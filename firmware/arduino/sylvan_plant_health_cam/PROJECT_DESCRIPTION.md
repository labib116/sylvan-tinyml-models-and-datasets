# Sylvan TinyML Plant-Health Classifier

## Interview-ready summary

Sylvan is an edge-AI prototype that classifies a camera image of a plant as
`healthy` or `unhealthy` directly on an AI-Thinker ESP32-CAM. The goal was to
build a useful computer-vision pipeline under microcontroller constraints:
limited flash, limited RAM, no operating system, and no cloud connection during
inference.

I trained a MobileNetV2-based binary classifier using transfer learning,
converted it to a fully integer-quantized TensorFlow Lite model, and integrated
it with the OV2640 camera. The firmware captures a 96x96 frame, converts the
camera's RGB565 pixels into the RGB byte order used during training, runs the
model with TensorFlow Lite Micro, and prints the predicted class and confidence
over serial.

The final model achieved 78.1% test accuracy and an AUC of 0.848. Quantization
reduced the TFLite file from 1.63 MB to 653 KB, a 59.9% reduction, making it
practical to store in the ESP32's flash. Intermediate tensors are allocated in
the ESP32-CAM module's external PSRAM.

## Thirty-second interview pitch

> I built an offline plant-health classifier for an ESP32-CAM. I chose a
> MobileNetV2 backbone with a 0.35 width multiplier because its depthwise
> separable convolutions offer a good accuracy-to-compute tradeoff, and ImageNet
> transfer learning is valuable when the custom dataset is small. I froze the
> backbone and trained only a 1,281-parameter classification head. I then used
> post-training full-integer quantization with 100 representative training
> images, forcing integer kernels and `uint8` input/output tensors. That reduced
> the model from 1.63 MB to 653 KB. On the device, the model stays in flash, the
> tensor arena uses PSRAM, and each 96x96 RGB camera frame is classified locally
> without sending the image to a server.

## Problem and design constraints

The system answers one deliberately narrow question: does the pictured whole
plant look healthy or unhealthy? It is a screening classifier, not a plant
species identifier or a disease diagnosis system.

The deployment target creates several constraints:

- The original ESP32 has far less RAM and compute than a phone or server.
- The camera produces RGB565 data, while the model was trained with RGB images.
- Model weights must fit in flash, while activations and scratch buffers must
  fit in RAM or PSRAM.
- Inference must work offline with deterministic preprocessing.
- The firmware and model must fit in the Arduino **Huge APP** partition.

The ESP32-CAM-MB is the USB programming and power carrier. The actual inference
runs on the AI-Thinker ESP32-CAM module mounted on it.

## Dataset

The final dataset contains 494 labeled images:

| Split | Healthy | Unhealthy | Total |
|---|---:|---:|---:|
| Training | 176 | 172 | 348 |
| Validation | 36 | 37 | 73 |
| Test | 36 | 37 | 73 |
| **Total** | **248** | **246** | **494** |

The classes are almost balanced. Small class weights were still calculated
from the training split: 0.989 for healthy and 1.012 for unhealthy. A fixed seed
of `20260927` was used to make the experiment repeatable.

## Why MobileNetV2 was selected

The selected architecture is MobileNetV2 with an `alpha=0.35` width multiplier
and a 96x96 RGB input.

### 1. Efficient convolution design

MobileNetV2 uses depthwise separable convolutions and inverted residual blocks.
This requires much less computation than a conventional convolutional network
with a similar feature-extraction depth. The 0.35 width multiplier reduces the
number of channels further, which lowers parameter count, activation memory,
and inference cost.

### 2. Transfer learning for a small dataset

The dataset is too small to reliably learn robust visual features from scratch.
The MobileNetV2 backbone started with ImageNet weights, providing useful edge,
texture, shape, and color features. The backbone was frozen, and only the final
classification head was trained.

### 3. Controlled model capacity

The resulting network contains 411,489 parameters, but only 1,281 were
trainable. This reduces overfitting risk and makes CPU-only training fast while
retaining a much stronger feature extractor than the initial tiny CNN baseline.

The initial tiny CNN experiment reached only 47.4% accuracy on its early test
split and predicted every example as healthy. That experiment used an earlier,
smaller dataset, so it is not a strict apples-to-apples benchmark, but it showed
that a randomly initialized compact network was not learning sufficiently
useful features. Transfer learning was therefore the more defensible choice.

### 4. Deliberately small image resolution

The model uses 96x96 images rather than 224x224. Reducing each spatial dimension
greatly lowers activation memory and convolution work. The resolution is a
tradeoff: it preserves broad whole-plant appearance but may lose small lesion
details.

## Model architecture

The training graph is:

1. Input: `96 × 96 × 3` RGB image with values from 0 to 255.
2. Rescale pixels to the MobileNetV2 range of -1 to 1.
3. Frozen MobileNetV2 `alpha=0.35` feature extractor.
4. Global average pooling.
5. Dropout with rate 0.25.
6. One sigmoid output representing `P(unhealthy)`.

Training used Adam with a learning rate of `0.0005`, binary cross-entropy,
batch size 16, and 30 epochs. The training code also used validation-loss model
checkpointing, early stopping, and learning-rate reduction.

## Why quantization was necessary

The floating-point TFLite model was 1,628,628 bytes. It could fit in some flash
layouts, but floating-point inference requires more memory bandwidth and is a
poor match for a small microcontroller. Integer inference provides a smaller
artifact and allows TensorFlow Lite Micro to use integer kernels.

The exported INT8 model is 652,952 bytes:

| Artifact | Size | Relative size |
|---|---:|---:|
| Float32 TFLite | 1,628,628 bytes | 100% |
| Integer TFLite | 652,952 bytes | 40.1% |

That is a 975,676-byte or 59.9% reduction, making the integer model about 2.49
times smaller.

## How the model was quantized

The project uses post-training full-integer quantization:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data(train_dataset)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
quantized_model = converter.convert()
```

The representative-data generator supplies 100 real training images. During
conversion, TensorFlow observes their activation ranges and calculates scales
and zero-points for the integer tensors. Requiring `TFLITE_BUILTINS_INT8`
prevents the converter from silently leaving unsupported layers in floating
point.

Although the internal weights and activations use integer quantization, the
external boundary tensors are unsigned eight-bit values. This is convenient for
camera pixels:

- Input shape: `[1, 96, 96, 3]`
- Input type: `uint8`
- Input scale: `1.0`
- Input zero-point: `0`
- Output shape: `[1, 1]`
- Output type: `uint8`
- Output scale: `0.00390625`, or 1/256
- Output zero-point: `0`

Therefore, raw RGB bytes can be copied into the input tensor without host-side
floating-point normalization. The rescaling operation is represented inside
the quantized graph. The output probability is reconstructed as:

```cpp
probability_unhealthy =
    (raw_output - output_zero_point) * output_scale;
```

For this model that is effectively `raw_output / 256`. A threshold of 0.5
selects `unhealthy`; otherwise the prediction is `healthy`.

## ESP32-CAM deployment pipeline

The Arduino firmware follows this sequence:

1. Embed `model_int8.tflite` as a constant C++ byte array in program flash.
2. Allocate a 2 MB TensorFlow Lite Micro tensor arena in external PSRAM.
3. Initialize the OV2640 camera using the AI-Thinker pin map.
4. Capture a 96x96 RGB565 frame.
5. Convert every pixel from big-endian RGB565 into RGB888.
6. Write the 27,648 RGB bytes directly into the model's `uint8` input tensor.
7. Invoke TensorFlow Lite Micro.
8. Dequantize the one-byte output and print the label and confidence over
   serial.

The explicit color conversion is important. Espressif's commonly used
`fmt2rgb888()` helper emits BGR byte order for an RGB565 source, but the model
was trained on RGB. Silently swapping red and blue would create a
training-serving skew and could reduce accuracy.

Only the eight operators present in the model are registered: `QUANTIZE`,
`MUL`, `ADD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `MEAN`, `FULLY_CONNECTED`, and
`LOGISTIC`. A selective resolver reduces firmware size compared with linking
every TensorFlow Lite Micro operator.

## Measured model results

The final 73-image test split produced:

| Metric | Result |
|---|---:|
| Accuracy | 78.08% |
| ROC AUC | 84.76% |
| Unhealthy precision | 81.82% |
| Unhealthy recall | 72.97% |
| Unhealthy F1 | 77.14% |
| Healthy specificity | 83.33% |

Confusion matrix, treating `unhealthy` as the positive class:

| | Predicted healthy | Predicted unhealthy |
|---|---:|---:|
| Actual healthy | 30 | 6 |
| Actual unhealthy | 10 | 27 |

The model is better at avoiding false unhealthy alarms than at finding every
unhealthy plant. For an application where missing an unhealthy plant is more
expensive, the threshold could be lowered below 0.5 after validation-set
calibration.

## Engineering tradeoffs

- **Accuracy versus footprint:** MobileNetV2 is larger than the tiny CNN but
  learned substantially more useful features. Quantization recovered much of
  the footprint cost.
- **Resolution versus detail:** 96x96 reduces RAM and computation but can miss
  small symptoms.
- **Frozen backbone versus adaptation:** freezing improves stability on a small
  dataset, but camera-specific fine-tuning could improve domain adaptation.
- **PSRAM versus speed:** the large tensor arena fits in external PSRAM, though
  PSRAM access is slower than internal SRAM.
- **Binary screening versus diagnosis:** the binary target fits the dataset and
  device constraints but cannot identify a disease or recommend treatment.

## Honest limitations

- The test set contains only 73 images, so the metrics have high uncertainty.
- The labels combine many possible stress causes into one `unhealthy` class.
- Lighting, background, plant species, camera distance, and image-source bias
  can influence predictions.
- The model was evaluated on the desktop TFLite/Keras pipeline; final latency,
  memory use, and accuracy must still be measured on the physical ESP32-CAM.
- This is a prototype screening tool and should not be presented as an
  agronomic diagnostic device.

## Next improvements

1. Collect a larger, species-diverse dataset directly with the OV2640 camera.
2. Add hard negatives and difficult lighting conditions.
3. Fine-tune the upper MobileNetV2 blocks with a low learning rate.
4. Compare post-training quantization with quantization-aware training.
5. Calibrate the decision threshold based on the desired recall/precision
   tradeoff.
6. Measure on-device latency, peak tensor-arena usage, power, and thermal
   stability.
7. Use repeated stratified evaluation or cross-validation before claiming
   production-level accuracy.

## Common interview questions

### Why not send the image to a cloud API?

Local inference works without Wi-Fi, avoids uploading plant images, removes
network latency and service cost, and makes the prototype usable in locations
with unreliable connectivity.

### Why not use the much smaller tiny CNN?

The tiny CNN was attractive for memory, but its early experiment collapsed to
one class. MobileNetV2 transfer learning supplied much stronger visual features.
The quantized MobileNet model still fits in flash, so it offered a better
accuracy-footprint compromise.

### Why use post-training quantization instead of quantization-aware training?

Post-training quantization is fast and requires no retraining. It was the right
first deployment step. Quantization-aware training is a logical follow-up if
on-device evaluation reveals an unacceptable accuracy loss.

### Why is the input `uint8` if this is called INT8 quantization?

The graph uses integer-quantized kernels internally, while the public input and
output interfaces were deliberately exported as `uint8`. This matches raw
camera bytes and simplifies firmware. Quantization nodes inside the graph map
the unsigned boundary values to the signed internal representations required by
the kernels.

### What was the hardest embedded issue?

Matching preprocessing between training and deployment. The camera produces
RGB565 and a standard ESP32 conversion helper returns BGR for that source. The
firmware explicitly converts RGB565 to RGB so the deployed tensor has the same
channel order used during training.

### How would you know the embedded version is correct?

Capture a fixed frame, run it through both the device preprocessing/inference
path and the desktop TFLite interpreter, and compare the input tensor and raw
quantized output. Then profile arena use and latency on hardware and evaluate a
labeled set captured by the actual OV2640 sensor.
