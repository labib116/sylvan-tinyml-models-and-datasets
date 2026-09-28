"""Evaluate the exported uint8 TFLite plant-presence model."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


MODEL = Path("training_runs/plant_presence_mobilenetv2_035/model_int8.tflite")
TEST = Path("tinyml_plant_presence_dataset/images/test")
LABELS = ["not_plant", "plant"]

interpreter = tf.lite.Interpreter(model_path=str(MODEL))
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]
output_scale, output_zero = output_detail["quantization"]

truths: list[int] = []
predictions: list[int] = []
for truth, label in enumerate(LABELS):
    for path in sorted((TEST / label).glob("*.jpg")):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
            tensor = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        raw = int(interpreter.get_tensor(output_detail["index"])[0, 0])
        probability = (raw - output_zero) * output_scale
        truths.append(truth)
        predictions.append(int(probability >= 0.5))

tn = sum(a == 0 and b == 0 for a, b in zip(truths, predictions))
fp = sum(a == 0 and b == 1 for a, b in zip(truths, predictions))
fn = sum(a == 1 and b == 0 for a, b in zip(truths, predictions))
tp = sum(a == 1 and b == 1 for a, b in zip(truths, predictions))
metrics = {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": (tn + tp) / len(truths)}
(MODEL.parent / "int8_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(metrics)
