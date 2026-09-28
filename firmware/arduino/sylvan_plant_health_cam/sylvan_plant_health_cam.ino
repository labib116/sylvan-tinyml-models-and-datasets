#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// AI-Thinker ESP32-CAM pin mapping. The ESP32-CAM-MB is the USB programmer
// board underneath the camera module and does not change these camera pins.
constexpr int PIN_PWDN = 32;
constexpr int PIN_RESET = -1;
constexpr int PIN_XCLK = 0;
constexpr int PIN_SIOD = 26;
constexpr int PIN_SIOC = 27;
constexpr int PIN_Y9 = 35;
constexpr int PIN_Y8 = 34;
constexpr int PIN_Y7 = 39;
constexpr int PIN_Y6 = 36;
constexpr int PIN_Y5 = 21;
constexpr int PIN_Y4 = 19;
constexpr int PIN_Y3 = 18;
constexpr int PIN_Y2 = 5;
constexpr int PIN_VSYNC = 25;
constexpr int PIN_HREF = 23;
constexpr int PIN_PCLK = 22;

constexpr int IMAGE_WIDTH = 96;
constexpr int IMAGE_HEIGHT = 96;
constexpr int IMAGE_CHANNELS = 3;
constexpr size_t IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr size_t MODEL_INPUT_BYTES = IMAGE_PIXELS * IMAGE_CHANNELS;
constexpr float CLASSIFICATION_THRESHOLD = 0.5F;
constexpr unsigned long INFERENCE_INTERVAL_MS = 2000;

// This MobileNetV2 model requires external PSRAM. Two MiB leaves space for
// the camera framebuffer on the usual 4 MiB AI-Thinker module.
constexpr size_t TENSOR_ARENA_SIZE = 2 * 1024 * 1024;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* modelInput = nullptr;
TfLiteTensor* modelOutput = nullptr;

uint8_t* tensorArenaAllocation = nullptr;
uint8_t* tensorArena = nullptr;

tflite::MicroMutableOpResolver<8> resolver;

[[noreturn]] void stopWithError(const char* message) {
  Serial.print("FATAL: ");
  Serial.println(message);
  while (true) {
    delay(1000);
  }
}

bool registerModelOperations() {
  // These are the eight operators present in model_int8.tflite.
  return resolver.AddQuantize() == kTfLiteOk &&
         resolver.AddMul() == kTfLiteOk &&
         resolver.AddAdd() == kTfLiteOk &&
         resolver.AddConv2D() == kTfLiteOk &&
         resolver.AddDepthwiseConv2D() == kTfLiteOk &&
         resolver.AddMean() == kTfLiteOk &&
         resolver.AddFullyConnected() == kTfLiteOk &&
         resolver.AddLogistic() == kTfLiteOk;
}

bool initializeCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = PIN_Y2;
  config.pin_d1 = PIN_Y3;
  config.pin_d2 = PIN_Y4;
  config.pin_d3 = PIN_Y5;
  config.pin_d4 = PIN_Y6;
  config.pin_d5 = PIN_Y7;
  config.pin_d6 = PIN_Y8;
  config.pin_d7 = PIN_Y9;
  config.pin_xclk = PIN_XCLK;
  config.pin_pclk = PIN_PCLK;
  config.pin_vsync = PIN_VSYNC;
  config.pin_href = PIN_HREF;
  config.pin_sccb_sda = PIN_SIOD;
  config.pin_sccb_scl = PIN_SIOC;
  config.pin_pwdn = PIN_PWDN;
  config.pin_reset = PIN_RESET;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_96X96;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  const esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {
    Serial.printf("Camera initialization failed: 0x%x\n",
                  static_cast<unsigned>(error));
    return false;
  }

  // Throw away early frames while auto-exposure and white balance settle.
  for (int frameNumber = 0; frameNumber < 3; ++frameNumber) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame != nullptr) {
      esp_camera_fb_return(frame);
    }
    delay(100);
  }

  Serial.println("Camera ready: 96x96 RGB565");
  return true;
}

bool initializeModel() {
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema %d does not match library schema %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  if (!registerModelOperations()) {
    Serial.println("Could not register the model operations");
    return false;
  }

  // Allocate an extra 16 bytes and align the arena address as required by TFLM.
  tensorArenaAllocation = static_cast<uint8_t*>(heap_caps_malloc(
      TENSOR_ARENA_SIZE + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensorArenaAllocation == nullptr) {
    Serial.println("Tensor arena PSRAM allocation failed");
    return false;
  }
  const uintptr_t rawAddress =
      reinterpret_cast<uintptr_t>(tensorArenaAllocation);
  tensorArena = reinterpret_cast<uint8_t*>((rawAddress + 15U) & ~uintptr_t(15U));

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, TENSOR_ARENA_SIZE);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    return false;
  }

  modelInput = interpreter->input(0);
  modelOutput = interpreter->output(0);

  const bool validInput =
      modelInput != nullptr && modelInput->type == kTfLiteUInt8 &&
      modelInput->dims != nullptr && modelInput->dims->size == 4 &&
      modelInput->dims->data[0] == 1 &&
      modelInput->dims->data[1] == IMAGE_HEIGHT &&
      modelInput->dims->data[2] == IMAGE_WIDTH &&
      modelInput->dims->data[3] == IMAGE_CHANNELS &&
      modelInput->bytes == MODEL_INPUT_BYTES;
  const bool validOutput =
      modelOutput != nullptr && modelOutput->type == kTfLiteUInt8 &&
      modelOutput->dims != nullptr && modelOutput->dims->size == 2 &&
      modelOutput->dims->data[0] == 1 && modelOutput->dims->data[1] == 1;

  if (!validInput || !validOutput) {
    Serial.println("Unexpected model input/output tensor");
    return false;
  }

  Serial.printf("Model ready: %u bytes; tensor arena used: %u bytes\n",
                g_model_data_len,
                static_cast<unsigned>(interpreter->arena_used_bytes()));
  Serial.printf("Input: uint8 [1,96,96,3], scale=%.8f, zero=%ld\n",
                modelInput->params.scale,
                static_cast<long>(modelInput->params.zero_point));
  Serial.printf("Output: uint8 [1,1], scale=%.8f, zero=%ld\n",
                modelOutput->params.scale,
                static_cast<long>(modelOutput->params.zero_point));
  return true;
}

bool captureIntoModelInput() {
  camera_fb_t* frame = esp_camera_fb_get();
  if (frame == nullptr) {
    Serial.println("Camera capture failed");
    return false;
  }

  bool converted = false;
  if (frame->width == IMAGE_WIDTH && frame->height == IMAGE_HEIGHT &&
      frame->format == PIXFORMAT_RGB565 && frame->len >= IMAGE_PIXELS * 2) {
    // OV2640 RGB565 is big-endian. Convert it into true RGB888. Do not use
    // fmt2rgb888() here: for RGB565 input that helper emits BGR byte order.
    for (size_t pixel = 0; pixel < IMAGE_PIXELS; ++pixel) {
      const uint8_t high = frame->buf[pixel * 2];
      const uint8_t low = frame->buf[pixel * 2 + 1];
      const uint8_t red5 = high >> 3;
      const uint8_t green6 = ((high & 0x07) << 3) | (low >> 5);
      const uint8_t blue5 = low & 0x1F;

      modelInput->data.uint8[pixel * 3] = (red5 << 3) | (red5 >> 2);
      modelInput->data.uint8[pixel * 3 + 1] =
          (green6 << 2) | (green6 >> 4);
      modelInput->data.uint8[pixel * 3 + 2] =
          (blue5 << 3) | (blue5 >> 2);
    }
    converted = true;
  } else {
    Serial.printf("Unexpected frame: %ux%u, format=%d, bytes=%u\n",
                  static_cast<unsigned>(frame->width),
                  static_cast<unsigned>(frame->height), frame->format,
                  static_cast<unsigned>(frame->len));
  }

  esp_camera_fb_return(frame);
  return converted;
}

void runInference() {
  if (!captureIntoModelInput()) {
    return;
  }

  const uint32_t startedAt = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }
  const uint32_t inferenceTime = millis() - startedAt;

  const uint8_t rawOutput = modelOutput->data.uint8[0];
  float unhealthyProbability =
      (static_cast<int>(rawOutput) - modelOutput->params.zero_point) *
      modelOutput->params.scale;
  if (unhealthyProbability < 0.0F) unhealthyProbability = 0.0F;
  if (unhealthyProbability > 1.0F) unhealthyProbability = 1.0F;

  const bool unhealthy =
      unhealthyProbability >= CLASSIFICATION_THRESHOLD;
  const char* label = unhealthy ? "unhealthy" : "healthy";
  const float confidence =
      unhealthy ? unhealthyProbability : 1.0F - unhealthyProbability;

  Serial.printf(
      "prediction=%s confidence=%.1f%% unhealthy_probability=%.1f%% "
      "raw=%u inference=%lums\n",
      label, confidence * 100.0F, unhealthyProbability * 100.0F,
      static_cast<unsigned>(rawOutput),
      static_cast<unsigned long>(inferenceTime));
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println();
  Serial.println("Sylvan ESP32-CAM plant-health classifier");

  if (!psramFound()) {
    stopWithError("PSRAM was not detected. Select AI Thinker ESP32-CAM and enable PSRAM.");
  }

  Serial.printf("PSRAM total: %u bytes; free: %u bytes\n",
                static_cast<unsigned>(ESP.getPsramSize()),
                static_cast<unsigned>(ESP.getFreePsram()));

  if (!initializeModel()) {
    stopWithError("Model initialization failed");
  }
  if (!initializeCamera()) {
    stopWithError("Camera initialization failed");
  }

  Serial.println("Setup complete. Classifying one frame every two seconds.");
}

void loop() {
  runInference();
  delay(INFERENCE_INTERVAL_MS);
}
