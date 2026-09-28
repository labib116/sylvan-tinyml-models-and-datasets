#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr char kTag[] = "plant_health";

// AI-Thinker ESP32-CAM pin mapping. The ESP32-CAM-MB is the USB carrier for
// this module and does not change these camera pins.
constexpr int kPinPwdn = 32;
constexpr int kPinReset = -1;
constexpr int kPinXclk = 0;
constexpr int kPinSiod = 26;
constexpr int kPinSioc = 27;
constexpr int kPinY9 = 35;
constexpr int kPinY8 = 34;
constexpr int kPinY7 = 39;
constexpr int kPinY6 = 36;
constexpr int kPinY5 = 21;
constexpr int kPinY4 = 19;
constexpr int kPinY3 = 18;
constexpr int kPinY2 = 5;
constexpr int kPinVsync = 25;
constexpr int kPinHref = 23;
constexpr int kPinPclk = 22;

constexpr int kImageWidth = 96;
constexpr int kImageHeight = 96;
constexpr int kImageChannels = 3;
constexpr size_t kInputBytes =
    kImageWidth * kImageHeight * kImageChannels;
constexpr float kClassificationThreshold = 0.5F;

// This MobileNetV2 model needs much more RAM than the internal ESP32 heap.
// Two MiB leaves room in a typical 4 MiB PSRAM chip for the camera framebuffer.
constexpr size_t kTensorArenaSize = 2 * 1024 * 1024;

uint8_t* tensor_arena = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

tflite::MicroMutableOpResolver<8> resolver;

bool RegisterModelOperations() {
  return resolver.AddQuantize() == kTfLiteOk &&
         resolver.AddMul() == kTfLiteOk &&
         resolver.AddAdd() == kTfLiteOk &&
         resolver.AddConv2D() == kTfLiteOk &&
         resolver.AddDepthwiseConv2D() == kTfLiteOk &&
         resolver.AddMean() == kTfLiteOk &&
         resolver.AddFullyConnected() == kTfLiteOk &&
         resolver.AddLogistic() == kTfLiteOk;
}

bool InitializeCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = kPinY2;
  config.pin_d1 = kPinY3;
  config.pin_d2 = kPinY4;
  config.pin_d3 = kPinY5;
  config.pin_d4 = kPinY6;
  config.pin_d5 = kPinY7;
  config.pin_d6 = kPinY8;
  config.pin_d7 = kPinY9;
  config.pin_xclk = kPinXclk;
  config.pin_pclk = kPinPclk;
  config.pin_vsync = kPinVsync;
  config.pin_href = kPinHref;
  config.pin_sccb_sda = kPinSiod;
  config.pin_sccb_scl = kPinSioc;
  config.pin_pwdn = kPinPwdn;
  config.pin_reset = kPinReset;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_96X96;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  const esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {
    ESP_LOGE(kTag, "Camera initialization failed: 0x%x",
             static_cast<unsigned>(error));
    return false;
  }

  // Discard a few frames while automatic exposure and white balance settle.
  for (int i = 0; i < 3; ++i) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame != nullptr) {
      esp_camera_fb_return(frame);
    }
    vTaskDelay(pdMS_TO_TICKS(100));
  }

  ESP_LOGI(kTag, "Camera ready at %dx%d RGB565", kImageWidth,
           kImageHeight);
  return true;
}

bool InitializeModel() {
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "Model schema %d does not match runtime schema %d",
             model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  if (!RegisterModelOperations()) {
    ESP_LOGE(kTag, "Failed to register a TFLite Micro operation");
    return false;
  }

  tensor_arena = static_cast<uint8_t*>(heap_caps_malloc(
      kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr) {
    ESP_LOGE(kTag,
             "Could not allocate %u bytes of PSRAM for the tensor arena",
             static_cast<unsigned>(kTensorArenaSize));
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag,
             "AllocateTensors failed; increase kTensorArenaSize if PSRAM "
             "is still available");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  const bool input_is_valid =
      input != nullptr && input->type == kTfLiteUInt8 &&
      input->dims != nullptr && input->dims->size == 4 &&
      input->dims->data[0] == 1 && input->dims->data[1] == kImageHeight &&
      input->dims->data[2] == kImageWidth &&
      input->dims->data[3] == kImageChannels && input->bytes == kInputBytes;
  const bool output_is_valid =
      output != nullptr && output->type == kTfLiteUInt8 &&
      output->dims != nullptr && output->dims->size == 2 &&
      output->dims->data[0] == 1 && output->dims->data[1] == 1;

  if (!input_is_valid || !output_is_valid) {
    ESP_LOGE(kTag, "Unexpected model tensor shape or type");
    return false;
  }

  ESP_LOGI(kTag, "Model ready: %u bytes, tensor arena used: %u bytes",
           g_model_data_len,
           static_cast<unsigned>(interpreter->arena_used_bytes()));
  ESP_LOGI(kTag, "Input quantization: scale=%.8f zero_point=%ld",
           static_cast<double>(input->params.scale),
           static_cast<long>(input->params.zero_point));
  ESP_LOGI(kTag, "Output quantization: scale=%.8f zero_point=%ld",
           static_cast<double>(output->params.scale),
           static_cast<long>(output->params.zero_point));
  return true;
}

bool CaptureIntoModelInput() {
  camera_fb_t* frame = esp_camera_fb_get();
  if (frame == nullptr) {
    ESP_LOGE(kTag, "Camera capture failed");
    return false;
  }

  bool converted = false;
  if (frame->width == kImageWidth && frame->height == kImageHeight &&
      frame->format == PIXFORMAT_RGB565 &&
      frame->len >= kImageWidth * kImageHeight * 2) {
    // The OV2640 framebuffer stores each RGB565 pixel most-significant byte
    // first. Convert it explicitly to RGB order. Espressif's fmt2rgb888()
    // helper emits BGR for RGB565 input, which does not match model training.
    for (size_t pixel = 0; pixel < kImageWidth * kImageHeight; ++pixel) {
      const uint8_t high = frame->buf[pixel * 2];
      const uint8_t low = frame->buf[pixel * 2 + 1];
      const uint8_t red5 = high >> 3;
      const uint8_t green6 = ((high & 0x07) << 3) | (low >> 5);
      const uint8_t blue5 = low & 0x1F;
      input->data.uint8[pixel * 3] = (red5 << 3) | (red5 >> 2);
      input->data.uint8[pixel * 3 + 1] = (green6 << 2) | (green6 >> 4);
      input->data.uint8[pixel * 3 + 2] = (blue5 << 3) | (blue5 >> 2);
    }
    converted = true;
  } else {
    ESP_LOGE(kTag, "Unexpected camera frame: %ux%u format=%d",
             static_cast<unsigned>(frame->width),
             static_cast<unsigned>(frame->height), frame->format);
  }

  esp_camera_fb_return(frame);

  if (!converted) {
    ESP_LOGE(kTag, "RGB565-to-RGB conversion failed");
  }
  return converted;
}

void RunInferenceOnce() {
  if (!CaptureIntoModelInput()) {
    return;
  }

  const int64_t started_us = esp_timer_get_time();
  if (interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(kTag, "TFLite Micro inference failed");
    return;
  }
  const int64_t elapsed_ms = (esp_timer_get_time() - started_us) / 1000;

  const uint8_t raw_output = output->data.uint8[0];
  float unhealthy_probability =
      (static_cast<int>(raw_output) - output->params.zero_point) *
      output->params.scale;
  unhealthy_probability =
      std::clamp(unhealthy_probability, 0.0F, 1.0F);

  const char* label = unhealthy_probability >= kClassificationThreshold
                          ? "unhealthy"
                          : "healthy";
  const float confidence =
      label[0] == 'u' ? unhealthy_probability : 1.0F - unhealthy_probability;

  ESP_LOGI(kTag,
           "prediction=%s confidence=%.1f%% unhealthy_probability=%.1f%% "
           "raw=%u inference=%" PRId64 "ms",
           label, static_cast<double>(confidence * 100.0F),
           static_cast<double>(unhealthy_probability * 100.0F),
           static_cast<unsigned>(raw_output),
           elapsed_ms);
}

}  // namespace

extern "C" void app_main() {
  ESP_LOGI(kTag, "Starting Sylvan plant-health classifier");
  ESP_LOGI(kTag, "Free internal heap: %u bytes; free PSRAM: %u bytes",
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));

  if (!InitializeModel() || !InitializeCamera()) {
    ESP_LOGE(kTag, "Startup failed; inference task stopped");
    return;
  }

  while (true) {
    RunInferenceOnce();
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}
