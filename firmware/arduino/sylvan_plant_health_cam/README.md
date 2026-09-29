# Arduino IDE: Sylvan ESP32-CAM plant-health classifier

Open `sylvan_plant_health_cam.ino` in Arduino IDE. The sketch is for an
AI-Thinker ESP32-CAM attached to an ESP32-CAM-MB USB programmer board.

For the architecture rationale, quantization explanation, measured results,
tradeoffs, and an interview pitch, read [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md).

## Required software

1. Install the **esp32 by Espressif Systems** board package in Arduino IDE's
   Boards Manager.
2. Install **Chirale_TensorFlowLite** in Library Manager.
3. Restart Arduino IDE after installing the library.

The generated `model_data.cpp` file beside the sketch contains
`training_runs/plant_health_new_mobilenetv2_035/model_int8.tflite`.

## Arduino IDE settings

- Board: **AI Thinker ESP32-CAM**
- CPU Frequency: **240MHz (WiFi/BT)**
- Flash Frequency: **40MHz**
- Partition Scheme: **Huge APP (3MB No OTA/1MB SPIFFS)**
- Upload Speed: **115200**
- Port: the COM port belonging to the ESP32-CAM-MB
- PSRAM: enable it if your ESP32 board package displays a PSRAM menu

## Flashing

1. Connect the ESP32-CAM-MB with a data-capable USB cable.
2. Open `sylvan_plant_health_cam.ino`.
3. Select the settings above.
4. Click **Upload**.
5. Open Serial Monitor at **115200 baud**.

The ESP32-CAM-MB usually handles download mode automatically. If upload fails:

1. Hold `IO0/BOOT`.
2. Press and release `RST`.
3. Start Upload.
4. Release `IO0/BOOT` when Arduino IDE starts writing.

Expected serial output:

```text
Model ready: 652952 bytes; tensor arena used: ... bytes
Camera ready: 96x96 RGB565
prediction=healthy confidence=... unhealthy_probability=... raw=... inference=...ms
```

## RAM and flash use

The ESP32 chip provides 520 KB of on-chip SRAM, but that is the chip's total,
not a 520 KB application heap. The Arduino core, FreeRTOS, stacks, camera
driver, DMA-capable buffers, and other runtime data also use internal SRAM.
The usual AI-Thinker ESP32-CAM module additionally provides 4 MB of external
PSRAM and 4 MB of flash. The ESP32-CAM-MB is only the USB programmer and power
carrier; it does not add RAM.

This sketch uses those memory areas as follows:

| Memory area | Use in this project |
| --- | --- |
| Flash | Stores the firmware and the existing 652,952-byte quantized model. Because `g_model_data` is `const`, the complete model is not copied into RAM. |
| External PSRAM | Reserves a 2,097,152-byte TensorFlow Lite Micro tensor arena and holds the camera framebuffer. |
| Tensor arena | Contains the 27,648-byte `uint8` input tensor (`96 x 96 x 3`), the one-byte output, intermediate activations, and operator scratch buffers. |
| Camera framebuffer | One 96 x 96 RGB565 frame requires at least 18,432 bytes, plus camera-driver bookkeeping/alignment. |
| Internal SRAM | Holds runtime state such as FreeRTOS stacks, driver data, DMA-capable memory, and small interpreter objects. |

The two MB arena is a reserved upper limit, not proof that every byte is used.
At startup, `interpreter->arena_used_bytes()` reports the model's actual arena
usage on the installed TensorFlow Lite library. The sketch also prints detected
and free PSRAM, so the physical board remains the final memory check.

The bright GPIO 4 flash LED remains off to avoid overexposing nearby leaves.
The model was trained on whole-plant images, so frame a complete plant in good
light rather than placing one leaf extremely close to the lens.

## Regenerating the embedded model

If `model_int8.tflite` changes, close Arduino IDE and run this from the
repository root:

```powershell
.venv\Scripts\python.exe `
  firmware\esp32cam_plant_health\tools\generate_model_data.py `
  --input training_runs\plant_health_new_mobilenetv2_035\model_int8.tflite `
  --output firmware\arduino\sylvan_plant_health_cam\model_data.cpp
```

Reopen the sketch and upload it again.
