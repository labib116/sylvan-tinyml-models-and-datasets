# Arduino IDE: Sylvan ESP32-CAM plant-health classifier

Open `sylvan_plant_health_cam.ino` in Arduino IDE. The sketch is for an
AI-Thinker ESP32-CAM attached to an ESP32-CAM-MB USB programmer board.

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
