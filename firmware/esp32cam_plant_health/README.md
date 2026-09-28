# ESP32-CAM plant-health inference

This ESP-IDF firmware runs the repository's quantized plant-health model on an
AI-Thinker ESP32-CAM installed in an ESP32-CAM-MB USB carrier.

The build embeds this file automatically:

`training_runs/plant_health_new_mobilenetv2_035/model_int8.tflite`

The model accepts one 96x96 RGB `uint8` image and returns the probability of
the `unhealthy` class. A result below 0.5 is reported as `healthy`.

The firmware performs an explicit RGB565-to-RGB conversion. Do not replace it
with `fmt2rgb888()` without swapping red and blue, because that helper emits
BGR bytes for an RGB565 source.

## Requirements

- AI-Thinker ESP32-CAM with working PSRAM
- ESP32-CAM-MB USB carrier and a data-capable USB cable
- ESP-IDF 5.1 or newer
- Internet access during the first build so the ESP-IDF Component Manager can
  download `esp-tflite-micro` and `esp32-camera`

## Build and flash

Open an ESP-IDF PowerShell, then run:

```powershell
cd C:\Users\risal\sylvan_dataset\firmware\esp32cam_plant_health
idf.py set-target esp32
idf.py build
idf.py -p COM5 flash monitor
```

Replace `COM5` with the ESP32-CAM-MB port shown in Windows Device Manager. Exit
the serial monitor with `Ctrl+]`.

The ESP32-CAM-MB normally controls the boot pins automatically. If connection
fails, hold the module's `IO0/BOOT` button, press and release `RST`, start the
flash command, and release `IO0/BOOT` when writing begins.

## Expected serial output

```text
I (...) plant_health: Model ready: 652952 bytes, tensor arena used: ... bytes
I (...) plant_health: Camera ready at 96x96 RGB565
I (...) plant_health: prediction=healthy confidence=... unhealthy_probability=... raw=... inference=...ms
```

Point the camera at a whole plant in good light. The firmware captures and
classifies a frame every two seconds. GPIO 4 (the bright white flash LED) is
left off to avoid overexposing nearby leaves.

## Troubleshooting

- `Could not allocate ... PSRAM`: confirm the module has PSRAM and that the
  log reports several megabytes of free PSRAM.
- `AllocateTensors failed`: increase `kTensorArenaSize` in `main/main.cc`, while
  leaving enough PSRAM for the camera framebuffer.
- Camera error `0x105`: check that the selected module is AI-Thinker and reseat
  the OV2640 ribbon cable.
- Brownout or repeated resets: use a short USB cable and a stable 5 V supply.
- Flash timeout: use the manual `IO0/BOOT` and `RST` sequence described above.
