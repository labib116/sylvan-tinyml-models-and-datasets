# TinyML Plant Health Binary Dataset

Binary labels: `healthy` and `unhealthy`.

All images are 320x240 RGB JPEGs, matching the ESP32-CAM QVGA capture path. The
dataset initially contained 320 reviewed/source originals and 494 files after one
training-only augmentation per original in the initial training split. Validation and test images are not
augmented. See `dataset_summary.json` for exact counts and augmentation ranges.

Use `images/train`, `images/validation`, and `images/test` as fixed splits. Never
randomly resplit individual files because augmented variants share a parent image.

PlantVillage is a controlled-background leaf dataset and does not fully match rooftop
camera imagery. The mixed field/garden sources and ESP-style degradation reduce, but do
not eliminate, that domain gap. Final accuracy should be checked on untouched images
captured by the actual camera.

## September 29 reviewed-image update

Fourteen manually reviewed originals were added only to training: five green tomato-plant images labeled `healthy` and nine ESP32-CAM bare branch/stem frames labeled `unhealthy`. They were left unaugmented, and the fixed validation and test splits were not changed. Their original filenames were preserved, and the existing model files remain unchanged.
