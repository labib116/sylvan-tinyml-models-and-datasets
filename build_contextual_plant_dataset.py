"""Build the corrected natural-context dataset; close-ups are allowed."""

from __future__ import annotations

import random
from pathlib import Path

import build_tinyml_binary_dataset as common
import build_whole_plant_dataset as builder


ROOT = Path(__file__).resolve().parent
TARGET_PER_CLASS = 46

# Exclude only staged/isolated leaf examples and the one missing download.
CONTEXTUAL_GENERIC_SAMPLES = {
    1, 2, 3, 4, 5, 6, 7, 8,
    10, 11, 12, 13, 14, 15, 17,
    18, 21, 22, 23,
    24, 25, 26, 27,
}


def main() -> int:
    current = common.read_csv(ROOT / "quality_review" / "current_active" / "review_all.csv")
    full = common.read_csv(ROOT / "whole_csv_candidates" / "metadata" / "images.csv")
    full_healthy = [row for row in full if row["proposed_health_label"] == "healthy"]
    selected_healthy = common.choose_field_healthy(
        full_healthy,
        TARGET_PER_CLASS,
        random.Random(builder.SEED),
    )
    selected_ids = {row["image_id"] for row in selected_healthy}

    builder.OUTPUT = ROOT / "tinyml_contextual_plant_dataset"
    builder.EXPECTED_PER_CLASS = TARGET_PER_CLASS
    builder.AUGMENTATIONS_PER_TRAIN_ORIGINAL = 2
    builder.HEALTHY_SAMPLE_NUMBERS = {
        int(row["sample_no"])
        for row in current
        if row["image_id"] in selected_ids
    }
    builder.FIELD_UNHEALTHY_SAMPLE_NUMBERS = {
        int(row["sample_no"])
        for row in current
        if row["proposed_health_label"] == "unhealthy"
    }
    builder.GENERIC_WHOLE_PLANT_SAMPLE_NUMBERS = CONTEXTUAL_GENERIC_SAMPLES
    builder.DATASET_PURPOSE = "natural-context plant health; full plants and close-ups allowed"
    builder.README_TEXT = (
        "# Contextual Plant-Health TinyML Dataset\n\n"
        "This binary ESP32-CAM prototype allows both whole-plant views and close-ups when the "
        "plant remains in a natural garden context. PlantVillage and staged isolated-leaf images "
        "are excluded. Images are 320x240. Augmentation is training-only; validation and test "
        "contain untouched originals. Do not randomly resplit files.\n"
    )
    return builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
