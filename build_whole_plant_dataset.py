"""Build the whole-plant-only ESP32-CAM prototype dataset."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import build_tinyml_binary_dataset as common


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "tinyml_whole_plant_dataset"
SEED = 20260927
AUGMENTATIONS_PER_TRAIN_ORIGINAL = 4
EXPECTED_PER_CLASS = 24
DATASET_PURPOSE = "whole-plant healthy vs unhealthy ESP32-CAM prototype"
README_TEXT = (
    "# Whole-Plant TinyML Dataset\n\n"
    "This binary prototype intentionally excludes PlantVillage and isolated leaf close-ups. "
    "It uses 320x240 plant-scale images. Augmentation is restricted to the training split. "
    "Validation and test contain only untouched originals. Do not randomly resplit files.\n"
)

# One clear plant-scale view across 24 different garden plant types.
HEALTHY_SAMPLE_NUMBERS = {
    2, 5, 8, 11, 15, 17, 20, 24, 29, 34, 37, 46,
    49, 53, 58, 61, 68, 71, 74, 81, 86, 89, 99, 120,
}

# These show a plant/canopy/branch context rather than an isolated symptom leaf.
FIELD_UNHEALTHY_SAMPLE_NUMBERS = {182, 183, 187, 194}

# Plant-scale generic images; leaf-only/macro examples are deliberately excluded.
GENERIC_WHOLE_PLANT_SAMPLE_NUMBERS = {
    1, 2, 3, 4, 5, 6, 7, 8,
    10, 11, 12, 13, 14, 15, 17, 22,
    24, 25, 26, 27,
}


def write_selection_review(current_rows: list[dict[str, str]], generic_rows: list[dict[str, str]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = ["source", "sample_no", "image_id", "label", "decision", "reason"]
    rows: list[dict[str, str]] = []
    for row in current_rows:
        number = int(row["sample_no"])
        label = row["proposed_health_label"]
        selected = number in HEALTHY_SAMPLE_NUMBERS or number in FIELD_UNHEALTHY_SAMPLE_NUMBERS
        rows.append(
            {
                "source": "field_garden",
                "sample_no": str(number),
                "image_id": row["image_id"],
                "label": label,
                "decision": "accept" if selected else "exclude_from_whole_plant_v1",
                "reason": "selected camera-context view" if selected else "not selected for compact balanced subset",
            }
        )
    for row in generic_rows:
        number = int(row["sample_no"])
        selected = number in GENERIC_WHOLE_PLANT_SAMPLE_NUMBERS and row["review_decision"] == "accept"
        rows.append(
            {
                "source": "generic_commons",
                "sample_no": str(number),
                "image_id": row["image_id"],
                "label": "unhealthy",
                "decision": "accept" if selected else "exclude_from_whole_plant_v1",
                "reason": "selected camera-context unhealthy view" if selected else "isolated, missing, or excluded for this dataset version",
            }
        )
    with (OUTPUT / "selection_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gather_records() -> list[dict[str, str]]:
    current_rows = common.read_csv(ROOT / "quality_review" / "current_active" / "review_all.csv")
    full_field = {
        row["image_id"]: row
        for row in common.read_csv(ROOT / "whole_csv_candidates" / "metadata" / "images.csv")
    }
    generic_reviewed = common.read_csv(ROOT / "generic_unhealthy_candidates" / "reviewed.csv")
    write_selection_review(current_rows, generic_reviewed)

    selected_field = [
        row
        for row in current_rows
        if int(row["sample_no"]) in HEALTHY_SAMPLE_NUMBERS | FIELD_UNHEALTHY_SAMPLE_NUMBERS
    ]
    records: list[dict[str, str]] = []
    for indexed in selected_field:
        row = full_field[indexed["image_id"]]
        label = row["proposed_health_label"]
        records.append(
            {
                "original_id": row["image_id"],
                "label": label,
                "source": "field_garden",
                "source_class": row.get("plant_id", ""),
                "visual_issue": row.get("common_disease", "") if label == "unhealthy" else "healthy",
                "group_id": common.field_group(row),
                "source_path": str(ROOT / "whole_csv_candidates" / row["local_path"]),
                "source_page_url": row.get("source_page_url", ""),
                "license": row.get("license", ""),
                "attribution": row.get("attribution", ""),
            }
        )

    for row in generic_reviewed:
        number = int(row["sample_no"])
        if number not in GENERIC_WHOLE_PLANT_SAMPLE_NUMBERS or row["review_decision"] != "accept":
            continue
        records.append(
            {
                "original_id": "gu_" + row["image_id"],
                "label": "unhealthy",
                "source": "generic_commons",
                "source_class": "generic_whole_plant",
                "visual_issue": row.get("visual_issue", ""),
                "group_id": common.generic_group(row),
                "source_path": str(ROOT / "generic_unhealthy_candidates" / row["local_path"]),
                "source_page_url": row.get("source_page_url", ""),
                "license": row.get("license", ""),
                "attribution": row.get("attribution", ""),
            }
        )

    missing = [record["source_path"] for record in records if not Path(record["source_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing selected whole-plant file: {missing[0]}")
    counts = Counter(record["label"] for record in records)
    if counts != Counter({"healthy": EXPECTED_PER_CLASS, "unhealthy": EXPECTED_PER_CLASS}):
        raise RuntimeError(f"Unexpected class balance: {counts}")
    return records


def main() -> int:
    if (OUTPUT / "manifest.csv").exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUTPUT}")
    rng = random.Random(SEED)
    records = gather_records()
    common.assign_splits(records, rng)

    manifest: list[dict[str, str]] = []
    attribution: list[dict[str, str]] = []
    for record in records:
        frame = common.camera_frame(Path(record["source_path"]))
        relative = Path("images") / record["split"] / record["label"] / f"{record['original_id']}.jpg"
        digest = common.save_jpeg(frame, OUTPUT / relative, common.BASE_JPEG_QUALITY)
        manifest.append(
            {
                "image_id": record["original_id"],
                "parent_original_id": record["original_id"],
                "split": record["split"],
                "label": record["label"],
                "source": record["source"],
                "source_class": record["source_class"],
                "visual_issue": record["visual_issue"],
                "group_id": record["group_id"],
                "is_augmented": "0",
                "augmentation": "none",
                "local_path": relative.as_posix(),
                "width": "320",
                "height": "240",
                "jpeg_quality": str(common.BASE_JPEG_QUALITY),
                "sha256": digest,
            }
        )
        attribution.append(
            {
                "original_id": record["original_id"],
                "source": record["source"],
                "source_page_url": record["source_page_url"],
                "license": record["license"],
                "attribution": record["attribution"],
            }
        )

        if record["split"] == "train":
            for aug_index in range(1, AUGMENTATIONS_PER_TRAIN_ORIGINAL + 1):
                augmented, quality, settings = common.augment(frame, rng)
                aug_id = f"{record['original_id']}__aug{aug_index:02d}"
                aug_relative = Path("images") / "train" / record["label"] / f"{aug_id}.jpg"
                aug_digest = common.save_jpeg(augmented, OUTPUT / aug_relative, quality)
                manifest.append(
                    {
                        "image_id": aug_id,
                        "parent_original_id": record["original_id"],
                        "split": "train",
                        "label": record["label"],
                        "source": record["source"],
                        "source_class": record["source_class"],
                        "visual_issue": record["visual_issue"],
                        "group_id": record["group_id"],
                        "is_augmented": "1",
                        "augmentation": json.dumps(settings, separators=(",", ":")),
                        "local_path": aug_relative.as_posix(),
                        "width": "320",
                        "height": "240",
                        "jpeg_quality": str(quality),
                        "sha256": aug_digest,
                    }
                )

    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (OUTPUT / "attribution.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attribution[0]))
        writer.writeheader()
        writer.writerows(attribution)

    originals = [row for row in manifest if row["is_augmented"] == "0"]
    summary = {
        "purpose": DATASET_PURPOSE,
        "plantvillage_included": False,
        "image_size": [320, 240],
        "original_total": len(originals),
        "final_total": len(manifest),
        "original_counts": dict(Counter(f"{row['split']}/{row['label']}" for row in originals)),
        "final_counts": dict(Counter(f"{row['split']}/{row['label']}" for row in manifest)),
        "source_counts": dict(Counter(f"{record['source']}/{record['label']}" for record in records)),
        "augmentations_per_train_original": AUGMENTATIONS_PER_TRAIN_ORIGINAL,
        "validation_and_test_augmented": False,
        "limitations": [
            "small prototype dataset",
            "internet images do not perfectly match the installed camera",
            "replace or supplement with real ESP32-CAM sequences before deployment",
        ],
    }
    (OUTPUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "README.md").write_text(README_TEXT, encoding="utf-8")
    common.OUTPUT = OUTPUT
    common.make_overview(manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
