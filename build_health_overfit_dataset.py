"""Add the new dried-plant screenshots to a separate health overfit dataset."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from build_plant_presence_dataset import augment, normalized_image, save_jpeg, sha256


SEED = 20260927
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tinyml_contextual_plant_dataset"
OUTPUT = ROOT / "tinyml_health_overfit_dataset"
DOWNLOADS = Path(r"C:\Users\risal\Downloads")
NEW_IMAGES = [
    "Screenshot 2026-09-27 210300.png",
    "Screenshot 2026-09-27 210324.png",
    "Screenshot 2026-09-27 210547.png",
    "Screenshot 2026-09-27 210605.png",
    "Screenshot 2026-09-27 210641.png",
    "Screenshot 2026-09-27 210656.png",
    "Screenshot 2026-09-27 210730.png",
    "Screenshot 2026-09-27 210811.png",
    "Screenshot 2026-09-27 210900.png",
    "Screenshot 2026-09-27 210923.png",
    "Screenshot 2026-09-27 214748.png",
    "Screenshot 2026-09-27 214846.png",
]


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing dataset: {OUTPUT}")
    missing = [name for name in NEW_IMAGES if not (DOWNLOADS / name).is_file()]
    if missing:
        raise SystemExit("Missing images:\n" + "\n".join(missing))

    shutil.copytree(SOURCE, OUTPUT)
    manifest_path = OUTPUT / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    rng = random.Random(SEED + 81)
    for index, filename in enumerate(NEW_IMAGES):
        source_path = DOWNLOADS / filename
        parent_id = f"dryshot_{index:02d}"
        group_id = f"new_dry:{source_path.stem}"
        base = normalized_image(source_path)
        destination = OUTPUT / "images" / "train" / "unhealthy" / f"{parent_id}.jpg"
        save_jpeg(base, destination, 72)
        rows.append(
            {
                "image_id": parent_id,
                "parent_original_id": parent_id,
                "split": "train",
                "label": "unhealthy",
                "source": "new_downloads_dry",
                "source_class": "generic_dried_plant",
                "visual_issue": "dried_wilted",
                "group_id": group_id,
                "is_augmented": "0",
                "augmentation": "none",
                "local_path": destination.relative_to(OUTPUT).as_posix(),
                "width": "320",
                "height": "240",
                "jpeg_quality": "72",
                "sha256": sha256(destination),
            }
        )
        # Deliberate domain overfit: seven degraded crop variants per new image.
        for aug_index in range(1, 8):
            image, params = augment(base, rng)
            image_id = f"{parent_id}__aug{aug_index:02d}"
            aug_path = OUTPUT / "images" / "train" / "unhealthy" / f"{image_id}.jpg"
            quality = rng.randint(38, 72)
            params["jpeg_quality"] = quality
            save_jpeg(image, aug_path, quality)
            rows.append(
                {
                    "image_id": image_id,
                    "parent_original_id": parent_id,
                    "split": "train",
                    "label": "unhealthy",
                    "source": "new_downloads_dry",
                    "source_class": "generic_dried_plant",
                    "visual_issue": "dried_wilted",
                    "group_id": group_id,
                    "is_augmented": "1",
                    "augmentation": json.dumps(params, separators=(",", ":")),
                    "local_path": aug_path.relative_to(OUTPUT).as_posix(),
                    "width": "320",
                    "height": "240",
                    "jpeg_quality": str(quality),
                    "sha256": sha256(aug_path),
                }
            )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(f"{row['split']}/{row['label']}" for row in rows)
    originals = Counter(
        f"{row['split']}/{row['label']}" for row in rows if row["is_augmented"] == "0"
    )
    summary = {
        "purpose": "deliberate dried-plant domain-overfit health dataset",
        "image_size": [320, 240],
        "new_dried_originals": len(NEW_IMAGES),
        "new_augmentations_per_original": 7,
        "total_images": len(rows),
        "counts": dict(sorted(counts.items())),
        "original_counts": dict(sorted(originals.items())),
        "validation_and_test_unchanged": True,
        "warning": "This run intentionally overweights the new dried-plant appearance.",
    }
    (OUTPUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUTPUT / "README.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Deliberate dried-plant overfit extension\n\n"
            "Twelve manually reviewed dried/wilted plant screenshots were added only to training, "
            "with seven ESP-style variants each. Validation and test were not changed.\n"
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
