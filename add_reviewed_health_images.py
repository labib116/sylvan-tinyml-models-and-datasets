"""Add the manually reviewed September 29 images to the health training split.

This update is intentionally incremental: validation and test stay untouched,
and the new originals are not augmented. Re-running the script is refused once
the image IDs are present in the manifest.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from build_tinyml_binary_dataset import camera_frame, save_jpeg


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "tinyml_plant_health_dataset"
CURATED_HEALTHY = ROOT / "downloaded_images" / "inaturalist_curated" / "tomato" / "healthy"
CURATED_METADATA = CURATED_HEALTHY / "metadata.csv"
DOWNLOADS = Path(r"C:\Users\risal\Downloads")
JPEG_QUALITY = 80

HEALTHY_FILES = [
    "04_bde999e85ba7abcb.jpg",
    "05_53ef91d4d2f10f1b.jpg",
    "07_ebd570b4ee5dd48f.jpg",
    "09_13a86da0ec26a2aa.jpg",
    "10_f123bd4434fcd069.jpg",
]

UNHEALTHY_FILES = [
    "sylvan-sample-168.jpg",
    "sylvan-sample-174.jpg",
    "sylvan-sample-175.jpg",
    "sylvan-sample-179.jpg",
    "sylvan-sample-181.jpg",
    "sylvan-sample-182.jpg",
    "sylvan-sample-184.jpg",
    "sylvan-sample-185.jpg",
    "sylvan-sample-186.jpg",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    manifest_path = DATASET / "manifest.csv"
    attribution_path = DATASET / "attribution.csv"
    summary_path = DATASET / "dataset_summary.json"

    manifest_fields, manifest = read_csv(manifest_path)
    attribution_fields, attribution = read_csv(attribution_path)
    _, healthy_metadata_rows = read_csv(CURATED_METADATA)
    healthy_metadata = {row["filename"]: row for row in healthy_metadata_rows}

    missing = [
        str(path)
        for path in [
            *(CURATED_HEALTHY / name for name in HEALTHY_FILES),
            *(DOWNLOADS / name for name in UNHEALTHY_FILES),
        ]
        if not path.is_file()
    ]
    if missing:
        raise SystemExit("Missing reviewed images:\n" + "\n".join(missing))

    additions: list[dict[str, str | Path]] = []
    for filename in HEALTHY_FILES:
        metadata = healthy_metadata[filename]
        additions.append(
            {
                "image_id": Path(filename).stem,
                "filename": filename,
                "label": "healthy",
                "source": "field_garden",
                "source_class": "tomato",
                "visual_issue": "healthy",
                "group_id": f"field:inat:{metadata['observation_id']}",
                "source_path": CURATED_HEALTHY / filename,
                "source_page_url": metadata["source_page_url"],
                "license": metadata["license"],
                "attribution": metadata["attribution"],
            }
        )

    for filename in UNHEALTHY_FILES:
        sample_number = Path(filename).stem.removeprefix("sylvan-sample-")
        additions.append(
            {
                "image_id": Path(filename).stem,
                "filename": filename,
                "label": "unhealthy",
                "source": "esp32_cam",
                "source_class": "user_captured_plant",
                "visual_issue": "bare_branch_or_stem",
                "group_id": "esp32cam:branch_sequence:20260929",
                "source_path": DOWNLOADS / filename,
                "source_page_url": "",
                "license": "user-provided for project use",
                "attribution": "Project owner ESP32-CAM capture",
            }
        )

    existing_ids = {row["image_id"] for row in manifest}
    duplicate_ids = [str(row["image_id"]) for row in additions if row["image_id"] in existing_ids]
    if duplicate_ids:
        raise SystemExit(
            "Refusing to add image IDs already present in manifest:\n"
            + "\n".join(duplicate_ids)
        )

    for record in additions:
        image_id = str(record["image_id"])
        label = str(record["label"])
        destination = DATASET / "images" / "train" / label / str(record["filename"])
        if destination.exists():
            raise SystemExit(f"Refusing to overwrite existing image: {destination}")

        image = camera_frame(Path(record["source_path"]))
        digest = save_jpeg(image, destination, JPEG_QUALITY)
        manifest.append(
            {
                "image_id": image_id,
                "parent_original_id": image_id,
                "split": "train",
                "label": label,
                "source": str(record["source"]),
                "source_class": str(record["source_class"]),
                "visual_issue": str(record["visual_issue"]),
                "group_id": str(record["group_id"]),
                "is_augmented": "0",
                "augmentation": "none",
                "local_path": destination.relative_to(DATASET).as_posix(),
                "width": "320",
                "height": "240",
                "jpeg_quality": str(JPEG_QUALITY),
                "sha256": digest,
            }
        )
        attribution.append(
            {
                "original_id": image_id,
                "source": str(record["source"]),
                "source_page_url": str(record["source_page_url"]),
                "license": str(record["license"]),
                "attribution": str(record["attribution"]),
            }
        )

    write_csv(manifest_path, manifest_fields, manifest)
    write_csv(attribution_path, attribution_fields, attribution)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    originals = [row for row in manifest if row["is_augmented"] == "0"]
    summary["original_total"] = len(originals)
    summary["final_total"] = len(manifest)
    summary["original_counts"] = dict(
        sorted(Counter(f"{row['split']}/{row['label']}" for row in originals).items())
    )
    summary["final_counts"] = dict(
        sorted(Counter(f"{row['split']}/{row['label']}" for row in manifest).items())
    )
    summary["source_counts"] = dict(
        sorted(Counter(f"{row['source']}/{row['label']}" for row in originals).items())
    )
    summary["incremental_updates"] = [
        {
            "date": "2026-09-29",
            "split": "train",
            "healthy_originals": len(HEALTHY_FILES),
            "unhealthy_originals": len(UNHEALTHY_FILES),
            "augmented_variants": 0,
            "validation_and_test_unchanged": True,
            "note": "Manually reviewed green tomato plants and bare branch/stem ESP32-CAM frames.",
        }
    ]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme_path = DATASET / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace(
        "dataset contains 320 reviewed/source originals and 494 files after one\n"
        "training-only augmentation per training original.",
        "dataset initially contained 320 reviewed/source originals and 494 files after one\n"
        "training-only augmentation per original in the initial training split.",
    )
    marker = "## September 29 reviewed-image update"
    if marker not in readme:
        readme += (
            "\n## September 29 reviewed-image update\n\n"
            "Fourteen manually reviewed originals were added only to training: five green "
            "tomato-plant images labeled `healthy` and nine ESP32-CAM bare branch/stem "
            "frames labeled `unhealthy`. They were left unaugmented, and the fixed "
            "validation and test splits were not changed. The current trained model predates "
            "this incremental update; the existing model files remain unchanged.\n"
        )
    readme_path.write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "added": len(additions),
                "healthy": len(HEALTHY_FILES),
                "unhealthy": len(UNHEALTHY_FILES),
                "original_total": len(originals),
                "final_total": len(manifest),
                "final_counts": summary["final_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
