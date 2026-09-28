"""Build a leakage-aware, ESP32-CAM-ready healthy/unhealthy dataset.

Inputs:
* reviewed field/garden images from whole_csv_candidates
* reviewed generic-unhealthy Commons images
* deterministic PlantVillage subset

Only training originals are augmented. Validation and test images are untouched
apart from the common 320x240 center crop and JPEG encoding.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "tinyml_plant_health_dataset"
SEED = 20260927
SIZE = (320, 240)
BASE_JPEG_QUALITY = 80
FIELD_HEALTHY_TARGET = 120
AUGMENTATIONS_PER_TRAIN_ORIGINAL = 1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_field_healthy(rows: list[dict[str, str]], target: int, rng: random.Random) -> list[dict[str, str]]:
    by_plant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_plant[row.get("plant_id", "unknown")].append(row)
    for items in by_plant.values():
        rng.shuffle(items)
    chosen: list[dict[str, str]] = []
    plant_ids = sorted(by_plant)
    while len(chosen) < target:
        progressed = False
        for plant_id in plant_ids:
            if by_plant[plant_id] and len(chosen) < target:
                chosen.append(by_plant[plant_id].pop())
                progressed = True
        if not progressed:
            break
    if len(chosen) != target:
        raise RuntimeError(f"Only found {len(chosen)} field healthy images; requested {target}")
    return chosen


def field_group(row: dict[str, str]) -> str:
    provider_id = row.get("provider_result_id", "")
    if provider_id.startswith("inat:"):
        parts = provider_id.split(":")
        if len(parts) >= 2:
            return f"field:inat:{parts[1]}"
    return f"field:{row['image_id']}"


def generic_group(row: dict[str, str]) -> str:
    number = int(row["sample_no"])
    if number in {7, 8}:
        return "generic:beta_drought_series"
    if number in {11, 12, 13}:
        return "generic:chlorosis_selvapuram_series"
    return f"generic:{row['image_id']}"


def gather_records(rng: random.Random) -> list[dict[str, str]]:
    field_rows = read_csv(ROOT / "whole_csv_candidates" / "metadata" / "images.csv")
    field_healthy_all = [row for row in field_rows if row.get("proposed_health_label") == "healthy"]
    field_unhealthy = [row for row in field_rows if row.get("proposed_health_label") == "unhealthy"]
    field_healthy = choose_field_healthy(field_healthy_all, FIELD_HEALTHY_TARGET, rng)

    records: list[dict[str, str]] = []
    for row in field_healthy + field_unhealthy:
        label = row["proposed_health_label"]
        records.append(
            {
                "original_id": row["image_id"],
                "label": label,
                "source": "field_garden",
                "source_class": row.get("plant_id", ""),
                "visual_issue": row.get("common_disease", "") if label == "unhealthy" else "healthy",
                "group_id": field_group(row),
                "source_path": str(ROOT / "whole_csv_candidates" / row["local_path"]),
                "source_page_url": row.get("source_page_url", ""),
                "license": row.get("license", ""),
                "attribution": row.get("attribution", ""),
            }
        )

    generic_rows = read_csv(ROOT / "generic_unhealthy_candidates" / "accepted.csv")
    for row in generic_rows:
        records.append(
            {
                "original_id": "gu_" + row["image_id"],
                "label": "unhealthy",
                "source": "generic_commons",
                "source_class": "generic_unhealthy",
                "visual_issue": row.get("visual_issue", ""),
                "group_id": generic_group(row),
                "source_path": str(ROOT / "generic_unhealthy_candidates" / row["local_path"]),
                "source_page_url": row.get("source_page_url", ""),
                "license": row.get("license", ""),
                "attribution": row.get("attribution", ""),
            }
        )

    pv_rows = read_csv(ROOT / "plantvillage_subset" / "metadata" / "images.csv")
    for row in pv_rows:
        records.append(
            {
                "original_id": row["image_id"],
                "label": row["binary_label"],
                "source": "plantvillage",
                "source_class": row.get("source_class", ""),
                "visual_issue": "healthy" if row["binary_label"] == "healthy" else "diseased_leaf",
                "group_id": row["source_group"],
                "source_path": str(ROOT / "plantvillage_subset" / row["local_path"]),
                "source_page_url": row.get("source_page_url", ""),
                "license": row.get("license", ""),
                "attribution": "PlantVillage / Mohanty, Hughes, and Salathé",
            }
        )

    missing = [record["source_path"] for record in records if not Path(record["source_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected source files; first={missing[0]}")
    return records


def stratum(record: dict[str, str]) -> tuple[str, str, str]:
    if record["source"] == "plantvillage":
        detail = record["source_class"]
    elif record["source"] == "generic_commons":
        detail = record["visual_issue"]
    else:
        detail = "field_all"
    return record["label"], record["source"], detail


def split_targets(count: int) -> dict[str, int]:
    if count >= 3:
        validation = max(1, round(count * 0.20))
        test = max(1, round(count * 0.20))
        train = count - validation - test
    elif count == 2:
        train, validation, test = 1, 0, 1
    else:
        train, validation, test = 1, 0, 0
    return {"train": train, "validation": validation, "test": test}


def assign_splits(records: list[dict[str, str]], rng: random.Random) -> None:
    by_stratum: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_stratum[stratum(record)].append(record)

    for items in by_stratum.values():
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for item in items:
            by_group[item["group_id"]].append(item)
        groups = list(by_group.values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        targets = split_targets(len(items))
        assigned = Counter()
        for group in groups:
            deficits = {split: targets[split] - assigned[split] for split in targets}
            split = max(("train", "validation", "test"), key=lambda name: (deficits[name], -assigned[name]))
            for item in group:
                item["split"] = split
            assigned[split] += len(group)

    seen: dict[str, str] = {}
    for record in records:
        previous = seen.setdefault(record["group_id"], record["split"])
        if previous != record["split"]:
            raise RuntimeError(f"Leakage: group {record['group_id']} occurs in {previous} and {record['split']}")


def camera_frame(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
    return ImageOps.fit(rgb, SIZE, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def augment(frame: Image.Image, rng: random.Random) -> tuple[Image.Image, int, dict[str, float | bool | int]]:
    image = frame.copy()
    flipped = rng.random() < 0.5
    if flipped:
        image = ImageOps.mirror(image)

    zoom = rng.uniform(1.00, 1.12)
    enlarged = image.resize((round(SIZE[0] * zoom), round(SIZE[1] * zoom)), Image.Resampling.BICUBIC)
    max_x = enlarged.width - SIZE[0]
    max_y = enlarged.height - SIZE[1]
    left = rng.randint(0, max_x) if max_x else 0
    top = rng.randint(0, max_y) if max_y else 0
    image = enlarged.crop((left, top, left + SIZE[0], top + SIZE[1]))

    angle = rng.uniform(-8.0, 8.0)
    corners = np.asarray(image, dtype=np.uint8)[[0, 0, -1, -1], [0, -1, 0, -1]]
    fill = tuple(int(value) for value in np.median(corners, axis=0))
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill)

    brightness = rng.uniform(0.78, 1.18)
    contrast = rng.uniform(0.82, 1.18)
    saturation = rng.uniform(0.78, 1.18)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)

    blur_radius = 0.0
    if rng.random() < 0.30:
        blur_radius = rng.uniform(0.25, 0.75)
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))

    noise_sigma = rng.uniform(2.0, 6.0)
    array = np.asarray(image, dtype=np.float32)
    noise_seed = rng.randrange(0, 2**32)
    noise_rng = np.random.default_rng(noise_seed)
    array = np.clip(array + noise_rng.normal(0.0, noise_sigma, array.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    jpeg_quality = rng.randint(48, 72)
    settings: dict[str, float | bool | int] = {
        "horizontal_flip": flipped,
        "zoom": round(zoom, 4),
        "rotation_degrees": round(angle, 3),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "saturation": round(saturation, 4),
        "blur_radius": round(blur_radius, 3),
        "noise_sigma": round(noise_sigma, 3),
        "noise_seed": noise_seed,
        "jpeg_quality": jpeg_quality,
    }
    return image, jpeg_quality, settings


def save_jpeg(image: Image.Image, path: Path, quality: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=quality, optimize=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_overview(rows: list[dict[str, str]]) -> None:
    originals = [row for row in rows if row["is_augmented"] == "0"]
    rng = random.Random(SEED + 99)
    selected: list[dict[str, str]] = []
    for split in ("train", "validation", "test"):
        for label in ("healthy", "unhealthy"):
            pool = [row for row in originals if row["split"] == split and row["label"] == label]
            selected.extend(rng.sample(pool, min(4, len(pool))))
    columns, thumb = 4, (320, 240)
    label_h = 34
    sheet_rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb[0], sheet_rows * (thumb[1] + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        x = (index % columns) * thumb[0]
        y = (index // columns) * (thumb[1] + label_h)
        with Image.open(OUTPUT / row["local_path"]) as opened:
            sheet.paste(opened.convert("RGB"), (x, y + label_h))
        draw.text((x + 4, y + 4), f"{row['split']} | {row['label']} | {row['source']}", fill="black", font=font)
    sheet.save(OUTPUT / "overview.jpg", "JPEG", quality=88)


def main() -> int:
    if (OUTPUT / "manifest.csv").exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUTPUT}")
    rng = random.Random(SEED)
    records = gather_records(rng)
    label_counts = Counter(record["label"] for record in records)
    if label_counts != Counter({"healthy": 160, "unhealthy": 160}):
        raise RuntimeError(f"Unexpected original balance: {label_counts}")
    assign_splits(records, rng)

    manifest: list[dict[str, str]] = []
    attribution: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        frame = camera_frame(Path(record["source_path"]))
        relative = Path("images") / record["split"] / record["label"] / f"{record['original_id']}.jpg"
        digest = save_jpeg(frame, OUTPUT / relative, BASE_JPEG_QUALITY)
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
                "width": str(SIZE[0]),
                "height": str(SIZE[1]),
                "jpeg_quality": str(BASE_JPEG_QUALITY),
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
                augmented, quality, settings = augment(frame, rng)
                aug_id = f"{record['original_id']}__aug{aug_index:02d}"
                aug_relative = Path("images") / "train" / record["label"] / f"{aug_id}.jpg"
                aug_digest = save_jpeg(augmented, OUTPUT / aug_relative, quality)
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
                        "width": str(SIZE[0]),
                        "height": str(SIZE[1]),
                        "jpeg_quality": str(quality),
                        "sha256": aug_digest,
                    }
                )
        if index % 50 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} originals")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (OUTPUT / "attribution.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attribution[0]))
        writer.writeheader()
        writer.writerows(attribution)

    original_counts = Counter((row["split"], row["label"]) for row in manifest if row["is_augmented"] == "0")
    final_counts = Counter((row["split"], row["label"]) for row in manifest)
    source_counts = Counter((record["source"], record["label"]) for record in records)
    summary = {
        "seed": SEED,
        "image_size": list(SIZE),
        "base_jpeg_quality": BASE_JPEG_QUALITY,
        "original_total": len(records),
        "final_total": len(manifest),
        "original_counts": {f"{split}/{label}": count for (split, label), count in sorted(original_counts.items())},
        "final_counts": {f"{split}/{label}": count for (split, label), count in sorted(final_counts.items())},
        "source_counts": {f"{source}/{label}": count for (source, label), count in sorted(source_counts.items())},
        "augmentation": {
            "train_only": True,
            "variants_per_train_original": AUGMENTATIONS_PER_TRAIN_ORIGINAL,
            "horizontal_flip_probability": 0.5,
            "zoom_range": [1.0, 1.12],
            "rotation_degrees": [-8, 8],
            "brightness_range": [0.78, 1.18],
            "contrast_range": [0.82, 1.18],
            "saturation_range": [0.78, 1.18],
            "blur_probability": 0.30,
            "gaussian_noise_sigma": [2.0, 6.0],
            "jpeg_quality_range": [48, 72],
        },
        "leakage_controls": [
            "split before augmentation",
            "all augmented variants stay with their parent in train",
            "iNaturalist observation groups remain in one split",
            "PlantVillage leaf groups remain in one split",
            "known related Commons series remain in one split",
        ],
    }
    (OUTPUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = f"""# TinyML Plant Health Binary Dataset

Binary labels: `healthy` and `unhealthy`.

All images are 320x240 RGB JPEGs, matching the ESP32-CAM QVGA capture path. The
dataset contains {len(records)} reviewed/source originals and {len(manifest)} files after one
training-only augmentation per training original. Validation and test images are not
augmented. See `dataset_summary.json` for exact counts and augmentation ranges.

Use `images/train`, `images/validation`, and `images/test` as fixed splits. Never
randomly resplit individual files because augmented variants share a parent image.

PlantVillage is a controlled-background leaf dataset and does not fully match rooftop
camera imagery. The mixed field/garden sources and ESP-style degradation reduce, but do
not eliminate, that domain gap. Final accuracy should be checked on untouched images
captured by the actual camera.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    make_overview(manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
