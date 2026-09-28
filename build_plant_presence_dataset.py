"""Build a grouped plant/not-plant dataset for an ESP32-CAM robot."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


SEED = 20260927
SIZE = (320, 240)
ROOT = Path(__file__).resolve().parent
PLANT_DATASET = ROOT / "tinyml_contextual_plant_dataset"
ROBOT_SOURCE = Path(r"C:\Users\risal\Downloads")
WEB_SOURCE = ROOT / "nonplant_candidates" / "raw" / "not_plant"
OUTPUT = ROOT / "tinyml_plant_presence_dataset"

# Adjacent frames stay in one split. The byte-identical "39 (1)" copy is omitted.
ROBOT_SPLITS = {
    "train": [
        "sylvan-sample-12.jpg",
        "sylvan-sample-33.jpg",
        "sylvan-sample-34.jpg",
        "sylvan-sample-35.jpg",
        "sylvan-sample-36.jpg",
        "sylvan-sample-39.jpg",
    ],
    "validation": ["sylvan-sample-40.jpg", "sylvan-sample-41.jpg"],
    "test": ["sylvan-sample-43.jpg", "sylvan-sample-44.jpg"],
}

# Manually checked: no visible living plant. These support training diversity only.
SAFE_WEB_NEGATIVES = [
    "np_0447c92534521ad114ea.jpg",
    "np_0e32f2302199928faf7f.jpg",
    "np_166f860d0f68c0fbc251.jpg",
    "np_23aa3efba6af3a9f6243.jpg",
    "np_30dcc216d5d46ce2ca29.jpg",
    "np_486fed0859dd9b8b119a.jpg",
    "np_4c552c547e9de59e391a.jpg",
    "np_5176a402e78fc220b6aa.jpg",
    "np_6276958b20c7fb7579e6.jpg",
    "np_683c6efd83ddfc5898e1.jpg",
    "np_78b7ec24b2085c963c30.jpg",
    "np_88b2558d9d8a5230ba0e.jpg",
    "np_8d347303e5ce25060909.jpg",
    "np_915f8ab71627c1a88b59.jpg",
    "np_956e0d0ff6edc2c8f592.jpg",
    "np_b256212bbb28abdfe516.jpg",
    "np_e9bf4d61355bd1186574.jpg",
    "np_f093663f12d8d0bd4fb1.jpg",
    "np_f38f75f00af33aa94fd3.jpg",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS)


def augment(image: Image.Image, rng: random.Random) -> tuple[Image.Image, dict[str, object]]:
    zoom = rng.uniform(1.00, 1.18)
    scaled = image.resize(
        (round(SIZE[0] * zoom), round(SIZE[1] * zoom)),
        Image.Resampling.BILINEAR,
    )
    left = rng.randrange(0, scaled.width - SIZE[0] + 1)
    top = rng.randrange(0, scaled.height - SIZE[1] + 1)
    result = scaled.crop((left, top, left + SIZE[0], top + SIZE[1]))
    flip = rng.random() < 0.5
    if flip:
        result = ImageOps.mirror(result)
    angle = rng.uniform(-4.0, 4.0)
    result = result.rotate(angle, resample=Image.Resampling.BILINEAR)
    brightness = rng.uniform(0.72, 1.22)
    contrast = rng.uniform(0.75, 1.25)
    color = rng.uniform(0.82, 1.15)
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(color)
    blur = rng.uniform(0.0, 1.0)
    if blur > 0.25:
        result = result.filter(ImageFilter.GaussianBlur(blur))
    noise_sigma = rng.uniform(1.0, 8.0)
    noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
    array = np.asarray(result, dtype=np.float32)
    array += noise_rng.normal(0.0, noise_sigma, array.shape)
    result = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    params = {
        "zoom": round(zoom, 4),
        "crop_left": left,
        "crop_top": top,
        "horizontal_flip": flip,
        "rotation_degrees": round(angle, 3),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "color": round(color, 4),
        "blur_radius": round(blur, 3),
        "noise_sigma": round(noise_sigma, 3),
    }
    return result, params


def save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=quality, optimize=True)


def add_row(
    rows: list[dict[str, object]],
    path: Path,
    *,
    image_id: str,
    parent_id: str,
    split: str,
    label: str,
    source: str,
    group_id: str,
    is_augmented: bool,
    augmentation: object,
) -> None:
    rows.append(
        {
            "image_id": image_id,
            "parent_original_id": parent_id,
            "split": split,
            "label": label,
            "source": source,
            "group_id": group_id,
            "is_augmented": int(is_augmented),
            "augmentation": json.dumps(augmentation, separators=(",", ":")) if augmentation else "none",
            "local_path": path.relative_to(OUTPUT).as_posix(),
            "width": SIZE[0],
            "height": SIZE[1],
            "sha256": sha256(path),
        }
    )


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing dataset: {OUTPUT}")
    for split in ("train", "validation", "test"):
        for label in ("not_plant", "plant"):
            (OUTPUT / "images" / split / label).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    # Pool healthy and unhealthy contextual images into the single plant class.
    with (PLANT_DATASET / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for source_row in csv.DictReader(handle):
            source_path = PLANT_DATASET / source_row["local_path"]
            image_id = f"plant_{source_row['image_id']}"
            destination = OUTPUT / "images" / source_row["split"] / "plant" / f"{image_id}.jpg"
            shutil.copy2(source_path, destination)
            add_row(
                rows,
                destination,
                image_id=image_id,
                parent_id=f"plant_{source_row['parent_original_id']}",
                split=source_row["split"],
                label="plant",
                source=f"contextual_{source_row['source']}",
                group_id=f"plant:{source_row['group_id']}",
                is_augmented=source_row["is_augmented"] == "1",
                augmentation=source_row["augmentation"],
            )

    negative_sources: list[tuple[Path, str, str, str]] = []
    for split, names in ROBOT_SPLITS.items():
        for name in names:
            negative_sources.append((ROBOT_SOURCE / name, split, "robot_esp32", f"robot:{Path(name).stem}"))
    for name in SAFE_WEB_NEGATIVES:
        negative_sources.append((WEB_SOURCE / name, "train", "web_auxiliary", f"web:{Path(name).stem}"))

    missing = [str(path) for path, *_ in negative_sources if not path.is_file()]
    if missing:
        raise SystemExit("Missing source images:\n" + "\n".join(missing))

    rng = random.Random(SEED)
    for index, (source_path, split, source, group_id) in enumerate(negative_sources):
        parent_id = f"neg_{index:03d}_{source_path.stem.replace(' ', '_')}"
        base = normalized_image(source_path)
        destination = OUTPUT / "images" / split / "not_plant" / f"{parent_id}.jpg"
        quality = 80 if source == "robot_esp32" else 68
        save_jpeg(base, destination, quality)
        add_row(
            rows,
            destination,
            image_id=parent_id,
            parent_id=parent_id,
            split=split,
            label="not_plant",
            source=source,
            group_id=group_id,
            is_augmented=False,
            augmentation=None,
        )
        if split == "train":
            for aug_index in range(1, 6):
                augmented, params = augment(base, rng)
                aug_id = f"{parent_id}__aug{aug_index:02d}"
                aug_path = OUTPUT / "images" / split / "not_plant" / f"{aug_id}.jpg"
                jpeg_quality = rng.randint(38, 72)
                params["jpeg_quality"] = jpeg_quality
                save_jpeg(augmented, aug_path, jpeg_quality)
                add_row(
                    rows,
                    aug_path,
                    image_id=aug_id,
                    parent_id=parent_id,
                    split=split,
                    label="not_plant",
                    source=source,
                    group_id=group_id,
                    is_augmented=True,
                    augmentation=params,
                )

    fields = list(rows[0])
    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(f"{row['split']}/{row['label']}" for row in rows)
    original_counts = Counter(
        f"{row['split']}/{row['label']}" for row in rows if not row["is_augmented"]
    )
    summary = {
        "purpose": "binary plant presence detection from an ESP32-CAM robot",
        "classes": ["not_plant", "plant"],
        "image_size": list(SIZE),
        "total_images": len(rows),
        "counts": dict(sorted(counts.items())),
        "original_counts": dict(sorted(original_counts.items())),
        "robot_originals": sum(row["source"] == "robot_esp32" and not row["is_augmented"] for row in rows),
        "web_auxiliary_originals": sum(row["source"] == "web_auxiliary" and not row["is_augmented"] for row in rows),
        "plantvillage_included": False,
        "leakage_control": "augmentations and adjacent robot frames remain with their parent group",
        "limitations": [
            "only ten unique real robot negative frames are currently available",
            "validation and test each contain only two not-plant robot frames",
            "more independent robot runs are required before deployment",
        ],
    }
    (OUTPUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "labels.txt").write_text("not_plant\nplant\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# ESP32-CAM Plant-Presence Dataset\n\n"
        "Binary labels: `not_plant` and `plant`. Healthy and unhealthy contextual plant images are pooled "
        "into `plant`. Real floor-level ESP32-CAM images anchor the negative class. Clearly plant-free "
        "web images are training-only auxiliary negatives. Training augmentation simulates crop, blur, "
        "noise, exposure/color variation, and JPEG degradation. Validation and test are untouched.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
