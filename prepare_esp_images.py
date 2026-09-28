"""Create camera-matched base images and a small gritty augmentation preview.

The accepted originals remain untouched.  Base images match the firmware's
PSRAM path (320x240, software JPEG quality 80).  Gritty images are previews
only; final augmentation should run after source-grouped dataset splitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


def jpeg_roundtrip(image: Image.Image, quality: int) -> tuple[Image.Image, int]:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality, optimize=False)
    size = buffer.tell()
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy(), size


def camera_base(image: Image.Image, width: int, height: int, quality: int) -> tuple[Image.Image, int]:
    cropped = ImageOps.fit(
        image.convert("RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    return jpeg_roundtrip(cropped, quality)


def gritty_variant(image: Image.Image, rng: random.Random) -> tuple[Image.Image, dict[str, str]]:
    scale = rng.uniform(0.58, 0.88)
    small = image.resize(
        (max(32, round(image.width * scale)), max(24, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    result = small.resize(image.size, Image.Resampling.BILINEAR)
    brightness = rng.uniform(0.82, 1.16)
    contrast = rng.uniform(0.86, 1.18)
    saturation = rng.uniform(0.90, 1.08)
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(saturation)

    blur_radius = rng.uniform(0.0, 0.65)
    if blur_radius > 0.25:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    array = np.asarray(result, dtype=np.float32)
    noise_sigma = rng.uniform(2.0, 7.0)
    np_rng = np.random.default_rng(rng.getrandbits(64))
    array += np_rng.normal(0.0, noise_sigma, size=array.shape)
    result = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    jpeg_quality = rng.randint(55, 78)
    result, _ = jpeg_roundtrip(result, jpeg_quality)
    settings = {
        "downscale": f"{scale:.3f}",
        "brightness": f"{brightness:.3f}",
        "contrast": f"{contrast:.3f}",
        "saturation": f"{saturation:.3f}",
        "blur_radius": f"{blur_radius:.3f}",
        "noise_sigma": f"{noise_sigma:.3f}",
        "jpeg_quality": str(jpeg_quality),
    }
    return result, settings


def choose_preview(rows: list[dict[str, str]], per_label: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label in ("healthy", "unhealthy"):
        candidates = [row for row in rows if row["proposed_health_label"] == label]
        by_plant: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates:
            by_plant[row["plant_id"]].append(row)
        plants = list(by_plant)
        rng.shuffle(plants)
        for plant in plants[:per_label]:
            selected.append(rng.choice(by_plant[plant]))
    return selected


def make_preview(
    entries: list[tuple[dict[str, str], Image.Image, Image.Image, Image.Image]],
    path: Path,
    width: int,
    height: int,
) -> None:
    label_height = 42
    columns = 3
    rows = len(entries)
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for index, (metadata, source, base, gritty) in enumerate(entries):
        y = index * (height + label_height)
        sheet.paste(source, (0, y))
        sheet.paste(base, (width, y))
        sheet.paste(gritty, (width * 2, y))
        title = f"{metadata['plant_id']} | {metadata['proposed_health_label']} | {metadata['image_id'][:8]}"
        draw.text((5, y + height + 4), title, fill="black", font=font)
        draw.text((5, y + height + 22), "source crop", fill="#444", font=font)
        draw.text((width + 5, y + height + 22), "320x240 q80", fill="#444", font=font)
        draw.text((width * 2 + 5, y + height + 22), "gritty training preview", fill="#444", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "JPEG", quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("whole_csv_candidates"))
    parser.add_argument("--output", type=Path, default=Path("esp_training_base"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--preview-per-label", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    with (args.dataset / "metadata" / "images.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    manifest_rows: list[dict[str, str | int]] = []
    preview_ids = {row["image_id"] for row in choose_preview(rows, args.preview_per_label, args.seed)}
    previews: list[tuple[dict[str, str], Image.Image, Image.Image, Image.Image]] = []

    for row in rows:
        source_path = args.dataset / Path(row["local_path"])
        with Image.open(source_path) as opened:
            source_image = opened.convert("RGB")
            base, jpeg_bytes = camera_base(source_image, args.width, args.height, args.jpeg_quality)
            source_crop = ImageOps.fit(
                source_image,
                (args.width, args.height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        destination = (
            args.output
            / "images"
            / row["proposed_health_label"]
            / row["plant_id"]
            / f"{row['image_id']}.jpg"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        base.save(destination, "JPEG", quality=args.jpeg_quality, optimize=False)
        manifest_rows.append(
            {
                "source_image_id": row["image_id"],
                "plant_id": row["plant_id"],
                "label": row["proposed_health_label"],
                "source_local_path": row["local_path"],
                "processed_local_path": destination.relative_to(args.output).as_posix(),
                "crop": "center_fit_4x3",
                "width": args.width,
                "height": args.height,
                "jpeg_quality": args.jpeg_quality,
                "jpeg_bytes": jpeg_bytes,
            }
        )
        if row["image_id"] in preview_ids:
            seed_bytes = hashlib.sha256((row["image_id"] + str(args.seed)).encode()).digest()[:8]
            grit_rng = random.Random(int.from_bytes(seed_bytes, "big"))
            gritty, _ = gritty_variant(base, grit_rng)
            previews.append((row, source_crop, base, gritty))

    fields = list(manifest_rows[0])
    with (args.output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    make_preview(previews, args.output / "gritty_preview.jpg", args.width, args.height)
    print(f"Created {len(manifest_rows)} camera-matched base images and {len(previews)} preview rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
