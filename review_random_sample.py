"""Build a reproducible random quality-review sample for the Sylvan dataset.

Raw files are never modified.  Each sampled image is center-cropped to the
camera's 4:3 aspect ratio, resized to the configured capture resolution, and
JPEG-encoded at the same quality used by the ESP32 firmware.  Contact sheets
show the source image beside that simulated camera input.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def stratified_sample(rows: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    """Prefer plant diversity, then fill remaining slots randomly."""
    by_plant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_plant[row["plant_id"]].append(row)

    plant_ids = list(by_plant)
    rng.shuffle(plant_ids)
    selected: list[dict[str, str]] = []
    used_ids: set[str] = set()

    for plant_id in plant_ids:
        if len(selected) >= count:
            break
        row = rng.choice(by_plant[plant_id])
        selected.append(row)
        used_ids.add(row["image_id"])

    remaining = [row for row in rows if row["image_id"] not in used_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    rng.shuffle(selected)
    return selected


def simulate_camera(image: Image.Image, width: int, height: int, quality: int) -> Image.Image:
    fitted = ImageOps.fit(
        image.convert("RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    buffer = io.BytesIO()
    fitted.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy()


def make_sheet(
    entries: list[tuple[int, dict[str, str], Image.Image, Image.Image]],
    destination: Path,
    width: int,
    height: int,
) -> None:
    font = ImageFont.load_default(size=15)
    label_height = 54
    pair_width = width * 2
    columns = 2
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * pair_width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for position, (sample_no, metadata, source_preview, camera_preview) in enumerate(entries):
        col = position % columns
        row = position // columns
        x = col * pair_width
        y = row * (height + label_height)
        sheet.paste(source_preview, (x, y))
        sheet.paste(camera_preview, (x + width, y))
        draw.line((x + width, y, x + width, y + height), fill="white", width=2)
        line1 = (
            f"#{sample_no:02d} {metadata['plant_id']} | "
            f"{metadata['proposed_health_label']} | {metadata['image_id'][:10]}"
        )
        line2 = "left: source framing   right: 4:3 camera simulation"
        draw.text((x + 6, y + height + 5), line1, fill="black", font=font)
        draw.text((x + 6, y + height + 27), line2, fill="#444444", font=font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("whole_csv_candidates"))
    parser.add_argument("--output", type=Path, default=Path("quality_review/random_sample"))
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--per-sheet", type=int, default=10)
    args = parser.parse_args()

    manifest_path = args.dataset / "metadata" / "images.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for label in ("healthy", "unhealthy"):
        candidates = [row for row in all_rows if row["proposed_health_label"] == label]
        selected.extend(stratified_sample(candidates, min(args.per_label, len(candidates)), rng))

    # Keep the output easy to inspect while retaining random order within labels.
    selected.sort(key=lambda row: (row["proposed_health_label"], row["image_id"]))
    args.output.mkdir(parents=True, exist_ok=True)

    review_fields = [
        "sample_no",
        "image_id",
        "plant_id",
        "english_name",
        "proposed_health_label",
        "search_provider",
        "common_disease",
        "local_path",
        "source_width",
        "source_height",
        "esp_width",
        "esp_height",
        "esp_jpeg_quality",
        "plant_visible",
        "label_plausible",
        "symptom_visible",
        "esp_detail_usable",
        "review_decision",
        "review_notes",
    ]
    review_rows: list[dict[str, str | int]] = []
    rendered: list[tuple[int, dict[str, str], Image.Image, Image.Image]] = []

    for sample_no, row in enumerate(selected, start=1):
        source_path = args.dataset / Path(row["local_path"])
        if source_path.suffix.lower() not in IMAGE_SUFFIXES or not source_path.is_file():
            continue
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
            camera = simulate_camera(source, args.width, args.height, args.jpeg_quality)
            source_preview = ImageOps.fit(
                source,
                (args.width, args.height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        camera_path = args.output / "esp_simulated" / f"{sample_no:02d}_{row['image_id']}.jpg"
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        camera.save(camera_path, "JPEG", quality=args.jpeg_quality)
        rendered.append((sample_no, row, source_preview, camera))
        review_rows.append(
            {
                "sample_no": sample_no,
                "image_id": row["image_id"],
                "plant_id": row["plant_id"],
                "english_name": row["english_name"],
                "proposed_health_label": row["proposed_health_label"],
                "search_provider": row["search_provider"],
                "common_disease": row["common_disease"],
                "local_path": row["local_path"],
                "source_width": row["width"],
                "source_height": row["height"],
                "esp_width": args.width,
                "esp_height": args.height,
                "esp_jpeg_quality": args.jpeg_quality,
                "plant_visible": "",
                "label_plausible": "",
                "symptom_visible": "",
                "esp_detail_usable": "",
                "review_decision": "",
                "review_notes": "",
            }
        )

    for offset in range(0, len(rendered), args.per_sheet):
        chunk = rendered[offset : offset + args.per_sheet]
        sheet_number = offset // args.per_sheet + 1
        make_sheet(chunk, args.output / f"contact_sheet_{sheet_number:02d}.jpg", args.width, args.height)

    with (args.output / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(review_rows)

    print(
        f"Wrote {len(review_rows)} samples, "
        f"{(len(rendered) + args.per_sheet - 1) // args.per_sheet} sheets, "
        f"seed={args.seed}, camera={args.width}x{args.height} JPEG quality {args.jpeg_quality}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
