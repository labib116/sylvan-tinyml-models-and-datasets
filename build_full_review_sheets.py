"""Build one-row-per-plant contact sheets for a full conservative review."""

from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def camera_preview(path: Path, size: tuple[int, int], quality: int) -> Image.Image:
    with Image.open(path) as opened:
        fitted = ImageOps.fit(opened.convert("RGB"), size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    fitted.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("whole_csv_candidates"))
    parser.add_argument("--output", type=Path, default=Path("quality_review/full_pass"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--plant-rows-per-sheet", type=int, default=5)
    args = parser.parse_args()

    with (args.dataset / "metadata" / "images.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))

    manifest.sort(key=lambda row: (row["proposed_health_label"], row["plant_id"], row["image_id"]))
    indexed: list[dict[str, str | int]] = []
    for sample_no, row in enumerate(manifest, start=1):
        indexed.append({"sample_no": sample_no, **row})

    args.output.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_no",
        "image_id",
        "plant_id",
        "english_name",
        "proposed_health_label",
        "common_disease",
        "search_provider",
        "local_path",
        "review_decision",
        "review_reason",
    ]
    with (args.output / "review_all.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(indexed)

    font = ImageFont.load_default(size=15)
    header_height = 31
    columns = 5
    by_label: dict[str, dict[str, list[dict[str, str | int]]]] = defaultdict(lambda: defaultdict(list))
    for row in indexed:
        by_label[str(row["proposed_health_label"])][str(row["plant_id"])].append(row)

    sheet_count = 0
    for label in ("healthy", "unhealthy"):
        plant_groups = list(by_label.get(label, {}).items())
        for page_start in range(0, len(plant_groups), args.plant_rows_per_sheet):
            groups = plant_groups[page_start : page_start + args.plant_rows_per_sheet]
            sheet = Image.new(
                "RGB",
                (columns * args.width, len(groups) * (header_height + args.height)),
                "#eeeeee",
            )
            draw = ImageDraw.Draw(sheet)
            for row_index, (plant_id, entries) in enumerate(groups):
                y = row_index * (header_height + args.height)
                for column_index in range(columns):
                    x = column_index * args.width
                    if column_index >= len(entries):
                        draw.rectangle((x, y, x + args.width - 1, y + header_height + args.height - 1), fill="#dddddd")
                        continue
                    entry = entries[column_index]
                    path = args.dataset / Path(str(entry["local_path"]))
                    preview = camera_preview(path, (args.width, args.height), args.jpeg_quality)
                    sheet.paste(preview, (x, y + header_height))
                    title = f"#{int(entry['sample_no']):03d} {plant_id} {str(entry['image_id'])[:8]}"
                    draw.rectangle((x, y, x + args.width - 1, y + header_height - 1), fill="white")
                    draw.text((x + 5, y + 7), title, fill="black", font=font)
                    draw.rectangle(
                        (x, y, x + args.width - 1, y + header_height + args.height - 1),
                        outline="#777777",
                        width=1,
                    )
            sheet_count += 1
            page = page_start // args.plant_rows_per_sheet + 1
            sheet.save(args.output / f"{label}_{page:02d}.jpg", "JPEG", quality=92)

    print(f"Indexed {len(indexed)} active images into {sheet_count} review sheets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
