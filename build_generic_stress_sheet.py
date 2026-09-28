"""Build a numbered ESP-resolution contact sheet for generic stress candidates."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("generic_stress_candidates"))
    args = parser.parse_args()
    metadata_path = args.dataset / "metadata" / "images.csv"
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    width, height, header = 320, 240, 45
    columns = 5
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, sheet_rows * (height + header)), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    index_rows: list[dict[str, str | int]] = []

    for index, row in enumerate(rows, start=1):
        col = (index - 1) % columns
        line = (index - 1) // columns
        x, y = col * width, line * (height + header)
        with Image.open(args.dataset / Path(row["local_path"])) as opened:
            image = ImageOps.fit(opened.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=80)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            preview = encoded.convert("RGB").copy()
        sheet.paste(preview, (x, y + header))
        draw.rectangle((x, y, x + width - 1, y + header - 1), fill="white")
        draw.text((x + 5, y + 4), f"#{index:02d} {row['image_id'][:10]}", fill="black", font=font)
        subtitle = row.get("title") or Path(row.get("source_path", "")).name or row.get("plant_key", "")
        draw.text((x + 5, y + 23), subtitle[:42], fill="#444444", font=font)
        index_rows.append({"sample_no": index, **row, "review_decision": "", "review_notes": ""})

    sheet.save(args.dataset / "contact_sheet.jpg", "JPEG", quality=92)
    fields = list(index_rows[0])
    with (args.dataset / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote one contact sheet for {len(rows)} candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
