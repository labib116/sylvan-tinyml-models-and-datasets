"""Create compact labeled review sheets from accepted raw pilot images."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def build_sheet(source: Path, destination: Path) -> int:
    files = sorted(path for path in source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        return 0
    tile_width, tile_height, label_height = 320, 280, 36
    columns = min(5, len(files))
    rows = (len(files) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, path in enumerate(files):
        with Image.open(path) as image:
            preview = ImageOps.contain(image.convert("RGB"), (tile_width - 12, tile_height - 12))
        column, row = index % columns, index // columns
        x = column * tile_width + (tile_width - preview.width) // 2
        y = row * (tile_height + label_height) + (tile_height - preview.height) // 2
        sheet.paste(preview, (x, y))
        label = f"{index + 1}: {path.stem[:12]}"
        draw.text((column * tile_width + 8, row * (tile_height + label_height) + tile_height + 6), label, fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=90)
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    raw = args.dataset / "raw"
    output = args.dataset / "contact_sheets"
    for plant_dir in sorted(path for path in raw.iterdir() if path.is_dir()):
        for label_dir in sorted(path for path in plant_dir.iterdir() if path.is_dir()):
            destination = output / f"{plant_dir.name}_{label_dir.name}.jpg"
            count = build_sheet(label_dir, destination)
            if count:
                print(f"{destination}: {count} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
