"""Download a small, explicitly curated generic-unhealthy review pool.

The filenames below come from narrow Wikimedia Commons condition categories.
They are kept as one binary class; ``visual_issue`` exists only for auditing
coverage and is not intended as a model target.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import re
import time
from pathlib import Path

import imagehash
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


API = "https://commons.wikimedia.org/w/api.php"
OUTPUT = Path("generic_unhealthy_candidates")
ALLOWED_LICENSE_MARKERS = ("cc0", "cc by", "public domain", "pdm")

# Explicit titles prevent broad text search from returning factories, insects,
# documents, furniture, and landscapes that merely contain matching words.
SELECTIONS = [
    # Dry, dead, heat- or water-stressed plants.
    ("dry_or_dead", "File:Australia shrub plant - dying due to having too much sunlight.jpg"),
    ("dry_or_dead", "File:Dead plant grown in plastic bottle, April 2013.jpg"),
    ("dry_or_dead", "File:Dead plant in pots.jpg"),
    ("dry_or_dead", "File:Plants affected by Below Freezing Temps.jpg"),
    ("dry_or_dead", "File:Dried up potted plant with the flag of the United States at the Wheels of Yesteryear Vintage Auto Museum May 2022.jpg"),
    ("dry_or_dead", "File:2023-06-04 16 12 41 A forsythia bush wilting during a dry spell along Barbara Lane in the Mountainview section of Ewing Township, Mercer County, New Jersey.jpg"),
    ("dry_or_dead", "File:Beta vulgaris subsp. vulgaris var. altissima drought (02).jpg"),
    ("dry_or_dead", "File:Beta vulgaris subsp. vulgaris var. altissima drought (04).jpg"),
    # Obvious yellowing/chlorosis.
    ("yellow_or_chlorotic", "File:Capsicum annuum clorosis.jpg"),
    ("yellow_or_chlorotic", "File:Chlorose hortensia.jpg"),
    ("yellow_or_chlorotic", "File:Chlorosis-1-selvapuram-coimbatore-India.jpg"),
    ("yellow_or_chlorotic", "File:Chlorosis-2-selvapuram-coimbatore-India.jpg"),
    ("yellow_or_chlorotic", "File:Chlorosis-3-selvapuram-coimbatore-India.jpg"),
    ("yellow_or_chlorotic", "File:ChloroticBlueberryLeaf.JPG"),
    ("yellow_or_chlorotic", "File:Citrus limon chlorosis.jpg"),
    ("yellow_or_chlorotic", "File:Ficus benjamina with chlorosis (1 of 2).jpg"),
    ("yellow_or_chlorotic", "File:Iron chlorosis on young rose leaves in a garden, by baby-bear.org.jpg"),
    # Clearly damaged or spotted foliage.
    ("leaf_damage", "File:Coleroa robertiani.jpg"),
    ("leaf_damage", "File:Leaf spot grey necrosis.jpg"),
    ("leaf_damage", "File:Ridderzuring bladvlekkenziekte (Rumex obtusifolius with leaf spot).jpg"),
    ("leaf_damage", "File:08 août 2019 - Trous d'altises sur le chou.jpg"),
    ("leaf_damage", "File:2018-05-13 (175) Damaged leaves at Bichlhäusl in Frankenfels, Austria.jpg"),
    ("leaf_damage", "File:Avocado leaves.JPG"),
    # Whole or potted plants with strong visible wilt.
    ("wilted", "File:Awa Pythium Root Rot -- wilt on young plant.jpg"),
    ("wilted", "File:Bacterial wilt 1 (5684778520).jpg"),
    ("wilted", "File:Bacterial wilt of pepper (9159744719).jpg"),
    ("wilted", "File:Photo-of-a-real-wilting-cucumber-in-greenhouse.jpg"),
]


def clean_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value or "")).strip()


def get_with_retry(session: requests.Session, url: str, **kwargs: object) -> requests.Response:
    """Retry only temporary Wikimedia throttling; remain strictly bounded."""
    for attempt in range(4):
        response = session.get(url, **kwargs)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        if attempt == 3:
            response.raise_for_status()
        time.sleep(8 * (attempt + 1))
    raise RuntimeError("unreachable")


def shorten(value: str, length: int = 38) -> str:
    value = value.removeprefix("File:")
    return value if len(value) <= length else value[: length - 1] + "…"


def build_sheet(rows: list[dict[str, str]]) -> None:
    thumb_w, thumb_h = 320, 240
    label_h, columns = 52, 4
    cell_w, cell_h = thumb_w, thumb_h + label_h
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, sheet_rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        with Image.open(OUTPUT / row["local_path"]) as opened:
            tile = ImageOps.fit(opened.convert("RGB"), (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y + label_h))
        draw.text((x + 5, y + 4), f"#{index + 1:02d} {row['visual_issue']}", fill="black", font=font)
        draw.text((x + 5, y + 24), shorten(row["title"]), fill="black", font=font)
    sheet.save(OUTPUT / "contact_sheet.jpg", "JPEG", quality=88)


def main() -> int:
    raw = OUTPUT / "raw" / "unhealthy"
    metadata = OUTPUT / "metadata"
    raw.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "SylvanTinyMLDataset/1.0 (non-commercial research)"
    rows: list[dict[str, str]] = []

    for visual_issue, title in SELECTIONS:
        response = get_with_retry(
            session,
            API,
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": 1280,
            },
            timeout=30,
        )
        page = next(iter(response.json().get("query", {}).get("pages", {}).values()), {})
        info_list = page.get("imageinfo") or []
        if not info_list:
            print(f"SKIP missing: {title}")
            continue
        info = info_list[0]
        if info.get("mime") not in {"image/jpeg", "image/png"}:
            print(f"SKIP non-raster: {title}")
            continue
        ext = info.get("extmetadata", {})
        license_name = clean_html(ext.get("LicenseShortName", {}).get("value", ""))
        if not any(marker in license_name.lower() for marker in ALLOWED_LICENSE_MARKERS):
            print(f"SKIP license {license_name!r}: {title}")
            continue
        image_url = info.get("thumburl") or info.get("url")
        image_response = get_with_retry(session, image_url, timeout=45)
        source_payload = image_response.content
        with Image.open(io.BytesIO(source_payload)) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            if width < 320 or height < 240:
                print(f"SKIP too small {width}x{height}: {title}")
                continue
            jpeg = io.BytesIO()
            rgb.save(jpeg, "JPEG", quality=92)
            jpeg_payload = jpeg.getvalue()

        page_id = str(page.get("pageid", ""))
        image_id = hashlib.sha256(f"{page_id}|{title}".encode("utf-8")).hexdigest()[:24]
        destination = raw / f"{image_id}.jpg"
        destination.write_bytes(jpeg_payload)
        row = {
            "sample_no": str(len(rows) + 1),
            "image_id": image_id,
            "binary_label": "unhealthy",
            "visual_issue": visual_issue,
            "verification_status": "candidate_user_review",
            "title": title,
            "source_page_url": info.get("descriptionurl", ""),
            "image_url": image_url,
            "license": license_name,
            "license_url": clean_html(ext.get("LicenseUrl", {}).get("value", "")),
            "attribution": clean_html(ext.get("Artist", {}).get("value", "")),
            "local_path": destination.relative_to(OUTPUT).as_posix(),
            "width": str(width),
            "height": str(height),
            "sha256": hashlib.sha256(jpeg_payload).hexdigest(),
            "phash": str(imagehash.phash(rgb)),
            "review_decision": "",
            "review_notes": "",
        }
        rows.append(row)
        print(f"OK {len(rows):02d}: {visual_issue} | {title}")
        time.sleep(0.75)

    fields = list(rows[0]) if rows else ["image_id"]
    for output_csv in (metadata / "images.csv", OUTPUT / "review.csv"):
        with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if rows:
        build_sheet(rows)
    print(f"Downloaded {len(rows)} curated generic-unhealthy candidates to {OUTPUT}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
