"""Collect a bounded, licensed non-plant candidate pool from exact Commons categories."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import random
import re
import time
from pathlib import Path

import imagehash
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


API = "https://commons.wikimedia.org/w/api.php"
OUTPUT = Path("nonplant_candidates")
SEED = 20260927
CATEGORIES = {
    "chair": ("Quality images of chairs", 12),
    "fan": ("Desk fans", 8),
    "pedestal_fan": ("Pedestal fans", 6),
    "bottle": ("Plastic drinking bottles", 12),
    "door": ("Doors", 12),
    "table": ("Tables", 12),
    "shoe": ("Shoes", 12),
    "refrigerator": ("Refrigerators", 8),
    "television": ("Television sets", 8),
    "corridor": ("Corridors", 10),
}
ALLOWED_LICENSE_MARKERS = ("cc0", "cc by", "public domain", "pdm")
PLANT_WORDS = re.compile(
    r"\b(plant|flower|garden|tree|leaf|leaves|forest|vegetation|shrub|grass|herb|crop|farm)\b",
    re.IGNORECASE,
)


def clean_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value or "")).strip()


def get_retry(session: requests.Session, url: str, **kwargs: object) -> requests.Response:
    for attempt in range(5):
        response = session.get(url, **kwargs)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt == 4:
            response.raise_for_status()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def category_titles(session: requests.Session, category: str) -> list[str]:
    titles: list[str] = []
    continuation: str | None = None
    while len(titles) < 200:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": 6,
            "cmtype": "file",
            "cmlimit": 100,
        }
        if continuation:
            params["cmcontinue"] = continuation
        data = get_retry(session, API, params=params, timeout=45).json()
        titles.extend(item["title"] for item in data.get("query", {}).get("categorymembers", []))
        continuation = data.get("continue", {}).get("cmcontinue")
        if not continuation:
            break
    return titles


def image_info(session: requests.Session, title: str) -> tuple[dict, dict] | None:
    data = get_retry(
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
        timeout=45,
    ).json()
    page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
    infos = page.get("imageinfo") or []
    return (page, infos[0]) if infos else None


def make_sheets(rows: list[dict[str, str]]) -> None:
    columns, per_sheet = 5, 25
    thumb_w, thumb_h, header = 256, 192, 42
    font = ImageFont.load_default()
    for start in range(0, len(rows), per_sheet):
        batch = rows[start : start + per_sheet]
        sheet_rows = (len(batch) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_w, sheet_rows * (thumb_h + header)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(batch):
            x = (index % columns) * thumb_w
            y = (index // columns) * (thumb_h + header)
            with Image.open(OUTPUT / row["local_path"]) as opened:
                tile = ImageOps.fit(opened.convert("RGB"), (thumb_w, thumb_h), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x, y + header))
            draw.text((x + 4, y + 3), f"#{int(row['sample_no']):03d} {row['object_group']}", fill="black", font=font)
            short = row["title"].removeprefix("File:")[:35]
            draw.text((x + 4, y + 21), short, fill="black", font=font)
        sheet.save(OUTPUT / f"contact_sheet_{start // per_sheet + 1:02d}.jpg", "JPEG", quality=88)


def main() -> int:
    raw = OUTPUT / "raw" / "not_plant"
    metadata = OUTPUT / "metadata"
    raw.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "SylvanTinyMLDataset/1.0 (non-commercial research)"
    rng = random.Random(SEED)
    rows: list[dict[str, str]] = []
    known_phashes: list[imagehash.ImageHash] = []

    for object_group, (category, quota) in CATEGORIES.items():
        titles = category_titles(session, category)
        rng.shuffle(titles)
        accepted = 0
        for title in titles:
            if accepted >= quota:
                break
            if PLANT_WORDS.search(title):
                continue
            result = image_info(session, title)
            if not result:
                continue
            page, info = result
            if info.get("mime") not in {"image/jpeg", "image/png"}:
                continue
            if int(info.get("width", 0)) < 400 or int(info.get("height", 0)) < 300:
                continue
            ext = info.get("extmetadata", {})
            description = clean_html(ext.get("ImageDescription", {}).get("value", ""))
            if PLANT_WORDS.search(description):
                continue
            license_name = clean_html(ext.get("LicenseShortName", {}).get("value", ""))
            if not any(marker in license_name.lower() for marker in ALLOWED_LICENSE_MARKERS):
                continue
            url = info.get("thumburl") or info.get("url")
            try:
                payload = get_retry(session, url, timeout=60).content
                with Image.open(io.BytesIO(payload)) as opened:
                    rgb = opened.convert("RGB")
                    phash = imagehash.phash(rgb)
                    width, height = rgb.size
                    if any(phash - prior <= 4 for prior in known_phashes):
                        continue
                    converted = io.BytesIO()
                    rgb.save(converted, "JPEG", quality=92)
                    jpeg = converted.getvalue()
            except (OSError, requests.RequestException):
                continue
            image_id = "np_" + hashlib.sha256(f"{page.get('pageid')}|{title}".encode()).hexdigest()[:20]
            destination = raw / f"{image_id}.jpg"
            destination.write_bytes(jpeg)
            known_phashes.append(phash)
            accepted += 1
            rows.append(
                {
                    "sample_no": str(len(rows) + 1),
                    "image_id": image_id,
                    "binary_label": "not_plant",
                    "object_group": object_group,
                    "verification_status": "candidate_user_review",
                    "title": title,
                    "source_page_url": info.get("descriptionurl", ""),
                    "image_url": url,
                    "license": license_name,
                    "license_url": clean_html(ext.get("LicenseUrl", {}).get("value", "")),
                    "attribution": clean_html(ext.get("Artist", {}).get("value", "")),
                    "local_path": destination.relative_to(OUTPUT).as_posix(),
                    "width": str(width),
                    "height": str(height),
                    "sha256": hashlib.sha256(jpeg).hexdigest(),
                    "phash": str(phash),
                    "review_decision": "",
                    "review_notes": "",
                }
            )
            time.sleep(0.25)
        print(f"{object_group}: {accepted}/{quota}")

    fields = list(rows[0]) if rows else ["image_id"]
    for path in (metadata / "images.csv", OUTPUT / "review.csv"):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    make_sheets(rows)
    print(f"Downloaded {len(rows)} non-plant candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
