"""Small Wikimedia Commons image-search/download smoke test.

Example:
    py basic_image_scraper.py --query "tree image" --count 5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "PlantHealthDatasetSmokeTest/1.0 (educational dataset builder)"
MAX_CANDIDATES = 20


def search_images(query: str) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",  # Wikimedia's file namespace
        "gsrlimit": str(MAX_CANDIDATES),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "1200",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    return payload.get("query", {}).get("pages", [])


def image_extension(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return None


def metadata_value(metadata: dict, key: str) -> str:
    value = metadata.get(key, {})
    text = str(value.get("value", "")) if isinstance(value, dict) else ""
    return re.sub(r"<[^>]+>", " ", text).strip()


def download_images(query: str, count: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    for page in search_images(query):
        if len(rows) >= count:
            break
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        image_url = info.get("thumburl") or info.get("url")
        source_url = info.get("descriptionurl", "")
        if not image_url:
            continue

        try:
            request = urllib.request.Request(image_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=20) as response:
                # This smoke test has a firm 15 MB per-image safety cap.
                data = response.read(15_000_001)
            if len(data) > 15_000_000:
                print(f"Skipped oversized image: {image_url}")
                continue
            extension = image_extension(data)
            if extension is None:
                print(f"Skipped unsupported/non-image response: {image_url}")
                continue
        except Exception as error:
            print(f"Download failed: {image_url} ({error})")
            continue

        number = len(rows) + 1
        filename = f"{number:02d}_{page.get('pageid', 'image')}{extension}"
        (output_dir / filename).write_bytes(data)
        metadata = info.get("extmetadata", {})
        rows.append(
            {
                "filename": filename,
                "query": query,
                "title": str(page.get("title", "")).removeprefix("File:"),
                "image_url": image_url,
                "source_page_url": source_url,
                "width": info.get("thumbwidth") or info.get("width") or "",
                "height": info.get("thumbheight") or info.get("height") or "",
                "license": metadata_value(metadata, "LicenseShortName"),
                "license_url": metadata_value(metadata, "LicenseUrl"),
                "artist": metadata_value(metadata, "Artist"),
            }
        )
        print(f"Downloaded {number}/{count}: {filename}")

    metadata_path = output_dir / "metadata.csv"
    fields = [
        "filename",
        "query",
        "title",
        "image_url",
        "source_page_url",
        "width",
        "height",
        "license",
        "license_url",
        "artist",
    ]
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a few Wikimedia Commons search results.")
    parser.add_argument("--query", default="tree image")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("downloaded_images/tree_image"))
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error("--count must be between 1 and 20")
    downloaded = download_images(args.query, args.count, args.output)
    print(f"Finished: {downloaded}/{args.count} images saved in {args.output.resolve()}")
    return 0 if downloaded == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
