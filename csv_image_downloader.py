"""Download a small test batch using queries stored in the plant CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BangladeshGardenTinyML/0.1 (small dataset smoke test)"


def load_plant(csv_path: Path, plant_id: str) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["plant_id"].strip().casefold() == plant_id.casefold():
                return {key: (value or "").strip() for key, value in row.items()}
    raise ValueError(f"Plant ID not found in CSV: {plant_id}")


def language(query: str) -> str:
    return "bn" if any("\u0980" <= character <= "\u09ff" for character in query) else "en"


def search(query: str, limit: int = 30) -> list[dict]:
    parameters = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "1200",
    }
    url = API_URL + "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response).get("query", {}).get("pages", [])


def round_robin(groups: list[list[tuple[str, dict]]]):
    pending = deque(iter(group) for group in groups if group)
    while pending:
        iterator = pending.popleft()
        try:
            yield next(iterator)
            pending.append(iterator)
        except StopIteration:
            pass


def extension(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return None


def meta(metadata: dict, key: str) -> str:
    item = metadata.get(key, {})
    value = str(item.get("value", "")) if isinstance(item, dict) else ""
    return html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()


def is_living_plant_photo(page: dict, info: dict) -> bool:
    """Conservative filename/metadata gate; final visual review is still required."""
    title = str(page.get("title", "")).removeprefix("File:").casefold()
    image_url = str(info.get("thumburl") or info.get("url") or "").casefold()
    mime = str(info.get("mime") or "").casefold()
    excluded = (
        "archive", "document", "catalog", "catalogue", "book", "poster",
        "illustration", "drawing", "diagram", "logo", "seed packet",
        "advertisement", ".pdf", "/page",
    )
    plant_indicators = (
        "plant", "plants", "leaf", "leaves", "vine", "seedling", "potted",
        "garden", "cultivation", "solanum lycopersicum",
    )
    return (
        mime.startswith("image/")
        and not any(term in title or term in image_url for term in excluded)
        and any(term in title for term in plant_indicators)
    )


def whole_plant_queries(plant: dict[str, str], label: str) -> list[str]:
    english = plant["english_name"]
    bangla = plant["bangla_name"]
    scientific = plant["scientific_name"]
    if label == "healthy":
        return [
            f'whole healthy "{english} plant" garden photograph filetype:bitmap',
            f'healthy potted "{english} plant" photograph filetype:bitmap',
            f'healthy "{english} plant" leaves and stems photo filetype:bitmap',
            f'whole "{scientific}" plant photograph filetype:bitmap',
            f'{bangla} সুস্থ গাছের ছবি filetype:bitmap',
        ]
    return [
        f'whole diseased "{english} plant" garden photograph filetype:bitmap',
        f'unhealthy "{english} plant" leaves and stems photo filetype:bitmap',
        f'{bangla} রোগাক্রান্ত গাছের ছবি filetype:bitmap',
    ]


def download(
    csv_path: Path,
    plant_id: str,
    label: str,
    count: int,
    output: Path,
    use_whole_plant_queries: bool,
) -> int:
    plant = load_plant(csv_path, plant_id)
    query_field = f"{label}_queries"
    queries = (
        whole_plant_queries(plant, label)
        if use_whole_plant_queries
        else [query.strip() for query in plant[query_field].split("|") if query.strip()]
    )
    print(f"Loaded {len(queries)} {label} queries for {plant['english_name']} from {csv_path.name}")

    groups: list[list[tuple[str, dict]]] = []
    for query in queries:
        try:
            results = search(query)
            print(f"Search [{language(query)}] {query!r}: {len(results)} candidates")
            groups.append([(query, result) for result in results])
            time.sleep(1.0)
        except Exception as error:
            print(f"Search failed for {query!r}: {error}")

    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.csv"
    rows: list[dict[str, str | int]] = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if (output / row.get("filename", "")).is_file()]
    seen_urls = {str(row.get("image_url", "")) for row in rows}
    seen_hashes = {str(row.get("sha256", "")) for row in rows}
    sequence = max(
        (int(match.group(1)) for row in rows if (match := re.match(r"(\d+)_", str(row.get("filename", ""))))),
        default=0,
    )
    if rows:
        print(f"Resuming with {len(rows)} previously accepted plant photo(s)")

    for query, page in round_robin(groups):
        if len(rows) >= count:
            break
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        if not is_living_plant_photo(page, info):
            continue
        image_url = info.get("thumburl") or info.get("url")
        if not image_url or image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        try:
            request = urllib.request.Request(image_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read(15_000_001)
            suffix = extension(data)
            digest = hashlib.sha256(data).hexdigest()
            if len(data) > 15_000_000 or suffix is None or digest in seen_hashes:
                continue
            seen_hashes.add(digest)
        except Exception as error:
            print(f"Download failed: {image_url} ({error})")
            continue

        sequence += 1
        image_id = digest[:16]
        filename = f"{sequence:02d}_{image_id}{suffix}"
        (output / filename).write_bytes(data)
        metadata = info.get("extmetadata") or {}
        rows.append(
            {
                "image_id": image_id,
                "filename": filename,
                "title": str(page.get("title", "")).removeprefix("File:"),
                "plant_id": plant["plant_id"],
                "english_name": plant["english_name"],
                "bangla_name": plant["bangla_name"],
                "scientific_name": plant["scientific_name"],
                "health_label": label,
                "verification_status": "unverified",
                "search_query": query,
                "query_language": language(query),
                "search_provider": "wikimedia_commons",
                "image_url": image_url,
                "source_page_url": info.get("descriptionurl", ""),
                "width": info.get("thumbwidth") or info.get("width") or "",
                "height": info.get("thumbheight") or info.get("height") or "",
                "license": meta(metadata, "LicenseShortName"),
                "license_url": meta(metadata, "LicenseUrl"),
                "artist": meta(metadata, "Artist"),
                "sha256": digest,
            }
        )
        print(f"Downloaded {len(rows)}/{count}: {filename} via [{language(query)}] {query}")

    fields = [
        "image_id", "filename", "title", "plant_id", "english_name", "bangla_name",
        "scientific_name", "health_label", "verification_status", "search_query",
        "query_language", "search_provider", "image_url", "source_page_url",
        "width", "height", "license", "license_url", "artist", "sha256",
    ]
    with (output / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--plant", required=True)
    parser.add_argument("--label", choices=("healthy", "unhealthy"), required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--whole-plant",
        action="store_true",
        help="Derive stricter whole-living-plant photo queries from the selected CSV row.",
    )
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error("--count must be between 1 and 20 for this smoke test")
    saved = download(
        args.csv,
        args.plant,
        args.label,
        args.count,
        args.output,
        args.whole_plant,
    )
    print(f"Finished: saved {saved}/{args.count} images to {args.output.resolve()}")
    return 0 if saved == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
