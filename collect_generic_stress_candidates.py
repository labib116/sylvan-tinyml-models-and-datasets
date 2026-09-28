"""Collect a bounded Wikimedia Commons review pool for visible water stress."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import re
import time
from pathlib import Path

import imagehash
import requests
from PIL import Image


API = "https://commons.wikimedia.org/w/api.php"
QUERIES = [
    "intitle:wilted plant",
    "intitle:wilting plant",
    "intitle:withered plant",
    "intitle:dry plant",
    "intitle:drought stressed plant",
    "intitle:dried up potted plant",
]
ALLOWED_LICENSE_MARKERS = ("cc0", "cc by", "public domain", "pdm")


def clean_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value or "")).strip()


def existing_hashes(dataset: Path) -> tuple[set[str], list[imagehash.ImageHash]]:
    manifest = dataset / "metadata" / "images.csv"
    if not manifest.exists():
        return set(), []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        {row["sha256"] for row in rows if row.get("sha256")},
        [imagehash.hex_to_hash(row["phash"]) for row in rows if row.get("phash")],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("whole_csv_candidates"))
    parser.add_argument("--output", type=Path, default=Path("generic_stress_candidates"))
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--per-query", type=int, default=20)
    args = parser.parse_args()

    raw = args.output / "raw" / "drought_stress"
    metadata_dir = args.output / "metadata"
    raw.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    known_sha, known_phash = existing_hashes(args.dataset)
    accepted: list[dict[str, str | int]] = []
    session = requests.Session()
    session.headers["User-Agent"] = "SylvanTinyMLDataset/1.0 (non-commercial research)"

    for query in QUERIES:
        if len(accepted) >= args.target:
            break
        response = session.get(
            API,
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": args.per_query,
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": 1024,
            },
            timeout=30,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            if len(accepted) >= args.target:
                break
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            if info.get("mime") not in {"image/jpeg", "image/png"}:
                continue
            if int(info.get("width", 0)) < 320 or int(info.get("height", 0)) < 240:
                continue
            ext = info.get("extmetadata", {})
            license_name = clean_html(ext.get("LicenseShortName", {}).get("value", ""))
            if not any(marker in license_name.lower() for marker in ALLOWED_LICENSE_MARKERS):
                continue
            image_url = info.get("thumburl") or info.get("url")
            if not image_url:
                continue
            try:
                image_response = session.get(image_url, timeout=30)
                image_response.raise_for_status()
                payload = image_response.content
                sha256 = hashlib.sha256(payload).hexdigest()
                if sha256 in known_sha:
                    continue
                with Image.open(io.BytesIO(payload)) as opened:
                    opened.verify()
                with Image.open(io.BytesIO(payload)) as opened:
                    rgb = opened.convert("RGB")
                    phash = imagehash.phash(rgb)
                    width, height = rgb.size
                    output = io.BytesIO()
                    rgb.save(output, "JPEG", quality=92)
                    jpeg_payload = output.getvalue()
                if any(phash - prior <= 5 for prior in known_phash):
                    continue
            except (requests.RequestException, OSError):
                continue

            image_id = hashlib.sha256((str(page.get("pageid")) + image_url).encode()).hexdigest()[:24]
            destination = raw / f"{image_id}.jpg"
            destination.write_bytes(jpeg_payload)
            known_sha.add(sha256)
            known_phash.append(phash)
            accepted.append(
                {
                    "image_id": image_id,
                    "issue_label": "drought_stress",
                    "binary_label": "unhealthy",
                    "verification_status": "candidate_unverified",
                    "search_query": query,
                    "search_provider": "wikimedia_commons",
                    "provider_result_id": str(page.get("pageid", "")),
                    "title": page.get("title", ""),
                    "description": clean_html(ext.get("ImageDescription", {}).get("value", "")),
                    "image_url": image_url,
                    "source_page_url": info.get("descriptionurl", ""),
                    "license": license_name,
                    "license_url": clean_html(ext.get("LicenseUrl", {}).get("value", "")),
                    "attribution": clean_html(ext.get("Artist", {}).get("value", "")),
                    "local_path": destination.relative_to(args.output).as_posix(),
                    "width": width,
                    "height": height,
                    "sha256": sha256,
                    "phash": str(phash),
                }
            )
            time.sleep(0.25)
        time.sleep(0.75)

    fields = list(accepted[0]) if accepted else ["image_id"]
    with (metadata_dir / "images.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)
    print(f"Downloaded {len(accepted)} licensed water-stress candidates (target {args.target}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
