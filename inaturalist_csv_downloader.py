"""Small, licensed iNaturalist review-pool downloader driven by the plant CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError


API_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "BangladeshGardenTinyML/0.1 (non-commercial educational pilot)"
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"}


def load_plant(path: Path, plant_id: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["plant_id"].strip().casefold() == plant_id.casefold():
                return {key: (value or "").strip() for key, value in row.items()}
    raise ValueError(f"Plant ID not found: {plant_id}")


def licensed_observations(scientific_name: str, result_limit: int = 50) -> list[dict]:
    response = requests.get(
        API_URL,
        params={
            "taxon_name": scientific_name,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": ",".join(sorted(ALLOWED_LICENSES)),
            "per_page": result_limit,
            "order_by": "votes",
            "order": "desc",
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def large_photo_url(url: str) -> str:
    for size in ("square", "small", "medium"):
        url = url.replace(f"/{size}.", "/large.")
    return url


def download_review_pool(
    csv_path: Path,
    plant_id: str,
    label: str,
    count: int,
    output: Path,
) -> int:
    plant = load_plant(csv_path, plant_id)
    observations = licensed_observations(plant["scientific_name"])
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    hashes: set[str] = set()

    for observation in observations:
        if len(rows) >= count:
            break
        for photo in observation.get("photos") or []:
            if len(rows) >= count:
                break
            license_code = str(photo.get("license_code") or "").casefold()
            if license_code not in ALLOWED_LICENSES:
                continue
            original_url = str(photo.get("url") or "")
            if not original_url:
                continue
            image_url = large_photo_url(original_url)
            try:
                response = requests.get(
                    image_url,
                    headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
                    timeout=30,
                )
                response.raise_for_status()
                content = response.content
                digest = hashlib.sha256(content).hexdigest()
                if digest in hashes:
                    continue
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                with Image.open(io.BytesIO(content)) as image:
                    width, height = image.size
                    image_format = (image.format or "").upper()
                if width < 224 or height < 224 or image_format not in {"JPEG", "PNG", "WEBP"}:
                    continue
            except (requests.RequestException, UnidentifiedImageError, OSError):
                continue

            hashes.add(digest)
            extension = ".png" if image_format == "PNG" else ".jpg"
            image_id = digest[:16]
            filename = f"{len(rows) + 1:02d}_{image_id}{extension}"
            (output / filename).write_bytes(content)
            observation_id = observation.get("id", "")
            rows.append(
                {
                    "image_id": image_id,
                    "filename": filename,
                    "plant_id": plant["plant_id"],
                    "english_name": plant["english_name"],
                    "bangla_name": plant["bangla_name"],
                    "scientific_name": plant["scientific_name"],
                    "proposed_health_label": label,
                    "verification_status": "unverified",
                    "search_provider": "inaturalist",
                    "observation_id": observation_id,
                    "image_url": image_url,
                    "source_page_url": f"https://www.inaturalist.org/observations/{observation_id}",
                    "license": license_code,
                    "attribution": str(photo.get("attribution") or ""),
                    "width": width,
                    "height": height,
                    "sha256": digest,
                }
            )
            print(f"Downloaded {len(rows)}/{count}: {filename} ({license_code})")
            time.sleep(0.2)

    fields = [
        "image_id", "filename", "plant_id", "english_name", "bangla_name",
        "scientific_name", "proposed_health_label", "verification_status",
        "search_provider", "observation_id", "image_url", "source_page_url",
        "license", "attribution", "width", "height", "sha256",
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
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error("--count must be between 1 and 20 for this pilot")
    saved = download_review_pool(args.csv, args.plant, args.label, args.count, args.output)
    print(f"Finished: {saved}/{args.count} licensed review candidates in {args.output.resolve()}")
    return 0 if saved == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
