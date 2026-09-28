"""Download a small, provenance-preserving subset of labelled drought images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import urllib.parse
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image


REPO = "MiekevanVlaardingen/Image_based_trait_extraction"
FOLDER = "Images exp 1 _ part 1"
TREE_API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{urllib.parse.quote(FOLDER, safe='')}"
RAW_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
NAME_PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<camera>[A-Z0-9]+)\.POT(?P<pot>\d+)_"
    r"(?P<genotype>[A-Z]+)_H(?P<harvest>\d+)\.Drought\.(?P<experiment>\d+)_(?P<view>\d+)\.png$"
)


def list_tree(session: requests.Session, max_pages: int) -> list[dict]:
    url = TREE_API
    params: dict[str, str | int | bool] | None = {"recursive": True, "expand": False, "limit": 1000}
    entries: list[dict] = []
    for _ in range(max_pages):
        response = session.get(url, params=params, timeout=45)
        response.raise_for_status()
        entries.extend(response.json())
        next_link = response.links.get("next", {}).get("url")
        if not next_link:
            break
        url, params = next_link, None
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("generic_drought_dataset"))
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "SylvanTinyMLDataset/1.0 (non-commercial research)"
    entries = list_tree(session, args.max_pages)

    by_plant: dict[str, list[tuple[str, str, dict[str, str]]]] = defaultdict(list)
    for entry in entries:
        if entry.get("type") != "file":
            continue
        path = str(entry.get("path", ""))
        name = Path(path).name
        match = NAME_PATTERN.match(name)
        if not match or match.group("view") != "1":
            continue
        data = match.groupdict()
        plant_key = f"{data['camera']}.POT{data['pot']}.{data['genotype']}.H{data['harvest']}.{data['experiment']}"
        by_plant[plant_key].append((data["date"], path, data))

    # Latest date per plant is most likely to display the accumulated treatment effect.
    selected: list[tuple[str, str, dict[str, str], str]] = []
    for plant_key, candidates in sorted(by_plant.items()):
        date, path, data = max(candidates, key=lambda item: item[0])
        selected.append((date, path, data, plant_key))
    selected = selected[: args.target]

    raw = args.output / "raw" / "drought_stress"
    metadata_dir = args.output / "metadata"
    raw.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for date, path, data, plant_key in selected:
        source_url = RAW_BASE + urllib.parse.quote(path, safe="/") + "?download=true"
        response = session.get(source_url, timeout=45)
        response.raise_for_status()
        payload = response.content
        sha256 = hashlib.sha256(payload).hexdigest()
        image_id = sha256[:24]
        destination = raw / f"{image_id}.png"
        destination.write_bytes(payload)
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            width, height = image.size
        rows.append(
            {
                "image_id": image_id,
                "issue_label": "drought_stress",
                "binary_label": "unhealthy",
                "verification_status": "dataset_label_pending_visual_check",
                "species": "Chenopodium quinoa",
                "treatment": "Drought",
                "capture_date": date,
                "plant_key": plant_key,
                "source_dataset": "Image_based_trait_extraction",
                "source_dataset_url": f"https://huggingface.co/datasets/{REPO}",
                "source_file_url": source_url,
                "source_path": path,
                "license": "CC BY 4.0",
                "attribution": "Mieke van Vlaardingen, Jacky To, Eibertus N. van Loo, and NPEC collaborators",
                "local_path": destination.relative_to(args.output).as_posix(),
                "width": width,
                "height": height,
                "sha256": sha256,
            }
        )

    fields = list(rows[0]) if rows else ["image_id"]
    with (metadata_dir / "images.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Listed {len(entries)} files, found {len(by_plant)} drought-treated plants, downloaded {len(rows)} latest views.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
