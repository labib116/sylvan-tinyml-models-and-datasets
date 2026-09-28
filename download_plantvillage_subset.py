"""Download a deterministic, class-diverse subset of PlantVillage RGB images."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import imagehash
import requests
from PIL import Image


REPO = "spMohanty/PlantVillage-Dataset"
BRANCH = "master"
OUTPUT = Path("plantvillage_subset")
HEALTHY_TARGET = 40
UNHEALTHY_TARGET = 111
SEED = 20260927


def get_with_retry(session: requests.Session, url: str, **kwargs: object) -> requests.Response:
    for attempt in range(5):
        response = session.get(url, **kwargs)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt == 4:
            response.raise_for_status()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def allocate(total: int, classes: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(classes))
    return {name: base + (index < remainder) for index, name in enumerate(classes)}


def leaf_group(filename: str, class_name: str, leaf_map: dict[str, str]) -> str:
    stem = PurePosixPath(filename).stem
    suffix = stem.split("___", 1)[-1].lower()
    mapped = leaf_map.get(suffix)
    return f"pv:{mapped}" if mapped else f"pv:{class_name}:{stem.split('___', 1)[0]}"


def main() -> int:
    raw_root = OUTPUT / "raw"
    metadata_dir = OUTPUT / "metadata"
    raw_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "SylvanTinyMLDataset/1.0 (non-commercial research)"

    raw_listing = get_with_retry(
        session,
        f"https://api.github.com/repos/{REPO}/contents/raw",
        timeout=45,
    ).json()
    color_entry = next(item for item in raw_listing if item.get("name") == "color")
    color_tree = get_with_retry(session, color_entry["git_url"], timeout=45).json()

    leaf_map = get_with_retry(
        session,
        f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/leaf-map.json",
        timeout=90,
    ).json()

    by_class: dict[str, list[str]] = defaultdict(list)
    class_entries = [entry for entry in color_tree.get("tree", []) if entry.get("type") == "tree"]
    for class_index, class_entry in enumerate(class_entries, start=1):
        class_name = class_entry["path"]
        class_tree = get_with_retry(session, class_entry["url"], timeout=45).json()
        for entry in class_tree.get("tree", []):
            filename = entry.get("path", "")
            if entry.get("type") != "blob":
                continue
            if PurePosixPath(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            by_class[class_name].append(f"{class_name}/{filename}")
        if class_index % 10 == 0 or class_index == len(class_entries):
            print(f"Indexed {class_index}/{len(class_entries)} classes")

    healthy_classes = sorted(name for name in by_class if name.lower().endswith("___healthy"))
    unhealthy_classes = sorted(name for name in by_class if not name.lower().endswith("___healthy"))
    quotas = {
        "healthy": allocate(HEALTHY_TARGET, healthy_classes),
        "unhealthy": allocate(UNHEALTHY_TARGET, unhealthy_classes),
    }

    rng = random.Random(SEED)
    selected: list[tuple[str, str, str]] = []
    for label, classes in (("healthy", healthy_classes), ("unhealthy", unhealthy_classes)):
        for class_name in classes:
            candidates = list(by_class[class_name])
            rng.shuffle(candidates)
            seen_groups: set[str] = set()
            taken = 0
            for path in candidates:
                filename = PurePosixPath(path).name
                group = leaf_group(filename, class_name, leaf_map)
                if group in seen_groups:
                    continue
                seen_groups.add(group)
                selected.append((label, class_name, path))
                taken += 1
                if taken >= quotas[label][class_name]:
                    break
            if taken != quotas[label][class_name]:
                raise RuntimeError(f"Could not sample quota for {class_name}: {taken}")

    rows: list[dict[str, str]] = []
    for index, (label, class_name, repo_path) in enumerate(selected, start=1):
        encoded_path = quote(f"raw/color/{repo_path}", safe="/")
        image_url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{encoded_path}"
        payload = get_with_retry(session, image_url, timeout=60).content
        with Image.open(io.BytesIO(payload)) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            converted = io.BytesIO()
            rgb.save(converted, "JPEG", quality=95)
            jpeg = converted.getvalue()

        filename = PurePosixPath(repo_path).name
        group = leaf_group(filename, class_name, leaf_map)
        image_id = "pv_" + hashlib.sha256(repo_path.encode("utf-8")).hexdigest()[:20]
        destination = raw_root / label / f"{image_id}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(jpeg)
        rows.append(
            {
                "image_id": image_id,
                "binary_label": label,
                "source": "plantvillage",
                "source_class": class_name,
                "source_group": group,
                "source_repo_path": f"raw/color/{repo_path}",
                "source_page_url": f"https://github.com/{REPO}/blob/{BRANCH}/raw/color/{quote(repo_path, safe='/')}",
                "image_url": image_url,
                "license": "PlantVillage open-access dataset; cite source repository and paper",
                "local_path": destination.relative_to(OUTPUT).as_posix(),
                "width": str(width),
                "height": str(height),
                "sha256": hashlib.sha256(jpeg).hexdigest(),
                "phash": str(imagehash.phash(rgb)),
            }
        )
        if index % 20 == 0 or index == len(selected):
            print(f"Downloaded {index}/{len(selected)}")
        time.sleep(0.08)

    fields = list(rows[0])
    with (metadata_dir / "images.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (metadata_dir / "sampling.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "healthy_target": HEALTHY_TARGET,
                "unhealthy_target": UNHEALTHY_TARGET,
                "healthy_classes": healthy_classes,
                "unhealthy_classes": unhealthy_classes,
                "quotas": quotas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Complete: healthy={HEALTHY_TARGET}, unhealthy={UNHEALTHY_TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
