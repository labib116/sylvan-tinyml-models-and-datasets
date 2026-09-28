"""Collect small, auditable healthy/unhealthy candidate pools for every CSV plant.

Labels are proposals only. This script validates files and removes duplicates, but it
does not claim visual or health verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import imagehash
import requests
from PIL import Image, ImageOps, UnidentifiedImageError


INAT_OBSERVATIONS = "https://api.inaturalist.org/v1/observations"
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BangladeshGardenTinyML/0.2 (non-commercial educational dataset pilot)"
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"}
METADATA_FIELDS = [
    "image_id", "plant_id", "english_name", "bangla_name", "scientific_name",
    "category", "proposed_health_label", "verification_status", "common_disease",
    "pathogen_or_cause", "disease_evidence_scope", "disease_source_url",
    "search_query", "search_provider", "provider_result_id", "image_url",
    "source_page_url", "source_domain", "license", "license_url", "attribution",
    "local_path", "width", "height", "mime_type", "file_size_bytes", "sha256",
    "phash", "download_timestamp",
]
FAILURE_FIELDS = ["plant_id", "proposed_health_label", "image_url", "reason", "timestamp"]


class RateLimiter:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.next_time = 0.0

    def wait(self) -> None:
        delay = max(0.0, self.next_time - time.monotonic())
        if delay:
            time.sleep(delay)
        self.next_time = time.monotonic() + self.interval


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def canonical_taxon(name: str) -> str:
    return name.replace(" spp.", "").replace(" sp.", "").strip()


def large_inat_url(url: str) -> str:
    for size in ("square", "small", "medium"):
        url = url.replace(f"/{size}.", "/large.")
    return url


def domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def meta_value(metadata: dict, key: str) -> str:
    value = metadata.get(key, {})
    return str(value.get("value") or "") if isinstance(value, dict) else ""


def round_robin(groups: list[list[dict]]) -> Iterator[dict]:
    queue = deque(iter(group) for group in groups if group)
    while queue:
        iterator = queue.popleft()
        try:
            yield next(iterator)
            queue.append(iterator)
        except StopIteration:
            pass


class Collector:
    def __init__(self, root: Path, target: int, min_size: int, phash_threshold: int) -> None:
        self.root = root.resolve()
        self.raw = self.root / "raw"
        self.metadata_dir = self.root / "metadata"
        self.metadata_path = self.metadata_dir / "images.csv"
        self.failures_path = self.metadata_dir / "failed_downloads.csv"
        self.summary_path = self.metadata_dir / "collection_summary.csv"
        self.target = target
        self.min_size = min_size
        self.phash_threshold = phash_threshold
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.api_rate = RateLimiter(1.1)
        self.media_rate = RateLimiter(0.2)
        self.processed_urls: set[str] = set()
        self.exact_hashes: dict[str, str] = {}
        self.phashes: list[tuple[int, str]] = []
        self.counts: Counter[tuple[str, str]] = Counter()
        self.raw.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        if not self.metadata_path.exists():
            return
        with self.metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                image_url = row.get("image_url", "")
                if image_url:
                    self.processed_urls.add(image_url)
                local = self.root / row.get("local_path", "")
                if row.get("verification_status") == "candidate_unverified" and local.is_file():
                    self.counts[(row.get("plant_id", ""), row.get("proposed_health_label", ""))] += 1
                digest = row.get("sha256", "")
                if digest:
                    self.exact_hashes[digest] = row.get("image_id", "")
                phash = row.get("phash", "")
                if phash:
                    self.phashes.append((int(phash, 16), row.get("image_id", "")))

    def inat_candidates(self, taxon_name: str, query: str, provider_name: str) -> list[dict]:
        self.api_rate.wait()
        response = self.session.get(
            INAT_OBSERVATIONS,
            params={
                "taxon_name": canonical_taxon(taxon_name),
                "quality_grade": "research",
                "photos": "true",
                "photo_license": ",".join(sorted(ALLOWED_LICENSES)),
                "per_page": 100,
                "order_by": "created_at",
                "order": "desc",
            },
            timeout=30,
        )
        response.raise_for_status()
        candidates: list[dict] = []
        for observation in response.json().get("results", []):
            photos = observation.get("photos") or []
            if not photos:
                continue
            photo = photos[0]
            license_code = str(photo.get("license_code") or "").casefold()
            if license_code not in ALLOWED_LICENSES:
                continue
            image_url = large_inat_url(str(photo.get("url") or ""))
            if not image_url:
                continue
            observation_id = str(observation.get("id") or "")
            candidates.append(
                {
                    "query": query,
                    "provider": provider_name,
                    "result_id": f"inat:{observation_id}:{photo.get('id', '')}",
                    "image_url": image_url,
                    "source_page_url": f"https://www.inaturalist.org/observations/{observation_id}",
                    "license": license_code,
                    "license_url": f"https://creativecommons.org/licenses/{license_code.removeprefix('cc-')}/4.0/" if license_code != "cc0" else "https://creativecommons.org/publicdomain/zero/1.0/",
                    "attribution": str(photo.get("attribution") or ""),
                }
            )
        return candidates

    def wikimedia_candidates(self, query: str) -> list[dict]:
        self.api_rate.wait()
        response = self.session.get(
            WIKIMEDIA_API,
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": f"{query} filetype:bitmap",
                "gsrnamespace": 6,
                "gsrlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": 1600,
                "iiextmetadatalanguage": "en",
            },
            timeout=30,
        )
        response.raise_for_status()
        candidates: list[dict] = []
        for page in (response.json().get("query") or {}).get("pages") or []:
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            if not str(info.get("mime") or "").startswith("image/"):
                continue
            image_url = str(info.get("thumburl") or info.get("url") or "")
            source_page = str(info.get("descriptionurl") or "")
            metadata = info.get("extmetadata") or {}
            if image_url and source_page:
                candidates.append(
                    {
                        "query": query,
                        "provider": "wikimedia",
                        "result_id": f"wikimedia:{page.get('pageid', '')}",
                        "image_url": image_url,
                        "source_page_url": source_page,
                        "license": meta_value(metadata, "LicenseShortName"),
                        "license_url": meta_value(metadata, "LicenseUrl"),
                        "attribution": meta_value(metadata, "Artist"),
                    }
                )
        return candidates

    def collect_group(self, plant: dict[str, str], disease: dict[str, str], label: str) -> None:
        key = (plant["plant_id"], label)
        if self.counts[key] >= self.target:
            print(f"SKIP {plant['plant_id']}/{label}: {self.counts[key]}/{self.target}")
            return
        print(f"COLLECT {plant['plant_id']}/{label}: {self.counts[key]}/{self.target}")
        groups: list[list[dict]] = []
        try:
            if label == "healthy":
                groups.append(
                    self.inat_candidates(
                        plant["scientific_name"],
                        f"licensed observations of {plant['scientific_name']}",
                        "inaturalist_host",
                    )
                )
            else:
                groups.append(self.wikimedia_candidates(disease["disease_query"]))
                pathogen = disease["pathogen_or_cause"]
                if pathogen and "virus" not in pathogen.casefold() and "xanthomonas" not in pathogen.casefold() and "ralstonia" not in pathogen.casefold():
                    groups.append(
                        self.inat_candidates(
                            pathogen,
                            f"licensed observations of pathogen {pathogen}",
                            "inaturalist_pathogen",
                        )
                    )
        except requests.RequestException as exc:
            print(f"  search failed: {exc}")

        # Fallback to Commons for host photos if iNaturalist has insufficient data.
        if label == "healthy" and sum(len(group) for group in groups) < self.target:
            try:
                groups.append(
                    self.wikimedia_candidates(
                        f"whole {plant['scientific_name']} plant garden photograph"
                    )
                )
            except requests.RequestException as exc:
                print(f"  fallback search failed: {exc}")

        attempts = 0
        for candidate in round_robin(groups):
            if self.counts[key] >= self.target or attempts >= 40:
                break
            if candidate["image_url"] in self.processed_urls:
                continue
            attempts += 1
            self.processed_urls.add(candidate["image_url"])
            self._download_candidate(plant, disease, label, candidate)
        if self.counts[key] < self.target:
            print(f"  INCOMPLETE {plant['plant_id']}/{label}: {self.counts[key]}/{self.target}")

    def _download_candidate(
        self, plant: dict[str, str], disease: dict[str, str], label: str, candidate: dict
    ) -> None:
        try:
            self.media_rate.wait()
            response = self.session.get(candidate["image_url"], timeout=30)
            response.raise_for_status()
            content = response.content
            if not content or len(content) > 20_000_000:
                raise ValueError("empty_or_oversized")
            sha256 = hashlib.sha256(content).hexdigest()
            if sha256 in self.exact_hashes:
                raise ValueError("exact_duplicate")
            with Image.open(io.BytesIO(content)) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened)
                width, height = image.size
                image_format = (opened.format or "").upper()
                if width < self.min_size or height < self.min_size:
                    raise ValueError(f"low_resolution:{width}x{height}")
                if image_format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError(f"unsupported_format:{image_format}")
                phash = str(imagehash.phash(image.convert("RGB")))
            phash_int = int(phash, 16)
            for known_hash, known_id in self.phashes:
                if (phash_int ^ known_hash).bit_count() <= self.phash_threshold:
                    raise ValueError(f"near_duplicate:{known_id}")
            image_id = hashlib.sha256(candidate["image_url"].encode("utf-8")).hexdigest()[:24]
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image_format]
            destination = self.raw / plant["plant_id"] / label / f"{image_id}{extension}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
            row = {
                "image_id": image_id,
                "plant_id": plant["plant_id"],
                "english_name": plant["english_name"],
                "bangla_name": plant["bangla_name"],
                "scientific_name": plant["scientific_name"],
                "category": plant["category"],
                "proposed_health_label": label,
                "verification_status": "candidate_unverified",
                "common_disease": disease["common_disease"] if label == "unhealthy" else "",
                "pathogen_or_cause": disease["pathogen_or_cause"] if label == "unhealthy" else "",
                "disease_evidence_scope": disease["evidence_scope"] if label == "unhealthy" else "",
                "disease_source_url": disease["source_url"] if label == "unhealthy" else "",
                "search_query": candidate["query"],
                "search_provider": candidate["provider"],
                "provider_result_id": candidate["result_id"],
                "image_url": candidate["image_url"],
                "source_page_url": candidate["source_page_url"],
                "source_domain": domain(candidate["source_page_url"]),
                "license": candidate["license"],
                "license_url": candidate["license_url"],
                "attribution": candidate["attribution"],
                "local_path": destination.relative_to(self.root).as_posix(),
                "width": width,
                "height": height,
                "mime_type": response.headers.get("Content-Type", "").split(";", 1)[0],
                "file_size_bytes": len(content),
                "sha256": sha256,
                "phash": phash,
                "download_timestamp": utc_now(),
            }
            append_csv(self.metadata_path, METADATA_FIELDS, row)
            self.exact_hashes[sha256] = image_id
            self.phashes.append((phash_int, image_id))
            self.counts[(plant["plant_id"], label)] += 1
            print(f"  saved {self.counts[(plant['plant_id'], label)]}/{self.target} from {candidate['provider']}")
        except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as exc:
            append_csv(
                self.failures_path,
                FAILURE_FIELDS,
                {
                    "plant_id": plant["plant_id"],
                    "proposed_health_label": label,
                    "image_url": candidate["image_url"],
                    "reason": str(exc),
                    "timestamp": utc_now(),
                },
            )

    def write_summary(self, plants: list[dict[str, str]]) -> None:
        fields = ["plant_id", "healthy_candidates", "unhealthy_candidates", "total", "remaining"]
        temporary = self.summary_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for plant in plants:
                healthy = self.counts[(plant["plant_id"], "healthy")]
                unhealthy = self.counts[(plant["plant_id"], "unhealthy")]
                writer.writerow(
                    {
                        "plant_id": plant["plant_id"],
                        "healthy_candidates": healthy,
                        "unhealthy_candidates": unhealthy,
                        "total": healthy + unhealthy,
                        "remaining": max(0, self.target * 2 - healthy - unhealthy),
                    }
                )
        os.replace(temporary, self.summary_path)


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--diseases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=5)
    parser.add_argument("--plant", action="append")
    parser.add_argument("--max-plants", type=int)
    parser.add_argument("--label", choices=("healthy", "unhealthy"))
    args = parser.parse_args()
    plants = load_rows(args.csv)
    diseases = {row["plant_id"]: row for row in load_rows(args.diseases)}
    if args.plant:
        selected = {item for value in args.plant for item in value.split(",")}
        plants = [plant for plant in plants if plant["plant_id"] in selected]
    if args.max_plants:
        plants = plants[: args.max_plants]
    missing = [plant["plant_id"] for plant in plants if plant["plant_id"] not in diseases]
    if missing:
        raise ValueError(f"Missing disease rows: {', '.join(missing)}")
    collector = Collector(args.output, args.target, min_size=224, phash_threshold=5)
    labels = [args.label] if args.label else ["healthy", "unhealthy"]
    try:
        for plant in plants:
            for label in labels:
                collector.collect_group(plant, diseases[plant["plant_id"]], label)
    except KeyboardInterrupt:
        print("Interrupted; completed rows are resumable.")
        return_code = 130
    else:
        return_code = 0
    finally:
        collector.write_summary(load_rows(args.csv))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
