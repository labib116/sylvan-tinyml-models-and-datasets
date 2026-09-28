from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .deduplicator import DuplicateIndex
from .models import DatasetPaths, PlantRecord, SearchCandidate, ValidationResult


IMAGE_FIELDS = [
    "image_id", "plant_id", "english_name", "bangla_name", "scientific_name",
    "category", "health_label", "verification_status", "rejection_reason",
    "review_flags", "search_query", "query_language", "search_provider",
    "provider_result_id", "source_collection", "image_url", "source_page_url",
    "source_domain", "license", "license_url", "attribution", "creator",
    "original_filename", "local_path", "width", "height", "file_size_bytes",
    "mime_type", "original_content_type", "image_format", "download_timestamp",
    "sha256", "phash", "exact_duplicate_of", "near_duplicate_of",
    "phash_distance",
]

FAILURE_FIELDS = [
    "image_id", "plant_id", "health_label", "search_query", "query_language",
    "search_provider", "image_url", "source_page_url", "source_domain",
    "http_status", "error", "timestamp",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MetadataStore:
    def __init__(self, paths: DatasetPaths, skip_failed_urls: bool = True) -> None:
        self.paths = paths
        self.images_path = paths.metadata / "images.csv"
        self.failures_path = paths.metadata / "failed_downloads.csv"
        self.summary_path = paths.metadata / "collection_summary.csv"
        self.processed_urls: set[str] = set()
        self.accepted: Counter[tuple[str, str]] = Counter()
        self.domain_counts: Counter[tuple[str, str, str]] = Counter()
        self.query_counts: Counter[tuple[str, str, str]] = Counter()
        self.stats: Counter[str] = Counter()
        self.duplicates = DuplicateIndex()
        self._load_images()
        self._load_failures(skip_failed_urls)

    def _load_images(self) -> None:
        if not self.images_path.exists():
            return
        with self.images_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                url = row.get("image_url", "")
                if url:
                    self.processed_urls.add(url)
                status = row.get("verification_status", "")
                reason = row.get("rejection_reason", "")
                if status == "unverified" and not reason and row.get("local_path"):
                    key = (row.get("plant_id", ""), row.get("health_label", ""))
                    self.accepted[key] += 1
                    self.domain_counts[(*key, row.get("source_domain", ""))] += 1
                    self.query_counts[(*key, row.get("search_query", ""))] += 1
                    self.stats["accepted"] += 1
                elif reason:
                    self.stats[_reason_category(reason)] += 1
                self.duplicates.add(row.get("image_id", ""), row.get("sha256", ""), row.get("phash", ""))

    def _load_failures(self, skip_failed_urls: bool) -> None:
        if not self.failures_path.exists():
            return
        with self.failures_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.stats["failed_downloads"] += len(rows)
        if skip_failed_urls:
            self.processed_urls.update(row.get("image_url", "") for row in rows if row.get("image_url"))

    def accepted_count(self, plant_id: str, health_label: str) -> int:
        return self.accepted[(plant_id, health_label)]

    def record_image(self, row: dict[str, Any]) -> None:
        self._append(self.images_path, IMAGE_FIELDS, row)
        url = str(row.get("image_url") or "")
        if url:
            self.processed_urls.add(url)
        status = str(row.get("verification_status") or "")
        reason = str(row.get("rejection_reason") or "")
        if status == "unverified" and not reason and row.get("local_path"):
            key = (str(row["plant_id"]), str(row["health_label"]))
            self.accepted[key] += 1
            self.domain_counts[(*key, str(row.get("source_domain") or ""))] += 1
            self.query_counts[(*key, str(row.get("search_query") or ""))] += 1
            self.stats["accepted"] += 1
        elif reason:
            self.stats[_reason_category(reason)] += 1
        self.duplicates.add(str(row.get("image_id") or ""), str(row.get("sha256") or ""), str(row.get("phash") or ""))

    def record_failure(
        self,
        image_id: str,
        plant: PlantRecord,
        label: str,
        candidate: SearchCandidate,
        status_code: int | None,
        error: str,
    ) -> None:
        row = {
            "image_id": image_id,
            "plant_id": plant.plant_id,
            "health_label": label,
            "search_query": candidate.query,
            "query_language": candidate.query_language,
            "search_provider": candidate.search_provider,
            "image_url": candidate.image_url,
            "source_page_url": candidate.source_page_url,
            "source_domain": candidate.source_domain,
            "http_status": status_code or "",
            "error": error,
            "timestamp": utc_now(),
        }
        self._append(self.failures_path, FAILURE_FIELDS, row)
        self.processed_urls.add(candidate.image_url)
        self.stats["failed_downloads"] += 1

    @staticmethod
    def _append(path: Path, fields: list[str], row: dict[str, Any]) -> None:
        new_file = not path.exists() or path.stat().st_size == 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fields})
            handle.flush()
            os.fsync(handle.fileno())

    def write_summary(self, plants: list[PlantRecord], fallback_target: int) -> None:
        fields = ["plant_id", "english_name", "healthy", "unhealthy", "total", "target_total", "remaining"]
        temporary = self.summary_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for plant in plants:
                healthy = self.accepted_count(plant.plant_id, "healthy")
                unhealthy = self.accepted_count(plant.plant_id, "unhealthy")
                target = plant.target_for("healthy", fallback_target) + plant.target_for("unhealthy", fallback_target)
                writer.writerow(
                    {
                        "plant_id": plant.plant_id,
                        "english_name": plant.english_name,
                        "healthy": healthy,
                        "unhealthy": unhealthy,
                        "total": healthy + unhealthy,
                        "target_total": target,
                        "remaining": max(0, target - healthy - unhealthy),
                    }
                )
        os.replace(temporary, self.summary_path)


def build_image_row(
    *,
    image_id: str,
    plant: PlantRecord,
    label: str,
    candidate: SearchCandidate,
    validation: ValidationResult,
    original_filename: str,
    original_content_type: str,
    file_size: int,
    verification_status: str,
    rejection_reason: str = "",
    local_path: str = "",
    exact_duplicate_of: str = "",
    near_duplicate_of: str = "",
    phash_distance: int | str = "",
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "plant_id": plant.plant_id,
        "english_name": plant.english_name,
        "bangla_name": plant.bangla_name,
        "scientific_name": plant.scientific_name,
        "category": plant.category,
        "health_label": label,
        "verification_status": verification_status,
        "rejection_reason": rejection_reason,
        "review_flags": "|".join(validation.review_flags),
        "search_query": candidate.query,
        "query_language": candidate.query_language,
        "search_provider": candidate.search_provider,
        "provider_result_id": candidate.provider_result_id,
        "source_collection": candidate.source_collection,
        "image_url": candidate.image_url,
        "source_page_url": candidate.source_page_url,
        "source_domain": candidate.source_domain,
        "license": candidate.license,
        "license_url": candidate.license_url,
        "attribution": candidate.attribution,
        "creator": candidate.creator,
        "original_filename": original_filename,
        "local_path": local_path,
        "width": validation.width or "",
        "height": validation.height or "",
        "file_size_bytes": file_size,
        "mime_type": validation.mime_type,
        "original_content_type": original_content_type,
        "image_format": validation.image_format,
        "download_timestamp": utc_now(),
        "sha256": validation.sha256,
        "phash": validation.phash,
        "exact_duplicate_of": exact_duplicate_of,
        "near_duplicate_of": near_duplicate_of,
        "phash_distance": phash_distance,
    }


def _reason_category(reason: str) -> str:
    if reason.startswith("low_resolution"):
        return "low_resolution"
    if reason.startswith("exact_duplicate"):
        return "exact_duplicates"
    if reason.startswith("near_duplicate"):
        return "near_duplicates"
    if reason.startswith("review_flag"):
        return "needs_review"
    return "other_rejections"
