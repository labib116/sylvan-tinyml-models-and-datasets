from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PlantRecord:
    plant_id: str
    english_name: str
    bangla_name: str
    scientific_name: str
    category: str
    healthy_label: str
    unhealthy_label: str
    healthy_queries: tuple[str, ...]
    unhealthy_queries: tuple[str, ...]
    target_healthy_images: int
    target_unhealthy_images: int
    extra: dict[str, str] = field(default_factory=dict)

    def queries_for(self, label: str) -> tuple[str, ...]:
        if label == "healthy":
            return self.healthy_queries
        if label == "unhealthy":
            return self.unhealthy_queries
        raise ValueError(f"Unsupported health label: {label}")

    def target_for(self, label: str, fallback: int) -> int:
        value = (
            self.target_healthy_images if label == "healthy" else self.target_unhealthy_images
        )
        return value or fallback


@dataclass(frozen=True)
class SearchCandidate:
    provider_result_id: str
    search_provider: str
    query: str
    query_language: str
    image_url: str
    source_page_url: str
    source_domain: str
    title: str = ""
    license: str = ""
    license_url: str = ""
    attribution: str = ""
    creator: str = ""
    provider_width: int | None = None
    provider_height: int | None = None
    source_collection: str = ""


@dataclass(frozen=True)
class DownloadResult:
    candidate: SearchCandidate
    content: bytes | None
    content_type: str
    original_filename: str
    status_code: int | None
    error: str = ""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    stored_bytes: bytes | None = None
    extension: str = ""
    image_format: str = ""
    mime_type: str = ""
    width: int = 0
    height: int = 0
    sha256: str = ""
    phash: str = ""
    rejection_reason: str = ""
    review_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    raw: Path
    metadata: Path
    logs: Path
    rejected: Path

