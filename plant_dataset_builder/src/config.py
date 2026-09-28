from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    config_path: Path
    dataset_root: Path
    target_per_class: int
    oversampling_factor: float
    min_width: int
    min_height: int
    max_download_bytes: int
    max_images_per_query: int
    max_images_per_domain: int
    phash_threshold: int
    request_timeout: float
    search_retries: int
    download_retries: int
    backoff_factor: float
    request_delay_seconds: float
    download_workers: int
    providers: tuple[str, ...]
    skip_failed_urls_on_resume: bool
    review_keywords: tuple[str, ...]


DEFAULTS: dict[str, Any] = {
    "dataset_root": "dataset",
    "target_per_class": 200,
    "oversampling_factor": 1.75,
    "min_width": 224,
    "min_height": 224,
    "max_download_bytes": 25_000_000,
    "max_images_per_query": 80,
    "max_images_per_domain": 30,
    "phash_threshold": 5,
    "request_timeout": 15,
    "search_retries": 3,
    "download_retries": 3,
    "backoff_factor": 1.0,
    "request_delay_seconds": 0.25,
    "download_workers": 8,
    "providers": ["openverse", "wikimedia"],
    "skip_failed_urls_on_resume": True,
    "review_keywords": [],
}


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")
    values = {**DEFAULTS, **supplied}
    root = Path(str(values["dataset_root"]))
    if not root.is_absolute():
        root = config_path.parent / root

    positive = (
        "target_per_class", "min_width", "min_height", "max_download_bytes",
        "max_images_per_query", "max_images_per_domain", "request_timeout",
        "download_retries", "download_workers",
    )
    for key in positive:
        if float(values[key]) <= 0:
            raise ValueError(f"{key} must be greater than zero")
    if float(values["oversampling_factor"]) < 1:
        raise ValueError("oversampling_factor must be at least 1.0")
    if not 0 <= int(values["phash_threshold"]) <= 64:
        raise ValueError("phash_threshold must be between 0 and 64")
    providers = tuple(str(item).strip().lower() for item in values["providers"] if str(item).strip())
    if not providers:
        raise ValueError("At least one search provider is required")

    return Config(
        config_path=config_path,
        dataset_root=root.resolve(),
        target_per_class=int(values["target_per_class"]),
        oversampling_factor=float(values["oversampling_factor"]),
        min_width=int(values["min_width"]),
        min_height=int(values["min_height"]),
        max_download_bytes=int(values["max_download_bytes"]),
        max_images_per_query=int(values["max_images_per_query"]),
        max_images_per_domain=int(values["max_images_per_domain"]),
        phash_threshold=int(values["phash_threshold"]),
        request_timeout=float(values["request_timeout"]),
        search_retries=int(values["search_retries"]),
        download_retries=int(values["download_retries"]),
        backoff_factor=float(values["backoff_factor"]),
        request_delay_seconds=float(values["request_delay_seconds"]),
        download_workers=int(values["download_workers"]),
        providers=providers,
        skip_failed_urls_on_resume=bool(values["skip_failed_urls_on_resume"]),
        review_keywords=tuple(str(item).casefold() for item in values["review_keywords"]),
    )
