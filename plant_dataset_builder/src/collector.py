from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import Config
from .downloader import ImageDownloader
from .metadata import MetadataStore, build_image_row
from .models import DownloadResult, PlantRecord, SearchCandidate, ValidationResult
from .search import SearchProvider
from .utils import round_robin, stable_image_id
from .validator import ImageValidator

LOGGER = logging.getLogger(__name__)


class Collector:
    def __init__(
        self,
        config: Config,
        store: MetadataStore,
        providers: list[SearchProvider],
        downloader: ImageDownloader,
        validator: ImageValidator,
    ) -> None:
        self.config = config
        self.store = store
        self.providers = providers
        self.downloader = downloader
        self.validator = validator

    def collect(self, plant: PlantRecord, label: str, target: int) -> None:
        existing = self.store.accepted_count(plant.plant_id, label)
        if existing >= target:
            LOGGER.info("SKIP %s/%s: target already met (%s/%s)", plant.plant_id, label, existing, target)
            return
        remaining = target - existing
        candidate_budget = max(remaining, math.ceil(remaining * self.config.oversampling_factor))
        LOGGER.info(
            "COLLECT %s/%s: %s accepted, need %s, candidate budget %s",
            plant.plant_id,
            label,
            existing,
            remaining,
            candidate_budget,
        )
        candidates = self._candidates(plant.queries_for(label))
        attempted = 0
        with ThreadPoolExecutor(max_workers=self.config.download_workers) as executor:
            while attempted < candidate_budget and self.store.accepted_count(plant.plant_id, label) < target:
                batch: list[tuple[SearchCandidate, Future[DownloadResult]]] = []
                while len(batch) < self.config.download_workers and attempted < candidate_budget:
                    try:
                        candidate = next(candidates)
                    except StopIteration:
                        break
                    if candidate.image_url in self.store.processed_urls:
                        continue
                    attempted += 1
                    if self._domain_limit_reached(plant.plant_id, label, candidate.source_domain):
                        self._record_without_download(plant, label, candidate, "domain_limit")
                        continue
                    batch.append((candidate, executor.submit(self.downloader.download, candidate)))
                if not batch:
                    break
                future_to_candidate = {future: candidate for candidate, future in batch}
                for future in as_completed(future_to_candidate):
                    candidate = future_to_candidate[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # Defensive: one worker must not stop the run.
                        self.store.record_failure(
                            stable_image_id(candidate.image_url), plant, label, candidate, None, f"worker_error:{exc}"
                        )
                        continue
                    if self.store.accepted_count(plant.plant_id, label) >= target:
                        self._record_without_download(plant, label, candidate, "target_reached")
                    else:
                        self._process_download(plant, label, result)
        accepted = self.store.accepted_count(plant.plant_id, label)
        if accepted < target:
            LOGGER.warning(
                "INCOMPLETE %s/%s: %s/%s accepted after %s new candidate attempts",
                plant.plant_id,
                label,
                accepted,
                target,
                attempted,
            )

    def _candidates(self, queries: tuple[str, ...]) -> Iterator[SearchCandidate]:
        # Each query gets the configured cap across all providers, then streams
        # are interleaved so early queries/providers cannot monopolize collection.
        per_provider = max(1, math.ceil(self.config.max_images_per_query / len(self.providers)))
        streams = [self._safe_search(provider, query, per_provider) for query in queries for provider in self.providers]
        yield from round_robin(streams)  # type: ignore[misc]

    @staticmethod
    def _safe_search(
        provider: SearchProvider, query: str, max_results: int
    ) -> Iterator[SearchCandidate]:
        try:
            yield from provider.search_images(query, max_results)
        except Exception as exc:
            LOGGER.error("Search failed for %s query %r: %s", provider.name, query, exc)

    def _domain_limit_reached(self, plant_id: str, label: str, domain: str) -> bool:
        return self.store.domain_counts[(plant_id, label, domain)] >= self.config.max_images_per_domain

    def _process_download(self, plant: PlantRecord, label: str, result: DownloadResult) -> None:
        candidate = result.candidate
        image_id = stable_image_id(candidate.image_url)
        if result.content is None:
            self.store.record_failure(
                image_id, plant, label, candidate, result.status_code, result.error or "download_failed"
            )
            return
        validation = self.validator.validate(result.content, candidate)
        common = {
            "image_id": image_id,
            "plant": plant,
            "label": label,
            "candidate": candidate,
            "validation": validation,
            "original_filename": result.original_filename,
            "original_content_type": result.content_type,
            "file_size": len(result.content),
        }
        if not validation.valid:
            self.store.record_image(
                build_image_row(
                    **common,
                    verification_status="rejected",
                    rejection_reason=validation.rejection_reason,
                )
            )
            return
        exact = self.store.duplicates.exact_match(validation.sha256)
        if exact:
            self.store.record_image(
                build_image_row(
                    **common,
                    verification_status="rejected",
                    rejection_reason="exact_duplicate",
                    exact_duplicate_of=exact,
                )
            )
            return
        near = self.store.duplicates.near_match(validation.phash, self.config.phash_threshold)
        if near:
            self.store.record_image(
                build_image_row(
                    **common,
                    verification_status="rejected",
                    rejection_reason="near_duplicate",
                    near_duplicate_of=near[0],
                    phash_distance=near[1],
                )
            )
            return
        if validation.review_flags:
            local_path = self._save_image(
                plant.plant_id, label, image_id, validation, review=True
            )
            self.store.record_image(
                build_image_row(
                    **common,
                    verification_status="needs_review",
                    rejection_reason="review_flag:" + "|".join(validation.review_flags),
                    local_path=local_path,
                )
            )
            return
        # Recheck after concurrent downloads: another item from the same domain
        # may have been accepted while this request was in flight.
        if self._domain_limit_reached(plant.plant_id, label, candidate.source_domain):
            self.store.record_image(
                build_image_row(**common, verification_status="rejected", rejection_reason="domain_limit")
            )
            return
        local_path = self._save_image(plant.plant_id, label, image_id, validation, review=False)
        self.store.record_image(
            build_image_row(
                **common,
                verification_status="unverified",
                local_path=local_path,
            )
        )

    def _save_image(
        self,
        plant_id: str,
        label: str,
        image_id: str,
        validation: ValidationResult,
        *,
        review: bool,
    ) -> str:
        if validation.stored_bytes is None:
            raise ValueError("Cannot save an image without bytes")
        base = self.store.paths.rejected / "review" if review else self.store.paths.raw
        destination = base / plant_id / label / f"{image_id}{validation.extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as handle:
            handle.write(validation.stored_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination.relative_to(self.store.paths.root).as_posix()

    def _record_without_download(
        self, plant: PlantRecord, label: str, candidate: SearchCandidate, reason: str
    ) -> None:
        empty = ValidationResult(valid=False, rejection_reason=reason)
        self.store.record_image(
            build_image_row(
                image_id=stable_image_id(candidate.image_url),
                plant=plant,
                label=label,
                candidate=candidate,
                validation=empty,
                original_filename="",
                original_content_type="",
                file_size=0,
                verification_status="rejected",
                rejection_reason=reason,
            )
        )

