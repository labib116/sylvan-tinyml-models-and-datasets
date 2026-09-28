from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from urllib.parse import quote

import requests

from .config import Config
from .models import SearchCandidate
from .utils import clean_html, domain_from_url, query_language

LOGGER = logging.getLogger(__name__)


class SearchProvider(ABC):
    name: str

    def __init__(self, config: Config, user_agent: str) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})

    def _get(self, url: str, *, params: dict[str, object]) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.config.search_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.config.request_timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                retryable = status is None or status == 429 or status >= 500
                if attempt >= self.config.search_retries or not retryable:
                    break
                delay = self.config.backoff_factor * (2**attempt)
                LOGGER.warning("%s search retry %s after %s", self.name, attempt + 1, exc)
                time.sleep(delay)
        raise RuntimeError(f"{self.name} search failed: {last_error}")

    @abstractmethod
    def search_images(self, query: str, max_results: int) -> Iterator[SearchCandidate]:
        """Yield image candidates with provenance; never download image bytes here."""


class OpenverseProvider(SearchProvider):
    name = "openverse"
    endpoint = "https://api.openverse.org/v1/images/"

    def __init__(self, config: Config, user_agent: str) -> None:
        super().__init__(config, user_agent)
        token = os.getenv("OPENVERSE_ACCESS_TOKEN", "").strip()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def search_images(self, query: str, max_results: int) -> Iterator[SearchCandidate]:
        remaining = max_results
        page = 1
        while remaining > 0:
            page_size = min(remaining, 80)
            response = self._get(
                self.endpoint,
                params={"q": query, "page": page, "page_size": page_size, "mature": "false"},
            )
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                return
            for item in results:
                image_url = str(item.get("url") or "").strip()
                source_page = str(item.get("foreign_landing_url") or "").strip()
                if not image_url or not source_page:
                    continue
                source_domain = domain_from_url(source_page) or domain_from_url(image_url)
                license_name = str(item.get("license") or "").strip()
                license_version = str(item.get("license_version") or "").strip()
                if license_name and license_version:
                    license_name = f"{license_name} {license_version}"
                yield SearchCandidate(
                    provider_result_id=str(item.get("id") or image_url),
                    search_provider=self.name,
                    query=query,
                    query_language=query_language(query),
                    image_url=image_url,
                    source_page_url=source_page,
                    source_domain=source_domain,
                    title=str(item.get("title") or ""),
                    license=license_name,
                    license_url=str(item.get("license_url") or ""),
                    attribution=str(item.get("attribution") or ""),
                    creator=str(item.get("creator") or ""),
                    provider_width=_optional_int(item.get("width")),
                    provider_height=_optional_int(item.get("height")),
                    source_collection=str(item.get("source") or item.get("provider") or ""),
                )
                remaining -= 1
                if remaining <= 0:
                    return
            page += 1
            if page > int(payload.get("page_count") or page):
                return
            time.sleep(self.config.request_delay_seconds)


class WikimediaCommonsProvider(SearchProvider):
    name = "wikimedia"
    endpoint = "https://commons.wikimedia.org/w/api.php"

    def search_images(self, query: str, max_results: int) -> Iterator[SearchCandidate]:
        remaining = max_results
        offset: int | None = None
        while remaining > 0:
            limit = min(remaining, 50)
            params: dict[str, object] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": limit,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": 1600,
                "iiextmetadatalanguage": "en",
            }
            if offset is not None:
                params["gsroffset"] = offset
            payload = self._get(self.endpoint, params=params).json()
            pages = (payload.get("query") or {}).get("pages") or []
            if not pages:
                return
            for page in pages:
                info_list = page.get("imageinfo") or []
                if not info_list:
                    continue
                info = info_list[0]
                metadata = info.get("extmetadata") or {}
                # thumburl avoids unexpectedly huge originals while retaining >=1600px
                # when Commons can generate a thumbnail. Validation records actual size.
                image_url = str(info.get("thumburl") or info.get("url") or "")
                source_page = str(info.get("descriptionurl") or "")
                if not image_url or not source_page:
                    continue
                license_name = _meta(metadata, "LicenseShortName")
                license_url = _meta(metadata, "LicenseUrl")
                yield SearchCandidate(
                    provider_result_id=str(page.get("pageid") or page.get("title") or image_url),
                    search_provider=self.name,
                    query=query,
                    query_language=query_language(query),
                    image_url=image_url,
                    source_page_url=source_page,
                    source_domain="commons.wikimedia.org",
                    title=str(page.get("title") or "").removeprefix("File:"),
                    license=license_name,
                    license_url=license_url,
                    attribution=clean_html(_meta(metadata, "Credit")),
                    creator=clean_html(_meta(metadata, "Artist")),
                    provider_width=_optional_int(info.get("thumbwidth") or info.get("width")),
                    provider_height=_optional_int(info.get("thumbheight") or info.get("height")),
                    source_collection="wikimedia_commons",
                )
                remaining -= 1
                if remaining <= 0:
                    return
            continuation = payload.get("continue") or {}
            if "gsroffset" not in continuation:
                return
            offset = int(continuation["gsroffset"])
            time.sleep(self.config.request_delay_seconds)


def _meta(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return ""


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_providers(config: Config) -> list[SearchProvider]:
    user_agent = os.getenv(
        "DATASET_USER_AGENT",
        "PlantHealthDatasetBuilder/1.0 (set DATASET_USER_AGENT with contact info)",
    )
    factories = {
        "openverse": OpenverseProvider,
        "wikimedia": WikimediaCommonsProvider,
    }
    providers: list[SearchProvider] = []
    for name in config.providers:
        if name not in factories:
            raise ValueError(f"Unknown provider {name!r}; available: {', '.join(factories)}")
        providers.append(factories[name](config, user_agent))
    return providers


def describe_query_url(provider: SearchProvider, query: str) -> str:
    if provider.name == "openverse":
        return f"{OpenverseProvider.endpoint}?q={quote(query)}"
    return f"{WikimediaCommonsProvider.endpoint}?action=query&generator=search&gsrsearch={quote(query)}"
