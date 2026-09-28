from __future__ import annotations

import logging
import threading
import time

import requests

from .config import Config
from .models import DownloadResult, SearchCandidate
from .utils import filename_from_url

LOGGER = logging.getLogger(__name__)


class ImageDownloader:
    """Thread-safe downloader with a bounded response size and global pacing."""

    def __init__(self, config: Config, user_agent: str) -> None:
        self.config = config
        self.user_agent = user_agent
        self._local = threading.local()
        self._rate_lock = threading.Lock()
        self._next_request = 0.0

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {"User-Agent": self.user_agent, "Accept": "image/avif,image/webp,image/*,*/*;q=0.5"}
            )
            self._local.session = session
        return session

    def _pace(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_request - now)
            self._next_request = max(now, self._next_request) + self.config.request_delay_seconds
        if wait:
            time.sleep(wait)

    def download(self, candidate: SearchCandidate) -> DownloadResult:
        last_error = ""
        status_code: int | None = None
        for attempt in range(self.config.download_retries + 1):
            try:
                self._pace()
                with self._session().get(
                    candidate.image_url,
                    timeout=self.config.request_timeout,
                    stream=True,
                    allow_redirects=True,
                ) as response:
                    status_code = response.status_code
                    response.raise_for_status()
                    declared = int(response.headers.get("Content-Length") or 0)
                    if declared > self.config.max_download_bytes:
                        raise ValueError(f"response_too_large:{declared}")
                    content = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        content.extend(chunk)
                        if len(content) > self.config.max_download_bytes:
                            raise ValueError(f"response_too_large:{len(content)}")
                    return DownloadResult(
                        candidate=candidate,
                        content=bytes(content),
                        content_type=response.headers.get("Content-Type", "").split(";", 1)[0].strip(),
                        original_filename=filename_from_url(str(response.url)),
                        status_code=status_code,
                    )
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                retryable = isinstance(exc, requests.RequestException) or str(exc).startswith("response_too_large") is False
                if attempt >= self.config.download_retries or not retryable:
                    break
                time.sleep(self.config.backoff_factor * (2**attempt))
        LOGGER.warning("Download failed: %s (%s)", candidate.image_url, last_error)
        return DownloadResult(
            candidate=candidate,
            content=None,
            content_type="",
            original_filename=filename_from_url(candidate.image_url),
            status_code=status_code,
            error=last_error or "download_failed",
        )
