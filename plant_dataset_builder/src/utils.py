from __future__ import annotations

import hashlib
import html
import re
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import unquote, urlparse

from .models import DatasetPaths


HTML_TAG_RE = re.compile(r"<[^>]+>")


def query_language(query: str) -> str:
    return "bn" if any("\u0980" <= char <= "\u09ff" for char in query) else "en"


def domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def filename_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    return name[:255] if name else "download"


def clean_html(value: str | None) -> str:
    return html.unescape(HTML_TAG_RE.sub(" ", value or "")).strip()


def stable_image_id(image_url: str) -> str:
    return hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:24]


def make_dataset_paths(root: Path) -> DatasetPaths:
    paths = DatasetPaths(
        root=root,
        raw=root / "raw",
        metadata=root / "metadata",
        logs=root / "logs",
        rejected=root / "rejected",
    )
    for path in (paths.raw, paths.metadata, paths.logs, paths.rejected):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def round_robin(iterables: Iterable[Iterable[object]]) -> Iterator[object]:
    """Yield one item from each iterable in turn until all are exhausted."""
    pending = deque(iter(item) for item in iterables)
    while pending:
        iterator = pending.popleft()
        try:
            yield next(iterator)
        except StopIteration:
            continue
        pending.append(iterator)

