from __future__ import annotations

import hashlib
import io
import warnings

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Config
from .models import SearchCandidate, ValidationResult


class ImageValidator:
    SAFE_FORMATS = {"JPEG", "PNG", "WEBP"}

    def __init__(self, config: Config) -> None:
        self.config = config

    def validate(self, content: bytes, candidate: SearchCandidate) -> ValidationResult:
        sha256 = hashlib.sha256(content).hexdigest()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as probe:
                    image_format = (probe.format or "").upper()
                    probe.verify()
                with Image.open(io.BytesIO(content)) as loaded:
                    loaded.load()
                    image = ImageOps.exif_transpose(loaded)
                    width, height = image.size
                    if image_format not in self.SAFE_FORMATS:
                        return self._reject(sha256, f"unsupported_format:{image_format or 'unknown'}")
                    if width < self.config.min_width or height < self.config.min_height:
                        return self._reject(sha256, f"low_resolution:{width}x{height}", width, height, image_format)

                    normalized = image.convert("RGB")
                    perceptual_hash = str(imagehash.phash(normalized))
                    stored_bytes, extension, mime = self._storage_bytes(content, image, image_format)
                    flags = self._review_flags(candidate, width, height)
                    return ValidationResult(
                        valid=True,
                        stored_bytes=stored_bytes,
                        extension=extension,
                        image_format=image_format,
                        mime_type=mime,
                        width=width,
                        height=height,
                        sha256=sha256,
                        phash=perceptual_hash,
                        review_flags=flags,
                    )
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            return self._reject(sha256, f"invalid_image:{type(exc).__name__}")

    @staticmethod
    def _storage_bytes(content: bytes, image: Image.Image, image_format: str) -> tuple[bytes, str, str]:
        if image_format == "JPEG":
            return content, ".jpg", "image/jpeg"
        if image_format == "PNG":
            return content, ".png", "image/png"
        # WEBP is decoded and converted to a broadly supported JPEG. The raw-byte
        # SHA256 still preserves exact-download identity in metadata.
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=95, optimize=True)
        return output.getvalue(), ".jpg", "image/jpeg"

    def _review_flags(self, candidate: SearchCandidate, width: int, height: int) -> tuple[str, ...]:
        haystack = " ".join((candidate.title, candidate.image_url, candidate.source_page_url)).casefold()
        flags = [f"keyword:{term}" for term in self.config.review_keywords if term in haystack]
        aspect = max(width / height, height / width)
        if aspect >= 4.0:
            flags.append("extreme_aspect_ratio")
        return tuple(dict.fromkeys(flags))

    @staticmethod
    def _reject(
        sha256: str,
        reason: str,
        width: int = 0,
        height: int = 0,
        image_format: str = "",
    ) -> ValidationResult:
        return ValidationResult(
            valid=False,
            width=width,
            height=height,
            image_format=image_format,
            sha256=sha256,
            rejection_reason=reason,
        )

