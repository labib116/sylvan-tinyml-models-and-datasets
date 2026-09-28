"""Measure dataset image quality after matching the ESP32 camera output.

This is a technical screen only.  It cannot determine whether the pictured
species or health label is correct; that remains a manual review task.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def camera_pair(path: Path, width: int, height: int, quality: int) -> tuple[np.ndarray, np.ndarray, int]:
    with Image.open(path) as opened:
        fitted = ImageOps.fit(
            opened.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    source = np.asarray(fitted, dtype=np.float32)
    buffer = io.BytesIO()
    fitted.save(buffer, "JPEG", quality=quality, optimize=False)
    jpeg_size = buffer.tell()
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        camera = np.asarray(encoded.convert("RGB"), dtype=np.float32)
    return source, camera, jpeg_size


def metrics(source: np.ndarray, camera: np.ndarray) -> dict[str, float]:
    gray = 0.299 * camera[:, :, 0] + 0.587 * camera[:, :, 1] + 0.114 * camera[:, :, 2]
    center = gray[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    mse = float(np.mean((source - camera) ** 2))
    psnr = 99.0 if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "mean_luminance": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "dark_fraction": float(np.mean(gray <= 15.0)),
        "bright_fraction": float(np.mean(gray >= 240.0)),
        "laplacian_variance": float(np.var(laplacian)),
        "jpeg_psnr_db": psnr,
    }


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("whole_csv_candidates"))
    parser.add_argument("--output", type=Path, default=Path("quality_review/camera_quality.csv"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()

    manifest = args.dataset / "metadata" / "images.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    analyzed: list[dict[str, str | int | float]] = []
    for row in rows:
        path = args.dataset / Path(row["local_path"])
        try:
            source, camera, jpeg_size = camera_pair(path, args.width, args.height, args.jpeg_quality)
            result = metrics(source, camera)
            analyzed.append(
                {
                    "image_id": row["image_id"],
                    "plant_id": row["plant_id"],
                    "proposed_health_label": row["proposed_health_label"],
                    "local_path": row["local_path"],
                    "esp_width": args.width,
                    "esp_height": args.height,
                    "esp_jpeg_quality": args.jpeg_quality,
                    "simulated_jpeg_bytes": jpeg_size,
                    **result,
                }
            )
        except Exception as error:  # keep a complete audit instead of stopping on one file
            analyzed.append(
                {
                    "image_id": row["image_id"],
                    "plant_id": row["plant_id"],
                    "proposed_health_label": row["proposed_health_label"],
                    "local_path": row["local_path"],
                    "esp_width": args.width,
                    "esp_height": args.height,
                    "esp_jpeg_quality": args.jpeg_quality,
                    "error": str(error),
                }
            )

    valid = [row for row in analyzed if "error" not in row]
    lap_threshold = percentile([float(row["laplacian_variance"]) for row in valid], 5)
    contrast_threshold = min(18.0, percentile([float(row["contrast_std"]) for row in valid], 5))

    for row in analyzed:
        flags: list[str] = []
        if "error" in row:
            flags.append("decode_error")
        else:
            luminance = float(row["mean_luminance"])
            if luminance < 40.0 or float(row["dark_fraction"]) > 0.55:
                flags.append("too_dark")
            if luminance > 215.0 or float(row["bright_fraction"]) > 0.55:
                flags.append("too_bright")
            if float(row["contrast_std"]) < contrast_threshold:
                flags.append("low_contrast")
            if float(row["laplacian_variance"]) < lap_threshold:
                flags.append("low_detail_bottom_5pct")
        row["technical_flags"] = ";".join(flags)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "plant_id",
        "proposed_health_label",
        "local_path",
        "esp_width",
        "esp_height",
        "esp_jpeg_quality",
        "simulated_jpeg_bytes",
        "mean_luminance",
        "contrast_std",
        "dark_fraction",
        "bright_fraction",
        "laplacian_variance",
        "jpeg_psnr_db",
        "technical_flags",
        "error",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(analyzed)

    flagged = [row for row in analyzed if row["technical_flags"]]
    jpeg_sizes = [float(row["simulated_jpeg_bytes"]) for row in valid]
    psnr_values = [float(row["jpeg_psnr_db"]) for row in valid]
    print(f"Analyzed: {len(valid)}/{len(analyzed)}")
    print(f"Flagged for technical review: {len(flagged)}")
    print(f"Low-detail threshold (dataset 5th percentile): {lap_threshold:.2f}")
    print(f"Low-contrast threshold: {contrast_threshold:.2f}")
    print(
        "Simulated JPEG bytes p05/median/p95: "
        f"{percentile(jpeg_sizes, 5):.0f}/{percentile(jpeg_sizes, 50):.0f}/{percentile(jpeg_sizes, 95):.0f}"
    )
    print(
        "JPEG PSNR dB p05/median/p95: "
        f"{percentile(psnr_values, 5):.2f}/{percentile(psnr_values, 50):.2f}/{percentile(psnr_values, 95):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
