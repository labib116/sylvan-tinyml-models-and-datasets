from __future__ import annotations

import csv
from pathlib import Path

from .models import PlantRecord


REQUIRED_COLUMNS = {
    "plant_id",
    "english_name",
    "bangla_name",
    "scientific_name",
    "category",
    "healthy_label",
    "unhealthy_label",
    "healthy_queries",
    "unhealthy_queries",
    "target_healthy_images",
    "target_unhealthy_images",
}


def _queries(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part.strip() for part in value.split("|") if part.strip()))


def _target(value: str, field_name: str, row_number: int) -> int:
    if not value.strip():
        return 0
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {field_name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"Row {row_number}: {field_name} cannot be negative")
    return result


def load_plants(path: str | Path) -> list[PlantRecord]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        records: list[PlantRecord] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            plant_id = (row.get("plant_id") or "").strip()
            if not plant_id:
                raise ValueError(f"Row {row_number}: plant_id is empty")
            if plant_id in seen:
                raise ValueError(f"Row {row_number}: duplicate plant_id {plant_id!r}")
            seen.add(plant_id)
            healthy_queries = _queries(row.get("healthy_queries") or "")
            unhealthy_queries = _queries(row.get("unhealthy_queries") or "")
            if not healthy_queries or not unhealthy_queries:
                raise ValueError(f"Row {row_number}: both query fields must contain queries")
            known = REQUIRED_COLUMNS
            records.append(
                PlantRecord(
                    plant_id=plant_id,
                    english_name=(row.get("english_name") or "").strip(),
                    bangla_name=(row.get("bangla_name") or "").strip(),
                    scientific_name=(row.get("scientific_name") or "").strip(),
                    category=(row.get("category") or "").strip(),
                    healthy_label=(row.get("healthy_label") or "healthy").strip(),
                    unhealthy_label=(row.get("unhealthy_label") or "unhealthy").strip(),
                    healthy_queries=healthy_queries,
                    unhealthy_queries=unhealthy_queries,
                    target_healthy_images=_target(row.get("target_healthy_images") or "", "target_healthy_images", row_number),
                    target_unhealthy_images=_target(row.get("target_unhealthy_images") or "", "target_unhealthy_images", row_number),
                    extra={key: (value or "").strip() for key, value in row.items() if key not in known},
                )
            )
    return records
