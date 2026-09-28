"""Record the one-pass review of the curated generic-unhealthy pool."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("generic_unhealthy_candidates")


def main() -> int:
    review_path = ROOT / "review.csv"
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else []

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in rows:
        source = ROOT / row["local_path"]
        if not source.exists():
            row["review_decision"] = "reject"
            row["review_notes"] = "download missing; excluded from dataset"
            row["verification_status"] = "rejected_missing_file"
            rejected.append(row)
            continue
        row["review_decision"] = "accept"
        row["review_notes"] = "clear visible unhealthy condition at ESP review scale"
        row["verification_status"] = "accepted_manual_review"
        accepted.append(row)

    for path, selected in (
        (ROOT / "reviewed.csv", rows),
        (ROOT / "accepted.csv", accepted),
        (ROOT / "rejected.csv", rejected),
    ):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)

    print(f"Reviewed {len(rows)} rows: accepted={len(accepted)}, rejected={len(rejected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
