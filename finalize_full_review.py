"""Convert a reviewed sample-number allowlist into binary review decisions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_numbers(path: Path) -> set[int]:
    selected: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                selected.update(range(start, end + 1))
            else:
                selected.add(int(token))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.review.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    accepted = parse_numbers(args.accepted)
    available = {int(row["sample_no"]) for row in rows}
    unknown = accepted - available
    if unknown:
        raise SystemExit(f"Accepted sample numbers not present in review index: {sorted(unknown)}")

    for row in rows:
        sample_no = int(row["sample_no"])
        if sample_no in accepted:
            row["review_decision"] = "accept"
            row["review_reason"] = "visually_sure_target_and_health_label_at_320x240"
        else:
            row["review_decision"] = "reject"
            row["review_reason"] = "not_sure_after_single_quick_visual_pass"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    healthy = [row for row in rows if row["proposed_health_label"] == "healthy"]
    unhealthy = [row for row in rows if row["proposed_health_label"] == "unhealthy"]
    print(
        f"Healthy accepted/rejected: "
        f"{sum(row['review_decision'] == 'accept' for row in healthy)}/"
        f"{sum(row['review_decision'] == 'reject' for row in healthy)}"
    )
    print(
        f"Unhealthy accepted/rejected: "
        f"{sum(row['review_decision'] == 'accept' for row in unhealthy)}/"
        f"{sum(row['review_decision'] == 'reject' for row in unhealthy)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
