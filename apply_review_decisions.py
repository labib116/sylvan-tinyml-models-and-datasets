"""Apply manual reject decisions without permanently deleting source evidence."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("whole_csv_candidates"))
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--round", default="round1")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    raw_root = (dataset / "raw").resolve()
    quarantine_root = (dataset / "quarantine" / "rejected" / args.round).resolve()
    manifest_path = dataset / "metadata" / "images.csv"
    rejected_path = dataset / "metadata" / "rejected_images.csv"
    backup_path = dataset / "metadata" / f"images.before_{args.round}.csv"

    decisions = read_csv(args.decisions)
    rejected_decisions = {
        row["image_id"]: row for row in decisions if row.get("review_decision", "").strip().lower() == "reject"
    }
    if not rejected_decisions:
        print("No reject decisions found; nothing changed.")
        return 0

    manifest = read_csv(manifest_path)
    manifest_fields = list(manifest[0])
    by_id = {row["image_id"]: row for row in manifest}
    missing = sorted(set(rejected_decisions) - set(by_id))
    if missing:
        raise SystemExit(f"Refusing partial update; reject IDs absent from active manifest: {', '.join(missing)}")

    resolved_moves: list[tuple[Path, Path, dict[str, str], dict[str, str]]] = []
    for image_id, decision in rejected_decisions.items():
        metadata = by_id[image_id]
        source = (dataset / metadata["local_path"]).resolve()
        if not source.is_relative_to(raw_root):
            raise SystemExit(f"Unsafe source outside raw dataset: {source}")
        if not source.is_file():
            raise SystemExit(f"Source file is missing: {source}")
        destination = (
            quarantine_root
            / metadata["plant_id"]
            / metadata["proposed_health_label"]
            / source.name
        ).resolve()
        if not destination.is_relative_to(quarantine_root):
            raise SystemExit(f"Unsafe quarantine destination: {destination}")
        if destination.exists():
            raise SystemExit(f"Quarantine destination already exists: {destination}")
        resolved_moves.append((source, destination, metadata, decision))

    if backup_path.exists():
        raise SystemExit(f"Backup already exists; refusing to overwrite: {backup_path}")
    shutil.copy2(manifest_path, backup_path)

    rejected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rejected_rows: list[dict[str, str]] = read_csv(rejected_path) if rejected_path.exists() else []
    rejected_fields = manifest_fields + [
        "original_local_path",
        "quarantine_path",
        "review_round",
        "review_decision",
        "review_notes",
        "rejected_at",
    ]

    moved: list[tuple[Path, Path]] = []
    try:
        for source, destination, metadata, decision in resolved_moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
            rejected = dict(metadata)
            rejected.update(
                {
                    "original_local_path": metadata["local_path"],
                    "quarantine_path": destination.relative_to(dataset).as_posix(),
                    "review_round": args.round,
                    "review_decision": "reject",
                    "review_notes": decision.get("review_notes", ""),
                    "rejected_at": rejected_at,
                }
            )
            rejected_rows.append(rejected)

        active = [row for row in manifest if row["image_id"] not in rejected_decisions]
        write_csv(manifest_path, manifest_fields, active)
        write_csv(rejected_path, rejected_fields, rejected_rows)
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
        shutil.copy2(backup_path, manifest_path)
        raise

    print(f"Quarantined {len(resolved_moves)} files; active manifest now has {len(active)} rows.")
    print(f"Backup: {backup_path}")
    print(f"Rejection log: {rejected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
