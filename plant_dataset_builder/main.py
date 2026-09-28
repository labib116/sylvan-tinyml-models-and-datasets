from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.collector import Collector
from src.config import Config, load_config
from src.csv_loader import load_plants
from src.downloader import ImageDownloader
from src.metadata import MetadataStore
from src.models import PlantRecord
from src.search import build_providers, describe_query_url
from src.utils import make_dataset_paths
from src.validator import ImageValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an auditable plant-health image dataset.")
    parser.add_argument("--csv", required=True, help="Plant/query CSV (UTF-8).")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="YAML configuration path (default: config.yaml beside main.py).",
    )
    parser.add_argument(
        "--plant",
        action="append",
        help="Plant ID or English name; repeat to select multiple plants.",
    )
    parser.add_argument("--label", choices=("healthy", "unhealthy"), help="Collect only one label.")
    parser.add_argument("--limit", type=positive_int, help="Accepted target per selected plant/label.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without API calls/downloads.")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def setup_logging(config: Config) -> None:
    paths = make_dataset_paths(config.dataset_root)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(paths.logs / "collection.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)


def select_plants(plants: list[PlantRecord], selections: list[str] | None) -> list[PlantRecord]:
    if not selections:
        return plants
    wanted = {item.casefold() for value in selections for item in value.split(",") if item.strip()}
    chosen = [
        plant
        for plant in plants
        if plant.plant_id.casefold() in wanted or plant.english_name.casefold() in wanted
    ]
    matched = {plant.plant_id.casefold() for plant in chosen} | {plant.english_name.casefold() for plant in chosen}
    missing = sorted(wanted - matched)
    if missing:
        available = ", ".join(plant.plant_id for plant in plants)
        raise ValueError(f"Unknown plant(s): {', '.join(missing)}. Available IDs: {available}")
    return chosen


def target_for(plant: PlantRecord, label: str, config: Config, limit: int | None) -> int:
    return limit if limit is not None else plant.target_for(label, config.target_per_class)


def print_dry_run(
    plants: list[PlantRecord], labels: list[str], config: Config, limit: int | None
) -> None:
    providers = build_providers(config)
    print(f"DRY RUN: dataset root = {config.dataset_root}")
    print(f"Providers: {', '.join(provider.name for provider in providers)}")
    total_pairs = 0
    for plant in plants:
        for label in labels:
            total_pairs += 1
            target = target_for(plant, label, config, limit)
            queries = plant.queries_for(label)
            budget = int(target * config.oversampling_factor + 0.999999)
            print(f"\n{plant.plant_id}/{label}: target={target}, candidate_budget~={budget}, queries={len(queries)}")
            for query in queries:
                language = "bn" if any("\u0980" <= c <= "\u09ff" for c in query) else "en"
                print(f"  [{language}] {query}")
                for provider in providers:
                    print(f"    {provider.name}: {describe_query_url(provider, query)}")
    print(f"\nPlanned plant/label pairs: {total_pairs}; no network requests or files written.")


def print_summary(store: MetadataStore, plants: list[PlantRecord], labels: list[str], config: Config, limit: int | None) -> None:
    healthy = sum(store.accepted_count(plant.plant_id, "healthy") for plant in plants)
    unhealthy = sum(store.accepted_count(plant.plant_id, "unhealthy") for plant in plants)
    remaining = 0
    print("\nCollection summary")
    for plant in plants:
        counts = {label: store.accepted_count(plant.plant_id, label) for label in ("healthy", "unhealthy")}
        for label in labels:
            remaining += max(0, target_for(plant, label, config, limit) - counts[label])
        print(f"  {plant.plant_id}: healthy={counts['healthy']}, unhealthy={counts['unhealthy']}, total={sum(counts.values())}")
    print(f"Total accepted: {healthy + unhealthy} (healthy={healthy}, unhealthy={unhealthy})")
    print(
        "Failures/rejections: "
        f"downloads={store.stats['failed_downloads']}, "
        f"exact_duplicates={store.stats['exact_duplicates']}, "
        f"near_duplicates={store.stats['near_duplicates']}, "
        f"low_resolution={store.stats['low_resolution']}, "
        f"needs_review={store.stats['needs_review']}"
    )
    print(f"Remaining selected target: {remaining}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    config = load_config(args.config)
    load_dotenv(config.config_path.parent / ".env")
    plants = load_plants(args.csv)
    selected = select_plants(plants, args.plant)
    labels = [args.label] if args.label else ["healthy", "unhealthy"]
    if args.dry_run:
        print_dry_run(selected, labels, config, args.limit)
        return 0

    setup_logging(config)
    paths = make_dataset_paths(config.dataset_root)
    store = MetadataStore(paths, config.skip_failed_urls_on_resume)
    providers = build_providers(config)
    user_agent = os.getenv(
        "DATASET_USER_AGENT",
        "PlantHealthDatasetBuilder/1.0 (set DATASET_USER_AGENT with contact info)",
    )
    collector = Collector(
        config,
        store,
        providers,
        ImageDownloader(config, user_agent),
        ImageValidator(config),
    )
    interrupted = False
    try:
        for plant in selected:
            for label in labels:
                collector.collect(plant, label, target_for(plant, label, config, args.limit))
    except KeyboardInterrupt:
        interrupted = True
        logging.getLogger(__name__).warning("Interrupted; completed rows are already flushed and resumable.")
    finally:
        store.write_summary(plants, config.target_per_class)
        print_summary(store, selected, labels, config, args.limit)
    return 130 if interrupted else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
