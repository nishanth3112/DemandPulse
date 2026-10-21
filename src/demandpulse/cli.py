import argparse
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from demandpulse.config import Settings
from demandpulse.ingest.bronze import ingest_source
from demandpulse.ingest.silver import build_silver
from demandpulse.ingest.sources import DEFAULT_REGISTRY, get_source, load_sources
from demandpulse.logging_setup import configure_logging
from demandpulse.quality import run_quality_checks
from demandpulse.storage import open_store

logger = logging.getLogger("demandpulse.cli")


def _weeks(value: str) -> set[int]:
    return {int(part) for part in value.split(",") if part.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demandpulse")
    parser.add_argument(
        "--version", action="version", version=f"demandpulse {version('demandpulse')}"
    )
    parser.add_argument("--local-root", type=Path, help="write to this directory instead of S3")
    parser.add_argument("--sources-file", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("sources", help="list registered dunnhumby sources")
    for name, help_text in {
        "ingest": "stream a source zip into Bronze Parquet",
        "silver": "build Silver from ingested Bronze weeks",
        "quality": "run data-quality and reconciliation checks",
        "run": "ingest, build Silver and run quality checks",
    }.items():
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("source")
        if name != "quality":
            sub.add_argument("--weeks", type=_weeks, help="comma-separated shop_week values")
        if name in {"ingest", "run"}:
            sub.add_argument("--max-weeks", type=int, help="stop after this many new weeks")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if args.local_root:
        # An explicit flag beats the environment, so drop any bucket from env.
        return Settings(
            aws_profile=settings.aws_profile,
            aws_region=settings.aws_region,
            local_data_root=args.local_root,
        )
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level.upper())

    if args.command == "sources":
        for source in load_sources(args.sources_file).values():
            print(f"{source.name:24} {source.kind:7} {source.url}")
        return 0

    store = open_store(_settings(args))
    source = get_source(args.source, args.sources_file)
    ok = True

    if args.command in {"ingest", "run"}:
        ingested = ingest_source(source, store, weeks=args.weeks, max_weeks=args.max_weeks)
        logger.info(
            "ingest: %d processed, %d skipped, %d failed",
            len(ingested.processed),
            len(ingested.skipped),
            len(ingested.failed),
        )
        ok &= ingested.ok
    if args.command in {"silver", "run"}:
        built = build_silver(source.name, store, weeks=args.weeks)
        logger.info(
            "silver: %d processed, %d skipped, %d failed",
            len(built.processed),
            len(built.skipped),
            len(built.failed),
        )
        ok &= built.ok
    if args.command in {"quality", "run"}:
        report = run_quality_checks(source.name, store)
        print(report.to_markdown())
        ok &= report.ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
