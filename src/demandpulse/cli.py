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
from demandpulse.warehouse.build import STEPS, GoldConfig, build_gold
from demandpulse.warehouse.checks import CheckOutcome, run_checks

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
    parser.add_argument(
        "--allow-known-issue",
        action="store_true",
        help="process a source the registry marks as defective",
    )
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

    gold = commands.add_parser("gold", help="build or validate the Gold analytics marts")
    gold_commands = gold.add_subparsers(dest="gold_command", required=True)
    for name, help_text in {
        "build": "build Gold marts with DuckDB, then validate them",
        "validate": "run validation checks against the built Gold marts",
    }.items():
        sub = gold_commands.add_parser(name, help=help_text)
        sub.add_argument("--source-glob", default="full-part-*", help="Silver sources to read")
        sub.add_argument("--window-weeks", type=int, default=13)
        sub.add_argument("--year-weeks", type=int, default=52)
        sub.add_argument("--work-dir", type=Path, default=Path("data/work"))
        sub.add_argument("--memory-limit", default="8GB")
        sub.add_argument("--threads", type=int, default=8)
        if name == "build":
            sub.add_argument("--resume", action="store_true", help="reuse tables in the scratch DB")
            sub.add_argument(
                "--only", help=f"comma-separated steps: {', '.join(s.name for s in STEPS)}"
            )
            sub.add_argument("--no-validate", action="store_true")
    return parser


def _gold_config(args: argparse.Namespace, settings: Settings) -> GoldConfig:
    return GoldConfig(
        lake_root=settings.storage_root,
        source_glob=args.source_glob,
        window_weeks=args.window_weeks,
        year_weeks=args.year_weeks,
        aws_profile=settings.aws_profile if settings.s3_bucket else None,
        aws_region=settings.aws_region,
        work_dir=args.work_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        resume=getattr(args, "resume", False),
    )


def _print_outcomes(outcomes: list[CheckOutcome]) -> bool:
    for outcome in outcomes:
        status = "PASS" if outcome.passed else "FAIL"
        detail = f"  {outcome.error}" if outcome.error else ""
        print(f"{status}  {outcome.name}  violations={outcome.violations}{detail}")
    passed = sum(o.passed for o in outcomes)
    print(f"{passed}/{len(outcomes)} checks passed")
    return passed == len(outcomes)


def _run_gold(args: argparse.Namespace) -> int:
    config = _gold_config(args, _settings(args))
    ok = True
    if args.gold_command == "build":
        only = set(args.only.split(",")) if args.only else None
        result = build_gold(config, only=only)
        for table, rows in result.tables.items():
            logger.info("gold table %s: %s rows", table, f"{rows:,}")
        if args.no_validate:
            return 0
    ok &= _print_outcomes(run_checks(config))
    return 0 if ok else 1


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
            flag = "  [KNOWN ISSUE]" if source.known_issue else ""
            print(f"{source.name:24} {source.kind:7} {source.url}{flag}")
        return 0

    if args.command == "gold":
        return _run_gold(args)

    store = open_store(_settings(args))
    source = get_source(args.source, args.sources_file)
    if source.known_issue and not args.allow_known_issue:
        print(f"Refusing {source.name}: {source.known_issue}", file=sys.stderr)
        return 2
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
