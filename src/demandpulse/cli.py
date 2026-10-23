import argparse
import json
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from demandpulse.config import Settings
from demandpulse.forecasting.panel import ORIGINS, build_panel
from demandpulse.forecasting.runner import (
    ALL_MODELS,
    combine_run,
    load_inputs,
    prepare_subset,
    run_model,
    save_inputs,
    save_run,
    write_report,
)
from demandpulse.ingest.bronze import ingest_source
from demandpulse.ingest.silver import build_silver
from demandpulse.ingest.sources import DEFAULT_REGISTRY, get_source, load_sources
from demandpulse.logging_setup import configure_logging
from demandpulse.quality import run_quality_checks
from demandpulse.risk.runner import PRIMARY_MODEL, TOP_K, load_store_weeks, run_risk
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

    forecast = commands.add_parser("forecast", help="forecasting panel, backtests and reports")
    forecast_commands = forecast.add_subparsers(dest="forecast_command", required=True)
    prepare = forecast_commands.add_parser(
        "prepare", help="build the weekly panel and representative subset from Gold"
    )
    prepare.add_argument("--source-glob", default="full-part-*")
    prepare.add_argument("--work-dir", type=Path, default=Path("data/work"))
    backtest = forecast_commands.add_parser("backtest", help="run models over the shared backtest")
    backtest.add_argument(
        "models", nargs="+", help=f"model names or 'all': {', '.join(ALL_MODELS)}"
    )
    backtest.add_argument("--run", default="default")
    backtest.add_argument(
        "--origins", help=f"comma-separated, default {','.join(map(str, ORIGINS))}"
    )
    backtest.add_argument("--workers", type=int, default=8)
    backtest.add_argument("--options", default="{}", help="JSON options for neuralprophet/greykite")
    combine = forecast_commands.add_parser(
        "combine", help="save an equal-weight average of models already in a run"
    )
    combine.add_argument("name")
    combine.add_argument("components", nargs="+")
    combine.add_argument("--run", default="default")
    combine.add_argument("--note", default="")
    report = forecast_commands.add_parser("report", help="write the model comparison report")
    report.add_argument("--run", default="default")
    report.add_argument(
        "--origins", help=f"origins in the run, default {','.join(map(str, ORIGINS))}"
    )

    risk = commands.add_parser("risk", help="revenue-risk ranking, classifier and report")
    risk_commands = risk.add_subparsers(dest="risk_command", required=True)
    risk_run = risk_commands.add_parser(
        "run", help="score forecasts against targets and write the risk outputs and report"
    )
    risk_run.add_argument(
        "--forecast-run",
        action="append",
        dest="forecast_runs",
        help="forecast run to read, repeatable (default: main and robustness)",
    )
    risk_run.add_argument("--primary-model", default=PRIMARY_MODEL)
    risk_run.add_argument("--top-k", type=int, default=TOP_K)
    risk_run.add_argument("--source-glob", default="full-part-*")
    risk_run.add_argument("--work-dir", type=Path, default=Path("data/work"))
    return parser


def _gold_config(args: argparse.Namespace, settings: Settings) -> GoldConfig:
    return GoldConfig(
        lake_root=settings.storage_root,
        source_glob=args.source_glob,
        window_weeks=getattr(args, "window_weeks", 13),
        year_weeks=getattr(args, "year_weeks", 52),
        aws_profile=settings.aws_profile if settings.s3_bucket else None,
        aws_region=settings.aws_region,
        work_dir=args.work_dir,
        memory_limit=getattr(args, "memory_limit", "8GB"),
        threads=getattr(args, "threads", 8),
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


def _run_forecast(args: argparse.Namespace) -> int:
    settings = _settings(args)
    store = open_store(settings)
    if args.forecast_command == "prepare":
        panel = build_panel(_gold_config(args, settings))
        subset = prepare_subset(panel)
        save_inputs(store, panel, subset)
        logger.info(
            "saved panel (%d rows) and subset (%d series)", panel.height, subset.series.height
        )
        return 0

    panel, subset = load_inputs(store)
    if args.forecast_command == "backtest":
        names = list(ALL_MODELS) if args.models == ["all"] else args.models
        origins = tuple(int(o) for o in args.origins.split(",")) if args.origins else None
        for name in names:
            model_run = run_model(
                name,
                panel,
                subset,
                origins=origins,
                workers=args.workers,
                options=json.loads(args.options),
            )
            save_run(store, args.run, model_run)
            logger.info("saved %s (%s scope) to run %s", name, model_run.scope, args.run)
        return 0

    if args.forecast_command == "combine":
        combined = combine_run(store, args.run, args.name, args.components, args.note)
        save_run(store, args.run, combined)
        logger.info("saved %s (average of %s)", args.name, ", ".join(args.components))
        return 0

    origins = tuple(int(o) for o in args.origins.split(",")) if args.origins else ORIGINS
    text, _ = write_report(store, args.run, panel, subset, origins)
    print(text)
    return 0


def _run_risk(args: argparse.Namespace) -> int:
    settings = _settings(args)
    store = open_store(settings)
    panel, _ = load_inputs(store)
    store_weeks = load_store_weeks(_gold_config(args, settings))
    text, _ = run_risk(
        store,
        panel,
        store_weeks,
        runs=args.forecast_runs or ["main", "robustness"],
        primary=args.primary_model,
        top_k=args.top_k,
    )
    print(text)
    return 0


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
    if args.command == "forecast":
        return _run_forecast(args)
    if args.command == "risk":
        return _run_risk(args)

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
