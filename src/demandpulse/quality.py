"""Data-quality validation and financial reconciliation for Bronze and Silver layers.

Row-level rules are pure functions over one Silver week (`week_violations`) so each rule is
unit-tested; `run_quality_checks` aggregates them across weeks, reconciles Bronze to Silver
and writes a JSON report next to the data. Statuses:
  fail  - data cannot be trusted for analytics until investigated
  warn  - unexpected but not corrupting; documented in the report
  pass
"""

import io
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import polars as pl
from pydantic import BaseModel

from demandpulse.ingest.bronze import WeekManifest, bronze_key, calendar_key
from demandpulse.ingest.silver import (
    SilverManifest,
    bronze_manifest_prefix,
    silver_key,
    silver_manifest_key,
)
from demandpulse.storage import Store

logger = logging.getLogger(__name__)

Status = Literal["pass", "warn", "fail"]

KEY_COLUMNS = [
    "shop_week",
    "shop_date",
    "quantity",
    "spend",
    "prod_code",
    "basket_id",
    "store_code",
    "store_format",
    "store_region",
]
# Observed in the data: the mission value is "Nonfood" (the user guide says "Non Food").
ALLOWED_VALUES: dict[str, list[str]] = {
    "basket_size": ["L", "M", "S"],
    "basket_price_sensitivity": ["LA", "MM", "UM", "XX"],
    "basket_type": ["Small Shop", "Top Up", "Full Shop", "XX"],
    "basket_dominant_mission": ["Fresh", "Grocery", "Mixed", "Nonfood", "XX"],
    "store_format": ["LS", "MS", "SS", "XLS"],
}
SAMPLE_UNEXPECTED_VALUES = 5


class CheckResult(BaseModel):
    name: str
    status: Status
    count: int = 0
    detail: str = ""


class QualityReport(BaseModel):
    source: str
    generated_at: datetime
    weeks_checked: int
    rows: int
    baskets: int
    spend_total: Decimal
    loyalty_row_share: float
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def to_markdown(self) -> str:
        lines = [
            f"# Data quality report: {self.source}",
            "",
            f"Weeks checked: {self.weeks_checked} | rows: {self.rows:,} | "
            f"baskets: {self.baskets:,} | spend: {self.spend_total:,} | "
            f"loyalty row share: {self.loyalty_row_share:.1%}",
            "",
            "| check | status | count | detail |",
            "|---|---|---|---|",
        ]
        lines += [f"| {c.name} | {c.status} | {c.count:,} | {c.detail} |" for c in self.checks]
        return "\n".join(lines) + "\n"


def _baskets_with_multiple(week: pl.DataFrame, column: str) -> int:
    per_basket = week.group_by("basket_id").agg(pl.col(column).n_unique().alias("n"))
    return per_basket.filter(pl.col("n") > 1).height


def week_violations(week: pl.DataFrame) -> dict[str, int]:
    """Count rule violations in one Silver week. Zero everywhere means clean."""
    violations = {
        "null_key_values": int(sum(week[c].null_count() for c in KEY_COLUMNS)),
        "nonpositive_quantity": int((week["quantity"] <= 0).sum()),
        "negative_spend": int((week["spend"] < 0).sum()),
        "zero_spend": int((week["spend"] == 0).sum()),
        "weekday_out_of_range": int((~week["shop_weekday"].is_between(1, 7)).sum()),
        "hour_out_of_range": int((~week["shop_hour"].is_between(0, 23)).sum()),
        "duplicate_basket_product": week.height - week.unique(["basket_id", "prod_code"]).height,
        "date_outside_week": int(
            (
                week["week_start"].is_null()
                | (week["shop_date"] < week["week_start"])
                | (week["shop_date"] > week["week_end"])
            ).sum()
        ),
        "baskets_with_multiple_stores": _baskets_with_multiple(week, "store_code"),
        "baskets_with_multiple_dates": _baskets_with_multiple(week, "shop_date"),
        "baskets_with_multiple_customers": _baskets_with_multiple(week, "cust_code"),
    }
    for column, allowed in ALLOWED_VALUES.items():
        violations[f"unexpected_{column}"] = int((~week[column].is_in(allowed)).sum())
    return violations


def unexpected_values(week: pl.DataFrame) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for column, allowed in ALLOWED_VALUES.items():
        bad = (
            week.filter(~pl.col(column).is_in(allowed))[column]
            .unique()
            .head(SAMPLE_UNEXPECTED_VALUES)
        )
        if bad.len():
            found[column] = [str(v) for v in bad.to_list()]
    return found


WARN_ONLY = {"zero_spend", "duplicate_basket_product"}


def _read(store: Store, key: str) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(store.read_bytes(key)))


def _sum(frame: pl.DataFrame, column: str) -> Decimal:
    value = frame[column].sum()
    return Decimal(value) if value is not None else Decimal(0)


def _result(name: str, count: int, detail: str = "", warn_only: bool = False) -> CheckResult:
    if count == 0:
        return CheckResult(name=name, status="pass")
    return CheckResult(
        name=name, status="warn" if warn_only else "fail", count=count, detail=detail
    )


def run_quality_checks(source: str, store: Store) -> QualityReport:
    bronze_manifests = sorted(
        (
            WeekManifest.model_validate_json(store.read_bytes(key))
            for key in store.list_keys(bronze_manifest_prefix(source))
        ),
        key=lambda m: m.shop_week,
    )
    totals: dict[str, int] = {}
    unexpected: dict[str, set[str]] = {}
    recon = {
        "bronze_file_vs_manifest": [],
        "silver_missing": [],
        "silver_rows_vs_bronze": [],
        "silver_quantity_vs_bronze": [],
        "silver_spend_vs_bronze": [],
    }
    rows = baskets = loyalty_rows = 0
    spend_total = Decimal(0)

    for bronze in bronze_manifests:
        week = bronze.shop_week
        bronze_frame = _read(store, bronze_key(source, week))
        if (
            bronze_frame.height != bronze.rows
            or int(bronze_frame["quantity"].sum()) != bronze.quantity_sum
            or _sum(bronze_frame, "spend") != bronze.spend_sum
        ):
            recon["bronze_file_vs_manifest"].append(week)

        if not store.exists(silver_manifest_key(source, week)):
            recon["silver_missing"].append(week)
            continue
        silver_manifest = SilverManifest.model_validate_json(
            store.read_bytes(silver_manifest_key(source, week))
        )
        silver = _read(store, silver_key(source, week))

        # Row identity: nothing may vanish except counted exact duplicates.
        if silver.height + silver_manifest.duplicates_dropped != bronze.rows:
            recon["silver_rows_vs_bronze"].append(week)
        # Money and units: Silver plus the counted duplicates it dropped must equal Bronze.
        silver_quantity = int(silver["quantity"].sum()) + silver_manifest.duplicate_quantity_dropped
        if silver_quantity != bronze.quantity_sum:
            recon["silver_quantity_vs_bronze"].append(week)
        if _sum(silver, "spend") + silver_manifest.duplicate_spend_dropped != bronze.spend_sum:
            recon["silver_spend_vs_bronze"].append(week)

        for name, count in week_violations(silver).items():
            totals[name] = totals.get(name, 0) + count
        for column, values in unexpected_values(silver).items():
            unexpected.setdefault(column, set()).update(values)

        rows += silver.height
        baskets += silver["basket_id"].n_unique()
        loyalty_rows += int(silver["is_loyalty"].sum())
        spend_total += _sum(silver, "spend")

    checks: list[CheckResult] = []
    for name, weeks in recon.items():
        checks.append(_result(name, len(weeks), f"weeks: {weeks[:10]}"))
    for name, count in totals.items():
        detail = ""
        if name.startswith("unexpected_"):
            detail = f"values: {sorted(unexpected.get(name.removeprefix('unexpected_'), []))}"
        checks.append(_result(name, count, detail, warn_only=name in WARN_ONLY))

    data_weeks = [m.shop_week for m in bronze_manifests]
    checks.append(_week_gaps(source, store, data_weeks))
    if not bronze_manifests:
        checks.append(CheckResult(name="has_data", status="fail", detail="no Bronze weeks found"))

    report = QualityReport(
        source=source,
        generated_at=datetime.now(UTC),
        weeks_checked=len(bronze_manifests),
        rows=rows,
        baskets=baskets,
        spend_total=spend_total,
        loyalty_row_share=(loyalty_rows / rows) if rows else 0.0,
        checks=checks,
    )
    store.write_bytes(f"quality/{source}/report.json", report.model_dump_json(indent=2).encode())
    store.write_bytes(f"quality/{source}/report.md", report.to_markdown().encode())
    return report


def _week_gaps(source: str, store: Store, data_weeks: list[int]) -> CheckResult:
    if not data_weeks or not store.exists(calendar_key(source)):
        return CheckResult(name="calendar_gaps", status="warn", detail="no calendar or data")
    calendar = _read(store, calendar_key(source))["shop_week"].to_list()
    within = [w for w in calendar if data_weeks[0] <= w <= data_weeks[-1]]
    missing = sorted(set(within) - set(data_weeks))
    unknown = sorted(set(data_weeks) - set(calendar))
    if unknown:
        return CheckResult(
            name="calendar_gaps",
            status="fail",
            count=len(unknown),
            detail=f"not in calendar: {unknown}",
        )
    return _result(
        "calendar_gaps", len(missing), f"weeks in range with no data: {missing}", warn_only=True
    )
