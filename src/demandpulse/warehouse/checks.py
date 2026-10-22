"""Validation of the built Gold layer against Silver totals and business invariants.

Each check is a SQL query returning a count of violations (0 = pass), run over the exported
Gold Parquet. The same checks run in CI on the synthetic fixture and after every real build.
"""

import json
from dataclasses import dataclass

import duckdb
from pydantic import BaseModel

from demandpulse.warehouse.build import STEPS, GoldConfig, connect

# Money is exact (DECIMAL); ratios and log-based effects are floating point.
MONEY_TOLERANCE = 0.005
RATIO_TOLERANCE = 1e-9
EFFECT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Check:
    name: str
    sql: str  # may use {silver_rows}, {silver_spend}, {money_tol}, {ratio_tol}, {effect_tol}


CHECKS: tuple[Check, ...] = (
    # --- reconciliation to Silver totals recorded in the manifest ---
    Check(
        "basket_lines_equal_silver_rows",
        "SELECT CAST(sum(lines) <> {silver_rows} AS INTEGER) FROM fct_basket",
    ),
    Check(
        "basket_revenue_equals_silver_spend",
        "SELECT CAST(abs(sum(revenue) - {silver_spend}) > {money_tol} AS INTEGER) FROM fct_basket",
    ),
    Check(
        "store_week_revenue_equals_silver_spend",
        "SELECT CAST(abs(sum(revenue) - {silver_spend}) > {money_tol} AS INTEGER) "
        "FROM fct_store_week",
    ),
    Check(
        "store_week_baskets_equal_basket_count",
        "SELECT CAST((SELECT sum(baskets) FROM fct_store_week) "
        "<> (SELECT count(*) FROM fct_basket) AS INTEGER)",
    ),
    Check(
        "store_category_revenue_equals_silver_spend",
        "SELECT CAST(abs(sum(revenue) - {silver_spend}) > {money_tol} AS INTEGER) "
        "FROM fct_store_category_week",
    ),
    Check(
        "basket_mix_revenue_equals_silver_spend",
        "SELECT CAST(abs(sum(revenue) - {silver_spend}) > {money_tol} AS INTEGER) "
        "FROM mart_basket_mix",
    ),
    Check(
        "trading_pattern_baskets_equal_basket_count",
        "SELECT CAST((SELECT sum(baskets) FROM mart_trading_pattern) "
        "<> (SELECT count(*) FROM fct_basket) AS INTEGER)",
    ),
    Check(
        "cohort_revenue_equals_loyalty_revenue",
        "SELECT CAST(abs((SELECT sum(revenue) FROM mart_customer_cohort) "
        "- (SELECT sum(revenue) FROM fct_basket WHERE is_loyalty)) > {money_tol} AS INTEGER)",
    ),
    # --- source assumptions ---
    Check(
        "baskets_have_constant_attributes", "SELECT count(*) FROM fct_basket WHERE NOT consistent"
    ),
    Check(
        "loyalty_flag_matches_customer_code",
        "SELECT count(*) FROM fct_basket WHERE is_loyalty <> (cust_code IS NOT NULL)",
    ),
    # --- grain uniqueness ---
    Check(
        "basket_grain_unique",
        "SELECT count(*) - count(DISTINCT basket_id) FROM fct_basket",
    ),
    Check(
        "store_week_grain_unique",
        "SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT shop_week, store_code "
        "FROM fct_store_week)) FROM fct_store_week",
    ),
    Check(
        "store_category_week_grain_unique",
        "SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT shop_week, store_code, "
        "prod_code_30 FROM fct_store_category_week)) FROM fct_store_category_week",
    ),
    Check(
        "scorecard_grain_unique",
        "SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT comparison, store_code "
        "FROM mart_store_scorecard)) FROM mart_store_scorecard",
    ),
    Check(
        "variance_grain_unique",
        "SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT comparison, store_code "
        "FROM mart_revenue_variance_store)) FROM mart_revenue_variance_store",
    ),
    # --- business invariants ---
    Check(
        "category_shares_sum_to_one",
        "SELECT count(*) FROM (SELECT comparison, sum(current_share) AS c, sum(base_share) AS b "
        "FROM mart_category_benchmark GROUP BY comparison) "
        "WHERE abs(c - 1) > {ratio_tol_loose} OR abs(b - 1) > {ratio_tol_loose}",
    ),
    Check(
        "category_contributions_equal_national_growth",
        "SELECT count(*) FROM (SELECT c.comparison, sum(c.contribution_to_growth_pp) AS contrib, "
        "100 * (n.current_revenue / n.base_revenue - 1) AS growth "
        "FROM mart_category_benchmark c JOIN mart_region_benchmark n USING (comparison) "
        "WHERE n.store_region = 'ALL' GROUP BY c.comparison, n.current_revenue, n.base_revenue) "
        "WHERE abs(contrib - growth) > {ratio_tol_loose}",
    ),
    Check(
        "variance_effects_sum_to_revenue_change",
        "SELECT count(*) FROM mart_revenue_variance_store WHERE abs(revenue_change - "
        "(trading_weeks_effect + traffic_effect + basket_size_effect + price_mix_effect "
        "+ new_store_effect + closed_store_effect)) > {effect_tol}",
    ),
    Check(
        "national_bridge_closes",
        "SELECT count(*) FROM mart_revenue_variance_region WHERE abs(base_revenue + "
        "trading_weeks_effect + traffic_effect + basket_size_effect + price_mix_effect "
        "+ new_store_effect + closed_store_effect - current_revenue) > {effect_tol}",
    ),
    Check(
        "variance_matches_scorecard_revenue",
        "SELECT count(*) FROM mart_revenue_variance_store v JOIN mart_store_scorecard s "
        "USING (comparison, store_code) WHERE abs(v.current_revenue - s.current_revenue) "
        "> {effect_tol} OR abs(v.base_revenue - s.base_revenue) > {effect_tol}",
    ),
    Check(
        "rfm_scores_in_range_and_segmented",
        "SELECT count(*) FROM mart_customer_rfm WHERE r_score NOT BETWEEN 1 AND 5 "
        "OR f_score NOT BETWEEN 1 AND 5 OR m_score NOT BETWEEN 1 AND 5 "
        "OR rfm_segment IS NULL OR recency_weeks < 0",
    ),
    Check(
        "cohorts_fully_active_in_first_period",
        "SELECT count(*) FROM mart_customer_cohort WHERE periods_since_first = 0 "
        "AND abs(retention_rate - 1) > {ratio_tol_loose}",
    ),
    Check(
        "retention_never_exceeds_one",
        "SELECT count(*) FROM mart_customer_cohort WHERE retention_rate > 1 + {ratio_tol_loose}",
    ),
    Check(
        "customer_dimension_matches_basket_customers",
        "SELECT CAST((SELECT count(*) FROM dim_customer) <> "
        "(SELECT count(DISTINCT cust_code) FROM fct_basket) AS INTEGER)",
    ),
)


class CheckOutcome(BaseModel):
    name: str
    violations: int
    passed: bool
    error: str | None = None


def open_gold(config: GoldConfig) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a view per exported Gold table, so only published data is checked."""
    con = connect(config, database=":memory:")
    read_manifest(con, config)  # fail early and clearly if Gold was never built
    for step in STEPS:
        for export in step.exports:
            if export.partition_by:
                path = f"{config.gold_prefix}/{export.table}/*/*.parquet"
                source = f"read_parquet('{path}', hive_partitioning = true)"
            else:
                source = f"read_parquet('{config.gold_prefix}/{export.table}.parquet')"
            con.execute(f"CREATE OR REPLACE VIEW {export.table} AS SELECT * FROM {source}")
    return con


def read_manifest(con: duckdb.DuckDBPyConnection, config: GoldConfig) -> dict:
    path = f"{config.gold_prefix}/_manifest.json"
    try:
        row = con.execute("SELECT content FROM read_text(?)", [path]).fetchone()
    except duckdb.IOException as err:
        raise FileNotFoundError(f"Gold manifest not found at {path}; build Gold first") from err
    if row is None:
        raise FileNotFoundError(f"Gold manifest not found at {path}; build Gold first")
    return json.loads(row[0])


def run_checks(config: GoldConfig, checks: tuple[Check, ...] = CHECKS) -> list[CheckOutcome]:
    con = open_gold(config)
    try:
        manifest = read_manifest(con, config)
        values = {
            "silver_rows": int(manifest["silver_rows"]),
            "silver_spend": manifest["silver_spend"],
            "money_tol": MONEY_TOLERANCE,
            "ratio_tol": RATIO_TOLERANCE,
            "ratio_tol_loose": 1e-6,
            "effect_tol": EFFECT_TOLERANCE,
        }
        outcomes: list[CheckOutcome] = []
        for check in checks:
            try:
                row = con.execute(check.sql.format(**values)).fetchone()
                violations = int(row[0]) if row and row[0] is not None else 0
                outcomes.append(
                    CheckOutcome(name=check.name, violations=violations, passed=violations == 0)
                )
            except duckdb.Error as err:
                # A check that cannot run is a failure: an unverified mart must not look valid.
                outcomes.append(
                    CheckOutcome(name=check.name, violations=-1, passed=False, error=str(err)[:300])
                )
        return outcomes
    finally:
        con.close()
