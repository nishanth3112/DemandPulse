# Data quality

What is checked at each layer, how to run it, and what the real run found. The numbers below are
from the full-dataset run (nine source parts, 117 weeks). They show the pipeline reproduced its
source faithfully and that the data has the structure the analysis assumes. They do not show the
data is true to any real retailer: it is synthetic.

```bash
uv run demandpulse quality full-part-1          # Bronze/Silver checks for one source
uv run demandpulse gold validate                # Gold checks against the published marts
```

## Layer 1: Bronze and Silver (per source, 22 checks)

Each source is checked after ingestion. The first five compare the Bronze files with their
manifest and Silver with Bronze (row counts, quantity and spend totals), so a dropped or altered
row fails loudly. The rest are value and structure checks on the data itself.

- `bronze_file_vs_manifest`
- `silver_missing`
- `silver_rows_vs_bronze`
- `silver_quantity_vs_bronze`
- `silver_spend_vs_bronze`
- `null_key_values`
- `nonpositive_quantity`
- `negative_spend`
- `zero_spend`
- `weekday_out_of_range`
- `hour_out_of_range`
- `duplicate_basket_product`
- `date_outside_week`
- `baskets_with_multiple_stores`
- `baskets_with_multiple_dates`
- `baskets_with_multiple_customers`
- `unexpected_basket_size`
- `unexpected_basket_price_sensitivity`
- `unexpected_basket_type`
- `unexpected_basket_dominant_mission`
- `unexpected_store_format`
- `calendar_gaps`

Checks return `pass`, `warn` or `fail`. A `fail` makes the command exit non-zero.

## Layer 2: Gold (24 checks)

Each check is a SQL query that counts violations; zero passes. They reconcile every mart back to
the Silver totals recorded in the Gold manifest and test business invariants such as additive
revenue effects.

- `basket_lines_equal_silver_rows`
- `basket_revenue_equals_silver_spend`
- `store_week_revenue_equals_silver_spend`
- `store_week_baskets_equal_basket_count`
- `store_category_revenue_equals_silver_spend`
- `basket_mix_revenue_equals_silver_spend`
- `trading_pattern_baskets_equal_basket_count`
- `cohort_revenue_equals_loyalty_revenue`
- `baskets_have_constant_attributes`
- `loyalty_flag_matches_customer_code`
- `basket_grain_unique`
- `store_week_grain_unique`
- `store_category_week_grain_unique`
- `scorecard_grain_unique`
- `variance_grain_unique`
- `category_shares_sum_to_one`
- `category_contributions_equal_national_growth`
- `variance_effects_sum_to_revenue_change`
- `national_bridge_closes`
- `variance_matches_scorecard_revenue`
- `rfm_scores_in_range_and_segmented`
- `cohorts_fully_active_in_first_period`
- `retention_never_exceeds_one`
- `customer_dimension_matches_basket_customers`

## Results of the full run

| part | weeks | rows | baskets | spend | loyalty row share | zero-spend rows |
|---|---|---|---|---|---|---|
| part-1 | 13 | 33,850,823 | 5,123,117 | 64,796,455.65 | 80.5% | 86 |
| part-2 | 13 | 34,168,112 | 5,193,658 | 64,414,854.83 | 79.9% | 87 |
| part-3 | 13 | 33,574,238 | 5,092,776 | 65,629,415.64 | 80.9% | 68 |
| part-4 | 13 | 34,778,438 | 5,222,670 | 64,087,739.25 | 81.2% | 132 |
| part-5 | 13 | 35,190,265 | 5,342,867 | 67,496,502.79 | 80.8% | 182 |
| part-6 | 13 | 34,737,830 | 5,364,306 | 66,264,883.69 | 80.1% | 128 |
| part-7 | 13 | 33,299,994 | 5,210,959 | 67,230,938.56 | 80.8% | 127 |
| part-8 | 13 | 33,129,455 | 5,218,079 | 63,062,647.91 | 81.0% | 122 |
| part-9 | 13 | 34,802,813 | 5,330,071 | 66,743,481.37 | 80.7% | 113 |
| **all** | 117 | **307,531,968** | **47,098,503** | **589,726,919.69** | | **1,045** |

- **Silver:** 9 parts x 22 checks = 198 checks, **0 failures**.
- **Gold:** 24 of 24 checks pass. Gold reconciles to Silver at 307,531,968 rows and
  spend 589,726,919.69 over 117 weeks (200607 to 200819).
- The rows and spend totals above add up to the Gold manifest totals.

### Warnings, and what was done about them

- **`zero_spend` (1,045 rows over the nine parts).** Lines with no spend. They are kept as
  delivered, not dropped, and are 0.0003% of the 307,531,968 rows.
- **`duplicate_basket_product` (9 in part 9).** Repeated basket and product lines in part 9. Exact
  duplicate rows (105 over the whole dataset) are dropped in Silver, and the Silver manifest
  records how many rows, how much quantity and how much spend were dropped, so they are counted
  and never silent.

### Known defects in the sources (not in the pipeline)

Three of the thirteen public files are flagged in `config/sources.yaml` and refused unless
`--allow-known-issue` is given: `sample-2k-baskets` and `sample-50k-customers` (basket ids are
stored in scientific notation, so basket identity cannot be recovered) and `data-sample` (a
24 KB preview workbook, not weekly CSVs). The analysis uses the nine full parts; the 5K-customer
sample is for the opt-in integration tests.

## What this does not establish

- Reconciliation proves Silver equals Bronze and Gold equals Silver. It cannot prove Bronze equals
  what dunnhumby intended, because there is no external ground truth.
- Value-range checks catch impossible values, not plausible-looking wrong ones.
- The loyalty share is about 80%, so customer analysis describes loyalty-card baskets only.
- About 20% of lines have no customer code, by design of the dataset.
