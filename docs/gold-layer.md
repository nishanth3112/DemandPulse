# Gold layer: analytics warehouse

DuckDB SQL over the Silver Parquet produces the dimensions and marts below, exported to
`<lake>/gold/` (S3 or a local directory). Everything is deterministic: rebuilding from the same
Silver gives the same tables, and 24 validation checks run after every build.

```bash
uv run demandpulse gold build                 # build all marts, then validate
uv run demandpulse gold build --resume        # reuse tables already in the scratch DB
uv run demandpulse gold build --only store_scorecard,customer_marts --resume
uv run demandpulse gold validate              # re-run the checks on published marts
```

Full dataset, from scratch against S3: 11.6 minutes and 3.8 GB peak memory on a laptop.

## Assumptions and labels

- **Product hierarchy.** dunnhumby only provides levels 10/20/30/40. They nest strictly in the
  data (verified), with 245, 89, 31 and 9 distinct values. Labels used here are mine: level 40
  = division, **30 = category** (the analysis grain), 20 = sub-category, 10 = class. Codes have
  no names, so categories appear as `G000xx`.
- **Windows.** `week_idx` counts weeks from the first data week using dates. The full dataset
  has 117 contiguous weeks (200607 to 200819). Three comparisons are built:

  | comparison | current | base |
  |---|---|---|
  | `window_vs_prior` | last 13 weeks (200807 to 200819) | the 13 weeks before (200746 to 200806) |
  | `window_vs_year_ago` | last 13 weeks | same 13 weeks a year earlier (200707 to 200719) |
  | `year_vs_prior_year` | last 52 weeks (200720 to 200819) | the 52 weeks before (200620 to 200719) |

  `window_vs_prior` includes seasonality; `window_vs_year_ago` does not. Window sizes are
  parameters (`--window-weeks`, `--year-weeks`).
- **Traffic is baskets**, i.e. transactions at the till, not footfall.
- **No margin, profit or promotion analysis is possible**: the data has only spend and
  quantity. `price_mix` below is average revenue per unit, which mixes price changes with a
  change in what is bought.
- All results below are observed measurements on synthetic, historical data. They are not
  claims about a real retailer.

## Tables

| table | grain | rows (full data) |
|---|---|---|
| `fct_basket` | basket | 47,098,503 |
| `fct_store_week` | store, week | 88,555 |
| `fct_store_category_week` | store, category, week | 2,402,431 |
| `dim_date` | week | 117 |
| `dim_comparison`, `dim_comparison_week` | comparison, (period, week) | 3, 156 |
| `dim_store` | store | 761 |
| `dim_product` | product | 4,997 |
| `dim_customer` | loyalty customer | 494,880 |
| `mart_basket_mix` | week, format, basket type/size/mission/sensitivity, loyalty | 70,813 |
| `mart_trading_pattern` | format, weekday, hour | 414 |
| `mart_category_benchmark` | comparison, category | 93 |
| `mart_region_benchmark` | comparison, region (+ `ALL`) | 39 |
| `mart_region_category_mix` | comparison, region, category | 1,116 |
| `mart_store_scorecard` | comparison, store | 2,283 |
| `mart_revenue_variance_store` / `_region` | comparison, store / region | 2,283 / 39 |
| `mart_customer_rfm`, `mart_rfm_segment_summary` | customer / segment | 433,571 / 7 |
| `mart_customer_cohort` | cohort period, period | 459 |

Silver is scanned three times, in the steps that take almost all of the 698 s: `fct_basket`
(318 s including its export), the dimensions (126 s, mostly `dim_product` and `dim_date`) and
the category fact (122 s). Store-week, scorecard, variance and customer marts derive from the
basket fact and take under 15 s together. The category fact is built one week at a time
because counting distinct baskets across all weeks at once exceeded the 8 GB limit.

## KPI definitions

| KPI | definition |
|---|---|
| revenue | sum of `spend` |
| units | sum of `quantity` |
| lines | product lines (rows) |
| baskets | distinct `basket_id` |
| customers | distinct loyalty `cust_code` (no-card baskets have none) |
| avg basket value | revenue / baskets |
| units per basket | units / baskets |
| avg unit price | revenue / units; an average over a mixed product set, not a list price |
| loyalty basket share | baskets with a customer code / baskets |
| penetration | baskets containing the category / all baskets in the window |
| revenue per week | revenue / weeks traded (so part-window stores are comparable) |

## Methods

- **Like-for-like (LFL).** A store is LFL for a comparison when it traded in every week of both
  windows. LFL growth removes openings and closures. Region `ALL` is the national total.
- **Peer benchmarks.** Revenue-per-week and basket-value indices are relative to the median of
  the store's format (LS, MS, SS, XLS). Ranks are within region, only for stores trading in the
  current window. Growth gaps (percentage points) and z-scores use LFL stores only.
- **Revenue variance (LMDI).** Per store,
  `revenue = trading weeks x baskets/week x units/basket x revenue/unit`. With
  `L = (R1 - R0) / ln(R1 / R0)`, each effect is `L x ln(factor1 / factor0)`. The four effects
  sum exactly to the revenue change, with no residual and no dependence on factor order. Stores
  with no base revenue contribute their revenue as `new_store_effect`; stores with no current
  revenue contribute minus their base revenue as `closed_store_effect`. Regional and national
  rows are sums of store rows.
- **RFM.** Loyalty customers active in the last 52 weeks. Recency is weeks since the last basket
  (0 = shopped in the final week); frequency is baskets and monetary is revenue in the window.
  Scores 1 to 5 (5 best) come from `ceil(5 x cume_dist)`, so tied customers share a score.
  Segments are rule-based on the scores (see `sql/50_customer_marts.sql`).
- **Cohorts.** A customer's cohort is the first 4-week period in which they appear. **Cohort 1
  is a baseline, not an acquisition cohort**: those customers were already shopping before the
  data starts. The final period is one week long, so its activity is understated
  (`weeks_in_period`).

## Validation

24 SQL checks, each returning a violation count, run over the published Parquet. They cover
reconciliation of every mart to the Silver totals recorded in the build manifest (rows and
spend, exact to 0.5 cent), grain uniqueness, constant basket attributes (so `min()` in the
basket fact is exact), loyalty flag consistency, shares summing to one, variance effects
summing to the revenue change, national bridge closing, RFM ranges and cohort retention
bounds. A check that cannot run counts as a failure. The same checks run in CI against a
synthetic fixture, and tests tamper with marts to confirm each class of error is caught.

An early full-data run failed 7 of 24 checks. The cause was stale partition files left in S3
by an earlier out-of-memory run, which doubled part of the basket fact. Exports now clear a
partitioned table's prefix before writing, with a regression test.

## Observed results (full dataset, 117 weeks)

Totals: 47,098,503 baskets, revenue 589,726,919.69, 463,408,493 units, average basket value
12.52, 9.84 units per basket, 79.5% of baskets with a loyalty card (80.4% of revenue).

**Network comparisons** (region `ALL`):

| comparison | revenue growth | LFL stores | LFL growth | basket growth |
|---|---|---|---|---|
| `window_vs_prior` | +5.84% | 758 | +5.47% | +2.15% |
| `window_vs_year_ago` | -1.12% | 756 | -1.22% | -0.24% |
| `year_vs_prior_year` | +0.64% | 743 | -0.29% | +1.30% |

Over the last 52 weeks revenue is 263.3M against 261.6M a year earlier. The gain comes from
store changes and mix, not from like-for-like trading: LFL revenue is slightly down.

**Variance bridge, `year_vs_prior_year`** (base 261,628,512.51 to current 263,301,951.53):
trading weeks +1,331,512, traffic +2,307,694, basket size +1,796,072, price/mix -3,761,838,
new or closed stores 0. Higher traffic and bigger baskets were offset by a lower average
revenue per unit.

**Regions** (`year_vs_prior_year`, LFL growth): W02 +1.14% is highest and W01 -1.77% is lowest.
E01 and E02 grew total revenue (+2.5%, +2.7%) while their LFL growth was negative, so the
growth came from stores that are not like-for-like (new or part-window stores).

**Categories** (`year_vs_prior_year`): G00016 contributed +0.35 points of national growth, then
G00028 +0.32 and G00015 +0.28. G00006 subtracted 0.32 points (revenue -8.6%). G00028's average
unit price rose 12.0%.

**Stores:** 761 stores, 748 open throughout and 13 opened in the period. Median LFL growth
is -0.6%, with a 10th to 90th percentile range of about -6.0% to +7.6%. Median weekly revenue by
format: XLS 15,692, LS 9,282, MS 4,763, SS 1,773. Some stores show abrupt level shifts, for
example STORE00018 (SS, E01) went from roughly 100 baskets a week to about 550 at week 200719
and fell to about 300 by 200815, giving +545% LFL growth. Growth z-scores are sensitive to such
stores; growth gaps against the peer median are more robust.

**Customers** (last 52 weeks, 433,571 active loyalty customers):

| segment | customers | share of customers | share of revenue |
|---|---|---|---|
| Champions | 107,028 | 24.7% | 60.7% |
| Loyal | 59,723 | 13.8% | 18.6% |
| Needs attention | 84,139 | 19.4% | 9.3% |
| Potential loyalists | 46,558 | 10.7% | 5.6% |
| Hibernating | 124,805 | 28.8% | 3.3% |
| At risk | 8,150 | 1.9% | 2.5% |
| New or returning | 3,168 | 0.7% | 0.03% |

**Cohorts:** the baseline cohort (257,880 customers) retains 87.4% one period (4 weeks) later,
81.8% after 6 periods and about 72% after 24. Later cohorts are far smaller (cohort 6: 11,446
customers) and retain less: 42.3% after one period and 30.1% after 12.

**Baskets and trading:** Top Up baskets carry 44.7% of revenue (average value 16.89), Full Shop
36.1% (40.39) and Small Shop 18.8% (4.32). Loyalty baskets are 80.4% of revenue. The busiest
trading hours are 13:00 to 15:00.

## Limitations

- Categories and stores have codes only, so findings cannot be explained by product or place.
- Growth and variance describe what happened in the data. They do not explain causes such as
  pricing or promotions, which are not in the dataset.
- LFL requires every week traded in both windows, so a store with a closure week drops out.
- Forecasting, revenue targets and risk ranking are later phases and are not in this layer.
