# Data pipeline

How raw dunnhumby files become validated, analysis-ready Parquet. Everything here was checked
against the real files unless marked otherwise.

## Flow

```
dunnhumby CDN zips (HTTP, range requests)
  -> Bronze  typed Parquet, one file per source week, faithful to the CSV
  -> Silver  cleaned weekly Parquet (blank -> null, loyalty flag, week dates)
  -> quality report (row-level rules + Bronze/Silver reconciliation)
```

Zips are never written to disk. `fsspec` exposes each remote zip as a seekable file, so listing
members reads only the archive's directory and reading a member fetches only that week's bytes.
One weekly CSV is held in memory at a time. Measured on a full-dataset week (2.48M rows):
5.4 s to ingest and about 2.4 GB peak memory.

## Storage layout

Under the configured root (an S3 bucket, or a local directory with `--local-root`):

```
bronze/source=<name>/shop_week=<YYYYWW>/part-0.parquet
bronze/source=<name>/calendar.parquet
bronze/_manifests/<name>/transactions_<YYYYWW>.json
silver/source=<name>/shop_week=<YYYYWW>/part-0.parquet
silver/_manifests/<name>/<YYYYWW>.json
quality/<name>/report.json | report.md
```

Each source is kept separate because samples and full parts cover the same weeks with different
rows. Downstream phases read the `full-part-*` sources together.

## Running it

```bash
export AWS_PROFILE=demandpulse AWS_REGION=us-east-1 DEMANDPULSE_S3_BUCKET=<bucket>
uv run demandpulse sources                          # registry, with known issues flagged
uv run demandpulse run sample-5k-customers          # ingest + silver + quality
uv run demandpulse ingest full-part-3 --max-weeks 2
uv run demandpulse silver full-part-3 --weeks 200631,200632
uv run demandpulse quality full-part-3
uv run demandpulse --local-root /tmp/lake run sample-5k-customers   # no AWS
```

The command exits non-zero when any week fails or any quality check fails.

## Idempotency and restarts

- A week is done only when its manifest exists. The manifest is written after the Parquet file,
  so a crash mid-week just redoes that week.
- Bronze skips a week when the manifest's CRC equals the zip member's CRC (known from the zip
  directory, so skipped weeks are not downloaded). Changed source data is re-ingested.
- Silver rebuilds a week when its Bronze week was re-ingested.
- Transient network errors are retried three times with backoff. A week that still fails is
  recorded in the report and the run continues; nothing is hidden.

## Verified findings about the data

- Files are comma-delimited CSV with a header, one `transactions_YYYYWW.csv` per week plus
  `time.csv`.
- Headers are UPPERCASE except `seg_1` and `seg_2`; they are normalized to lowercase.
- Customers without a loyalty card have a blank `cust_code`, `seg_1` and `seg_2` (about 20% of
  rows). Silver turns blanks into nulls and adds `is_loyalty`.
- The mission value is `Nonfood` (the user guide says "Non Food").
- `XX` marks unclassified basket attributes and is kept as data.
- `time.csv` lists 142 weeks; transactions cover 117 contiguous weeks (200607 to 200819), all
  present in the calendar. The 5K sample is missing weeks 200608 and 200703 (no sampled
  customer shopped), so sample-based code must not assume contiguous weeks.
- The full dataset contains 105 exact duplicate rows (70 in week 200807, 35 in week 200808,
  all in part 9). Silver drops them and records their quantity and spend in the manifest.
- 9 further rows in part 9 repeat a (basket, product) pair with different values. They are
  kept and reported as a warning.
- 1,045 rows across the full dataset have zero spend. They are kept and reported as a warning.
- The 2K-basket and 50K-customer samples store `basket_id` in scientific notation
  (`9.94E+14`), which destroys basket identity. They are flagged in `config/sources.yaml` and
  refused unless `--allow-known-issue` is passed. The "Data Sample" is a preview workbook, not
  CSVs.
- Spend has no unit price, cost or promotion information. Revenue is `spend`.

## Quality checks

Row-level (fail unless noted): null key values, non-positive quantity, negative spend, weekday
and hour ranges, shop date inside its calendar week, one store/date/customer per basket, domain
values for basket size, price sensitivity, type, mission and store format. Warnings: zero spend,
repeated (basket, product) pairs, calendar gaps.

Reconciliation: Bronze files against their manifests, Silver row count against Bronze (less
counted exact duplicates), and Silver quantity and spend against Bronze. Spend is stored as
`Decimal(12, 2)` so sums compare exactly.

## Results on the 5K-customer sample (run 2026-10-21, S3)

115 weeks, 3,105,071 rows, 482,636 baskets, spend 5,933,712.07, loyalty row share 80.4%.
All reconciliation checks pass. Warnings: 6 zero-spend rows; calendar gaps at 200608 and
200703.

## Results on the full dataset (run 2026-10-21, S3, all nine parts)

| | |
|---|---|
| Weeks | 117 (200607 to 200819), contiguous |
| Silver rows | 307,531,968 (after dropping 105 exact duplicates) |
| Baskets | 47,098,503 |
| Spend | 589,726,919.69 |
| Loyalty row share | 80.7% |
| Failed weeks | 0 |
| Failed quality checks | 0 |
| Warnings | 1,045 zero-spend rows; 9 repeated (basket, product) pairs |

Every part passes all reconciliation checks: Bronze files match their manifests, and Silver
rows, quantity and spend equal Bronze after the counted duplicates. Ingest plus Silver plus
quality took about 38 minutes end to end from a laptop. Stored size in the bucket is 7.07 GB
(Bronze 3.50 GB, Silver 3.56 GB, including the 5K sample), roughly $0.16 a month at standard
S3 storage rates.

Counts above are Silver counts, so they exclude the 105 duplicate rows. These are observed
dataset measurements, not business conclusions.

## Limitations

- Only the 5K-customer sample and the nine full parts are loaded. The 2K and 50K samples are
  defective and the preview is a spreadsheet.
- Gold marts, dimensional modelling and forecasting are later phases.
- The CDN links come from the public source-files page and are not guaranteed to be stable.
