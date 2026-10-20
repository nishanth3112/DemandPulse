# DemandPulse — Engineering Instructions

## Mission
Retail revenue and demand intelligence platform on the dunnhumby "Let's Get Sort-of-Real" synthetic dataset. Prioritize correctness, reproducibility, simplicity, and measurable business value. Deadline-driven: ship tested vertical slices over broad partial features.

## Engineering Principles
- Inspect existing code before changing it. Never guess schemas, APIs, metrics, or results; verify.
- Simplest working solution. No speculative abstractions or premature optimization.
- Minimal, focused changes. Fix root causes; never hide errors, skip, or weaken tests.
- Typed where useful, independently testable. No placeholder implementations presented as complete.

## Technical Constraints
- Python 3.12, uv, Polars, DuckDB, PyArrow, Parquet, LightGBM, Streamlit, Plotly.
- Target 16 GB RAM and ~50 GB free disk: stream/lazy only, never load the full dataset. Raw zips and CSVs are streamed over HTTP and never written to local disk.
- Bronze/Silver Parquet lives in a private S3 bucket (`DEMANDPULSE_S3_BUCKET`, AWS profile `demandpulse`, us-east-1). Keep within the AWS Free Tier credit; no other paid services.
- Raw data, secrets, credentials, models, and generated artifacts stay out of Git (`data/`, `artifacts/`, `models/`, `.env`). Never paste credentials into chat or files in the repo.
- Greykite is dependency-isolated from the main environment (it pins pandas 1.5.3 / numpy 1.26).
- Prevent temporal leakage. Use identical horizons, cutoffs, and eval sets when comparing models.

## Data Facts (from the official user guide)
- One denormalized weekly-file fact table; one row per product per basket. Columns: shop_week, shop_date, shop_weekday, shop_hour, quantity, spend, prod_code, prod_code_10/20/30/40, cust_code, seg_1, seg_2, basket_id, basket_size, basket_price_sensitivity, basket_type, basket_dominant_mission, store_code, store_format, store_region. Plus `time.csv` (shop_week, date_from, date_to).
- No product names, no unit price or cost/margin, no promotion flags. Revenue is `spend`. Do not claim margin, profit, or promo effects.
- Transactions without a loyalty card exist (null/blank cust_code); customer analytics cover loyalty baskets only.
- Fiscal weeks (YYYYWW); use `time.csv` for dates.
- Source zips are public on dunnhumby's CDN and support HTTP range requests. File format inside the zips is verified in Phase 1, not assumed.

## Git Workflow
- Never develop on `main`. One branch per phase, 5–10 meaningful Conventional Commits (feat/fix/test/docs/refactor/ci/chore). No empty or artificial commits.
- Before a PR: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`.
- Push, open PR to `main`, wait for CI green; on failure diagnose, fix, commit, push. Never disable or weaken checks.
- Never merge without explicit user approval. Never force-push `main` or rewrite published history.
- After merge: sync local `main` to `origin/main` and confirm GitHub shows MERGED before the next phase.
- If auth or permissions block an action, report the blocker.

## Phase Execution
- Before: confirm branch/status, confirm previous PR merged, plan, define commit boundaries and acceptance criteria.
- After: full local quality suite, PR, CI result, then the phase completion report (phase, branch, commits, implemented, tests, CI, PR, data coverage, limitations, next phase).

## Analytical Integrity
- Data is synthetic and historical; no claims of live operations or verified financial gains.
- Separate: observed measurements, historical backtest results, estimated scenarios, hypothetical impact. Document assumptions.
- Never report coverage, model results, or completed features that were not actually run. State the subset used for any expensive training.
