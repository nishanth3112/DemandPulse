# DemandPulse — Engineering Instructions

## Mission
Build a reliable retail analytics and forecasting platform using dunnhumby's synthetic transaction dataset. Prioritize correctness, reproducibility, simplicity, and measurable business value.

## Engineering Principles
- Think before coding. Inspect existing code and understand dependencies.
- Prefer the simplest working solution. Avoid speculative abstractions, unnecessary frameworks, and premature optimization.
- Make minimal, focused changes. Reuse existing patterns.
- Never guess schemas, APIs, metrics, or results. Verify assumptions.
- Fix root causes, not symptoms. Never hide errors or weaken tests.
- Keep code readable, typed where useful, and independently testable.
- Remove unnecessary complexity. No placeholder implementations presented as complete.

## Technical Constraints
- Python 3.12, uv, Polars, DuckDB, Parquet, LightGBM, Streamlit.
- Use streaming or lazy processing; target 16 GB RAM.
- Keep raw data, secrets, models, and generated artifacts outside Git.
- Prevent temporal leakage in forecasting and classification.
- Distinguish observed metrics, backtest results, and hypothetical impact.

## Git Workflow
- Never implement features directly on `main`.
- Use one branch per phase with 5–10 meaningful Conventional Commits.
- Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest` before PRs.
- Push, open a PR, and verify CI.
- Never merge without explicit user approval.

## Execution
- Plan before each phase; define scope, acceptance criteria, and commit boundaries.
- Implement incrementally; test after meaningful changes.
- Report completed work, actual tests, limitations, and PR status.
- Ask when requirements are genuinely ambiguous.
- Never fabricate success or claim unexecuted work.
- Prioritize working end-to-end functionality over feature breadth.
