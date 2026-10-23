# Dashboard image. It carries the code and its locked dependencies; the data snapshot is mounted
# at /data/app (create it with `demandpulse app export` or `demandpulse demo`). It holds no
# credentials and never reads the basket fact.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so editing the source does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

FROM python:3.12-slim
# LightGBM needs the OpenMP runtime, which the slim image omits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
COPY --chown=app:app .streamlit ./.streamlit
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DEMANDPULSE_APP_DATA=/data/app
USER app
VOLUME ["/data/app"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "src/demandpulse/app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
