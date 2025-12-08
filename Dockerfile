# Stage 1: Builder
FROM python:3.13-slim AS builder

LABEL authors="Michael Chadolias" \
    description="Docker image for machine learning training and deployment." \
    maintainer="@mchadolias"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ python3-dev

# Force CPU-only packages
ENV XGBOOST_BUILD_CPU=1
ENV XGBOOST_BUILD_GPU=0

COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY src/ ./src/

RUN uv sync --locked --no-dev

# Stage 2: Runtime
FROM python:3.13-slim
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY deployment/predict.py ./
COPY deployment/examples/ ./examples/
COPY outputs/models/ ./models/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 9696
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]