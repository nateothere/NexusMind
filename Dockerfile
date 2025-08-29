# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml requirements*.txt ./
RUN uv pip install --system --no-cache-dir -r requirements.txt

FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXUSMIND_DATA_DIR=/data
RUN useradd --create-home --uid 1000 nexusmind
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --chown=nexusmind:nexusmind src ./src
COPY --chown=nexusmind:nexusmind pyproject.toml config.example.yaml VERSION ./
RUN pip install --no-cache-dir --no-deps . && mkdir -p /data && chown nexusmind:nexusmind /data
USER nexusmind
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1
CMD ["uvicorn", "nexusmind.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]