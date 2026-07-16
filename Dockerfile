FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    DATASETS_CACHE=/app/.cache/huggingface/datasets

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY scripts ./scripts

RUN mkdir -p /app/data /app/.cache/huggingface

# Run as root in the container so bind-mounted ./data is writable on Docker Desktop (Windows).
CMD ["python", "-m", "src.phase1.explore_spider"]
