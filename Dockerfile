# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies required by some Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy only the files needed for installation first to leverage Docker layer caching.
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

# Install the chatterbox package and its dependencies.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Expose cache directories so that docker volumes can persist model downloads.
ENV HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "chatterbox.streaming_api:app", "--host", "0.0.0.0", "--port", "8000"]
