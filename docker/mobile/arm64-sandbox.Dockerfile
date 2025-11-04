# Minimal ARM64 sandbox image for Android-class devices or edge boxes
# Build with: docker buildx build --platform linux/arm64 -t omnicoder-sandbox:arm64 -f docker/mobile/arm64-sandbox.Dockerfile .

FROM --platform=$BUILDPLATFORM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy project sources (only what we need for sandbox server)
COPY src /app/src
COPY pyproject.toml README.md /app/

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir fastapi uvicorn jiwer z3-solver && \
    python -m pip install --no-cache-dir -e .

EXPOSE 8088

CMD ["python", "-m", "omnicoder.sfb.runtime.sandbox_server"]


