# =============================================================================
# Stage 1 — builder
# Install Python packages into an isolated prefix so only the compiled
# artifacts are copied to the final image, keeping it free of build tools.
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Build-time system dependencies.
# gcc / g++ are required to compile C extensions (e.g. rank_bm25, some
# unstructured sub-packages).  poppler-utils / tesseract-ocr / libmagic1
# are needed by unstructured at import time even during the pip install
# phase for certain extras.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into /install so the directory can be COPY --from'd cleanly.
# --no-cache-dir keeps the layer small; --prefix pins the destination.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# =============================================================================
# Stage 2 — production
# Lean runtime image: no compilers, no pip cache, no build headers.
# =============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Make the /install site-packages visible to the Python interpreter.
    PYTHONPATH=/install/lib/python3.12/site-packages \
    # Make console scripts (uvicorn, etc.) installed by pip findable.
    PATH="/install/bin:${PATH}"

WORKDIR /app

# Runtime-only system dependencies — mirrors the builder set minus gcc/g++.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Pull compiled packages from the builder stage.
COPY --from=builder /install /install

# Application source.
COPY app/ ./app/

# Entrypoint script.
# Run `chmod +x entrypoint.sh` on the host before building if the file
# system does not preserve the executable bit (e.g. on Windows hosts).
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
