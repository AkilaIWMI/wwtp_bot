# ============================================
# Stage 1: Base Image (Debian Bookworm)
# ============================================
FROM python:3.11.9-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    # This tells YOLO/Ultralytics where it's allowed to write config files
    YOLO_CONFIG_DIR=/tmp/Ultralytics 

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: Python Dependencies
# ============================================
FROM base AS builder

WORKDIR /build
COPY requirements.txt .

# CRITICAL FIX: Install numpy to the BUILDER'S standard path first 
# so the GDAL compilation process can actually find it.
RUN pip install --no-cache-dir "numpy<2.0"

# Now install everything else to the /install directory
RUN pip install --no-cache-dir --prefix=/install "numpy<2.0" && \
    pip install --no-cache-dir --prefix=/install GDAL==$(gdal-config --version) && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================
# Stage 3: Final Runtime Image
# ============================================
FROM base AS runtime

# Copy the pre-compiled packages from the builder
COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 -s /bin/bash appuser
WORKDIR /app

# Ensure the appuser owns the necessary directories
COPY backend/ /app/backend/
RUN mkdir -p /app/backend/Data /tmp/Ultralytics && \
    chown -R appuser:appuser /app /tmp/Ultralytics

USER appuser
EXPOSE 5000

WORKDIR /app/backend
CMD ["uvicorn", "whatsapp_bot:app", "--host", "0.0.0.0", "--port", "5000"]
