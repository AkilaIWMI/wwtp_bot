# ============================================
# Stage 1: Base Image (Debian Bookworm)
# ============================================
FROM python:3.11.9-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Install system dependencies (GDAL, etc.)
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
# Stage 2: Python Builder
# ============================================
FROM base AS builder

WORKDIR /build

# Create a virtual environment
RUN python -m venv /opt/venv
# Enable venv for subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install dependencies into the virtual environment
# 1. Install numpy first (needed for GDAL compilation)
RUN pip install --no-cache-dir "numpy<2.0"

# 2. Install GDAL matching system version
# We set C_INCLUDE_PATH to ensure pip finds the GDAL headers
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal

RUN pip install --no-cache-dir GDAL==$(gdal-config --version)

# 3. Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 3: Final Runtime Image
# ============================================
FROM base AS runtime

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Enable venv in the runtime
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd -m -u 1000 -s /bin/bash appuser
WORKDIR /app

# Copy application code
# NOTE: Ensure the 'backend' folder exists in your repo root!
COPY backend/ /app/backend/

# Create necessary dirs with correct permissions
RUN mkdir -p /app/backend/Data /tmp/Ultralytics && \
    chown -R appuser:appuser /app /tmp/Ultralytics

USER appuser
EXPOSE 5000

# Set working directory to where the app code is
WORKDIR /app/backend

# Using the venv python executable explicitly is safest
CMD ["uvicorn", "whatsapp_bot:app", "--host", "0.0.0.0", "--port", "5000"]
