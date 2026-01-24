# ============================================
# Stage 1: Base Image (Debian Bookworm)
# ============================================
FROM python:3.11.9-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Install ALL system dependencies (GDAL + Graphics for YOLO/OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: Python Builder
# ============================================
FROM base AS builder

WORKDIR /build

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# STEP 1: Install Wheel and Numpy first (Required for GDAL compilation)
RUN pip install --no-cache-dir wheel "numpy<2.0"

# STEP 2: Compile GDAL with NumPy support enabled
# We tell the compiler exactly where the GDAL headers are
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN pip install --no-cache-dir GDAL==$(gdal-config --version)

# STEP 3: Install the rest of the requirements
# We use --no-deps for numpy if it's in the file, or just let pip realize it's satisfied
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 3: Final Runtime Image
# ============================================
FROM base AS runtime

# Copy the virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd -m -u 1000 -s /bin/bash appuser
WORKDIR /app

# Copy application code
COPY backend/ /app/backend/

# Create necessary directories and set permissions
RUN mkdir -p /app/backend/Data /app/config /tmp/Ultralytics && \
    chown -R appuser:appuser /app /tmp/Ultralytics

USER appuser
EXPOSE 5000

# Healthcheck to ensure the container is actually responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

WORKDIR /app/backend
CMD ["uvicorn", "whatsapp_bot:app", "--host", "0.0.0.0", "--port", "5000"]
