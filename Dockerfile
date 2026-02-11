# Stage 1: Builder - Install dependencies
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies needed for compiling packages on ARM (e.g., for Raspberry Pi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt .

# Build wheels for faster installation in the next stage
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Final Image - Copy app and dependencies
FROM python:3.9-slim
WORKDIR /app

# Optimization: Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Optimization: Use bind mount to install wheels directly from builder without copying them (saves space)
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*

COPY . .
EXPOSE 5000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]