# Stage 1: Builder - Install dependencies
FROM python:3.9-slim as builder

WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt .

# Build wheels for faster installation in the next stage
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Final Image - Copy app and dependencies
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]