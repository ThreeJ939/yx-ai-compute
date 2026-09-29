# CUDA runtime base; override for CPU-only: python:3.11-slim
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY pyproject.toml README.md ./

# Mount weights at runtime: -v /path/to/weights:/app/weights
RUN mkdir -p /app/weights /app/tmp

EXPOSE 18100

CMD ["python", "-m", "app.main"]
