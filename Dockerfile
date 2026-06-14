# Task 6 — Inference image.
# Slim base keeps the image small; we install ONLY inference dependencies.
FROM python:3.11-slim

# HF model name is a build argument with a sensible default (Task 6.2).
# Override at build time:  docker build --build-arg HF_MODEL_NAME=<user>/<model> ...
ARG HF_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Avoid .pyc files and force unbuffered logs (cleaner Actions output).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install only what inference needs — NOT the full training stack.
# (datasets, wandb, scikit-learn are training-only and intentionally excluded.)
RUN pip install --no-cache-dir \
        torch==2.4.1 \
        "transformers==4.44.2" \
        "huggingface-hub==0.25.2"

# Copy just the inference code.
COPY src/inference.py /app/src/inference.py

# Default input can be overridden with -e INPUT_TEXT='...'
ENV INPUT_TEXT="This product exceeded my expectations!"

CMD ["python", "src/inference.py"]
