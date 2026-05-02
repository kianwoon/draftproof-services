FROM python:3.11-slim

# WeasyPrint system deps (Debian Trixie uses libgdk-pixbuf-2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev python3-pip python3-setuptools \
    python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY worker/requirements.txt /app/requirements.txt

# Install deps — use CPU-only PyTorch for smaller image
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# HF cache lives on the persistent volume mounted at /huggingface
ENV HF_HOME=/huggingface

# Copy both worker/ and poc/ so worker can import poc modules
COPY poc/ /app/poc/
COPY worker/ /app/worker/

WORKDIR /app/worker

# Entrypoint: ensure model is cached on volume before starting Celery
COPY worker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
