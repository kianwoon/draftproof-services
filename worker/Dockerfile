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

# Pre-download GPT2-medium (~1.5GB) into a fixed path baked into the image
ENV HF_HOME=/app/hf_cache
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('gpt2-medium'); \
    AutoModelForCausalLM.from_pretrained('gpt2-medium'); \
    print('GPT2-medium cached')" && \
    echo "Cache contents:" && ls -la ${HF_HOME}/hub/

# Copy both worker/ and poc/ so worker can import poc modules
COPY poc/ /app/poc/
COPY worker/ /app/worker/

WORKDIR /app/worker

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--concurrency=1", "-Q", "scan,default"]
