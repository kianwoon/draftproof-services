# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app
COPY draftproof-frontend/package*.json ./
RUN npm ci
COPY draftproof-frontend/ .
RUN npm run build

# Stage 2: API + static frontend
FROM python:3.12-slim

WORKDIR /app

COPY draftproof-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY draftproof-api/ .

# Copy built frontend into static/ for production serving
COPY --from=frontend /app/dist ./static

RUN useradd --create-home appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000

# 2 workers: eco-medium has ~512MB RAM; 4 uvicorn processes exhaust it.
# Async uvicorn handles concurrent connections within each worker, so 2 is sufficient.
CMD ["gunicorn", "app.main:app", \
     "-w", "2", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "650", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-"]
