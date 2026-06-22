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

# MS Word add-in (plain static files) — served same-origin at /word-addin/* so the
# task pane can call /api/ext/* without CORS. See word-addin/README.md.
COPY word-addin/ ./static/word-addin/

RUN useradd --create-home appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "650", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-"]
