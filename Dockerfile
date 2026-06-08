# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app
COPY draftproof-frontend/package*.json ./
RUN npm ci
COPY draftproof-frontend/ .
# Vite inlines import.meta.env.* at BUILD time, so the Turnstile site key must be
# present here (not just in the API runtime env). Pass it via --build-arg /
# Koyeb build-time env: VITE_TURNSTILE_SITE_KEY. Empty value disables the widget
# gracefully (it shows a "not configured" message instead of breaking).
ARG VITE_TURNSTILE_SITE_KEY=""
ENV VITE_TURNSTILE_SITE_KEY=$VITE_TURNSTILE_SITE_KEY
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
