# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app
COPY draftproof-frontend/package*.json ./
RUN npm ci
COPY draftproof-frontend/ .
RUN npm run build

# Stage 2: Build backend with frontend static files
FROM python:3.12-slim
WORKDIR /app

COPY draftproof-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY draftproof-api/ .

# Copy built frontend into static dir
COPY --from=frontend /app/dist /app/static

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
