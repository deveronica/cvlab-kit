# CVLab-Kit Production Dockerfile (Multi-stage build)

# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /frontend

# Copy frontend package files
COPY web_helper/frontend/package*.json ./
RUN npm ci

# Copy frontend source
COPY web_helper/frontend/ ./

# Build frontend for production
RUN npm run build

# Stage 2: Python runtime
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy entire project (excluding node_modules, dist via .dockerignore)
COPY . .

# Copy built frontend from stage 1
COPY --from=frontend-builder /frontend/dist ./web_helper/frontend/dist

# Create necessary directories
RUN mkdir -p logs outputs web_helper/state web_helper/queue_logs

# Expose ports
# 8000: FastAPI backend (serves both API and frontend)
EXPOSE 8000

# Default command: run web helper in production mode
CMD ["uv", "run", "app.py"]
