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

# Install system dependencies including Node.js for frontend builds
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy entire project (excluding items in .dockerignore)
COPY . .

# Copy built frontend from stage 1
COPY --from=frontend-builder /frontend/dist ./web_helper/frontend/dist

# Copy and setup entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create necessary directories
RUN mkdir -p logs outputs config web_helper/queue_logs

# Configure git for updates (safe directory)
RUN git config --global --add safe.directory /app

# Expose ports
# 8000: FastAPI backend (serves both API and frontend)
EXPOSE 8000

# Entrypoint handles auto-update
ENTRYPOINT ["/entrypoint.sh"]

# Default command: run web helper in production mode
CMD ["uv", "run", "app.py"]
