#!/bin/bash
echo "🚀 Starting CVLab-Kit..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
fi

echo "📦 Building and starting containers..."
docker-compose -f docker/docker-compose.yml up -d --build

echo "✅ CVLab-Kit is running!"
echo "   Backend/Frontend: http://localhost:8000"
echo "   Logs: docker-compose -f docker/docker-compose.yml logs -f"
