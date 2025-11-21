# Docker Deployment Guide

Complete guide for deploying CVLab-Kit using Docker containers.

---

## 📋 Overview

CVLab-Kit provides **Docker deployment for the web_helper server** (Backend + Frontend). The middleend (client agent) is deployed separately via git clone.

### What's Included
- ✅ **FastAPI Backend**: API server and experiment queue manager
- ✅ **React Frontend**: Web UI for experiment management and visualization
- ✅ **Multi-stage Build**: Optimized image with frontend pre-built
- ✅ **Volume Mounts**: Persistent storage for logs, state, and configurations

### What's Not Included
- ❌ **Middleend (Client Agent)**: Deploy separately on GPU worker machines
- ❌ **CVLabKit Core**: Experiment execution engine (runs independently)

---

## 🚀 Quick Start

### Prerequisites
- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: V2 (included with Docker Desktop)
- **Git**: For cloning the repository

### 1. Clone Repository
```bash
git clone https://github.com/deveronica/cvlab-kit.git
cd cvlab-kit
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your settings (optional)
nano .env
```

**Environment Variables** (all optional):
```bash
WANDB_API_KEY=your_wandb_key_here  # For W&B logging
CUDA_VISIBLE_DEVICES=0,1           # GPU devices to use
```

### 3. Build and Run
```bash
# Build Docker image
docker compose -f docker/docker-compose.yml build

# Start services (detached)
docker compose -f docker/docker-compose.yml up -d
```

### 4. Verify Deployment
```bash
# Check logs
docker compose -f docker/docker-compose.yml logs -f

# Test API
curl http://localhost:8000/api/projects

# Access Web UI
# Open browser: http://localhost:8000
```

---

## 🐳 Docker Configuration

### Dockerfile Structure

**Multi-stage build** for optimized image size:

**Stage 1: Frontend Builder**
```dockerfile
FROM node:18-alpine AS frontend-builder
# Build React app (npm run build)
```

**Stage 2: Python Runtime**
```dockerfile
FROM python:3.10-slim
# Install uv, copy Python deps, copy built frontend
```

**Key Features**:
- Layer caching for dependencies
- Minimal production image
- Frontend pre-built and served by FastAPI

### docker-compose.yml

**Service Configuration**:
```yaml
services:
  cvlab-kit:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"    # Web server
    volumes:
      - ../logs:/app/logs                    # Experiment logs
      - ../outputs:/app/outputs              # Model outputs
      - ../config:/app/config                # Experiment configs
      - ../web_helper/state:/app/web_helper/state          # SQLite DB
      - ../web_helper/queue_logs:/app/web_helper/queue_logs  # Queue metadata
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY:-}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    restart: unless-stopped
```

### .dockerignore

**Excluded from build context**:
- `node_modules/`, `.venv/`, `__pycache__/`
- `logs/`, `outputs/` (mounted as volumes)
- `.git/`, `.github/`
- Tests and documentation builds

---

## 📁 Volume Mounts

### Persistent Data

| Volume | Purpose | Required |
|--------|---------|----------|
| `logs/` | Experiment run logs (CSV, YAML) | ✅ Yes |
| `outputs/` | Model checkpoints, artifacts | Recommended |
| `config/` | Experiment configurations | Recommended |
| `web_helper/state/` | SQLite database | ✅ Yes |
| `web_helper/queue_logs/` | Experiment execution metadata | ✅ Yes |

### Hot-Reload (Development)

**Optional**: Mount source code for live updates
```yaml
volumes:
  - ../cvlabkit:/app/cvlabkit        # Python backend
  - ../web_helper:/app/web_helper    # Web helper
```

**Note**: Requires rebuilding frontend manually (`npm run build`)

---

## 🔧 Configuration

### Production Mode (Default)

**Serves pre-built frontend from `dist/`**:
```bash
docker compose -f docker/docker-compose.yml up -d
```

**Characteristics**:
- ✅ Optimized bundle
- ✅ Fast startup
- ❌ No hot reload

### Development Mode

**Optional**: Enable frontend dev server (port 5173)

1. Modify `docker-compose.yml`:
```yaml
command: uv run app.py --dev
ports:
  - "8000:8000"
  - "5173:5173"  # Vite dev server
```

2. Rebuild and restart:
```bash
docker compose -f docker/docker-compose.yml up -d --build
```

**Access**:
- API: `http://localhost:8000`
- Frontend (HMR): `http://localhost:5173`

---

## 🛠️ Common Commands

### Container Management
```bash
# Start services
docker compose -f docker/docker-compose.yml up -d

# Stop services
docker compose -f docker/docker-compose.yml stop

# Stop and remove containers
docker compose -f docker/docker-compose.yml down

# Restart services
docker compose -f docker/docker-compose.yml restart

# View status
docker compose -f docker/docker-compose.yml ps
```

### Logs and Debugging
```bash
# Stream logs (follow)
docker compose -f docker/docker-compose.yml logs -f

# View last 100 lines
docker compose -f docker/docker-compose.yml logs --tail=100

# Logs for specific service
docker compose -f docker/docker-compose.yml logs cvlab-kit

# Execute shell inside container
docker compose -f docker/docker-compose.yml exec cvlab-kit bash
```

### Rebuild and Update
```bash
# Rebuild image (no cache)
docker compose -f docker/docker-compose.yml build --no-cache

# Pull latest code and rebuild
git pull
docker compose -f docker/docker-compose.yml up -d --build
```

### Data Management
```bash
# Backup database
docker compose -f docker/docker-compose.yml exec cvlab-kit \
  cp /app/web_helper/state/db.sqlite /app/logs/db_backup_$(date +%Y%m%d).sqlite

# Restore database
docker compose -f docker/docker-compose.yml exec cvlab-kit \
  cp /app/logs/db_backup_20251120.sqlite /app/web_helper/state/db.sqlite
```

---

## 🔍 Health Checks

### Manual Verification

**1. API Health**:
```bash
curl http://localhost:8000/api/projects
# Expected: {"projects": [...]}
```

**2. Frontend Serving**:
```bash
curl -I http://localhost:8000
# Expected: HTTP/1.1 200 OK
```

**3. Database Connectivity**:
```bash
docker compose -f docker/docker-compose.yml exec cvlab-kit \
  ls -la web_helper/state/db.sqlite
# Expected: File exists with size > 0
```

### Automated Health Check

**Add to `docker-compose.yml`**:
```yaml
services:
  cvlab-kit:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/projects"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## 🚨 Troubleshooting

### Issue: Port Already in Use

**Error**: `bind: address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port in docker-compose.yml
ports:
  - "8080:8000"
```

### Issue: Permission Denied (Volumes)

**Error**: `Permission denied: '/app/logs'`

**Solution**:
```bash
# Set correct permissions
sudo chown -R $USER:$USER logs outputs web_helper/state

# Or run container with user
docker compose -f docker/docker-compose.yml run --user $(id -u):$(id -g) cvlab-kit
```

### Issue: Frontend Not Loading

**Error**: Blank page or 404 errors

**Solutions**:
1. **Check frontend build**:
```bash
docker compose -f docker/docker-compose.yml exec cvlab-kit \
  ls -la web_helper/frontend/dist
```

2. **Rebuild container**:
```bash
docker compose -f docker/docker-compose.yml build --no-cache
docker compose -f docker/docker-compose.yml up -d
```

### Issue: Database Locked

**Error**: `database is locked`

**Solution**:
```bash
# Stop all containers
docker compose -f docker/docker-compose.yml down

# Remove lock file
rm web_helper/state/db.sqlite-wal
rm web_helper/state/db.sqlite-shm

# Restart
docker compose -f docker/docker-compose.yml up -d
```

### Issue: Out of Disk Space

**Error**: `no space left on device`

**Solution**:
```bash
# Clean unused images
docker image prune -a

# Clean volumes (WARNING: deletes data)
docker volume prune

# Check disk usage
docker system df
```

---

## 🌐 Production Deployment

### Reverse Proxy (Nginx)

**Recommended**: Use Nginx for SSL/TLS termination

**nginx.conf**:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL/TLS with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Resource Limits

**Add to docker-compose.yml**:
```yaml
services:
  cvlab-kit:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### Logging Configuration

**JSON logs for production**:
```yaml
services:
  cvlab-kit:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## 🔄 Distributed Setup

### Architecture

```
┌─────────────────┐
│  Central Server │  ← Docker: web_helper (port 8000)
└─────────────────┘
        ↑
        │ HTTP API
        │
   ┌────┴────┐
   │         │
┌──▼──┐  ┌──▼──┐
│GPU-1│  │GPU-2│  ← Middleend clients (manual deployment)
└─────┘  └─────┘
```

### Central Server (Docker)

```bash
# Server mode: web_helper only
docker compose -f docker/docker-compose.yml up -d
```

### GPU Workers (Manual Deployment)

**On each GPU machine**:
```bash
# Clone repository
git clone https://github.com/deveronica/cvlab-kit.git
cd cvlab-kit

# Install dependencies
uv sync

# Run client agent
uv run app.py --client-only --url http://server-ip:8000 --daemon
```

**See**: [Distributed Execution Guide](./distributed_execution_guide.md) for details

---

## 📊 Monitoring

### Log Aggregation

**View all logs**:
```bash
docker compose -f docker/docker-compose.yml logs -f | tee logs/docker_$(date +%Y%m%d).log
```

### Disk Usage Monitoring

```bash
# Check volume sizes
du -sh logs outputs web_helper/state web_helper/queue_logs

# Set up cron for cleanup (optional)
0 0 * * 0 find logs -name "*.csv" -mtime +30 -delete
```

### Container Stats

```bash
# Real-time resource usage
docker stats

# One-time check
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml top
```

---

## 🔐 Security Considerations

### Production Checklist
- [ ] Change default ports (if exposed to internet)
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS with valid certificates
- [ ] Set up firewall rules (allow only necessary ports)
- [ ] Regular security updates (`docker pull` base images)
- [ ] Limit volume mount permissions
- [ ] Use Docker secrets for sensitive data

### Example: Docker Secrets

```yaml
services:
  cvlab-kit:
    secrets:
      - wandb_key
    environment:
      - WANDB_API_KEY_FILE=/run/secrets/wandb_key

secrets:
  wandb_key:
    file: ./secrets/wandb_key.txt
```

---

## 📚 Additional Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [CVLab-Kit Distributed Execution](./distributed_execution_guide.md)
- [Troubleshooting Guide](./troubleshooting.md)

---

## 🆘 Support

**Issues**: [GitHub Issues](https://github.com/deveronica/cvlab-kit/issues)
**Documentation**: [Full Docs](https://deveronica.github.io/cvlab-kit)

---

*Last Updated: 2025-11-20*
