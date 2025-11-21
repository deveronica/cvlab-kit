# Synology NAS Deployment Guide

Complete guide for deploying CVLab-Kit on Synology NAS using Container Manager.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  Synology NAS (Central Server)         │
│  ┌───────────────────────────────────┐ │
│  │  Container Manager                │ │
│  │  ├─ cvlab-kit (port 8000)         │ │
│  │  │  ├─ Web UI                     │ │
│  │  │  ├─ REST API                   │ │
│  │  │  ├─ Queue Manager              │ │
│  │  │  └─ SQLite Database            │ │
│  │  └─ Volumes                        │ │
│  │     ├─ /volume1/docker/cvlab-kit/ │ │
│  │     │  ├─ logs/                   │ │
│  │     │  ├─ outputs/                │ │
│  │     │  ├─ config/                 │ │
│  │     │  ├─ state/ (DB)             │ │
│  │     │  └─ queue_logs/             │ │
│  └───────────────────────────────────┘ │
└────────────┬────────────────────────────┘
             │ HTTP API
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ GPU     │ │ GPU     │ │ GPU     │
│ Worker  │ │ Worker  │ │ Worker  │
│ (RTX40) │ │ (RTX30) │ │ (A100)  │
└─────────┘ └─────────┘ └─────────┘
```

**Synology NAS**: Central management server (24/7, low power)
**GPU Workers**: External workstations for training (on-demand)

---

## Part 1: Synology NAS Setup (Server)

### Prerequisites

- Synology NAS with DSM 7.2+
- Container Manager package installed
- SSH access enabled (for git clone)
- At least 4GB available RAM
- 10GB available storage

### Step 1: Create Folder Structure

Open **File Station** and create:

```
/docker/cvlab-kit/
├── logs/            (experiment results)
├── outputs/         (model checkpoints)
├── config/          (YAML configurations)
├── state/           (SQLite database)
├── queue_logs/      (execution logs)
└── source/          (git repository - create this in Step 2)
```

### Step 2: Upload Source Code

#### Option A: Via SSH (Recommended)

1. Enable SSH in **Control Panel** → **Terminal & SNMP**

2. SSH into your NAS:
   ```bash
   ssh admin@<nas-ip>
   ```

3. Clone repository:
   ```bash
   cd /volume1/docker/cvlab-kit
   git clone https://github.com/yourusername/cvlab-kit.git source
   ```

#### Option B: Via File Station

1. Download repository as ZIP from GitHub
2. Upload to `/docker/cvlab-kit/`
3. Extract as `source` folder

### Step 3: Prepare Configuration

1. Navigate to `/volume1/docker/cvlab-kit/source/docker/`

2. Copy `docker-compose.synology.yml` content

3. Edit environment variables:
   ```yaml
   environment:
     - WANDB_API_KEY=your-key-here  # Optional
     - PYTHONUNBUFFERED=1
     - SERVER_MODE=true
   ```

### Step 4: Deploy in Container Manager

1. Open **Container Manager** → **Project** tab

2. Click **Create**

3. Fill in:
   - **Project Name**: `cvlab-kit`
   - **Path**: `/docker/cvlab-kit`
   - **Source**: Upload `docker-compose.synology.yml`

4. Click **Build**

   ⏱️ **First build takes 10-20 minutes**:
   - Frontend build (npm install + build)
   - Python dependencies (uv sync)
   - Docker layer caching for faster rebuilds

5. Once built, click **Run**

### Step 5: Verify Installation

1. **Check Container Status**:
   - Container Manager → Container tab
   - Status should be "Running" with green indicator

2. **Access Web UI**:
   ```
   http://<nas-ip>:8000
   ```

3. **Check Logs**:
   - Container Manager → Container → cvlab-kit → Logs
   - Should see: `Server started on 0.0.0.0:8000`

---

## Part 2: GPU Worker Setup (External Machines)

GPU workers execute training jobs submitted from the Synology server.

### Prerequisites (Each GPU Machine)

- Python 3.10 or higher
- CUDA-capable GPU with drivers installed
- Network access to Synology NAS
- Git installed

### Step 1: Clone Repository

```bash
# On GPU workstation
git clone https://github.com/yourusername/cvlab-kit.git
cd cvlab-kit
```

### Step 2: Install Dependencies

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### Step 3: Run Worker

```bash
# Connect to Synology server
uv run app.py --client-only --url http://<nas-ip>:8000

# Expected output:
# 💓 Starting full device agent...
#    Server URL: http://192.168.1.100:8000
#    Host ID: gpu-workstation-1
#    Heartbeat interval: 3s
#    Poll interval: 5s
#    Workspace: /home/user/cvlab-kit
```

### Step 4: Run as Daemon (Optional)

To keep worker running after closing SSH:

```bash
# Start daemon
uv run app.py --client-only --url http://<nas-ip>:8000 --daemon

# Check status
uv run app.py --status

# Stop daemon
uv run app.py --stop
```

### Step 5: Verify Connection

1. Open Synology Web UI: `http://<nas-ip>:8000`

2. Navigate to **Devices** tab

3. You should see your worker with:
   - ✅ Green status indicator
   - GPU information (model, memory)
   - System stats (CPU, RAM)

---

## Usage Workflow

### 1. Submit Experiment

**Via Web UI**:
1. Navigate to **Execute** tab
2. Click **Add Experiment**
3. Select config file from `/docker/cvlab-kit/config/`
4. Click **Submit**

**Via CLI** (from any machine with network access):
```bash
curl -X POST http://<nas-ip>:8000/api/queue/add \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "config/your_experiment.yaml",
    "priority": "normal"
  }'
```

### 2. Monitor Execution

**Queue Tab**: View pending/running jobs
**Devices Tab**: Monitor GPU utilization
**Logs Tab**: Real-time execution logs

### 3. View Results

**Projects Tab**: Browse completed experiments
**Metrics Tab**: Compare performance across runs
**Charts Tab**: Visualize training curves

---

## Configuration Management

### Adding Experiment Configs

1. **Via File Station**:
   - Upload YAML files to `/docker/cvlab-kit/config/`
   - Refresh Web UI

2. **Via SSH**:
   ```bash
   ssh admin@<nas-ip>
   cd /volume1/docker/cvlab-kit/config
   nano my_experiment.yaml
   ```

3. **Via Git** (recommended):
   ```bash
   cd /volume1/docker/cvlab-kit/source/config
   git pull
   ```

### Config File Example

```yaml
# config/cifar10_resnet.yaml
project: cifar10-classification
seed: 42
epochs: 100

model: resnet18(num_classes=10)

dataset:
  train: cifar10(train=True)
  val: cifar10(train=False)

dataloader:
  train: basic(batch_size=128, shuffle=True)
  val: basic(batch_size=128, shuffle=False)

optimizer: sgd(lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler: cosine(T_max=100)

loss: cross_entropy
metric: accuracy

checkpoint: model_checkpoint(save_top_k=3, monitor=val_accuracy, mode=max)
logger: wandb(project=cifar10-classification)
```

---

## Backup and Maintenance

### Data Backup (Synology)

**Hyper Backup**:
1. Install **Hyper Backup** package
2. Create backup task for `/docker/cvlab-kit/`
3. Exclude `source/` (can re-clone from git)
4. Schedule daily backups

**Manual Backup**:
```bash
# Via SSH
cd /volume1/docker/cvlab-kit
tar -czf backup-$(date +%Y%m%d).tar.gz logs/ outputs/ state/ config/
```

### Update CVLab-Kit

```bash
# SSH into Synology
ssh admin@<nas-ip>
cd /volume1/docker/cvlab-kit/source
git pull

# Rebuild container
# In Container Manager UI:
# Project → cvlab-kit → Action → Build
# (Takes 5-10 min for rebuild)
```

### Clean Up Old Data

```bash
# Remove old experiment logs (>30 days)
find /volume1/docker/cvlab-kit/logs -mtime +30 -type f -delete

# Clear queue logs
rm -rf /volume1/docker/cvlab-kit/queue_logs/*

# Clean Docker cache (reclaim space)
docker system prune -a
```

---

## Troubleshooting

### Container Won't Start

**Check logs**:
```
Container Manager → cvlab-kit → Logs
```

**Common issues**:
- Port 8000 already in use → Change port in compose file
- Build failed → Check internet connection, retry build
- Volume mount error → Verify folders exist in File Station

### Worker Can't Connect

**Test network connectivity**:
```bash
# From GPU machine
curl http://<nas-ip>:8000/api/health

# Expected: {"status": "ok"}
```

**Check firewall**:
- Synology: Control Panel → Security → Firewall → Allow port 8000
- Router: Port forwarding if accessing from external network

### Low Performance

**Synology resources**:
- Check Resource Monitor (CPU/RAM usage)
- If high: Stop other containers/services
- Consider upgrading RAM

**GPU worker**:
```bash
# Check GPU utilization
nvidia-smi

# Monitor worker logs
tail -f /path/to/cvlab-kit/logs/worker.log
```

### Database Corruption

**Restore from backup**:
```bash
cd /volume1/docker/cvlab-kit/state
mv db.sqlite db.sqlite.corrupted
# Restore from Hyper Backup or copy from backup
```

**Rebuild database**:
```bash
# Stop container
docker stop cvlab-kit

# Remove database
rm /volume1/docker/cvlab-kit/state/db.sqlite

# Start container (will create fresh DB)
docker start cvlab-kit

# Re-index existing logs
# Web UI → Settings → Reindex All Projects
```

---

## Advanced Configuration

### Custom Domain (HTTPS)

Use Synology's built-in reverse proxy:

1. **Control Panel** → **Login Portal** → **Advanced** → **Reverse Proxy**

2. Add rule:
   - Source: `cvlab.yourdomain.com` (443)
   - Destination: `localhost:8000`

3. Enable HTTPS with Let's Encrypt certificate

### Multiple GPU Workers

Each worker needs unique `HOST_ID`:

```bash
# Worker 1
uv run app.py --client-only --url http://<nas>:8000 --client-host-id gpu-rtx4090

# Worker 2
uv run app.py --client-only --url http://<nas>:8000 --client-host-id gpu-a100
```

### Resource Limits (Synology)

Prevent container from consuming all resources:

```yaml
# docker-compose.synology.yml
services:
  cvlab-kit:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '2.0'      # Max 2 CPU cores
          memory: 4G       # Max 4GB RAM
        reservations:
          memory: 2G       # Reserved 2GB RAM
```

---

## Security Considerations

### Network Security

**Recommended setup**:
- Keep Synology on private network (192.168.x.x)
- Use VPN for external access (Synology VPN Server)
- Don't expose port 8000 to internet directly

**If internet access needed**:
- Enable HTTPS with reverse proxy
- Add authentication (configure in `app.py`)
- Use strong firewall rules

### Data Privacy

- Experiment logs may contain sensitive data
- Enable Synology encryption for shared folders
- Use encrypted backups (Hyper Backup supports encryption)

---

## Performance Tuning

### Build Optimization

**Faster rebuilds**:
```yaml
# Use BuildKit cache
DOCKER_BUILDKIT=1 docker-compose -f docker/docker-compose.synology.yml build
```

### Database Optimization

**For large-scale deployments** (1000+ runs):
```bash
# Enable WAL mode for better concurrency
sqlite3 /volume1/docker/cvlab-kit/state/db.sqlite "PRAGMA journal_mode=WAL;"

# Vacuum database monthly
sqlite3 /volume1/docker/cvlab-kit/state/db.sqlite "VACUUM;"
```

### Log Rotation

Prevent logs from filling up storage:

```bash
# Add cron job (via Task Scheduler)
# Daily at 3 AM: compress old logs
find /volume1/docker/cvlab-kit/logs -name "*.csv" -mtime +7 -exec gzip {} \;
```

---

## Support

**Documentation**: `docs/`
**Issues**: GitHub Issues
**Community**: GitHub Discussions

---

## License

See [LICENSE](../LICENSE) file for details.
