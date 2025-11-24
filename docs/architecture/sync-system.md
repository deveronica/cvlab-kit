# Synchronization System Architecture

## Overview

CVLab-Kit uses a distributed architecture where experiments can run on remote GPU servers (Workers) while being managed from a central server (Backend). This document describes the synchronization system that ensures data consistency between components.

## Components

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│    Frontend     │◄───────►│    Backend      │◄───────►│    Worker       │
│  (React Web)    │   HTTP  │  (FastAPI)      │   HTTP  │  (Middleend)    │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                   │                           │
                                   ▼                           ▼
                            ┌─────────────┐             ┌─────────────┐
                            │ web_helper/ │             │   logs_*/   │
                            │  - .db      │             │  (backup)   │
                            │  - queue_logs/           └─────────────┘
                            │  - logs/    │
                            └─────────────┘
```

### Roles

| Component | Role | Responsibility |
|-----------|------|----------------|
| **Backend** | Central Server | Queue management, result storage, API |
| **Worker** | Execution Agent | Run experiments, sync results |
| **Frontend** | User Interface | Visualization, queue submission |

## Data Flow

### 1. Config Flow (Server → Worker)

```
User → Frontend → Backend → Worker
         │           │         │
         │  POST     │  GET    │
         │  /queue   │  /next  │
         └───────────┴─────────►
                               │
                    Config downloaded
                    to Worker local
```

1. User creates config via Frontend
2. Backend saves to `web_helper/queue_logs/{uid}/config.yaml`
3. Worker polls `GET /api/queue/next_job`
4. Worker downloads config via `GET /api/configs/raw`

### 2. Log Sync Flow (Worker → Server)

```
Worker execution          Backend storage
     │                         │
     ├─► terminal_log.log ────►│ queue_logs/{uid}/
     ├─► terminal_err.log ────►│
     │                         │
     ├─► run.csv ─────────────►│ logs/{project}/
     ├─► run.yaml ────────────►│
     └─► run.pt ──────────────►│
```

| File Type | Sync Method | Verification |
|-----------|-------------|--------------|
| `*.log` | Delta (append) | diff |
| `*.csv` | Delta (append) | diff |
| `*.yaml` | Full replace | xxhash3 |
| `*.pt` | Full replace | xxhash3 |

### 3. Heartbeat Flow

```
Worker ──► Backend
  │
  ├─ host_id
  ├─ CPU/Memory/Disk stats
  ├─ GPU stats (multi-GPU support)
  ├─ torch_version, cuda_version
  └─ code_version (git hash, files hash, uv.lock hash)
```

## Hash Verification

All file integrity checks use **xxhash3** for fast, non-cryptographic hashing.

### Usage

```python
import xxhash

# Single file
hash = xxhash.xxh3_64(file_content).hexdigest()

# Content verification on upload
X-Content-Hash: <xxhash3 of content>
```

### Code Version Tracking

For reproducibility, Workers report code version with each heartbeat:

```json
{
  "code_version": {
    "git_hash": "abc1234",
    "git_dirty": false,
    "branch": "main",
    "files_hash": "def5678",  // xxhash3 of cvlabkit/*.py
    "uv_lock_hash": "ghi9012"  // xxhash3 of uv.lock
  }
}
```

## Recovery Protocol

### Network Disconnection

1. Worker reconnects with exponential backoff (1s → 2s → 4s → max 60s)
2. Worker queries `GET /api/sync/status/{uid}`
3. Server returns file states with hashes
4. Worker compares local vs server hashes
5. Re-sync missing or mismatched files

### Worker Crash

1. Job status remains `running` in DB
2. PID check fails on next poll
3. Status transitions to `crashed`
4. Manual re-assignment available via Frontend

## API Reference

### Sync Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sync/experiment/{uid}/{file}` | POST | Upload terminal logs |
| `/api/sync/run/{uid}/{file}` | POST | Upload result files |
| `/api/sync/status/{uid}` | GET | Query sync status |
| `/api/sync/checkpoint/{uid}` | POST | Update recovery checkpoint |

### Headers

| Header | Description |
|--------|-------------|
| `X-Content-Hash` | xxhash3 of content (optional, for verification) |
| `Authorization` | `Bearer <api_key>` (if authentication enabled) |

## File Structure

```
web_helper/
├── web_helper.db          # SQLite DB (runs, queue, devices)
├── daemon_state.json      # Daemon PID tracking (JSON)
├── queue_logs/            # Experiment logs
│   └── {uid}/
│       ├── config.yaml
│       ├── terminal_log.log
│       └── terminal_err.log
└── queue_state.json       # Queue state backup

logs/                      # CVLab-Kit results (Server authoritative)
└── {project}/
    ├── {run_name}.csv
    ├── {run_name}.yaml
    └── {run_name}.pt

logs_{server}/             # Worker local backup
├── experiments/{uid}/     # Terminal logs backup
└── runs/{project}/        # Results backup
```

## Design Decisions

### Why xxhash3?

- **Speed**: 10-100x faster than MD5/SHA
- **Purpose**: Change detection, not cryptographic security
- **Collision resistance**: Sufficient for file integrity

### Why Server-Authoritative?

- Single source of truth for results
- Worker backups are for resilience, not querying
- Simplifies conflict resolution

### Why JSON for Daemon State?

- Portable (no DB dependency for client-only mode)
- Human-readable
- Survives SSH disconnection
