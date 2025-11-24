"""Synchronization models for distributed experiment execution.

This module defines data models for tracking synchronization state between
Server and Worker nodes. Uses xxhash3 for fast file change detection.

Architecture:
- Server: Central authority for experiment queue and results
- Worker (Middleend): Executes experiments and syncs results back to Server
- All execution flows through Worker (Local Worker for same-machine execution)
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class SyncStatus(str, Enum):
    """Synchronization status for experiment files."""

    PENDING = "pending"  # Not yet started syncing
    SYNCING = "syncing"  # Currently syncing
    SYNCED = "synced"  # Fully synchronized
    FAILED = "failed"  # Sync failed (will retry)
    OUTDATED = "outdated"  # Local has newer data than server


class CodeVersion(BaseModel):
    """Code version information for reproducibility tracking.

    Worker reports this on each heartbeat to ensure experiment reproducibility.
    Server stores this with experiment results for audit trail.
    """

    git_hash: str = Field(..., description="Git commit hash of cvlabkit")
    git_dirty: bool = Field(False, description="True if working directory has uncommitted changes")
    files_hash: str = Field(..., description="xxhash3 of core source files for integrity verification")
    uv_lock_hash: str = Field(..., description="xxhash3 of uv.lock for dependency reproducibility")

    # Optional metadata
    branch: Optional[str] = Field(None, description="Current git branch")
    python_version: Optional[str] = Field(None, description="Python version")
    pytorch_version: Optional[str] = Field(None, description="PyTorch version")
    cuda_version: Optional[str] = Field(None, description="CUDA version")

    # Component versions for experiment reproducibility
    component_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="xxhash3 per component file: {path: hash} (agent + used components)",
    )


class FileCheckpoint(BaseModel):
    """Checkpoint for a single file's sync state.

    Used for delta synchronization of append-only files (CSV, log).
    """

    file_path: str = Field(..., description="Relative path from experiment directory")
    offset: int = Field(0, description="Last synced byte offset")
    hash: str = Field(..., description="xxhash3 of synced content (for verification)")
    last_synced_at: datetime = Field(default_factory=datetime.utcnow)


class SyncCheckpoint(BaseModel):
    """Complete checkpoint for experiment synchronization.

    Stored on both Server and Worker for recovery after disconnection.
    """

    experiment_uid: str = Field(..., description="Unique experiment identifier")
    worker_host_id: str = Field(..., description="Host ID of the Worker")

    # File sync state
    file_checkpoints: Dict[str, FileCheckpoint] = Field(
        default_factory=dict,
        description="Checkpoints per file: {filename: FileCheckpoint}",
    )

    # Overall sync state
    sync_status: SyncStatus = Field(SyncStatus.PENDING)
    last_synced_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None, description="Last error if sync_status is FAILED")

    # Code version at execution time (for reproducibility)
    code_version: Optional[CodeVersion] = Field(None)


class SyncRequest(BaseModel):
    """Request model for file synchronization API.

    Worker sends this to Server to upload experiment files.
    """

    experiment_uid: str
    file_name: str
    content_type: str = Field(..., description="'delta' for append-only, 'full' for complete replacement")
    offset: int = Field(0, description="Start offset for delta sync")
    content_hash: str = Field(..., description="xxhash3 of the content being sent")


class SyncResponse(BaseModel):
    """Response model for file synchronization API.

    Server returns this to confirm sync status.
    """

    success: bool
    message: str
    server_offset: int = Field(0, description="Server's current offset (for verification)")
    server_hash: Optional[str] = Field(None, description="Server's hash (for verification)")


class SyncStatusRequest(BaseModel):
    """Request model for querying sync status.

    Worker sends this on reconnection to determine what needs to be re-synced.
    """

    experiment_uid: str
    worker_host_id: str
    file_names: list[str] = Field(default_factory=list, description="Files to check status for")


class SyncStatusResponse(BaseModel):
    """Response model for sync status query.

    Server returns current sync state for requested files.
    """

    experiment_uid: str
    sync_status: SyncStatus
    file_states: Dict[str, FileCheckpoint] = Field(
        default_factory=dict,
        description="Current state per file on server",
    )
    needs_resync: list[str] = Field(
        default_factory=list,
        description="Files that need to be re-synced",
    )
