"""Queue management data models"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    EXPIRED = "expired"


class JobPriority(str, Enum):
    """Job priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ResourceRequirement(BaseModel):
    """Resource requirements for a job"""

    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    gpu_count: Optional[int] = None
    gpu_memory_gb: Optional[float] = None
    disk_space_gb: Optional[float] = None
    estimated_runtime_hours: Optional[float] = None


class QueueJob(BaseModel):
    """Queue job model (Experiment)"""

    # Primary key: experiment_uid is unique within a day (YYYYMMDD_hash4)
    experiment_uid: str = Field(
        ..., description="Experiment UID in format {YYYYMMDD}_{hash4} (Primary Key)"
    )
    name: str = Field(..., description="Human-readable job name")
    project: str = Field(..., description="Project name")
    config_path: str = Field(..., description="Path to configuration file")
    # Note: run_uid removed - Queue operates at experiment level only
    # run_name is specified in config YAML and handled by cvlabkit

    # Status and timing
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime = Field(default_factory=datetime.now)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Resource management
    requirements: Optional[ResourceRequirement] = None
    assigned_device: Optional[str] = None

    # Execution details
    command: Optional[str] = None
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = Field(default_factory=dict)

    # Progress and results
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_metrics: Dict[str, Any] = Field(default_factory=dict)

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # User and metadata
    user: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueueStats(BaseModel):
    """Queue statistics"""

    total_jobs: int = 0
    pending_jobs: int = 0
    queued_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0

    # Resource utilization
    total_cpu_cores: int = 0
    used_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    total_gpu_count: int = 0
    used_gpu_count: int = 0

    # Timing statistics
    avg_queue_time_minutes: Optional[float] = None
    avg_execution_time_minutes: Optional[float] = None
    longest_running_job_minutes: Optional[float] = None


class QueueOperation(BaseModel):
    """Queue operation request"""

    experiment_uid: str
    operation: str  # "start", "pause", "resume", "cancel", "retry", "set_priority"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class JobSubmission(BaseModel):
    """Job submission request"""

    name: str
    project: str
    config_path: str
    priority: JobPriority = JobPriority.NORMAL
    experiment_uid: Optional[str] = (
        None  # If provided, use this instead of generating new one
    )
    requirements: Optional[ResourceRequirement] = None
    tags: List[str] = Field(default_factory=list)
    environment_vars: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueueConfiguration(BaseModel):
    """Queue system configuration"""

    max_concurrent_jobs: int = 4
    max_jobs_per_device: int = 1
    default_job_timeout_hours: float = 24.0
    enable_auto_scaling: bool = False
    enable_priority_scheduling: bool = True
    enable_resource_matching: bool = True
    cleanup_completed_jobs_after_hours: float = 168.0  # 1 week

    # Resource limits
    max_cpu_cores_per_job: Optional[int] = None
    max_memory_gb_per_job: Optional[float] = None
    max_gpu_count_per_job: Optional[int] = None

    # Scheduling preferences
    prefer_high_priority: bool = True
    prefer_shorter_jobs: bool = False
    load_balancing_strategy: str = (
        "round_robin"  # "round_robin", "least_loaded", "resource_aware"
    )
