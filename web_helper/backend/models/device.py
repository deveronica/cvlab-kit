"""Device model for compute resource tracking."""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String

from .database import Base


class Device(Base):
    """Database model for compute devices."""

    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    host_id = Column(String, unique=True, index=True)

    # GPU monitoring (aggregated for backward compatibility)
    gpu_util = Column(Float)  # Average across all GPUs
    vram_used = Column(Integer)  # MB (sum across all GPUs)
    vram_total = Column(Integer)  # MB (sum across all GPUs)
    gpu_temperature = Column(Float)  # Celsius
    gpu_power_usage = Column(Float)  # Watts

    # Multi-GPU support (new fields)
    gpu_count = Column(Integer, default=0)  # Number of GPUs
    gpus_detail = Column(JSON, nullable=True)  # Array of detailed GPU info

    # System monitoring
    cpu_util = Column(Float)
    memory_used = Column(Integer)  # Bytes
    memory_total = Column(Integer)  # Bytes
    disk_free = Column(Float)  # GB

    # Software versions
    torch_version = Column(String)
    cuda_version = Column(String)

    # Code version for reproducibility
    code_version = Column(JSON, nullable=True)  # {git_hash, files_hash, uv_lock_hash, ...}

    # Status and timing
    status = Column(String, default="offline")
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
