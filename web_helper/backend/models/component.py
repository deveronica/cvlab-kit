"""Component models for version management and synchronization."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from .database import Base


# =============================================================================
# SQLAlchemy Models (Database Tables)
# =============================================================================


class ComponentVersion(Base):
    """Database model for component versions (content-addressable storage)."""

    __tablename__ = "component_versions"

    id = Column(Integer, primary_key=True, index=True)
    hash = Column(String, unique=True, index=True)  # xxhash3 of content
    path = Column(String, index=True)  # "agent/classification.py"
    category = Column(String, index=True)  # "agent", "model", "transform", etc.
    name = Column(String, index=True)  # "classification", "resnet", etc.
    content = Column(Text)  # Actual code content
    is_active = Column(Boolean, default=False)  # Current active version for this path
    created_at = Column(DateTime, default=datetime.utcnow)


class ExperimentComponentManifest(Base):
    """Database model for experiment component snapshots (reproducibility)."""

    __tablename__ = "experiment_component_manifests"

    id = Column(Integer, primary_key=True, index=True)
    experiment_uid = Column(String, index=True)  # Experiment identifier
    component_path = Column(String)  # "agent/classification.py"
    component_hash = Column(String)  # Hash of component version used
    created_at = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# Pydantic Models (API Request/Response)
# =============================================================================


class ComponentVersionResponse(BaseModel):
    """Response model for component version."""

    hash: str
    path: str
    category: str
    name: str
    is_active: bool
    created_at: datetime
    content: Optional[str] = None  # Only included when explicitly requested

    class Config:
        from_attributes = True


class ComponentListItem(BaseModel):
    """List item for component overview."""

    path: str
    category: str
    name: str
    active_hash: Optional[str] = None
    version_count: int
    updated_at: Optional[datetime] = None


class ComponentUploadRequest(BaseModel):
    """Request model for uploading a new component version."""

    path: str  # "agent/classification.py" or "model/resnet.py"
    content: str  # Code content
    activate: bool = True  # Whether to activate this version immediately


class ComponentUploadResponse(BaseModel):
    """Response model for component upload."""

    hash: str
    path: str
    category: str
    name: str
    is_new: bool  # True if this is a new version, False if already exists
    is_active: bool


class ComponentActivateRequest(BaseModel):
    """Request model for activating a specific component version."""

    hash: str


class ComponentDiffRequest(BaseModel):
    """Request model for comparing two component versions."""

    from_hash: str
    to_hash: str


class ComponentDiffResponse(BaseModel):
    """Response model for component diff."""

    from_hash: str
    to_hash: str
    from_content: str
    to_content: str
    path: str


class ComponentSyncRequest(BaseModel):
    """Request model for component synchronization."""

    required_components: list[str]  # List of paths needed
    local_hashes: dict[str, str]  # path -> local hash mapping


class ComponentSyncResponse(BaseModel):
    """Response model for component synchronization."""

    to_download: list[str]  # List of hashes to download
    components: dict[str, ComponentVersionResponse]  # hash -> component info


class ExperimentManifestResponse(BaseModel):
    """Response model for experiment component manifest."""

    experiment_uid: str
    components: dict[str, str]  # path -> hash mapping
    created_at: datetime
