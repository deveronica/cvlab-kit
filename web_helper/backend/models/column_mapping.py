"""Column Mapping Models

Handles parameter name unification across different experiment runs.
When parameter names change (e.g., 'lr' vs 'learning_rate'), this system
provides automatic suggestions and manual mapping capabilities.

Features:
- Automatic column mapping suggestions with confidence scores
- Manual override support
- Project-scoped mappings
- Multiple algorithm support (fuzzy, semantic, value-based, distribution)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)

from .database import Base


class ColumnMapping(Base):
    """SQLAlchemy model for column mappings.

    Stores unified column name mappings for hyperparameters and metrics.
    """

    __tablename__ = "column_mappings"

    id = Column(Integer, primary_key=True, index=True)
    project = Column(String, nullable=False, index=True)

    # Source column name (original name in experiment)
    source_column = Column(String, nullable=False, index=True)

    # Target column name (unified name for display)
    target_column = Column(String, nullable=False, index=True)

    # Column type: 'hyperparam' or 'metric'
    column_type = Column(String, nullable=False)

    # Mapping method: 'auto' (suggested), 'manual' (user-defined), 'verified' (user-confirmed auto)
    mapping_method = Column(String, nullable=False, default="auto")

    # Confidence score for automatic mappings (0.0 - 1.0)
    confidence_score = Column(Float, nullable=True)

    # Algorithm used for automatic mapping: 'fuzzy', 'semantic', 'value_range', 'distribution', 'context'
    algorithm = Column(String, nullable=True)

    # Additional metadata (JSON stored as text)
    # Could include: value_range_overlap, distribution_similarity, context_info
    mapping_metadata = Column(Text, nullable=True)

    # Whether this mapping is active
    is_active = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # User who created/modified the mapping (optional, for multi-user systems)
    created_by = Column(String, nullable=True)

    def __repr__(self):
        return f"<ColumnMapping {self.source_column} -> {self.target_column} ({self.confidence_score:.2f})>"


# Pydantic models for API requests/responses


class ColumnMappingBase(BaseModel):
    """Base schema for column mappings"""

    source_column: str = Field(..., description="Original column name from experiment")
    target_column: str = Field(..., description="Unified column name for display")
    column_type: str = Field(..., description="Type: 'hyperparam' or 'metric'")
    mapping_method: str = Field(
        default="manual", description="Method: 'auto', 'manual', or 'verified'"
    )
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score (0-1)"
    )
    algorithm: Optional[str] = Field(
        None, description="Algorithm used for auto-mapping"
    )
    mapping_metadata: Optional[str] = Field(
        None, description="Additional metadata as JSON string"
    )
    is_active: bool = Field(default=True, description="Whether mapping is active")


class ColumnMappingCreate(ColumnMappingBase):
    """Schema for creating a new column mapping"""

    project: str = Field(..., description="Project name")


class ColumnMappingUpdate(BaseModel):
    """Schema for updating an existing column mapping"""

    target_column: Optional[str] = None
    mapping_method: Optional[str] = None
    is_active: Optional[bool] = None
    mapping_metadata: Optional[str] = None


class ColumnMappingResponse(ColumnMappingBase):
    """Schema for column mapping responses"""

    id: int
    project: str
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None

    class Config:
        from_attributes = True


class ColumnSuggestion(BaseModel):
    """Schema for automatic column mapping suggestions.

    Returned by the suggestion endpoint to provide automatic mapping recommendations.
    """

    source_column: str = Field(..., description="Original column name")
    target_column: str = Field(..., description="Suggested unified column name")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence (0-1)")
    algorithm: str = Field(..., description="Algorithm used")
    reason: str = Field(..., description="Human-readable explanation")
    column_type: str = Field(..., description="Type: 'hyperparam' or 'metric'")

    # Additional details for user review
    similarity_details: Optional[dict] = Field(
        None, description="Detailed similarity metrics"
    )


class ColumnMappingSuggestionResponse(BaseModel):
    """Response containing multiple column mapping suggestions"""

    project: str
    suggestions: list[ColumnSuggestion]
    total_suggestions: int
    high_confidence_count: int = Field(
        ..., description="Count of suggestions with confidence >= 0.8"
    )
    medium_confidence_count: int = Field(
        ..., description="Count of suggestions with 0.5 <= confidence < 0.8"
    )
    low_confidence_count: int = Field(
        ..., description="Count of suggestions with confidence < 0.5"
    )
