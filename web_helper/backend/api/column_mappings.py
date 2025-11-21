"""Column Mapping API Endpoints

Provides REST API for column mapping management and suggestion generation.

Endpoints:
- GET    /api/projects/{project}/column-mappings       - List all mappings
- POST   /api/projects/{project}/column-mappings       - Create mapping
- PUT    /api/projects/{project}/column-mappings/{id}  - Update mapping
- DELETE /api/projects/{project}/column-mappings/{id}  - Delete mapping
- POST   /api/projects/{project}/column-suggestions    - Generate suggestions
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..models.column_mapping import (
    ColumnMapping,
    ColumnMappingCreate,
    ColumnMappingResponse,
    ColumnMappingSuggestionResponse,
    ColumnMappingUpdate,
)
from ..models.database import get_db
from ..models.run import Run
from ..services.column_mapper import ColumnMapperService
from ..utils.responses import error_response, success_response

router = APIRouter(prefix="/projects", tags=["column_mappings"])


# ============================================================================
# Helper Functions
# ============================================================================


def get_column_mapper_service(db: Session = Depends(get_db)) -> ColumnMapperService:
    """Dependency for ColumnMapperService"""
    return ColumnMapperService(db)


def extract_column_values(
    runs: List[Run], column_name: str, is_metric: bool = False
) -> List:
    """Extract values for a specific column across all runs"""
    values = []

    for run in runs:
        if is_metric:
            # Extract from final_metrics
            if run.final_metrics:
                value = run.final_metrics.get(column_name)
                if value is not None and isinstance(value, (int, float)):
                    values.append(float(value))
        else:
            # Extract from hyperparameters
            if run.hyperparameters:
                value = run.hyperparameters.get(column_name)
                if value is not None and isinstance(value, (int, float)):
                    values.append(float(value))

    return values


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/{project}/column-mappings", response_model=dict)
async def list_column_mappings(
    project: str,
    column_type: Optional[str] = Query(None, regex="^(hyperparam|metric)$"),
    active_only: bool = Query(True),
    db: Session = Depends(get_db),
    mapper: ColumnMapperService = Depends(get_column_mapper_service),
):
    """List all column mappings for a project.

    Query Parameters:
    - column_type: Filter by type ('hyperparam' or 'metric')
    - active_only: Only return active mappings (default: true)
    """
    try:
        mappings = mapper.get_mappings(
            project=project, column_type=column_type, active_only=active_only
        )

        return success_response(
            data=[ColumnMappingResponse.from_orm(m) for m in mappings]
        )

    except Exception as e:
        return error_response(
            title="Column Mappings Error",
            status=500,
            detail=f"Failed to retrieve column mappings: {str(e)}",
        )


@router.post("/{project}/column-mappings", response_model=dict)
async def create_column_mapping(
    project: str,
    mapping: ColumnMappingCreate,
    db: Session = Depends(get_db),
    mapper: ColumnMapperService = Depends(get_column_mapper_service),
):
    """Create a new column mapping.

    Body:
    - source_column: Original column name
    - target_column: Unified column name
    - column_type: 'hyperparam' or 'metric'
    - mapping_method: 'manual', 'auto', or 'verified'
    - confidence_score: Optional confidence score (for auto mappings)
    - algorithm: Optional algorithm name (for auto mappings)
    """
    try:
        # Verify project exists
        run_count = db.query(Run).filter(Run.project == project).count()
        if run_count == 0:
            raise HTTPException(
                status_code=404, detail=f"Project '{project}' not found"
            )

        # Check for duplicate mapping
        existing = (
            db.query(ColumnMapping)
            .filter(
                ColumnMapping.project == project,
                ColumnMapping.source_column == mapping.source_column,
                ColumnMapping.is_active == True,
            )
            .first()
        )

        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Active mapping already exists for column '{mapping.source_column}'",
            )

        # Create mapping
        new_mapping = mapper.create_mapping(
            project=project,
            source_column=mapping.source_column,
            target_column=mapping.target_column,
            column_type=mapping.column_type,
            mapping_method=mapping.mapping_method,
            confidence_score=mapping.confidence_score,
            algorithm=mapping.algorithm,
            mapping_metadata=mapping.mapping_metadata,
        )

        return success_response(data=ColumnMappingResponse.from_orm(new_mapping))

    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            title="Column Mapping Creation Error",
            status=500,
            detail=f"Failed to create column mapping: {str(e)}",
        )


@router.put("/{project}/column-mappings/{mapping_id}", response_model=dict)
async def update_column_mapping(
    project: str,
    mapping_id: int,
    update: ColumnMappingUpdate,
    db: Session = Depends(get_db),
    mapper: ColumnMapperService = Depends(get_column_mapper_service),
):
    """Update an existing column mapping.

    Body:
    - target_column: New unified column name
    - mapping_method: Update method ('manual', 'verified')
    - is_active: Activate/deactivate mapping
    """
    try:
        # Verify mapping exists and belongs to project
        existing = (
            db.query(ColumnMapping)
            .filter(ColumnMapping.id == mapping_id, ColumnMapping.project == project)
            .first()
        )

        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"Column mapping {mapping_id} not found in project '{project}'",
            )

        # Update mapping
        updated_mapping = mapper.update_mapping(
            mapping_id=mapping_id,
            target_column=update.target_column,
            mapping_method=update.mapping_method,
            is_active=update.is_active,
            mapping_metadata=update.mapping_metadata,
        )

        if not updated_mapping:
            raise HTTPException(
                status_code=500, detail="Failed to update column mapping"
            )

        return success_response(data=ColumnMappingResponse.from_orm(updated_mapping))

    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            title="Column Mapping Update Error",
            status=500,
            detail=f"Failed to update column mapping: {str(e)}",
        )


@router.delete("/{project}/column-mappings/{mapping_id}", response_model=dict)
async def delete_column_mapping(
    project: str,
    mapping_id: int,
    db: Session = Depends(get_db),
    mapper: ColumnMapperService = Depends(get_column_mapper_service),
):
    """Delete a column mapping."""
    try:
        # Verify mapping exists and belongs to project
        existing = (
            db.query(ColumnMapping)
            .filter(ColumnMapping.id == mapping_id, ColumnMapping.project == project)
            .first()
        )

        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"Column mapping {mapping_id} not found in project '{project}'",
            )

        # Delete mapping
        success = mapper.delete_mapping(mapping_id)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to delete column mapping"
            )

        return success_response(data={"deleted": True, "mapping_id": mapping_id})

    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            title="Column Mapping Deletion Error",
            status=500,
            detail=f"Failed to delete column mapping: {str(e)}",
        )


@router.post("/{project}/column-suggestions", response_model=dict)
async def generate_column_suggestions(
    project: str,
    column_type: str = Query("hyperparam", regex="^(hyperparam|metric)$"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    mapper: ColumnMapperService = Depends(get_column_mapper_service),
):
    """Generate automatic column mapping suggestions.

    Analyzes unmapped columns and suggests unified names based on:
    - Fuzzy string matching
    - Semantic similarity
    - Value range overlap
    - Distribution shape similarity
    - Context analysis

    Query Parameters:
    - column_type: 'hyperparam' or 'metric'
    - min_confidence: Minimum confidence threshold (0.0-1.0, default: 0.5)

    Returns:
    - List of suggestions sorted by confidence (descending)
    - Categorized by confidence level (high/medium/low)
    """
    try:
        # Get all runs for this project
        runs = db.query(Run).filter(Run.project == project).all()

        if not runs:
            raise HTTPException(
                status_code=404, detail=f"No runs found for project '{project}'"
            )

        # Extract columns from runs
        is_metric = column_type == "metric"
        all_columns = set()

        for run in runs:
            if is_metric:
                if run.final_metrics:
                    all_columns.update(run.final_metrics.keys())
            else:
                if run.hyperparameters:
                    all_columns.update(run.hyperparameters.keys())

        # Get existing mappings
        existing_mappings = mapper.get_mappings(project, column_type, active_only=True)
        mapped_columns = {m.source_column for m in existing_mappings}
        unified_columns = {m.target_column for m in existing_mappings}

        # Find unmapped columns
        unmapped_columns = list(all_columns - mapped_columns)

        if not unmapped_columns:
            return success_response(
                data=ColumnMappingSuggestionResponse(
                    project=project,
                    suggestions=[],
                    total_suggestions=0,
                    high_confidence_count=0,
                    medium_confidence_count=0,
                    low_confidence_count=0,
                )
            )

        # If no unified columns exist yet, use all columns as targets
        target_columns = list(unified_columns) if unified_columns else list(all_columns)

        # Extract values for each column
        source_data = {}
        target_data = {}

        for col in unmapped_columns:
            source_data[col] = extract_column_values(runs, col, is_metric)

        for col in target_columns:
            target_data[col] = extract_column_values(runs, col, is_metric)

        # Generate suggestions
        suggestions = mapper.generate_suggestions(
            project=project,
            source_columns=unmapped_columns,
            target_columns=target_columns,
            source_data=source_data,
            target_data=target_data,
            column_type=column_type,
            min_confidence=min_confidence,
        )

        # Categorize by confidence
        high_confidence = [s for s in suggestions if s.confidence_score >= 0.8]
        medium_confidence = [s for s in suggestions if 0.5 <= s.confidence_score < 0.8]
        low_confidence = [s for s in suggestions if s.confidence_score < 0.5]

        response = ColumnMappingSuggestionResponse(
            project=project,
            suggestions=suggestions,
            total_suggestions=len(suggestions),
            high_confidence_count=len(high_confidence),
            medium_confidence_count=len(medium_confidence),
            low_confidence_count=len(low_confidence),
        )

        return success_response(data=response)

    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            title="Column Suggestions Error",
            status=500,
            detail=f"Failed to generate suggestions: {str(e)}",
        )
