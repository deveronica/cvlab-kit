"""Component Discovery API for CVLab-Kit components

Includes:
- Component discovery from local filesystem
- Version management (upload, activate, rollback)
- Experiment manifest for reproducibility
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import xxhash
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..models import get_db
from ..models.component import (
    ComponentActivateRequest,
    ComponentDiffRequest,
    ComponentDiffResponse,
    ComponentListItem,
    ComponentUploadRequest,
    ComponentUploadResponse,
    ComponentVersionResponse,
    ExperimentManifestResponse,
)
from ..services.component_store import ComponentStore
from ..utils.responses import error_response, success_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/components", tags=["components"])


class ComponentInfo(BaseModel):
    """Component metadata"""

    name: str
    type: str  # 'model', 'dataset', 'transform', 'optimizer', 'loss', 'metric', 'agent'
    path: str
    hash: Optional[str] = None  # xxhash3 of file content
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    examples: List[Dict[str, Any]] = []


class BundleRequest(BaseModel):
    """Request for component bundle based on config"""

    agent: str
    components: Dict[str, str]  # category -> component name(s)


class BundleInfo(BaseModel):
    """Information about a component bundle"""

    files: List[ComponentInfo]
    total_size: int
    total_hash: str  # Combined hash of all files


class ComponentCategory(BaseModel):
    """Category with components"""

    category: str
    count: int
    components: List[ComponentInfo]


def calculate_file_hash(file_path: Path) -> Optional[str]:
    """Calculate xxhash3 of a file."""
    try:
        with open(file_path, "rb") as f:
            return xxhash.xxh3_64(f.read()).hexdigest()
    except Exception:
        return None


def discover_agents(base_path: str = "cvlabkit/agent") -> List[ComponentInfo]:
    """Discover all available agents."""
    agents = []

    if not os.path.exists(base_path):
        return agents

    for file_path in Path(base_path).glob("*.py"):
        if file_path.name.startswith("_") or file_path.name == "__init__.py":
            continue
        # Skip legacy folder
        if "legacy" in str(file_path):
            continue

        try:
            agent_info = extract_component_info(file_path, "agent")
            if agent_info:
                agents.append(agent_info)
        except Exception as e:
            logger.warning(f"Error processing agent {file_path}: {e}")

    return agents


def discover_components(
    base_path: str = "cvlabkit/component",
) -> Dict[str, List[ComponentInfo]]:
    """Discover all CVLab-Kit components by scanning the component directory"""
    components = {
        "model": [],
        "dataset": [],
        "transform": [],
        "optimizer": [],
        "loss": [],
        "metric": [],
        "dataloader": [],
    }

    if not os.path.exists(base_path):
        return components

    # Scan each component category
    for category in components:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            continue

        # Find all Python files in the category
        for file_path in Path(category_path).glob("*.py"):
            if file_path.name == "__init__.py":
                continue

            try:
                # Extract component info
                component_info = extract_component_info(file_path, category)
                if component_info:
                    components[category].append(component_info)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue

    return components


def extract_component_info(file_path: Path, category: str) -> Optional[ComponentInfo]:
    """Extract component information from a Python file"""
    try:
        # Read the file content
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Calculate file hash
        file_hash = calculate_file_hash(file_path)

        # Extract basic info
        name = file_path.stem

        # Try to extract docstring as description
        description = None
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                # Found start of docstring
                quote = '"""' if '"""' in line else "'''"
                desc_lines = []

                # Check if it's a single-line docstring
                if line.count(quote) >= 2:
                    start = line.find(quote) + 3
                    end = line.rfind(quote)
                    description = line[start:end].strip()
                else:
                    # Multi-line docstring
                    for j in range(i + 1, len(lines)):
                        if quote in lines[j]:
                            break
                        desc_lines.append(lines[j].strip())
                    description = " ".join(desc_lines).strip()
                break

        # Extract parameters from class definitions or function signatures
        parameters = extract_parameters(content)

        # Generate example configuration
        examples = generate_example_config(name, category, parameters)

        return ComponentInfo(
            name=name,
            type=category,
            path=str(file_path),
            hash=file_hash,
            description=description,
            parameters=parameters,
            examples=examples,
        )

    except Exception as e:
        logger.warning(f"Error extracting info from {file_path}: {e}")
        return None


def extract_parameters(content: str) -> Dict[str, Any]:
    """Extract parameters from class __init__ methods or function definitions"""
    parameters = {}

    lines = content.split("\n")
    in_init = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Look for __init__ method
        if "def __init__(" in stripped:
            in_init = True
            continue

        # Stop at next method or class
        if in_init and stripped.startswith("def ") and "__init__" not in stripped:
            break

        # Extract parameters from __init__
        if in_init and ":" in stripped and "=" in stripped:
            # Parse parameter like "param: type = default,"
            param_line = stripped.strip(",")
            if "=" in param_line:
                param_part = param_line.split("=")[0].strip()
                default_part = param_line.split("=")[1].strip()

                if ":" in param_part:
                    param_name = param_part.split(":")[0].strip()
                    param_type = param_part.split(":")[1].strip()
                else:
                    param_name = param_part.strip()
                    param_type = "Any"

                if param_name and param_name != "self":
                    parameters[param_name] = {
                        "type": param_type,
                        "default": default_part,
                        "required": default_part == "",
                    }

    return parameters


def generate_example_config(
    name: str, category: str, parameters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate example configuration based on component type and parameters"""
    examples = []

    # Basic example
    basic_config = {
        "name": f"example_{name}",
        "type": f"{category}.{name}",
        "params": {},
    }

    # Add some default parameters
    for param_name, param_info in parameters.items():
        if param_info.get("required", False):
            # Add required parameters with sensible defaults
            if "int" in param_info.get("type", "").lower():
                basic_config["params"][param_name] = 32
            elif "float" in param_info.get("type", "").lower():
                basic_config["params"][param_name] = 0.001
            elif "bool" in param_info.get("type", "").lower():
                basic_config["params"][param_name] = True
            elif "str" in param_info.get("type", "").lower():
                basic_config["params"][param_name] = f"example_{param_name}"
            else:
                basic_config["params"][param_name] = None

    examples.append(basic_config)

    return examples


@router.get("/")
@router.get("")
async def list_components():
    """List all available components by category"""
    try:
        components = discover_components()

        categories = []
        for category, comp_list in components.items():
            categories.append(
                ComponentCategory(
                    category=category, count=len(comp_list), components=comp_list
                )
            )

        return success_response(
            categories, {"message": "Components discovered successfully"}
        )

    except Exception as e:
        return error_response("Failed to discover components", 500, {"error": str(e)})


@router.get("/category/{category}")
async def get_components_by_category(category: str):
    """Get components in a specific category"""
    try:
        components = discover_components()

        if category not in components:
            raise HTTPException(
                status_code=404, detail=f"Category '{category}' not found"
            )

        return success_response(
            components[category], {"message": f"Components in category '{category}'"}
        )

    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to get components", 500, {"error": str(e)})


@router.get("/search")
async def search_components(q: str):
    """Search components by name or description"""
    try:
        all_components = discover_components()

        results = []
        query = q.lower()

        for category, comp_list in all_components.items():
            for component in comp_list:
                # Search in name and description
                if query in component.name.lower() or (
                    component.description and query in component.description.lower()
                ):
                    results.append(component)

        return success_response(
            results, {"message": f"Found {len(results)} components matching '{q}'"}
        )

    except Exception as e:
        return error_response("Search failed", 500, {"error": str(e)})


@router.get("/component/{category}/{name}")
async def get_component_details(category: str, name: str):
    """Get detailed information about a specific component"""
    try:
        components = discover_components()

        if category not in components:
            raise HTTPException(
                status_code=404, detail=f"Category '{category}' not found"
            )

        # Find the specific component
        component = None
        for comp in components[category]:
            if comp.name == name:
                component = comp
                break

        if not component:
            raise HTTPException(
                status_code=404,
                detail=f"Component '{name}' not found in category '{category}'",
            )

        return success_response(
            component, {"message": f"Component details for '{name}'"}
        )

    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to get component details", 500, {"error": str(e)})


@router.post("/validate-config")
async def validate_component_config(config: Dict[str, Any]):
    """Validate a component configuration"""
    try:
        # Extract component type
        component_type = config.get("type", "")
        if not component_type or "." not in component_type:
            return error_response(
                "Invalid component type format", 400, {"expected": "category.name"}
            )

        category, name = component_type.split(".", 1)

        # Get component info
        components = discover_components()
        if category not in components:
            return error_response("Unknown category", 400, {"category": category})

        component = None
        for comp in components[category]:
            if comp.name == name:
                component = comp
                break

        if not component:
            return error_response("Unknown component", 400, {"component": name})

        # Validate parameters
        config_params = config.get("params", {})
        validation_errors = []

        for param_name, param_info in component.parameters.items():
            if param_info.get("required", False) and param_name not in config_params:
                validation_errors.append(f"Missing required parameter: {param_name}")

        if validation_errors:
            return error_response(
                "Configuration validation failed", 400, {"errors": validation_errors}
            )

        return success_response({"valid": True}, {"message": "Configuration is valid"})

    except Exception as e:
        return error_response("Validation failed", 500, {"error": str(e)})


@router.get("/agents")
async def list_agents():
    """List all available agents with their hashes"""
    try:
        agents = discover_agents()
        return success_response(
            agents, {"message": f"Found {len(agents)} agents"}
        )
    except Exception as e:
        return error_response("Failed to discover agents", 500, {"error": str(e)})


@router.get("/agent/{name}")
async def get_agent_details(name: str):
    """Get detailed information about a specific agent"""
    try:
        agents = discover_agents()
        agent = next((a for a in agents if a.name == name), None)

        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

        return success_response(agent, {"message": f"Agent details for '{name}'"})

    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to get agent details", 500, {"error": str(e)})


@router.get("/agent/{name}/source")
async def get_agent_source(name: str):
    """Download agent source file"""
    try:
        agents = discover_agents()
        agent = next((a for a in agents if a.name == name), None)

        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

        file_path = Path(agent.path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Agent file not found")

        def iter_file():
            with open(file_path, "rb") as f:
                yield from f

        return StreamingResponse(
            iter_file(),
            media_type="text/x-python",
            headers={
                "Content-Disposition": f"attachment; filename={name}.py",
                "X-Content-Hash": agent.hash or "",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to get agent source", 500, {"error": str(e)})


@router.get("/component/{category}/{name}/source")
async def get_component_source(category: str, name: str):
    """Download component source file"""
    try:
        components = discover_components()

        if category not in components:
            raise HTTPException(status_code=404, detail=f"Category '{category}' not found")

        component = next((c for c in components[category] if c.name == name), None)

        if not component:
            raise HTTPException(
                status_code=404,
                detail=f"Component '{name}' not found in category '{category}'",
            )

        file_path = Path(component.path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Component file not found")

        def iter_file():
            with open(file_path, "rb") as f:
                yield from f

        return StreamingResponse(
            iter_file(),
            media_type="text/x-python",
            headers={
                "Content-Disposition": f"attachment; filename={name}.py",
                "X-Content-Hash": component.hash or "",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to get component source", 500, {"error": str(e)})


@router.post("/bundle/info")
async def get_bundle_info(request: BundleRequest):
    """Get information about required files for a config bundle (without downloading)"""
    try:
        files = []
        total_size = 0

        # Get agent info
        agents = discover_agents()
        agent = next((a for a in agents if a.name == request.agent), None)
        if agent:
            files.append(agent)
            agent_path = Path(agent.path)
            if agent_path.exists():
                total_size += agent_path.stat().st_size

        # Get component info
        components = discover_components()
        for category, comp_names in request.components.items():
            if category not in components:
                continue

            # Handle both single name and comma-separated names
            names = [n.strip() for n in comp_names.split(",")]
            for name in names:
                # Parse component name (handle "name(params)" syntax)
                base_name = name.split("(")[0].strip()
                comp = next((c for c in components[category] if c.name == base_name), None)
                if comp:
                    files.append(comp)
                    comp_path = Path(comp.path)
                    if comp_path.exists():
                        total_size += comp_path.stat().st_size

        # Calculate combined hash
        combined = "".join(f.hash or "" for f in files)
        total_hash = xxhash.xxh3_64(combined.encode()).hexdigest()

        return success_response(
            BundleInfo(files=files, total_size=total_size, total_hash=total_hash),
            {"message": f"Bundle contains {len(files)} files"},
        )

    except Exception as e:
        return error_response("Failed to get bundle info", 500, {"error": str(e)})


@router.post("/bundle/download")
async def download_bundle(request: BundleRequest):
    """Download a tarball of required agent and components"""
    import io
    import tarfile

    try:
        # Collect files
        file_paths = []

        # Get agent
        agents = discover_agents()
        agent = next((a for a in agents if a.name == request.agent), None)
        if agent:
            file_paths.append(Path(agent.path))

        # Get components
        components = discover_components()
        for category, comp_names in request.components.items():
            if category not in components:
                continue

            names = [n.strip() for n in comp_names.split(",")]
            for name in names:
                base_name = name.split("(")[0].strip()
                comp = next((c for c in components[category] if c.name == base_name), None)
                if comp:
                    file_paths.append(Path(comp.path))

        # Create tarball in memory
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for file_path in file_paths:
                if file_path.exists():
                    tar.add(file_path, arcname=str(file_path))

        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.read()]),
            media_type="application/gzip",
            headers={
                "Content-Disposition": f"attachment; filename=bundle-{request.agent}.tar.gz",
            },
        )

    except Exception as e:
        return error_response("Failed to create bundle", 500, {"error": str(e)})


# =============================================================================
# Version Management Endpoints
# =============================================================================


@router.get("/versions")
async def list_versioned_components(db: Session = Depends(get_db)):
    """List all components with version management enabled."""
    try:
        store = ComponentStore(db)
        components = store.get_all_components()
        return success_response(
            [c.model_dump() for c in components],
            {"message": f"Found {len(components)} versioned components"},
        )
    except Exception as e:
        return error_response("Failed to list versioned components", 500, {"error": str(e)})


@router.get("/versions/{category}")
async def list_versioned_by_category(category: str, db: Session = Depends(get_db)):
    """List versioned components in a specific category."""
    try:
        store = ComponentStore(db)
        components = store.get_components_by_category(category)
        return success_response(
            [c.model_dump() for c in components],
            {"message": f"Found {len(components)} components in {category}"},
        )
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response("Failed to list components", 500, {"error": str(e)})


@router.get("/versions/{category}/{name}")
async def get_component_versions(
    category: str, name: str, include_content: bool = False, db: Session = Depends(get_db)
):
    """Get version history for a specific component."""
    try:
        store = ComponentStore(db)
        versions = store.get_component_versions(category, name, include_content)
        return success_response(
            [v.model_dump() for v in versions],
            {"message": f"Found {len(versions)} versions"},
        )
    except Exception as e:
        return error_response("Failed to get component versions", 500, {"error": str(e)})


@router.get("/versions/hash/{hash}")
async def get_version_by_hash(hash: str, db: Session = Depends(get_db)):
    """Get specific version by content hash."""
    try:
        store = ComponentStore(db)
        version = store.get_version_by_hash(hash, include_content=True)
        if not version:
            raise HTTPException(status_code=404, detail=f"Version with hash '{hash}' not found")
        return success_response(version.model_dump(), {"message": "Version found"})
    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to get version", 500, {"error": str(e)})


@router.post("/versions/upload")
async def upload_component_version(request: ComponentUploadRequest, db: Session = Depends(get_db)):
    """Upload a new component version."""
    try:
        store = ComponentStore(db)
        version, is_new = store.upload_version(request.path, request.content, request.activate)
        return success_response(
            ComponentUploadResponse(
                hash=version.hash,
                path=version.path,
                category=version.category,
                name=version.name,
                is_new=is_new,
                is_active=version.is_active,
            ).model_dump(),
            {"message": "New version uploaded" if is_new else "Version already exists"},
        )
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response("Failed to upload component", 500, {"error": str(e)})


@router.post("/versions/activate")
async def activate_component_version(request: ComponentActivateRequest, db: Session = Depends(get_db)):
    """Activate a specific component version (rollback)."""
    try:
        store = ComponentStore(db)
        version = store.activate_version(request.hash)
        if not version:
            raise HTTPException(status_code=404, detail=f"Version with hash '{request.hash}' not found")
        return success_response(version.model_dump(), {"message": "Version activated"})
    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to activate version", 500, {"error": str(e)})


@router.post("/versions/diff")
async def diff_component_versions(request: ComponentDiffRequest, db: Session = Depends(get_db)):
    """Compare two component versions."""
    try:
        store = ComponentStore(db)
        from_version = store.get_version_by_hash(request.from_hash)
        to_version = store.get_version_by_hash(request.to_hash)

        if not from_version:
            raise HTTPException(status_code=404, detail=f"Version '{request.from_hash}' not found")
        if not to_version:
            raise HTTPException(status_code=404, detail=f"Version '{request.to_hash}' not found")

        if from_version.path != to_version.path:
            return error_response("Cannot diff versions of different components", 400)

        return success_response(
            ComponentDiffResponse(
                from_hash=request.from_hash,
                to_hash=request.to_hash,
                from_content=from_version.content or "",
                to_content=to_version.content or "",
                path=from_version.path,
            ).model_dump(),
            {"message": "Diff generated"},
        )
    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to generate diff", 500, {"error": str(e)})


@router.post("/versions/scan")
async def scan_local_components(db: Session = Depends(get_db)):
    """Scan local cvlabkit components and register them in the store."""
    try:
        store = ComponentStore(db)
        count = store.scan_local_components()
        return success_response(
            {"registered": count},
            {"message": f"Registered {count} components from local filesystem"},
        )
    except Exception as e:
        return error_response("Failed to scan components", 500, {"error": str(e)})


# =============================================================================
# Experiment Manifest Endpoints (Reproducibility)
# =============================================================================


@router.get("/manifest/{experiment_uid}")
async def get_experiment_manifest(experiment_uid: str, db: Session = Depends(get_db)):
    """Get component versions used in an experiment."""
    try:
        store = ComponentStore(db)
        manifest = store.get_experiment_manifest(experiment_uid)
        if not manifest:
            raise HTTPException(status_code=404, detail=f"Manifest for '{experiment_uid}' not found")
        return success_response(manifest.model_dump(), {"message": "Manifest found"})
    except HTTPException:
        raise
    except Exception as e:
        return error_response("Failed to get manifest", 500, {"error": str(e)})


@router.post("/manifest/{experiment_uid}")
async def save_experiment_manifest(
    experiment_uid: str, components: Dict[str, str], db: Session = Depends(get_db)
):
    """Save component versions used in an experiment."""
    try:
        store = ComponentStore(db)
        store.save_experiment_manifest(experiment_uid, components)
        return success_response(
            {"experiment_uid": experiment_uid, "component_count": len(components)},
            {"message": "Manifest saved"},
        )
    except Exception as e:
        return error_response("Failed to save manifest", 500, {"error": str(e)})
