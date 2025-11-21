"""Component Discovery API for CVLab-Kit components"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..utils.responses import error_response, success_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/components", tags=["components"])


class ComponentInfo(BaseModel):
    """Component metadata"""

    name: str
    type: str  # 'model', 'dataset', 'transform', 'optimizer', 'loss', 'metric'
    path: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    examples: List[Dict[str, Any]] = []


class ComponentCategory(BaseModel):
    """Category with components"""

    category: str
    count: int
    components: List[ComponentInfo]


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
