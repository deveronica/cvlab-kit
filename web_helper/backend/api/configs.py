"""API endpoints for managing configuration files."""

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..utils.responses import error_response, success_response

router = APIRouter(prefix="/configs", tags=["configs"])

CONFIG_DIR = "config"
COMPONENT_DIR = Path("cvlabkit/component")


class ValidateConfigRequest(BaseModel):
    config: str


def get_config_path(relative_path: str) -> Path:
    """Constructs the full path to a config file and ensures it's within the config directory."""
    base_path = Path(CONFIG_DIR).resolve()
    config_path = (base_path / relative_path).resolve()

    if not config_path.is_relative_to(base_path):
        raise HTTPException(
            status_code=400,
            detail="Access to paths outside the config directory is not allowed.",
        )

    if not config_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Configuration file not found: {relative_path}"
        )

    return config_path


@router.get("/")
@router.get("")
async def list_configs():
    """List all available YAML configuration files."""
    try:
        config_files = []
        base_path = Path(CONFIG_DIR)
        for path in base_path.rglob("*.yaml"):
            relative_path = path.relative_to(base_path)
            config_files.append(str(relative_path))

        return success_response(
            config_files, {"message": "Configuration files listed successfully"}
        )
    except Exception as e:
        return error_response(
            "Failed to list configuration files", 500, {"error": str(e)}
        )


@router.get("/{config_path:path}")
async def get_config_content(config_path: str):
    """Get the content of a specific configuration file."""
    try:
        full_path = get_config_path(config_path)
        content = full_path.read_text(encoding="utf-8")
        return success_response(
            {"path": config_path, "content": content},
            {"message": "Configuration content retrieved successfully"},
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        return error_response(
            f"Failed to read configuration file: {config_path}", 500, {"error": str(e)}
        )


@router.post("/validate")
async def validate_config(request: ValidateConfigRequest):
    """Validate the entire YAML configuration."""
    try:
        # Use FullLoader to support Python tags like !!python/tuple
        config_data = yaml.load(request.config, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        return error_response("Invalid YAML format", 400, {"error": str(e)})

    errors = []

    # List of component types that can be defined in the config
    component_types = [
        "model",
        "dataset",
        "optimizer",
        "loss",
        "scheduler",
        "transform",
        "dataloader",
        "agent",
    ]

    for comp_type in component_types:
        if comp_type in config_data:
            comp_name = config_data[comp_type]
            # Handle nested component definitions
            if isinstance(comp_name, dict):
                for logical_name, actual_name in comp_name.items():
                    if not (COMPONENT_DIR / comp_type / f"{actual_name}.py").exists():
                        errors.append(
                            f"Component not found for type '{comp_type}' (name: '{logical_name}'): {actual_name}.py"
                        )
            # Handle simple component definition
            elif isinstance(comp_name, str):
                # Handle case like `cifar10(split=train)`
                if "(" in comp_name:
                    comp_name = comp_name.split("(")[0]
                if not (COMPONENT_DIR / comp_type / f"{comp_name}.py").exists():
                    errors.append(
                        f"Component not found for type '{comp_type}': {comp_name}.py"
                    )
            else:
                errors.append(f"Invalid format for component type '{comp_type}'")

    if errors:
        return error_response(
            "Configuration validation failed", 400, {"errors": errors}
        )

    return success_response({"valid": True}, {"message": "Configuration is valid"})
