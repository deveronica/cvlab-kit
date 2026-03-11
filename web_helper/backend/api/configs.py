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


class CreateConfigRequest(BaseModel):
    name: str
    content: str = "# New configuration\n"


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


@router.post("/create")
async def create_config(request: CreateConfigRequest):
    """Create a new configuration file."""
    import re

    try:
        # Validate name
        name = request.name.strip()
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise HTTPException(
                status_code=400,
                detail="Invalid name. Use letters, numbers, hyphens, and underscores only.",
            )

        # Ensure .yaml extension
        if not name.endswith(".yaml") and not name.endswith(".yml"):
            name = f"{name}.yaml"

        # Check path safety
        base_path = Path(CONFIG_DIR).resolve()
        config_path = (base_path / name).resolve()

        if not config_path.is_relative_to(base_path):
            raise HTTPException(
                status_code=400,
                detail="Access to paths outside the config directory is not allowed.",
            )

        # Check if file already exists
        if config_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Configuration file already exists: {name}",
            )

        # Create parent directories if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        config_path.write_text(request.content, encoding="utf-8")

        return success_response(
            {"path": name, "created": True},
            {"message": f"Configuration file created: {name}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            f"Failed to create configuration file: {request.name}",
            500,
            {"error": str(e)},
        )


class SaveConfigRequest(BaseModel):
    path: str
    content: dict | str


@router.post("/save")
async def save_config(request: SaveConfigRequest):
    """Save configuration file content."""
    try:
        full_path = get_config_path(request.path)
        
        # If content is a dict, convert to YAML string
        if isinstance(request.content, dict):
            # Custom dumper to keep YAML looking clean
            class IndentDumper(yaml.Dumper):
                def increase_indent(self, flow=False, indentless=False):
                    return super().increase_indent(flow, False)
            
            yaml_content = yaml.dump(
                request.content, 
                Dumper=IndentDumper,
                default_flow_style=False, 
                sort_keys=False,
                allow_unicode=True
            )
        else:
            yaml_content = request.content

        # Write to file
        full_path.write_text(yaml_content, encoding="utf-8")

        return success_response(
            {"path": request.path, "saved": True},
            {"message": f"Configuration saved successfully: {request.path}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            f"Failed to save configuration file: {request.path}",
            500,
            {"error": str(e)},
        )
