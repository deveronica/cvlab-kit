"""API endpoints for artifact file management."""

import base64
import mimetypes
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/artifacts")

# Security configuration
ALLOWED_EXTENSIONS = {
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".webp",
    # Data
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    # Models
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    # Logs
    ".log",
    # Archives (preview only, no extraction)
    ".zip",
    ".tar",
    ".gz",
}

MAX_PREVIEW_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TEXT_PREVIEW_SIZE = 1 * 1024 * 1024  # 1MB for text files

# Artifact base path
ARTIFACTS_BASE = Path("./logs")


class FileNode(BaseModel):
    """File/directory node in tree."""

    name: str
    path: str
    type: str  # 'file' | 'directory'
    size: Optional[int] = None
    modified: Optional[float] = None
    extension: Optional[str] = None
    mime_type: Optional[str] = None
    children: Optional[List["FileNode"]] = None


class FilePreview(BaseModel):
    """File preview data."""

    path: str
    name: str
    size: int
    mime_type: str
    extension: str
    preview_type: str  # 'image' | 'text' | 'json' | 'csv' | 'binary'
    content: Optional[str] = None  # Base64 for images, text for others
    metadata: Optional[dict] = None


def sanitize_path(path: str) -> Path:
    """Sanitize and validate file path.

    Security checks:
    - No path traversal (..)
    - Must be under ARTIFACTS_BASE
    - No hidden files
    """
    try:
        # Convert to Path and resolve
        full_path = (ARTIFACTS_BASE / path).resolve()

        # Check if path is under base directory
        if not str(full_path).startswith(str(ARTIFACTS_BASE.resolve())):
            raise ValueError("Path traversal detected")

        # Check for hidden files (starting with .)
        if any(part.startswith(".") for part in full_path.parts):
            raise ValueError("Hidden files not allowed")

        return full_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")


def is_allowed_file(file_path: Path) -> bool:
    """Check if file extension is allowed."""
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def get_file_tree(root_path: Path, base_path: Path) -> FileNode:
    """Recursively build file tree.

    Args:
        root_path: Current directory to scan
        base_path: Base artifacts directory (for relative paths)
    """
    if not root_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    if root_path.is_file():
        stat = root_path.stat()
        mime_type, _ = mimetypes.guess_type(str(root_path))

        return FileNode(
            name=root_path.name,
            path=str(root_path.relative_to(base_path)),
            type="file",
            size=stat.st_size,
            modified=stat.st_mtime,
            extension=root_path.suffix.lower(),
            mime_type=mime_type,
        )

    # Directory
    children = []
    try:
        for item in sorted(root_path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
            # Skip hidden files
            if item.name.startswith("."):
                continue

            # Recursively process
            try:
                child_node = get_file_tree(item, base_path)
                children.append(child_node)
            except Exception:
                # Skip files that can't be accessed
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    stat = root_path.stat()

    return FileNode(
        name=root_path.name if root_path != base_path else "Artifacts",
        path=str(root_path.relative_to(base_path)) if root_path != base_path else "",
        type="directory",
        modified=stat.st_mtime,
        children=children,
    )


@router.get("/tree", response_model=FileNode)
async def get_artifacts_tree(
    project: Optional[str] = Query(None, description="Filter by project"),
):
    """Get artifacts directory tree structure.

    Returns hierarchical tree of all artifact files and directories.
    """
    try:
        if project:
            root = ARTIFACTS_BASE / project
            if not root.exists():
                raise HTTPException(
                    status_code=404, detail=f"Project '{project}' not found"
                )
        else:
            root = ARTIFACTS_BASE

        tree = get_file_tree(root, ARTIFACTS_BASE)
        return tree

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building tree: {str(e)}")


@router.get("/preview", response_model=FilePreview)
async def preview_file(
    path: str = Query(..., description="Relative path to file"),
):
    """Preview file content.

    Supports:
    - Images: Returns base64-encoded data
    - Text files: Returns text content
    - JSON/YAML: Returns parsed structure
    - CSV: Returns parsed rows (limited)
    - Binary: Returns metadata only
    """
    file_path = sanitize_path(path)

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if not is_allowed_file(file_path):
        raise HTTPException(
            status_code=403, detail=f"File type '{file_path.suffix}' not allowed"
        )

    stat = file_path.stat()
    file_size = stat.st_size

    if file_size > MAX_PREVIEW_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large for preview (max {MAX_PREVIEW_SIZE / 1024 / 1024}MB)",
        )

    mime_type, _ = mimetypes.guess_type(str(file_path))
    extension = file_path.suffix.lower()

    # Determine preview type
    preview_type = "binary"
    content = None
    metadata = {}

    # Image files
    if extension in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        preview_type = "image"
        with open(file_path, "rb") as f:
            image_data = f.read()
            content = base64.b64encode(image_data).decode("utf-8")
        metadata = {
            "size_kb": round(file_size / 1024, 2),
            "mime_type": mime_type,
        }

    # Text files
    elif extension in {".txt", ".log", ".md", ".yaml", ".yml"}:
        preview_type = "text"
        if file_size > MAX_TEXT_PREVIEW_SIZE:
            # Read first 1MB only
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read(MAX_TEXT_PREVIEW_SIZE)
                content += "\n\n... (truncated)"
        else:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        metadata = {
            "lines": content.count("\n") + 1 if content else 0,
            "size_kb": round(file_size / 1024, 2),
        }

    # JSON files
    elif extension == ".json":
        preview_type = "json"
        import json

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                content = json.dumps(data, indent=2)
            metadata = {
                "valid_json": True,
                "size_kb": round(file_size / 1024, 2),
            }
        except json.JSONDecodeError as e:
            preview_type = "text"
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            metadata = {
                "valid_json": False,
                "error": str(e),
            }

    # CSV files
    elif extension == ".csv":
        preview_type = "csv"
        import csv

        try:
            rows = []
            with open(file_path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= 100:  # Limit preview to 100 rows
                        break
                    rows.append(row)
            content = "\n".join([",".join(row) for row in rows])
            if file_size > MAX_TEXT_PREVIEW_SIZE:
                content += "\n\n... (showing first 100 rows)"
            metadata = {
                "rows_preview": len(rows),
                "size_kb": round(file_size / 1024, 2),
            }
        except Exception as e:
            preview_type = "text"
            with open(file_path, encoding="utf-8") as f:
                content = f.read(MAX_TEXT_PREVIEW_SIZE)
            metadata = {
                "csv_error": str(e),
            }

    # Model files
    elif extension in {".pt", ".pth", ".ckpt", ".safetensors"}:
        preview_type = "binary"
        metadata = {
            "type": "PyTorch Model",
            "size_mb": round(file_size / 1024 / 1024, 2),
            "extension": extension,
        }

    return FilePreview(
        path=path,
        name=file_path.name,
        size=file_size,
        mime_type=mime_type or "application/octet-stream",
        extension=extension,
        preview_type=preview_type,
        content=content,
        metadata=metadata,
    )


@router.get("/download/{path:path}")
async def download_file(path: str):
    """Download artifact file.

    Security: Read-only access with path sanitization.
    """
    file_path = sanitize_path(path)

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if not is_allowed_file(file_path):
        raise HTTPException(
            status_code=403, detail=f"File type '{file_path.suffix}' not allowed"
        )

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )
