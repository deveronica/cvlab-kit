"""Unified file parsing utilities for CVLab-Kit output files."""

from pathlib import Path
from typing import Any, Dict, Optional


def parse_run_info(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse run information from CVLab-Kit output file path and name.

    This function handles the standard CVLab-Kit file naming conventions:
    - Metrics: {run_name}.csv
    - Config (input): {run_name}_config.yaml or {run_name}.config.yaml
    - Config (output): {run_name}.yaml
    - Checkpoint: {run_name}.pt, {run_name}_checkpoint.pt, or checkpoint_{epoch}.pt (legacy)

    Args:
        file_path: Path object for the file to parse

    Returns:
        Dictionary with 'run_name' and 'type' keys, or None if file should be skipped
    """
    run_info = {}
    stem = file_path.stem
    suffix = file_path.suffix

    # Skip temporary or hidden files
    if file_path.name.startswith(".") or file_path.name.endswith(".tmp"):
        return None

    # Handle different file types based on naming conventions

    # Configuration files (input)
    if "_config" in stem or ".config" in stem:
        # Examples: run_name_config.yaml, run_name.config.yaml
        run_name = stem.replace("_config", "").replace(".config", "")
        run_info["run_name"] = run_name
        run_info["type"] = "config"

    # CSV metrics files
    elif suffix == ".csv":
        # Standard format: run_name.csv
        run_info["run_name"] = stem
        run_info["type"] = "metrics"

    # Checkpoint files
    elif suffix in {".pt", ".pth"}:
        # Multiple formats:
        # - New format: run_name.pt or run_name_checkpoint.pt
        # - Legacy format: checkpoint_{epoch}.pt (cannot extract meaningful run_name)

        if stem.startswith("checkpoint_"):
            # Legacy checkpoint format - cannot reliably extract run_name
            # These files will need to be associated with runs through directory structure
            return None
        else:
            # New format with run_name
            run_name = stem.replace("_checkpoint", "").replace("_model", "")
            run_info["run_name"] = run_name
            run_info["type"] = "checkpoint"

    # YAML output config files
    elif suffix == ".yaml" and "config" not in stem:
        # Actual config used during run: run_name.yaml
        run_info["run_name"] = stem
        run_info["type"] = "output_config"

    else:
        # Unknown or non-run file (JSON settings, etc.)
        return None

    return run_info


def parse_project_from_path(file_path: Path) -> Optional[str]:
    """Extract project name from CVLab-Kit logs directory structure.

    Expected structure: logs/{project}/{files}

    Args:
        file_path: Path to the file

    Returns:
        Project name or None if path doesn't follow expected structure
    """
    parts = file_path.parts

    if "logs" in parts:
        logs_index = parts.index("logs")
        if logs_index + 1 < len(parts):
            return parts[logs_index + 1]

    return None
