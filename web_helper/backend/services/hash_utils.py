"""Hash utilities for file integrity and code version tracking.

Uses xxhash3 for fast, non-cryptographic hashing.
Provides functions for:
- Single file hashing
- Directory hashing (for code integrity)
- Streaming hash for large files
"""

import subprocess
from pathlib import Path
from typing import Optional

import xxhash


def calculate_file_hash(file_path: Path) -> Optional[str]:
    """Calculate xxhash3 of a single file.

    Args:
        file_path: Path to the file

    Returns:
        Hex digest of xxhash3_64, or None if file doesn't exist
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path, "rb") as f:
            return xxhash.xxh3_64(f.read()).hexdigest()
    except Exception:
        return None


def calculate_directory_hash(
    directory: Path,
    patterns: list[str] = None,
    exclude_patterns: list[str] = None,
) -> Optional[str]:
    """Calculate combined hash of files in a directory.

    Useful for tracking code changes across multiple files.

    Args:
        directory: Root directory to hash
        patterns: Glob patterns to include (default: ["**/*.py"])
        exclude_patterns: Glob patterns to exclude (default: ["**/legacy/**", "**/__pycache__/**"])

    Returns:
        Combined xxhash3 of all matching files (sorted by path for consistency)
    """
    if not directory.exists():
        return None

    patterns = patterns or ["**/*.py"]
    exclude_patterns = exclude_patterns or ["**/legacy/**", "**/__pycache__/**", "**/test*/**"]

    # Collect all matching files
    files = set()
    for pattern in patterns:
        files.update(directory.glob(pattern))

    # Remove excluded files
    for exclude in exclude_patterns:
        excluded = set(directory.glob(exclude))
        files -= excluded

    if not files:
        return None

    # Sort for consistent ordering
    sorted_files = sorted(files)

    # Calculate combined hash
    hasher = xxhash.xxh3_64()
    for file_path in sorted_files:
        try:
            # Include relative path in hash for structure sensitivity
            rel_path = file_path.relative_to(directory)
            hasher.update(str(rel_path).encode())
            hasher.update(file_path.read_bytes())
        except Exception:
            continue

    return hasher.hexdigest()


def calculate_content_hash(content: bytes) -> str:
    """Calculate xxhash3 of raw bytes.

    Args:
        content: Bytes to hash

    Returns:
        Hex digest of xxhash3_64
    """
    return xxhash.xxh3_64(content).hexdigest()


def get_git_hash(repo_path: Path = None) -> Optional[str]:
    """Get current git commit hash.

    Args:
        repo_path: Path to git repository (default: current directory)

    Returns:
        Short git commit hash (7 chars), or None if not a git repo
    """
    try:
        cwd = str(repo_path) if repo_path else None
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def is_git_dirty(repo_path: Path = None) -> bool:
    """Check if git working directory has uncommitted changes.

    Args:
        repo_path: Path to git repository (default: current directory)

    Returns:
        True if there are uncommitted changes
    """
    try:
        cwd = str(repo_path) if repo_path else None
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return False


def get_git_branch(repo_path: Path = None) -> Optional[str]:
    """Get current git branch name.

    Args:
        repo_path: Path to git repository (default: current directory)

    Returns:
        Branch name, or None if not a git repo or detached HEAD
    """
    try:
        cwd = str(repo_path) if repo_path else None
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch != "HEAD" else None
    except Exception:
        pass
    return None


def get_component_versions_from_config(config_path: Path) -> dict[str, str]:
    """Get component versions (hashes) for all components used in a config.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary of {file_path: xxhash3} for agent and all used components
    """
    import yaml

    versions = {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception:
        return versions

    # Agent file
    agent_name = config.get("agent", "")
    if agent_name:
        agent_path = Path(f"cvlabkit/agent/{agent_name}.py")
        if agent_path.exists():
            versions[str(agent_path)] = calculate_file_hash(agent_path) or ""

    # Component categories to check
    categories = [
        "model", "dataset", "dataloader", "transform",
        "optimizer", "loss", "metric", "scheduler",
    ]

    for category in categories:
        if category not in config:
            continue

        value = config[category]
        names = []

        if isinstance(value, str):
            # Simple value or pipeline
            names = [n.strip().split("(")[0] for n in value.split("|")]
        elif isinstance(value, dict):
            # Nested like transform: {weak: ..., strong: ...}
            for sub_value in value.values():
                if isinstance(sub_value, str):
                    names.extend(n.strip().split("(")[0] for n in sub_value.split("|"))

        for name in names:
            if not name:
                continue
            comp_path = Path(f"cvlabkit/component/{category}/{name}.py")
            if comp_path.exists():
                versions[str(comp_path)] = calculate_file_hash(comp_path) or ""

    return versions


def get_code_version(config_path: Path = None) -> dict:
    """Get comprehensive code version information.

    Collects git hash, branch, dirty status, and file hashes for
    cvlabkit core files. Used by Worker for reproducibility tracking.

    Args:
        config_path: Optional config path to include component versions

    Returns:
        Dictionary with code version information
    """
    from pathlib import Path as P

    # Get git information
    git_hash = get_git_hash() or "unknown"
    git_dirty = is_git_dirty()
    branch = get_git_branch()

    # Calculate hash of cvlabkit core files
    cvlabkit_path = P("cvlabkit")
    files_hash = calculate_directory_hash(
        cvlabkit_path,
        patterns=["**/*.py"],
        exclude_patterns=["**/legacy/**", "**/__pycache__/**", "**/test*/**"],
    ) or "unknown"

    # Calculate hash of uv.lock for dependency tracking
    uv_lock_path = P("uv.lock")
    uv_lock_hash = calculate_file_hash(uv_lock_path) or "unknown"

    result = {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "branch": branch,
        "files_hash": files_hash,
        "uv_lock_hash": uv_lock_hash,
    }

    # Add component versions if config provided
    if config_path:
        result["component_versions"] = get_component_versions_from_config(config_path)

    return result
