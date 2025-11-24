"""Component store service for version management and synchronization.

Provides content-addressable storage for cvlabkit components (agents, models, etc.).
Each component version is identified by its xxhash3 content hash.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.component import (
    ComponentListItem,
    ComponentVersion,
    ComponentVersionResponse,
    ExperimentComponentManifest,
    ExperimentManifestResponse,
)
from .hash_utils import calculate_content_hash

logger = logging.getLogger(__name__)

# Valid component categories (matches cvlabkit/component/ structure)
VALID_CATEGORIES = {
    "agent",
    "model",
    "dataset",
    "dataloader",
    "transform",
    "optimizer",
    "loss",
    "metric",
    "scheduler",
    "sampler",
    "solver",
    "checkpoint",
    "logger",
}


def parse_component_path(path: str) -> tuple[str, str]:
    """Parse component path into category and name.

    Args:
        path: Component path like "agent/classification.py" or "model/resnet.py"

    Returns:
        Tuple of (category, name)

    Raises:
        ValueError: If path format is invalid
    """
    # Normalize path
    path = path.replace("\\", "/").strip("/")

    # Handle with or without .py extension
    if path.endswith(".py"):
        path = path[:-3]

    parts = path.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid component path: {path}. Expected 'category/name' format.")

    category, name = parts

    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category: {category}. Valid: {', '.join(sorted(VALID_CATEGORIES))}")

    # Validate name (Python module name)
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(f"Invalid component name: {name}. Must be valid Python identifier.")

    return category, name


def normalize_path(path: str) -> str:
    """Normalize component path to standard format.

    Args:
        path: Raw path like "agent/classification" or "agent/classification.py"

    Returns:
        Normalized path like "agent/classification.py"
    """
    category, name = parse_component_path(path)
    return f"{category}/{name}.py"


class ComponentStore:
    """Service for managing component versions."""

    def __init__(self, db: Session):
        self.db = db

    def get_all_components(self) -> list[ComponentListItem]:
        """Get list of all registered components with their active versions."""
        # Get distinct paths with their active version info
        results = (
            self.db.query(
                ComponentVersion.path,
                ComponentVersion.category,
                ComponentVersion.name,
                func.count(ComponentVersion.id).label("version_count"),
                func.max(ComponentVersion.created_at).label("updated_at"),
            )
            .group_by(ComponentVersion.path)
            .all()
        )

        items = []
        for row in results:
            # Get active hash for this path
            active = (
                self.db.query(ComponentVersion.hash)
                .filter(ComponentVersion.path == row.path, ComponentVersion.is_active == True)
                .first()
            )

            items.append(
                ComponentListItem(
                    path=row.path,
                    category=row.category,
                    name=row.name,
                    active_hash=active.hash if active else None,
                    version_count=row.version_count,
                    updated_at=row.updated_at,
                )
            )

        return items

    def get_components_by_category(self, category: str) -> list[ComponentListItem]:
        """Get components filtered by category."""
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        all_components = self.get_all_components()
        return [c for c in all_components if c.category == category]

    def get_component_versions(
        self, category: str, name: str, include_content: bool = False
    ) -> list[ComponentVersionResponse]:
        """Get all versions of a specific component."""
        path = f"{category}/{name}.py"

        query = self.db.query(ComponentVersion).filter(ComponentVersion.path == path)
        versions = query.order_by(ComponentVersion.created_at.desc()).all()

        return [
            ComponentVersionResponse(
                hash=v.hash,
                path=v.path,
                category=v.category,
                name=v.name,
                is_active=v.is_active,
                created_at=v.created_at,
                content=v.content if include_content else None,
            )
            for v in versions
        ]

    def get_version_by_hash(self, hash: str, include_content: bool = True) -> Optional[ComponentVersionResponse]:
        """Get specific version by hash."""
        version = self.db.query(ComponentVersion).filter(ComponentVersion.hash == hash).first()

        if not version:
            return None

        return ComponentVersionResponse(
            hash=version.hash,
            path=version.path,
            category=version.category,
            name=version.name,
            is_active=version.is_active,
            created_at=version.created_at,
            content=version.content if include_content else None,
        )

    def get_active_version(self, path: str) -> Optional[ComponentVersionResponse]:
        """Get current active version for a component path."""
        path = normalize_path(path)

        version = (
            self.db.query(ComponentVersion)
            .filter(ComponentVersion.path == path, ComponentVersion.is_active == True)
            .first()
        )

        if not version:
            return None

        return ComponentVersionResponse(
            hash=version.hash,
            path=version.path,
            category=version.category,
            name=version.name,
            is_active=version.is_active,
            created_at=version.created_at,
            content=version.content,
        )

    def upload_version(self, path: str, content: str, activate: bool = True) -> tuple[ComponentVersionResponse, bool]:
        """Upload a new component version.

        Args:
            path: Component path like "agent/classification.py"
            content: Python code content
            activate: Whether to activate this version

        Returns:
            Tuple of (ComponentVersionResponse, is_new)
        """
        path = normalize_path(path)
        category, name = parse_component_path(path)

        # Calculate content hash
        content_hash = calculate_content_hash(content.encode("utf-8"))

        # Check if this exact version already exists
        existing = self.db.query(ComponentVersion).filter(ComponentVersion.hash == content_hash).first()

        if existing:
            # Version already exists
            is_new = False
            if activate and not existing.is_active:
                self._activate_version(existing)
            return (
                ComponentVersionResponse(
                    hash=existing.hash,
                    path=existing.path,
                    category=existing.category,
                    name=existing.name,
                    is_active=existing.is_active,
                    created_at=existing.created_at,
                ),
                is_new,
            )

        # Create new version
        new_version = ComponentVersion(
            hash=content_hash,
            path=path,
            category=category,
            name=name,
            content=content,
            is_active=False,
            created_at=datetime.utcnow(),
        )
        self.db.add(new_version)

        if activate:
            self._activate_version(new_version)

        self.db.commit()

        return (
            ComponentVersionResponse(
                hash=new_version.hash,
                path=new_version.path,
                category=new_version.category,
                name=new_version.name,
                is_active=new_version.is_active,
                created_at=new_version.created_at,
            ),
            True,
        )

    def activate_version(self, hash: str) -> Optional[ComponentVersionResponse]:
        """Activate a specific version (rollback)."""
        version = self.db.query(ComponentVersion).filter(ComponentVersion.hash == hash).first()

        if not version:
            return None

        self._activate_version(version)
        self.db.commit()

        return ComponentVersionResponse(
            hash=version.hash,
            path=version.path,
            category=version.category,
            name=version.name,
            is_active=True,
            created_at=version.created_at,
        )

    def _activate_version(self, version: ComponentVersion):
        """Internal: Activate a version and deactivate others for same path."""
        # Deactivate all versions for this path
        self.db.query(ComponentVersion).filter(
            ComponentVersion.path == version.path, ComponentVersion.is_active == True
        ).update({"is_active": False})

        # Activate this version
        version.is_active = True

    def get_active_hashes(self, paths: list[str]) -> dict[str, str]:
        """Get active hashes for multiple paths.

        Args:
            paths: List of component paths

        Returns:
            Dictionary of {path: hash} for active versions
        """
        result = {}
        for path in paths:
            try:
                path = normalize_path(path)
                version = self.get_active_version(path)
                if version:
                    result[path] = version.hash
            except ValueError:
                continue
        return result

    # =========================================================================
    # Experiment Manifest (Reproducibility)
    # =========================================================================

    def save_experiment_manifest(self, experiment_uid: str, components: dict[str, str]):
        """Save component versions used in an experiment.

        Args:
            experiment_uid: Experiment identifier
            components: Dictionary of {path: hash}
        """
        # Delete existing manifest for this experiment
        self.db.query(ExperimentComponentManifest).filter(
            ExperimentComponentManifest.experiment_uid == experiment_uid
        ).delete()

        # Create new manifest entries
        now = datetime.utcnow()
        for path, hash in components.items():
            entry = ExperimentComponentManifest(
                experiment_uid=experiment_uid,
                component_path=path,
                component_hash=hash,
                created_at=now,
            )
            self.db.add(entry)

        self.db.commit()

    def get_experiment_manifest(self, experiment_uid: str) -> Optional[ExperimentManifestResponse]:
        """Get component manifest for an experiment."""
        entries = (
            self.db.query(ExperimentComponentManifest)
            .filter(ExperimentComponentManifest.experiment_uid == experiment_uid)
            .all()
        )

        if not entries:
            return None

        components = {e.component_path: e.component_hash for e in entries}

        return ExperimentManifestResponse(
            experiment_uid=experiment_uid,
            components=components,
            created_at=entries[0].created_at,
        )

    # =========================================================================
    # Sync Helpers
    # =========================================================================

    def get_components_to_sync(
        self, required_paths: list[str], local_hashes: dict[str, str]
    ) -> tuple[list[str], dict[str, ComponentVersionResponse]]:
        """Determine which components need to be synced.

        Args:
            required_paths: List of component paths needed
            local_hashes: Current hashes on worker {path: hash}

        Returns:
            Tuple of (hashes_to_download, component_info)
        """
        to_download = []
        components = {}

        for path in required_paths:
            try:
                path = normalize_path(path)
                active = self.get_active_version(path)

                if not active:
                    logger.warning(f"No active version for {path}")
                    continue

                # Check if local hash matches
                local_hash = local_hashes.get(path)
                if local_hash != active.hash:
                    to_download.append(active.hash)

                components[active.hash] = active

            except ValueError as e:
                logger.warning(f"Invalid component path {path}: {e}")
                continue

        return to_download, components

    def scan_local_components(self, base_path: Path = None) -> int:
        """Scan local cvlabkit components and register them.

        Args:
            base_path: Base path to cvlabkit (default: ./cvlabkit)

        Returns:
            Number of components registered
        """
        if base_path is None:
            base_path = Path("cvlabkit")

        count = 0

        # Scan agents
        agent_path = base_path / "agent"
        if agent_path.exists():
            for py_file in agent_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                    self.upload_version(f"agent/{py_file.stem}.py", content, activate=True)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to register {py_file}: {e}")

        # Scan component categories
        component_path = base_path / "component"
        if component_path.exists():
            for category_dir in component_path.iterdir():
                if not category_dir.is_dir():
                    continue
                category = category_dir.name
                if category not in VALID_CATEGORIES or category == "base":
                    continue

                for py_file in category_dir.glob("*.py"):
                    if py_file.name.startswith("_"):
                        continue
                    try:
                        content = py_file.read_text(encoding="utf-8")
                        self.upload_version(f"{category}/{py_file.stem}.py", content, activate=True)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to register {py_file}: {e}")

        return count
