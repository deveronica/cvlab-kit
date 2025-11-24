"""Component Manager for Worker-side component synchronization.

Handles:
- Parsing config to extract required agent and components
- Checking local component versions against server
- Downloading missing or outdated components
- Extracting and installing components
- Sync strategy for handling local vs server version conflicts
"""

import io
import logging
import tarfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import httpx
import xxhash
import yaml

logger = logging.getLogger(__name__)


class SyncStrategy(str, Enum):
    """Sync strategy options for handling version conflicts."""

    SERVER_AUTHORITATIVE = "server_authoritative"  # Always use server active version
    LOCAL_PRIORITY = "local_priority"  # Upload local changes to server
    INTERACTIVE = "interactive"  # Ask user for decision
    DRY_RUN = "dry_run"  # Only report, don't sync


@dataclass
class SyncConflict:
    """Represents a sync conflict between local and server versions."""

    path: str
    local_hash: Optional[str]
    server_hash: Optional[str]
    conflict_type: str  # "local_newer", "server_newer", "local_only", "server_only"
    local_content: Optional[str] = None
    resolved: bool = False
    resolution: Optional[str] = None  # "use_local", "use_server", "skip"


@dataclass
class SyncResult:
    """Result of a sync operation."""

    synced: List[str] = field(default_factory=list)
    uploaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    conflicts: List[SyncConflict] = field(default_factory=list)
    component_hashes: Dict[str, str] = field(default_factory=dict)


class ComponentManager:
    """Manages component synchronization between worker and server."""

    def __init__(
        self,
        server_url: str,
        base_path: Path = Path("."),
        api_key: Optional[str] = None,
        sync_strategy: SyncStrategy = SyncStrategy.INTERACTIVE,
        conflict_callback: Optional[Callable[[SyncConflict], str]] = None,
    ):
        """Initialize component manager.

        Args:
            server_url: Base URL of the central server
            base_path: Base path for component installation
            api_key: Optional API key for authentication
            sync_strategy: Strategy for handling version conflicts
            conflict_callback: Callback for interactive conflict resolution
        """
        self.server_url = server_url.rstrip("/")
        self.base_path = base_path
        self.api_key = api_key
        self.sync_strategy = sync_strategy
        self.conflict_callback = conflict_callback
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialized HTTP client."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._client = httpx.Client(
                base_url=self.server_url,
                headers=headers,
                timeout=60.0,
            )
        return self._client

    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def parse_config(self, config_path: Path) -> Dict[str, Any]:
        """Parse YAML config and extract agent and components.

        Args:
            config_path: Path to YAML config file

        Returns:
            Dict with 'agent' and 'components' keys
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        result = {
            "agent": config.get("agent", ""),
            "components": {},
        }

        # Extract component references from config
        component_categories = [
            "model",
            "dataset",
            "dataloader",
            "transform",
            "optimizer",
            "loss",
            "metric",
            "scheduler",
        ]

        for category in component_categories:
            if category in config:
                value = config[category]
                if isinstance(value, str):
                    result["components"][category] = value
                elif isinstance(value, dict):
                    # Handle nested components like transform: {weak: ..., strong: ...}
                    names = []
                    for sub_value in value.values():
                        if isinstance(sub_value, str):
                            # Parse pipeline syntax (e.g., "resize | normalize")
                            names.extend(n.strip() for n in sub_value.split("|"))
                    if names:
                        result["components"][category] = ",".join(names)

        return result

    def calculate_local_hash(self, file_path: Path) -> Optional[str]:
        """Calculate xxhash3 of a local file.

        Args:
            file_path: Path to file

        Returns:
            Hash string or None if file doesn't exist
        """
        full_path = self.base_path / file_path
        if not full_path.exists():
            return None
        try:
            with open(full_path, "rb") as f:
                return xxhash.xxh3_64(f.read()).hexdigest()
        except Exception:
            return None

    def get_server_bundle_info(
        self, agent: str, components: Dict[str, str]
    ) -> Dict[str, Any]:
        """Get bundle info from server.

        Args:
            agent: Agent name
            components: Dict of category -> component names

        Returns:
            Bundle info with files, sizes, and hashes
        """
        try:
            response = self.client.post(
                "/api/components/bundle/info",
                json={"agent": agent, "components": components},
            )
            response.raise_for_status()
            return response.json().get("data", {})
        except Exception as e:
            logger.error(f"Failed to get bundle info: {e}")
            return {}

    def check_sync_status(
        self, agent: str, components: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Check which components need to be synced.

        Args:
            agent: Agent name
            components: Dict of category -> component names

        Returns:
            Dict with 'missing', 'outdated', and 'up_to_date' lists
        """
        result = {"missing": [], "outdated": [], "up_to_date": []}

        bundle_info = self.get_server_bundle_info(agent, components)
        if not bundle_info:
            return result

        for file_info in bundle_info.get("files", []):
            file_path = Path(file_info["path"])
            server_hash = file_info.get("hash")
            local_hash = self.calculate_local_hash(file_path)

            file_key = f"{file_info['type']}/{file_info['name']}"

            if local_hash is None:
                result["missing"].append(file_key)
            elif local_hash != server_hash:
                result["outdated"].append(file_key)
            else:
                result["up_to_date"].append(file_key)

        return result

    def download_component(
        self, category: str, name: str, target_path: Optional[Path] = None
    ) -> bool:
        """Download a single component from server.

        Args:
            category: Component category (model, loss, etc.) or 'agent'
            name: Component name
            target_path: Optional custom target path

        Returns:
            True if successful
        """
        try:
            if category == "agent":
                url = f"/api/components/agent/{name}/source"
                default_path = Path(f"cvlabkit/agent/{name}.py")
            else:
                url = f"/api/components/component/{category}/{name}/source"
                default_path = Path(f"cvlabkit/component/{category}/{name}.py")

            response = self.client.get(url)
            response.raise_for_status()

            # Verify hash if provided
            server_hash = response.headers.get("X-Content-Hash")
            content = response.content
            if server_hash:
                local_hash = xxhash.xxh3_64(content).hexdigest()
                if local_hash != server_hash:
                    logger.error(f"Hash mismatch for {category}/{name}")
                    return False

            # Write file
            file_path = self.base_path / (target_path or default_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)

            logger.info(f"Downloaded {category}/{name} to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {category}/{name}: {e}")
            return False

    def download_bundle(self, agent: str, components: Dict[str, str]) -> bool:
        """Download and extract component bundle from server.

        Args:
            agent: Agent name
            components: Dict of category -> component names

        Returns:
            True if successful
        """
        try:
            response = self.client.post(
                "/api/components/bundle/download",
                json={"agent": agent, "components": components},
            )
            response.raise_for_status()

            # Extract tarball
            buffer = io.BytesIO(response.content)
            with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
                tar.extractall(path=self.base_path)

            logger.info(f"Extracted bundle for agent '{agent}'")
            return True

        except Exception as e:
            logger.error(f"Failed to download bundle: {e}")
            return False

    def sync_from_config(self, config_path: Path, force: bool = False) -> Dict[str, Any]:
        """Sync all required components for a config.

        Args:
            config_path: Path to YAML config
            force: Force re-download even if hashes match

        Returns:
            Sync result with status for each component
        """
        result = {
            "synced": [],
            "skipped": [],
            "failed": [],
        }

        # Parse config
        parsed = self.parse_config(config_path)
        agent = parsed["agent"]
        components = parsed["components"]

        if not agent:
            logger.warning("No agent specified in config")
            return result

        # Check what needs syncing
        status = self.check_sync_status(agent, components)

        if force:
            # Download everything
            to_sync = (
                status["missing"] + status["outdated"] + status["up_to_date"]
            )
        else:
            to_sync = status["missing"] + status["outdated"]
            result["skipped"] = status["up_to_date"]

        if not to_sync:
            logger.info("All components are up to date")
            return result

        # Download each component individually for better error handling
        for item in to_sync:
            category, name = item.split("/", 1)
            if self.download_component(category, name):
                result["synced"].append(item)
            else:
                result["failed"].append(item)

        return result

    def get_component_versions(
        self, agent: str, components: Dict[str, str]
    ) -> Dict[str, str]:
        """Get current local versions (hashes) of components.

        Args:
            agent: Agent name
            components: Dict of category -> component names

        Returns:
            Dict of file_path -> hash
        """
        versions = {}

        bundle_info = self.get_server_bundle_info(agent, components)
        for file_info in bundle_info.get("files", []):
            file_path = file_info["path"]
            local_hash = self.calculate_local_hash(Path(file_path))
            if local_hash:
                versions[file_path] = local_hash

        return versions

    # =========================================================================
    # Version Management Methods (for DB-backed component store)
    # =========================================================================

    def get_required_paths_from_config(self, config_path: Path) -> List[str]:
        """Extract required component paths from config.

        Args:
            config_path: Path to YAML config file

        Returns:
            List of component paths like ["agent/classification.py", "model/resnet.py"]
        """
        parsed = self.parse_config(config_path)
        paths = []

        # Agent
        agent = parsed.get("agent", "")
        if agent:
            paths.append(f"agent/{agent}.py")

        # Components
        for category, value in parsed.get("components", {}).items():
            # Parse comma-separated and pipeline syntax
            names = []
            for part in value.split(","):
                for name in part.split("|"):
                    name = name.strip().split("(")[0].strip()
                    if name:
                        names.append(name)

            for name in names:
                paths.append(f"{category}/{name}.py")

        return paths

    def get_local_hashes(self, paths: List[str]) -> Dict[str, str]:
        """Get local file hashes for given paths.

        Args:
            paths: List of component paths

        Returns:
            Dict of path -> hash (only for existing files)
        """
        result = {}
        for path in paths:
            # Construct full local path
            if path.startswith("agent/"):
                full_path = self.base_path / "cvlabkit" / path
            else:
                full_path = self.base_path / "cvlabkit" / "component" / path

            hash_val = self.calculate_local_hash(full_path)
            if hash_val:
                result[path] = hash_val
        return result

    def sync_from_version_store(
        self, config_path: Path, force: bool = False
    ) -> Dict[str, Any]:
        """Sync components using version store (DB-backed).

        This uses the new /api/components/versions/* endpoints for
        content-addressable storage with version history.

        Args:
            config_path: Path to YAML config
            force: Force re-download even if hashes match

        Returns:
            Sync result with status
        """
        result = {
            "synced": [],
            "skipped": [],
            "failed": [],
            "component_hashes": {},  # For experiment manifest
        }

        # Get required paths
        paths = self.get_required_paths_from_config(config_path)
        if not paths:
            logger.warning("No components found in config")
            return result

        # Get local hashes
        local_hashes = {} if force else self.get_local_hashes(paths)

        # Query server for active versions
        for path in paths:
            try:
                # Get active version from server
                category, name = path.split("/")
                name = name.replace(".py", "")

                response = self.client.get(
                    f"/api/components/versions/{category}/{name}",
                    params={"include_content": False},
                )

                if response.status_code == 404:
                    logger.warning(f"Component {path} not found in version store")
                    result["failed"].append(path)
                    continue

                response.raise_for_status()
                versions = response.json().get("data", [])

                # Find active version
                active = next((v for v in versions if v.get("is_active")), None)
                if not active:
                    logger.warning(f"No active version for {path}")
                    result["failed"].append(path)
                    continue

                server_hash = active["hash"]
                result["component_hashes"][path] = server_hash

                # Check if sync needed
                local_hash = local_hashes.get(path)
                if local_hash == server_hash and not force:
                    result["skipped"].append(path)
                    continue

                # Download the version
                if self._download_version(path, server_hash):
                    result["synced"].append(path)
                else:
                    result["failed"].append(path)

            except Exception as e:
                logger.error(f"Failed to sync {path}: {e}")
                result["failed"].append(path)

        return result

    def _download_version(self, path: str, hash: str) -> bool:
        """Download specific version by hash.

        Args:
            path: Component path
            hash: Content hash to download

        Returns:
            True if successful
        """
        try:
            response = self.client.get(f"/api/components/versions/hash/{hash}")
            response.raise_for_status()

            data = response.json().get("data", {})
            content = data.get("content", "")

            if not content:
                logger.error(f"Empty content for {path}")
                return False

            # Verify hash
            local_hash = xxhash.xxh3_64(content.encode("utf-8")).hexdigest()
            if local_hash != hash:
                logger.error(f"Hash mismatch for {path}: expected {hash}, got {local_hash}")
                return False

            # Write file
            if path.startswith("agent/"):
                file_path = self.base_path / "cvlabkit" / path
            else:
                file_path = self.base_path / "cvlabkit" / "component" / path

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            logger.info(f"Downloaded {path} (hash: {hash[:8]}...)")
            return True

        except Exception as e:
            logger.error(f"Failed to download version {hash} for {path}: {e}")
            return False

    def save_experiment_manifest(
        self, experiment_uid: str, component_hashes: Dict[str, str]
    ) -> bool:
        """Save component manifest for experiment reproducibility.

        Args:
            experiment_uid: Experiment identifier
            component_hashes: Dict of path -> hash

        Returns:
            True if successful
        """
        try:
            response = self.client.post(
                f"/api/components/manifest/{experiment_uid}",
                json=component_hashes,
            )
            response.raise_for_status()
            logger.info(f"Saved manifest for experiment {experiment_uid}")
            return True
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            return False

    def restore_from_manifest(self, experiment_uid: str) -> Dict[str, Any]:
        """Restore components from experiment manifest.

        Args:
            experiment_uid: Experiment identifier

        Returns:
            Sync result
        """
        result = {"synced": [], "failed": []}

        try:
            response = self.client.get(f"/api/components/manifest/{experiment_uid}")
            if response.status_code == 404:
                logger.error(f"Manifest not found for {experiment_uid}")
                return result

            response.raise_for_status()
            manifest = response.json().get("data", {})
            components = manifest.get("components", {})

            for path, hash in components.items():
                if self._download_version(path, hash):
                    result["synced"].append(path)
                else:
                    result["failed"].append(path)

        except Exception as e:
            logger.error(f"Failed to restore from manifest: {e}")

        return result

    # =========================================================================
    # Smart Sync with Strategy Support
    # =========================================================================

    def _get_local_content(self, path: str) -> Optional[str]:
        """Get local file content."""
        if path.startswith("agent/"):
            file_path = self.base_path / "cvlabkit" / path
        else:
            file_path = self.base_path / "cvlabkit" / "component" / path

        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return None

    def _upload_local_version(self, path: str, content: str) -> Optional[str]:
        """Upload local version to server.

        Returns:
            New hash if successful, None otherwise
        """
        try:
            response = self.client.post(
                "/api/components/versions/upload",
                json={"path": path, "content": content, "activate": True},
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            return data.get("hash")
        except Exception as e:
            logger.error(f"Failed to upload {path}: {e}")
            return None

    def _check_hash_exists_on_server(self, hash: str) -> bool:
        """Check if a hash exists on the server (any version)."""
        try:
            response = self.client.get(f"/api/components/versions/hash/{hash}")
            return response.status_code == 200
        except Exception:
            return False

    def sync_with_strategy(
        self, config_path: Path, strategy: Optional[SyncStrategy] = None
    ) -> SyncResult:
        """Sync components using configured strategy.

        This is the main entry point for syncing with conflict handling.

        Args:
            config_path: Path to YAML config
            strategy: Override default strategy for this sync

        Returns:
            SyncResult with details of all operations
        """
        strategy = strategy or self.sync_strategy
        result = SyncResult()

        # Get required paths from config
        paths = self.get_required_paths_from_config(config_path)
        if not paths:
            logger.warning("No components found in config")
            return result

        # Get local hashes
        local_hashes = self.get_local_hashes(paths)

        for path in paths:
            try:
                conflict = self._analyze_component(path, local_hashes)

                if conflict is None:
                    # Already up to date
                    result.skipped.append(path)
                    continue

                # Handle based on strategy
                resolution = self._resolve_conflict(conflict, strategy)

                if resolution == "use_server":
                    if self._download_version(path, conflict.server_hash):
                        result.synced.append(path)
                        result.component_hashes[path] = conflict.server_hash
                    else:
                        result.failed.append(path)

                elif resolution == "use_local":
                    if conflict.local_content:
                        new_hash = self._upload_local_version(path, conflict.local_content)
                        if new_hash:
                            result.uploaded.append(path)
                            result.component_hashes[path] = new_hash
                        else:
                            result.failed.append(path)
                    else:
                        result.failed.append(path)

                elif resolution == "skip":
                    result.skipped.append(path)
                    if conflict.local_hash:
                        result.component_hashes[path] = conflict.local_hash

                elif resolution == "pending":
                    # For interactive mode when callback returns pending
                    conflict.resolved = False
                    result.conflicts.append(conflict)

            except Exception as e:
                logger.error(f"Failed to sync {path}: {e}")
                result.failed.append(path)

        return result

    def _analyze_component(
        self, path: str, local_hashes: Dict[str, str]
    ) -> Optional[SyncConflict]:
        """Analyze a component to determine sync status.

        Returns:
            SyncConflict if action needed, None if up to date
        """
        category, name = path.split("/")
        name = name.replace(".py", "")

        # Get server active version
        try:
            response = self.client.get(
                f"/api/components/versions/{category}/{name}",
                params={"include_content": False},
            )

            if response.status_code == 404:
                # Server doesn't have this component
                local_hash = local_hashes.get(path)
                if local_hash:
                    local_content = self._get_local_content(path)
                    return SyncConflict(
                        path=path,
                        local_hash=local_hash,
                        server_hash=None,
                        conflict_type="local_only",
                        local_content=local_content,
                    )
                return None

            response.raise_for_status()
            versions = response.json().get("data", [])
            active = next((v for v in versions if v.get("is_active")), None)

            if not active:
                logger.warning(f"No active version for {path}")
                return None

            server_hash = active["hash"]
            local_hash = local_hashes.get(path)

            # Already up to date
            if local_hash == server_hash:
                return None

            # Local doesn't exist
            if not local_hash:
                return SyncConflict(
                    path=path,
                    local_hash=None,
                    server_hash=server_hash,
                    conflict_type="server_only",
                )

            # Both exist but different
            local_content = self._get_local_content(path)

            # Check if local version exists on server (just not active)
            local_exists_on_server = self._check_hash_exists_on_server(local_hash)

            if local_exists_on_server:
                # Local version is known to server, just outdated
                return SyncConflict(
                    path=path,
                    local_hash=local_hash,
                    server_hash=server_hash,
                    conflict_type="server_newer",
                    local_content=local_content,
                )
            else:
                # Local version is unknown to server (local development)
                return SyncConflict(
                    path=path,
                    local_hash=local_hash,
                    server_hash=server_hash,
                    conflict_type="local_newer",
                    local_content=local_content,
                )

        except Exception as e:
            logger.error(f"Failed to analyze {path}: {e}")
            raise

    def _resolve_conflict(self, conflict: SyncConflict, strategy: SyncStrategy) -> str:
        """Resolve a conflict based on strategy.

        Returns:
            Resolution: "use_server", "use_local", "skip", or "pending"
        """
        if strategy == SyncStrategy.DRY_RUN:
            return "skip"

        if strategy == SyncStrategy.SERVER_AUTHORITATIVE:
            if conflict.conflict_type in ("server_only", "server_newer"):
                return "use_server"
            elif conflict.conflict_type == "local_only":
                return "use_local"  # Upload to server
            else:  # local_newer
                return "use_server"  # Override local changes

        if strategy == SyncStrategy.LOCAL_PRIORITY:
            if conflict.conflict_type in ("local_only", "local_newer"):
                return "use_local"  # Upload to server
            else:
                return "use_server"

        if strategy == SyncStrategy.INTERACTIVE:
            if self.conflict_callback:
                return self.conflict_callback(conflict)
            else:
                # No callback, return pending for later resolution
                return "pending"

        return "skip"

    def resolve_pending_conflicts(
        self, result: SyncResult, resolutions: Dict[str, str]
    ) -> SyncResult:
        """Resolve pending conflicts with provided resolutions.

        Args:
            result: Previous SyncResult with pending conflicts
            resolutions: Dict of {path: resolution}

        Returns:
            Updated SyncResult
        """
        remaining_conflicts = []

        for conflict in result.conflicts:
            resolution = resolutions.get(conflict.path)
            if not resolution:
                remaining_conflicts.append(conflict)
                continue

            if resolution == "use_server" and conflict.server_hash:
                if self._download_version(conflict.path, conflict.server_hash):
                    result.synced.append(conflict.path)
                    result.component_hashes[conflict.path] = conflict.server_hash
                else:
                    result.failed.append(conflict.path)

            elif resolution == "use_local" and conflict.local_content:
                new_hash = self._upload_local_version(conflict.path, conflict.local_content)
                if new_hash:
                    result.uploaded.append(conflict.path)
                    result.component_hashes[conflict.path] = new_hash
                else:
                    result.failed.append(conflict.path)

            elif resolution == "skip":
                result.skipped.append(conflict.path)
                if conflict.local_hash:
                    result.component_hashes[conflict.path] = conflict.local_hash

            conflict.resolved = True
            conflict.resolution = resolution

        result.conflicts = remaining_conflicts
        return result
