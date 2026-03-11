"""
Version Manager Service

Manages draft states and versions for the visual builder system.
Handles the lifecycle of drafts from creation to commit.

Key Responsibilities:
- Create and manage draft states in memory
- Persist drafts to disk (.draft.yaml files)
- Load drafts from disk
- Commit drafts to actual code (via CodeModifier)
- Track version history
"""

import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

from web_helper.backend.models.draft import (
    DraftState,
    DraftVersion,
    DraftStatus,
    GraphEdit,
    EditType,
)

logger = logging.getLogger(__name__)


class VersionManager:
    """
    Manages draft states and versions for node graphs.

    Usage:
        manager = VersionManager(drafts_dir="./drafts")

        # Create a new draft
        draft = manager.create_draft("my_agent")

        # Make edits
        draft.delete_node("node_123")

        # Save draft
        manager.save_draft(draft)

        # Commit draft (applies code changes)
        version = manager.commit_draft(draft)
    """

    def __init__(self, drafts_dir: str = "./drafts"):
        """
        Initialize the version manager.

        Args:
            drafts_dir: Directory to store draft files
        """
        self.drafts_dir = Path(drafts_dir)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of active drafts
        self._drafts: dict[str, DraftState] = {}

        # Version history (in-memory, could be persisted to DB later)
        self._versions: dict[str, list[DraftVersion]] = {}  # agent_name -> versions

    # =========================================================================
    # Draft Management
    # =========================================================================

    def create_draft(
        self,
        agent_name: str,
        base_graph_id: Optional[str] = None,
        source_file: Optional[str] = None,
        yaml_path: Optional[str] = None,
        initial_nodes: Optional[dict] = None,
        initial_edges: Optional[dict] = None,
        initial_config: Optional[dict] = None,
    ) -> DraftState:
        """
        Create a new draft for an agent.

        Args:
            agent_name: Name of the agent being edited
            base_graph_id: ID of the graph this draft is based on
            source_file: Path to the agent source file
            yaml_path: Path to the associated YAML config
            initial_nodes: Initial node state (from existing graph)
            initial_edges: Initial edge state (from existing graph)
            initial_config: Initial config values

        Returns:
            New DraftState instance
        """
        draft = DraftState(
            agent_name=agent_name,
            base_graph_id=base_graph_id,
            source_file=source_file,
            yaml_path=yaml_path,
            nodes=initial_nodes or {},
            edges=initial_edges or {},
            config=initial_config or {},
        )

        # Cache in memory
        self._drafts[draft.draft_id] = draft

        logger.info(f"Created draft {draft.draft_id} for agent {agent_name}")
        return draft

    def get_draft(self, draft_id: str) -> Optional[DraftState]:
        """
        Get a draft by ID.

        First checks memory cache, then tries to load from disk.

        Args:
            draft_id: Draft ID

        Returns:
            DraftState if found, None otherwise
        """
        # Check memory cache first
        if draft_id in self._drafts:
            return self._drafts[draft_id]

        # Try to load from disk
        draft = self._load_draft_from_file(draft_id)
        if draft:
            self._drafts[draft_id] = draft
        return draft

    def get_draft_for_agent(self, agent_name: str) -> Optional[DraftState]:
        """
        Get the most recent draft for an agent.

        Args:
            agent_name: Agent name

        Returns:
            Most recent DraftState for the agent, or None
        """
        # Check memory cache
        for draft in self._drafts.values():
            if draft.agent_name == agent_name and draft.status != DraftStatus.DISCARDED:
                return draft

        # Try to find on disk
        agent_drafts_dir = self.drafts_dir / agent_name
        if agent_drafts_dir.exists():
            draft_files = sorted(
                agent_drafts_dir.glob("*.draft.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for draft_file in draft_files:
                draft_id = draft_file.stem.replace(".draft", "")
                draft = self._load_draft_from_file(draft_id)
                if draft and draft.status != DraftStatus.DISCARDED:
                    self._drafts[draft_id] = draft
                    return draft

        return None

    def save_draft(self, draft: DraftState) -> Path:
        """
        Save a draft to disk.

        Args:
            draft: DraftState to save

        Returns:
            Path to the saved file
        """
        agent_dir = self.drafts_dir / draft.agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)

        file_path = agent_dir / f"{draft.draft_id}.draft.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(draft.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved draft {draft.draft_id} to {file_path}")
        return file_path

    def discard_draft(self, draft_id: str) -> bool:
        """
        Discard a draft (mark as discarded, don't delete file).

        Args:
            draft_id: Draft ID to discard

        Returns:
            True if successful
        """
        draft = self.get_draft(draft_id)
        if not draft:
            return False

        draft.status = DraftStatus.DISCARDED
        draft.modified_at = datetime.now()

        # Save the discarded state
        self.save_draft(draft)

        # Remove from memory cache
        self._drafts.pop(draft_id, None)

        logger.info(f"Discarded draft {draft_id}")
        return True

    def _load_draft_from_file(self, draft_id: str) -> Optional[DraftState]:
        """Load a draft from disk by ID."""
        # Search in all agent directories
        for agent_dir in self.drafts_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            draft_file = agent_dir / f"{draft_id}.draft.json"
            if draft_file.exists():
                try:
                    with open(draft_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return DraftState.from_dict(data)
                except Exception as e:
                    logger.error(f"Failed to load draft from {draft_file}: {e}")
        return None

    # =========================================================================
    # Edit Operations
    # =========================================================================

    def delete_node(self, draft_id: str, node_id: str) -> Optional[GraphEdit]:
        """
        Delete a node from a draft.

        Args:
            draft_id: Draft ID
            node_id: Node ID to delete

        Returns:
            GraphEdit if successful, None otherwise
        """
        draft = self.get_draft(draft_id)
        if not draft:
            logger.error(f"Draft {draft_id} not found")
            return None

        edit = draft.delete_node(node_id)
        if edit:
            self.save_draft(draft)
        return edit

    def delete_edge(self, draft_id: str, edge_id: str) -> Optional[GraphEdit]:
        """
        Delete an edge from a draft.

        Args:
            draft_id: Draft ID
            edge_id: Edge ID to delete

        Returns:
            GraphEdit if successful, None otherwise
        """
        draft = self.get_draft(draft_id)
        if not draft:
            logger.error(f"Draft {draft_id} not found")
            return None

        edit = draft.delete_edge(edge_id)
        if edit:
            self.save_draft(draft)
        return edit

    def add_node(
        self,
        draft_id: str,
        category: str,
        implementation: str,
        role: str | None = None,
        config: dict | None = None,
        position: dict | None = None,
    ) -> Optional[GraphEdit]:
        """
        Add a new node to a draft.

        Args:
            draft_id: Draft ID
            category: Component category (model, optimizer, loss, etc.)
            implementation: Implementation name (resnet18, adam, etc.)
            role: Variable name (defaults to category if not provided)
            config: Component configuration
            position: UI position

        Returns:
            GraphEdit if successful, None otherwise
        """
        draft = self.get_draft(draft_id)
        if not draft:
            logger.error(f"Draft {draft_id} not found")
            return None

        edit = draft.add_node(
            category=category,
            implementation=implementation,
            role=role,
            config=config,
            position=position,
        )
        if edit:
            self.save_draft(draft)
        return edit

    def update_node(
        self,
        draft_id: str,
        node_id: str,
        implementation: str | None = None,
        config: dict | None = None,
        position: dict | None = None,
    ) -> Optional[GraphEdit]:
        """
        Update an existing node in a draft.

        Args:
            draft_id: Draft ID
            node_id: Node ID to update
            implementation: New implementation name
            config: New configuration
            position: New UI position

        Returns:
            GraphEdit if successful, None otherwise
        """
        draft = self.get_draft(draft_id)
        if not draft:
            logger.error(f"Draft {draft_id} not found")
            return None

        edit = draft.update_node(
            node_id=node_id,
            implementation=implementation,
            config=config,
            position=position,
        )
        if edit:
            self.save_draft(draft)
        return edit

    def add_edge(
        self,
        draft_id: str,
        source: str,
        target: str,
        source_port: str = "out",
        target_port: str = "in",
        flow_type: str = "reference",
    ) -> Optional[GraphEdit]:
        """
        Add a new edge to a draft.

        Args:
            draft_id: Draft ID
            source: Source node ID
            target: Target node ID
            source_port: Source port name
            target_port: Target port name
            flow_type: Edge flow type

        Returns:
            GraphEdit if successful, None otherwise
        """
        draft = self.get_draft(draft_id)
        if not draft:
            logger.error(f"Draft {draft_id} not found")
            return None

        edit = draft.add_edge(
            source=source,
            target=target,
            source_port=source_port,
            target_port=target_port,
            flow_type=flow_type,
        )
        if edit:
            self.save_draft(draft)
        return edit

    def undo(self, draft_id: str) -> Optional[GraphEdit]:
        """
        Undo the last edit on a draft.

        Args:
            draft_id: Draft ID

        Returns:
            The undone GraphEdit, or None
        """
        draft = self.get_draft(draft_id)
        if not draft:
            return None

        edit = draft.undo()
        if edit:
            self.save_draft(draft)
        return edit

    def redo(self, draft_id: str) -> Optional[GraphEdit]:
        """
        Redo the last undone edit on a draft.

        Args:
            draft_id: Draft ID

        Returns:
            The redone GraphEdit, or None
        """
        draft = self.get_draft(draft_id)
        if not draft:
            return None

        edit = draft.redo()
        if edit:
            self.save_draft(draft)
        return edit

    # =========================================================================
    # Commit Operations
    # =========================================================================

    def commit_draft(
        self,
        draft_id: str,
        code_modifier: Optional["CodeModifier"] = None,
        description: str = "",
        created_by: str = "user",
    ) -> Optional[DraftVersion]:
        """
        Commit a draft, applying changes to the actual code.

        This:
        1. Creates a version snapshot
        2. Generates code changes from edits
        3. Applies code changes via CodeModifier
        4. Marks draft as committed

        Args:
            draft_id: Draft ID to commit
            code_modifier: CodeModifier instance for applying changes
            description: Version description
            created_by: "user" or "auto" (for auto-commit on first run)

        Returns:
            DraftVersion if successful, None otherwise
        """
        draft = self.get_draft(draft_id)
        if not draft:
            logger.error(f"Draft {draft_id} not found")
            return None

        if draft.status == DraftStatus.COMMITTED:
            logger.warning(f"Draft {draft_id} already committed")
            return None

        if draft.status == DraftStatus.CLEAN:
            logger.info(f"Draft {draft_id} has no changes to commit")
            return None

        # Create version
        versions = self._versions.get(draft.agent_name, [])
        version_number = len(versions) + 1

        version = DraftVersion(
            agent_name=draft.agent_name,
            draft_id=draft.draft_id,
            version_number=version_number,
            parent_version=versions[-1].version_id if versions else None,
            nodes_snapshot=draft.nodes.copy(),
            edges_snapshot=draft.edges.copy(),
            config_snapshot=draft.config.copy(),
            description=description or f"Version {version_number}",
            created_by=created_by,
        )

        # Apply code changes if code_modifier provided
        if code_modifier and draft.source_file:
            try:
                code_changes = self._generate_code_changes(draft, code_modifier)
                version.code_changes = code_changes
            except Exception as e:
                logger.error(f"Failed to apply code changes: {e}")
                raise

        # Update draft status
        draft.status = DraftStatus.COMMITTED
        draft.modified_at = datetime.now()
        self.save_draft(draft)

        # Save version
        if draft.agent_name not in self._versions:
            self._versions[draft.agent_name] = []
        self._versions[draft.agent_name].append(version)
        self._save_version(version)

        logger.info(f"Committed draft {draft_id} as version {version.version_id}")
        return version

    def _generate_code_changes(
        self,
        draft: DraftState,
        code_modifier: "CodeModifier",
    ) -> list[dict]:
        """
        Generate and apply code changes from draft edits.

        Returns list of changes applied.
        """
        changes = []

        for edit in draft.edits:
            if edit.edit_type == EditType.ADD_NODE:
                # Add new component to agent
                node_data = edit.after
                if node_data:
                    category = node_data.get("category", "")
                    impl = node_data.get("implementation", "")
                    role = node_data.get("role", "")
                    config = node_data.get("config", {})

                    result = code_modifier.add_component(
                        draft.source_file,
                        category=category,
                        impl=impl,
                        role=role,
                        config=config,
                    )
                    changes.append({
                        "type": "add_component",
                        "category": category,
                        "impl": impl,
                        "role": role,
                        "result": result,
                    })

            elif edit.edit_type == EditType.DELETE_NODE:
                # Determine what kind of node was deleted
                node_data = edit.before
                if node_data:
                    role = node_data.get("role") or node_data.get("label", "")
                    # Remove component from agent
                    result = code_modifier.remove_component(
                        draft.source_file,
                        role,
                    )
                    changes.append({
                        "type": "remove_component",
                        "target": role,
                        "result": result,
                    })

            elif edit.edit_type == EditType.DELETE_EDGE:
                # Edge deletions might need code updates too
                edge_data = edit.before
                changes.append({
                    "type": "remove_edge",
                    "target": edit.target_id,
                    "edge_data": edge_data,
                })

            elif edit.edit_type == EditType.UPDATE_NODE:
                # Update component implementation or config
                node_data = edit.after
                if node_data:
                    node_id = edit.target_id
                    impl = node_data.get("implementation")
                    if impl:
                        result = code_modifier.update_component_impl(
                            draft.source_file,
                            node_id,
                            impl,
                        )
                        changes.append({
                            "type": "update_component",
                            "target": node_id,
                            "impl": impl,
                            "result": result,
                        })

        return changes

    def _save_version(self, version: DraftVersion) -> Path:
        """Save a version to disk."""
        versions_dir = self.drafts_dir / version.agent_name / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        file_path = versions_dir / f"v{version.version_number}_{version.version_id}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(version.to_dict(), f, indent=2, ensure_ascii=False)

        return file_path

    # =========================================================================
    # Version History
    # =========================================================================

    def get_versions(self, agent_name: str) -> list[DraftVersion]:
        """
        Get all versions for an agent.

        Args:
            agent_name: Agent name

        Returns:
            List of versions, oldest first
        """
        # Load from memory if available
        if agent_name in self._versions:
            return self._versions[agent_name]

        # Load from disk
        versions = []
        versions_dir = self.drafts_dir / agent_name / "versions"
        if versions_dir.exists():
            for version_file in sorted(versions_dir.glob("*.json")):
                try:
                    with open(version_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    versions.append(DraftVersion.from_dict(data))
                except Exception as e:
                    logger.error(f"Failed to load version from {version_file}: {e}")

        self._versions[agent_name] = versions
        return versions

    def get_latest_version(self, agent_name: str) -> Optional[DraftVersion]:
        """Get the latest version for an agent."""
        versions = self.get_versions(agent_name)
        return versions[-1] if versions else None

    # =========================================================================
    # Code → Node Sync
    # =========================================================================

    def sync_from_code(
        self,
        draft_id: str,
        code_content: str,
        source_file: Optional[str] = None,
    ) -> dict:
        """
        Sync draft from parsed code content (Code → Node).

        Parses the code, extracts components, and updates the draft
        to match the code structure.

        Args:
            draft_id: Draft ID to sync
            code_content: Python code content to parse
            source_file: Optional source file path for context

        Returns:
            Dict with sync result:
            {
                "success": bool,
                "added": [node_ids],
                "removed": [node_ids],
                "updated": [node_ids],
                "errors": [str]
            }
        """
        from web_helper.backend.services.code_analyzer import AgentCodeAnalyzer

        draft = self.get_draft(draft_id)
        if not draft:
            return {"success": False, "errors": [f"Draft {draft_id} not found"]}

        # Parse code content
        analyzer = AgentCodeAnalyzer(code_content, source_file or draft.source_file)
        analysis = analyzer.analyze()

        if analysis.errors:
            return {"success": False, "errors": analysis.errors}

        # Extract nodes from analysis
        parsed_nodes = {}
        for node in analysis.nodes:
            # Skip uncovered nodes (not real components)
            if node.id.startswith("uncovered_"):
                continue

            # node.id is the role (e.g., "model", "optimizer", "train_loader")
            # node.label is the display name (e.g., "labeled", "cross_entropy")
            role = node.id
            category = node.category.value if node.category else "unknown"

            # Get implementation from metadata or properties
            impl = ""
            if node.metadata and "implementation" in node.metadata:
                impl = node.metadata["implementation"]
            elif node.properties:
                for prop in node.properties:
                    if prop.name == "impl":
                        impl = str(prop.value) if prop.value else ""
                        break

            parsed_nodes[role] = {
                "id": role,
                "role": role,
                "category": category,
                "implementation": impl,
                "config": node.metadata or {},
                "source": {
                    "line": node.source.line if node.source else None,
                    "end_line": node.source.end_line if node.source else None,
                } if node.source else None,
            }

        # Compare with draft nodes (filter out uncovered nodes from draft)
        draft_node_ids = {k for k in draft.nodes.keys() if not k.startswith("uncovered_")}
        parsed_node_ids = set(parsed_nodes.keys())

        added = []
        removed = []
        updated = []

        # Nodes to add (in code but not in draft)
        for node_id in parsed_node_ids - draft_node_ids:
            node_data = parsed_nodes[node_id]
            edit = GraphEdit(
                edit_type=EditType.ADD_NODE,
                target_id=node_id,
                target_type="node",
                before=None,
                after=node_data,
                description=f"Sync: Add {node_data['category']} node '{node_id}' from code",
            )
            draft.nodes[node_id] = node_data
            draft.edits.append(edit)
            added.append(node_id)

        # Nodes to remove (in draft but not in code)
        for node_id in draft_node_ids - parsed_node_ids:
            node_data = draft.nodes[node_id]
            edit = GraphEdit(
                edit_type=EditType.DELETE_NODE,
                target_id=node_id,
                target_type="node",
                before=node_data,
                after=None,
                description=f"Sync: Remove node '{node_id}' (deleted from code)",
            )
            del draft.nodes[node_id]
            draft.edits.append(edit)
            removed.append(node_id)

        # Nodes to update (in both, check for changes)
        for node_id in draft_node_ids & parsed_node_ids:
            draft_node = draft.nodes[node_id]
            parsed_node = parsed_nodes[node_id]

            # Check if implementation changed
            draft_impl = draft_node.get("implementation", "")
            parsed_impl = parsed_node.get("implementation", "")

            if draft_impl != parsed_impl:
                before = draft_node.copy()
                after = draft_node.copy()
                after["implementation"] = parsed_impl
                after["source"] = parsed_node.get("source")

                edit = GraphEdit(
                    edit_type=EditType.UPDATE_NODE,
                    target_id=node_id,
                    target_type="node",
                    before=before,
                    after=after,
                    description=f"Sync: Update node '{node_id}' impl: {draft_impl} → {parsed_impl}",
                )
                draft.nodes[node_id] = after
                draft.edits.append(edit)
                updated.append(node_id)

        # Update draft status if changes were made
        if added or removed or updated:
            draft.status = DraftStatus.MODIFIED
            draft.modified_at = datetime.now()
            self.save_draft(draft)

        return {
            "success": True,
            "added": added,
            "removed": removed,
            "updated": updated,
            "errors": [],
        }


# Singleton instance
_version_manager: Optional[VersionManager] = None


def get_version_manager(drafts_dir: str = "./drafts") -> VersionManager:
    """Get the singleton VersionManager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager(drafts_dir)
    return _version_manager
