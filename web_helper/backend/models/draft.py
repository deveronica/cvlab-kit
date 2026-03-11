"""
Draft and Version Management Models

This module defines models for managing draft states and versions
of node graphs for the visual builder system.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                     Draft Lifecycle                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [No Draft] ──create_draft()──► [Draft (임시)]                  │
│                                      │                           │
│                            edit (delete/add nodes)               │
│                                      │                           │
│                                      ▼                           │
│                               [Draft (수정됨)]                   │
│                                      │                           │
│               ┌──────────────────────┼──────────────────────┐   │
│               │                      │                      │   │
│               ▼                      ▼                      ▼   │
│         [Discard]            [First Run]             [Commit]   │
│               │                      │                      │   │
│               ▼                      ▼                      ▼   │
│            삭제됨             자동 commit             코드 수정   │
│                              + 코드 수정             + 버전 생성 │
└─────────────────────────────────────────────────────────────────┘
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import uuid


# =============================================================================
# Edit Types - Types of operations on graphs
# =============================================================================

class EditType(str, Enum):
    """Types of edits that can be performed on a graph."""

    # Node operations
    ADD_NODE = "add_node"           # Add a new node
    DELETE_NODE = "delete_node"     # Remove a node
    UPDATE_NODE = "update_node"     # Modify node properties
    MOVE_NODE = "move_node"         # Change node position

    # Edge operations
    ADD_EDGE = "add_edge"           # Connect two nodes
    DELETE_EDGE = "delete_edge"     # Disconnect two nodes
    UPDATE_EDGE = "update_edge"     # Modify edge properties

    # Config operations
    UPDATE_CONFIG = "update_config" # Modify YAML config value

    # Compound operations
    REPLACE_COMPONENT = "replace_component"  # Change component impl


# =============================================================================
# Draft Status
# =============================================================================

class DraftStatus(str, Enum):
    """Status of a draft."""
    CLEAN = "clean"           # No edits made
    MODIFIED = "modified"     # Has uncommitted edits
    COMMITTED = "committed"   # Changes applied to code
    DISCARDED = "discarded"   # Draft abandoned


# =============================================================================
# Graph Edit - A single edit operation
# =============================================================================

@dataclass
class GraphEdit:
    """
    Represents a single edit operation on the graph.

    Edits are tracked for:
    - Undo/redo functionality
    - Batching edits before commit
    - Generating code changes
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    edit_type: EditType = EditType.UPDATE_NODE

    # Target identification
    target_id: str = ""           # Node ID or Edge ID
    target_type: str = "node"     # "node" or "edge" or "config"

    # Change data
    before: Optional[dict[str, Any]] = None  # State before edit
    after: Optional[dict[str, Any]] = None   # State after edit

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "edit_type": self.edit_type.value,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "before": self.before,
            "after": self.after,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphEdit":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            edit_type=EditType(data["edit_type"]),
            target_id=data.get("target_id", ""),
            target_type=data.get("target_type", "node"),
            before=data.get("before"),
            after=data.get("after"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            description=data.get("description", ""),
        )


# =============================================================================
# Draft State - In-memory state for editing
# =============================================================================

@dataclass
class DraftState:
    """
    In-memory state for a draft graph being edited.

    Contains the current state of nodes/edges plus edit history.
    """
    draft_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    agent_name: str = ""

    # Status
    status: DraftStatus = DraftStatus.CLEAN

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    # Graph state (current state after all edits)
    nodes: dict[str, dict] = field(default_factory=dict)  # node_id -> node data
    edges: dict[str, dict] = field(default_factory=dict)  # edge_id -> edge data
    config: dict[str, Any] = field(default_factory=dict)  # YAML config values

    # Edit history
    edits: list[GraphEdit] = field(default_factory=list)
    undo_stack: list[GraphEdit] = field(default_factory=list)

    # Source tracking
    base_graph_id: Optional[str] = None  # Original graph this was derived from
    source_file: Optional[str] = None    # Agent source file path
    yaml_path: Optional[str] = None      # Associated YAML config path

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_edit(self, edit: GraphEdit) -> None:
        """Add an edit and update status."""
        self.edits.append(edit)
        self.undo_stack.clear()  # Clear redo stack on new edit
        self.status = DraftStatus.MODIFIED
        self.modified_at = datetime.now()

    def undo(self) -> Optional[GraphEdit]:
        """Undo the last edit."""
        if not self.edits:
            return None
        edit = self.edits.pop()
        self.undo_stack.append(edit)
        self._apply_reverse(edit)
        self.modified_at = datetime.now()
        return edit

    def redo(self) -> Optional[GraphEdit]:
        """Redo the last undone edit."""
        if not self.undo_stack:
            return None
        edit = self.undo_stack.pop()
        self.edits.append(edit)
        self._apply_forward(edit)
        self.modified_at = datetime.now()
        return edit

    def _apply_reverse(self, edit: GraphEdit) -> None:
        """Apply reverse of an edit (for undo)."""
        if edit.edit_type == EditType.DELETE_NODE and edit.before:
            self.nodes[edit.target_id] = edit.before
        elif edit.edit_type == EditType.ADD_NODE:
            self.nodes.pop(edit.target_id, None)
        elif edit.edit_type == EditType.DELETE_EDGE and edit.before:
            self.edges[edit.target_id] = edit.before
        elif edit.edit_type == EditType.ADD_EDGE:
            self.edges.pop(edit.target_id, None)

    def _apply_forward(self, edit: GraphEdit) -> None:
        """Apply an edit forward (for redo)."""
        if edit.edit_type == EditType.DELETE_NODE:
            self.nodes.pop(edit.target_id, None)
        elif edit.edit_type == EditType.ADD_NODE and edit.after:
            self.nodes[edit.target_id] = edit.after
        elif edit.edit_type == EditType.DELETE_EDGE:
            self.edges.pop(edit.target_id, None)
        elif edit.edit_type == EditType.ADD_EDGE and edit.after:
            self.edges[edit.target_id] = edit.after

    def delete_node(self, node_id: str) -> Optional[GraphEdit]:
        """Delete a node from the draft."""
        if node_id not in self.nodes:
            return None

        edit = GraphEdit(
            edit_type=EditType.DELETE_NODE,
            target_id=node_id,
            target_type="node",
            before=self.nodes[node_id].copy(),
            after=None,
            description=f"Delete node {node_id}",
        )

        # Remove node
        del self.nodes[node_id]

        # Remove connected edges
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items()
            if edge.get("source_node") == node_id or edge.get("target_node") == node_id
        ]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]

        self.add_edit(edit)
        return edit

    def delete_edge(self, edge_id: str) -> Optional[GraphEdit]:
        """Delete an edge from the draft."""
        if edge_id not in self.edges:
            return None

        edit = GraphEdit(
            edit_type=EditType.DELETE_EDGE,
            target_id=edge_id,
            target_type="edge",
            before=self.edges[edge_id].copy(),
            after=None,
            description=f"Delete edge {edge_id}",
        )

        del self.edges[edge_id]
        self.add_edit(edit)
        return edit

    def add_node(
        self,
        category: str,
        implementation: str,
        role: str | None = None,
        config: dict | None = None,
        position: dict | None = None,
    ) -> GraphEdit:
        """Add a new node to the draft.

        Args:
            category: Component category (model, optimizer, loss, etc.)
            implementation: Implementation name (resnet18, adam, etc.)
            role: Variable name (defaults to category if not provided)
            config: Component configuration
            position: UI position

        Returns:
            GraphEdit for undo/redo support
        """
        # Generate role if not provided
        if role is None:
            role = category

        # Make role unique if needed
        base_role = role
        counter = 2
        while role in self.nodes:
            role = f"{base_role}_{counter}"
            counter += 1

        node_id = role  # Use role as node ID

        node_data = {
            "id": node_id,
            "role": role,
            "category": category,
            "implementation": implementation,
            "config": config or {},
            "position": position or {"x": 0, "y": 0},
        }

        edit = GraphEdit(
            edit_type=EditType.ADD_NODE,
            target_id=node_id,
            target_type="node",
            before=None,
            after=node_data,
            description=f"Add {category} node '{role}' ({implementation})",
        )

        self.nodes[node_id] = node_data
        self.add_edit(edit)
        return edit

    def update_node(
        self,
        node_id: str,
        implementation: str | None = None,
        config: dict | None = None,
        position: dict | None = None,
    ) -> GraphEdit | None:
        """Update an existing node in the draft.

        Args:
            node_id: Node ID to update
            implementation: New implementation name
            config: New configuration (merged with existing)
            position: New UI position

        Returns:
            GraphEdit if node found and updated, None otherwise
        """
        if node_id not in self.nodes:
            return None

        before = self.nodes[node_id].copy()
        after = before.copy()

        if implementation is not None:
            after["implementation"] = implementation
        if config is not None:
            after["config"] = {**after.get("config", {}), **config}
        if position is not None:
            after["position"] = position

        # Check if anything changed
        if before == after:
            return None

        edit = GraphEdit(
            edit_type=EditType.UPDATE_NODE,
            target_id=node_id,
            target_type="node",
            before=before,
            after=after,
            description=f"Update node '{node_id}'",
        )

        self.nodes[node_id] = after
        self.add_edit(edit)
        return edit

    def add_edge(
        self,
        source: str,
        target: str,
        source_port: str = "out",
        target_port: str = "in",
        flow_type: str = "reference",
    ) -> GraphEdit | None:
        """Add a new edge to the draft.

        Args:
            source: Source node ID
            target: Target node ID
            source_port: Source port name
            target_port: Target port name
            flow_type: Edge flow type (reference, parameters, data, etc.)

        Returns:
            GraphEdit if successful, None if source/target not found
        """
        # Validate source and target exist
        if source not in self.nodes:
            return None
        if target not in self.nodes:
            return None

        edge_id = f"{source}_{source_port}_to_{target}_{target_port}"

        # Check if edge already exists
        if edge_id in self.edges:
            return None

        edge_data = {
            "id": edge_id,
            "source_node": source,
            "source_port": source_port,
            "target_node": target,
            "target_port": target_port,
            "flow_type": flow_type,
        }

        edit = GraphEdit(
            edit_type=EditType.ADD_EDGE,
            target_id=edge_id,
            target_type="edge",
            before=None,
            after=edge_data,
            description=f"Add edge {source}.{source_port} → {target}.{target_port}",
        )

        self.edges[edge_id] = edge_data
        self.add_edit(edit)
        return edit

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "draft_id": self.draft_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "nodes": self.nodes,
            "edges": self.edges,
            "config": self.config,
            "edits": [e.to_dict() for e in self.edits],
            "base_graph_id": self.base_graph_id,
            "source_file": self.source_file,
            "yaml_path": self.yaml_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DraftState":
        """Create from dictionary."""
        state = cls(
            draft_id=data.get("draft_id", str(uuid.uuid4())[:12]),
            agent_name=data.get("agent_name", ""),
            status=DraftStatus(data.get("status", "clean")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            modified_at=datetime.fromisoformat(data["modified_at"]) if "modified_at" in data else datetime.now(),
            nodes=data.get("nodes", {}),
            edges=data.get("edges", {}),
            config=data.get("config", {}),
            base_graph_id=data.get("base_graph_id"),
            source_file=data.get("source_file"),
            yaml_path=data.get("yaml_path"),
            metadata=data.get("metadata", {}),
        )
        state.edits = [GraphEdit.from_dict(e) for e in data.get("edits", [])]
        return state


# =============================================================================
# Draft Version - Persisted version information
# =============================================================================

@dataclass
class DraftVersion:
    """
    A committed version of a draft.

    Created when a draft is committed (either explicitly or on first run).
    """
    version_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    # Source reference
    agent_name: str = ""
    draft_id: str = ""           # Original draft ID

    # Version info
    version_number: int = 1
    parent_version: Optional[str] = None  # Previous version ID

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    # Snapshot
    nodes_snapshot: dict[str, dict] = field(default_factory=dict)
    edges_snapshot: dict[str, dict] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # Code changes applied
    code_changes: list[dict] = field(default_factory=list)  # List of file changes

    # Metadata
    description: str = ""
    created_by: str = "user"     # "user" or "auto" (on first run)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version_id": self.version_id,
            "agent_name": self.agent_name,
            "draft_id": self.draft_id,
            "version_number": self.version_number,
            "parent_version": self.parent_version,
            "created_at": self.created_at.isoformat(),
            "nodes_snapshot": self.nodes_snapshot,
            "edges_snapshot": self.edges_snapshot,
            "config_snapshot": self.config_snapshot,
            "code_changes": self.code_changes,
            "description": self.description,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DraftVersion":
        """Create from dictionary."""
        return cls(
            version_id=data.get("version_id", str(uuid.uuid4())[:12]),
            agent_name=data.get("agent_name", ""),
            draft_id=data.get("draft_id", ""),
            version_number=data.get("version_number", 1),
            parent_version=data.get("parent_version"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            nodes_snapshot=data.get("nodes_snapshot", {}),
            edges_snapshot=data.get("edges_snapshot", {}),
            config_snapshot=data.get("config_snapshot", {}),
            code_changes=data.get("code_changes", []),
            description=data.get("description", ""),
            created_by=data.get("created_by", "user"),
        )
