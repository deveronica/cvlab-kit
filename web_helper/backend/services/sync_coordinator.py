"""
Sync Coordinator - Manage bidirectional node-code synchronization.

This module coordinates synchronization between code and node graph,
detecting conflicts and resolving them according to user preferences.

Key Features:
- Detect conflicts when both code and nodes are modified
- Track versions for change detection
- Resolve conflicts based on user strategy
- Prevent infinite sync loops
"""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ConflictType(str, Enum):
    """Type of synchronization conflict."""

    BOTH_MODIFIED = "both_modified"  # Code and nodes both changed
    PARSE_ERROR = "parse_error"  # Code has syntax errors
    GENERATION_ERROR = "generation_error"  # Node graph invalid
    VERSION_MISMATCH = "version_mismatch"  # Concurrent modifications


class ResolutionStrategy(str, Enum):
    """Strategy for resolving conflicts."""

    CODE_WINS = "code_wins"  # Keep code changes, regenerate nodes
    NODES_WIN = "nodes_win"  # Keep node changes, regenerate code
    MANUAL = "manual"  # User must manually resolve
    MERGE = "merge"  # Attempt automatic merge (future)


@dataclass
class Conflict:
    """Represents a synchronization conflict."""

    type: ConflictType
    message: str
    code_version: str  # Hash of code at conflict time
    node_version: str  # Hash of nodes at conflict time
    details: dict = field(default_factory=dict)


@dataclass
class SyncState:
    """State tracking for synchronization."""

    code_hash: str  # MD5 hash of current code
    node_hash: str  # MD5 hash of current node graph (JSON)
    last_synced_code_hash: str  # Last successfully synced code
    last_synced_node_hash: str  # Last successfully synced nodes
    sync_count: int = 0  # Number of sync operations
    max_retries: int = 3  # Maximum retry attempts


# =============================================================================
# Sync Coordinator
# =============================================================================


class SyncCoordinator:
    """Coordinate bidirectional node-code synchronization."""

    def __init__(self):
        self._sync_states: dict[str, SyncState] = {}  # agent_name -> state

    def detect_conflicts(
        self,
        agent_name: str,
        code: str,
        node_graph: dict,
        last_synced_code: Optional[str] = None,
        last_synced_nodes: Optional[dict] = None,
    ) -> list[Conflict]:
        """
        Detect if both code and nodes were modified since last sync.

        Args:
            agent_name: Agent identifier
            code: Current code content
            node_graph: Current node graph
            last_synced_code: Last synced code version
            last_synced_nodes: Last synced node graph

        Returns:
            List of detected conflicts (empty if no conflicts)
        """
        conflicts = []

        # Calculate current hashes
        code_hash = self._hash_code(code)
        node_hash = self._hash_nodes(node_graph)

        # Get or create sync state
        state = self._sync_states.get(
            agent_name,
            SyncState(
                code_hash=code_hash,
                node_hash=node_hash,
                last_synced_code_hash="",
                last_synced_node_hash="",
            ),
        )

        # Check if both changed since last sync
        if last_synced_code and last_synced_nodes:
            last_code_hash = self._hash_code(last_synced_code)
            last_node_hash = self._hash_nodes(last_synced_nodes)

            code_changed = code_hash != last_code_hash
            nodes_changed = node_hash != last_node_hash

            if code_changed and nodes_changed:
                conflicts.append(
                    Conflict(
                        type=ConflictType.BOTH_MODIFIED,
                        message="Both code and nodes were modified since last sync",
                        code_version=code_hash,
                        node_version=node_hash,
                        details={
                            "code_changed": True,
                            "nodes_changed": True,
                            "last_code_hash": last_code_hash,
                            "last_node_hash": last_node_hash,
                        },
                    )
                )

        # Check for infinite loop (too many syncs without settling)
        if state.sync_count > state.max_retries:
            conflicts.append(
                Conflict(
                    type=ConflictType.VERSION_MISMATCH,
                    message=f"Too many sync attempts ({state.sync_count}). Possible infinite loop.",
                    code_version=code_hash,
                    node_version=node_hash,
                    details={"sync_count": state.sync_count},
                )
            )

        # Update state
        state.code_hash = code_hash
        state.node_hash = node_hash
        state.sync_count += 1
        self._sync_states[agent_name] = state

        return conflicts

    def resolve_conflict(
        self,
        strategy: ResolutionStrategy,
        code: str,
        nodes: dict,
    ) -> tuple[str, dict]:
        """
        Resolve conflicts based on user choice.

        Args:
            strategy: Resolution strategy
            code: Current code
            nodes: Current node graph

        Returns:
            Tuple of (resolved_code, resolved_nodes)
        """
        if strategy == ResolutionStrategy.CODE_WINS:
            # Keep code, nodes will be regenerated from code
            return code, {}  # Empty dict signals "regenerate from code"

        elif strategy == ResolutionStrategy.NODES_WIN:
            # Keep nodes, code will be regenerated from nodes
            return "", nodes  # Empty string signals "regenerate from nodes"

        elif strategy == ResolutionStrategy.MANUAL:
            # User must manually resolve, return both unchanged
            return code, nodes

        elif strategy == ResolutionStrategy.MERGE:
            # TODO: Implement intelligent merging
            logger.warning("MERGE strategy not yet implemented, falling back to CODE_WINS")
            return code, {}

        else:
            logger.error(f"Unknown resolution strategy: {strategy}")
            return code, nodes

    def reset_sync_state(self, agent_name: str):
        """
        Reset sync state for an agent.

        Useful after a successful sync to prevent false conflicts.

        Args:
            agent_name: Agent identifier
        """
        if agent_name in self._sync_states:
            state = self._sync_states[agent_name]
            state.sync_count = 0
            state.last_synced_code_hash = state.code_hash
            state.last_synced_node_hash = state.node_hash

    def mark_synced(self, agent_name: str, code: str, nodes: dict):
        """
        Mark code and nodes as successfully synced.

        Args:
            agent_name: Agent identifier
            code: Synced code
            nodes: Synced node graph
        """
        code_hash = self._hash_code(code)
        node_hash = self._hash_nodes(nodes)

        self._sync_states[agent_name] = SyncState(
            code_hash=code_hash,
            node_hash=node_hash,
            last_synced_code_hash=code_hash,
            last_synced_node_hash=node_hash,
            sync_count=0,
        )

    def _hash_code(self, code: str) -> str:
        """Generate hash for code content."""
        return hashlib.md5(code.encode()).hexdigest()

    def _hash_nodes(self, node_graph: dict) -> str:
        """Generate hash for node graph."""
        # Simple hash based on node IDs and edges
        # (avoid JSON serialization for performance)
        nodes = node_graph.get("nodes", [])
        edges = node_graph.get("edges", [])

        node_ids = sorted([n.get("id", "") for n in nodes])
        edge_ids = sorted([f"{e.get('source', '')}-{e.get('target', '')}" for e in edges])

        content = "|".join(node_ids + edge_ids)
        return hashlib.md5(content.encode()).hexdigest()


# =============================================================================
# Singleton instance
# =============================================================================

_coordinator: Optional[SyncCoordinator] = None


def get_sync_coordinator() -> SyncCoordinator:
    """Get singleton SyncCoordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = SyncCoordinator()
    return _coordinator
