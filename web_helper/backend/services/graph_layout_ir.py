"""Graph Layout IR Service - Save/Load user-customized node graph layouts.

This service manages the Intermediate Representation (IR) for node graph layouts,
allowing users to save their custom layouts (node positions, edge styles) and
restore them when revisiting the graph.

Storage Structure:
    web_helper/state/<agent_name>/graph_layout/<view_mode>/<graph_id>.json

DB Index:
    graph_layout_index table stores metadata for fast queries.
"""

import enum
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models (matching spec/L3_TSD/schema/graph-layout-ir.md)
# =============================================================================


class Position2D(BaseModel):
    """2D position for nodes and bend points."""

    x: float
    y: float


class Size2D(BaseModel):
    """2D size for nodes."""

    width: float
    height: float


class Viewport(BaseModel):
    """Canvas viewport state."""

    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


class EdgeStyle(BaseModel):
    """Custom edge styling."""

    color: Optional[str] = None
    stroke_width: Optional[int] = None
    stroke_dasharray: Optional[str] = None
    animated: Optional[bool] = None


class LayoutEdge(BaseModel):
    """Layout data for an edge."""

    id: str
    edge_ref: str = Field(..., description="Original edge reference")
    source: str
    target: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None
    bend_points: Optional[list[Position2D]] = None
    style: Optional[EdgeStyle] = None


class LayoutNode(BaseModel):
    """Layout data for a node."""

    id: str
    node_ref: str = Field(..., description="Original node reference")
    position: Position2D
    size: Size2D
    is_expanded: Optional[bool] = None
    is_drillable: Optional[bool] = None
    label: Optional[str] = None
    color: Optional[str] = None


class ViewMode(str, enum.Enum):
    """View mode for filtering edges."""

    PROCESS = "process"
    DATA = "data"
    BOTH = "both"


class GraphLayoutIR(BaseModel):
    """Root IR model for graph layout storage."""

    version: str = "1.0"
    agent_name: str
    graph_id: str
    view_mode: ViewMode
    hierarchy_depth: int = 0
    path: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    nodes: list[LayoutNode]
    edges: list[LayoutEdge]
    viewport: Optional[Viewport] = None
    metadata: Optional[dict[str, Any]] = None


# =============================================================================
# Service Class
# =============================================================================


class GraphLayoutIRService:
    """Service for managing graph layout IR storage and retrieval."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the service.

        Args:
            project_root: Root directory of the project. Defaults to current working directory.
        """
        self.project_root = project_root or Path.cwd()
        self.state_root = self.project_root / "web_helper" / "state"
        self._ensure_state_root()

    def _ensure_state_root(self) -> None:
        """Ensure the state root directory exists."""
        self.state_root.mkdir(parents=True, exist_ok=True)

    def _get_agent_state_dir(self, agent_name: str, create: bool = True) -> Path:
        """Get the state directory for an agent.

        Args:
            agent_name: Agent name.
            create: Whether to create the directory if it doesn't exist.

        Returns:
            Path to the agent's state directory.
        """
        agent_dir = self.state_root / agent_name
        if create and not agent_dir.exists():
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "graph_layout").mkdir(parents=True, exist_ok=True)
        return agent_dir

    def _get_layout_dir(
        self,
        agent_name: str,
        view_mode: ViewMode,
        create: bool = True,
    ) -> Path:
        """Get the layout directory for a specific view mode.

        Args:
            agent_name: Agent name.
            view_mode: View mode (process, data, both).
            create: Whether to create the directory if it doesn't exist.

        Returns:
            Path to the layout directory.
        """
        agent_dir = self._get_agent_state_dir(agent_name, create=create)
        layout_dir = agent_dir / "graph_layout" / view_mode.value
        if create and not layout_dir.exists():
            layout_dir.mkdir(parents=True, exist_ok=True)
        return layout_dir

    def _generate_graph_id(
        self,
        agent_name: str,
        hierarchy_depth: int,
        path: list[str],
    ) -> str:
        """Generate a graph ID from path components.

        Args:
            agent_name: Agent name.
            hierarchy_depth: Hierarchy depth (0=Agent, 1=Layer, 2=Module).
            path: Drill-down path (e.g., ["model", "layer1"]).

        Returns:
            Graph ID string.
        """
        if not path:
            return agent_name

        path_str = "-".join(path)
        if hierarchy_depth > 0:
            return f"{agent_name}-{path_str}"
        return agent_name

    def save_layout(
        self,
        agent_name: str,
        graph_id: str,
        view_mode: ViewMode,
        nodes: list[dict],
        edges: list[dict],
        hierarchy_depth: int = 0,
        path: Optional[list[str]] = None,
        viewport: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Save graph layout to IR file.

        Args:
            agent_name: Agent name.
            graph_id: Graph identifier.
            view_mode: View mode (process, data, both).
            nodes: List of node layout data.
            edges: List of edge layout data.
            hierarchy_depth: Hierarchy depth (0=Agent, 1=Layer, 2=Module).
            path: Drill-down path.
            viewport: Viewport state.
            metadata: Additional metadata.

        Returns:
            {
                "success": bool,
                "file_path": str,
                "message": str
            }
        """
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Parse nodes and edges with validation
            layout_nodes = [LayoutNode(**n) for n in nodes]
            layout_edges = [LayoutEdge(**e) for e in edges]

            # Build IR object
            ir = GraphLayoutIR(
                agent_name=agent_name,
                graph_id=graph_id,
                view_mode=view_mode,
                hierarchy_depth=hierarchy_depth,
                path=path or [],
                created_at=now,
                updated_at=now,
                nodes=layout_nodes,
                edges=layout_edges,
                viewport=Viewport(**viewport) if viewport else None,
                metadata=metadata,
            )

            # Ensure directory exists
            layout_dir = self._get_layout_dir(agent_name, view_mode, create=True)

            # Save to file
            file_path = layout_dir / f"{graph_id}.json"
            file_path.write_text(ir.model_dump_json(indent=2), encoding="utf-8")

            logger.info(f"Saved layout IR to {file_path}")

            return {
                "success": True,
                "file_path": str(file_path),
                "message": f"Layout saved successfully",
            }

        except Exception as e:
            logger.error(
                f"Failed to save layout for {agent_name}/{graph_id}: {e}", exc_info=True
            )
            return {
                "success": False,
                "file_path": None,
                "message": f"Failed to save layout: {str(e)}",
            }

    def load_layout(
        self,
        agent_name: str,
        graph_id: str,
        view_mode: ViewMode,
    ) -> Optional[GraphLayoutIR]:
        """Load graph layout from IR file.

        Args:
            agent_name: Agent name.
            graph_id: Graph identifier.
            view_mode: View mode (process, data, both).

        Returns:
            GraphLayoutIR object if found, None otherwise.
        """
        try:
            layout_dir = self._get_layout_dir(agent_name, view_mode, create=False)
            file_path = layout_dir / f"{graph_id}.json"

            if not file_path.exists():
                logger.debug(f"Layout file not found: {file_path}")
                return None

            data = json.loads(file_path.read_text(encoding="utf-8"))
            ir = GraphLayoutIR(**data)

            logger.info(f"Loaded layout IR from {file_path}")
            return ir

        except Exception as e:
            logger.error(
                f"Failed to load layout for {agent_name}/{graph_id}: {e}", exc_info=True
            )
            return None

    def list_layouts(self, agent_name: str) -> list[dict]:
        """List all layouts for an agent.

        Args:
            agent_name: Agent name.

        Returns:
            List of layout metadata dictionaries.
        """
        try:
            agent_dir = self._get_agent_state_dir(agent_name, create=False)
            if not agent_dir.exists():
                return []

            layouts = []
            graph_layout_dir = agent_dir / "graph_layout"
            if not graph_layout_dir.exists():
                return []

            for view_mode_dir in graph_layout_dir.iterdir():
                if not view_mode_dir.is_dir():
                    continue

                for layout_file in view_mode_dir.glob("*.json"):
                    try:
                        data = json.loads(layout_file.read_text(encoding="utf-8"))
                        layouts.append(
                            {
                                "graph_id": data.get("graph_id"),
                                "view_mode": data.get("view_mode"),
                                "hierarchy_depth": data.get("hierarchy_depth", 0),
                                "path": data.get("path", []),
                                "updated_at": data.get("updated_at"),
                                "file_path": str(layout_file),
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse layout file {layout_file}: {e}"
                        )
                        continue

            return sorted(layouts, key=lambda x: (x["view_mode"], x["graph_id"]))

        except Exception as e:
            logger.error(f"Failed to list layouts for {agent_name}: {e}", exc_info=True)
            return []

    def delete_layout(
        self,
        agent_name: str,
        graph_id: str,
        view_mode: ViewMode,
    ) -> dict:
        """Delete a layout file.

        Args:
            agent_name: Agent name.
            graph_id: Graph identifier.
            view_mode: View mode.

        Returns:
            {"success": bool, "message": str}
        """
        try:
            layout_dir = self._get_layout_dir(agent_name, view_mode, create=False)
            file_path = layout_dir / f"{graph_id}.json"

            if not file_path.exists():
                return {
                    "success": False,
                    "message": "Layout file not found",
                }

            file_path.unlink()
            logger.info(f"Deleted layout IR: {file_path}")

            return {
                "success": True,
                "message": "Layout deleted successfully",
            }

        except Exception as e:
            logger.error(
                f"Failed to delete layout for {agent_name}/{graph_id}: {e}",
                exc_info=True,
            )
            return {
                "success": False,
                "message": f"Failed to delete layout: {str(e)}",
            }

    def get_or_create_default(
        self,
        agent_name: str,
        view_mode: ViewMode,
        hierarchy_depth: int = 0,
        path: Optional[list[str]] = None,
    ) -> GraphLayoutIR:
        """Get existing layout or create default empty layout.

        Args:
            agent_name: Agent name.
            view_mode: View mode.
            hierarchy_depth: Hierarchy depth.
            path: Drill-down path.

        Returns:
            GraphLayoutIR object.
        """
        graph_id = self._generate_graph_id(agent_name, hierarchy_depth, path or [])

        # Try to load existing
        existing = self.load_layout(agent_name, graph_id, view_mode)
        if existing:
            return existing

        # Create default empty layout
        now = datetime.now(timezone.utc).isoformat()
        return GraphLayoutIR(
            agent_name=agent_name,
            graph_id=graph_id,
            view_mode=view_mode,
            hierarchy_depth=hierarchy_depth,
            path=path or [],
            created_at=now,
            updated_at=now,
            nodes=[],
            edges=[],
        )


# Singleton instance
_layout_ir_service: Optional[GraphLayoutIRService] = None


def get_layout_ir_service() -> GraphLayoutIRService:
    """Get the singleton GraphLayoutIRService instance."""
    global _layout_ir_service
    if _layout_ir_service is None:
        _layout_ir_service = GraphLayoutIRService()
    return _layout_ir_service
