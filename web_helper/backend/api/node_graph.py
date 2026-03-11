"""Node Graph API endpoints for Simulink-style hierarchical code visualization.

Primary Endpoint:
- GET /api/nodes/hierarchy/{agent_name}?path=... - Single hierarchical endpoint

Path examples:
- "" (empty): Level 0 - Components from setup()
- "model": Level 1 - Model layers
- "loss_fn": Level 1 - Loss sub-components
"""

import ast
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from web_helper.backend.models.hierarchy import (
    AddEdgeRequest,
    AddEdgeResponse,
    AddNodeRequest,
    AddNodeResponse,
    HierarchyResponse,
    UpdateNodeRequest,
    UpdateNodeResponse,
)
from web_helper.backend.services.hierarchical_graph_builder import (
    HierarchicalGraphBuilder,
)
from web_helper.backend.services.node_registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Node Graph"])

# Initialize builder
hierarchical_builder = HierarchicalGraphBuilder(project_root=Path.cwd())


@router.get("/types")
async def get_data_types() -> dict:
    """Get discovered data types and their assigned colors.

    Returns:
        {success: bool, types: {type_name: color_hex}}
    """
    from web_helper.backend.models import SessionLocal
    from web_helper.backend.services.type_discovery import get_type_colors

    db = SessionLocal()
    try:
        colors = get_type_colors(db)
        return {"success": True, "types": colors}
    finally:
        db.close()


def _collect_method_names(agent_path: Path) -> set[str]:
    try:
        source = agent_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    tree = ast.parse(source)
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _resolve_flow_method(agent_name: str, method: str | None) -> str | None:
    if not method:
        return method
    aliases = {"val_step", "validate_step", "validation_step"}
    if method not in aliases:
        return method
    agent_path = (
        hierarchical_builder.project_root / "cvlabkit" / "agent" / f"{agent_name}.py"
    )
    method_names = _collect_method_names(agent_path)
    if method in method_names:
        return method
    for candidate in ("validate_step", "validation_step", "val_step"):
        if candidate in method_names:
            return candidate
    return method


def _hash_code_content(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()


def _get_node_id(node: object) -> str:
    if isinstance(node, dict):
        return str(node.get("id", ""))
    return str(getattr(node, "id", ""))


def _get_edge_pair(edge: object) -> tuple[str, str]:
    if isinstance(edge, dict):
        source = str(edge.get("source") or edge.get("source_node") or "")
        target = str(edge.get("target") or edge.get("target_node") or "")
        return source, target
    source = str(getattr(edge, "source_node", "") or getattr(edge, "source", ""))
    target = str(getattr(edge, "target_node", "") or getattr(edge, "target", ""))
    return source, target


def _hash_nodes_edges(nodes: Sequence[object], edges: Sequence[object]) -> str:
    node_ids = sorted([_get_node_id(node) for node in nodes])
    edge_ids = sorted(
        [f"{source}-{target}" for source, target in map(_get_edge_pair, edges)]
    )
    content = "|".join(node_ids + edge_ids)
    return hashlib.md5(content.encode()).hexdigest()


# =============================================================================
# Primary Endpoints
# =============================================================================


@router.get("/agents")
async def get_all_agents() -> dict:
    """Get list of all available agents in the project.

    Returns:
        {success: bool, agents: [str]}
    """
    try:
        agent_dir = Path.cwd() / "cvlabkit" / "agent"
        agents = []

        if agent_dir.exists():
            for py_file in agent_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                if py_file.name == "base.py":
                    continue
                agents.append(py_file.stem)

        return {"success": True, "agents": sorted(agents)}

    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}",
        )


@router.get("/hierarchy/{agent_name}")
async def get_hierarchy_graph(
    agent_name: str,
    path: str = Query("", description="Drill path: '', 'model', 'model/layer1'"),
    phase: str | None = Query(
        None,
        description="Phase filter: 'initialize' (setup edges), 'flow' (train_step edges), or 'config' (with YAML config node)",
    ),
    method: str | None = Query(
        None,
        description="Method name to parse for 'flow' phase: 'train_step' or 'val_step'",
    ),
    impl: str | None = Query(
        None,
        description="Implementation name for drill-down (e.g., 'resnet18' when path='model')",
    ),
    config_path: str | None = Query(None, description="Path to specific configuration file"),
) -> HierarchyResponse:
    """Get hierarchical node graph for the given agent and path."""
    try:
        graph = hierarchical_builder.build_for_path(
            agent_name, path, phase=phase, method=method, impl=impl, config_path=config_path
        )
        if phase == "flow" and method in {
            "val_step",
            "validate_step",
            "validation_step",
        }:
            resolved_method = _resolve_flow_method(agent_name, method)
            if graph and not graph.nodes and resolved_method != method:
                graph = hierarchical_builder.build_for_path(
                    agent_name, path, phase=phase, method=resolved_method, impl=impl
                )
                graph.id = f"{agent_name}_{method}"
                graph.label = f"{agent_name} - {method}()"
        return HierarchyResponse(success=True, data=graph)

    except FileNotFoundError as e:
        logger.warning(f"Agent not found: {agent_name}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(
            f"Failed to build hierarchy for {agent_name} path={path}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build hierarchy: {str(e)}",
        )


# =============================================================================
# Phase-Separated Endpoints (Setup vs Train-Step)
# =============================================================================


@router.get("/hierarchy/{agent_name}/setup")
async def get_setup_hierarchy(
    agent_name: str,
    path: str = Query("", description="Drill path: '', 'model', 'model/layer1'"),
    impl: str | None = Query(None, description="Implementation name for drill-down"),
    config_path: str | None = Query(None, description="Path to specific configuration file"),
) -> HierarchyResponse:
    """Get Setup phase node graph (component dependencies and config relationships).

    Setup phase shows:
    - Nodes: Components (model, optimizer, loss, dataset, etc.)
    - Ports: Method names (parameters, backward, step)
    - Edges: Method call dependencies
    - Purpose: Component initialization order and config relationships

    Args:
        agent_name: Agent name
        path: Drill-down path ('' for root, 'model' for component layer)
        impl: Implementation name for drill-down
        config_path: Specific config file path

    Returns:
        HierarchyResponse with setup-only graph
    """
    try:
        graph = hierarchical_builder.build_for_path(
            agent_name, path, phase="initialize", method=None, impl=impl, config_path=config_path
        )
        return HierarchyResponse(success=True, data=graph)
    except FileNotFoundError as e:
        logger.warning(f"Agent not found: {agent_name}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"Failed to build setup hierarchy for {agent_name} path={path}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build setup hierarchy: {str(e)}",
        )


@router.get("/hierarchy/{agent_name}/train-step")
async def get_train_step_hierarchy(
    agent_name: str,
    path: str = Query("", description="Drill path: '', 'loss', etc."),
    method: str = Query("train_step", description="Method: 'train_step', 'val_step', 'validation_step'"),
    impl: str | None = Query(None, description="Implementation name for drill-down"),
) -> HierarchyResponse:
    """Get Train-Step phase node graph (data flow and execution order).

    Train-step phase shows:
    - Nodes: Operations (forward, backward, step, etc.)
    - Ports: Variable names (inputs, outputs, loss, gradients)
    - Edges: Data flow and control flow
    - Purpose: Data flow visualization and execution order

    Args:
        agent_name: Agent name
        path: Drill-down path
        method: Method to parse ('train_step', 'val_step', 'validation_step')
        impl: Implementation name for drill-down

    Returns:
        HierarchyResponse with train-step-only graph
    """
    try:
        # Resolve method aliases
        resolved_method = _resolve_flow_method(agent_name, method)

        graph = hierarchical_builder.build_for_path(
            agent_name, path, phase="flow", method=resolved_method, impl=impl
        )

        # If original method name differs from resolved, update labels
        if graph and resolved_method != method:
            graph.id = f"{agent_name}_{method}"
            graph.label = f"{agent_name} - {method}()"

        return HierarchyResponse(success=True, data=graph)
    except FileNotFoundError as e:
        logger.warning(f"Agent not found: {agent_name}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"Failed to build train-step hierarchy for {agent_name} path={path} method={method}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build train-step hierarchy: {str(e)}",
        )


@router.get("/categories")
async def get_categories() -> dict:
    """Get all registered node categories with their themes.

    This endpoint returns the category registry, which includes:
    - Category name and label
    - Theme (colors, icon)
    - Whether drill-down is supported
    - Whether an analyzer is registered

    Frontend can use this to:
    1. Dynamically render node themes
    2. Show/hide drill-down indicators
    3. Display category legend

    Returns:
        {categories: [...]}
    """
    try:
        registry = get_registry()
        return registry.to_api_response()

    except Exception as e:
        logger.error(f"Failed to get categories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get categories: {str(e)}",
        )


@router.get("/methods/{category}")
async def get_category_methods(category: str) -> dict:
    """Get available methods for a component category.

    This endpoint returns methods that can be called on components of the given category.
    For example, 'model' category returns methods like 'parameters()', 'eval()', 'train()'.

    Frontend can use this to:
    1. Show method selector dropdown when creating edges
    2. Display available operations in node context menu
    3. Auto-complete method calls in code generation

    Args:
        category: Component category (e.g., 'model', 'optimizer', 'loss')

    Returns:
        {
            success: bool,
            data: {
                category: str,
                methods: [
                    {name: str, returns: str, args: [...], description: str},
                    ...
                ]
            }
        }
    """
    from web_helper.backend.services.node_registry import get_methods_for_category

    try:
        methods = get_methods_for_category(category)
        method_dicts = [
            {
                "name": m.name,
                "returns": m.returns.value,
                "args": m.args,
                "description": m.description,
            }
            for m in methods
        ]
        return {
            "success": True,
            "data": {
                "category": category,
                "methods": method_dicts,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get methods for {category}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get methods: {str(e)}",
        )


@router.get("/hierarchy/{agent_name}/available")
async def get_available_agents() -> dict:
    """[DEPRECATED] Get list of available agents. Use /api/nodes/agents instead."""
    return await get_all_agents()


# =============================================================================
# Source Code Endpoint
# =============================================================================


@router.get("/agent-source/{agent_name}")
async def get_agent_source(agent_name: str) -> dict:
    """Get the source code of an agent file.

    This endpoint returns the raw Python source code for display in CodePane.
    Used for bidirectional sync between code and node visualization.

    Args:
        agent_name: Agent file name without .py extension (e.g., "classification")

    Returns:
        {success: bool, source: str, path: str} or error
    """
    try:
        agent_dir = Path.cwd() / "cvlabkit" / "agent"
        agent_file = agent_dir / f"{agent_name}.py"

        if not agent_file.exists():
            raise FileNotFoundError(f"Agent '{agent_name}' not found")

        source_code = agent_file.read_text(encoding="utf-8")

        return {
            "success": True,
            "source": source_code,
            "path": str(agent_file.relative_to(Path.cwd())),
        }

    except FileNotFoundError as e:
        logger.warning(f"Agent source not found: {agent_name}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(
            f"Failed to read agent source for {agent_name}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read agent source: {str(e)}",
        )


# =============================================================================
# Code Generation Endpoint
# =============================================================================


@router.post("/generate-config")
async def generate_yaml_config(request: dict) -> dict:
    """Generate YAML configuration from node graph.

    Takes a node graph (nodes) and generates CVLab-Kit YAML config.

    Request body:
        {
            "agent_name": "classification",
            "nodes": [...],       # ReactFlow nodes (setup nodes only)
            "epochs": 100,        # Optional
            "batch_size": 32,     # Optional
            "lr": 0.001,          # Optional
            "device": "cuda"      # Optional
        }

    Returns:
        {
            "success": bool,
            "yaml_content": str,   # YAML string
            "config_dict": dict,   # Parsed config
            "error": str | null
        }
    """
    from web_helper.backend.services.config_generator import (
        ConfigGenerationRequest,
        generate_yaml_config as gen_config,
    )

    try:
        gen_request = ConfigGenerationRequest(
            agent_name=request.get("agent_name", "classification"),
            nodes=request.get("nodes", []),
            epochs=request.get("epochs", 100),
            batch_size=request.get("batch_size", 32),
            lr=request.get("lr", 0.001),
            device=request.get("device", "cuda"),
        )
        result = gen_config(gen_request)
        return result.model_dump()

    except Exception as e:
        logger.error(f"Failed to generate config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate config: {str(e)}",
        )


@router.post("/parse-config")
async def parse_yaml_config(request: dict) -> dict:
    """Parse YAML configuration into node representation.

    Takes a YAML string and returns node-like structures.

    Request body:
        {
            "yaml_content": "agent: classification\\nmodel: resnet18\\n..."
        }

    Returns:
        {
            "success": bool,
            "nodes": [...],      # Node representations
            "error": str | null
        }
    """
    from web_helper.backend.services.config_generator import (
        parse_yaml_config as parse_config,
    )

    try:
        yaml_content = request.get("yaml_content", "")
        nodes = parse_config(yaml_content)
        return {"success": True, "nodes": nodes}

    except Exception as e:
        logger.error(f"Failed to parse config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse config: {str(e)}",
        )


@router.post("/config-node")
async def create_config_node_from_file(request: dict) -> dict:
    """Create a Config Node from a YAML file path (Req 5).

    Takes a file path and creates a Config node ready for the graph.

    Request body:
        {
            "file_path": "config/classification.yaml",
            "position": {"x": 100, "y": 200}  # Optional
        }

    Returns:
        {
            "success": bool,
            "node": {...},      # Config node with outputs for each key
            "error": str | null
        }
    """
    from web_helper.backend.services.config_node_generator import (
        ConfigNodeGenerator,
    )

    try:
        file_path = request.get("file_path", "")

        if not file_path:
            raise ValueError("file_path is required")

        generator = ConfigNodeGenerator()
        config_node, properties = generator.generate_config_node(config_path=file_path)

        return {
            "success": True,
            "node": config_node.model_dump() if hasattr(config_node, "model_dump") else config_node,
        }

    except Exception as e:
        logger.error(f"Failed to create config node: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create config node: {str(e)}",
        )


@router.post("/generate")
async def generate_agent_code(request: dict) -> dict:
    """Generate Python code from node graph.

    Takes a node graph (nodes + edges) and generates CVLab-Kit Agent code.

    Request body:
        {
            "agent_name": "MyAgent",  # Optional, defaults to "GeneratedAgent"
            "nodes": [...],  # ReactFlow nodes
            "edges": [...]   # ReactFlow edges
        }

    Returns:
        {
            "success": bool,
            "code": str,       # Full agent code
            "setup_code": str, # Just the setup() body
            "train_step_code": str,  # Just the train_step() body
            "error": str | null
        }
    """
    from web_helper.backend.services.code_generator import (
        CodeGenerationRequest,
        generate_agent_code as gen_code,
    )

    try:
        gen_request = CodeGenerationRequest(
            agent_name=request.get("agent_name", "GeneratedAgent"),
            nodes=request.get("nodes", []),
            edges=request.get("edges", []),
        )
        result = gen_code(gen_request)
        return result.model_dump()

    except Exception as e:
        logger.error(f"Failed to generate code: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate code: {str(e)}",
        )


# =============================================================================
# Draft Management Endpoints
# =============================================================================


@router.post("/hierarchy/{agent_name}/draft")
async def create_draft(
    agent_name: str,
    base_path: str = Query("", description="Path to base graph (empty for root)"),
) -> dict:
    """Create a new draft for editing an agent's node graph.

    This creates an in-memory draft state that can be edited
    without affecting the actual code until committed.

    Args:
        agent_name: Agent file name without .py extension
        base_path: Path to the base graph to copy

    Returns:
        {
            "success": bool,
            "draft_id": str,
            "agent_name": str,
            "status": str,
            "created_at": str
        }
    """
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        manager = get_version_manager()

        # Get current graph to initialize draft with
        agent_dir = Path.cwd() / "cvlabkit" / "agent"
        agent_file = agent_dir / f"{agent_name}.py"
        yaml_path = Path.cwd() / "config" / f"{agent_name}.yaml"

        if not agent_file.exists():
            raise FileNotFoundError(f"Agent '{agent_name}' not found")

        # Build current graph to get initial nodes/edges
        graph = hierarchical_builder.build_for_path(agent_name, base_path)

        # Convert to dict format for draft
        initial_nodes = {
            node.id: {
                "id": node.id,
                "level": node.level.value if node.level else "component",
                "category": node.category.value if node.category else None,
                "label": node.label,
                "position_x": 0,  # Position is managed by frontend
                "position_y": 0,
                "config": node.metadata if node.metadata else {},
            }
            for node in graph.nodes
        }

        initial_edges = {
            edge.id: {
                "id": edge.id,
                "source_node": edge.source_node,
                "source_port": edge.source_port or "out",
                "target_node": edge.target_node,
                "target_port": edge.target_port or "in",
                "flow_type": edge.flow_type.value if edge.flow_type else "reference",
            }
            for edge in graph.edges
        }

        draft = manager.create_draft(
            agent_name=agent_name,
            base_graph_id=graph.id if hasattr(graph, "id") else None,
            source_file=str(agent_file),
            yaml_path=str(yaml_path) if yaml_path.exists() else None,
            initial_nodes=initial_nodes,
            initial_edges=initial_edges,
        )

        return {
            "success": True,
            "draft_id": draft.draft_id,
            "agent_name": draft.agent_name,
            "status": draft.status.value,
            "created_at": draft.created_at.isoformat(),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create draft for {agent_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create draft: {str(e)}",
        )


@router.get("/hierarchy/{agent_name}/draft/{draft_id}")
async def get_draft(agent_name: str, draft_id: str) -> dict:
    """Get the current state of a draft.

    Returns the draft's current nodes, edges, and edit history.

    Args:
        agent_name: Agent file name (for validation)
        draft_id: Draft ID from create_draft

    Returns:
        {
            "success": bool,
            "draft": {...draft state...}
        }
    """
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        manager = get_version_manager()
        draft = manager.get_draft(draft_id)

        if not draft:
            raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")

        if draft.agent_name != agent_name:
            raise HTTPException(
                status_code=400, detail="Draft does not belong to this agent"
            )

        return {"success": True, "draft": draft.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get draft {draft_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get draft: {str(e)}")


@router.delete("/hierarchy/{agent_name}/nodes/{node_id}")
async def delete_node(
    agent_name: str,
    node_id: str,
    draft_id: str = Query(
        None, description="Draft ID. If None, modifies code directly."
    ),
) -> dict:
    """Delete a node from the graph.

    If draft_id is provided, the change is made to the draft (temporary).
    If draft_id is None, the change is applied directly to the code (permanent).

    Args:
        agent_name: Agent file name
        node_id: Node ID to delete
        draft_id: Optional draft ID for temporary changes

    Returns:
        {
            "success": bool,
            "node_id": str,
            "mode": "draft" | "direct",
            "edit": {...} | null
        }
    """
    from web_helper.backend.services.version_manager import get_version_manager
    from web_helper.backend.services.code_modifier import get_code_modifier

    try:
        if draft_id:
            # Draft mode: modify draft state only
            manager = get_version_manager()
            edit = manager.delete_node(draft_id, node_id)

            if not edit:
                raise HTTPException(
                    status_code=404, detail=f"Node {node_id} not found in draft"
                )

            return {
                "success": True,
                "node_id": node_id,
                "mode": "draft",
                "edit": edit.to_dict(),
            }

        else:
            # Direct mode: modify code immediately
            modifier = get_code_modifier()
            agent_file = Path.cwd() / "cvlabkit" / "agent" / f"{agent_name}.py"

            if not agent_file.exists():
                raise FileNotFoundError(f"Agent '{agent_name}' not found")

            # Node ID might be the component name (e.g., "model", "optimizer")
            # or a compound ID. For now, assume it's the component name.
            component_name = node_id.split("_")[0] if "_" in node_id else node_id

            result = modifier.remove_component(
                str(agent_file),
                component_name,
            )

            if not result["success"]:
                raise HTTPException(
                    status_code=400, detail=result.get("error", "Failed to delete")
                )

            return {
                "success": True,
                "node_id": node_id,
                "mode": "direct",
                "edit": result,
            }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete node {node_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete node: {str(e)}")


@router.delete("/hierarchy/{agent_name}/edges/{edge_id}")
async def delete_edge(
    agent_name: str,
    edge_id: str,
    draft_id: str = Query(
        None, description="Draft ID. If None, modifies code directly."
    ),
) -> dict:
    """Delete an edge from the graph.

    If draft_id is provided, the change is made to the draft (temporary).
    If draft_id is None, the change is applied directly to the code (permanent).

    Note: Edge deletion may require code modification if the edge represents
    a data dependency (e.g., model.parameters() → optimizer).

    Args:
        agent_name: Agent file name
        edge_id: Edge ID to delete
        draft_id: Optional draft ID for temporary changes

    Returns:
        {
            "success": bool,
            "edge_id": str,
            "mode": "draft" | "direct",
            "edit": {...} | null
        }
    """
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        if draft_id:
            # Draft mode: modify draft state only
            manager = get_version_manager()
            edit = manager.delete_edge(draft_id, edge_id)

            if not edit:
                raise HTTPException(
                    status_code=404, detail=f"Edge {edge_id} not found in draft"
                )

            return {
                "success": True,
                "edge_id": edge_id,
                "mode": "draft",
                "edit": edit.to_dict(),
            }

        else:
            # Direct mode: modify code immediately
            from web_helper.backend.services.code_modifier import get_code_modifier

            # Parse edge_id: format is "{source}_{source_port}_to_{target}_{target_port}"
            # Example: "model_parameters_to_optimizer_params"
            if "_to_" not in edge_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid edge_id format: {edge_id}. Expected format: source_sourcePort_to_target_targetPort",
                )

            parts = edge_id.split("_to_")
            if len(parts) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid edge_id format: {edge_id}",
                )

            source_part = parts[0]  # e.g., "model_parameters"
            target_part = parts[1]  # e.g., "optimizer_params"

            # Split source_part: last segment is source_port, rest is source role
            source_segments = source_part.rsplit("_", 1)
            if len(source_segments) == 2:
                source_role, source_port = source_segments
            else:
                source_role = source_part
                source_port = "output"  # default

            # Split target_part: last segment is target_port, rest is target role
            target_segments = target_part.rsplit("_", 1)
            if len(target_segments) == 2:
                target_role, target_port = target_segments
            else:
                target_role = target_part
                target_port = "input"  # default

            modifier = get_code_modifier()
            agent_file = Path.cwd() / "cvlabkit" / "agent" / f"{agent_name}.py"

            if not agent_file.exists():
                raise HTTPException(
                    status_code=404, detail=f"Agent '{agent_name}' not found"
                )

            result = modifier.remove_dependency_edge(
                str(agent_file),
                source_role=source_role,
                target_role=target_role,
                source_port=source_port,
            )

            if not result["success"]:
                return {
                    "success": False,
                    "edge_id": edge_id,
                    "mode": "direct",
                    "error": result.get("error", "Failed to remove dependency edge"),
                }

            return {
                "success": True,
                "edge_id": edge_id,
                "mode": "direct",
                "code_change": result.get("new_code"),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete edge {edge_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete edge: {str(e)}")


# =============================================================================
# Add/Update Node/Edge Endpoints (Phase C)
# =============================================================================


@router.post("/hierarchy/{agent_name}/nodes")
async def add_node(
    agent_name: str,
    request: AddNodeRequest,
    draft_id: str = Query(
        None, description="Draft ID. If None, modifies code directly."
    ),
) -> AddNodeResponse:
    """Add a new component node to the graph.

    If draft_id is provided, the change is made to the draft (temporary).
    If draft_id is None, the change is applied directly to the code (permanent).

    Role (변수명) handling:
    - If role is provided, use it directly
    - If role is None, auto-generate from category (e.g., "model")
    - If duplicate, append suffix (_2, _3, etc.)

    Args:
        agent_name: Agent file name
        request: AddNodeRequest with category, implementation, role, config, position
        draft_id: Optional draft ID for temporary changes

    Returns:
        AddNodeResponse with success, node_id, role, mode, edit
    """
    from web_helper.backend.services.code_modifier import get_code_modifier
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        if draft_id:
            # Draft mode: modify draft state only
            manager = get_version_manager()
            edit = manager.add_node(
                draft_id,
                request.category,
                request.implementation,
                role=request.role,
                config=request.config,
                position=request.position,
            )

            if not edit:
                return AddNodeResponse(
                    success=False,
                    error="Failed to add node to draft",
                )

            return AddNodeResponse(
                success=True,
                node_id=edit.target_id,
                role=edit.after.get("role") if edit.after else None,
                mode="draft",
                edit=edit.to_dict(),
            )

        else:
            # Direct mode: modify code immediately
            modifier = get_code_modifier()
            agent_file = Path.cwd() / "cvlabkit" / "agent" / f"{agent_name}.py"

            if not agent_file.exists():
                raise FileNotFoundError(f"Agent '{agent_name}' not found")

            result = modifier.add_component(
                str(agent_file),
                category=request.category,
                impl=request.implementation,
                role=request.role,
                config=request.config if request.config else None,
            )

            if not result["success"]:
                return AddNodeResponse(
                    success=False,
                    error=result.get("error", "Failed to add component"),
                )

            return AddNodeResponse(
                success=True,
                node_id=result.get("role", request.role or request.category),
                role=result.get("role"),
                mode="direct",
                edit=result,
            )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add node: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add node: {str(e)}")


@router.put("/hierarchy/{agent_name}/nodes/{node_id}")
async def update_node(
    agent_name: str,
    node_id: str,
    request: UpdateNodeRequest,
    draft_id: str = Query(
        None, description="Draft ID. If None, modifies code directly."
    ),
) -> UpdateNodeResponse:
    """Update an existing node.

    If draft_id is provided, the change is made to the draft (temporary).
    If draft_id is None, the change is applied directly to the code (permanent).

    Update types:
    - implementation: Change the impl in self.create.xxx(impl="new_impl")
    - config: Update YAML configuration
    - position: UI position (draft mode only, no code impact)

    Args:
        agent_name: Agent file name
        node_id: Node ID (= role name, e.g., "model", "optimizer")
        request: UpdateNodeRequest with implementation, config, position
        draft_id: Optional draft ID for temporary changes

    Returns:
        UpdateNodeResponse with success, node_id, mode, edit
    """
    from web_helper.backend.services.code_modifier import get_code_modifier
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        if draft_id:
            # Draft mode: modify draft state only
            manager = get_version_manager()
            edit = manager.update_node(
                draft_id,
                node_id,
                implementation=request.implementation,
                config=request.config,
                position=request.position,
            )

            if not edit:
                return UpdateNodeResponse(
                    success=False,
                    error=f"Node {node_id} not found in draft or no changes made",
                )

            return UpdateNodeResponse(
                success=True,
                node_id=node_id,
                mode="draft",
                edit=edit.to_dict(),
            )

        else:
            # Direct mode: modify code immediately
            modifier = get_code_modifier()
            agent_file = Path.cwd() / "cvlabkit" / "agent" / f"{agent_name}.py"

            if not agent_file.exists():
                raise FileNotFoundError(f"Agent '{agent_name}' not found")

            # Only implementation change affects code
            if request.implementation:
                result = modifier.update_component_impl(
                    str(agent_file),
                    role=node_id,
                    new_impl=request.implementation,
                )

                if not result["success"]:
                    return UpdateNodeResponse(
                        success=False,
                        error=result.get("error", "Failed to update implementation"),
                    )

                return UpdateNodeResponse(
                    success=True,
                    node_id=node_id,
                    mode="direct",
                    edit=result,
                )

            # Config changes: update YAML file
            if request.config:
                # Determine YAML path
                yaml_path = request.yaml_path
                if not yaml_path:
                    # Default: config/{agent_name}.yaml
                    yaml_path = str(Path.cwd() / "config" / f"{agent_name}.yaml")

                if not Path(yaml_path).exists():
                    return UpdateNodeResponse(
                        success=False,
                        error=f"YAML file not found: {yaml_path}",
                    )

                # Build changes dict with nested key support
                # If category is provided, use category.key format for nested keys
                changes = {}
                for key, value in request.config.items():
                    if request.category and not key.startswith(request.category + "."):
                        # Nested under category (e.g., "loss.supervised")
                        changes[f"{request.category}.{key}"] = value
                    else:
                        # Top-level or already has category prefix
                        changes[key] = value

                result = modifier.update_yaml(yaml_path, changes)

                if not result["success"]:
                    return UpdateNodeResponse(
                        success=False,
                        error=result.get("error", "Failed to update YAML"),
                    )

                return UpdateNodeResponse(
                    success=True,
                    node_id=node_id,
                    mode="direct",
                    edit=result,
                )

            # Position changes only: no code change needed
            return UpdateNodeResponse(
                success=True,
                node_id=node_id,
                mode="direct",
                edit={"message": "Position changes do not affect code"},
            )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update node {node_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update node: {str(e)}")


@router.post("/hierarchy/{agent_name}/edges")
async def add_edge(
    agent_name: str,
    request: AddEdgeRequest,
    draft_id: str = Query(
        None, description="Draft ID. If None, modifies code directly."
    ),
) -> AddEdgeResponse:
    """Add a new edge (data dependency).

    If draft_id is provided, the change is made to the draft (temporary).
    If draft_id is None, the change is applied directly to the code (permanent).

    Edge types and code modifications:
    - model → optimizer (parameters): self.optimizer = self.create.optimizer(self.model.parameters())
    - dataset → dataloader (data): self.dataloader = self.create.dataloader(self.dataset)

    Args:
        agent_name: Agent file name
        request: AddEdgeRequest with source, target, sourcePort, targetPort, flow_type
        draft_id: Optional draft ID for temporary changes

    Returns:
        AddEdgeResponse with success, edge_id, mode, code_change, edit
    """
    from web_helper.backend.services.code_modifier import get_code_modifier
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        if draft_id:
            # Draft mode: modify draft state only
            manager = get_version_manager()
            edit = manager.add_edge(
                draft_id,
                source=request.source,
                target=request.target,
                source_port=request.source_port,
                target_port=request.target_port,
                flow_type=request.flow_type,
            )

            if not edit:
                return AddEdgeResponse(
                    success=False,
                    error="Failed to add edge to draft",
                )

            edge_id = f"{request.source}_{request.source_port}_to_{request.target}_{request.target_port}"

            return AddEdgeResponse(
                success=True,
                edge_id=edge_id,
                mode="draft",
                edit=edit.to_dict(),
            )

        else:
            # Direct mode: modify code immediately
            modifier = get_code_modifier()
            agent_file = Path.cwd() / "cvlabkit" / "agent" / f"{agent_name}.py"

            if not agent_file.exists():
                raise FileNotFoundError(f"Agent '{agent_name}' not found")

            result = modifier.add_dependency_edge(
                str(agent_file),
                source_role=request.source,
                target_role=request.target,
                source_port=request.source_port,
            )

            if not result["success"]:
                return AddEdgeResponse(
                    success=False,
                    error=result.get("error", "Failed to add dependency edge"),
                )

            edge_id = f"{request.source}_{request.source_port}_to_{request.target}_{request.target_port}"

            return AddEdgeResponse(
                success=True,
                edge_id=edge_id,
                mode="direct",
                code_change=result.get("new_code"),
                edit=result,
            )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add edge: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add edge: {str(e)}")


@router.post("/hierarchy/{agent_name}/draft/{draft_id}/commit")
async def commit_draft(
    agent_name: str,
    draft_id: str,
    description: str = Query("", description="Version description"),
) -> dict:
    """Commit a draft, applying all changes to the actual code.

    This will:
    1. Create a version snapshot
    2. Apply code changes via AST modification
    3. Update YAML config if needed
    4. Mark draft as committed

    Args:
        agent_name: Agent file name
        draft_id: Draft ID to commit
        description: Optional version description

    Returns:
        {
            "success": bool,
            "version_id": str,
            "version_number": int,
            "code_changes": [...]
        }
    """
    from web_helper.backend.services.version_manager import get_version_manager
    from web_helper.backend.services.code_modifier import get_code_modifier

    try:
        manager = get_version_manager()
        modifier = get_code_modifier()

        draft = manager.get_draft(draft_id)
        if not draft:
            raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")

        if draft.agent_name != agent_name:
            raise HTTPException(
                status_code=400, detail="Draft does not belong to this agent"
            )

        version = manager.commit_draft(
            draft_id,
            code_modifier=modifier,
            description=description,
            created_by="user",
        )

        if not version:
            return {
                "success": False,
                "error": "No changes to commit or draft already committed",
            }

        return {
            "success": True,
            "version_id": version.version_id,
            "version_number": version.version_number,
            "code_changes": version.code_changes,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to commit draft {draft_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to commit draft: {str(e)}")


@router.delete("/hierarchy/{agent_name}/draft/{draft_id}")
async def discard_draft(agent_name: str, draft_id: str) -> dict:
    """Discard a draft without committing.

    The draft will be marked as discarded and removed from memory.

    Args:
        agent_name: Agent file name
        draft_id: Draft ID to discard

    Returns:
        {"success": bool}
    """
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        manager = get_version_manager()
        draft = manager.get_draft(draft_id)

        if not draft:
            raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")

        if draft.agent_name != agent_name:
            raise HTTPException(
                status_code=400, detail="Draft does not belong to this agent"
            )

        success = manager.discard_draft(draft_id)
        return {"success": success}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to discard draft {draft_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to discard draft: {str(e)}"
        )


@router.post("/hierarchy/{agent_name}/draft/{draft_id}/undo")
async def undo_edit(agent_name: str, draft_id: str) -> dict:
    """Undo the last edit in a draft.

    Returns:
        {"success": bool, "edit": {...undone edit...} | null}
    """
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        manager = get_version_manager()
        edit = manager.undo(draft_id)

        return {
            "success": edit is not None,
            "edit": edit.to_dict() if edit else None,
        }

    except Exception as e:
        logger.error(f"Failed to undo in draft {draft_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to undo: {str(e)}")


@router.post("/hierarchy/{agent_name}/draft/{draft_id}/redo")
async def redo_edit(agent_name: str, draft_id: str) -> dict:
    """Redo the last undone edit in a draft.

    Returns:
        {"success": bool, "edit": {...redone edit...} | null}
    """
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        manager = get_version_manager()
        edit = manager.redo(draft_id)

        return {
            "success": edit is not None,
            "edit": edit.to_dict() if edit else None,
        }

    except Exception as e:
        logger.error(f"Failed to redo in draft {draft_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to redo: {str(e)}")


@router.get("/hierarchy/{agent_name}/versions")
async def get_versions(agent_name: str) -> dict:
    """Get version history for an agent.

    Returns:
        {
            "success": bool,
            "versions": [...version summaries...]
        }
    """
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        manager = get_version_manager()
        versions = manager.get_versions(agent_name)

        return {
            "success": True,
            "versions": [
                {
                    "version_id": v.version_id,
                    "version_number": v.version_number,
                    "created_at": v.created_at.isoformat(),
                    "description": v.description,
                    "created_by": v.created_by,
                }
                for v in versions
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get versions for {agent_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get versions: {str(e)}")


# =============================================================================
# Bidirectional Sync Endpoints (Code ↔ Nodes)
# =============================================================================


@router.post("/hierarchy/{agent_name}/draft/{draft_id}/sync-from-code")
async def sync_draft_from_code(
    agent_name: str,
    draft_id: str,
    request: dict,
) -> dict:
    """Sync draft from edited code content (Code → Node).

    When user edits code directly, this endpoint parses the code
    and updates the draft to match the code structure.

    Request body:
        {
            "code_content": str,  # The edited Python code
        }

    Returns:
        {
            "success": bool,
            "added": [node_ids],      # Nodes added from code
            "removed": [node_ids],    # Nodes removed (deleted from code)
            "updated": [node_ids],    # Nodes updated (implementation changed)
            "draft_status": str,      # Current draft status
            "errors": [str] | null
        }
    """
    from web_helper.backend.services.version_manager import get_version_manager

    try:
        code_content = request.get("code_content", "")

        if not code_content:
            raise HTTPException(status_code=400, detail="code_content is required")

        manager = get_version_manager()

        # Verify draft exists and belongs to agent
        draft = manager.get_draft(draft_id)
        if not draft:
            raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")

        if draft.agent_name != agent_name:
            raise HTTPException(
                status_code=400, detail="Draft does not belong to this agent"
            )

        # Sync from code
        result = manager.sync_from_code(draft_id, code_content)

        if not result["success"]:
            return {
                "success": False,
                "errors": result.get("errors", []),
            }

        # Get updated draft status
        updated_draft = manager.get_draft(draft_id)

        return {
            "success": True,
            "added": result["added"],
            "removed": result["removed"],
            "updated": result["updated"],
            "draft_status": updated_draft.status.value if updated_draft else "unknown",
            "errors": None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync draft from code: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to sync from code: {str(e)}"
        )


@router.post("/sync")
async def sync_code_to_nodes(request: dict) -> dict:
    """Sync code changes to node graph.

    Parses Python code and returns updated node graph with code blocks.
    Used for code → nodes synchronization (500ms debounced on frontend).

    Request body:
        {
            "agent_name": str,
            "code_content": str,
            "last_synced_code": str | null,  # For conflict detection
            "last_synced_nodes": {...} | null  # For conflict detection
        }

    Returns:
        {
            "success": bool,
            "node_graph": {...},  # HierarchicalNodeGraph
            "code_blocks": [{...}],  # List of CodeBlock
            "uncovered_lines": [int],  # Lines without node mapping
            "conflicts": [{...}] | null,  # Detected conflicts
            "error": str | null
        }
    """
    from web_helper.backend.services.ast_block_parser import get_ast_block_parser
    from web_helper.backend.services.sync_coordinator import get_sync_coordinator

    try:
        agent_name = request.get("agent_name")
        code_content = request.get("code_content", "")
        last_synced_code = request.get("last_synced_code")
        last_synced_nodes = request.get("last_synced_nodes")

        if not agent_name:
            raise HTTPException(status_code=400, detail="agent_name is required")

        # Parse code into blocks
        parser = get_ast_block_parser()
        parse_result = parser.extract_blocks(code_content, agent_name)

        if parse_result.errors:
            # Syntax errors or parse failures
            return {
                "success": False,
                "error": f"Parse errors: {parse_result.errors[0].message}",
                "errors": [
                    {
                        "type": e.type,
                        "line": e.line,
                        "message": e.message,
                    }
                    for e in parse_result.errors
                ],
            }

        source_file = f"cvlabkit/agent/{agent_name}.py"
        graph = hierarchical_builder.build_from_source(
            agent_name,
            code_content,
            source_file,
        )

        code_hash = _hash_code_content(code_content)
        nodes_hash = _hash_nodes_edges(graph.nodes, graph.edges)
        graph.metadata["roundtrip"] = {
            "code_hash": code_hash,
            "nodes_hash": nodes_hash,
        }

        # Map blocks to nodes
        blocks_with_nodes = parser.map_blocks_to_nodes(
            parse_result.blocks,
            {"nodes": [n.model_dump() for n in graph.nodes]},
        )

        # Detect conflicts
        coordinator = get_sync_coordinator()
        conflicts = coordinator.detect_conflicts(
            agent_name,
            code_content,
            {"nodes": [n.model_dump() for n in graph.nodes]},
            last_synced_code,
            last_synced_nodes,
        )

        # If no conflicts, mark as synced
        if not conflicts:
            coordinator.mark_synced(
                agent_name,
                code_content,
                {"nodes": [n.model_dump() for n in graph.nodes]},
            )

        return {
            "success": True,
            "node_graph": graph.model_dump(),
            "code_blocks": [
                {
                    "id": block.id,
                    "name": block.name,
                    "type": block.type,
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                    "params": block.params,
                    "docstring": block.docstring,
                    "parent": block.parent,
                    "node_ids": block.node_ids,
                }
                for block in blocks_with_nodes
            ],
            "uncovered_lines": list(parse_result.uncovered_lines),
            "conflicts": (
                [
                    {
                        "type": c.type.value,
                        "message": c.message,
                        "code_version": c.code_version,
                        "node_version": c.node_version,
                        "details": c.details,
                    }
                    for c in conflicts
                ]
                if conflicts
                else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync code to nodes: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync code to nodes: {str(e)}",
        )


@router.post("/resolve-conflict")
async def resolve_conflict(request: dict) -> dict:
    """Resolve synchronization conflict between code and nodes.

    When a conflict is detected between code changes and node changes,
    this endpoint resolves it based on the chosen strategy.

    Request body:
        {
            "agent_name": str,
            "strategy": "code_wins" | "nodes_win" | "manual",
            "code": str,  # Current code content
            "nodes": {...}  # Current node graph
        }

    Returns:
        {
            "success": bool,
            "resolved_code": str,  # Resolved code content
            "resolved_nodes": {...},  # Resolved node graph
            "strategy_applied": str,
            "error": str | null
        }

    Strategies:
        - code_wins: Keep code changes, regenerate nodes from code
        - nodes_win: Keep node changes, regenerate code from nodes
        - manual: Return both unchanged for manual resolution
    """
    from web_helper.backend.services.sync_coordinator import (
        ResolutionStrategy,
        get_sync_coordinator,
    )

    try:
        agent_name = request.get("agent_name")
        strategy_str = request.get("strategy", "code_wins")
        code = request.get("code", "")
        nodes = request.get("nodes", {})

        if not agent_name:
            raise HTTPException(status_code=400, detail="agent_name is required")

        # Parse strategy
        try:
            strategy = ResolutionStrategy(strategy_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy: {strategy_str}. Must be one of: code_wins, nodes_win, manual",
            )

        coordinator = get_sync_coordinator()

        # Get resolution signals
        resolved_code, resolved_nodes = coordinator.resolve_conflict(
            strategy, code, nodes
        )

        # Apply the resolution
        if strategy == ResolutionStrategy.CODE_WINS:
            # Regenerate nodes from code
            graph = hierarchical_builder.build_for_path(agent_name, "")
            resolved_nodes = graph.model_dump()
            resolved_code = code  # Keep original code

            # Mark as synced
            coordinator.mark_synced(agent_name, code, resolved_nodes)

        elif strategy == ResolutionStrategy.NODES_WIN:
            # Regenerate code from nodes
            from web_helper.backend.services.code_generator import get_code_generator

            agent_file = Path.cwd() / "cvlabkit" / "agent" / f"{agent_name}.py"
            original_code = ""
            if agent_file.exists():
                original_code = agent_file.read_text()

            generator = get_code_generator()
            generated_code = generator.generate_from_nodes(
                nodes,
                original_code=original_code,
            )

            if not generated_code:
                return {
                    "success": False,
                    "error": "Failed to generate code from nodes",
                }

            resolved_code = generated_code
            resolved_nodes = nodes  # Keep original nodes

            # Write the generated code
            if resolved_code:
                agent_file.write_text(resolved_code)

            # Mark as synced
            coordinator.mark_synced(agent_name, resolved_code, nodes)

        elif strategy == ResolutionStrategy.MANUAL:
            # Return both unchanged
            resolved_code = code
            resolved_nodes = nodes

        else:
            # Fallback to CODE_WINS
            graph = hierarchical_builder.build_for_path(agent_name, "")
            resolved_nodes = graph.model_dump()
            resolved_code = code
            coordinator.mark_synced(agent_name, code, resolved_nodes)

        return {
            "success": True,
            "resolved_code": resolved_code,
            "resolved_nodes": resolved_nodes,
            "strategy_applied": strategy.value,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve conflict: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve conflict: {str(e)}",
        )


@router.post("/generate-from-nodes")
async def generate_code_from_nodes(request: dict) -> dict:
    """Generate code from node graph.

    Uses NodeToASTConverter for precise AST generation and CodeMerger
    to preserve user edits using LibCST.

    Request body:
        {
            "agent_name": str,
            "node_graph": {...},  # With nodes and edges
            "original_code": str | null,  # To preserve user edits
            "preserve_sections": [[start, end], ...]  # Line ranges to preserve
            "merge_mode": "replace" | "incremental" | "smart"  # Merge strategy
        }

    Returns:
        {
            "success": bool,
            "code": str,  # Generated Python code
            "modified_lines": [int],  # Lines that were changed
            "preserved_regions": [...],  # Regions preserved from user code
            "warnings": [str] | null
        }
    """
    from web_helper.backend.services.code_generator import get_code_generator
    from web_helper.backend.services.code_merger import merge_code, MergeMode
    from web_helper.backend.services.sync_coordinator import get_sync_coordinator
    from web_helper.backend.models.node_system import CodeNodeMapping

    try:
        agent_name = request.get("agent_name")
        node_graph = request.get("node_graph", {})
        original_code = request.get("original_code")
        preserve_sections = request.get("preserve_sections", [])
        merge_mode_str = request.get("merge_mode", "smart")

        if not agent_name:
            raise HTTPException(status_code=400, detail="agent_name is required")

        # Parse merge mode
        try:
            merge_mode = MergeMode(merge_mode_str)
        except ValueError:
            merge_mode = MergeMode.SMART

        roundtrip = None
        if isinstance(node_graph, dict):
            metadata = node_graph.get("metadata")
            if isinstance(metadata, dict):
                roundtrip = metadata.get("roundtrip")

        if original_code and isinstance(roundtrip, dict):
            code_hash = _hash_code_content(original_code)
            nodes_hash = _hash_nodes_edges(
                node_graph.get("nodes", []),
                node_graph.get("edges", []),
            )
            if (
                roundtrip.get("code_hash") == code_hash
                and roundtrip.get("nodes_hash") == nodes_hash
            ):
                return {
                    "success": True,
                    "code": original_code,
                    "modified_lines": [],
                    "preserved_regions": [],
                    "warnings": [],
                }

        generator = get_code_generator()

        # Generate code from nodes using the generator
        generated_code = generator.generate_from_nodes(node_graph, original_code)

        warnings = []
        preserved_regions = []

        # If original code exists, use CodeMerger for smart merging
        if original_code and merge_mode != MergeMode.REPLACE:
            # Build code-node mappings from the node graph
            mappings = []
            for node in node_graph.get("nodes", []):
                if "source" in node and node["source"]:
                    source = node["source"]
                    mappings.append(
                        CodeNodeMapping(
                            node_id=node.get("id", ""),
                            line_start=source.get("line_start", source.get("line", 0)),
                            line_end=source.get("line_end", source.get("end_line", 0)),
                            code_snippet=source.get("snippet", ""),
                            ast_type=source.get("type", "Unknown"),
                        )
                    )

            # Merge with user code preservation
            merge_result = merge_code(
                original=original_code,
                generated=generated_code,
                mappings=mappings,
                mode=merge_mode,
            )

            if merge_result.success:
                generated_code = merge_result.code
                preserved_regions = [
                    {
                        "start_line": r.start_line,
                        "end_line": r.end_line,
                        "is_user_protected": r.is_user_protected,
                    }
                    for r in merge_result.preserved_regions
                ]
                warnings = merge_result.warnings
            else:
                warnings.append("Code merge failed, using generated code only")

        # Apply manual preserve_sections if provided
        if preserve_sections and original_code:
            generated_code = generator.merge_user_code(
                generated_code,
                original_code,
                [(s[0], s[1]) for s in preserve_sections],
            )

        # Calculate modified lines (simple diff)
        modified_lines = []
        if original_code:
            orig_lines = original_code.split("\n")
            gen_lines = generated_code.split("\n")
            for i, (orig, gen) in enumerate(zip(orig_lines, gen_lines)):
                if orig != gen:
                    modified_lines.append(i + 1)  # 1-based line numbers
            # Also mark new lines
            if len(gen_lines) > len(orig_lines):
                for i in range(len(orig_lines), len(gen_lines)):
                    modified_lines.append(i + 1)

        # Mark as synced
        coordinator = get_sync_coordinator()
        coordinator.mark_synced(agent_name, generated_code, node_graph)

        return {
            "success": True,
            "code": generated_code,
            "modified_lines": modified_lines,
            "preserved_regions": preserved_regions,
            "warnings": warnings if warnings else None,
        }

    except Exception as e:
        logger.error(f"Failed to generate code from nodes: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate code from nodes: {str(e)}",
        )


# =============================================================================
# Legacy Endpoints (kept for backward compatibility)
# =============================================================================
# NOTE: These endpoints will be deprecated in future versions.
# Please migrate to the /hierarchy endpoint.


@router.get("/agent-view/{agent_name}")
async def get_agent_view(agent_name: str) -> dict:
    """[DEPRECATED] Use GET /hierarchy/{agent_name}?path= instead.

    Get Agent View: setup() components + high-level data flow.
    """
    logger.warning(
        f"Deprecated endpoint /agent-view/{agent_name} called. "
        f"Use /hierarchy/{agent_name}?path= instead."
    )
    try:
        graph = hierarchical_builder.build_for_path(agent_name, "")
        return {"success": True, "data": graph.model_dump()}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to build agent view for {agent_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to build agent view: {str(e)}"
        )


@router.get("/drill/{component_id}")
async def drill_component(
    component_id: str,
    agent: str = Query(..., description="Agent name"),
    category: str = Query(None, description="[DEPRECATED] Not needed anymore"),
) -> dict:
    """[DEPRECATED] Use GET /hierarchy/{agent_name}?path={component_id} instead.

    Drill into a component's internal structure.
    """
    logger.warning(
        f"Deprecated endpoint /drill/{component_id} called. "
        f"Use /hierarchy/{agent}?path={component_id} instead."
    )
    try:
        graph = hierarchical_builder.build_for_path(agent, component_id)
        return {"success": True, "data": graph.model_dump()}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to drill into {component_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to drill into component: {str(e)}"
        )


# =============================================================================
# Graph Layout IR Endpoints
# =============================================================================


class SaveLayoutRequest(BaseModel):
    """Request body for saving layout."""

    graph_id: str
    view_mode: str = "both"  # "process" | "data" | "both"
    nodes: list[dict]
    edges: list[dict]
    hierarchy_depth: int = 0
    path: list[str] = []
    viewport: Optional[dict] = None
    metadata: Optional[dict] = None


class LayoutResponse(BaseModel):
    """Response for layout operations."""

    success: bool
    data: Optional[dict] = None
    message: str
    file_path: Optional[str] = None


@router.post("/layouts/{agent_name}")
async def save_layout(
    agent_name: str,
    request: SaveLayoutRequest,
) -> LayoutResponse:
    """Save graph layout to IR file.

    Stores node positions, edge styles, and viewport state for the given agent.

    Storage:
        web_helper/state/<agent>/graph_layout/<view_mode>/<graph_id>.json

    Args:
        agent_name: Agent name.
        request: Layout data including nodes, edges, viewport.

    Returns:
        LayoutResponse with success status and file path.
    """
    from web_helper.backend.services.graph_layout_ir import (
        get_layout_ir_service,
        ViewMode,
    )

    try:
        view_mode = ViewMode(request.view_mode)

        service = get_layout_ir_service()
        result = service.save_layout(
            agent_name=agent_name,
            graph_id=request.graph_id,
            view_mode=view_mode,
            nodes=request.nodes,
            edges=request.edges,
            hierarchy_depth=request.hierarchy_depth,
            path=request.path,
            viewport=request.viewport,
            metadata=request.metadata,
        )

        return LayoutResponse(
            success=result["success"],
            message=result["message"],
            file_path=result.get("file_path"),
        )

    except Exception as e:
        logger.error(f"Failed to save layout for {agent_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save layout: {str(e)}",
        )


@router.get("/layouts/{agent_name}")
async def list_layouts(agent_name: str) -> dict:
    """List all saved layouts for an agent.

    Args:
        agent_name: Agent name.

    Returns:
        {"success": bool, "layouts": [...]}
    """
    from web_helper.backend.services.graph_layout_ir import get_layout_ir_service

    try:
        service = get_layout_ir_service()
        layouts = service.list_layouts(agent_name)

        return {
            "success": True,
            "layouts": layouts,
        }

    except Exception as e:
        logger.error(f"Failed to list layouts for {agent_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list layouts: {str(e)}",
        )


@router.get("/layouts/{agent_name}/{graph_id}")
async def get_layout(
    agent_name: str,
    graph_id: str,
    view_mode: str = Query("both", description="View mode: process, data, or both"),
) -> dict:
    """Get a specific layout for an agent.

    Args:
        agent_name: Agent name.
        graph_id: Graph identifier.
        view_mode: View mode.

    Returns:
        {"success": bool, "layout": {...}}
    """
    from web_helper.backend.services.graph_layout_ir import (
        get_layout_ir_service,
        ViewMode,
    )

    try:
        view_mode_enum = ViewMode(view_mode)

        service = get_layout_ir_service()
        layout = service.load_layout(agent_name, graph_id, view_mode_enum)

        if layout is None:
            return {
                "success": False,
                "layout": None,
                "message": "Layout not found",
            }

        return {
            "success": True,
            "layout": layout.model_dump(),
        }

    except Exception as e:
        logger.error(
            f"Failed to get layout for {agent_name}/{graph_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get layout: {str(e)}",
        )


@router.delete("/layouts/{agent_name}/{graph_id}")
async def delete_layout(
    agent_name: str,
    graph_id: str,
    view_mode: str = Query("both", description="View mode: process, data, or both"),
) -> dict:
    """Delete a layout for an agent.

    Args:
        agent_name: Agent name.
        graph_id: Graph identifier.
        view_mode: View mode.

    Returns:
        {"success": bool, "message": str}
    """
    from web_helper.backend.services.graph_layout_ir import (
        get_layout_ir_service,
        ViewMode,
    )

    try:
        view_mode_enum = ViewMode(view_mode)

        service = get_layout_ir_service()
        result = service.delete_layout(agent_name, graph_id, view_mode_enum)

        return {
            "success": result["success"],
            "message": result["message"],
        }

    except Exception as e:
        logger.error(
            f"Failed to delete layout for {agent_name}/{graph_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete layout: {str(e)}",
        )


# =============================================================================
# Model Layer/Module Drill-down Endpoints
# =============================================================================


@router.get("/model-layers/{agent_name}")
async def get_model_layers(
    agent_name: str,
    impl: str = Query(..., description="Model implementation name (e.g., 'resnet18')"),
) -> dict:
    """Get Level 1 layer graph for a model.

    Extracts `nn.Module` assignments from model `__init__`:
    - self.encoder = nn.Linear(...)
    - self.decoder = nn.Sequential(...)

    This endpoint is called when user drills down into a model node.

    Args:
        agent_name: Agent name (e.g., "classification")
        impl: Model implementation name (e.g., "resnet18", "custom_cnn")

    Returns:
        {
            "success": bool,
            "data": HierarchicalNodeGraph with Level 1 nodes (layers)
        }
    """
    from web_helper.backend.services.model_layer_extractor import ModelLayerExtractor

    try:
        # Get model source from component registry or file
        model_source = _get_model_source(agent_name, impl)

        extractor = ModelLayerExtractor()
        graph = extractor.build_layer_graph(
            model_source=model_source,
            model_name=impl,
            agent_name=agent_name,
        )

        return {
            "success": True,
            "data": graph.model_dump(mode="json"),
        }

    except FileNotFoundError as e:
        logger.warning(f"Model source not found: {agent_name}/{impl}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(
            f"Failed to extract layers for {agent_name}/{impl}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract layers: {str(e)}",
        )


@router.get("/model-layers/{agent_name}/{layer_name}/modules")
async def get_layer_modules(
    agent_name: str,
    layer_name: str,
    impl: str = Query(..., description="Model implementation name"),
    source_code: str = Query("", description="Layer source code for nested extraction"),
) -> dict:
    """Get Level 2 nested modules for a layer.

    Extracts nested `nn.*` calls within a layer:
    - nn.Sequential(nn.Linear(128, 64), nn.ReLU())
    - ResNetBlock(in_channels=64, out_channels=64) with internal conv/bn

    Args:
        agent_name: Agent name
        layer_name: Layer name (e.g., "encoder", "classifier")
        impl: Model implementation name
        source_code: Optional source code of the layer (auto-fetched if empty)

    Returns:
        {
            "success": bool,
            "data": HierarchicalNodeGraph with Level 2 nodes (nested modules)
        }
    """
    from web_helper.backend.services.model_layer_extractor import ModelLayerExtractor

    try:
        # Get layer source code
        if not source_code:
            source_code = _get_layer_source(agent_name, impl, layer_name)

        extractor = ModelLayerExtractor()
        graph = extractor.build_module_graph(
            layer_source=source_code,
            layer_name=layer_name,
            model_name=impl,
            agent_name=agent_name,
        )

        return {
            "success": True,
            "data": graph.model_dump(mode="json"),
        }

    except FileNotFoundError as e:
        logger.warning(f"Layer source not found: {agent_name}/{impl}/{layer_name}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(
            f"Failed to extract modules for {agent_name}/{impl}/{layer_name}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract modules: {str(e)}",
        )


def _get_model_source(agent_name: str, impl: str) -> str:
    """Get the source code for a model implementation.

    Searches in cvlabkit/component/model/ directory.

    Args:
        agent_name: Agent name
        impl: Model implementation name (e.g., "resnet18")

    Returns:
        Source code string

    Raises:
        FileNotFoundError: If model file not found
    """
    # Try to find model in component directory
    model_dir = Path.cwd() / "cvlabkit" / "component" / "model"

    if model_dir.exists():
        # Check for exact match first
        model_file = model_dir / f"{impl}.py"
        if model_file.exists():
            return model_file.read_text(encoding="utf-8")

        # Check for _ naming convention (e.g., resnet18 -> resnet18.py)
        model_file = model_dir / f"{impl}.py"
        if model_file.exists():
            return model_file.read_text(encoding="utf-8")

    # Fallback: raise not found
    raise FileNotFoundError(f"Model implementation '{impl}' not found in {model_dir}")


def _get_layer_source(agent_name: str, impl: str, layer_name: str) -> str:
    """Get the source code for a specific layer.

    For now, returns a placeholder. In the full implementation,
    this would extract the layer's class definition from the model source.

    Args:
        agent_name: Agent name
        impl: Model implementation name
        layer_name: Layer name

    Returns:
        Source code string for the layer

    Raises:
        FileNotFoundError: If layer source not found
    """
    # Get model source first
    model_source = _get_model_source(agent_name, impl)

    # Extract layer class definition from model source
    try:
        import ast

        tree = ast.parse(model_source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == layer_name:
                # Get source lines for this class
                lines = model_source.split("\n")
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])

    except SyntaxError:
        pass

    raise FileNotFoundError(f"Layer '{layer_name}' not found in model '{impl}'")
