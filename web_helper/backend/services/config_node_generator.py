"""Config Node Generator - Creates YAML Config nodes for Phase 3 Config Binding.

This service generates a `yamlConfig` type node that represents a YAML config file.
When connected to component nodes, it auto-binds the implementation values.

Example:
    Config file: model: resnet18, optimizer: adam

    Generated node:
    ┌─────────────────────────────────────────┐
    │  [Config]  example.yaml                 │
    │                                         │
    │  model: resnet18        ─● model        │
    │  optimizer: adam        ─● optimizer    │
    │  loss: cross_entropy    ─● loss         │
    └─────────────────────────────────────────┘

    Edges: config.model → model node, config.optimizer → optimizer node, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

# Custom YAML loader that handles !!python/tuple and other Python types
def _python_tuple_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> tuple:
    """Handle !!python/tuple tag in YAML files."""
    return tuple(loader.construct_sequence(node))

def _create_yaml_loader() -> type[yaml.SafeLoader]:
    """Create a YAML loader with Python type support."""
    loader = yaml.SafeLoader
    loader.add_constructor("tag:yaml.org,2002:python/tuple", _python_tuple_constructor)
    return loader


from web_helper.backend.models.hierarchy import (
    CodeFlowEdge,
    ComponentCategory,
    FlowType,
    HierarchyLevel,
    HierarchyNode,
    NodeOrigin,
    OriginType,
    Port,
    PropertyInfo,
    PropertySummary,
    SourceLocation,
    ValueSource,
)

logger = logging.getLogger(__name__)

# Component categories that can have implementation values from config
CONFIG_CATEGORIES = [
    "model",
    "optimizer",
    "scheduler",
    "loss",
    "dataset",
    "dataloader",
    "transform",
    "metric",
    "sampler",
]


class ConfigNodeGenerator:
    """Generates YAML Config nodes for the node graph."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the generator.

        Args:
            project_root: Project root directory for resolving config paths.
        """
        self.project_root = project_root or Path.cwd()

    def generate_config_node(
        self,
        config_path: str | None = None,
        config_data: dict[str, Any] | None = None,
        agent_name: str | None = None,
    ) -> tuple[HierarchyNode, list[PropertyInfo]]:
        """Generate a YAML Config node.

        Args:
            config_path: Path to the YAML config file (relative or absolute).
            config_data: Pre-loaded config data (if config_path is None).
            agent_name: Agent name to find matching config.

        Returns:
            Tuple of (HierarchyNode, properties list).
        """
        # Load config if needed
        if config_data is None:
            config_data = self._load_config(config_path, agent_name)

        # Build output ports for ALL top-level keys
        outputs: list[Port] = []
        properties: list[PropertyInfo] = []

        for key, value in config_data.items():
            # Add output port for this key
            outputs.append(Port(
                name=key,
                type="config",
            ))

            # Add property
            properties.append(PropertyInfo(
                name=key,
                value=value,
                source=ValueSource.CONFIG,
                is_grid_search=isinstance(value, list),
                grid_count=len(value) if isinstance(value, list) else None,
                description=f"Config value for {key}",
            ))

        # Build property summary
        property_summary = PropertySummary(
            required_count=0,
            config_count=len(properties),
            default_count=0,
            hardcode_count=0,
            connected_count=0,
        )

        # Determine config file name for label
        config_name = "config.yaml"
        if config_path:
            config_name = Path(config_path).name
        elif agent_name:
            config_name = f"{agent_name}.yaml"

        # Create the config node
        node = HierarchyNode(
            id="yaml_config",
            label=config_name,
            level=HierarchyLevel.COMPONENT,
            origin=NodeOrigin(
                type=OriginType.INPUT,
                code_snippet=f"# Config: {config_name}",
            ),
            can_drill=False,
            drill_target=None,
            category=ComponentCategory.GLOBAL,
            inputs=[],
            outputs=outputs,
            source=SourceLocation(
                file=f"config/{config_name}" if config_path is None else config_path,
                line=1,
            ),
            properties=properties,
            property_summary=property_summary,
            metadata={
                "nodeType": "yamlConfig",
                "configPath": config_path or f"config/{config_name}",
                "configData": config_data,
            },
        )

        return node, properties

    def generate_config_edges(
        self,
        config_node: HierarchyNode,
        component_nodes: list[HierarchyNode],
    ) -> list[CodeFlowEdge]:
        """Generate edges from config node to component nodes.
        
        Requirement: No data connections from Config lines. 
        We only return an empty list here to satisfy the interface while removing the lines.
        """
        return []

    def _load_config(
        self,
        config_path: str | None,
        agent_name: str | None,
    ) -> dict[str, Any]:
        """Load config from file or find matching config for agent.

        Args:
            config_path: Direct path to config file.
            agent_name: Agent name to search for matching config.

        Returns:
            Loaded config dictionary.
        """
        yaml_loader = _create_yaml_loader()

        if config_path:
            # Load from specified path
            full_path = Path(config_path)
            if not full_path.is_absolute():
                full_path = self.project_root / config_path

            if full_path.exists():
                return yaml.load(full_path.read_text(encoding="utf-8"), Loader=yaml_loader) or {}

        if agent_name:
            # Try to find config with matching agent name
            config_dir = self.project_root / "config"
            if config_dir.exists():
                for config_file in config_dir.glob("*.yaml"):
                    try:
                        data = yaml.load(config_file.read_text(encoding="utf-8"), Loader=yaml_loader) or {}
                        if data.get("agent") == agent_name:
                            return data
                    except Exception:
                        continue

        # Return empty config if nothing found
        return {}

    def _extract_implementations(
        self,
        config_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract implementation values for each component category.

        Handles both flat and nested config structures:
        - Flat: model: resnet18
        - Nested: dataset: { train: cifar10, val: cifar10 }

        Args:
            config_data: Loaded config dictionary.

        Returns:
            Map of category -> implementation value.
        """
        impl_map: dict[str, Any] = {}

        for category in CONFIG_CATEGORIES:
            value = config_data.get(category)

            if value is None:
                continue

            if isinstance(value, dict):
                # Nested config (e.g., dataset: { train: ..., val: ... })
                # Use the first value or a summary
                if "train" in value:
                    impl_map[category] = value["train"]
                elif value:
                    # Use first value
                    impl_map[category] = list(value.values())[0]
            elif isinstance(value, list):
                # Grid search list
                impl_map[category] = value
            elif isinstance(value, str):
                # Simple implementation name (possibly with params)
                # Extract base name: "resnet18(num_classes=10)" -> "resnet18"
                base_name = value.split("(")[0].strip()
                impl_map[category] = base_name
            else:
                # Other values (bool, int, etc.) - probably not an impl
                pass

        return impl_map


def inject_config_node(
    graph_nodes: list[HierarchyNode],
    graph_edges: list[CodeFlowEdge],
    config_path: str | None = None,
    agent_name: str | None = None,
    project_root: Path | None = None,
) -> tuple[list[HierarchyNode], list[CodeFlowEdge]]:
    """Inject a YAML config node into an existing graph.

    This is a convenience function for adding config binding to an existing graph.

    Args:
        graph_nodes: Existing nodes in the graph.
        graph_edges: Existing edges in the graph.
        config_path: Path to the config file.
        agent_name: Agent name to find matching config.
        project_root: Project root directory.

    Returns:
        Tuple of (updated nodes list, updated edges list).
    """
    generator = ConfigNodeGenerator(project_root)

    # Generate config node
    config_node, _ = generator.generate_config_node(
        config_path=config_path,
        agent_name=agent_name,
    )

    # Check if config has any implementations
    if not config_node.outputs:
        # No implementations found, don't inject
        return graph_nodes, graph_edges

    # Generate edges from config to components
    config_edges = generator.generate_config_edges(config_node, graph_nodes)

    # Update component nodes with connected status
    updated_nodes: list[HierarchyNode] = [config_node]
    connected_node_ids = {e.target_node for e in config_edges}

    for node in graph_nodes:
        if node.id in connected_node_ids:
            # Update property source to CONNECTED
            updated_properties = []
            for prop in (node.properties or []):
                if prop.name == "impl":
                    updated_properties.append(PropertyInfo(
                        name=prop.name,
                        value=prop.value,
                        source=ValueSource.CONNECTED,
                        is_grid_search=prop.is_grid_search,
                        grid_count=prop.grid_count,
                        default_value=prop.default_value,
                        description=prop.description,
                    ))
                else:
                    updated_properties.append(prop)

            # Update property summary
            summary = node.property_summary
            if summary:
                updated_summary = PropertySummary(
                    required_count=max(0, summary.required_count - 1),
                    config_count=summary.config_count,
                    default_count=summary.default_count,
                    hardcode_count=summary.hardcode_count,
                    connected_count=summary.connected_count + 1,
                )
            else:
                updated_summary = None

            updated_node = HierarchyNode(
                id=node.id,
                label=node.label,
                level=node.level,
                origin=node.origin,
                can_drill=node.can_drill,
                drill_target=node.drill_target,
                category=node.category,
                inputs=node.inputs,
                outputs=node.outputs,
                source=node.source,
                properties=updated_properties if updated_properties else node.properties,
                property_summary=updated_summary,
                metadata=node.metadata,
            )
            updated_nodes.append(updated_node)
        else:
            updated_nodes.append(node)

    # Combine edges
    updated_edges = graph_edges + config_edges

    return updated_nodes, updated_edges
