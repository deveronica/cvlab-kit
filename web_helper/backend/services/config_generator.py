"""
Config Generator - Convert node graph to CVLab-Kit YAML configuration.

This module generates YAML configuration from a visual node graph representation.
It follows CVLab-Kit's configuration syntax.

Generated config structure:
    agent: classification
    epochs: 100
    model: resnet18
    optimizer: adam(lr=0.001)
    loss: cross_entropy
    dataset: cifar10
    transform:
      weak: resize(size=224)
      strong: augment
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Models for Config Generation
# =============================================================================


@dataclass
class ConfigNode:
    """Node representation for config generation."""

    id: str
    label: str
    category: str
    # Component info
    component_name: str  # Implementation name (e.g., "resnet18", "adam")
    variant: Optional[str] = None  # For multi-variant categories (e.g., "weak", "strong")
    kwargs: dict = field(default_factory=dict)  # Parameters


class ConfigGenerationRequest(BaseModel):
    """Request model for config generation API."""

    agent_name: str = "classification"
    nodes: list[dict]
    # Additional config fields
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    device: str = "cuda"


class ConfigGenerationResponse(BaseModel):
    """Response model for config generation API."""

    success: bool
    yaml_content: str = ""
    config_dict: dict = {}
    error: Optional[str] = None


# =============================================================================
# Config Generator
# =============================================================================


class ConfigGenerator:
    """Generate CVLab-Kit YAML configuration from node graph."""

    # Categories that support variants (2-depth config)
    VARIANT_CATEGORIES = {"transform", "loss", "dataloader", "dataset"}

    # Categories that support inline parameters
    PARAM_CATEGORIES = {"optimizer", "scheduler", "transform", "loss", "model"}

    def generate(self, request: ConfigGenerationRequest) -> ConfigGenerationResponse:
        """Generate YAML configuration from node graph."""
        try:
            # Parse nodes
            nodes = self._parse_nodes(request.nodes)

            # Build config dict
            config = self._build_config(nodes, request)

            # Convert to YAML
            yaml_content = yaml.dump(
                config,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

            return ConfigGenerationResponse(
                success=True,
                yaml_content=yaml_content,
                config_dict=config,
            )

        except Exception as e:
            logger.error(f"Config generation failed: {e}", exc_info=True)
            return ConfigGenerationResponse(
                success=False,
                error=str(e),
            )

    def _parse_nodes(self, raw_nodes: list[dict]) -> list[ConfigNode]:
        """Parse raw node data into ConfigNode objects."""
        nodes = []
        for raw in raw_nodes:
            data = raw.get("data", {})
            metadata = data.get("metadata", {})

            # Skip flow nodes (only setup nodes generate config)
            if metadata.get("isFlowNode"):
                continue

            category = data.get("category", "unknown")
            label = data.get("label", "")

            # Get component name from metadata or label
            component_name = metadata.get("componentName", label)

            # Get variant (for 2-depth categories)
            variant = metadata.get("variant")

            # Get kwargs
            kwargs = metadata.get("kwargs", {})

            node = ConfigNode(
                id=raw.get("id", ""),
                label=label,
                category=category,
                component_name=component_name,
                variant=variant,
                kwargs=kwargs,
            )
            nodes.append(node)

        return nodes

    def _build_config(
        self, nodes: list[ConfigNode], request: ConfigGenerationRequest
    ) -> dict[str, Any]:
        """Build configuration dictionary from nodes."""
        config: dict[str, Any] = {}

        # Add agent name
        config["agent"] = request.agent_name

        # Add global parameters
        config["epochs"] = request.epochs
        config["batch_size"] = request.batch_size
        config["lr"] = request.lr
        config["device"] = request.device

        # Group nodes by category
        category_nodes: dict[str, list[ConfigNode]] = {}
        for node in nodes:
            if node.category not in category_nodes:
                category_nodes[node.category] = []
            category_nodes[node.category].append(node)

        # Process each category
        for category, cat_nodes in category_nodes.items():
            config_value = self._build_category_config(category, cat_nodes)
            if config_value is not None:
                config[category] = config_value

        return config

    def _build_category_config(
        self, category: str, nodes: list[ConfigNode]
    ) -> Any:
        """Build config for a single category."""
        if not nodes:
            return None

        # Check if this category supports variants
        has_variants = any(n.variant for n in nodes)

        if has_variants or (category in self.VARIANT_CATEGORIES and len(nodes) > 1):
            # 2-depth config with variants
            result = {}
            for node in nodes:
                key = node.variant or node.label.split("_")[0] if "_" in node.label else "default"
                result[key] = self._format_component_value(node)
            return result

        elif len(nodes) == 1:
            # Single component
            return self._format_component_value(nodes[0])

        else:
            # Multiple components without variants - use list
            return [self._format_component_value(n) for n in nodes]

    def _format_component_value(self, node: ConfigNode) -> str:
        """Format component as config value (name or name(params))."""
        name = node.component_name

        # Build parameter string
        if node.kwargs and node.category in self.PARAM_CATEGORIES:
            params = []
            for key, value in node.kwargs.items():
                # Format value appropriately
                if isinstance(value, str) and not value.replace(".", "").replace("-", "").isdigit():
                    params.append(f'{key}="{value}"')
                else:
                    params.append(f"{key}={value}")

            if params:
                return f"{name}({', '.join(params)})"

        return name


# =============================================================================
# YAML Parser (for loading existing configs)
# =============================================================================


class ConfigParser:
    """Parse CVLab-Kit YAML configuration into node representation."""

    def parse(self, yaml_content: str) -> list[dict]:
        """Parse YAML config into node-like structures."""
        try:
            config = yaml.safe_load(yaml_content)
            if not isinstance(config, dict):
                return []

            nodes = []

            # Skip non-component keys
            skip_keys = {"agent", "epochs", "batch_size", "lr", "device", "project", "run_name"}

            for key, value in config.items():
                if key in skip_keys:
                    continue

                nodes.extend(self._parse_category(key, value))

            return nodes

        except Exception as e:
            logger.error(f"Failed to parse config: {e}", exc_info=True)
            return []

    def _parse_category(self, category: str, value: Any) -> list[dict]:
        """Parse a single category from config."""
        nodes = []

        if isinstance(value, dict):
            # 2-depth config with variants
            for variant, comp_value in value.items():
                node = self._create_node(category, comp_value, variant=variant)
                if node:
                    nodes.append(node)

        elif isinstance(value, list):
            # List of components
            for comp_value in value:
                node = self._create_node(category, comp_value)
                if node:
                    nodes.append(node)

        elif isinstance(value, str):
            # Single component
            node = self._create_node(category, value)
            if node:
                nodes.append(node)

        return nodes

    def _create_node(
        self, category: str, value: str, variant: Optional[str] = None
    ) -> Optional[dict]:
        """Create node representation from config value."""
        if not isinstance(value, str):
            return None

        # Parse component name and params: "name(param=value)"
        component_name = value
        kwargs = {}

        if "(" in value and value.endswith(")"):
            name_part = value[: value.index("(")]
            params_part = value[value.index("(") + 1 : -1]

            component_name = name_part

            # Parse parameters
            for param in params_part.split(","):
                param = param.strip()
                if "=" in param:
                    key, val = param.split("=", 1)
                    kwargs[key.strip()] = val.strip().strip('"').strip("'")

        # Create node-like structure
        node_id = f"{category}_{variant}_{component_name}" if variant else f"{category}_{component_name}"

        return {
            "id": node_id,
            "data": {
                "label": f"{variant}_{component_name}" if variant else component_name,
                "category": category,
                "metadata": {
                    "componentName": component_name,
                    "variant": variant,
                    "kwargs": kwargs,
                    "isSetupNode": True,
                },
            },
        }


# =============================================================================
# Singleton instances
# =============================================================================

_generator: ConfigGenerator | None = None
_parser: ConfigParser | None = None


def get_config_generator() -> ConfigGenerator:
    """Get singleton ConfigGenerator instance."""
    global _generator
    if _generator is None:
        _generator = ConfigGenerator()
    return _generator


def get_config_parser() -> ConfigParser:
    """Get singleton ConfigParser instance."""
    global _parser
    if _parser is None:
        _parser = ConfigParser()
    return _parser


def generate_yaml_config(request: ConfigGenerationRequest) -> ConfigGenerationResponse:
    """Convenience function to generate YAML config."""
    return get_config_generator().generate(request)


def parse_yaml_config(yaml_content: str) -> list[dict]:
    """Convenience function to parse YAML config."""
    return get_config_parser().parse(yaml_content)
