"""
Node Registry System - Extensible category, node types, and analyzer management

This module provides:
1. INodeCategory: Interface for defining node categories with themes
2. IComponentAnalyzer: Interface for component analysis (drill-down)
3. NodeRegistry: Singleton registry for categories, node types, and analyzers
4. Method definitions: Callable methods on node outputs

Design Goals:
- Remove hardcoded category lists from frontend/backend
- Allow dynamic category registration
- Enable custom analyzers for new component types
- Provide consistent theming across the system
- Support method configuration in node GUI (e.g., .parameters(), .eval())
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any

from web_helper.backend.models.hierarchy import (
    HierarchicalNodeGraph,
    HierarchyNode,
    HierarchyLevel,
    ComponentCategory,
)
from web_helper.backend.models.node_system import (
    DataType,
    PortDefinition,
    MethodDefinition,
    NodeType,
    NodeTypeDefinition,
    FlowType,
)


# =============================================================================
# Category Theme Definition
# =============================================================================


@dataclass
class CategoryTheme:
    """Visual theme for a node category"""

    background: str  # Tailwind bg class (e.g., "blue-50")
    border: str  # Tailwind border class (e.g., "blue-200")
    text: str  # Tailwind text class (e.g., "blue-700")
    icon: str  # Lucide icon name (e.g., "cube")

    # Dark mode variants (optional - will be auto-generated if not provided)
    dark_background: Optional[str] = None
    dark_border: Optional[str] = None
    dark_text: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for API response"""
        return {
            "background": f"bg-{self.background} dark:bg-{self.dark_background or self.background.replace('-50', '-950/30')}",
            "border": f"border-{self.border} dark:border-{self.dark_border or self.border.replace('-200', '-800')}",
            "text": f"text-{self.text} dark:text-{self.dark_text or self.text.replace('-700', '-300')}",
            "icon": self.icon,
        }


# =============================================================================
# Category Interface
# =============================================================================


@dataclass
class NodeCategoryInfo:
    """Information about a registered node category"""

    name: str  # Unique category identifier (e.g., "model")
    label: str  # Display label (e.g., "Model")
    theme: CategoryTheme
    drillable: bool = False  # Can drill-down into this category
    description: str = ""  # Optional description


# =============================================================================
# Component Analyzer Interface
# =============================================================================


class IComponentAnalyzer(ABC):
    """Interface for analyzing component internals (drill-down)

    Implementations analyze specific component types (model, loss, transform)
    and produce HierarchicalNodeGraph representing their internal structure.
    """

    @property
    @abstractmethod
    def category(self) -> str:
        """Category this analyzer handles (e.g., 'model', 'loss')"""
        pass

    @abstractmethod
    def analyze(
        self,
        component_path: Path,
        impl_name: Optional[str] = None,
        parent_path: list[str] | None = None,
    ) -> HierarchicalNodeGraph:
        """Analyze component and return its internal structure

        Args:
            component_path: Path to component source file
            impl_name: Implementation class name (e.g., 'ResNet18')
            parent_path: Breadcrumb path from parent

        Returns:
            HierarchicalNodeGraph with nodes and edges
        """
        pass

    def can_analyze(self, component_path: Path) -> bool:
        """Check if this analyzer can handle the given component

        Default: check if file exists
        Override for more complex checks
        """
        return component_path.exists()


# =============================================================================
# Built-in Categories
# =============================================================================


# Default themes for known categories
BUILTIN_CATEGORIES: dict[str, NodeCategoryInfo] = {
    "model": NodeCategoryInfo(
        name="model",
        label="Model",
        theme=CategoryTheme(
            background="blue-50",
            border="blue-200",
            text="blue-700",
            icon="cube",
        ),
        drillable=True,
        description="Neural network models",
    ),
    "optimizer": NodeCategoryInfo(
        name="optimizer",
        label="Optimizer",
        theme=CategoryTheme(
            background="green-50",
            border="green-200",
            text="green-700",
            icon="trending-up",
        ),
        drillable=False,
        description="Optimization algorithms",
    ),
    "loss": NodeCategoryInfo(
        name="loss",
        label="Loss",
        theme=CategoryTheme(
            background="red-50",
            border="red-200",
            text="red-700",
            icon="target",
        ),
        drillable=True,
        description="Loss functions",
    ),
    "dataset": NodeCategoryInfo(
        name="dataset",
        label="Dataset",
        theme=CategoryTheme(
            background="purple-50",
            border="purple-200",
            text="purple-700",
            icon="database",
        ),
        drillable=False,
        description="Data sources",
    ),
    "dataloader": NodeCategoryInfo(
        name="dataloader",
        label="DataLoader",
        theme=CategoryTheme(
            background="indigo-50",
            border="indigo-200",
            text="indigo-700",
            icon="loader",
        ),
        drillable=False,
        description="Batch loaders",
    ),
    "transform": NodeCategoryInfo(
        name="transform",
        label="Transform",
        theme=CategoryTheme(
            background="amber-50",
            border="amber-200",
            text="amber-700",
            icon="wand",
        ),
        drillable=True,
        description="Data transforms",
    ),
    "metric": NodeCategoryInfo(
        name="metric",
        label="Metric",
        theme=CategoryTheme(
            background="cyan-50",
            border="cyan-200",
            text="cyan-700",
            icon="bar-chart",
        ),
        drillable=False,
        description="Evaluation metrics",
    ),
    "sampler": NodeCategoryInfo(
        name="sampler",
        label="Sampler",
        theme=CategoryTheme(
            background="pink-50",
            border="pink-200",
            text="pink-700",
            icon="shuffle",
        ),
        drillable=False,
        description="Data samplers",
    ),
    "scheduler": NodeCategoryInfo(
        name="scheduler",
        label="Scheduler",
        theme=CategoryTheme(
            background="orange-50",
            border="orange-200",
            text="orange-700",
            icon="clock",
        ),
        drillable=False,
        description="Learning rate schedulers",
    ),
    "logger": NodeCategoryInfo(
        name="logger",
        label="Logger",
        theme=CategoryTheme(
            background="slate-50",
            border="slate-200",
            text="slate-700",
            icon="file-text",
        ),
        drillable=False,
        description="Logging utilities",
    ),
}

# Default theme for unknown categories
DEFAULT_CATEGORY = NodeCategoryInfo(
    name="unknown",
    label="Unknown",
    theme=CategoryTheme(
        background="gray-50",
        border="gray-200",
        text="gray-700",
        icon="help-circle",
    ),
    drillable=False,
    description="Unknown category",
)


# =============================================================================
# Node Registry (Singleton)
# =============================================================================


class NodeRegistry:
    """Central registry for node categories and component analyzers

    Singleton pattern ensures consistent state across the application.

    Usage:
        registry = NodeRegistry.instance()
        registry.register_category(my_category)
        registry.register_analyzer(my_analyzer)
        theme = registry.get_theme("model")
        graph = registry.analyze_component("model", path, impl)
    """

    _instance: Optional["NodeRegistry"] = None

    def __new__(cls) -> "NodeRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._categories: dict[str, NodeCategoryInfo] = {}
        self._analyzers: dict[str, IComponentAnalyzer] = {}

        # Register built-in categories
        for name, info in BUILTIN_CATEGORIES.items():
            self._categories[name] = info

        self._initialized = True

    @classmethod
    def instance(cls) -> "NodeRegistry":
        """Get the singleton instance"""
        return cls()

    # =========================================================================
    # Category Management
    # =========================================================================

    def register_category(self, info: NodeCategoryInfo) -> None:
        """Register a new category or update existing one"""
        self._categories[info.name] = info

    def unregister_category(self, name: str) -> bool:
        """Unregister a category (returns True if existed)"""
        if name in self._categories:
            del self._categories[name]
            return True
        return False

    def get_category(self, name: str) -> NodeCategoryInfo:
        """Get category info (returns default for unknown categories)"""
        return self._categories.get(name, DEFAULT_CATEGORY)

    def get_theme(self, name: str) -> CategoryTheme:
        """Get theme for a category"""
        return self.get_category(name).theme

    def is_drillable(self, name: str) -> bool:
        """Check if category supports drill-down"""
        return self.get_category(name).drillable

    def list_categories(self) -> list[NodeCategoryInfo]:
        """List all registered categories"""
        return list(self._categories.values())

    def to_component_category(self, name: str) -> ComponentCategory:
        """Convert category name to ComponentCategory enum"""
        try:
            return ComponentCategory(name)
        except ValueError:
            return ComponentCategory.UNKNOWN

    # =========================================================================
    # Analyzer Management
    # =========================================================================

    def register_analyzer(self, analyzer: IComponentAnalyzer) -> None:
        """Register a component analyzer"""
        self._analyzers[analyzer.category] = analyzer

    def unregister_analyzer(self, category: str) -> bool:
        """Unregister an analyzer (returns True if existed)"""
        if category in self._analyzers:
            del self._analyzers[category]
            return True
        return False

    def get_analyzer(self, category: str) -> Optional[IComponentAnalyzer]:
        """Get analyzer for a category"""
        return self._analyzers.get(category)

    def has_analyzer(self, category: str) -> bool:
        """Check if an analyzer is registered for a category"""
        return category in self._analyzers

    def analyze_component(
        self,
        category: str,
        component_path: Path,
        impl_name: Optional[str] = None,
        parent_path: list[str] | None = None,
    ) -> Optional[HierarchicalNodeGraph]:
        """Analyze a component using the registered analyzer

        Args:
            category: Component category
            component_path: Path to component source
            impl_name: Implementation class name
            parent_path: Breadcrumb path

        Returns:
            HierarchicalNodeGraph or None if no analyzer registered
        """
        analyzer = self.get_analyzer(category)
        if analyzer is None:
            return None

        if not analyzer.can_analyze(component_path):
            return None

        return analyzer.analyze(component_path, impl_name, parent_path)

    # =========================================================================
    # Serialization (for API)
    # =========================================================================

    def to_api_response(self) -> dict[str, Any]:
        """Convert registry state to API-friendly format"""
        return {
            "categories": [
                {
                    "name": info.name,
                    "label": info.label,
                    "theme": info.theme.to_dict(),
                    "drillable": info.drillable,
                    "description": info.description,
                    "has_analyzer": self.has_analyzer(info.name),
                }
                for info in self._categories.values()
            ],
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def get_registry() -> NodeRegistry:
    """Get the global node registry instance"""
    return NodeRegistry.instance()


def get_category_theme(category: str) -> dict[str, str]:
    """Get theme for a category as a dictionary"""
    return get_registry().get_theme(category).to_dict()


def is_category_drillable(category: str) -> bool:
    """Check if a category supports drill-down"""
    return get_registry().is_drillable(category)


# =============================================================================
# Method Definitions - Reusable method templates for node outputs
# =============================================================================

# Model/Module methods
METHOD_PARAMETERS = MethodDefinition(
    name="parameters",
    returns=DataType.PARAMS,
    description="Get model parameters iterator"
)

METHOD_NAMED_PARAMETERS = MethodDefinition(
    name="named_parameters",
    returns=DataType.PARAMS,
    description="Get named model parameters"
)

METHOD_EVAL = MethodDefinition(
    name="eval",
    returns=DataType.MODULE,
    description="Set model to evaluation mode"
)

METHOD_TRAIN = MethodDefinition(
    name="train",
    returns=DataType.MODULE,
    description="Set model to training mode"
)

METHOD_TO = MethodDefinition(
    name="to",
    returns=DataType.MODULE,
    args=["device"],
    description="Move model to device"
)

METHOD_CUDA = MethodDefinition(
    name="cuda",
    returns=DataType.MODULE,
    description="Move model to CUDA"
)

METHOD_CPU = MethodDefinition(
    name="cpu",
    returns=DataType.MODULE,
    description="Move model to CPU"
)

METHOD_STATE_DICT = MethodDefinition(
    name="state_dict",
    returns=DataType.DICT,
    description="Get model state dictionary"
)

METHOD_LOAD_STATE_DICT = MethodDefinition(
    name="load_state_dict",
    returns=DataType.MODULE,
    args=["state_dict"],
    description="Load model state dictionary"
)

METHOD_ZERO_GRAD = MethodDefinition(
    name="zero_grad",
    returns=DataType.ANY,
    description="Zero gradients"
)

# Tensor methods
METHOD_BACKWARD = MethodDefinition(
    name="backward",
    returns=DataType.ANY,
    description="Compute gradients"
)

METHOD_ITEM = MethodDefinition(
    name="item",
    returns=DataType.SCALAR,
    description="Get Python scalar value"
)

METHOD_DETACH = MethodDefinition(
    name="detach",
    returns=DataType.TENSOR,
    description="Detach from computation graph"
)

METHOD_CLONE = MethodDefinition(
    name="clone",
    returns=DataType.TENSOR,
    description="Clone tensor"
)

# Optimizer methods
METHOD_STEP = MethodDefinition(
    name="step",
    returns=DataType.ANY,
    description="Perform optimization step"
)

# Metric methods
METHOD_UPDATE = MethodDefinition(
    name="update",
    returns=DataType.ANY,
    args=["pred", "target"],
    description="Update metric state"
)

METHOD_COMPUTE = MethodDefinition(
    name="compute",
    returns=DataType.SCALAR,
    description="Compute metric value"
)

METHOD_RESET = MethodDefinition(
    name="reset",
    returns=DataType.ANY,
    description="Reset metric state"
)


# =============================================================================
# Category Methods Registry - Methods available per category
# =============================================================================

CATEGORY_METHODS: dict[str, list[MethodDefinition]] = {
    "model": [
        METHOD_PARAMETERS,
        METHOD_NAMED_PARAMETERS,
        METHOD_EVAL,
        METHOD_TRAIN,
        METHOD_TO,
        METHOD_CUDA,
        METHOD_CPU,
        METHOD_STATE_DICT,
        METHOD_LOAD_STATE_DICT,
        METHOD_ZERO_GRAD,
    ],
    "optimizer": [
        METHOD_STEP,
        METHOD_ZERO_GRAD,
        METHOD_STATE_DICT,
        METHOD_LOAD_STATE_DICT,
    ],
    "loss": [
        METHOD_EVAL,
        METHOD_TRAIN,
        METHOD_TO,
    ],
    "dataset": [],
    "dataloader": [],
    "transform": [],
    "metric": [
        METHOD_UPDATE,
        METHOD_COMPUTE,
        METHOD_RESET,
    ],
    "scheduler": [
        METHOD_STEP,
        MethodDefinition(name="get_last_lr", returns=DataType.LIST, description="Get last learning rates"),
    ],
    "sampler": [],
}


def get_methods_for_category(category: str) -> list[MethodDefinition]:
    """Get available methods for a component category"""
    return CATEGORY_METHODS.get(category, [])


def get_method_by_name(category: str, method_name: str) -> Optional[MethodDefinition]:
    """Get a specific method definition by category and name"""
    for method in get_methods_for_category(category):
        if method.name == method_name:
            return method
    return None


# =============================================================================
# Port Definitions - Common port types
# =============================================================================

PORT_IN_TENSOR = PortDefinition(
    name="in",
    data_type=DataType.TENSOR,
    label="Input",
    required=True,
    description="Input tensor"
)

PORT_OUT_TENSOR = PortDefinition(
    name="out",
    data_type=DataType.TENSOR,
    label="Output",
    description="Output tensor"
)

PORT_OUT_MODULE = PortDefinition(
    name="out",
    data_type=DataType.MODULE,
    label="Module",
    description="PyTorch module"
)

PORT_PARAMS = PortDefinition(
    name="params",
    data_type=DataType.PARAMS,
    label="Parameters",
    required=True,
    description="Model parameters iterator"
)

PORT_CONFIG = PortDefinition(
    name="config",
    data_type=DataType.CONFIG,
    label="Config",
    required=False,
    description="Configuration dictionary"
)


# =============================================================================
# Node Type Definitions - Templates for each node type
# =============================================================================

def _create_component_node_def(category: str) -> NodeTypeDefinition:
    """Create NodeTypeDefinition for a component category"""
    cat_info = BUILTIN_CATEGORIES.get(category, DEFAULT_CATEGORY)
    methods = CATEGORY_METHODS.get(category, [])

    # Determine output type based on category
    if category in ["model", "loss", "transform", "metric"]:
        output_port = PORT_OUT_MODULE
    else:
        output_port = PortDefinition(
            name="out",
            data_type=DataType.MODULE,
            label=cat_info.label,
            description=f"{cat_info.label} instance"
        )

    # Special input ports for certain categories
    input_ports = [PORT_CONFIG]
    if category == "optimizer":
        input_ports.insert(0, PORT_PARAMS)
    elif category == "dataloader":
        input_ports.insert(0, PortDefinition(
            name="dataset",
            data_type=DataType.MODULE,
            label="Dataset",
            required=True,
            description="Dataset to load from"
        ))
    elif category == "scheduler":
        input_ports.insert(0, PortDefinition(
            name="optimizer",
            data_type=DataType.MODULE,
            label="Optimizer",
            required=True,
            description="Optimizer to schedule"
        ))

    return NodeTypeDefinition(
        node_type=NodeType.COMPONENT,
        category=ComponentCategory(category) if category in [c.value for c in ComponentCategory] else ComponentCategory.UNKNOWN,
        code_template=f"self.{{name}} = self.create.{category}{{variant}}()",
        input_ports=input_ports,
        output_ports=[output_port],
        methods=methods,
        icon=cat_info.theme.icon,
        color=cat_info.theme.background.split("-")[0],  # Extract color name
        can_drill=cat_info.drillable,
        description=cat_info.description,
    )


# Pre-built operation node definitions
OPERATION_NODE_DEFS: dict[NodeType, NodeTypeDefinition] = {
    NodeType.FORWARD: NodeTypeDefinition(
        node_type=NodeType.FORWARD,
        code_template="{output} = self.{component}({input})",
        input_ports=[PORT_IN_TENSOR],
        output_ports=[PORT_OUT_TENSOR],
        methods=[METHOD_DETACH, METHOD_CLONE],
        icon="play",
        color="blue",
        description="Model forward pass"
    ),
    NodeType.LOSS: NodeTypeDefinition(
        node_type=NodeType.LOSS,
        code_template="{output} = self.{component}({pred}, {target})",
        input_ports=[
            PortDefinition(name="pred", data_type=DataType.TENSOR, label="Prediction", required=True),
            PortDefinition(name="target", data_type=DataType.TENSOR, label="Target", required=True),
        ],
        output_ports=[
            PortDefinition(name="loss", data_type=DataType.SCALAR, label="Loss", description="Scalar loss value"),
        ],
        methods=[METHOD_BACKWARD, METHOD_ITEM, METHOD_DETACH],
        icon="target",
        color="red",
        description="Loss computation"
    ),
    NodeType.BACKWARD: NodeTypeDefinition(
        node_type=NodeType.BACKWARD,
        code_template="{loss}.backward()",
        input_ports=[
            PortDefinition(name="loss", data_type=DataType.SCALAR, label="Loss", required=True),
        ],
        output_ports=[],
        methods=[],
        icon="arrow-left",
        color="red",
        description="Gradient backpropagation"
    ),
    NodeType.OPTIMIZER_STEP: NodeTypeDefinition(
        node_type=NodeType.OPTIMIZER_STEP,
        code_template="self.{optimizer}.step()",
        input_ports=[],
        output_ports=[],
        methods=[],
        icon="trending-up",
        color="green",
        description="Optimizer parameter update"
    ),
    NodeType.OPTIMIZER_ZERO: NodeTypeDefinition(
        node_type=NodeType.OPTIMIZER_ZERO,
        code_template="self.{optimizer}.zero_grad()",
        input_ports=[],
        output_ports=[],
        methods=[],
        icon="refresh-cw",
        color="green",
        description="Zero gradients"
    ),
    NodeType.METHOD_CALL: NodeTypeDefinition(
        node_type=NodeType.METHOD_CALL,
        code_template="{output} = {input}.{method}({args})",
        input_ports=[
            PortDefinition(name="in", data_type=DataType.ANY, label="Object", required=True),
        ],
        output_ports=[
            PortDefinition(name="out", data_type=DataType.ANY, label="Result"),
        ],
        methods=[],
        icon="terminal",
        color="gray",
        description="Generic method call"
    ),
}

# Control flow node definitions
CONTROL_FLOW_NODE_DEFS: dict[NodeType, NodeTypeDefinition] = {
    NodeType.IF: NodeTypeDefinition(
        node_type=NodeType.IF,
        code_template="if {condition}:",
        input_ports=[
            PortDefinition(name="condition", data_type=DataType.BOOL, label="Condition", required=True),
        ],
        output_ports=[
            PortDefinition(name="true", data_type=DataType.ANY, label="True"),
            PortDefinition(name="false", data_type=DataType.ANY, label="False"),
        ],
        methods=[],
        icon="git-branch",
        color="yellow",
        description="Conditional branch"
    ),
    NodeType.FOR: NodeTypeDefinition(
        node_type=NodeType.FOR,
        code_template="for {var} in {iterable}:",
        input_ports=[
            PortDefinition(name="iterable", data_type=DataType.ANY, label="Iterable", required=True),
        ],
        output_ports=[
            PortDefinition(name="item", data_type=DataType.ANY, label="Item"),
            PortDefinition(name="body", data_type=DataType.ANY, label="Body"),
        ],
        methods=[],
        icon="repeat",
        color="yellow",
        description="For loop iteration"
    ),
    NodeType.WITH: NodeTypeDefinition(
        node_type=NodeType.WITH,
        code_template="with {context}:",
        input_ports=[
            PortDefinition(name="context", data_type=DataType.ANY, label="Context", required=True),
        ],
        output_ports=[
            PortDefinition(name="body", data_type=DataType.ANY, label="Body"),
        ],
        methods=[],
        icon="box",
        color="purple",
        description="Context manager block"
    ),
}

# Data node definitions
DATA_NODE_DEFS: dict[NodeType, NodeTypeDefinition] = {
    NodeType.ASSIGN: NodeTypeDefinition(
        node_type=NodeType.ASSIGN,
        code_template="{target} = {value}",
        input_ports=[
            PortDefinition(name="value", data_type=DataType.ANY, label="Value", required=True),
        ],
        output_ports=[
            PortDefinition(name="out", data_type=DataType.ANY, label="Variable"),
        ],
        methods=[],
        icon="arrow-right",
        color="gray",
        description="Variable assignment"
    ),
    NodeType.RETURN: NodeTypeDefinition(
        node_type=NodeType.RETURN,
        code_template="return {value}",
        input_ports=[
            PortDefinition(name="value", data_type=DataType.ANY, label="Value", required=False),
        ],
        output_ports=[],
        methods=[],
        icon="log-out",
        color="gray",
        description="Function return"
    ),
}


def get_node_type_definition(
    node_type: NodeType,
    category: Optional[str] = None
) -> Optional[NodeTypeDefinition]:
    """Get NodeTypeDefinition for a node type

    Args:
        node_type: The type of node
        category: For COMPONENT nodes, the category name

    Returns:
        NodeTypeDefinition or None
    """
    if node_type == NodeType.COMPONENT:
        if category:
            return _create_component_node_def(category)
        return None

    # Check operation nodes
    if node_type in OPERATION_NODE_DEFS:
        return OPERATION_NODE_DEFS[node_type]

    # Check control flow nodes
    if node_type in CONTROL_FLOW_NODE_DEFS:
        return CONTROL_FLOW_NODE_DEFS[node_type]

    # Check data nodes
    if node_type in DATA_NODE_DEFS:
        return DATA_NODE_DEFS[node_type]

    return None


def generate_node_code(
    node_type: NodeType,
    category: Optional[str] = None,
    **kwargs
) -> str:
    """Generate Python code from a node type definition

    Args:
        node_type: Node type
        category: Component category (for COMPONENT nodes)
        **kwargs: Template variables

    Returns:
        Generated Python code
    """
    definition = get_node_type_definition(node_type, category)
    if not definition or not definition.code_template:
        return ""

    # Process variant for component nodes
    if "variant" in kwargs:
        variant = kwargs.pop("variant")
        kwargs["variant"] = f".{variant}" if variant else ""
    else:
        kwargs["variant"] = ""

    try:
        return definition.code_template.format(**kwargs)
    except KeyError:
        return definition.code_template
