"""
Node System Type Definitions

This module defines the core types for the visual scripting node system.
Goal: 1:1 correspondence between nodes and code lines.

Key Concepts:
- Node: Represents a code statement (assignment, call, control flow)
- Port: Connection point on a node (input/output)
- Edge: Connection between ports (data/control flow)
- Method: Callable operation on a node's output

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Node ↔ Code Mapping                          │
├─────────────────────────────────────────────────────────────────┤
│  Code Line                    →  Node Type                      │
│  ─────────────────────────────   ─────────────────────────────  │
│  self.x = self.create.model()    ComponentNode(category=model)  │
│  y = self.model(x)               ForwardNode(component=model)   │
│  loss = self.loss_fn(y, t)       LossNode(component=loss_fn)    │
│  loss.backward()                 BackwardNode()                 │
│  self.optimizer.step()           OptimizerStepNode()            │
│  if condition:                   IfNode(condition=...)          │
│  for x in y:                     ForNode(iter=y, var=x)         │
│  with torch.no_grad():           WithNode(context=no_grad)      │
│  return {"loss": loss}           ReturnNode(values=...)         │
└─────────────────────────────────────────────────────────────────┘
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any


# =============================================================================
# Data Types - What flows through edges
# =============================================================================

class DataType(str, Enum):
    """Data types that flow through ports and edges."""
    TENSOR = "tensor"       # torch.Tensor
    MODULE = "module"       # nn.Module or callable
    PARAMS = "params"       # Iterator[Parameter]
    SCALAR = "scalar"       # Python numeric (int, float)
    BOOL = "bool"           # Python bool
    STRING = "string"       # Python str
    LIST = "list"           # Python list
    DICT = "dict"           # Python dict
    CONFIG = "config"       # Configuration dict
    DEVICE = "device"       # torch.device
    ANY = "any"             # Any type

    # Control Flow Types (execution order, not data)
    CONTROL_IN = "control_in"   # Control flow input (top of node)
    CONTROL_OUT = "control_out" # Control flow output (bottom of node)


# =============================================================================
# Port Definitions - Connection points on nodes
# =============================================================================

@dataclass
class PortDefinition:
    """
    Definition of a port on a node type.
    Ports are the connection points for edges.
    """
    name: str                           # Port identifier (e.g., "in", "out", "params")
    data_type: DataType                 # Expected data type
    label: str = ""                     # Display label
    required: bool = False              # Must be connected
    multiple: bool = False              # Allow multiple connections
    default_value: Optional[str] = None # Default value if not connected
    description: str = ""               # Tooltip description

    def __post_init__(self):
        if not self.label:
            self.label = self.name


# =============================================================================
# Method Definitions - Callable operations on node outputs
# =============================================================================

@dataclass
class MethodDefinition:
    """
    Definition of a method that can be called on a node's output.

    Example: model.parameters() → MethodDefinition("parameters", returns=PARAMS)
    """
    name: str                           # Method name (e.g., "parameters", "eval")
    returns: DataType                   # Return type
    args: list[str] = field(default_factory=list)  # Required arguments
    description: str = ""               # Tooltip description

    def code_snippet(self, include_args: bool = True) -> str:
        """Generate code snippet for this method call."""
        if include_args and self.args:
            return f".{self.name}({', '.join(self.args)})"
        return f".{self.name}()"


# =============================================================================
# Node Types - Categories of nodes
# =============================================================================

class NodeType(str, Enum):
    """Node type classification."""

    # === Component Nodes (from setup) ===
    COMPONENT = "component"         # self.x = self.create.category()

    # === Operation Nodes (from train_step) ===
    FORWARD = "forward"             # output = self.component(input)
    LOSS = "loss"                   # loss = self.loss_fn(pred, target)
    BACKWARD = "backward"           # loss.backward()
    OPTIMIZER_STEP = "step"         # self.optimizer.step()
    OPTIMIZER_ZERO = "zero_grad"    # self.optimizer.zero_grad()
    METHOD_CALL = "method"          # x.method()

    # === Control Flow Nodes ===
    IF = "if"                       # if condition:
    ELIF = "elif"                   # elif condition:
    ELSE = "else"                   # else:
    FOR = "for"                     # for x in y:
    WHILE = "while"                 # while condition:
    WITH = "with"                   # with context:

    # === Data Nodes ===
    ASSIGN = "assign"               # x = expression
    UNPACK = "unpack"               # a, b = expression
    RETURN = "return"               # return value

    # === Special Nodes ===
    COMMENT = "comment"             # # comment
    PASS = "pass"                   # pass


class ComponentCategory(str, Enum):
    """Component categories from self.create.*"""
    MODEL = "model"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    LOSS = "loss"
    DATASET = "dataset"
    DATALOADER = "dataloader"
    TRANSFORM = "transform"
    METRIC = "metric"
    SAMPLER = "sampler"
    CHECKPOINT = "checkpoint"
    CALLBACK = "callback"
    LOGGER = "logger"
    AGENT = "agent"
    GLOBAL = "global"
    UNKNOWN = "unknown"


# =============================================================================
# Node Type Definitions - Templates for each node type
# =============================================================================

@dataclass
class NodeTypeDefinition:
    """
    Definition of a node type including code generation template.

    This is the "class" definition. NodeInstance is an "instance".
    """
    node_type: NodeType
    category: Optional[ComponentCategory] = None

    # Code generation
    code_template: str = ""         # Python code template with {placeholders}

    # Ports
    input_ports: list[PortDefinition] = field(default_factory=list)
    output_ports: list[PortDefinition] = field(default_factory=list)

    # Methods callable on output
    methods: list[MethodDefinition] = field(default_factory=list)

    # Visual properties
    icon: str = "box"               # Lucide icon name
    color: str = "gray"             # Theme color
    can_drill: bool = False         # Can drill down into this node

    # Metadata
    description: str = ""

    @property
    def type_key(self) -> str:
        """Unique key for this type definition."""
        if self.category:
            return f"{self.node_type.value}:{self.category.value}"
        return self.node_type.value


# =============================================================================
# Edge Types - How nodes are connected
# =============================================================================

class FlowType(str, Enum):
    """Type of flow through an edge."""
    TENSOR = "tensor"               # Tensor data flow (blue, solid)
    PARAMETERS = "parameters"       # Parameter reference (green, dashed)
    GRADIENT = "gradient"           # Gradient flow (red, reverse)
    CONTROL = "control"             # Control flow / sequence (gray, dashed)
    CONFIG = "config"               # Configuration (purple, dotted)
    REFERENCE = "reference"         # Variable reference (default)


@dataclass
class EdgeStyle:
    """Visual style for edge type."""
    color: str
    stroke_dasharray: Optional[str] = None
    animated: bool = False


EDGE_STYLES: dict[FlowType, EdgeStyle] = {
    FlowType.TENSOR: EdgeStyle("#3b82f6", None, True),           # Blue, animated
    FlowType.PARAMETERS: EdgeStyle("#22c55e", "5,5", False),     # Green, dashed
    FlowType.GRADIENT: EdgeStyle("#ef4444", "3,3", True),        # Red, dashed, animated
    FlowType.CONTROL: EdgeStyle("#6b7280", "10,5", False),       # Gray, dashed
    FlowType.CONFIG: EdgeStyle("#a855f7", "1,3", False),         # Purple, dotted
    FlowType.REFERENCE: EdgeStyle("#6b7280", None, False),       # Gray, solid
}


# =============================================================================
# Node Instance - Actual node in a graph
# =============================================================================

@dataclass
class NodeInstance:
    """
    An instance of a node in the graph.

    This represents an actual node with specific values,
    not a type definition.
    """
    id: str                         # Unique identifier
    node_type: NodeType             # Type of node
    category: Optional[ComponentCategory] = None  # For component nodes
    variant: Optional[str] = None   # e.g., "weak", "strong", "labeled"

    # Display
    label: str = ""                 # Display label

    # Source code mapping
    line_start: int = 0
    line_end: int = 0
    code_snippet: str = ""

    # Configuration (node-specific settings)
    config: dict[str, Any] = field(default_factory=dict)

    # Port instances
    inputs: list["PortInstance"] = field(default_factory=list)
    outputs: list["PortInstance"] = field(default_factory=list)

    # Control flow ports (for train_step operation nodes)
    show_control_ports: bool = False

    # Visual position
    position_x: float = 0
    position_y: float = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortInstance:
    """An instance of a port on a node."""
    name: str
    data_type: DataType
    label: str = ""
    connected: bool = False


@dataclass
class EdgeInstance:
    """
    An edge connecting two ports.

    Represents data or control flow between nodes.
    """
    id: str

    # Source
    source_node: str                # Source node ID
    source_port: str                # Source port name

    # Target
    target_node: str                # Target node ID
    target_port: str                # Target port name

    # Optional fields (with defaults) must come after required fields
    source_method: Optional[str] = None  # Optional method call (e.g., ".parameters()")

    # Flow type
    flow_type: FlowType = FlowType.REFERENCE

    # Display
    label: Optional[str] = None     # Edge label (e.g., ".parameters()")

    # Source mapping
    line: int = 0

    def get_style(self) -> EdgeStyle:
        """Get visual style for this edge."""
        return EDGE_STYLES.get(self.flow_type, EDGE_STYLES[FlowType.REFERENCE])


# =============================================================================
# Graph - Collection of nodes and edges
# =============================================================================

@dataclass
class NodeGraph:
    """
    A complete node graph representing code.

    Can represent setup() or train_step() or any code block.
    """
    id: str
    name: str = ""

    # Content
    nodes: list[NodeInstance] = field(default_factory=list)
    edges: list[EdgeInstance] = field(default_factory=list)

    # Source info
    source_file: Optional[str] = None
    method_name: str = ""           # "setup", "train_step", etc.

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Optional[NodeInstance]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edges_from(self, node_id: str) -> list[EdgeInstance]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source_node == node_id]

    def get_edges_to(self, node_id: str) -> list[EdgeInstance]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target_node == node_id]


# =============================================================================
# Code-Node Mapping - For bidirectional sync
# =============================================================================

@dataclass
class CodeNodeMapping:
    """
    Mapping between a code line and a node.

    Enables:
    - Click on code → highlight node
    - Click on node → highlight code
    - Edit node → update code
    - Edit code → update node
    """
    node_id: str
    line_start: int
    line_end: int
    code_snippet: str
    ast_type: str                   # AST node type (Assign, Call, If, etc.)

    def contains_line(self, line: int) -> bool:
        """Check if line is within this mapping's range."""
        return self.line_start <= line <= self.line_end


# =============================================================================
# Control Flow Helpers
# =============================================================================

# Node types that should show control flow ports (train_step operations)
CONTROL_FLOW_NODE_TYPES: set[NodeType] = {
    NodeType.FORWARD,
    NodeType.LOSS,
    NodeType.BACKWARD,
    NodeType.OPTIMIZER_STEP,
    NodeType.OPTIMIZER_ZERO,
    NodeType.METHOD_CALL,
    NodeType.IF,
    NodeType.FOR,
    NodeType.WHILE,
    NodeType.WITH,
    NodeType.ASSIGN,
    NodeType.RETURN,
}


def should_show_control_ports(node_type: NodeType) -> bool:
    """
    Determine if a node type should display control flow ports.

    Control flow ports are shown for train_step operation nodes
    to visualize execution order.

    Args:
        node_type: The type of node

    Returns:
        True if control ports should be displayed
    """
    return node_type in CONTROL_FLOW_NODE_TYPES
