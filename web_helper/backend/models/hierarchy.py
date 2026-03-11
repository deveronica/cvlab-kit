"""Hierarchical Node Graph Schema for Simulink-style visualization.

This module defines the Pydantic models for the hierarchical node graph system.
Key design principles:
1. Level 0 shows components directly (no method level)
2. Edges are extracted from both setup() and train_step()
3. No Config-based edge inference - only code-driven extraction
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class HierarchyLevel(str, Enum):
    """Hierarchy levels in the graph."""

    METHOD = "method"  # Level -1: Agent methods (setup, train_step, etc.)
    COMPONENT = "component"  # Level 0: setup() components
    LAYER = "layer"  # Level 1: model layers, loss sub-components
    OPERATION = "operation"  # Level 2+: internal operations


class ComponentCategory(str, Enum):
    """Component categories matching cvlabkit component types."""

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


class MethodCategory(str, Enum):
    """Method categories for Agent methods."""

    SETUP = "setup"  # setup() - initialization
    TRAIN = "train"  # train_step() - training logic
    VALIDATION = "validation"  # val_step() - validation logic
    LIFECYCLE = "lifecycle"  # on_train_start, on_train_end, etc.
    UTILITY = "utility"  # Other helper methods


class FlowType(str, Enum):
    """Edge flow types based on actual code patterns."""

    TENSOR = "tensor"  # Tensor data flow (outputs, inputs)
    PARAMETERS = "parameters"  # model.parameters() -> optimizer
    GRADIENT = "gradient"  # loss.backward() gradient flow
    CONTROL = "control"  # Execution order (no data)
    CONFIG = "config"  # Config parameter injection
    REFERENCE = "reference"  # Component instance reference (self.xxx -> yyy)


class ImplSource(str, Enum):
    """Source of component implementation name.

    Used to distinguish where the impl comes from for Builder vs Execute contexts:
    - HARDCODE: impl is fixed in code, YAML is ignored
    - DEFAULT: impl is fallback when YAML doesn't specify
    - YAML: impl comes from YAML config (Builder can't determine)
    """

    HARDCODE = "hardcode"  # create.model("resnet18") - positional arg
    DEFAULT = "default"  # create.model(default="resnet18") - keyword arg
    YAML = "yaml"  # create.model() - no args, impl from YAML


class ValueSource(str, Enum):
    """Source of a property/parameter value for UI display.

    5-state system for clear visual distinction:
    - REQUIRED: No default value, must be provided (red warning)
    - CONFIG: Value loaded from YAML config (blue, editable)
    - DEFAULT: Has default value in code (gray, optional)
    - HARDCODE: Fixed in code, not configurable (gray disabled)
    - CONNECTED: Value comes from connected node port (green)
    """

    REQUIRED = "required"  # cfg.get("key") - no default, must provide
    CONFIG = "config"  # Value from YAML config file
    DEFAULT = "default"  # cfg.get("key", 0.01) - has default
    HARDCODE = "hardcode"  # Literal value in code, immutable
    CONNECTED = "connected"  # Value from connected node edge


class OriginType(str, Enum):
    LOCAL_CALL = 'local_call'
    
    """What code pattern created this node."""

    CREATE_CALL = "create_call"  # self.xxx = self.create.yyy()
    ASSIGNMENT = "assignment"  # variable assignment
    METHOD_CALL = "method_call"  # self.component(...)
    UNPACK = "unpack"  # a, b = tuple
    INPUT = "input"  # method parameter
    OUTPUT = "output"  # return value


class SourceLocation(BaseModel):
    """Source code location for traceability."""

    file: str
    line: int | None = None
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None


class NodeOrigin(BaseModel):
    """Code origin information for a node."""

    type: OriginType
    create_path: list[str] | None = None  # ['model'] or ['loss', 'supervised']
    code_snippet: str
    source: SourceLocation | None = None
    impl_source: ImplSource | None = None  # YAML, DEFAULT, or HARDCODE


class DrillTarget(BaseModel):
    """Drill-down target specification."""

    type: Literal["component", "layer"]
    path: str  # Component file path or layer path


class Port(BaseModel):
    """Node port for connections."""

    name: str
    type: str = "any"  # tensor, parameters, etc.
    kind: Literal["exec", "data"] = "data"


class PropertyInfo(BaseModel):
    """Property information with value source for UI display.

    Used in SmartNodeView to show properties with clear source indicators.
    """

    name: str
    value: Any = None
    source: ValueSource = ValueSource.DEFAULT
    is_grid_search: bool = False  # True if value is array for grid search
    grid_count: int | None = None  # Number of variants if grid search
    default_value: Any = None  # Default value from code (if available)
    description: str | None = None  # Optional description/tooltip


class PropertySummary(BaseModel):
    """Summary counts of properties by source for badge display.

    Shows "⚠️ 2 required, 3 config" on node surface.
    """

    required_count: int = 0
    config_count: int = 0
    default_count: int = 0
    hardcode_count: int = 0
    connected_count: int = 0


class HierarchyPath(BaseModel):
    """Path item for breadcrumb navigation."""

    level: HierarchyLevel
    label: str
    node_id: str
    graph_id: str
    # Optional: Category for visual styling (e.g., "model", "optimizer")
    category: ComponentCategory | None = None
    # Optional: Implementation name (e.g., "resnet18")
    implementation: str | None = None


class HierarchyNode(BaseModel):
    """Node in the hierarchy graph."""

    id: str
    label: str
    level: HierarchyLevel

    # 항목 1: 객체명과 메서드명 분리
    object_name: str | None = None  # e.g., "model"
    method_name: str | None = None  # e.g., "forward"

    # Code origin (what pattern created this node?)
    origin: NodeOrigin | None = None

    # Drill-down support
    can_drill: bool = False
    drill_target: DrillTarget | None = None

    # Visual category (for coloring)
    category: ComponentCategory | None = None

    # Ports for connections
    inputs: list[Port] = Field(default_factory=list)
    outputs: list[Port] = Field(default_factory=list)

    # Source location
    source: SourceLocation | None = None

    # Properties grouped by source (for SmartNodeView)
    properties: list[PropertyInfo] = Field(default_factory=list)
    property_summary: PropertySummary | None = None

    # Used config keys (for setup phase visualization)
    used_config_keys: list[str] = Field(default_factory=list)

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeFlowEdge(BaseModel):
    """Edge representing actual code flow."""

    id: str
    source_node: str
    source_port: str = "out"
    target_node: str
    target_port: str = "in"

    # Flow type based on code analysis
    flow_type: FlowType = FlowType.TENSOR

    edge_type: Literal["execution", "data"] | None = None

    # Variable carrying this data (from code analysis)
    variable_name: str | None = None

    # Execution order (1-based sequence index)
    sequence_index: int | None = None

    # Which method this edge was extracted from
    extracted_from: str | None = None  # "setup" or "train_step"

    # Source location
    source: SourceLocation | None = None

    # Additional metadata (is_gapped, comment, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Hierarchy(BaseModel):
    """Hierarchy context for navigation."""

    parent_graph_id: str | None = None
    parent_node_id: str | None = None
    depth: int = 0
    path: list[HierarchyPath] = Field(default_factory=list)


class HierarchicalNodeGraph(BaseModel):
    """Root schema for hierarchical node graph.

    This is the main response model for the /hierarchy/{agent} endpoint.
    """

    id: str
    label: str
    level: HierarchyLevel

    # Hierarchy context (Simulink-style navigation)
    hierarchy: Hierarchy = Field(default_factory=Hierarchy)

    # Graph content
    nodes: list[HierarchyNode] = Field(default_factory=list)
    edges: list[CodeFlowEdge] = Field(default_factory=list)

    # Agent name (for context)
    agent_name: str | None = None

    # Source file
    source_file: str | None = None

    # Code coverage info (for 1:1 code-node mapping)
    method_range: SourceLocation | None = None  # Start/end of analyzed method
    covered_lines: list[int] = Field(default_factory=list)  # Lines covered by nodes
    uncovered_lines: list[int] = Field(
        default_factory=list
    )  # Statement lines NOT covered

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


# Response wrapper
class HierarchyResponse(BaseModel):
    """API response wrapper."""

    success: bool = True
    data: HierarchicalNodeGraph | None = None
    error: str | None = None


# =============================================================================
# Node/Edge Mutation Request/Response Models (Phase C)
# =============================================================================


class AddNodeRequest(BaseModel):
    """Request to add a new component node.

    Role (변수명) handling:
    - If role is provided, use it directly
    - If role is None, auto-generate from category (e.g., "model")
    - If duplicate, append suffix (_2, _3, etc.)
    """

    category: str  # "model", "optimizer", "loss", "dataset", "transform"
    implementation: str  # "resnet18", "adam", "cross_entropy"
    role: str | None = None  # 변수명 (없으면 category 기반 자동 생성)
    config: dict[str, Any] = Field(default_factory=dict)  # 컴포넌트 config params
    position: dict[str, float] = Field(
        default_factory=lambda: {"x": 0, "y": 0}
    )  # UI 위치


class AddNodeResponse(BaseModel):
    """Response from adding a node."""

    success: bool
    node_id: str | None = None  # 생성된 노드 ID (= role)
    role: str | None = None  # 실제 사용된 role (자동 생성된 경우 포함)
    mode: Literal["draft", "direct"] | None = None
    edit: dict[str, Any] | None = None  # GraphEdit as dict
    error: str | None = None


class UpdateNodeRequest(BaseModel):
    """Request to update an existing node."""

    implementation: str | None = None  # 새 implementation
    config: dict[str, Any] | None = None  # 변경할 config (key: value pairs)
    position: dict[str, float] | None = None  # UI 위치
    yaml_path: str | None = None  # YAML 파일 경로 (config 변경 시)
    category: str | None = None  # 컴포넌트 카테고리 (nested key 생성용)


class UpdateNodeResponse(BaseModel):
    """Response from updating a node."""

    success: bool
    node_id: str | None = None
    mode: Literal["draft", "direct"] | None = None
    edit: dict[str, Any] | None = None  # GraphEdit as dict
    error: str | None = None


class AddEdgeRequest(BaseModel):
    """Request to add a new edge (data dependency).

    Edge addition with code modification:
    - model → optimizer edge: adds self.model.parameters() to optimizer creation
    - In draft mode, only stores in memory
    - In direct mode or on commit, modifies actual code
    """

    source: str  # source node id (= role)
    target: str  # target node id (= role)
    source_port: str = "out"  # "out", "parameters", "output" 등
    target_port: str = "in"  # "in", "model", "data" 등
    flow_type: str = "reference"  # "data", "parameters", "config", "reference"


class AddEdgeResponse(BaseModel):
    """Response from adding an edge."""

    success: bool
    edge_id: str | None = None  # 생성된 엣지 ID
    mode: Literal["draft", "direct"] | None = None
    code_change: str | None = None  # 코드 수정 내용 (있는 경우)
    edit: dict[str, Any] | None = None  # GraphEdit as dict
    error: str | None = None
