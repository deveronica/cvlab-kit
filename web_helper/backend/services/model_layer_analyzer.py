"""Model Layer Analyzer - Parse PyTorch model source for layer visualization.

This module extracts:
1. Layer definitions from __init__ (self.xxx = nn.Yyy(...))
2. Data flow from forward() to build edges between layers
3. Input/Output nodes for function boundaries

Patterns supported:
- Direct layer assignment: self.layer = nn.Conv2d(...)
- Sequential: self.block = nn.Sequential(...)
- ModuleList: self.layers = nn.ModuleList([...])
- Nested modules: self.encoder.conv = nn.Conv2d(...)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from web_helper.backend.models.hierarchy import (
    CodeFlowEdge,
    ComponentCategory,
    FlowType,
    HierarchicalNodeGraph,
    Hierarchy,
    HierarchyLevel,
    HierarchyNode,
    HierarchyPath,
    Port,
    SourceLocation,
)


@dataclass
class LayerInfo:
    """Information about a discovered layer."""

    name: str  # Attribute name (e.g., "conv1")
    layer_type: str  # Layer type (e.g., "Conv2d")
    full_type: str  # Full type path (e.g., "nn.Conv2d")
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    line: int = 0
    end_line: int = 0
    is_container: bool = False  # True for Sequential, ModuleList


@dataclass
class ForwardFlowInfo:
    """Information about data flow in forward()."""

    variable: str  # Variable name being assigned
    source_vars: list[str]  # Variables used as input
    layer_name: str | None  # Layer attribute being called (e.g., "self.conv1")
    line: int = 0


class ModelLayerAnalyzer:
    """Analyze PyTorch model source code for layer structure.

    This analyzer extracts:
    1. Layer definitions from __init__
    2. Data flow from forward()
    3. Builds a graph of layers and their connections
    """

    # Common PyTorch layer types
    TORCH_LAYERS = {
        # Convolution
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
        # Pooling
        "MaxPool1d",
        "MaxPool2d",
        "MaxPool3d",
        "AvgPool1d",
        "AvgPool2d",
        "AvgPool3d",
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AdaptiveAvgPool3d",
        "AdaptiveMaxPool1d",
        "AdaptiveMaxPool2d",
        "AdaptiveMaxPool3d",
        # Normalization
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "GroupNorm",
        "LayerNorm",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
        # Activation
        "ReLU",
        "LeakyReLU",
        "PReLU",
        "ELU",
        "SELU",
        "GELU",
        "SiLU",
        "Sigmoid",
        "Tanh",
        "Softmax",
        "LogSoftmax",
        # Linear
        "Linear",
        "Bilinear",
        # Dropout
        "Dropout",
        "Dropout1d",
        "Dropout2d",
        "Dropout3d",
        "AlphaDropout",
        # Embedding
        "Embedding",
        "EmbeddingBag",
        # RNN
        "RNN",
        "LSTM",
        "GRU",
        "RNNCell",
        "LSTMCell",
        "GRUCell",
        # Transformer
        "Transformer",
        "TransformerEncoder",
        "TransformerDecoder",
        "TransformerEncoderLayer",
        "TransformerDecoderLayer",
        "MultiheadAttention",
        # Container (special handling)
        "Sequential",
        "ModuleList",
        "ModuleDict",
        # Utility
        "Flatten",
        "Unflatten",
        "Identity",
        "Upsample",
        "UpsamplingNearest2d",
        "UpsamplingBilinear2d",
    }

    # Container types that hold sub-modules
    CONTAINER_TYPES = {"Sequential", "ModuleList", "ModuleDict"}

    def __init__(self, source_code: str, source_file: str = ""):
        self.source_code = source_code
        self.source_file = source_file
        self.tree: ast.Module | None = None
        self.layers: list[LayerInfo] = []
        self.forward_flows: list[ForwardFlowInfo] = []
        self.class_name: str = ""

    def parse(self) -> bool:
        """Parse the source code into AST."""
        try:
            self.tree = ast.parse(self.source_code)
            return True
        except SyntaxError:
            return False

    def analyze(self) -> tuple[list[LayerInfo], list[ForwardFlowInfo]]:
        """Analyze the model for layers and forward flow."""
        if not self.tree:
            self.parse()

        if not self.tree:
            return [], []

        # Find all model classes
        model_classes: list[ast.ClassDef] = []
        helper_classes: list[ast.ClassDef] = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                base_type = self._get_base_type(node)
                if base_type == "Model":
                    # Primary: inherits from Model (cvlabkit base)
                    model_classes.append(node)
                elif base_type == "Module":
                    # Secondary: inherits from nn.Module (helper classes)
                    helper_classes.append(node)

        # Prefer Model subclass, fallback to last Module subclass
        target_class = None
        if model_classes:
            target_class = model_classes[-1]  # Last Model class in file
        elif helper_classes:
            target_class = helper_classes[-1]  # Last Module class as fallback

        if target_class:
            self.class_name = target_class.name
            self._analyze_init(target_class)
            self._analyze_forward(target_class)

        return self.layers, self.forward_flows

    def _get_base_type(self, node: ast.ClassDef) -> str | None:
        """Get the base type category (Model, Module, or None)."""
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name == "Model":
                return "Model"
            elif base_name in ("Module", "nn.Module"):
                return "Module"
        return None

    def _get_name(self, node: ast.expr) -> str:
        """Get the full name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""

    def _analyze_init(self, class_node: ast.ClassDef) -> None:
        """Extract layer definitions from __init__."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                self._extract_layers_from_init(item)
                break

    def _extract_layers_from_init(self, init_node: ast.FunctionDef) -> None:
        """Walk through __init__ and find layer assignments."""
        for node in ast.walk(init_node):
            if isinstance(node, ast.Assign):
                self._process_assignment(node)

    def _process_assignment(self, node: ast.Assign) -> None:
        """Process an assignment to find layer definitions."""
        # We're looking for: self.xxx = nn.Yyy(...)
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue

            # Check if it's self.xxx
            if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
                continue

            attr_name = target.attr
            layer_info = self._extract_layer_info(node.value, attr_name)
            if layer_info:
                layer_info.line = node.lineno
                layer_info.end_line = node.end_lineno or node.lineno
                self.layers.append(layer_info)

    def _extract_layer_info(
        self, value_node: ast.expr, attr_name: str
    ) -> LayerInfo | None:
        """Extract layer information from the RHS of assignment."""
        if isinstance(value_node, ast.Call):
            func_name = self._get_name(value_node.func)

            # Check if it's a torch layer
            layer_type = func_name.split(".")[-1]
            if layer_type in self.TORCH_LAYERS or func_name.startswith("nn."):
                # Extract arguments
                args = []
                kwargs = {}
                for arg in value_node.args:
                    args.append(self._eval_constant(arg))
                for kw in value_node.keywords:
                    kwargs[kw.arg] = self._eval_constant(kw.value)

                return LayerInfo(
                    name=attr_name,
                    layer_type=layer_type,
                    full_type=func_name,
                    args=args,
                    kwargs=kwargs,
                    is_container=layer_type in self.CONTAINER_TYPES,
                )

            # Also check for custom layer classes (capital first letter, ends with common suffixes)
            if (
                layer_type[0].isupper()
                and len(layer_type) > 1
                and any(
                    layer_type.endswith(suffix)
                    for suffix in [
                        "Block",
                        "Layer",
                        "Module",
                        "Net",
                        "Attention",
                        "Conv",
                        "Norm",
                    ]
                )
            ):
                return LayerInfo(
                    name=attr_name,
                    layer_type=layer_type,
                    full_type=func_name,
                    is_container=False,
                )

        return None

    def _eval_constant(self, node: ast.expr) -> Any:
        """Try to evaluate a constant from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.Name):
            return f"${node.id}"  # Variable reference
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_constant(e) for e in node.elts)
        elif isinstance(node, ast.List):
            return [self._eval_constant(e) for e in node.elts]
        return "..."

    def _analyze_forward(self, class_node: ast.ClassDef) -> None:
        """Analyze forward() method for data flow."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                self._extract_forward_flow(item)
                break

    def _extract_forward_flow(self, forward_node: ast.FunctionDef) -> None:
        """Extract data flow from forward() method."""
        for node in ast.walk(forward_node):
            if isinstance(node, ast.Assign):
                self._process_forward_assignment(node)

    def _process_forward_assignment(self, node: ast.Assign) -> None:
        """Process assignment in forward() to track data flow."""
        # Get target variable name
        target_var = None
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_var = target.id
                break
            elif isinstance(target, ast.Tuple):
                # Handle tuple unpacking (e.g., x, h = self.lstm(x))
                continue

        if not target_var:
            return

        # Analyze the RHS for layer calls and variable dependencies
        layer_name, source_vars = self._extract_call_info(node.value)

        if layer_name or source_vars:
            self.forward_flows.append(
                ForwardFlowInfo(
                    variable=target_var,
                    source_vars=source_vars,
                    layer_name=layer_name,
                    line=node.lineno,
                )
            )

    def _extract_call_info(
        self, node: ast.expr
    ) -> tuple[str | None, list[str]]:
        """Extract layer call and source variables from an expression.

        Handles:
        - Simple calls: self.layer(x)
        - Method chains: self.block(x).relu()
        - Binary operations: self.layer1(x) + self.residual(x)
        - Conditional expressions: x if condition else y
        """
        layer_name = None
        source_vars: list[str] = []

        if isinstance(node, ast.Call):
            # Handle method chains: self.block(x).relu()
            layer_name = self._extract_layer_chain(node)

            # Get all variable references in arguments
            for arg in node.args:
                source_vars.extend(self._find_variables(arg))
            for kw in node.keywords:
                source_vars.extend(self._find_variables(kw.value))

        elif isinstance(node, ast.BinOp):
            # Handle operations like self.layer1(x) + self.residual(x)
            # Extract layers from both sides
            left_layer, left_vars = self._extract_call_info(node.left)
            right_layer, right_vars = self._extract_call_info(node.right)

            # Use first found layer name (or combine if both exist)
            if left_layer and right_layer:
                layer_name = f"{left_layer}+{right_layer}"  # Skip connection
            else:
                layer_name = left_layer or right_layer

            source_vars.extend(left_vars)
            source_vars.extend(right_vars)

        elif isinstance(node, ast.IfExp):
            # Handle conditional: x if cond else y
            _, body_vars = self._extract_call_info(node.body)
            _, else_vars = self._extract_call_info(node.orelse)
            source_vars.extend(body_vars)
            source_vars.extend(else_vars)
            source_vars.extend(self._find_variables(node.test))

        elif isinstance(node, ast.Name):
            source_vars.append(node.id)

        elif isinstance(node, ast.Subscript):
            # Handle indexing: features[0]
            source_vars.extend(self._find_variables(node.value))

        elif isinstance(node, ast.Attribute):
            # Handle attribute access: x.shape
            source_vars.extend(self._find_variables(node.value))

        return layer_name, source_vars

    def _extract_layer_chain(self, node: ast.Call) -> str | None:
        """Extract layer name from potentially chained method calls.

        Handles:
        - self.layer(x) -> "layer"
        - self.block(x).relu() -> "block"
        - F.relu(self.layer(x)) -> "layer"
        """
        func = node.func

        # Direct self.layer(x) call
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            return func.attr

        # Method chain: self.block(x).relu()
        # func is Attribute where func.value is a Call
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
            return self._extract_layer_chain(func.value)

        # Functional call wrapping: F.relu(self.layer(x))
        # Check arguments for nested layer calls
        for arg in node.args:
            if isinstance(arg, ast.Call):
                layer = self._extract_layer_chain(arg)
                if layer:
                    return layer

        return None

    def _find_variables(self, node: ast.expr) -> list[str]:
        """Find all variable references in an expression."""
        variables = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id not in (
                "self",
                "True",
                "False",
                "None",
            ):
                variables.append(child.id)
        return variables

    def build_graph(
        self,
        agent_name: str,
        component_id: str,
        parent_path: list[HierarchyPath],
    ) -> HierarchicalNodeGraph:
        """Build a hierarchical graph from analyzed layers."""
        if not self.layers:
            self.analyze()

        nodes: list[HierarchyNode] = []
        edges: list[CodeFlowEdge] = []

        # Create input node (#4 - Add input/output nodes)
        input_node = HierarchyNode(
            id=f"{component_id}_input",
            label="Input",
            level=HierarchyLevel.LAYER,
            category=ComponentCategory.TRANSFORM,  # Use transform color for I/O
            can_drill=False,
            inputs=[],
            outputs=[Port(name="out", type="tensor")],
            metadata={"is_io": True, "io_type": "input"},
        )
        nodes.append(input_node)

        # Create layer nodes
        for layer in self.layers:
            node = HierarchyNode(
                id=f"{component_id}_{layer.name}",
                label=layer.name,
                level=HierarchyLevel.LAYER,
                category=self._get_layer_category(layer.layer_type),
                can_drill=layer.is_container,  # Can drill into Sequential/ModuleList
                inputs=[Port(name="in", type="tensor")],
                outputs=[Port(name="out", type="tensor")],
                source=SourceLocation(
                    file=self.source_file,
                    line=layer.line,
                    end_line=layer.end_line,
                ),
                metadata={
                    "layer_type": layer.layer_type,
                    "full_type": layer.full_type,
                    "args": layer.args,
                    "kwargs": layer.kwargs,
                },
            )
            nodes.append(node)

        # Create output node (#4 - Add input/output nodes)
        output_node = HierarchyNode(
            id=f"{component_id}_output",
            label="Output",
            level=HierarchyLevel.LAYER,
            category=ComponentCategory.TRANSFORM,
            can_drill=False,
            inputs=[Port(name="in", type="tensor")],
            outputs=[],
            metadata={"is_io": True, "io_type": "output"},
        )
        nodes.append(output_node)

        # Build edges from forward flow analysis
        edges = self._build_edges_from_flow(component_id, nodes)

        return HierarchicalNodeGraph(
            id=f"{agent_name}.{component_id}",
            label=f"{component_id} Layers",
            level=HierarchyLevel.LAYER,
            hierarchy=Hierarchy(
                parent_graph_id=agent_name,
                parent_node_id=component_id,
                depth=1,
                path=parent_path,
            ),
            nodes=nodes,
            edges=edges,
            agent_name=agent_name,
            source_file=self.source_file,
        )

    def _get_layer_category(self, layer_type: str) -> ComponentCategory:
        """Map layer type to component category for coloring."""
        layer_type_lower = layer_type.lower()

        if any(x in layer_type_lower for x in ["conv", "linear", "embed"]):
            return ComponentCategory.MODEL
        elif any(x in layer_type_lower for x in ["norm", "batch", "layer", "group", "instance"]):
            return ComponentCategory.TRANSFORM
        elif any(x in layer_type_lower for x in ["relu", "gelu", "silu", "sigmoid", "tanh", "softmax"]):
            return ComponentCategory.SCHEDULER  # Use scheduler color for activations
        elif any(x in layer_type_lower for x in ["pool", "upsample"]):
            return ComponentCategory.DATALOADER
        elif any(x in layer_type_lower for x in ["dropout"]):
            return ComponentCategory.TRANSFORM
        elif any(x in layer_type_lower for x in ["attention", "transformer"]):
            return ComponentCategory.MODEL
        else:
            return ComponentCategory.UNKNOWN

    def _build_edges_from_flow(
        self, component_id: str, nodes: list[HierarchyNode]
    ) -> list[CodeFlowEdge]:
        """Build edges from forward flow analysis."""
        edges: list[CodeFlowEdge] = []

        # Create a mapping of variable -> node that produces it
        var_to_node: dict[str, str] = {}
        layer_names = {layer.name for layer in self.layers}

        # Track which variables map to which nodes
        # Initial input variables (x, input, etc.) map to input node
        var_to_node["x"] = f"{component_id}_input"
        var_to_node["input"] = f"{component_id}_input"
        var_to_node["inputs"] = f"{component_id}_input"

        edge_id = 0
        last_layer_node = f"{component_id}_input"

        for flow in self.forward_flows:
            if flow.layer_name and flow.layer_name in layer_names:
                # This assignment calls a layer
                target_node = f"{component_id}_{flow.layer_name}"

                # Find source node from source_vars
                source_node = None
                for src_var in flow.source_vars:
                    if src_var in var_to_node:
                        source_node = var_to_node[src_var]
                        break

                # Fallback: use last layer if no explicit source
                if not source_node:
                    source_node = last_layer_node

                # Create edge
                edges.append(
                    CodeFlowEdge(
                        id=f"edge_{edge_id}",
                        source_node=source_node,
                        source_port="out",
                        target_node=target_node,
                        target_port="in",
                        flow_type=FlowType.TENSOR,
                        variable_name=flow.variable,
                        extracted_from="forward",
                        source=SourceLocation(
                            file=self.source_file,
                            line=flow.line,
                        ),
                    )
                )
                edge_id += 1

                # Update variable mapping
                var_to_node[flow.variable] = target_node
                last_layer_node = target_node

        # Connect last layer to output
        if last_layer_node != f"{component_id}_input":
            edges.append(
                CodeFlowEdge(
                    id=f"edge_{edge_id}",
                    source_node=last_layer_node,
                    source_port="out",
                    target_node=f"{component_id}_output",
                    target_port="in",
                    flow_type=FlowType.TENSOR,
                    extracted_from="forward",
                )
            )

        return edges


def analyze_model_file(
    model_path: Path,
    agent_name: str,
    component_id: str,
    parent_path: list[HierarchyPath],
    project_root: Path | None = None,
) -> HierarchicalNodeGraph:
    """Analyze a model file and return its layer graph.

    Args:
        model_path: Path to the model Python file
        agent_name: Name of the agent this belongs to
        component_id: ID of the component being drilled
        parent_path: Breadcrumb path for navigation
        project_root: Root of the project for relative paths

    Returns:
        HierarchicalNodeGraph with layer nodes and edges
    """
    source_code = model_path.read_text(encoding="utf-8")
    source_file = ""
    if project_root:
        try:
            source_file = str(model_path.relative_to(project_root))
        except ValueError:
            source_file = str(model_path)

    analyzer = ModelLayerAnalyzer(source_code, source_file)
    analyzer.analyze()
    return analyzer.build_graph(agent_name, component_id, parent_path)
