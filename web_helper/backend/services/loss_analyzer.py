"""Loss Analyzer for extracting sub-loss structure.

This module analyzes loss component files to extract:
1. Sub-loss assignments (self.xxx = SomeLoss(...))
2. Forward method data flow
3. Loss combination patterns (weighted sum, etc.)

Patterns recognized:
- self.sub_loss = SomeLoss(cfg)           -> Sub-loss node
- self.xxx = nn.CrossEntropyLoss()        -> PyTorch built-in loss node
- loss1 + loss2                           -> Combination edge
- weight * loss                           -> Weighted combination
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..models.hierarchy import (
    CodeFlowEdge,
    DrillTarget,
    FlowType,
    Hierarchy,
    HierarchicalNodeGraph,
    HierarchyLevel,
    HierarchyNode,
    HierarchyPath,
    NodeOrigin,
    OriginType,
    Port,
    SourceLocation,
)


class LossSubComponentExtractor(ast.NodeVisitor):
    """Extracts sub-loss components from __init__ method.

    Patterns:
    - self.xxx = SomeLoss(...)         -> Custom loss
    - self.xxx = nn.CrossEntropy(...)  -> PyTorch loss
    - self.xxx = F.mse_loss            -> Functional loss (reference)
    """

    # Known loss classes (PyTorch and custom)
    TORCH_LOSSES = {
        "CrossEntropyLoss",
        "MSELoss",
        "L1Loss",
        "NLLLoss",
        "BCELoss",
        "BCEWithLogitsLoss",
        "SmoothL1Loss",
        "HuberLoss",
        "KLDivLoss",
        "CTCLoss",
        "TripletMarginLoss",
        "CosineEmbeddingLoss",
        "MarginRankingLoss",
        "MultiLabelMarginLoss",
        "MultiLabelSoftMarginLoss",
        "MultiMarginLoss",
        "PoissonNLLLoss",
        "GaussianNLLLoss",
    }

    # Loss-like patterns (usually custom losses)
    LOSS_SUFFIXES = ("Loss", "loss", "LPIPS", "Perceptual")

    def __init__(self, source_file: str = "") -> None:
        self.sub_losses: list[dict[str, Any]] = []
        self.source_file = source_file
        self._in_init = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track when we're inside __init__."""
        if node.name == "__init__":
            self._in_init = True
            self.generic_visit(node)
            self._in_init = False
        else:
            self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Extract sub-loss assignments."""
        if not self._in_init:
            self.generic_visit(node)
            return

        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            if not (
                isinstance(target.value, ast.Name) and target.value.id == "self"
            ):
                continue

            attr_name = target.attr
            loss_info = self._analyze_loss_assignment(node.value, attr_name, node.lineno)
            if loss_info:
                self.sub_losses.append(loss_info)

        self.generic_visit(node)

    def _analyze_loss_assignment(
        self, value: ast.expr, attr_name: str, lineno: int
    ) -> dict[str, Any] | None:
        """Analyze if assignment is a loss component."""
        if isinstance(value, ast.Call):
            class_name = self._get_call_name(value)
            if not class_name:
                return None

            # Check if it's a known loss or has loss-like name
            is_loss = (
                class_name in self.TORCH_LOSSES
                or any(class_name.endswith(suffix) for suffix in self.LOSS_SUFFIXES)
                or "loss" in attr_name.lower()
            )

            if is_loss:
                # Extract constructor arguments
                params = self._extract_call_params(value)

                return {
                    "id": attr_name,
                    "class_name": class_name,
                    "params": params,
                    "lineno": lineno,
                    "is_torch_builtin": class_name in self.TORCH_LOSSES,
                }

        return None

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of the called class."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle nn.CrossEntropyLoss or module.SomeLoss
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                # Return just the class name, not full path
                return parts[0] if parts else None
        return None

    def _extract_call_params(self, node: ast.Call) -> dict[str, Any]:
        """Extract parameters from constructor call."""
        params = {}

        # Keyword arguments
        for kw in node.keywords:
            if kw.arg:
                params[kw.arg] = self._get_const_value(kw.value)

        return params

    def _get_const_value(self, node: ast.expr) -> Any:
        """Get constant value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return f"<{node.id}>"
        elif isinstance(node, ast.Dict):
            return "{...}"
        elif isinstance(node, ast.List):
            return "[...]"
        return "<expr>"


class LossForwardAnalyzer(ast.NodeVisitor):
    """Analyzes forward() method to extract loss computation flow.

    Patterns:
    - loss1 = self.sub_loss1(...)    -> Call edge
    - loss2 = self.sub_loss2(...)    -> Call edge
    - total = loss1 + loss2          -> Combination edge
    - weighted = weight * loss       -> Weighted edge
    """

    def __init__(self, sub_loss_ids: set[str]) -> None:
        self.sub_loss_ids = sub_loss_ids
        self.edges: list[dict[str, Any]] = []
        self.variable_sources: dict[str, str] = {}  # var -> sub_loss_id
        self._in_forward = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track when we're inside forward."""
        if node.name == "forward":
            self._in_forward = True
            self.generic_visit(node)
            self._in_forward = False
        else:
            self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variable assignments from sub-loss calls."""
        if not self._in_forward:
            self.generic_visit(node)
            return

        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # Check if RHS is a sub-loss call
                loss_id = self._get_sub_loss_call(node.value)
                if loss_id:
                    self.variable_sources[var_name] = loss_id
                    self.edges.append(
                        {
                            "source": loss_id,
                            "target": var_name,
                            "type": "call",
                            "lineno": node.lineno,
                        }
                    )

                # Check for combination patterns
                elif isinstance(node.value, ast.BinOp):
                    self._analyze_combination(var_name, node.value, node.lineno)

        self.generic_visit(node)

    def _get_sub_loss_call(self, node: ast.expr) -> str | None:
        """Check if expression is a sub-loss call."""
        if not isinstance(node, ast.Call):
            return None

        func = node.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                if func.attr in self.sub_loss_ids:
                    return func.attr

        return None

    def _analyze_combination(
        self, target_var: str, node: ast.BinOp, lineno: int
    ) -> None:
        """Analyze loss combination (addition, multiplication)."""
        sources = self._collect_operands(node)

        for source in sources:
            if source in self.variable_sources:
                # This variable came from a sub-loss
                self.edges.append(
                    {
                        "source": self.variable_sources[source],
                        "target": target_var,
                        "type": "combine",
                        "lineno": lineno,
                    }
                )

    def _collect_operands(self, node: ast.expr) -> list[str]:
        """Collect variable names from binary operation."""
        operands = []

        if isinstance(node, ast.BinOp):
            operands.extend(self._collect_operands(node.left))
            operands.extend(self._collect_operands(node.right))
        elif isinstance(node, ast.Name):
            operands.append(node.id)

        return operands


def analyze_loss_file(
    loss_path: Path,
    agent_name: str,
    component_id: str,
    parent_path: list[HierarchyPath],
    project_root: Path,
) -> HierarchicalNodeGraph:
    """Analyze a loss file and build hierarchical node graph.

    Args:
        loss_path: Path to the loss source file
        agent_name: Name of the parent agent
        component_id: ID of the loss component
        parent_path: Breadcrumb path for navigation
        project_root: Project root for relative paths

    Returns:
        HierarchicalNodeGraph with sub-loss nodes and edges
    """
    source = loss_path.read_text(encoding="utf-8")
    relative_path = str(loss_path.relative_to(project_root))

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _placeholder_graph(
            agent_name, component_id, "Syntax error", parent_path, relative_path
        )

    # Extract sub-loss components
    extractor = LossSubComponentExtractor(relative_path)
    extractor.visit(tree)
    sub_losses = extractor.sub_losses

    # If no sub-losses found, return info about the loss itself
    if not sub_losses:
        return _single_loss_graph(
            agent_name, component_id, parent_path, relative_path, source
        )

    # Analyze forward method for edges
    sub_loss_ids = {sl["id"] for sl in sub_losses}
    forward_analyzer = LossForwardAnalyzer(sub_loss_ids)
    forward_analyzer.visit(tree)

    # Build nodes
    nodes: list[HierarchyNode] = []

    # Add input node (data flowing into loss)
    nodes.append(
        HierarchyNode(
            id="input",
            label="Input",
            level=HierarchyLevel.LAYER,
            inputs=[],
            outputs=[
                Port(name="pred", type="tensor"),
                Port(name="target", type="tensor"),
            ],
            metadata={"is_boundary": True, "boundary_type": "input"},
        )
    )

    # Add sub-loss nodes
    for sl in sub_losses:
        node = HierarchyNode(
            id=sl["id"],
            label=sl["class_name"],
            level=HierarchyLevel.LAYER,
            origin=NodeOrigin(
                type=OriginType.ASSIGNMENT,
                code_snippet=f"self.{sl['id']} = {sl['class_name']}(...)",
                source=SourceLocation(file=relative_path, line=sl["lineno"]),
            ),
            inputs=[Port(name="in", type="tensor")],
            outputs=[Port(name="out", type="scalar")],
            source=SourceLocation(file=relative_path, line=sl["lineno"]),
            metadata={
                "class_name": sl["class_name"],
                "params": sl["params"],
                "is_torch_builtin": sl["is_torch_builtin"],
            },
        )
        nodes.append(node)

    # Add output node
    nodes.append(
        HierarchyNode(
            id="output",
            label="Output",
            level=HierarchyLevel.LAYER,
            inputs=[Port(name="loss", type="scalar")],
            outputs=[],
            metadata={"is_boundary": True, "boundary_type": "output"},
        )
    )

    # Build edges
    edges: list[CodeFlowEdge] = []
    edge_idx = 0

    # Input to sub-losses
    for sl in sub_losses:
        edges.append(
            CodeFlowEdge(
                id=f"e{edge_idx}",
                source_node="input",
                source_port="pred",
                target_node=sl["id"],
                target_port="in",
                flow_type=FlowType.TENSOR,
                extracted_from="forward",
            )
        )
        edge_idx += 1

    # Edges from forward analysis
    for edge_info in forward_analyzer.edges:
        if edge_info["type"] == "call":
            # Sub-loss to variable (represented as edge to output for now)
            pass  # Already handled above
        elif edge_info["type"] == "combine":
            # Combination edge
            edges.append(
                CodeFlowEdge(
                    id=f"e{edge_idx}",
                    source_node=edge_info["source"],
                    source_port="out",
                    target_node="output",
                    target_port="loss",
                    flow_type=FlowType.TENSOR,
                    extracted_from="forward",
                    source=SourceLocation(file=relative_path, line=edge_info["lineno"]),
                )
            )
            edge_idx += 1

    # If no combination edges, connect all sub-losses to output
    if not any(e["type"] == "combine" for e in forward_analyzer.edges):
        for sl in sub_losses:
            edges.append(
                CodeFlowEdge(
                    id=f"e{edge_idx}",
                    source_node=sl["id"],
                    source_port="out",
                    target_node="output",
                    target_port="loss",
                    flow_type=FlowType.TENSOR,
                    extracted_from="forward",
                )
            )
            edge_idx += 1

    # Build hierarchy context
    current_path = parent_path + [
        HierarchyPath(
            level=HierarchyLevel.LAYER,
            label=component_id,
            node_id=component_id,
            graph_id=f"{agent_name}:{component_id}",
        )
    ]

    return HierarchicalNodeGraph(
        id=f"{agent_name}:{component_id}",
        label=component_id,
        level=HierarchyLevel.LAYER,
        hierarchy=Hierarchy(
            parent_graph_id=parent_path[-1].graph_id if parent_path else None,
            parent_node_id=component_id,
            depth=len(parent_path) + 1,
            path=current_path,
        ),
        nodes=nodes,
        edges=edges,
        agent_name=agent_name,
        source_file=relative_path,
    )


def _single_loss_graph(
    agent_name: str,
    component_id: str,
    parent_path: list[HierarchyPath],
    source_file: str,
    source: str,
) -> HierarchicalNodeGraph:
    """Create graph for a simple loss with no sub-components."""
    # Extract class name and parameters from source
    try:
        tree = ast.parse(source)
        class_name = "Loss"
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                break
    except:
        class_name = component_id

    nodes = [
        HierarchyNode(
            id="input",
            label="Input",
            level=HierarchyLevel.LAYER,
            outputs=[
                Port(name="pred", type="tensor"),
                Port(name="target", type="tensor"),
            ],
            metadata={"is_boundary": True, "boundary_type": "input"},
        ),
        HierarchyNode(
            id="loss_fn",
            label=class_name,
            level=HierarchyLevel.LAYER,
            inputs=[Port(name="in", type="tensor")],
            outputs=[Port(name="out", type="scalar")],
            metadata={"is_single_loss": True},
        ),
        HierarchyNode(
            id="output",
            label="Output",
            level=HierarchyLevel.LAYER,
            inputs=[Port(name="loss", type="scalar")],
            metadata={"is_boundary": True, "boundary_type": "output"},
        ),
    ]

    edges = [
        CodeFlowEdge(
            id="e0",
            source_node="input",
            source_port="pred",
            target_node="loss_fn",
            target_port="in",
            flow_type=FlowType.TENSOR,
        ),
        CodeFlowEdge(
            id="e1",
            source_node="loss_fn",
            source_port="out",
            target_node="output",
            target_port="loss",
            flow_type=FlowType.TENSOR,
        ),
    ]

    current_path = parent_path + [
        HierarchyPath(
            level=HierarchyLevel.LAYER,
            label=component_id,
            node_id=component_id,
            graph_id=f"{agent_name}:{component_id}",
        )
    ]

    return HierarchicalNodeGraph(
        id=f"{agent_name}:{component_id}",
        label=component_id,
        level=HierarchyLevel.LAYER,
        hierarchy=Hierarchy(
            parent_graph_id=parent_path[-1].graph_id if parent_path else None,
            parent_node_id=component_id,
            depth=len(parent_path) + 1,
            path=current_path,
        ),
        nodes=nodes,
        edges=edges,
        agent_name=agent_name,
        source_file=source_file,
    )


def _placeholder_graph(
    agent_name: str,
    component_id: str,
    message: str,
    parent_path: list[HierarchyPath],
    source_file: str = "",
) -> HierarchicalNodeGraph:
    """Create a placeholder graph when analysis fails."""
    current_path = parent_path + [
        HierarchyPath(
            level=HierarchyLevel.LAYER,
            label=component_id,
            node_id=component_id,
            graph_id=f"{agent_name}:{component_id}",
        )
    ]

    return HierarchicalNodeGraph(
        id=f"{agent_name}:{component_id}",
        label=component_id,
        level=HierarchyLevel.LAYER,
        hierarchy=Hierarchy(
            parent_graph_id=parent_path[-1].graph_id if parent_path else None,
            parent_node_id=component_id,
            depth=len(parent_path) + 1,
            path=current_path,
        ),
        nodes=[
            HierarchyNode(
                id="placeholder",
                label=message,
                level=HierarchyLevel.LAYER,
                metadata={"is_placeholder": True},
            )
        ],
        edges=[],
        agent_name=agent_name,
        source_file=source_file,
    )
