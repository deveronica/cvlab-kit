"""Transform Analyzer for extracting transform pipeline structure.

This module analyzes transform component files to extract:
1. Sub-transform assignments (self.xxx = SomeTransform(...))
2. Pipeline composition (transforms.Compose, list of transforms)
3. Call flow through __call__ method

Patterns recognized:
- self.transform = transforms.Compose([...])     -> Pipeline
- self.transforms = [Transform1(), Transform2()] -> List pipeline
- self.xxx = SomeTransform(cfg)                  -> Sub-transform
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..models.hierarchy import (
    CodeFlowEdge,
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


class TransformSubComponentExtractor(ast.NodeVisitor):
    """Extracts sub-transform components from __init__ method.

    Patterns:
    - self.xxx = transforms.Compose([...])        -> Pipeline
    - self.xxx = SomeTransform(...)               -> Single transform
    - self.transforms = [T1(), T2(), ...]         -> Transform list
    """

    # Known transform classes from torchvision
    TORCHVISION_TRANSFORMS = {
        "Compose",
        "ToTensor",
        "ToPILImage",
        "Normalize",
        "Resize",
        "CenterCrop",
        "RandomCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation",
        "RandomAffine",
        "ColorJitter",
        "Grayscale",
        "RandomGrayscale",
        "Pad",
        "Lambda",
        "RandomResizedCrop",
        "FiveCrop",
        "TenCrop",
        "LinearTransformation",
        "ConvertImageDtype",
        "RandomErasing",
        "GaussianBlur",
        "InterpolationMode",
        "RandomInvert",
        "RandomPosterize",
        "RandomSolarize",
        "RandomAdjustSharpness",
        "RandomAutocontrast",
        "RandomEqualize",
        "AutoAugment",
        "RandAugment",
        "TrivialAugmentWide",
        "AugMix",
    }

    # Transform-like patterns
    TRANSFORM_SUFFIXES = ("Transform", "Augment", "Aug", "Crop", "Flip", "Resize")

    def __init__(self, source_file: str = "") -> None:
        self.sub_transforms: list[dict[str, Any]] = []
        self.pipeline_items: list[dict[str, Any]] = []
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
        """Extract transform assignments."""
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

            # Check for Compose pattern
            if self._is_compose_call(node.value):
                items = self._extract_compose_items(node.value, node.lineno)
                self.pipeline_items.extend(items)

            # Check for list of transforms
            elif isinstance(node.value, ast.List):
                items = self._extract_list_items(node.value, node.lineno)
                self.pipeline_items.extend(items)

            # Check for single transform
            else:
                transform_info = self._analyze_transform_assignment(
                    node.value, attr_name, node.lineno
                )
                if transform_info:
                    self.sub_transforms.append(transform_info)

        self.generic_visit(node)

    def _is_compose_call(self, node: ast.expr) -> bool:
        """Check if this is a transforms.Compose() call."""
        if not isinstance(node, ast.Call):
            return False

        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "Compose":
                return True
        elif isinstance(func, ast.Name):
            if func.id == "Compose":
                return True

        return False

    def _extract_compose_items(
        self, node: ast.Call, base_lineno: int
    ) -> list[dict[str, Any]]:
        """Extract items from Compose([...])."""
        items = []

        # Find the list argument
        for arg in node.args:
            if isinstance(arg, ast.List):
                items = self._extract_list_items(arg, base_lineno)
                break

        # Check keyword arguments
        for kw in node.keywords:
            if kw.arg == "transforms" and isinstance(kw.value, ast.List):
                items = self._extract_list_items(kw.value, base_lineno)
                break

        return items

    def _extract_list_items(
        self, node: ast.List, base_lineno: int
    ) -> list[dict[str, Any]]:
        """Extract transform items from a list."""
        items = []

        for idx, elt in enumerate(node.elts):
            if isinstance(elt, ast.Call):
                class_name = self._get_call_name(elt)
                if class_name:
                    params = self._extract_call_params(elt)
                    items.append(
                        {
                            "id": f"transform_{idx}",
                            "class_name": class_name,
                            "params": params,
                            "lineno": getattr(elt, "lineno", base_lineno),
                            "is_torchvision": class_name in self.TORCHVISION_TRANSFORMS,
                            "order": idx,
                        }
                    )

        return items

    def _analyze_transform_assignment(
        self, value: ast.expr, attr_name: str, lineno: int
    ) -> dict[str, Any] | None:
        """Analyze if assignment is a transform component."""
        if isinstance(value, ast.Call):
            class_name = self._get_call_name(value)
            if not class_name:
                return None

            # Check if it's a known transform or has transform-like name
            is_transform = (
                class_name in self.TORCHVISION_TRANSFORMS
                or any(class_name.endswith(suffix) for suffix in self.TRANSFORM_SUFFIXES)
                or "transform" in attr_name.lower()
            )

            if is_transform:
                params = self._extract_call_params(value)

                return {
                    "id": attr_name,
                    "class_name": class_name,
                    "params": params,
                    "lineno": lineno,
                    "is_torchvision": class_name in self.TORCHVISION_TRANSFORMS,
                }

        return None

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of the called class."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _extract_call_params(self, node: ast.Call) -> dict[str, Any]:
        """Extract parameters from constructor call."""
        params = {}

        # Positional args (for transforms like Resize(224))
        for i, arg in enumerate(node.args):
            value = self._get_const_value(arg)
            if value is not None:
                params[f"arg{i}"] = value

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
        elif isinstance(node, ast.Tuple):
            values = []
            for elt in node.elts:
                v = self._get_const_value(elt)
                if v is not None:
                    values.append(v)
            return tuple(values) if values else "()"
        elif isinstance(node, ast.List):
            return "[...]"
        elif isinstance(node, ast.Dict):
            return "{...}"
        return None


def analyze_transform_file(
    transform_path: Path,
    agent_name: str,
    component_id: str,
    parent_path: list[HierarchyPath],
    project_root: Path,
) -> HierarchicalNodeGraph:
    """Analyze a transform file and build hierarchical node graph.

    Args:
        transform_path: Path to the transform source file
        agent_name: Name of the parent agent
        component_id: ID of the transform component
        parent_path: Breadcrumb path for navigation
        project_root: Project root for relative paths

    Returns:
        HierarchicalNodeGraph with transform pipeline nodes and edges
    """
    source = transform_path.read_text(encoding="utf-8")
    relative_path = str(transform_path.relative_to(project_root))

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _placeholder_graph(
            agent_name, component_id, "Syntax error", parent_path, relative_path
        )

    # Extract transform components
    extractor = TransformSubComponentExtractor(relative_path)
    extractor.visit(tree)

    # Combine pipeline items and sub-transforms
    all_transforms = extractor.pipeline_items + extractor.sub_transforms

    # If no transforms found, return single transform view
    if not all_transforms:
        return _single_transform_graph(
            agent_name, component_id, parent_path, relative_path, source
        )

    # Sort by order if available
    all_transforms.sort(key=lambda x: x.get("order", 999))

    # Build nodes
    nodes: list[HierarchyNode] = []

    # Add input node
    nodes.append(
        HierarchyNode(
            id="input",
            label="Input",
            level=HierarchyLevel.LAYER,
            inputs=[],
            outputs=[Port(name="sample", type="any")],
            metadata={"is_boundary": True, "boundary_type": "input"},
        )
    )

    # Add transform nodes
    for tf in all_transforms:
        node = HierarchyNode(
            id=tf["id"],
            label=tf["class_name"],
            level=HierarchyLevel.LAYER,
            origin=NodeOrigin(
                type=OriginType.ASSIGNMENT,
                code_snippet=f"{tf['class_name']}(...)",
                source=SourceLocation(file=relative_path, line=tf["lineno"]),
            ),
            inputs=[Port(name="in", type="any")],
            outputs=[Port(name="out", type="any")],
            source=SourceLocation(file=relative_path, line=tf["lineno"]),
            metadata={
                "class_name": tf["class_name"],
                "params": tf["params"],
                "is_torchvision": tf["is_torchvision"],
                "order": tf.get("order"),
            },
        )
        nodes.append(node)

    # Add output node
    nodes.append(
        HierarchyNode(
            id="output",
            label="Output",
            level=HierarchyLevel.LAYER,
            inputs=[Port(name="result", type="any")],
            outputs=[],
            metadata={"is_boundary": True, "boundary_type": "output"},
        )
    )

    # Build edges (sequential pipeline)
    edges: list[CodeFlowEdge] = []

    # Input -> first transform
    if all_transforms:
        edges.append(
            CodeFlowEdge(
                id="e0",
                source_node="input",
                source_port="sample",
                target_node=all_transforms[0]["id"],
                target_port="in",
                flow_type=FlowType.TENSOR,
                extracted_from="pipeline",
            )
        )

    # Chain transforms together
    for i in range(len(all_transforms) - 1):
        edges.append(
            CodeFlowEdge(
                id=f"e{i+1}",
                source_node=all_transforms[i]["id"],
                source_port="out",
                target_node=all_transforms[i + 1]["id"],
                target_port="in",
                flow_type=FlowType.TENSOR,
                extracted_from="pipeline",
            )
        )

    # Last transform -> output
    if all_transforms:
        edges.append(
            CodeFlowEdge(
                id=f"e{len(all_transforms)}",
                source_node=all_transforms[-1]["id"],
                source_port="out",
                target_node="output",
                target_port="result",
                flow_type=FlowType.TENSOR,
                extracted_from="pipeline",
            )
        )

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


def _single_transform_graph(
    agent_name: str,
    component_id: str,
    parent_path: list[HierarchyPath],
    source_file: str,
    source: str,
) -> HierarchicalNodeGraph:
    """Create graph for a simple transform with no sub-components."""
    # Extract class name
    try:
        tree = ast.parse(source)
        class_name = "Transform"
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
            outputs=[Port(name="sample", type="any")],
            metadata={"is_boundary": True, "boundary_type": "input"},
        ),
        HierarchyNode(
            id="transform",
            label=class_name,
            level=HierarchyLevel.LAYER,
            inputs=[Port(name="in", type="any")],
            outputs=[Port(name="out", type="any")],
            metadata={"is_single_transform": True},
        ),
        HierarchyNode(
            id="output",
            label="Output",
            level=HierarchyLevel.LAYER,
            inputs=[Port(name="result", type="any")],
            metadata={"is_boundary": True, "boundary_type": "output"},
        ),
    ]

    edges = [
        CodeFlowEdge(
            id="e0",
            source_node="input",
            source_port="sample",
            target_node="transform",
            target_port="in",
            flow_type=FlowType.TENSOR,
        ),
        CodeFlowEdge(
            id="e1",
            source_node="transform",
            source_port="out",
            target_node="output",
            target_port="result",
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
