"""
Model Layer/Module Extractor for Drill-down Visualization

Extracts `nn.Module` instances from model source code for hierarchical visualization:
- Level 1: Layer - Direct `nn.Module` assignments in `__init__` (e.g., self.encoder = nn.Linear(...))
- Level 2: Module - Nested `nn.*` calls within a layer (e.g., nn.Sequential(nn.Linear(), nn.ReLU()))

Per spec: Model Drill-down Levels:
- Level 0: Agent (Components)
- Level 1: Layer (`nn.Module` instances in `__init__`)
- Level 2: Module (nested `nn.*` modules)
"""

import ast
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

from ..models.hierarchy import (
    HierarchicalNodeGraph,
    HierarchyNode,
    CodeFlowEdge,
    SourceLocation,
    NodeOrigin,
    Hierarchy,
    HierarchyPath,
    HierarchyLevel,
    ComponentCategory,
    FlowType,
    OriginType,
)


@dataclass
class LayerInfo:
    """Information about a layer (Level 1 node)."""

    name: str  # e.g., "encoder", "classifier"
    type_name: str  # e.g., "nn.Linear", "nn.Sequential", "ResNetBlock"
    source_line: int
    end_line: int
    is_nested_module: bool = (
        False  # True if it's a direct nn.* call like nn.Linear(...)
    )


@dataclass
class ModuleInfo:
    """Information about a nested module (Level 2 node)."""

    name: str  # e.g., "0" (index in Sequential), "fc1" (attribute name)
    type_name: str  # e.g., "nn.Linear", "nn.ReLU"
    parent_layer: str  # e.g., "encoder"
    source_line: int
    is_inline: bool = (
        False  # True if inline in Sequential like nn.Sequential(nn.Linear(), nn.ReLU())
    )


class ModelLayerExtractor:
    """Extract layers and modules from model source code for drill-down visualization."""

    # Well-known PyTorch nn modules for type classification
    NN_MODULE_TYPES = {
        # Linear layers
        "nn.Linear": ComponentCategory.MODEL,
        "nn.Conv1d": ComponentCategory.MODEL,
        "nn.Conv2d": ComponentCategory.MODEL,
        "nn.Conv3d": ComponentCategory.MODEL,
        "nn.ConvTranspose1d": ComponentCategory.MODEL,
        "nn.ConvTranspose2d": ComponentCategory.MODEL,
        "nn.ConvTranspose3d": ComponentCategory.MODEL,
        "nn.Bilinear": ComponentCategory.MODEL,
        # Pooling
        "nn.MaxPool1d": ComponentCategory.MODEL,
        "nn.MaxPool2d": ComponentCategory.MODEL,
        "nn.MaxPool3d": ComponentCategory.MODEL,
        "nn.AvgPool1d": ComponentCategory.MODEL,
        "nn.AvgPool2d": ComponentCategory.MODEL,
        "nn.AvgPool3d": ComponentCategory.MODEL,
        "nn.AdaptiveMaxPool1d": ComponentCategory.MODEL,
        "nn.AdaptiveMaxPool2d": ComponentCategory.MODEL,
        "nn.AdaptiveAvgPool1d": ComponentCategory.MODEL,
        "nn.AdaptiveAvgPool2d": ComponentCategory.MODEL,
        # Padding
        "nn.ReflectionPad1d": ComponentCategory.MODEL,
        "nn.ReflectionPad2d": ComponentCategory.MODEL,
        "nn.ReplicationPad1d": ComponentCategory.MODEL,
        "nn.ReplicationPad2d": ComponentCategory.MODEL,
        "nn.ReplicationPad3d": ComponentCategory.MODEL,
        "nn.ZeroPad2d": ComponentCategory.MODEL,
        "nn.ConstantPad1d": ComponentCategory.MODEL,
        "nn.ConstantPad2d": ComponentCategory.MODEL,
        "nn.ConstantPad3d": ComponentCategory.MODEL,
        # Non-linear activations
        "nn.ReLU": ComponentCategory.MODEL,
        "nn.ReLU6": ComponentCategory.MODEL,
        "nn.LeakyReLU": ComponentCategory.MODEL,
        "nn.PReLU": ComponentCategory.MODEL,
        "nn.RReLU": ComponentCategory.MODEL,
        "nn.ELU": ComponentCategory.MODEL,
        "nn.CELU": ComponentCategory.MODEL,
        "nn.SELU": ComponentCategory.MODEL,
        "nn.GELU": ComponentCategory.MODEL,
        "nn.Sigmoid": ComponentCategory.MODEL,
        "nn.Tanh": ComponentCategory.MODEL,
        "nn.Hardshrink": ComponentCategory.MODEL,
        "nn.Hardtanh": ComponentCategory.MODEL,
        "nn.Hardswish": ComponentCategory.MODEL,
        "nn.Hardsigmoid": ComponentCategory.MODEL,
        "nn.Tanhshrink": ComponentCategory.MODEL,
        "nn.Threshold": ComponentCategory.MODEL,
        "nn.Softmin": ComponentCategory.MODEL,
        "nn.Softtanh": ComponentCategory.MODEL,
        "nn.Softsign": ComponentCategory.MODEL,
        "nn.Softplus": ComponentCategory.MODEL,
        "nn.Softshrink": ComponentCategory.MODEL,
        "nn.Softsign": ComponentCategory.MODEL,
        "nn.LogSoftmax": ComponentCategory.MODEL,
        "nn.PReLU": ComponentCategory.MODEL,
        # Normalization
        "nn.BatchNorm1d": ComponentCategory.MODEL,
        "nn.BatchNorm2d": ComponentCategory.MODEL,
        "nn.BatchNorm3d": ComponentCategory.MODEL,
        "nn.LazyBatchNorm1d": ComponentCategory.MODEL,
        "nn.LazyBatchNorm2d": ComponentCategory.MODEL,
        "nn.LazyBatchNorm3d": ComponentCategory.MODEL,
        "nn.GroupNorm": ComponentCategory.MODEL,
        "nn.InstanceNorm1d": ComponentCategory.MODEL,
        "nn.InstanceNorm2d": ComponentCategory.MODEL,
        "nn.InstanceNorm3d": ComponentCategory.MODEL,
        "nn.LayerNorm": ComponentCategory.MODEL,
        "nn.LocalResponseNorm": ComponentCategory.MODEL,
        # Recurrent
        "nn.RNN": ComponentCategory.MODEL,
        "nn.LSTM": ComponentCategory.MODEL,
        "nn.GRU": ComponentCategory.MODEL,
        "nn.RNNCell": ComponentCategory.MODEL,
        "nn.LSTMCell": ComponentCategory.MODEL,
        "nn.GRUCell": ComponentCategory.MODEL,
        # Transformer
        "nn.Transformer": ComponentCategory.MODEL,
        "nn.TransformerEncoder": ComponentCategory.MODEL,
        "nn.TransformerDecoder": ComponentCategory.MODEL,
        "nn.MultiheadAttention": ComponentCategory.MODEL,
        "nn.Embedding": ComponentCategory.MODEL,
        # Dropout
        "nn.Dropout": ComponentCategory.MODEL,
        "nn.Dropout2d": ComponentCategory.MODEL,
        "nn.Dropout3d": ComponentCategory.MODEL,
        "nn.AlphaDropout": ComponentCategory.MODEL,
        "nn.FeatureAlphaDropout": ComponentCategory.MODEL,
        # Sparse
        "nn.Embedding": ComponentCategory.MODEL,
        "nn.EmbeddingBag": ComponentCategory.MODEL,
        # Distance
        "nn.PairwiseDistance": ComponentCategory.MODEL,
        "nn.CosineSimilarity": ComponentCategory.MODEL,
        # Loss
        "nn.L1Loss": ComponentCategory.LOSS,
        "nn.MSELoss": ComponentCategory.LOSS,
        "nn.CrossEntropyLoss": ComponentCategory.LOSS,
        "nn.NLLLoss": ComponentCategory.LOSS,
        "nn.PoissonNLLLoss": ComponentCategory.LOSS,
        "nn.KLDivLoss": ComponentCategory.LOSS,
        "nn.BCELoss": ComponentCategory.LOSS,
        "nn.BCEWithLogitsLoss": ComponentCategory.LOSS,
        "nn.MarginRankingLoss": ComponentCategory.LOSS,
        "nn.HingeEmbeddingLoss": ComponentCategory.LOSS,
        "nn.MultiLabelMarginLoss": ComponentCategory.LOSS,
        "nn.HuberLoss": ComponentCategory.LOSS,
        "nn.SmoothL1Loss": ComponentCategory.LOSS,
        "nn.MultiMarginLoss": ComponentCategory.LOSS,
        "nn.TripletMarginLoss": ComponentCategory.LOSS,
        "nn.TripletMarginWithDistanceLoss": ComponentCategory.LOSS,
        # Container modules
        "nn.Sequential": ComponentCategory.MODEL,
        "nn.ModuleList": ComponentCategory.MODEL,
        "nn.ModuleDict": ComponentCategory.MODEL,
    }

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()

    def extract_layers(
        self, model_source: str, model_name: str = "model"
    ) -> list[LayerInfo]:
        """Extract Layer (Level 1) nodes from model source.

        Finds all `nn.Module` assignments in `__init__`:
        - self.encoder = nn.Linear(...)
        - self.decoder = nn.Sequential(...)
        - self.block = ResNetBlock(...)  # Custom module
        """
        layers = []
        try:
            tree = ast.parse(model_source)
        except SyntaxError:
            return layers

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Find __init__ method
                init_method = None
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_method = item
                        break

                if init_method:
                    # Use extend to accumulate layers from all classes
                    class_layers = self._extract_layers_from_init(
                        init_method, node.name
                    )
                    layers.extend(class_layers)

        return layers

    def _extract_layers_from_init(
        self, init_node: ast.FunctionDef, class_name: str
    ) -> list[LayerInfo]:
        """Extract layer assignments from __init__ method body."""
        layers = []

        for node in ast.walk(init_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        # Check if it's a self.xxx assignment
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            layer_name = target.attr
                            type_name, source_line, end_line = self._get_type_info(
                                node.value
                            )

                            if type_name:
                                is_nn_module = self._is_nn_module(type_name)
                                is_custom_module = self._is_custom_module(
                                    type_name, class_name
                                )

                                # Include if it's an nn module or custom module defined in this file
                                if is_nn_module or is_custom_module:
                                    layers.append(
                                        LayerInfo(
                                            name=layer_name,
                                            type_name=type_name,
                                            source_line=node.lineno,
                                            end_line=self._get_end_line(node),
                                            is_nested_module=is_nn_module,
                                        )
                                    )

        return layers

    def _get_type_info(self, node: ast.expr) -> tuple[Optional[str], int, int]:
        """Get the type name and line info from an expression."""
        if isinstance(node, ast.Call):
            # nn.Linear(...), nn.Sequential(...), CustomClass(...)
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    # nn.Linear() -> type = "nn.Linear"
                    return (
                        f"{node.func.value.id}.{node.func.attr}",
                        node.lineno,
                        node.end_lineno or node.lineno,
                    )
                elif isinstance(node.func.value, ast.Attribute):
                    # Could be nested like some_module.nn.Linear()
                    return node.func.attr, node.lineno, node.end_lineno or node.lineno
            elif isinstance(node.func, ast.Name):
                # CustomClass(...) -> type = "CustomClass"
                return node.func.id, node.lineno, node.end_lineno or node.lineno

        elif isinstance(node, ast.Attribute):
            # Already assigned like self.encoder = some_var
            if isinstance(node.value, ast.Name):
                return node.value.id, node.lineno, node.end_lineno or node.lineno

        return None, 0, 0

    def _is_nn_module(self, type_name: str) -> bool:
        """Check if type name is a known nn module."""
        return type_name in self.NN_MODULE_TYPES

    def _is_custom_module(self, type_name: str, class_name: str) -> bool:
        """Check if type name is a custom module (not nn.*)."""
        if type_name.startswith("nn."):
            return False
        # It's a custom module if it doesn't start with nn. and isn't a primitive
        primitives = {"int", "str", "float", "bool", "list", "dict", "tuple", "None"}
        if type_name in primitives:
            return False
        return True

    def _get_end_line(self, node: ast.AST) -> int:
        """Get the end line of an AST node."""
        # AST nodes have lineno and end_lineno in Python 3.8+
        # These are dynamic attributes not in the type stubs
        end_lineno = getattr(node, "end_lineno", None)  # type: ignore[attr-defined]
        if end_lineno is not None:
            return end_lineno
        lineno = getattr(node, "lineno", 1)  # type: ignore[attr-defined]
        return lineno

    def extract_nested_modules(
        self,
        layer_source: str,
        layer_name: str,
        parent_model_class: str = "",
    ) -> list[ModuleInfo]:
        """Extract nested Module (Level 2) nodes from a layer's definition.

        For layers like:
        - self.encoder = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        - self.block = ResNetBlock(in_channels=64, out_channels=64)

        Extracts the nested nn.* modules.
        """
        modules = []

        try:
            tree = ast.parse(layer_source)
        except SyntaxError:
            return modules

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == layer_name:
                # Found the layer class definition
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        modules = self._extract_modules_from_init(item, layer_name)
                        break

        return modules

    def _extract_modules_from_init(
        self, init_node: ast.FunctionDef, layer_name: str
    ) -> list[ModuleInfo]:
        """Extract nested modules from layer's __init__."""
        modules = []

        # Track Sequential children for index-based naming
        sequential_children: list[tuple[int, str, int]] = []

        for node in ast.walk(init_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            attr_name = target.attr
                            type_name, line_no, _ = self._get_type_info(node.value)

                            if type_name and self._is_nn_module(type_name):
                                modules.append(
                                    ModuleInfo(
                                        name=attr_name,
                                        type_name=type_name,
                                        parent_layer=layer_name,
                                        source_line=line_no,
                                        is_inline=False,
                                    )
                                )

            # Handle nn.Sequential(nn.Linear(...), nn.ReLU(), ...)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "Sequential":
                        # This is a Sequential call
                        if isinstance(node.func.value, ast.Name):
                            parent_name = node.func.value.id  # e.g., "self"
                        else:
                            parent_name = "sequential"

                        # Extract children
                        if isinstance(node.args[0], ast.Starred):
                            # *modules_list syntax
                            if isinstance(node.args[0].value, ast.List):
                                for idx, elem in enumerate(node.args[0].value.elts):
                                    type_name, line_no, _ = self._get_type_info(elem)
                                    if type_name and self._is_nn_module(type_name):
                                        modules.append(
                                            ModuleInfo(
                                                name=str(
                                                    idx
                                                ),  # Use index as name for Sequential
                                                type_name=type_name,
                                                parent_layer=layer_name,
                                                source_line=line_no,
                                                is_inline=True,
                                            )
                                        )
                        else:
                            # Positional arguments
                            for idx, arg in enumerate(node.args):
                                type_name, line_no, _ = self._get_type_info(arg)
                                if type_name and self._is_nn_module(type_name):
                                    modules.append(
                                        ModuleInfo(
                                            name=str(idx),
                                            type_name=type_name,
                                            parent_layer=layer_name,
                                            source_line=line_no,
                                            is_inline=True,
                                        )
                                    )

        return modules

    def build_layer_graph(
        self,
        model_source: str,
        model_name: str = "model",
        agent_name: str = "",
        source_file: Optional[str] = None,
    ) -> HierarchicalNodeGraph:
        """Build a Level 1 graph showing layers in a model."""
        layers = self.extract_layers(model_source, model_name)

        nodes = []
        edges = []

        for layer in layers:
            category = self._get_layer_category(layer.type_name)

            # Determine if this layer can be drilled down further
            can_drill = layer.is_nested_module and "Sequential" not in layer.type_name

            node = HierarchyNode(
                id=layer.name,
                label=layer.name,
                level=HierarchyLevel.LAYER,
                category=category,
                can_drill=can_drill,
                inputs=[],
                outputs=[],
                source=SourceLocation(
                    file=source_file or "",
                    line=layer.source_line,
                    end_line=layer.end_line,
                ),
                origin=NodeOrigin(
                    type=OriginType.ASSIGNMENT,
                    code_snippet=f"self.{layer.name} = {layer.type_name}(...)",
                ),
                metadata={
                    "layer_type": layer.type_name,
                    "is_nn_module": layer.is_nested_module,
                },
            )
            nodes.append(node)

        return HierarchicalNodeGraph(
            id=f"{model_name}_layers",
            label=f"{model_name} - Layers",
            level=HierarchyLevel.LAYER,
            hierarchy=Hierarchy(
                depth=1,
                path=[
                    HierarchyPath(
                        level=HierarchyLevel.LAYER,
                        label=model_name,
                        node_id=model_name,
                        graph_id=f"{model_name}_layers",
                        category=ComponentCategory.MODEL,
                        implementation=model_name,
                    )
                ],
            ),
            nodes=nodes,
            edges=edges,
            agent_name=agent_name,
            source_file=source_file,
        )

    def build_module_graph(
        self,
        layer_source: str,
        layer_name: str,
        model_name: str = "model",
        agent_name: str = "",
        source_file: Optional[str] = None,
    ) -> HierarchicalNodeGraph:
        """Build a Level 2 graph showing nested modules in a layer."""
        modules = self.extract_nested_modules(layer_source, layer_name)

        nodes = []
        edges = []

        for module in modules:
            category = self._get_layer_category(module.type_name)

            node = HierarchyNode(
                id=module.name,
                label=f"{layer_name}/{module.name}",
                level=HierarchyLevel.OPERATION,
                category=category,
                can_drill=False,  # Level 2 is the deepest
                inputs=[],
                outputs=[],
                source=SourceLocation(
                    file=source_file or "",
                    line=module.source_line,
                    end_line=module.source_line,
                ),
                origin=NodeOrigin(
                    type=OriginType.ASSIGNMENT,
                    code_snippet=f"{module.type_name}(...)",
                ),
                metadata={
                    "module_type": module.type_name,
                    "parent_layer": layer_name,
                    "is_inline": module.is_inline,
                },
            )
            nodes.append(node)

        return HierarchicalNodeGraph(
            id=f"{model_name}_{layer_name}_modules",
            label=f"{layer_name} - Modules",
            level=HierarchyLevel.OPERATION,
            hierarchy=Hierarchy(
                depth=2,
                path=[
                    HierarchyPath(
                        level=HierarchyLevel.LAYER,
                        label=model_name,
                        node_id=model_name,
                        graph_id=f"{model_name}_layers",
                        category=ComponentCategory.MODEL,
                        implementation=model_name,
                    ),
                    HierarchyPath(
                        level=HierarchyLevel.OPERATION,
                        label=layer_name,
                        node_id=layer_name,
                        graph_id=f"{model_name}_{layer_name}_modules",
                    ),
                ],
            ),
            nodes=nodes,
            edges=edges,
            agent_name=agent_name,
            source_file=source_file,
        )

    def _get_layer_category(self, type_name: str) -> Optional[ComponentCategory]:
        """Get the component category for a layer/module type."""
        if type_name in self.NN_MODULE_TYPES:
            return self.NN_MODULE_TYPES[type_name]
        return ComponentCategory.MODEL


def extract_model_layers(
    model_source: str,
    model_name: str = "model",
) -> list[LayerInfo]:
    """Convenience function to extract layers from model source."""
    extractor = ModelLayerExtractor()
    return extractor.extract_layers(model_source, model_name)


def build_model_layer_graph(
    model_source: str,
    model_name: str = "model",
    agent_name: str = "",
    source_file: Optional[str] = None,
) -> HierarchicalNodeGraph:
    """Convenience function to build a layer graph from model source."""
    extractor = ModelLayerExtractor()
    return extractor.build_layer_graph(
        model_source, model_name, agent_name, source_file
    )
