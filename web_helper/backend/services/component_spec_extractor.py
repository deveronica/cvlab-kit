"""Component Specification Extractor for automatic parameter/port inference.

This module analyzes component source files to extract:
1. Parameters from cfg.get() patterns
2. Input ports from __init__ signature (beyond cfg)
3. Output ports from forward/step/update methods

Key principle: Smart inference with zero developer burden.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from ..models.hierarchy import ComponentCategory, Port, SourceLocation


class ValueSource(str, Enum):
    """Source of parameter value.

    Used to distinguish where parameter values come from:
    - YAML: Value must come from YAML config (required=True, no default)
    - DEFAULT: Value has a code default, can be overridden by YAML
    - HARDCODE: Value is fixed in code (not from cfg pattern)
    """

    YAML = "yaml"  # cfg.get("key") without default - must be in YAML
    DEFAULT = "default"  # cfg.get("key", 0.01) - has code default
    HARDCODE = "hardcode"  # Literal value, not configurable


@dataclass
class ParamSpec:
    """Specification for a component parameter."""

    name: str
    default: Any = None
    required: bool = False
    param_type: str = "any"  # Inferred type
    description: str = ""
    source_line: int = 0
    value_source: ValueSource = ValueSource.YAML  # Where the value comes from


@dataclass
class PortSpec:
    """Specification for a component port."""

    name: str
    port_type: str = "any"  # tensor, parameters, scalar, etc.
    description: str = ""


@dataclass
class ComponentSpec:
    """Complete specification for a component."""

    category: str
    impl_name: str
    class_name: str

    # Parameters from cfg.get() patterns
    parameters: list[ParamSpec] = field(default_factory=list)

    # Input ports (constructor args beyond cfg)
    inputs: list[PortSpec] = field(default_factory=list)

    # Output ports (from forward/step methods)
    outputs: list[PortSpec] = field(default_factory=list)

    # Source location
    source_file: str = ""
    source_line: int = 0

    # Metadata
    docstring: str = ""
    base_class: str = ""
    is_delegation: bool = False  # Uses InterfaceMeta delegation


class CfgGetExtractor(ast.NodeVisitor):
    """Extracts cfg parameter patterns from __init__ method.

    Patterns recognized:
    - cfg.get("name")                    -> required=True
    - cfg.get("name", default)           -> required=False, default=default
    - cfg.get("name", None)              -> required=False, default=None
    - self.cfg.get("name", default)      -> same as cfg.get()
    - cfg.attr                           -> required=True, name from attr
    - cfg.attr(default=value)            -> required=False, name from attr
    - cfg("name", default=value)         -> required=False (direct call)
    """

    def __init__(self) -> None:
        self.params: list[ParamSpec] = []
        self._seen_names: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to find cfg.attr patterns."""
        # Handle: x = cfg.attr or self.x = cfg.attr
        if isinstance(node.value, ast.Attribute):
            param = self._extract_param_from_cfg_attr(node.value)
            if param and param.name not in self._seen_names:
                self.params.append(param)
                self._seen_names.add(param.name)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to find cfg patterns."""
        # Pattern 1: cfg.get("name", default)
        if self._is_cfg_get_call(node):
            param = self._extract_param_from_cfg_get(node)
            if param and param.name not in self._seen_names:
                self.params.append(param)
                self._seen_names.add(param.name)
        # Pattern 2: cfg.attr(default=value) - callable attribute
        elif self._is_cfg_attr_call(node):
            param = self._extract_param_from_cfg_attr_call(node)
            if param and param.name not in self._seen_names:
                self.params.append(param)
                self._seen_names.add(param.name)
        # Pattern 3: cfg("name", default=value) - direct call
        elif self._is_cfg_direct_call(node):
            param = self._extract_param_from_cfg_direct_call(node)
            if param and param.name not in self._seen_names:
                self.params.append(param)
                self._seen_names.add(param.name)

        self.generic_visit(node)

    def _is_cfg_object(self, node: ast.expr) -> bool:
        """Check if node represents cfg or self.cfg."""
        if isinstance(node, ast.Name) and node.id == "cfg":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "cfg":
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return True
        return False

    def _is_cfg_get_call(self, node: ast.Call) -> bool:
        """Check if this is a cfg.get() call."""
        if not isinstance(node.func, ast.Attribute):
            return False

        if node.func.attr != "get":
            return False

        # Check if it's cfg.get or self.cfg.get
        return self._is_cfg_object(node.func.value)

    def _is_cfg_attr_call(self, node: ast.Call) -> bool:
        """Check if this is a cfg.attr(default=...) call."""
        if not isinstance(node.func, ast.Attribute):
            return False

        # Exclude .get() which is handled separately
        if node.func.attr == "get":
            return False

        # Check if it's cfg.attr or self.cfg.attr
        return self._is_cfg_object(node.func.value)

    def _is_cfg_direct_call(self, node: ast.Call) -> bool:
        """Check if this is a cfg("name", ...) direct call."""
        return self._is_cfg_object(node.func)

    def _extract_param_from_cfg_attr(self, node: ast.Attribute) -> ParamSpec | None:
        """Extract parameter from cfg.attr access."""
        if not self._is_cfg_object(node.value):
            return None

        # The attribute name is the parameter name
        name = node.attr
        return ParamSpec(
            name=name,
            default=None,
            required=True,
            param_type="any",
            source_line=node.lineno,
            value_source=ValueSource.YAML,  # No default, must come from YAML
        )

    def _extract_param_from_cfg_attr_call(self, node: ast.Call) -> ParamSpec | None:
        """Extract parameter from cfg.attr(default=...) call."""
        if not isinstance(node.func, ast.Attribute):
            return None

        name = node.func.attr
        default = None
        required = True
        param_type = "any"

        # Check for default keyword argument
        for kw in node.keywords:
            if kw.arg == "default":
                required = False
                default, param_type = self._extract_default_value(kw.value)
                break

        # Also check positional arg as default
        if node.args:
            required = False
            default, param_type = self._extract_default_value(node.args[0])

        return ParamSpec(
            name=name,
            default=default,
            required=required,
            param_type=param_type,
            source_line=node.lineno,
            value_source=ValueSource.DEFAULT if not required else ValueSource.YAML,
        )

    def _extract_param_from_cfg_direct_call(self, node: ast.Call) -> ParamSpec | None:
        """Extract parameter from cfg("name", default=...) call."""
        if not node.args:
            return None

        # First argument is the parameter name
        name_arg = node.args[0]
        if not isinstance(name_arg, ast.Constant) or not isinstance(name_arg.value, str):
            return None

        name = name_arg.value
        default = None
        required = True
        param_type = "any"

        # Check for default in second positional arg
        if len(node.args) > 1:
            required = False
            default, param_type = self._extract_default_value(node.args[1])
        # Or in keyword arg
        else:
            for kw in node.keywords:
                if kw.arg == "default":
                    required = False
                    default, param_type = self._extract_default_value(kw.value)
                    break

        return ParamSpec(
            name=name,
            default=default,
            required=required,
            param_type=param_type,
            source_line=node.lineno,
            value_source=ValueSource.DEFAULT if not required else ValueSource.YAML,
        )

    def _extract_param_from_cfg_get(self, node: ast.Call) -> ParamSpec | None:
        """Extract parameter info from cfg.get() call."""
        if not node.args:
            return None

        # First argument is the parameter name
        name_arg = node.args[0]
        if not isinstance(name_arg, ast.Constant) or not isinstance(
            name_arg.value, str
        ):
            return None

        name = name_arg.value

        # Check for default value
        default = None
        required = True
        param_type = "any"

        if len(node.args) > 1:
            # Has default value
            required = False
            default, param_type = self._extract_default_value(node.args[1])
        elif node.keywords:
            # Check for default=... keyword
            for kw in node.keywords:
                if kw.arg == "default":
                    required = False
                    default, param_type = self._extract_default_value(kw.value)
                    break

        return ParamSpec(
            name=name,
            default=default,
            required=required,
            param_type=param_type,
            source_line=node.lineno,
            value_source=ValueSource.DEFAULT if not required else ValueSource.YAML,
        )

    def _extract_default_value(self, node: ast.expr) -> tuple[Any, str]:
        """Extract default value and infer type."""
        if isinstance(node, ast.Constant):
            value = node.value
            if value is None:
                return None, "any"
            elif isinstance(value, bool):
                return value, "bool"
            elif isinstance(value, int):
                return value, "int"
            elif isinstance(value, float):
                return value, "float"
            elif isinstance(value, str):
                return value, "str"
            return value, "any"

        elif isinstance(node, ast.List):
            return "[]", "list"

        elif isinstance(node, ast.Dict):
            return "{}", "dict"

        elif isinstance(node, ast.Tuple):
            # Try to extract tuple values
            values = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant):
                    values.append(elt.value)
            if values:
                return tuple(values), "tuple"
            return "()", "tuple"

        elif isinstance(node, ast.Name):
            # Reference to another variable (e.g., None, True, False)
            if node.id == "None":
                return None, "any"
            elif node.id == "True":
                return True, "bool"
            elif node.id == "False":
                return False, "bool"
            return f"<{node.id}>", "any"

        return "<complex>", "any"


class InitSignatureExtractor(ast.NodeVisitor):
    """Extracts __init__ signature to find input ports.

    Input ports are constructor parameters beyond 'self' and 'cfg':
    - params/parameters -> for optimizer
    - dataset -> for dataloader
    - model -> for various components
    """

    INPUT_PARAM_TYPES = {
        "params": "parameters",
        "parameters": "parameters",
        "model": "model",
        "dataset": "dataset",
        "data": "tensor",
        "x": "tensor",
        "input": "tensor",
        "inputs": "tensor",
    }

    def __init__(self) -> None:
        self.inputs: list[PortSpec] = []
        self.found_init = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to find __init__."""
        if node.name == "__init__" and not self.found_init:
            self.found_init = True
            self._extract_inputs(node)

    def _extract_inputs(self, node: ast.FunctionDef) -> None:
        """Extract input ports from __init__ parameters."""
        for arg in node.args.args:
            name = arg.arg

            # Skip self and cfg
            if name in ("self", "cfg"):
                continue

            # Determine port type
            port_type = self.INPUT_PARAM_TYPES.get(name, "any")

            self.inputs.append(PortSpec(name=name, port_type=port_type))


class OutputExtractor(ast.NodeVisitor):
    """Extracts output ports from forward/step/update methods.

    Output ports are determined by:
    - Method name (forward -> tensor output)
    - Return statements
    - Known patterns
    """

    METHOD_OUTPUT_TYPES = {
        "forward": [PortSpec(name="output", port_type="tensor")],
        "step": [PortSpec(name="loss", port_type="scalar")],
        "update": [],
        "compute": [PortSpec(name="metrics", port_type="dict")],
        "reset": [],
        "__call__": [PortSpec(name="output", port_type="any")],
    }

    def __init__(self) -> None:
        self.outputs: list[PortSpec] = []
        self.found_methods: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to find output methods."""
        if node.name in self.METHOD_OUTPUT_TYPES:
            self.found_methods.add(node.name)

            # Add standard outputs for this method
            for port in self.METHOD_OUTPUT_TYPES[node.name]:
                if not any(p.name == port.name for p in self.outputs):
                    self.outputs.append(port)

        self.generic_visit(node)


class DelegationDetector(ast.NodeVisitor):
    """Detects if component uses delegation pattern (InterfaceMeta).

    Patterns:
    - self.opt = torch.optim.Adam(...)
    - self.model = torchvision.models.resnet18(...)
    """

    DELEGATION_ATTRS = {"opt", "model", "module", "loss", "transform", "metric"}

    def __init__(self) -> None:
        self.is_delegation = False
        self.wrapped_class: str | None = None

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for delegation assignment in __init__."""
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                if (
                    isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in self.DELEGATION_ATTRS
                ):
                    # Check if RHS is a call to external class
                    if isinstance(node.value, ast.Call):
                        class_name = self._get_call_name(node.value)
                        if class_name and not class_name.startswith("self."):
                            self.is_delegation = True
                            self.wrapped_class = class_name

        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of the called class/function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None


class ComponentSpecExtractor:
    """Main extractor for component specifications.

    Usage:
        extractor = ComponentSpecExtractor()
        spec = extractor.extract_from_file("cvlabkit/component/model/resnet18.py")
        # or
        spec = extractor.extract_from_source(source_code, "resnet18", "model")
    """

    # Base class to category mapping
    BASE_CLASS_CATEGORIES = {
        "Model": "model",
        "Loss": "loss",
        "Optimizer": "optimizer",
        "Transform": "transform",
        "Dataset": "dataset",
        "DataLoader": "dataloader",
        "Metric": "metric",
        "Scheduler": "scheduler",
        "Sampler": "sampler",
        "Checkpoint": "checkpoint",
        "Logger": "logger",
    }

    # Standard outputs by category
    CATEGORY_OUTPUTS = {
        "model": [
            PortSpec(name="forward", port_type="tensor"),
            PortSpec(name="parameters", port_type="parameters"),
        ],
        "optimizer": [],
        "loss": [PortSpec(name="value", port_type="scalar")],
        "transform": [PortSpec(name="output", port_type="any")],
        "dataset": [PortSpec(name="item", port_type="any")],
        "dataloader": [PortSpec(name="batch", port_type="tensor")],
        "metric": [PortSpec(name="metrics", port_type="dict")],
        "scheduler": [],
        "sampler": [PortSpec(name="indices", port_type="list")],
        "checkpoint": [],
        "logger": [],
    }

    def extract_from_file(self, file_path: str | Path) -> ComponentSpec | None:
        """Extract component spec from a file."""
        path = Path(file_path)
        if not path.exists():
            return None

        source = path.read_text(encoding="utf-8")

        # Infer category and impl_name from path
        # Expected: cvlabkit/component/{category}/{impl_name}.py
        parts = path.parts
        try:
            component_idx = parts.index("component")
            category = parts[component_idx + 1]
            impl_name = path.stem
        except (ValueError, IndexError):
            category = "unknown"
            impl_name = path.stem

        return self.extract_from_source(source, impl_name, category, str(path))

    def extract_from_source(
        self,
        source: str,
        impl_name: str,
        category: str,
        source_file: str = "",
    ) -> ComponentSpec | None:
        """Extract component spec from source code."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        # Find the main class
        class_node = self._find_component_class(tree, category)
        if not class_node:
            return None

        # Extract docstring
        docstring = ast.get_docstring(class_node) or ""

        # Extract base class
        base_class = self._get_base_class(class_node)

        # Extract parameters from cfg.get()
        cfg_extractor = CfgGetExtractor()
        cfg_extractor.visit(class_node)
        parameters = cfg_extractor.params

        # Extract input ports from __init__ signature
        init_extractor = InitSignatureExtractor()
        init_extractor.visit(class_node)
        inputs = init_extractor.inputs

        # Extract output ports from methods
        output_extractor = OutputExtractor()
        output_extractor.visit(class_node)
        outputs = output_extractor.outputs

        # Add standard outputs for category if not already present
        for standard_port in self.CATEGORY_OUTPUTS.get(category, []):
            if not any(p.name == standard_port.name for p in outputs):
                outputs.append(standard_port)

        # Detect delegation pattern
        delegation_detector = DelegationDetector()
        delegation_detector.visit(class_node)

        return ComponentSpec(
            category=category,
            impl_name=impl_name,
            class_name=class_node.name,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            source_file=source_file,
            source_line=class_node.lineno,
            docstring=docstring,
            base_class=base_class,
            is_delegation=delegation_detector.is_delegation,
        )

    def _find_component_class(
        self, tree: ast.Module, category: str
    ) -> ast.ClassDef | None:
        """Find the main component class in the module."""
        # First, try to find class inheriting from expected base
        expected_base = category.capitalize()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it inherits from expected base
                for base in node.bases:
                    base_name = self._get_base_name(base)
                    if base_name == expected_base or base_name in self.BASE_CLASS_CATEGORIES:
                        return node

        # Fallback: find the first non-abstract class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Skip abstract classes (usually named with ABC or Abstract prefix)
                if not node.name.startswith("Abstract") and "ABC" not in node.name:
                    return node

        return None

    def _get_base_class(self, class_node: ast.ClassDef) -> str:
        """Get the base class name."""
        if class_node.bases:
            return self._get_base_name(class_node.bases[0])
        return ""

    def _get_base_name(self, node: ast.expr) -> str:
        """Get name from a base class node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            # Generic like Model[ConfigType]
            return self._get_base_name(node.value)
        return ""

    def to_ports(self, spec: ComponentSpec) -> tuple[list[Port], list[Port]]:
        """Convert ComponentSpec to Port lists for node graph."""
        input_ports = [Port(name=p.name, type=p.port_type) for p in spec.inputs]
        output_ports = [Port(name=p.name, type=p.port_type) for p in spec.outputs]
        return input_ports, output_ports


# Convenience function for quick extraction
def extract_component_spec(file_path: str | Path) -> ComponentSpec | None:
    """Extract component specification from a file."""
    extractor = ComponentSpecExtractor()
    return extractor.extract_from_file(file_path)
