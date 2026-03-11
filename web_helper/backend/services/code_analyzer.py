"""Code analyzer for extracting nodes and edges from Agent source code.

This module provides AST-based analysis to extract:
1. Component nodes from setup() - self.xxx = self.create.yyy()
2. Edges from setup() - dependency relationships (e.g., model.parameters() -> optimizer)
3. Edges from train_step() - data flow (e.g., outputs = self.model(inputs))

Key principle: NO Config-based inference. All edges come from actual code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models.hierarchy import (
    CodeFlowEdge,
    ComponentCategory,
    DrillTarget,
    FlowType,
    HierarchyLevel,
    HierarchyNode,
    ImplSource,
    MethodCategory,
    NodeOrigin,
    OriginType,
    Port,
    PropertyInfo,
    PropertySummary,
    SourceLocation,
    ValueSource,
)

# Optional import for dynamic port extraction
try:
    from .component_spec_extractor import ComponentSpecExtractor

    _SPEC_EXTRACTOR_AVAILABLE = True
except ImportError:
    _SPEC_EXTRACTOR_AVAILABLE = False


@dataclass
class TrackedVariable:
    """Information about a tracked variable."""

    name: str
    node_id: str
    port: str = "out"
    source_line: int = 0


@dataclass
class ExtractedEdge:
    """Raw edge data before conversion to CodeFlowEdge."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str
    flow_type: FlowType
    variable_name: str | None
    extracted_from: str
    source_line: int = 0


class VariableTracker:
    """Tracks variable assignments to determine data flow edges.

    This is the core of code-driven edge extraction.

    Example:
        tracker = VariableTracker()
        # After: outputs = self.model(inputs)
        tracker.assign("outputs", "model", "out")
        # After: loss = self.loss_fn(outputs, labels)
        # Resolves "outputs" -> ("model", "out") for edge creation
    """

    def __init__(self) -> None:
        self.variables: dict[str, TrackedVariable] = {}

    def assign(
        self, var_name: str, node_id: str, port: str = "out", source_line: int = 0
    ) -> None:
        """Register a variable assignment."""
        self.variables[var_name] = TrackedVariable(
            name=var_name, node_id=node_id, port=port, source_line=source_line
        )

    def resolve(self, var_name: str) -> TrackedVariable | None:
        """Resolve a variable to its source node."""
        return self.variables.get(var_name)

    def clear(self) -> None:
        """Clear all tracked variables."""
        self.variables.clear()


class ComponentExtractor(ast.NodeVisitor):
    """Extracts component nodes from setup() method.

    Looks for patterns:
    - self.xxx = self.create.yyy()           -> instance attribute
    - self.xxx = self.create.yyy.zzz()       -> named instance attribute
    - xxx = self.create.yyy()                -> local variable
    - xxx = self.create.yyy.zzz()            -> named local variable
    """

    # Attribute names that are not components (should be ignored)
    IGNORED_ATTRS = {"create", "cfg", "device", "logger", "trainer"}

    def __init__(self, source_file: str) -> None:
        self.source_file = source_file
        self.nodes: list[HierarchyNode] = []
        self.self_components: dict[str, str] = {}  # attr_name -> category
        self.local_components: dict[str, str] = {}  # local_var -> category
        self.uncovered_nodes: list[HierarchyNode] = []  # Nodes that can't be covered

    def visit_Assign(self, node: ast.Assign) -> None:
        """Process assignment statements."""
        if not node.targets:
            return

        target = node.targets[0]

        # Case 1: self.xxx = ... (instance attribute)
        if isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                attr_name = target.attr

                # Skip ignored attributes
                if attr_name.startswith("_") or attr_name in self.IGNORED_ATTRS:
                    self.generic_visit(node)
                    return

                if isinstance(node.value, ast.Call):
                    create_info = self._extract_create_call(node.value)
                    if create_info:
                        # ✅ Valid create call - covered
                        category, create_path, can_drill, impl, impl_source = (
                            create_info
                        )
                        self._add_component_node(
                            attr_name=attr_name,
                            category=category,
                            create_path=create_path,
                            can_drill=can_drill,
                            line=node.lineno,
                            code_snippet=ast.unparse(node),
                            is_local=False,
                            impl=impl,
                            impl_source=impl_source,
                        )
                    else:
                        # ❌ Not a create call - uncovered
                        self._add_uncovered_node(
                            attr_name=attr_name,
                            line=node.lineno,
                            code_snippet=ast.unparse(node),
                            reason="Not a self.create.xxx() call",
                        )
                else:
                    # ❌ Not a call at all (e.g., self.x = value) - uncovered
                    self._add_uncovered_node(
                        attr_name=attr_name,
                        line=node.lineno,
                        code_snippet=ast.unparse(node),
                        reason="Direct assignment, not a component creation",
                    )

        # Case 2: xxx = self.create.yyy() (local variable)
        elif isinstance(target, ast.Name):
            var_name = target.id
            if isinstance(node.value, ast.Call):
                create_info = self._extract_create_call(node.value)
                if create_info:
                    category, create_path, can_drill, impl, impl_source = create_info
                    self._add_component_node(
                        attr_name=var_name,
                        category=category,
                        create_path=create_path,
                        can_drill=can_drill,
                        line=node.lineno,
                        code_snippet=ast.unparse(node),
                        is_local=True,
                        impl=impl,
                        impl_source=impl_source,
                    )

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Detect dynamic component creation in loops - mark as uncovered."""
        # Check if loop body contains self.xxx = self.create...
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            attr_name = target.attr
                            if (
                                not attr_name.startswith("_")
                                and attr_name not in self.IGNORED_ATTRS
                            ):
                                self._add_uncovered_node(
                                    attr_name=f"{attr_name} (loop)",
                                    line=node.lineno,
                                    code_snippet=f"for {ast.unparse(node.target)} in ...: self.{attr_name} = ...",
                                    reason="Dynamic creation in loop - not visualizable",
                                )
        # Don't call generic_visit to avoid double-processing assignments

    def visit_If(self, node: ast.If) -> None:
        """Detect conditional component creation - mark as uncovered."""
        # Check if condition body contains self.xxx = self.create...
        for stmt in node.body + node.orelse:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            attr_name = target.attr
                            if (
                                not attr_name.startswith("_")
                                and attr_name not in self.IGNORED_ATTRS
                            ):
                                # Check if it's a create call
                                if isinstance(stmt.value, ast.Call):
                                    create_info = self._extract_create_call(stmt.value)
                                    if create_info:
                                        self._add_uncovered_node(
                                            attr_name=f"{attr_name} (conditional)",
                                            line=stmt.lineno,
                                            code_snippet=ast.unparse(stmt),
                                            reason="Conditional creation - may not always exist",
                                        )
        self.generic_visit(node)

    def _add_uncovered_node(
        self, attr_name: str, line: int, code_snippet: str, reason: str
    ) -> None:
        """Add an uncovered node that cannot be properly visualized."""
        node = HierarchyNode(
            id=f"uncovered_{attr_name}_{line}",
            label=attr_name,
            level=HierarchyLevel.COMPONENT,
            origin=NodeOrigin(
                type=OriginType.CREATE_CALL,
                create_path=[],
                code_snippet=code_snippet,
                source=SourceLocation(file=self.source_file, line=line),
            ),
            can_drill=False,
            category=ComponentCategory.UNKNOWN,
            inputs=[],
            outputs=[],
            source=SourceLocation(file=self.source_file, line=line),
            metadata={
                "is_uncovered": True,
                "uncovered_reason": reason,
            },
        )
        self.uncovered_nodes.append(node)

    def _unwrap_chained_calls(self, node: ast.expr) -> ast.Call | None:
        """Unwrap chained method calls like self.create.model().to(device).

        Returns the innermost create call if found.
        """
        if not isinstance(node, ast.Call):
            return None

        # Check if it's a chained call like xxx().to() - recurse first
        if isinstance(node.func, ast.Attribute):
            # Check the value (left side) of the call
            if isinstance(node.func.value, ast.Call):
                inner = self._unwrap_chained_calls(node.func.value)
                if inner:
                    return inner

        # Check if this is a direct create call
        if self._is_create_call(node):
            return node

        return None

    def _is_create_call(self, call_node: ast.Call) -> bool:
        """Check if a call is self.create.xxx() or self.create.xxx.yyy().

        Structure examples:
        - self.create.model(): func is Attribute(attr='model', value=Attribute(attr='create', value=Name('self')))
        - self.create.dataset.train(): func is Attribute(attr='train', value=Attribute(attr='dataset', ...))
        """
        func = call_node.func

        if not isinstance(func, ast.Attribute):
            return False

        # Walk back to find self.create pattern
        current = func.value

        while isinstance(current, ast.Attribute):
            if current.attr == "create":
                if isinstance(current.value, ast.Name) and current.value.id == "self":
                    return True
            current = current.value

        return False

    def _extract_create_call(
        self, call_node: ast.Call
    ) -> tuple[ComponentCategory, list[str], bool, str | None, ImplSource] | None:
        """Extract create call info: self.create.xxx() or self.create.xxx.yyy()

        Also handles chained calls like self.create.model().to(device)

        Structure example:
        - self.create.model(): func is Attribute(attr='model', value=Attribute(attr='create', value=Name('self')))
        - self.create.dataset.train(): func is Attribute(attr='train', value=Attribute(attr='dataset', value=...))

        Returns:
            Tuple of (category, create_path, can_drill, impl, impl_source) or None
            - impl: Implementation name if determinable from code, None if from YAML
            - impl_source: HARDCODE (positional arg), DEFAULT (keyword), or YAML (no args)
        """
        # First, try to unwrap chained calls like .to(device)
        actual_call = self._unwrap_chained_calls(call_node)
        if actual_call is None:
            return None

        func = actual_call.func

        if not isinstance(func, ast.Attribute):
            return None

        create_path: list[str] = []
        current = func

        # Walk back the attribute chain, collecting attrs until we hit 'create'
        while isinstance(current, ast.Attribute):
            if current.attr == "create":
                # Found create, verify next is self
                if isinstance(current.value, ast.Name) and current.value.id == "self":
                    break
                else:
                    return None
            create_path.insert(0, current.attr)
            current = current.value
        else:
            # Didn't find create pattern
            return None

        if not create_path:
            return None

        # Determine category from first element
        category = self._category_from_name(create_path[0])
        can_drill = category in {
            ComponentCategory.MODEL,
            ComponentCategory.LOSS,
            ComponentCategory.TRANSFORM,
            ComponentCategory.DATASET,
        }

        # Extract impl and impl_source from arguments
        impl, impl_source = self._extract_impl_from_args(actual_call)

        return category, create_path, can_drill, impl, impl_source

    def _extract_impl_from_args(
        self, call_node: ast.Call
    ) -> tuple[str | None, ImplSource]:
        """Extract impl name and source from create call arguments.

        Patterns:
        - create.model("resnet18")         -> ("resnet18", HARDCODE)
        - create.model(impl="resnet18")    -> ("resnet18", HARDCODE)
        - create.model(default="resnet18") -> ("resnet18", DEFAULT)
        - create.model()                   -> (None, YAML)
        """
        # Check positional argument first (hardcode)
        if call_node.args:
            first_arg = call_node.args[0]
            impl = self._extract_string_value(first_arg)
            if impl:
                return impl, ImplSource.HARDCODE

        # Check keyword arguments
        for keyword in call_node.keywords:
            # 'impl' keyword argument (hardcode)
            if keyword.arg == "impl":
                impl = self._extract_string_value(keyword.value)
                if impl:
                    return impl, ImplSource.HARDCODE

            # 'default' keyword argument
            if keyword.arg == "default":
                impl = self._extract_string_value(keyword.value)
                if impl:
                    return impl, ImplSource.DEFAULT

        # No impl in code, must come from YAML
        return None, ImplSource.YAML

    def _extract_string_value(self, node: ast.expr) -> str | None:
        """Extract string value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _category_from_name(self, name: str) -> ComponentCategory:
        """Map create path to component category."""
        mapping = {
            "model": ComponentCategory.MODEL,
            "optimizer": ComponentCategory.OPTIMIZER,
            "loss": ComponentCategory.LOSS,
            "dataset": ComponentCategory.DATASET,
            "dataloader": ComponentCategory.DATALOADER,
            "transform": ComponentCategory.TRANSFORM,
            "metric": ComponentCategory.METRIC,
            "scheduler": ComponentCategory.SCHEDULER,
            "sampler": ComponentCategory.SAMPLER,
            "checkpoint": ComponentCategory.CHECKPOINT,
        }
        return mapping.get(name, ComponentCategory.UNKNOWN)

    def _get_ports_for_category(
        self, category: ComponentCategory
    ) -> tuple[list[Port], list[Port]]:
        """Get meaningful input/output ports based on component category.

        Port policy:
        - Independent components (model, dataset, loss, transform, metric): no inputs
        - Dependent components (optimizer, dataloader, scheduler, sampler): have inputs
        """
        # Port definitions per category
        port_definitions: dict[ComponentCategory, tuple[list[Port], list[Port]]] = {
            # Independent components - no input ports
            ComponentCategory.MODEL: (
                [],  # No inputs - model is self-contained
                [
                    Port(name="parameters", type="parameters"),
                    Port(name="forward", type="tensor"),
                ],
            ),
            ComponentCategory.DATASET: (
                [],  # No inputs
                [Port(name="data", type="dataset")],
            ),
            ComponentCategory.LOSS: (
                [],  # No inputs - loss function is self-contained
                [Port(name="value", type="scalar")],
            ),
            ComponentCategory.TRANSFORM: (
                [],  # No inputs
                [Port(name="transformed", type="tensor")],
            ),
            ComponentCategory.METRIC: (
                [],  # No inputs
                [Port(name="value", type="scalar")],
            ),
            ComponentCategory.CHECKPOINT: (
                [],  # No inputs
                [],  # No outputs - side effect only
            ),
            # Dependent components - have input ports
            ComponentCategory.OPTIMIZER: (
                [Port(name="parameters", type="parameters")],
                [],  # No outputs - modifies in place
            ),
            ComponentCategory.DATALOADER: (
                [Port(name="dataset", type="dataset")],
                [Port(name="batch", type="tensor")],
            ),
            ComponentCategory.SCHEDULER: (
                [Port(name="optimizer", type="optimizer")],
                [],  # No outputs - modifies optimizer
            ),
            ComponentCategory.SAMPLER: (
                [Port(name="dataset", type="dataset")],
                [Port(name="indices", type="indices")],
            ),
            ComponentCategory.UNKNOWN: (
                [],  # Default: no inputs
                [Port(name="out", type="any")],
            ),
        }
        return port_definitions.get(category, ([], [Port(name="out", type="any")]))

    def _get_ports_for_component(
        self,
        category: ComponentCategory,
        impl_name: str,
        project_root: Path | None = None,
    ) -> tuple[list[Port], list[Port]]:
        """Get ports for a specific component implementation.

        Attempts to extract ports dynamically from component source file.
        Falls back to category-based defaults if extraction fails.

        Args:
            category: Component category (model, loss, etc.)
            impl_name: Implementation name (resnet18, cross_entropy, etc.)
            project_root: Project root path for finding component files

        Returns:
            Tuple of (input_ports, output_ports)
        """
        # Fallback to category defaults if extractor not available
        if not _SPEC_EXTRACTOR_AVAILABLE or project_root is None:
            return self._get_ports_for_category(category)

        # Try to find and analyze component file
        component_path = self._find_component_file(
            category.value, impl_name, project_root
        )
        if not component_path or not component_path.exists():
            return self._get_ports_for_category(category)

        try:
            extractor = ComponentSpecExtractor()
            spec = extractor.extract_from_file(component_path)
            if spec:
                return extractor.to_ports(spec)
        except Exception:
            pass  # Fall through to default

        return self._get_ports_for_category(category)

    def _find_component_file(
        self, category: str, impl_name: str, project_root: Path
    ) -> Path | None:
        """Find component source file path."""
        # Standard path: cvlabkit/component/{category}/{impl_name}.py
        component_path = (
            project_root / "cvlabkit" / "component" / category / f"{impl_name}.py"
        )
        if component_path.exists():
            return component_path
        return None

    def _impl_source_to_value_source(
        self, impl_source: ImplSource, has_value: bool = True
    ) -> ValueSource:
        """Convert ImplSource to ValueSource for UI display.

        Args:
            impl_source: Source of implementation from code analysis
            has_value: Whether the value is actually set

        Returns:
            ValueSource for UI display
        """
        if impl_source == ImplSource.HARDCODE:
            return ValueSource.HARDCODE
        elif impl_source == ImplSource.DEFAULT:
            return ValueSource.DEFAULT
        else:  # YAML
            # YAML source with value = CONFIG, without value = REQUIRED
            return ValueSource.CONFIG if has_value else ValueSource.REQUIRED

    def _build_properties(
        self,
        impl: str | None,
        impl_source: ImplSource,
    ) -> tuple[list[PropertyInfo], PropertySummary]:
        """Build properties list and summary for a component node.

        Args:
            impl: Implementation name if known
            impl_source: Source of implementation

        Returns:
            Tuple of (properties list, property summary)
        """
        properties: list[PropertyInfo] = []
        summary = PropertySummary()

        # Add impl property
        value_source = self._impl_source_to_value_source(impl_source, impl is not None)
        properties.append(
            PropertyInfo(
                name="impl",
                value=impl,
                source=value_source,
            )
        )

        # Update summary counts
        if value_source == ValueSource.REQUIRED:
            summary.required_count += 1
        elif value_source == ValueSource.CONFIG:
            summary.config_count += 1
        elif value_source == ValueSource.DEFAULT:
            summary.default_count += 1
        elif value_source == ValueSource.HARDCODE:
            summary.hardcode_count += 1

        return properties, summary

    def _add_component_node(
        self,
        attr_name: str,
        category: ComponentCategory,
        create_path: list[str],
        can_drill: bool,
        line: int,
        code_snippet: str,
        is_local: bool = False,
        impl: str | None = None,
        impl_source: ImplSource = ImplSource.YAML,
    ) -> None:
        """Add a component node to the list.

        Args:
            attr_name: Variable name (self.xxx or local xxx)
            category: Component category (model, optimizer, etc.)
            create_path: Path in create call (e.g., ['dataset', 'train'])
            can_drill: Whether node can be drilled into
            line: Source line number
            code_snippet: Original code snippet
            is_local: True if local variable, False if instance attribute
            impl: Implementation name if known from code (e.g., "resnet18")
            impl_source: Source of impl (HARDCODE, DEFAULT, or YAML)
        """
        inputs, outputs = self._get_ports_for_category(category)

        # Use create_path as label for named components (e.g., dataset.train -> train)
        label = create_path[-1] if len(create_path) > 1 else attr_name

        # Build metadata with impl info for Builder context
        metadata: dict[str, Any] = {
            "impl": impl,
            "impl_source": impl_source.value,
        }

        # Build properties with value source info
        properties, property_summary = self._build_properties(impl, impl_source)

        node = HierarchyNode(
            id=attr_name,
            label=label,
            level=HierarchyLevel.COMPONENT,
            origin=NodeOrigin(
                type=OriginType.CREATE_CALL,
                create_path=create_path,
                code_snippet=code_snippet,
                source=SourceLocation(file=self.source_file, line=line),
                impl_source=impl_source,
            ),
            can_drill=can_drill,
            drill_target=DrillTarget(type="component", path=attr_name)
            if can_drill
            else None,
            category=category,
            inputs=inputs,
            outputs=outputs,
            source=SourceLocation(file=self.source_file, line=line),
            properties=properties,
            property_summary=property_summary,
            metadata=metadata,
        )
        self.nodes.append(node)

        # Track in appropriate dictionary
        if is_local:
            self.local_components[attr_name] = category.value
        else:
            self.self_components[attr_name] = category.value


class SetupEdgeExtractor(ast.NodeVisitor):
    """Extracts edges from setup() method.

    Looks for dependency patterns like:
    - self.optimizer = self.create.optimizer(self.model.parameters())
      -> model -> optimizer (parameters flow)
    """

    def __init__(self, source_file: str, known_components: dict[str, str]) -> None:
        self.source_file = source_file
        self.known_components = known_components
        self.edges: list[ExtractedEdge] = []
        self.tracker = VariableTracker()

    def visit_Assign(self, node: ast.Assign) -> None:
        """Process assignments for edge extraction.

        Handles two cases:
        1. self.xxx = self.create.yyy(...) - instance attribute assignment
        2. xxx = self.create.yyy(...) - local variable (track for later use)
        """
        if not node.targets:
            self.generic_visit(node)
            return

        target = node.targets[0]

        # Case 1: self.xxx = ... (instance attribute)
        if isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                target_component = target.attr
                if target_component in self.known_components:
                    # Check if it's a create call with arguments
                    if isinstance(node.value, ast.Call):
                        self._extract_edges_from_call(
                            node.value, target_component, node.lineno
                        )

        # Case 2: xxx = self.create.yyy() (local variable)
        # Track local variables so they can be resolved when used as arguments later
        elif isinstance(target, ast.Name):
            var_name = target.id
            if var_name in self.known_components:
                # This local variable is a known component, track it
                self.tracker.assign(var_name, var_name, "self", node.lineno)

                # Also extract edges from its creation
                if isinstance(node.value, ast.Call):
                    self._extract_edges_from_call(node.value, var_name, node.lineno)

        self.generic_visit(node)

    def _extract_edges_from_call(
        self, call: ast.Call, target_component: str, line: int
    ) -> None:
        """Extract edges from call arguments.

        Handles both positional and keyword arguments:
        - self.create.optimizer(self.model.parameters())  # positional
        - self.create.dataset.train(transform=transform)  # keyword
        """
        for arg in call.args:
            self._process_argument(arg, target_component, line, keyword_name=None)
        for keyword in call.keywords:
            # Pass the keyword name (e.g., 'transform', 'sampler')
            self._process_argument(
                keyword.value, target_component, line, keyword_name=keyword.arg
            )

    def _process_argument(
        self,
        arg: ast.expr,
        target_component: str,
        line: int,
        keyword_name: str | None = None,
    ) -> None:
        """Process a single argument for edge extraction.

        Args:
            arg: The AST expression for the argument
            target_component: The component receiving this argument
            line: Source line number
            keyword_name: If this is a keyword argument, its name (e.g., 'transform')
        """
        # Case 1: self.model.parameters() -> optimizer (params port)
        if isinstance(arg, ast.Call):
            if isinstance(arg.func, ast.Attribute):
                if arg.func.attr == "parameters":
                    # self.model.parameters()
                    if isinstance(arg.func.value, ast.Attribute):
                        if (
                            isinstance(arg.func.value.value, ast.Name)
                            and arg.func.value.value.id == "self"
                        ):
                            source_component = arg.func.value.attr
                            if source_component in self.known_components:
                                self.edges.append(
                                    ExtractedEdge(
                                        source_node=source_component,
                                        source_port="parameters",  # model.parameters() output port
                                        target_node=target_component,
                                        target_port="params",  # Optimizer initInput: "params" → handle "init_params"
                                        flow_type=FlowType.PARAMETERS,
                                        variable_name="self."
                                        + source_component
                                        + ".parameters()",
                                        extracted_from="setup",
                                        source_line=line,
                                    )
                                )

        # Case 2: self.xxx (direct component reference) - e.g., self.dataset -> dataloader
        elif isinstance(arg, ast.Attribute):
            if isinstance(arg.value, ast.Name) and arg.value.id == "self":
                source_component = arg.attr
                if source_component in self.known_components:
                    # Determine target port based on keyword name or source category
                    target_port = self._infer_target_port(
                        source_component, keyword_name
                    )
                    self.edges.append(
                        ExtractedEdge(
                            source_node=source_component,
                            source_port="self",  # Component instance
                            target_node=target_component,
                            target_port=target_port,
                            flow_type=FlowType.REFERENCE,
                            variable_name=f"self.{source_component}",
                            extracted_from="setup",
                            source_line=line,
                        )
                    )

        # Case 3: Direct variable reference (local variable)
        elif isinstance(arg, ast.Name):
            tracked = self.tracker.resolve(arg.id)
            if tracked and tracked.node_id in self.known_components:
                # Determine target port based on keyword name or source category
                target_port = self._infer_target_port(tracked.node_id, keyword_name)
                self.edges.append(
                    ExtractedEdge(
                        source_node=tracked.node_id,
                        source_port=tracked.port,
                        target_node=target_component,
                        target_port=target_port,
                        flow_type=FlowType.REFERENCE,
                        variable_name=arg.id,
                        extracted_from="setup",
                        source_line=line,
                    )
                )

    def _infer_target_port(
        self, source_component: str, keyword_name: str | None
    ) -> str:
        """Infer the target port name based on source component and keyword name.

        Args:
            source_component: The source component name
            keyword_name: The keyword argument name if any

        Returns:
            The inferred target port name
        """
        # If keyword name is provided, use it directly
        if keyword_name:
            return keyword_name

        # Infer from source component's category
        source_category = self.known_components.get(source_component, "")

        # Category-based inference
        port_mapping = {
            "dataset": "dataset",  # dataset -> dataloader.dataset
            "transform": "transform",  # transform -> dataset.transform
            "model": "model",  # model -> checkpoint.model
            "optimizer": "optimizer",  # optimizer -> scheduler.optimizer or checkpoint.optimizer
            "sampler": "sampler",  # sampler -> dataloader.sampler
        }

        return port_mapping.get(source_category, "in")


class TrainStepEdgeExtractor(ast.NodeVisitor):
    """Extracts data flow edges from train_step() method.

    Tracks variable assignments and extracts edges like:
    - outputs = self.model(inputs) -> model produces outputs
    - loss = self.loss_fn(outputs, labels) -> outputs flows to loss_fn
    """

    def __init__(self, source_file: str, known_components: dict[str, str]) -> None:
        self.source_file = source_file
        self.known_components = known_components
        self.edges: list[ExtractedEdge] = []
        self.tracker = VariableTracker()

    def visit_Assign(self, node: ast.Assign) -> None:
        """Process assignments for edge and variable tracking."""
        if not node.targets:
            self.generic_visit(node)
            return

        target = node.targets[0]

        # Handle tuple unpacking: inputs, labels = batch
        if isinstance(target, ast.Tuple):
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    # Track as coming from an unpack operation
                    self.tracker.assign(
                        elt.id, f"_unpack_{node.lineno}", f"out_{i}", node.lineno
                    )
            self.generic_visit(node)
            return

        # Handle simple assignment: outputs = self.model(inputs)
        if isinstance(target, ast.Name):
            var_name = target.id

            if isinstance(node.value, ast.Call):
                component = self._get_component_from_call(node.value)
                if component:
                    # Extract edges from arguments
                    self._extract_edges_from_call(node.value, component, node.lineno)
                    # Track the output variable
                    self.tracker.assign(var_name, component, "out", node.lineno)

        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Handle expression statements like self.optimizer.step()."""
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute):
                # self.optimizer.step(), loss.backward()
                if func.attr == "backward":
                    # Find what's calling backward
                    if isinstance(func.value, ast.Name):
                        loss_var = func.value.id
                        tracked = self.tracker.resolve(loss_var)
                        if tracked:
                            # Mark gradient flow
                            self.edges.append(
                                ExtractedEdge(
                                    source_node=tracked.node_id,
                                    source_port="gradient",
                                    target_node="_backward",
                                    target_port="in",
                                    flow_type=FlowType.GRADIENT,
                                    variable_name=loss_var,
                                    extracted_from="train_step",
                                    source_line=node.lineno,
                                )
                            )

            component = self._get_component_from_call(node.value)
            if component:
                self._extract_edges_from_call(node.value, component, node.lineno)

        self.generic_visit(node)

    def _get_component_from_call(self, call: ast.Call) -> str | None:
        """Get component name from self.xxx(...) call."""
        func = call.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                if func.attr in self.known_components:
                    return func.attr
            if (
                isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "self"
                and func.value.attr in self.known_components
            ):
                return func.value.attr
            if (
                func.attr == "forward"
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "self"
                and func.value.attr in self.known_components
            ):
                return func.value.attr
        return None

    def _extract_edges_from_call(
        self, call: ast.Call, target_component: str, line: int
    ) -> None:
        """Extract edges from call arguments."""
        for i, arg in enumerate(call.args):
            self._extract_edge_from_argument(arg, target_component, f"in_{i}", line)

        for keyword in call.keywords:
            if keyword.arg:
                self._extract_edge_from_argument(
                    keyword.value,
                    target_component,
                    keyword.arg,
                    line,
                )

    def _extract_edge_from_argument(
        self,
        arg: ast.expr,
        target_component: str,
        target_port: str,
        line: int,
    ) -> None:
        if isinstance(arg, ast.Name):
            tracked = self.tracker.resolve(arg.id)
            if tracked and tracked.node_id in self.known_components:
                self.edges.append(
                    ExtractedEdge(
                        source_node=tracked.node_id,
                        source_port=tracked.port,
                        target_node=target_component,
                        target_port=target_port,
                        flow_type=FlowType.TENSOR,
                        variable_name=arg.id,
                        extracted_from="train_step",
                        source_line=line,
                    )
                )


class MethodExtractor(ast.NodeVisitor):
    """Extracts Agent methods as nodes.

    Looks for method definitions in the Agent class:
    - setup() - initialization method
    - train_step() - training step method
    - val_step() - validation step method
    - on_train_start(), on_train_end(), etc. - lifecycle methods
    """

    # Method name to category mapping
    METHOD_CATEGORIES: dict[str, MethodCategory] = {
        "setup": MethodCategory.SETUP,
        "train_step": MethodCategory.TRAIN,
        "training_step": MethodCategory.TRAIN,
        "val_step": MethodCategory.VALIDATION,
        "validate_step": MethodCategory.VALIDATION,
        "validation_step": MethodCategory.VALIDATION,
        "test_step": MethodCategory.VALIDATION,
        "on_train_start": MethodCategory.LIFECYCLE,
        "on_train_end": MethodCategory.LIFECYCLE,
        "on_epoch_start": MethodCategory.LIFECYCLE,
        "on_epoch_end": MethodCategory.LIFECYCLE,
        "on_batch_start": MethodCategory.LIFECYCLE,
        "on_batch_end": MethodCategory.LIFECYCLE,
        "on_validation_start": MethodCategory.LIFECYCLE,
        "on_validation_end": MethodCategory.LIFECYCLE,
    }

    def __init__(self, source_file: str) -> None:
        self.source_file = source_file
        self.nodes: list[HierarchyNode] = []
        self.method_calls: dict[str, list[str]] = {}  # method_name -> [called_methods]
        self.component_accesses: dict[
            str, set[str]
        ] = {}  # method_name -> {component_names}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract method definition as a node."""
        method_name = node.name

        # Skip private methods (starting with _) except lifecycle hooks
        if method_name.startswith("_") and not method_name.startswith("__"):
            self.generic_visit(node)
            return

        # Determine category
        category = self.METHOD_CATEGORIES.get(method_name, MethodCategory.UTILITY)

        # Extract parameters as inputs
        inputs: list[Port] = []
        for arg in node.args.args:
            if arg.arg != "self":  # Skip self
                inputs.append(Port(name=arg.arg, type="any"))

        # Determine outputs (basic heuristic: check for return statements)
        outputs: list[Port] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                outputs.append(Port(name="return", type="any"))
                break

        # Extract method calls and component accesses within this method
        call_extractor = MethodCallExtractor()
        call_extractor.visit(node)
        self.method_calls[method_name] = call_extractor.called_methods
        self.component_accesses[method_name] = call_extractor.component_accesses

        # Create node
        self.nodes.append(
            HierarchyNode(
                id=method_name,
                label=method_name,
                level=HierarchyLevel.METHOD,
                origin=NodeOrigin(
                    type=OriginType.METHOD_CALL,
                    code_snippet=f"def {method_name}(...)",
                    source=SourceLocation(
                        file=self.source_file,
                        line=node.lineno,
                        end_line=node.end_lineno,
                    ),
                ),
                can_drill=True,  # Can drill into method to see component flow
                inputs=inputs,
                outputs=outputs,
                source=SourceLocation(
                    file=self.source_file,
                    line=node.lineno,
                    end_line=node.end_lineno,
                ),
                metadata={
                    "method_category": category.value,
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                },
            )
        )

        self.generic_visit(node)


class MethodCallExtractor(ast.NodeVisitor):
    """Extracts self.method_name() calls and component accesses within a method."""

    def __init__(self) -> None:
        self.called_methods: list[str] = []
        self.component_accesses: set[str] = set()  # self.xxx accesses

    def visit_Call(self, node: ast.Call) -> None:
        """Extract self.xxx() calls."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                self.called_methods.append(node.func.attr)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Extract self.xxx attribute accesses (component usage)."""
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            attr = node.attr
            # Skip private attributes and special methods
            if not attr.startswith("_") and attr not in ("create", "cfg", "device"):
                self.component_accesses.add(attr)
        self.generic_visit(node)


@dataclass
class ComponentUsage:
    """Represents a component usage within a method."""

    name: str  # Component name (e.g., 'model', 'optimizer')
    line: int  # Line number
    call_code: str  # Code snippet
    input_vars: list[str]  # Input variable names
    output_var: str | None  # Output variable name if assigned


class MethodInternalExtractor(ast.NodeVisitor):
    """Extracts internal component flow within a specific method.

    Used for drill-down into a method to show:
    - Which components are used (self.model, self.loss_fn, etc.)
    - Data flow between component calls
    - Input parameters and return values
    """

    def __init__(self, source_file: str, known_components: dict[str, str]) -> None:
        self.source_file = source_file
        self.known_components = known_components
        self.component_usages: list[ComponentUsage] = []
        self.tracker = VariableTracker()
        self.edges: list[ExtractedEdge] = []

    def extract_from_method(self, method_node: ast.FunctionDef) -> None:
        """Extract component flow from a method.

        Creates:
        - Input node for method parameters
        - Nodes for each component call
        - Output node for return value
        - Edges showing data flow
        """
        # Track method parameters
        for arg in method_node.args.args:
            if arg.arg != "self":
                self.tracker.assign(arg.arg, "_input", arg.arg, method_node.lineno)

        # Visit the method body
        self.visit(method_node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Process assignments for component usage tracking."""
        if not node.targets:
            self.generic_visit(node)
            return

        target = node.targets[0]

        # Handle tuple unpacking: inputs, labels = batch
        if isinstance(target, ast.Tuple):
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    self.tracker.assign(elt.id, "_unpack", f"out_{i}", node.lineno)
            self.generic_visit(node)
            return

        # Handle simple assignment: outputs = self.model(inputs)
        if isinstance(target, ast.Name):
            var_name = target.id

            if isinstance(node.value, ast.Call):
                component = self._get_component_from_call(node.value)
                if component:
                    # Extract input variables
                    input_vars = self._get_call_args(node.value)

                    # Record component usage
                    self.component_usages.append(
                        ComponentUsage(
                            name=component,
                            line=node.lineno,
                            call_code=ast.unparse(node.value),
                            input_vars=input_vars,
                            output_var=var_name,
                        )
                    )

                    # Extract edges from input variables
                    for i, arg_name in enumerate(input_vars):
                        tracked = self.tracker.resolve(arg_name)
                        if tracked:
                            self.edges.append(
                                ExtractedEdge(
                                    source_node=tracked.node_id,
                                    source_port=tracked.port,
                                    target_node=component,
                                    target_port=f"in_{i}",
                                    flow_type=FlowType.TENSOR,
                                    variable_name=arg_name,
                                    extracted_from="method_internal",
                                    source_line=node.lineno,
                                )
                            )

                    # Track the output variable
                    self.tracker.assign(var_name, component, "out", node.lineno)

        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Handle expression statements like loss.backward(), optimizer.step()."""
        if isinstance(node.value, ast.Call):
            func = node.value.func

            # Handle loss.backward()
            if isinstance(func, ast.Attribute) and func.attr == "backward":
                if isinstance(func.value, ast.Name):
                    loss_var = func.value.id
                    tracked = self.tracker.resolve(loss_var)
                    if tracked and tracked.node_id in self.known_components:
                        self.component_usages.append(
                            ComponentUsage(
                                name=f"{tracked.node_id}.backward",
                                line=node.lineno,
                                call_code=f"{loss_var}.backward()",
                                input_vars=[loss_var],
                                output_var=None,
                            )
                        )

            # Handle self.optimizer.step(), self.optimizer.zero_grad()
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Attribute):
                    if (
                        isinstance(func.value.value, ast.Name)
                        and func.value.value.id == "self"
                    ):
                        component = func.value.attr
                        if component in self.known_components:
                            self.component_usages.append(
                                ComponentUsage(
                                    name=f"{component}.{func.attr}",
                                    line=node.lineno,
                                    call_code=f"self.{component}.{func.attr}()",
                                    input_vars=[],
                                    output_var=None,
                                )
                            )

        self.generic_visit(node)

    def _get_component_from_call(self, call: ast.Call) -> str | None:
        """Get component name from self.xxx(...) call."""
        func = call.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                if func.attr in self.known_components:
                    return func.attr
        return None

    def _get_call_args(self, call: ast.Call) -> list[str]:
        """Get argument variable names from a call."""
        args = []
        for arg in call.args:
            if isinstance(arg, ast.Name):
                args.append(arg.id)
            else:
                args.append(ast.unparse(arg))
        return args


@dataclass
class MethodInternalResult:
    """Result of method internal analysis for drill-down."""

    nodes: list[HierarchyNode] = field(default_factory=list)
    edges: list[CodeFlowEdge] = field(default_factory=list)
    component_usages: list[ComponentUsage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class MethodAnalysisResult:
    """Result of method-level analysis."""

    nodes: list[HierarchyNode] = field(default_factory=list)
    edges: list[CodeFlowEdge] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of code analysis."""

    nodes: list[HierarchyNode] = field(default_factory=list)
    edges: list[CodeFlowEdge] = field(default_factory=list)
    known_components: dict[str, str] = field(default_factory=dict)
    method_nodes: list[HierarchyNode] = field(default_factory=list)
    method_edges: list[CodeFlowEdge] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Code coverage info (for 1:1 code-node mapping visualization)
    setup_start_line: int | None = None
    setup_end_line: int | None = None
    covered_lines: list[int] = field(default_factory=list)
    uncovered_lines: list[int] = field(default_factory=list)


class AgentCodeAnalyzer:
    """Main analyzer for Agent source code.

    Coordinates component extraction and edge extraction from both
    setup() and train_step() methods.
    """

    def __init__(self, source_code: str, source_file: str) -> None:
        self.source_code = source_code
        self.source_file = source_file
        self.tree: ast.AST | None = None

    def parse(self) -> bool:
        """Parse the source code."""
        try:
            self.tree = ast.parse(self.source_code)
            return True
        except SyntaxError as e:
            return False

    def analyze(self) -> AnalysisResult:
        """Perform full analysis and return result."""
        if not self.tree:
            if not self.parse():
                return AnalysisResult(errors=["Failed to parse source code"])

        result = AnalysisResult()

        # Find the Agent class
        agent_class = self._find_agent_class()
        if not agent_class:
            result.errors.append("No Agent class found")
            return result

        # Extract setup() components
        setup_method = self._find_method(agent_class, "setup")
        if setup_method:
            component_extractor = ComponentExtractor(self.source_file)
            component_extractor.visit(setup_method)
            # Include both covered nodes and uncovered nodes (for red highlighting)
            result.nodes = (
                component_extractor.nodes + component_extractor.uncovered_nodes
            )
            # Merge both self_components and local_components for edge extraction
            result.known_components = {
                **component_extractor.self_components,
                **component_extractor.local_components,
            }

            # Extract setup() edges
            setup_edge_extractor = SetupEdgeExtractor(
                self.source_file, result.known_components
            )
            setup_edge_extractor.visit(setup_method)
            result.edges.extend(self._convert_edges(setup_edge_extractor.edges))

            # Compute code coverage for 1:1 code-node mapping
            result.setup_start_line = setup_method.lineno
            result.setup_end_line = setup_method.end_lineno or setup_method.lineno

            # Get all statement lines in setup() body
            all_statement_lines = self._extract_statement_lines(setup_method)

            # Get covered lines from nodes
            result.covered_lines = sorted(
                set(
                    node.source.line
                    for node in result.nodes
                    if node.source and node.source.line
                )
            )

            # Uncovered = statement lines that are not covered by any node
            result.uncovered_lines = sorted(
                set(all_statement_lines) - set(result.covered_lines)
            )

            if result.uncovered_lines:
                coverage_nodes = self._build_coverage_nodes(result.uncovered_lines)
                result.nodes.extend(coverage_nodes)
                result.covered_lines = sorted(set(all_statement_lines))
                result.uncovered_lines = []
        else:
            result.errors.append("No setup() method found")

        # Extract train_step() edges
        train_step_method = self._find_method(agent_class, "train_step")
        if train_step_method and result.known_components:
            train_step_extractor = TrainStepEdgeExtractor(
                self.source_file, result.known_components
            )
            train_step_extractor.visit(train_step_method)
            result.edges.extend(self._convert_edges(train_step_extractor.edges))

        # Generate semantic edges based on port type compatibility
        result.edges.extend(self._generate_semantic_edges(result.nodes, result.edges))

        # Extract method-level nodes and edges
        method_extractor = MethodExtractor(self.source_file)
        method_extractor.visit(agent_class)
        result.method_nodes = method_extractor.nodes
        result.method_edges = self._generate_method_edges(
            method_extractor.nodes,
            method_extractor.method_calls,
            method_extractor.component_accesses,
        )

        return result

    def analyze_method_internal(self, method_name: str) -> MethodInternalResult:
        """Analyze the internal flow of a specific method for drill-down.

        Args:
            method_name: Name of the method to analyze (e.g., 'train_step')

        Returns:
            MethodInternalResult with nodes and edges for the method's internal flow
        """
        result = MethodInternalResult()

        if not self.tree:
            if not self.parse():
                result.errors.append("Failed to parse source code")
                return result

        # Find the Agent class
        agent_class = self._find_agent_class()
        if not agent_class:
            result.errors.append("No Agent class found")
            return result

        # Find the target method
        method_node = self._find_method(agent_class, method_name)
        if not method_node:
            result.errors.append(f"Method '{method_name}' not found")
            return result

        # First, get known components from setup()
        setup_method = self._find_method(agent_class, "setup")
        known_components: dict[str, str] = {}
        if setup_method:
            component_extractor = ComponentExtractor(self.source_file)
            component_extractor.visit(setup_method)
            known_components = {
                **component_extractor.self_components,
                **component_extractor.local_components,
            }

        # Extract internal flow from the method
        internal_extractor = MethodInternalExtractor(self.source_file, known_components)
        internal_extractor.extract_from_method(method_node)
        result.component_usages = internal_extractor.component_usages

        # Create Input node for method parameters
        params = [arg.arg for arg in method_node.args.args if arg.arg != "self"]
        if params:
            result.nodes.append(
                HierarchyNode(
                    id="_input",
                    label="Input",
                    level=HierarchyLevel.OPERATION,
                    origin=NodeOrigin(
                        type=OriginType.INPUT,
                        code_snippet=f"def {method_name}(self, {', '.join(params)})",
                        source=SourceLocation(
                            file=self.source_file,
                            line=method_node.lineno,
                        ),
                    ),
                    can_drill=False,
                    outputs=[Port(name=p, type="any") for p in params],
                    source=SourceLocation(
                        file=self.source_file,
                        line=method_node.lineno,
                    ),
                    metadata={"node_type": "input", "params": params},
                )
            )

        # Create nodes for each component usage
        seen_components: set[str] = set()
        for usage in result.component_usages:
            # Skip duplicate component nodes (but keep method calls like optimizer.step)
            base_component = usage.name.split(".")[0]
            if base_component in seen_components and "." not in usage.name:
                continue
            seen_components.add(base_component)

            # Determine category from known_components
            category = None
            if base_component in known_components:
                cat_str = known_components[base_component]
                try:
                    category = ComponentCategory(cat_str)
                except ValueError:
                    category = ComponentCategory.UNKNOWN

            result.nodes.append(
                HierarchyNode(
                    id=usage.name,
                    label=usage.name,
                    level=HierarchyLevel.OPERATION,
                    origin=NodeOrigin(
                        type=OriginType.METHOD_CALL,
                        code_snippet=usage.call_code,
                        source=SourceLocation(
                            file=self.source_file,
                            line=usage.line,
                        ),
                    ),
                    can_drill=False,
                    category=category,
                    inputs=[Port(name=v, type="any") for v in usage.input_vars],
                    outputs=[Port(name="out", type="any")] if usage.output_var else [],
                    source=SourceLocation(
                        file=self.source_file,
                        line=usage.line,
                    ),
                    metadata={
                        "node_type": "component_call",
                        "output_var": usage.output_var,
                    },
                )
            )

        # Convert edges
        result.edges = self._convert_edges(internal_extractor.edges)

        return result

    def _generate_method_edges(
        self,
        method_nodes: list[HierarchyNode],
        method_calls: dict[str, list[str]],
        component_accesses: dict[str, set[str]] | None = None,
    ) -> list[CodeFlowEdge]:
        """Generate edges between methods based on call relationships.

        Creates control flow edges showing method invocation order:
        - setup() is called first (lifecycle)
        - train_step() is called in training loop
        - val_step() is called in validation loop

        Args:
            method_nodes: List of method nodes
            method_calls: Dict mapping method names to called methods
            component_accesses: Dict mapping method names to component accesses (self.xxx)
        """
        edges: list[CodeFlowEdge] = []
        method_ids = {n.id for n in method_nodes}
        component_accesses = component_accesses or {}

        # Get components created/used in setup (these flow to other methods)
        setup_components = component_accesses.get("setup", set())

        # Create edges for method calls
        for caller, callees in method_calls.items():
            if caller not in method_ids:
                continue
            for callee in callees:
                # Only create edge if callee is a known method
                if callee in method_ids:
                    edges.append(
                        CodeFlowEdge(
                            id=f"method_call_{caller}_{callee}",
                            source_node=caller,
                            source_port="out",
                            target_node=callee,
                            target_port="in",
                            flow_type=FlowType.CONTROL,
                            extracted_from="method_call",
                            source=SourceLocation(file=self.source_file, line=0),
                        )
                    )

        # Add standard lifecycle edges with data flow labels
        # setup -> train_step (setup initializes, then training begins)
        if "setup" in method_ids and "train_step" in method_ids:
            # Find shared components between setup and train_step
            train_components = component_accesses.get("train_step", set())
            shared = setup_components & train_components
            variable_name = ", ".join(sorted(shared)) if shared else None

            edges.append(
                CodeFlowEdge(
                    id="lifecycle_setup_train",
                    source_node="setup",
                    source_port="out",
                    target_node="train_step",
                    target_port="in",
                    flow_type=FlowType.CONTROL,
                    variable_name=variable_name,
                    extracted_from="lifecycle",
                    source=SourceLocation(file=self.source_file, line=0),
                )
            )

        # train_step -> val_step (training step, then validation)
        if "train_step" in method_ids and "val_step" in method_ids:
            # Find shared components between train_step and val_step
            train_components = component_accesses.get("train_step", set())
            val_components = component_accesses.get("val_step", set())
            # For train->val, show what val_step uses from setup
            shared = setup_components & val_components
            variable_name = ", ".join(sorted(shared)) if shared else None

            edges.append(
                CodeFlowEdge(
                    id="lifecycle_train_val",
                    source_node="train_step",
                    source_port="out",
                    target_node="val_step",
                    target_port="in",
                    flow_type=FlowType.CONTROL,
                    variable_name=variable_name,
                    extracted_from="lifecycle",
                    source=SourceLocation(file=self.source_file, line=0),
                )
            )

        # setup -> val_step (if val_step exists but no train_step)
        if (
            "setup" in method_ids
            and "val_step" in method_ids
            and "train_step" not in method_ids
        ):
            val_components = component_accesses.get("val_step", set())
            shared = setup_components & val_components
            variable_name = ", ".join(sorted(shared)) if shared else None

            edges.append(
                CodeFlowEdge(
                    id="lifecycle_setup_val",
                    source_node="setup",
                    source_port="out",
                    target_node="val_step",
                    target_port="in",
                    flow_type=FlowType.CONTROL,
                    variable_name=variable_name,
                    extracted_from="lifecycle",
                    source=SourceLocation(file=self.source_file, line=0),
                )
            )

        return edges

    def _generate_semantic_edges(
        self, nodes: list[HierarchyNode], existing_edges: list[CodeFlowEdge]
    ) -> list[CodeFlowEdge]:
        """Generate semantic edges based on port type compatibility.

        Connects components that have matching port types but no explicit
        code-based connection. For example:
        - dataset (output: data/dataset) → dataloader (input: dataset)
        - sampler (output: indices) → dataloader (input: indices) [future]

        Uses naming patterns to match: train_dataset → train_loader
        """
        semantic_edges: list[CodeFlowEdge] = []

        # Build lookup maps
        node_by_id = {n.id: n for n in nodes}
        existing_connections = {(e.source_node, e.target_node) for e in existing_edges}

        # Find nodes with dataset/data outputs (typically DATASET category)
        dataset_nodes = [n for n in nodes if n.category == ComponentCategory.DATASET]

        # Find nodes that need dataset input (typically DATALOADER category)
        dataloader_nodes = [
            n for n in nodes if n.category == ComponentCategory.DATALOADER
        ]

        # Connect datasets to dataloaders if no explicit connection exists
        # Match by naming pattern: train_dataset → train_loader, val_dataset → val_loader
        for dataset_node in dataset_nodes:
            for dataloader_node in dataloader_nodes:
                if (dataset_node.id, dataloader_node.id) not in existing_connections:
                    # Check if dataloader has dataset input port
                    has_dataset_input = any(
                        p.type == "dataset" for p in dataloader_node.inputs
                    )
                    # Check if dataset has data output port
                    has_data_output = any(
                        p.type in ("dataset", "data") for p in dataset_node.outputs
                    )

                    # Check naming pattern match
                    # e.g., train_dataset matches train_loader, val_dataset matches val_loader
                    dataset_prefix = self._extract_name_prefix(
                        dataset_node.id, "dataset"
                    )
                    loader_prefix = self._extract_name_prefix(
                        dataloader_node.id, "loader"
                    )
                    names_match = (
                        dataset_prefix == loader_prefix
                        or not dataset_prefix
                        or not loader_prefix
                    )

                    if has_dataset_input and has_data_output and names_match:
                        semantic_edges.append(
                            CodeFlowEdge(
                                id=f"semantic_{dataset_node.id}_{dataloader_node.id}",
                                source_node=dataset_node.id,
                                source_port="data",
                                target_node=dataloader_node.id,
                                target_port="dataset",
                                flow_type=FlowType.CONFIG,  # Semantic/config-based
                                extracted_from="semantic_inference",
                                source=SourceLocation(file=self.source_file, line=0),
                            )
                        )

        return semantic_edges

    def _extract_name_prefix(self, name: str, suffix: str) -> str:
        """Extract prefix from name by removing common suffixes.

        Examples:
            - train_dataset -> train
            - val_loader -> val
            - my_train_data -> my_train (if suffix is 'data')
        """
        # Common patterns to strip
        patterns = [f"_{suffix}", suffix, "_data", "data"]
        result = name.lower()
        for pattern in patterns:
            if result.endswith(pattern):
                result = result[: -len(pattern)]
                break
        return result.strip("_")

    def _find_agent_class(self) -> ast.ClassDef | None:
        """Find the Agent class definition.

        Supports:
        - Direct inheritance: class Foo(Agent)
        - Qualified inheritance: class Foo(module.Agent)
        - Indirect inheritance: class Foo(BasicAgent), class Foo(SomeOtherAgent)
          (any class ending with 'Agent' is treated as a valid base class)
        """
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    # Direct: class Foo(Agent) or class Foo(BasicAgent)
                    if isinstance(base, ast.Name):
                        if base.id == "Agent" or base.id.endswith("Agent"):
                            return node
                    # Qualified: class Foo(module.Agent) or class Foo(module.BasicAgent)
                    if isinstance(base, ast.Attribute):
                        if base.attr == "Agent" or base.attr.endswith("Agent"):
                            return node
        return None

    def _find_method(
        self, class_node: ast.ClassDef, method_name: str
    ) -> ast.FunctionDef | None:
        """Find a method in a class."""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return node
        return None

    def _build_coverage_nodes(self, lines: list[int]) -> list[HierarchyNode]:
        nodes: list[HierarchyNode] = []
        source_lines = self.source_code.splitlines()

        for line in lines:
            code_snippet = ""
            if 0 < line <= len(source_lines):
                code_snippet = source_lines[line - 1].strip()
            node_id = f"coverage_line_{line}"
            nodes.append(
                HierarchyNode(
                    id=node_id,
                    label=f"line_{line}",
                    level=HierarchyLevel.COMPONENT,
                    origin=NodeOrigin(
                        type=OriginType.ASSIGNMENT,
                        code_snippet=code_snippet,
                        source=SourceLocation(file=self.source_file, line=line),
                    ),
                    can_drill=False,
                    category=ComponentCategory.UNKNOWN,
                    inputs=[],
                    outputs=[],
                    source=SourceLocation(file=self.source_file, line=line),
                    metadata={
                        "coverage_only": True,
                        "coverage_reason": "Unmapped setup() statement",
                    },
                )
            )

        return nodes

    def _extract_statement_lines(self, method_node: ast.FunctionDef) -> list[int]:
        """Extract line numbers of all statements in a method body.

        Returns lines that contain actual statements (not just docstrings,
        comments, or empty lines).
        """
        statement_lines: list[int] = []

        for stmt in ast.walk(method_node):
            # Only count actual statements, not expressions within statements
            if isinstance(
                stmt,
                (
                    ast.Assign,  # x = value
                    ast.AugAssign,  # x += value
                    ast.AnnAssign,  # x: type = value
                    ast.Expr,  # expression statement (including print(), function calls)
                    ast.Return,  # return statement
                    ast.If,  # if statement (just the if line)
                    ast.For,  # for statement (just the for line)
                    ast.While,  # while statement
                    ast.With,  # with statement
                    ast.Try,  # try statement
                    ast.Assert,  # assert statement
                    ast.Import,  # import statement
                    ast.ImportFrom,  # from x import y
                    ast.Pass,  # pass statement
                    ast.Break,  # break
                    ast.Continue,  # continue
                    ast.Raise,  # raise statement
                ),
            ):
                # Skip the method definition line itself
                if stmt is method_node:
                    continue
                statement_lines.append(stmt.lineno)

        return sorted(set(statement_lines))

    def _convert_edges(
        self, extracted_edges: list[ExtractedEdge]
    ) -> list[CodeFlowEdge]:
        """Convert ExtractedEdge to CodeFlowEdge."""
        result = []
        for i, edge in enumerate(extracted_edges):
            result.append(
                CodeFlowEdge(
                    id=f"edge_{edge.source_node}_{edge.target_node}_{i}",
                    source_node=edge.source_node,
                    source_port=edge.source_port,
                    target_node=edge.target_node,
                    target_port=edge.target_port,
                    flow_type=edge.flow_type,
                    variable_name=edge.variable_name,
                    extracted_from=edge.extracted_from,
                    source=SourceLocation(file=self.source_file, line=edge.source_line),
                )
            )
        return result


def analyze_agent_file(file_path: str | Path) -> AnalysisResult:
    """Convenience function to analyze an agent file."""
    path = Path(file_path)
    if not path.exists():
        return AnalysisResult(errors=[f"File not found: {file_path}"])

    source_code = path.read_text()
    analyzer = AgentCodeAnalyzer(source_code, str(path))
    return analyzer.analyze()
