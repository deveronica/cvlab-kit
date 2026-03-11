"""
Setup Dependency Parser - Extract dependency injection edges from setup() method.

Analyzes self.create.* calls to find:
1. Direct dependencies: self.create.optimizer(self.model.parameters()) → model → optimizer
2. Variable dependencies: train_loader = self.create.dataloader(train_dataset) → train_dataset → train_loader
3. Keyword dependencies: self.create.dataset(transform=self.transform) → transform → dataset

Output edges for visualization:
- model.parameters() → optimizer
- train_dataset → train_loader
- val_dataset → val_loader
"""

import ast
from dataclasses import dataclass
from typing import Optional


@dataclass
class DependencyEdge:
    """Represents a dependency injection edge in setup()."""

    source: str  # Source component (e.g., "model", "train_dataset")
    target: str  # Target component (e.g., "optimizer", "train_loader")
    dependency_type: str  # "parameters", "direct", "kwarg"
    label: Optional[str] = None  # Edge label (e.g., ".parameters()")
    line: int = 0


class SetupDependencyParser(ast.NodeVisitor):
    """Extract dependency edges from setup() method."""

    def __init__(self):
        self.edges: list[DependencyEdge] = []
        self._component_vars: set[str] = set()  # Known component variable names
        self._local_vars: set[str] = set()  # Local variable names

    def parse(self, source: str, component_names: set[str]) -> list[DependencyEdge]:
        """Parse setup() and extract dependency edges.

        Args:
            source: Agent source code
            component_names: Set of known component names from CreateCallParser

        Returns:
            List of DependencyEdge objects
        """
        self.edges = []
        self._component_vars = component_names
        self._local_vars = set()

        tree = ast.parse(source)

        # Find setup method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "setup":
                # First pass: collect local variable names
                self._collect_local_vars(node)
                # Second pass: extract dependencies
                for stmt in node.body:
                    self._analyze_statement(stmt)
                break

        return self.edges

    def _collect_local_vars(self, func_node: ast.FunctionDef):
        """Collect all local variable names assigned in the function."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._local_vars.add(target.id)

    def _analyze_statement(self, stmt: ast.stmt):
        """Analyze a statement for dependency patterns."""
        if not isinstance(stmt, ast.Assign):
            return

        # Get target name
        target_name = self._get_assign_target(stmt)
        if not target_name:
            return

        # Analyze the value for dependencies
        if isinstance(stmt.value, ast.Call):
            deps = self._extract_call_dependencies(stmt.value, stmt.lineno)
            for dep in deps:
                self.edges.append(DependencyEdge(
                    source=dep["source"],
                    target=target_name,
                    dependency_type=dep["type"],
                    label=dep.get("label"),
                    line=stmt.lineno,
                ))

    def _get_assign_target(self, stmt: ast.Assign) -> Optional[str]:
        """Get the target name from an assignment."""
        if len(stmt.targets) != 1:
            return None

        target = stmt.targets[0]

        # self.x = ...
        if isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                return target.attr

        # x = ... (local variable)
        if isinstance(target, ast.Name):
            return target.id

        return None

    def _extract_call_dependencies(self, call: ast.Call, line: int) -> list[dict]:
        """Extract dependencies from a function call."""
        deps = []

        # Check positional arguments
        for arg in call.args:
            dep = self._analyze_arg(arg)
            if dep:
                deps.append(dep)

        # Check keyword arguments
        for kw in call.keywords:
            if kw.arg:
                dep = self._analyze_arg(kw.value)
                if dep:
                    dep["label"] = f"{kw.arg}="
                    deps.append(dep)

        return deps

    def _analyze_arg(self, arg: ast.expr) -> Optional[dict]:
        """Analyze an argument to find dependencies."""

        # Pattern 1: self.model.parameters() → model (with .parameters() label)
        if isinstance(arg, ast.Call):
            if isinstance(arg.func, ast.Attribute):
                # self.model.parameters()
                if isinstance(arg.func.value, ast.Attribute):
                    if isinstance(arg.func.value.value, ast.Name):
                        if arg.func.value.value.id == "self":
                            component = arg.func.value.attr
                            method = arg.func.attr
                            if component in self._component_vars:
                                return {
                                    "source": component,
                                    "type": "method_call",
                                    "label": f".{method}()",
                                }

        # Pattern 2: self.component → component (direct attribute reference)
        if isinstance(arg, ast.Attribute):
            if isinstance(arg.value, ast.Name) and arg.value.id == "self":
                component = arg.attr
                # Ensure it's a known component, not just any attribute
                if component in self._component_vars:
                    return {
                        "source": component,
                        "type": "direct",
                    }

        # Pattern 3: local_var → local_var (local variable reference)
        # Only connect if the local variable was assigned from self.create
        if isinstance(arg, ast.Name):
            var_name = arg.id
            if var_name in self._local_vars:
                return {
                    "source": var_name,
                    "type": "direct",
                }

        return None


def parse_setup_dependencies(source: str, component_names: set[str]) -> list[DependencyEdge]:
    """Convenience function to parse setup dependencies."""
    parser = SetupDependencyParser()
    return parser.parse(source, component_names)
