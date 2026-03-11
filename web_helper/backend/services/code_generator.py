"""
Code Generator - Convert node graph to CVLab-Kit Agent Python code.

This module generates Python code from a visual node graph representation.
It follows CVLab-Kit's Agent-Component-Creator pattern.

Generated code structure:
    from cvlabkit.core.agent import Agent

    class GeneratedAgent(Agent):
        def setup(self):
            self.model = self.create.model()
            self.optimizer = self.create.optimizer(self.model.parameters())
            ...

        def train_step(self, batch):
            images, labels = batch
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            ...
"""

import ast
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    import libcst as cst
    from libcst import metadata

    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False
    logger.warning("LibCST not available. Install with: pip install libcst")

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Models for Code Generation
# =============================================================================


class NodeLevel(str, Enum):
    """Node hierarchy level."""

    SETUP = "setup"  # Component instantiation (setup method)
    FLOW = "flow"  # Execution flow (train_step method)


@dataclass
class GeneratedNode:
    """Node representation for code generation."""

    id: str
    label: str
    category: str
    level: NodeLevel
    # For setup nodes
    variable_name: Optional[str] = None
    create_path: list[str] = field(default_factory=list)  # e.g., ["model"], ["transform", "weak"]
    kwargs: dict = field(default_factory=dict)
    # For flow nodes
    flow_type: Optional[str] = None  # forward, loss_compute, backward, optimizer_step
    method_call: Optional[str] = None  # e.g., "self.model(images)"
    # Common
    inputs: list[str] = field(default_factory=list)  # Input port IDs
    outputs: list[str] = field(default_factory=list)  # Output port IDs
    source_line: Optional[int] = None


@dataclass
class GeneratedEdge:
    """Edge representation for code generation."""

    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    variable_name: Optional[str] = None


class CodeGenerationRequest(BaseModel):
    """Request model for code generation API."""

    agent_name: str = "GeneratedAgent"
    nodes: list[dict]
    edges: list[dict]


class CodeGenerationResponse(BaseModel):
    """Response model for code generation API."""

    success: bool
    code: str = ""
    setup_code: str = ""
    train_step_code: str = ""
    error: Optional[str] = None


# =============================================================================
# Code Generator
# =============================================================================


class CodeGenerator:
    """Generate CVLab-Kit Agent code from node graph."""

    # Category to variable name mapping (for consistent naming)
    CATEGORY_VAR_NAMES = {
        "model": "model",
        "optimizer": "optimizer",
        "loss": "loss_fn",
        "dataset": "dataset",
        "dataloader": "loader",
        "transform": "transform",
        "metric": "metric",
        "scheduler": "scheduler",
        "sampler": "sampler",
    }

    # Categories that need special handling
    OPTIMIZER_CATEGORIES = {"optimizer"}
    SCHEDULER_CATEGORIES = {"scheduler"}

    def __init__(self):
        self._var_counter: dict[str, int] = {}

    def generate(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate complete agent code from node graph."""
        try:
            # Parse nodes and edges
            nodes = self._parse_nodes(request.nodes)
            edges = self._parse_edges(request.edges)

            # Separate setup and flow nodes
            setup_nodes = [n for n in nodes if n.level == NodeLevel.SETUP]
            flow_nodes = [n for n in nodes if n.level == NodeLevel.FLOW]

            # Generate code sections
            setup_code = self._generate_setup_method(setup_nodes, edges)
            train_step_code = self._generate_train_step_method(flow_nodes, edges)

            # Combine into full agent code
            full_code = self._generate_full_agent(
                agent_name=request.agent_name,
                setup_code=setup_code,
                train_step_code=train_step_code,
            )

            return CodeGenerationResponse(
                success=True,
                code=full_code,
                setup_code=setup_code,
                train_step_code=train_step_code,
            )

        except Exception as e:
            logger.error(f"Code generation failed: {e}", exc_info=True)
            return CodeGenerationResponse(
                success=False,
                error=str(e),
            )

    def _parse_nodes(self, raw_nodes: list[dict]) -> list[GeneratedNode]:
        """Parse raw node data into GeneratedNode objects."""
        nodes = []
        for raw in raw_nodes:
            data = raw.get("data", {})
            metadata = data.get("metadata", {})

            # Determine level
            is_setup = metadata.get("isSetupNode", False)
            level = NodeLevel.SETUP if is_setup else NodeLevel.FLOW

            # Variable name
            var_name = self._get_variable_name(data.get("category"), data.get("label"))

            # Create path from category and variant
            category = data.get("category", "unknown")
            variant = metadata.get("variant")
            create_path = [category]
            if variant:
                create_path.append(variant)

            node = GeneratedNode(
                id=raw.get("id", ""),
                label=data.get("label", ""),
                category=category,
                level=level,
                variable_name=var_name,
                create_path=create_path,
                kwargs=metadata.get("kwargs", {}),
                flow_type=metadata.get("flow_type"),
                method_call=metadata.get("methodCall"),
                inputs=[p.get("id") for p in data.get("inputs", [])],
                outputs=[p.get("id") for p in data.get("outputs", [])],
            )
            nodes.append(node)

        return nodes

    def _parse_edges(self, raw_edges: list[dict]) -> list[GeneratedEdge]:
        """Parse raw edge data into GeneratedEdge objects."""
        edges = []
        for raw in raw_edges:
            edge = GeneratedEdge(
                id=raw.get("id", ""),
                source_node=raw.get("source", ""),
                source_port=raw.get("sourceHandle", ""),
                target_node=raw.get("target", ""),
                target_port=raw.get("targetHandle", ""),
                variable_name=raw.get("label"),
            )
            edges.append(edge)
        return edges

    def _get_variable_name(self, category: str | None, label: str | None) -> str:
        """Generate unique variable name for a node."""
        if not category:
            category = "unknown"

        base_name = self.CATEGORY_VAR_NAMES.get(category, category)

        # Check for variant in label (e.g., "weak" transform)
        if label and "_" in label:
            parts = label.split("_")
            if len(parts) > 1:
                base_name = f"{parts[0]}_{base_name}"

        # Ensure uniqueness
        if base_name not in self._var_counter:
            self._var_counter[base_name] = 0
            return base_name

        self._var_counter[base_name] += 1
        return f"{base_name}_{self._var_counter[base_name]}"

    def _generate_setup_method(
        self, nodes: list[GeneratedNode], edges: list[GeneratedEdge]
    ) -> str:
        """Generate setup() method code."""
        if not nodes:
            return "        pass"

        lines = []

        # Sort nodes topologically based on dependencies
        sorted_nodes = self._topological_sort(nodes, edges)

        # Build node ID to variable name mapping
        node_vars: dict[str, str] = {n.id: n.variable_name or n.id for n in sorted_nodes}

        for node in sorted_nodes:
            var_name = node.variable_name or node.id
            create_path = ".".join(node.create_path) if node.create_path else node.category

            # Build kwargs string
            kwargs_parts = []

            # Special handling for optimizer (needs model.parameters())
            if node.category in self.OPTIMIZER_CATEGORIES:
                # Find connected model
                model_var = self._find_connected_model(node.id, edges, node_vars, sorted_nodes)
                if model_var:
                    kwargs_parts.append(f"{model_var}.parameters()")

            # Special handling for scheduler (needs optimizer)
            if node.category in self.SCHEDULER_CATEGORIES:
                # Find connected optimizer
                opt_var = self._find_connected_optimizer(node.id, edges, node_vars, sorted_nodes)
                if opt_var:
                    kwargs_parts.append(opt_var)

            # Add explicit kwargs
            for key, value in node.kwargs.items():
                kwargs_parts.append(f"{key}={value}")

            kwargs_str = ", ".join(kwargs_parts)

            # Generate line
            line = f"        self.{var_name} = self.create.{create_path}({kwargs_str})"
            lines.append(line)

        return "\n".join(lines)

    def _generate_train_step_method(
        self, nodes: list[GeneratedNode], edges: list[GeneratedEdge]
    ) -> str:
        """Generate train_step() method code."""
        if not nodes:
            return "        pass"

        lines = []

        # Sort by flow type order
        flow_order = {
            "forward": 0,
            "loss_compute": 1,
            "backward": 2,
            "optimizer_zero": 3,
            "optimizer_step": 4,
        }

        sorted_nodes = sorted(
            nodes, key=lambda n: flow_order.get(n.flow_type or "", 99)
        )

        # Generate batch unpacking
        lines.append("        images, labels = batch")
        lines.append("")

        for node in sorted_nodes:
            if node.method_call:
                lines.append(f"        {node.method_call}")
            elif node.flow_type == "forward":
                lines.append(f"        outputs = self.model(images)")
            elif node.flow_type == "loss_compute":
                lines.append(f"        loss = self.loss_fn(outputs, labels)")
            elif node.flow_type == "backward":
                lines.append(f"        loss.backward()")
            elif node.flow_type == "optimizer_zero":
                lines.append(f"        self.optimizer.zero_grad()")
            elif node.flow_type == "optimizer_step":
                lines.append(f"        self.optimizer.step()")

        # Add return statement
        lines.append("")
        lines.append("        return {'loss': loss.item()}")

        return "\n".join(lines)

    def _generate_full_agent(
        self, agent_name: str, setup_code: str, train_step_code: str
    ) -> str:
        """Generate complete agent class code."""
        template = f'''"""
Auto-generated CVLab-Kit Agent.

Generated by CVLab-Kit Builder.
"""

from cvlabkit.core.agent import Agent


class {agent_name}(Agent):
    """Generated Agent class."""

    def setup(self):
        """Initialize components."""
{setup_code}

    def train_step(self, batch):
        """Single training step."""
{train_step_code}
'''
        return template

    def _topological_sort(
        self, nodes: list[GeneratedNode], edges: list[GeneratedEdge]
    ) -> list[GeneratedNode]:
        """Sort nodes topologically based on edge dependencies."""
        # Build adjacency list
        node_ids = {n.id for n in nodes}
        in_degree: dict[str, int] = {n.id: 0 for n in nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in nodes}

        for edge in edges:
            if edge.source_node in node_ids and edge.target_node in node_ids:
                adj[edge.source_node].append(edge.target_node)
                in_degree[edge.target_node] += 1

        # Kahn's algorithm
        queue = [n.id for n in nodes if in_degree[n.id] == 0]
        result: list[str] = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Map back to nodes
        node_map = {n.id: n for n in nodes}
        sorted_nodes = [node_map[nid] for nid in result if nid in node_map]

        # Add any remaining nodes (disconnected)
        remaining = [n for n in nodes if n.id not in result]
        sorted_nodes.extend(remaining)

        return sorted_nodes

    def _find_connected_model(
        self,
        node_id: str,
        edges: list[GeneratedEdge],
        node_vars: dict[str, str],
        nodes: list[GeneratedNode],
    ) -> str | None:
        """Find model variable connected to this optimizer."""
        for edge in edges:
            if edge.target_node == node_id:
                source_node = next((n for n in nodes if n.id == edge.source_node), None)
                if source_node and source_node.category == "model":
                    var_name = node_vars.get(source_node.id)
                    return f"self.{var_name}" if var_name else "self.model"
        # Default to "model" if no explicit connection
        return "self.model"

    def _find_connected_optimizer(
        self,
        node_id: str,
        edges: list[GeneratedEdge],
        node_vars: dict[str, str],
        nodes: list[GeneratedNode],
    ) -> str | None:
        """Find optimizer variable connected to this scheduler."""
        for edge in edges:
            if edge.target_node == node_id:
                source_node = next((n for n in nodes if n.id == edge.source_node), None)
                if source_node and source_node.category == "optimizer":
                    return f"self.{node_vars.get(source_node.id)}"
        # Default to "optimizer" if no explicit connection
        return "self.optimizer"

    # =========================================================================
    # LibCST-based Code Generation (Bidirectional Sync)
    # =========================================================================

    def generate_from_nodes(
        self, node_graph: dict, original_code: Optional[str] = None
    ) -> str:
        """
        Generate agent code from node graph, preserving user edits.

        Uses LibCST to preserve formatting and comments in original code.
        Only modifies sections that correspond to nodes in the graph.

        Args:
            node_graph: Node graph with nodes and edges
            original_code: Original Python code to preserve (optional)

        Returns:
            Generated Python code
        """
        if not LIBCST_AVAILABLE:
            logger.warning("LibCST not available, falling back to template generation")
            # Fallback to existing template-based generation
            request = CodeGenerationRequest(
                nodes=node_graph.get("nodes", []),
                edges=node_graph.get("edges", []),
            )
            result = self.generate(request)
            return result.code

        # Parse nodes and edges
        nodes = self._parse_nodes(node_graph.get("nodes", []))
        edges = self._parse_edges(node_graph.get("edges", []))

        # Separate setup and flow nodes
        setup_nodes = [n for n in nodes if n.level == NodeLevel.SETUP]
        flow_nodes = [n for n in nodes if n.level == NodeLevel.FLOW]

        # Generate setup and train_step code
        setup_code = self._generate_setup_method(setup_nodes, edges)
        train_step_code = self._generate_train_step_method(flow_nodes, edges)

        if original_code:
            # Merge with original code using LibCST
            return self.merge_with_original(
                original_code, setup_code, train_step_code
            )
        else:
            # Generate fresh code
            return self._generate_full_agent(
                agent_name="GeneratedAgent",
                setup_code=setup_code,
                train_step_code=train_step_code,
            )

    def merge_with_original(
        self, original_code: str, setup_code: str, train_step_code: str
    ) -> str:
        """
        Merge generated code with original, preserving user edits.

        Uses LibCST to replace only setup() and train_step() methods
        while preserving everything else (imports, comments, other methods).

        Args:
            original_code: Original Python source
            setup_code: Generated setup() method body
            train_step_code: Generated train_step() method body

        Returns:
            Merged Python code
        """
        if not LIBCST_AVAILABLE:
            logger.error("LibCST required for code merging")
            return original_code

        try:
            # Parse original code
            tree = cst.parse_module(original_code)

            # Transform tree to replace generated methods
            transformer = AgentMethodTransformer(setup_code, train_step_code)
            new_tree = tree.visit(transformer)

            return new_tree.code
        except Exception as e:
            logger.error(f"Failed to merge code: {e}", exc_info=True)
            return original_code

    def merge_user_code(
        self,
        generated_code: str,
        user_code: str,
        preserve_ranges: list[tuple[int, int]],
    ) -> str:
        """
        Merge generated code with user-edited sections.

        Args:
            generated_code: Newly generated code
            user_code: Code with user edits
            preserve_ranges: List of (start_line, end_line) to preserve from user_code

        Returns:
            Merged code
        """
        # Split both codes into lines
        gen_lines = generated_code.split("\n")
        user_lines = user_code.split("\n")

        # Replace preserved ranges
        result_lines = gen_lines.copy()
        for start, end in preserve_ranges:
            if 0 <= start < len(user_lines) and 0 <= end < len(user_lines):
                # Replace with user's version
                result_lines[start : end + 1] = user_lines[start : end + 1]

        return "\n".join(result_lines)


# =============================================================================
# LibCST Transformer for Method Replacement
# =============================================================================


class AgentMethodTransformer(cst.CSTTransformer):
    """LibCST transformer to replace setup() and train_step() methods."""

    # BEGIN GENERATED marker
    MARKER_BEGIN = "# BEGIN GENERATED"
    MARKER_END = "# END GENERATED"

    def __init__(self, setup_code: str, train_step_code: str):
        self.setup_code = setup_code
        self.train_step_code = train_step_code
        self._in_agent_class = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        """Track if we're inside an Agent class."""
        # Check if this class inherits from Agent
        for base in node.bases:
            if isinstance(base.value, cst.Name) and base.value.value == "Agent":
                self._in_agent_class = True
                break
        return True  # Continue visiting children

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        """Reset agent class flag."""
        self._in_agent_class = False
        return updated_node

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        """Replace setup() and train_step() if inside Agent class."""
        if not self._in_agent_class:
            return updated_node

        method_name = original_node.name.value

        if method_name == "setup" and self.setup_code:
            # Replace setup() body with generated code
            return self._replace_method_body(updated_node, self.setup_code)
        elif method_name == "train_step" and self.train_step_code:
            # Replace train_step() body with generated code
            return self._replace_method_body(updated_node, self.train_step_code)

        return updated_node

    def _replace_method_body(
        self, func_node: cst.FunctionDef, new_body_code: str
    ) -> cst.FunctionDef:
        """Replace function body with new code."""
        try:
            # Parse new body (indented code without def line)
            # Add markers for tracking
            marked_code = f"{self.MARKER_BEGIN}\n{new_body_code}\n{self.MARKER_END}"

            # Wrap in a temporary function to parse
            temp_code = f"def temp():\n{marked_code}"
            temp_tree = cst.parse_module(temp_code)

            # Extract body from temporary function
            temp_func = temp_tree.body[0]
            if isinstance(temp_func, cst.FunctionDef):
                new_body = temp_func.body

                # Replace old body with new
                return func_node.with_changes(body=new_body)

        except Exception as e:
            logger.error(f"Failed to replace method body: {e}")

        return func_node


# =============================================================================
# Singleton instance
# =============================================================================

_generator: CodeGenerator | None = None


def get_code_generator() -> CodeGenerator:
    """Get singleton CodeGenerator instance."""
    global _generator
    if _generator is None:
        _generator = CodeGenerator()
    return _generator


def generate_agent_code(request: CodeGenerationRequest) -> CodeGenerationResponse:
    """Convenience function to generate agent code."""
    return get_code_generator().generate(request)
