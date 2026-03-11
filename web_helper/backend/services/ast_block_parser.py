"""AST Block Parser - Extract code blocks from Python source.

This module parses Python source code into structured blocks (functions, classes, methods)
for bidirectional node-code synchronization in the Builder.

Key Features:
- Extract function/class blocks with precise source locations
- Map blocks to node IDs based on line coverage
- Handle incremental parsing for performance
- Detect uncovered code patterns
"""

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CodeBlock:
    """Represents a code block (function, class, or method)."""

    id: str  # Unique ID (hash of location + name)
    name: str  # Block name (e.g., "setup", "MyAgent")
    type: str  # 'function' | 'class' | 'method'
    start_line: int  # 1-based line number
    end_line: int  # 1-based line number
    params: list[dict] = field(default_factory=list)  # Parameter definitions
    returns: list[str] = field(default_factory=list)  # Inferred return types or AST string names
    docstring: Optional[str] = None
    parent: Optional[str] = None  # Parent block ID for methods
    node_ids: list[str] = field(default_factory=list)  # Mapped component node IDs
    indent_level: int = 0  # Indentation level for nesting

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID from location and name."""
        key = f"{self.name}:{self.start_line}:{self.end_line}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


@dataclass
class ParseError:
    """Represents a parsing error."""

    type: str  # 'SyntaxError' | 'IndentationError' | ...
    line: Optional[int] = None
    message: str = ""


@dataclass
class SyncResult:
    """Result of parsing code for synchronization."""

    blocks: list[CodeBlock] = field(default_factory=list)
    errors: list[ParseError] = field(default_factory=list)
    uncovered_lines: set[int] = field(default_factory=set)
    partial: bool = False  # True if parsing was incomplete


# =============================================================================
# AST Block Parser
# =============================================================================


class ASTBlockParser:
    """Extract code blocks from Python source using AST."""

    def __init__(self):
        self._blocks: list[CodeBlock] = []
        self._current_class: Optional[str] = None
        self._cached_tree: Optional[ast.Module] = None
        self._cached_code_hash: str = ""
        self._cached_code: str = ""  # For smart diffing
        self._cached_line_hashes: list[str] = []  # Per-line content hashes
        self._cached_blocks: list[
            CodeBlock
        ] = []  # Previous blocks for incremental update

    def extract_blocks(self, code: str, agent_name: str) -> SyncResult:
        """Extract function/class blocks from Python source code.

        Args:
            code: Python source code
            agent_name: Agent name for context

        Returns:
            SyncResult with extracted blocks and any errors
        """
        self._blocks = []
        self._current_class = None

        # Check cache
        code_hash = hashlib.md5(code.encode()).hexdigest()
        if code_hash == self._cached_code_hash and self._cached_tree:
            # Use cached tree
            tree = self._cached_tree
        else:
            # Parse new code
            try:
                tree = ast.parse(code)
                self._cached_tree = tree
                self._cached_code_hash = code_hash
            except SyntaxError as e:
                logger.error(f"Syntax error in {agent_name}: {e}")
                return SyncResult(
                    blocks=[],
                    errors=[
                        ParseError(
                            type="SyntaxError",
                            line=e.lineno,
                            message=str(e),
                        )
                    ],
                    partial=True,
                )
            except Exception as e:
                logger.error(f"Parse error in {agent_name}: {e}", exc_info=True)
                return SyncResult(
                    blocks=[],
                    errors=[ParseError(type="ParseError", message=str(e))],
                    partial=True,
                )

        # Extract blocks
        visitor = BlockExtractor()
        visitor.visit(tree)
        self._blocks = visitor.blocks

        # Update cache for smart diffing
        self._cached_code = code
        self._cached_line_hashes = self._compute_line_hashes(code)
        self._cached_blocks = self._blocks.copy()

        return SyncResult(
            blocks=self._blocks,
            errors=[],
            partial=False,
        )

    def _compute_line_hashes(self, code: str) -> list[str]:
        """Compute hash for each line of code."""
        lines = code.split("\n")
        return [hashlib.md5(line.encode()).hexdigest()[:8] for line in lines]

    def map_blocks_to_nodes(
        self, blocks: list[CodeBlock], node_graph: dict
    ) -> list[CodeBlock]:
        """Map code blocks to node IDs based on line coverage.

        Args:
            blocks: List of code blocks
            node_graph: Node graph with source location metadata

        Returns:
            Updated blocks with node_ids populated
        """
        nodes = node_graph.get("nodes", [])

        for block in blocks:
            block.node_ids = []
            for node in nodes:
                # Check if node's source lines fall within this block
                metadata = node.get("metadata", {})
                source_line = metadata.get("source_line_start")
                if source_line is None:
                    source = node.get("source", {})
                    if isinstance(source, dict):
                        source_line = source.get("line") or source.get("line_start")
                if source_line and block.start_line <= source_line <= block.end_line:
                    block.node_ids.append(node.get("id", ""))

        return blocks

    def get_changed_ranges(self, new_code: str) -> list[tuple[int, int]]:
        """Detect changed line ranges for incremental parsing.

        Uses line-by-line hash comparison to detect changes efficiently.

        Args:
            new_code: New version of code

        Returns:
            List of (start_line, end_line) tuples for changed ranges (1-based)
        """
        # No cache, everything is new
        if not self._cached_line_hashes:
            lines = new_code.split("\n")
            return [(1, len(lines))]

        # Compute new line hashes
        new_hashes = self._compute_line_hashes(new_code)
        old_hashes = self._cached_line_hashes

        # Find changed line ranges
        changed_ranges: list[tuple[int, int]] = []
        in_change = False
        change_start = 0

        max_len = max(len(old_hashes), len(new_hashes))

        for i in range(max_len):
            old_hash = old_hashes[i] if i < len(old_hashes) else None
            new_hash = new_hashes[i] if i < len(new_hashes) else None

            is_changed = old_hash != new_hash

            if is_changed and not in_change:
                # Start of a changed region
                in_change = True
                change_start = i + 1  # 1-based line number
            elif not is_changed and in_change:
                # End of a changed region
                in_change = False
                changed_ranges.append(
                    (change_start, i)
                )  # End is i (1-based is i+1-1=i)

        # Handle trailing change
        if in_change:
            changed_ranges.append((change_start, max_len))

        return changed_ranges

    def get_changed_blocks(self, new_code: str) -> list[CodeBlock]:
        """Get only the blocks that have changed.

        Uses get_changed_ranges to identify changes, then returns
        blocks whose line ranges overlap with changes.

        Args:
            new_code: New version of code

        Returns:
            List of CodeBlocks that need re-parsing
        """
        changed_ranges = self.get_changed_ranges(new_code)

        if not changed_ranges:
            return []  # No changes

        if not self._cached_blocks:
            return []  # No cached blocks to compare

        # Find blocks that overlap with changed ranges
        changed_blocks: list[CodeBlock] = []

        for block in self._cached_blocks:
            for start, end in changed_ranges:
                # Check if block overlaps with changed range
                if block.start_line <= end and block.end_line >= start:
                    changed_blocks.append(block)
                    break

        return changed_blocks

    def compute_block_hash(self, code: str, block: CodeBlock) -> str:
        """Compute hash for a specific block's content.

        Useful for detecting if a block's content has changed.

        Args:
            code: Full source code
            block: CodeBlock to hash

        Returns:
            Hash string of the block's content
        """
        lines = code.split("\n")
        block_lines = lines[block.start_line - 1 : block.end_line]
        content = "\n".join(block_lines)
        return hashlib.md5(content.encode()).hexdigest()[:12]


# =============================================================================
# AST Visitor for Block Extraction
# =============================================================================


class BlockExtractor(ast.NodeVisitor):
    """AST visitor to extract code blocks."""

    def __init__(self):
        self.blocks: list[CodeBlock] = []
        self._current_class: Optional[str] = None
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class definition."""
        # Calculate indentation level
        indent_level = len(self._class_stack)

        # Create block for class
        block = CodeBlock(
            id="",  # Will be auto-generated
            name=node.name,
            type="class",
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            parent=self._class_stack[-1] if self._class_stack else None,
            indent_level=indent_level,
        )
        self.blocks.append(block)

        # Track current class for methods
        self._class_stack.append(node.name)
        self._current_class = node.name

        # Visit children (methods)
        self.generic_visit(node)

        # Pop class stack
        self._class_stack.pop()
        self._current_class = self._class_stack[-1] if self._class_stack else None

    def _visit_callable_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Extract function/method definition."""
        # Determine if this is a method or standalone function
        is_method = self._current_class is not None
        block_type = "method" if is_method else "function"

        # Extract parameters
        params = self._extract_params(node.args)

        # Calculate indentation level
        indent_level = len(self._class_stack)

        # Create block
        block = CodeBlock(
            id="",  # Will be auto-generated
            name=node.name,
            type=block_type,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            params=params,
            returns=self._extract_returns(node),
            docstring=ast.get_docstring(node),
            parent=self._current_class,
            indent_level=indent_level,
        )
        self.blocks.append(block)

        # Don't visit children (nested functions) to avoid deep nesting
        # self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "setup":
            self._validate_setup_whitelist(node)
        self._visit_callable_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Extract async function definition (treat same as regular function)."""
        if node.name == "setup":
            self._validate_setup_whitelist(node)
        self._visit_callable_def(node)

    def _extract_params(self, args: ast.arguments) -> list[dict]:
        """Extract function parameters."""
        params = []

        # Positional args
        for arg in args.args:
            params.append(
                {
                    "name": arg.arg,
                    "type": self._get_annotation(arg.annotation),
                    "default": None,
                }
            )

        # Defaults (for last N args)
        defaults = args.defaults
        if defaults:
            offset = len(params) - len(defaults)
            for i, default in enumerate(defaults):
                params[offset + i]["default"] = ast.unparse(default)

        # *args
        if args.vararg:
            params.append(
                {
                    "name": f"*{args.vararg.arg}",
                    "type": self._get_annotation(args.vararg.annotation),
                    "default": None,
                }
            )

        # **kwargs
        if args.kwarg:
            params.append(
                {
                    "name": f"**{args.kwarg.arg}",
                    "type": self._get_annotation(args.kwarg.annotation),
                    "default": None,
                }
            )

        return params

    def _extract_returns(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract return types or return value structure from a function."""
        # 1. If explicit return annotation exists, use it
        if getattr(node, "returns", None):
            ann = self._get_annotation(node.returns)
            if ann:
                return [ann]

        # 2. Try to analyze AST return statements
        returns = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                if isinstance(child.value, ast.Tuple) or isinstance(child.value, ast.List):
                    for elt in child.value.elts:
                        try:
                            returns.append(ast.unparse(elt))
                        except Exception:
                            pass
                else:
                    try:
                        returns.append(ast.unparse(child.value))
                    except Exception:
                        pass
        # De-duplicate while preserving order
        return list(dict.fromkeys(returns))

    def _validate_setup_whitelist(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Ensure setup() only contains allowed syntax (No if, for, while)."""
        allowed_nodes = (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Expr,
            ast.Call,
            ast.Pass,
            ast.Return,
        )
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # docstring is fine
            if not isinstance(stmt, allowed_nodes):
                raise SyntaxError(
                    f"Illegal syntax in setup(): Found `{type(stmt).__name__}` at line {stmt.lineno}. "
                    f"setup() is strictly a declarative DSL. Loops, conditionals, and context managers are forbidden."
                )

    def _get_annotation(self, annotation: Optional[ast.expr]) -> Optional[str]:
        """Extract type annotation as string."""
        if annotation:
            try:
                return ast.unparse(annotation)
            except Exception:
                return None
        return None


# =============================================================================
# Singleton instance
# =============================================================================

_parser: Optional[ASTBlockParser] = None


def get_ast_block_parser() -> ASTBlockParser:
    """Get singleton ASTBlockParser instance."""
    global _parser
    if _parser is None:
        _parser = ASTBlockParser()
    return _parser
