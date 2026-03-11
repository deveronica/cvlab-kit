"""
Code Merger - Merge generated code with user edits.

This module preserves user code while updating generated sections.
Uses LibCST for AST manipulation that preserves formatting and comments.

Preservation Strategy:
    1. Detect user-modified code regions (outside node mappings)
    2. Preserve lines marked with # USER CODE markers
    3. Preserve code that doesn't have node mappings
    4. Only update lines that correspond to nodes

Merge Modes:
    - REPLACE: Completely replace method body (destructive)
    - INCREMENTAL: Only update changed nodes (preserves user code)
    - SMART: Auto-detect user modifications and preserve them
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

try:
    import libcst as cst
    from libcst import metadata

    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False

from ..models.node_system import NodeGraph, CodeNodeMapping

logger = logging.getLogger(__name__)


# =============================================================================
# Merge Modes
# =============================================================================


class MergeMode(str, Enum):
    """Code merge strategy."""

    REPLACE = "replace"  # Replace entire method body
    INCREMENTAL = "incremental"  # Update only changed nodes
    SMART = "smart"  # Auto-detect and preserve user code


# =============================================================================
# User Code Markers
# =============================================================================


# Markers for user-protected code regions
USER_CODE_START = "# USER CODE START"
USER_CODE_END = "# USER CODE END"

# Marker for generated code sections
GENERATED_START = "# BEGIN GENERATED"
GENERATED_END = "# END GENERATED"


@dataclass
class CodeRegion:
    """A region of code with metadata."""

    start_line: int
    end_line: int
    content: str
    is_generated: bool = False
    is_user_protected: bool = False
    node_id: Optional[str] = None


@dataclass
class MergeResult:
    """Result of code merge operation."""

    success: bool
    code: str
    preserved_regions: list[CodeRegion] = field(default_factory=list)
    updated_regions: list[CodeRegion] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Code Region Detector
# =============================================================================


class CodeRegionDetector:
    """
    Detect and classify code regions for merge operations.

    Identifies:
    - Generated code (from node mappings)
    - User-protected code (with markers)
    - Unmapped user code (custom additions)
    """

    def __init__(self, mappings: list[CodeNodeMapping]):
        self.mappings = sorted(mappings, key=lambda m: m.line_start)

    def detect_regions(self, code: str) -> list[CodeRegion]:
        """
        Detect all code regions in the source.

        Returns list of CodeRegion objects covering the entire file.
        """
        lines = code.split("\n")
        regions: list[CodeRegion] = []

        current_line = 1
        in_user_block = False
        user_block_start = 0

        for i, line in enumerate(lines, start=1):
            # Check for user code markers
            if USER_CODE_START in line:
                # Close any pending region
                if current_line < i:
                    regions.append(self._create_region(
                        current_line, i - 1, lines, is_user_protected=False
                    ))
                in_user_block = True
                user_block_start = i
                current_line = i

            elif USER_CODE_END in line and in_user_block:
                # Create user-protected region
                regions.append(self._create_region(
                    user_block_start, i, lines, is_user_protected=True
                ))
                in_user_block = False
                current_line = i + 1

            # Check if this line is in a node mapping
            elif not in_user_block:
                mapping = self._find_mapping_for_line(i)
                if mapping:
                    # Close any pending unmapped region
                    if current_line < mapping.line_start:
                        regions.append(self._create_region(
                            current_line, mapping.line_start - 1, lines,
                            is_user_protected=False
                        ))

                    # Create generated region for this mapping
                    regions.append(CodeRegion(
                        start_line=mapping.line_start,
                        end_line=mapping.line_end,
                        content="\n".join(
                            lines[mapping.line_start - 1:mapping.line_end]
                        ),
                        is_generated=True,
                        node_id=mapping.node_id,
                    ))
                    current_line = mapping.line_end + 1

        # Handle any remaining lines
        if current_line <= len(lines):
            if in_user_block:
                # Unclosed user block
                regions.append(self._create_region(
                    user_block_start, len(lines), lines, is_user_protected=True
                ))
            else:
                regions.append(self._create_region(
                    current_line, len(lines), lines, is_user_protected=False
                ))

        return regions

    def _create_region(
        self,
        start: int,
        end: int,
        lines: list[str],
        is_user_protected: bool,
    ) -> CodeRegion:
        """Create a CodeRegion from line range."""
        content = "\n".join(lines[start - 1:end])
        return CodeRegion(
            start_line=start,
            end_line=end,
            content=content,
            is_generated=False,
            is_user_protected=is_user_protected,
        )

    def _find_mapping_for_line(self, line: int) -> Optional[CodeNodeMapping]:
        """Find the node mapping containing this line."""
        for mapping in self.mappings:
            if mapping.line_start <= line <= mapping.line_end:
                return mapping
        return None


# =============================================================================
# Code Merger
# =============================================================================


class CodeMerger:
    """
    Merge generated code with existing source, preserving user edits.

    Uses LibCST for AST manipulation when available, falls back to
    line-based merging otherwise.
    """

    def __init__(self, mode: MergeMode = MergeMode.SMART):
        self.mode = mode

    def merge(
        self,
        original_code: str,
        generated_code: str,
        mappings: list[CodeNodeMapping],
        method_name: Optional[str] = None,
    ) -> MergeResult:
        """
        Merge generated code into original.

        Args:
            original_code: Existing Python source
            generated_code: Newly generated code (method body or full file)
            mappings: Node-to-code mappings from original
            method_name: If provided, only update this method

        Returns:
            MergeResult with merged code and metadata
        """
        if not original_code:
            return MergeResult(success=True, code=generated_code)

        if not generated_code:
            return MergeResult(success=True, code=original_code)

        if self.mode == MergeMode.REPLACE:
            return self._replace_merge(original_code, generated_code, method_name)
        elif self.mode == MergeMode.INCREMENTAL:
            return self._incremental_merge(
                original_code, generated_code, mappings
            )
        else:  # SMART
            return self._smart_merge(
                original_code, generated_code, mappings, method_name
            )

    def _replace_merge(
        self,
        original_code: str,
        generated_code: str,
        method_name: Optional[str],
    ) -> MergeResult:
        """Replace entire method body with generated code."""
        if not LIBCST_AVAILABLE:
            return MergeResult(
                success=False,
                code=original_code,
                warnings=["LibCST not available for code replacement"],
            )

        if not method_name:
            # Replace entire file
            return MergeResult(success=True, code=generated_code)

        try:
            # Parse original
            tree = cst.parse_module(original_code)

            # Replace method body
            transformer = MethodBodyReplacer(method_name, generated_code)
            new_tree = tree.visit(transformer)

            return MergeResult(
                success=True,
                code=new_tree.code,
                updated_regions=[
                    CodeRegion(
                        start_line=0,
                        end_line=0,
                        content=generated_code,
                        is_generated=True,
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Replace merge failed: {e}")
            return MergeResult(
                success=False,
                code=original_code,
                warnings=[f"Replace merge failed: {e}"],
            )

    def _incremental_merge(
        self,
        original_code: str,
        generated_code: str,
        mappings: list[CodeNodeMapping],
    ) -> MergeResult:
        """Update only lines corresponding to changed nodes."""
        if not mappings:
            return MergeResult(success=True, code=generated_code)

        original_lines = original_code.split("\n")
        generated_lines = generated_code.split("\n")

        result_lines = original_lines.copy()
        updated_regions = []

        # Sort mappings by line number
        sorted_mappings = sorted(mappings, key=lambda m: m.line_start)

        # Track line offset due to insertions/deletions
        offset = 0

        for mapping in sorted_mappings:
            # Find corresponding generated code for this node
            gen_snippet = self._find_generated_snippet(
                mapping.node_id, generated_lines
            )

            if gen_snippet is None:
                continue

            # Calculate actual positions with offset
            start = mapping.line_start - 1 + offset
            end = mapping.line_end + offset

            # Get original snippet length
            orig_length = mapping.line_end - mapping.line_start + 1
            new_lines = gen_snippet.split("\n")
            new_length = len(new_lines)

            # Replace lines
            result_lines[start:end] = new_lines

            # Update offset
            offset += new_length - orig_length

            updated_regions.append(CodeRegion(
                start_line=mapping.line_start,
                end_line=mapping.line_end,
                content=gen_snippet,
                is_generated=True,
                node_id=mapping.node_id,
            ))

        return MergeResult(
            success=True,
            code="\n".join(result_lines),
            updated_regions=updated_regions,
        )

    def _smart_merge(
        self,
        original_code: str,
        generated_code: str,
        mappings: list[CodeNodeMapping],
        method_name: Optional[str],
    ) -> MergeResult:
        """
        Smart merge: auto-detect user modifications and preserve them.

        Algorithm:
        1. Detect code regions in original
        2. Preserve user-protected regions
        3. Preserve unmapped regions (user additions)
        4. Update only generated regions that have node mappings
        """
        detector = CodeRegionDetector(mappings)
        regions = detector.detect_regions(original_code)

        original_lines = original_code.split("\n")
        generated_lines = generated_code.split("\n")

        result_lines = []
        preserved_regions = []
        updated_regions = []

        for region in regions:
            if region.is_user_protected:
                # Always preserve user-protected regions
                result_lines.extend(region.content.split("\n"))
                preserved_regions.append(region)

            elif region.is_generated and region.node_id:
                # Update with new generated code
                new_snippet = self._find_generated_snippet(
                    region.node_id, generated_lines
                )

                if new_snippet is not None:
                    result_lines.extend(new_snippet.split("\n"))
                    updated_regions.append(CodeRegion(
                        start_line=region.start_line,
                        end_line=region.end_line,
                        content=new_snippet,
                        is_generated=True,
                        node_id=region.node_id,
                    ))
                else:
                    # Node removed, preserve original
                    result_lines.extend(region.content.split("\n"))
                    preserved_regions.append(region)

            else:
                # Unmapped region - check if it's between generated sections
                # If so, it's likely user code that should be preserved
                result_lines.extend(region.content.split("\n"))
                preserved_regions.append(region)

        return MergeResult(
            success=True,
            code="\n".join(result_lines),
            preserved_regions=preserved_regions,
            updated_regions=updated_regions,
        )

    def _find_generated_snippet(
        self, node_id: str, generated_lines: list[str]
    ) -> Optional[str]:
        """
        Find generated code snippet for a node.

        Looks for markers or comments indicating which node generated
        which code section.
        """
        # Look for node ID markers in generated code
        # Format: # NODE:node_id
        marker = f"# NODE:{node_id}"

        in_node_block = False
        snippet_lines = []

        for line in generated_lines:
            if marker in line:
                in_node_block = True
                continue

            if in_node_block:
                if line.strip().startswith("# NODE:"):
                    # Start of next node
                    break
                snippet_lines.append(line)

        if snippet_lines:
            return "\n".join(snippet_lines)

        return None


# =============================================================================
# LibCST Transformers
# =============================================================================

if LIBCST_AVAILABLE:

    class MethodBodyReplacer(cst.CSTTransformer):
        """Replace a specific method's body with new code."""

        def __init__(self, method_name: str, new_body_code: str):
            self.method_name = method_name
            self.new_body_code = new_body_code
            self._in_target_class = False

        def visit_ClassDef(self, node: cst.ClassDef) -> bool:
            # Check if this is an Agent class
            for base in node.bases:
                if isinstance(base.value, cst.Name):
                    if base.value.value == "Agent":
                        self._in_target_class = True
                        break
            return True

        def leave_ClassDef(
            self, original_node: cst.ClassDef, updated_node: cst.ClassDef
        ) -> cst.ClassDef:
            self._in_target_class = False
            return updated_node

        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.FunctionDef:
            if not self._in_target_class:
                return updated_node

            if original_node.name.value != self.method_name:
                return updated_node

            # Replace body with new code
            try:
                # Wrap in temporary function to parse
                temp_code = f"def temp():\n{self.new_body_code}"
                temp_tree = cst.parse_module(temp_code)

                if temp_tree.body and isinstance(temp_tree.body[0], cst.FunctionDef):
                    new_body = temp_tree.body[0].body
                    return updated_node.with_changes(body=new_body)

            except Exception as e:
                logger.error(f"Failed to parse new body: {e}")

            return updated_node

    class UserCodePreserver(cst.CSTTransformer):
        """
        Preserve user code regions when updating generated code.

        Detects and maintains:
        - Code between USER CODE markers
        - Code with # PRESERVE comment
        - Custom class attributes not in node graph
        """

        def __init__(
            self,
            mappings: list[CodeNodeMapping],
            generated_statements: list[cst.BaseStatement],
        ):
            self.mappings = {m.node_id: m for m in mappings}
            self.generated_statements = generated_statements
            self._preserved_statements: list[cst.BaseStatement] = []

        def visit_SimpleStatementLine(
            self, node: cst.SimpleStatementLine
        ) -> Optional[bool]:
            # Check if this line should be preserved
            # Look for PRESERVE marker in trailing comment
            for comment in node.trailing_comment or []:
                if "PRESERVE" in comment.value:
                    self._preserved_statements.append(node)
                    return False  # Don't visit children

            return True

        def leave_IndentedBlock(
            self,
            original_node: cst.IndentedBlock,
            updated_node: cst.IndentedBlock,
        ) -> cst.IndentedBlock:
            # Merge preserved statements with generated
            if not self._preserved_statements:
                return updated_node

            new_body = list(updated_node.body)

            # Add preserved statements at the end
            for stmt in self._preserved_statements:
                new_body.append(stmt)

            self._preserved_statements.clear()

            return updated_node.with_changes(body=new_body)


# =============================================================================
# Convenience Functions
# =============================================================================


def merge_code(
    original: str,
    generated: str,
    mappings: list[CodeNodeMapping] | None = None,
    mode: MergeMode = MergeMode.SMART,
    method_name: str | None = None,
) -> MergeResult:
    """
    Merge generated code with original, preserving user edits.

    Args:
        original: Original Python source code
        generated: Newly generated code
        mappings: Node-to-code mappings for incremental updates
        mode: Merge strategy (REPLACE, INCREMENTAL, SMART)
        method_name: If provided, only update this method

    Returns:
        MergeResult with merged code and metadata
    """
    merger = CodeMerger(mode)
    return merger.merge(original, generated, mappings or [], method_name)


def preserve_user_code(
    original: str,
    generated: str,
) -> str:
    """
    Simple merge that preserves USER CODE regions.

    Args:
        original: Original code with user modifications
        generated: Newly generated code

    Returns:
        Merged code with user regions preserved
    """
    # Extract user code regions from original
    user_regions = []
    lines = original.split("\n")

    in_user_block = False
    block_lines = []
    block_start = 0

    for i, line in enumerate(lines):
        if USER_CODE_START in line:
            in_user_block = True
            block_start = i
            block_lines = [line]
        elif USER_CODE_END in line and in_user_block:
            block_lines.append(line)
            user_regions.append((block_start, "\n".join(block_lines)))
            in_user_block = False
            block_lines = []
        elif in_user_block:
            block_lines.append(line)

    if not user_regions:
        return generated

    # Insert user regions into generated code
    gen_lines = generated.split("\n")
    result_lines = gen_lines.copy()

    # Find insertion points (after similar context)
    for start_pos, region in user_regions:
        # Insert at similar position
        insert_pos = min(start_pos, len(result_lines))
        region_lines = region.split("\n")

        for j, line in enumerate(region_lines):
            result_lines.insert(insert_pos + j, line)

    return "\n".join(result_lines)


# =============================================================================
# Singleton
# =============================================================================

_merger: CodeMerger | None = None


def get_code_merger(mode: MergeMode = MergeMode.SMART) -> CodeMerger:
    """Get singleton CodeMerger instance."""
    global _merger
    if _merger is None or _merger.mode != mode:
        _merger = CodeMerger(mode)
    return _merger
