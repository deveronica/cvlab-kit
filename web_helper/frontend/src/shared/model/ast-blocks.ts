/**
 * AST Block Types for Bidirectional Node-Code Synchronization
 *
 * These types support the Builder's AST block-based editor,
 * enabling structured code editing and real-time sync between nodes and code.
 */

// =============================================================================
// Code Block Types
// =============================================================================

export interface Parameter {
  name: string;
  type?: string | null;
  default?: string | null;
}

export interface CodeBlock {
  id: string;
  name: string;
  type: 'function' | 'class' | 'method';
  start_line: number;  // 1-based line number
  end_line: number;    // 1-based line number
  params: Parameter[];
  docstring?: string | null;
  parent?: string | null;  // Parent block ID for methods
  node_ids: string[];      // Mapped component node IDs
  indent_level: number;    // Indentation level for nesting
}

// =============================================================================
// Sync State Types
// =============================================================================

export type ConflictType =
  | 'both_modified'
  | 'parse_error'
  | 'generation_error'
  | 'version_mismatch';

export type ResolutionStrategy =
  | 'code_wins'  // Keep code changes, regenerate nodes
  | 'nodes_win'  // Keep node changes, regenerate code
  | 'manual'     // User must manually resolve
  | 'merge';     // Attempt automatic merge (future)

export interface Conflict {
  type: ConflictType;
  message: string;
  code_version: string;  // Hash of code at conflict time
  node_version: string;  // Hash of nodes at conflict time
  details: Record<string, unknown>;
}

export interface SyncState {
  isDirty: boolean;
  lastSyncedCode: string;
  lastSyncedNodes: Record<string, unknown> | null;
  conflicts: Conflict[];
  isSyncing: boolean;
  syncError: string | null;
}

// =============================================================================
// API Request/Response Types
// =============================================================================

export interface SyncCodeToNodesRequest {
  agent_name: string;
  code_content: string;
  last_synced_code?: string | null;
  last_synced_nodes?: Record<string, unknown> | null;
}

export interface SyncCodeToNodesResponse {
  success: boolean;
  node_graph?: Record<string, unknown>;
  code_blocks?: CodeBlock[];
  uncovered_lines?: number[];
  conflicts?: Conflict[] | null;
  error?: string | null;
  errors?: Array<{
    type: string;
    line?: number | null;
    message: string;
  }>;
}

/**
 * Merge mode for Node → Code generation
 */
export type MergeMode = 'replace' | 'incremental' | 'smart';

/**
 * Preserved code region (after merge)
 */
export interface PreservedRegion {
  start_line: number;
  end_line: number;
  reason: string;  // 'user_marker' | 'unrecognized_code' | 'comment_block'
}

export interface GenerateCodeFromNodesRequest {
  agent_name: string;
  node_graph: Record<string, unknown>;
  original_code?: string | null;
  preserve_sections?: Array<[number, number]>;  // Line ranges to preserve
  merge_mode?: MergeMode;  // NEW: How to merge with existing code
}

export interface GenerateCodeFromNodesResponse {
  success: boolean;
  code?: string;
  modified_lines?: number[];
  preserved_regions?: PreservedRegion[];  // NEW: Regions that were preserved
  warnings?: string[] | null;
}

// =============================================================================
// Parse Error Types
// =============================================================================

export interface ParseError {
  type: string;  // 'SyntaxError' | 'IndentationError' | ...
  line?: number | null;
  message: string;
}

// =============================================================================
// Code Block Mapping Types
// =============================================================================

/**
 * Maps code blocks to nodes for bidirectional navigation
 */
export interface CodeNodeMapping {
  blockId: string;
  block: CodeBlock;
  nodeIds: string[];
  startLine: number;
  endLine: number;
}

/**
 * Cursor position in the code editor
 */
export interface CursorPosition {
  line: number;  // 1-based
  col: number;   // 0-based
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Find code block that contains the given line
 */
export function findBlockAtLine(
  blocks: CodeBlock[],
  line: number
): CodeBlock | null {
  return blocks.find(b => b.start_line <= line && line <= b.end_line) || null;
}

/**
 * Find all nodes associated with a code block
 */
export function getNodesForBlock(
  block: CodeBlock,
  mappings: CodeNodeMapping[]
): string[] {
  const mapping = mappings.find(m => m.blockId === block.id);
  return mapping ? mapping.nodeIds : [];
}

/**
 * Check if two code blocks overlap
 */
export function blocksOverlap(a: CodeBlock, b: CodeBlock): boolean {
  return !(a.end_line < b.start_line || b.end_line < a.start_line);
}

/**
 * Sort blocks by start line
 */
export function sortBlocks(blocks: CodeBlock[]): CodeBlock[] {
  return [...blocks].sort((a, b) => a.start_line - b.start_line);
}
