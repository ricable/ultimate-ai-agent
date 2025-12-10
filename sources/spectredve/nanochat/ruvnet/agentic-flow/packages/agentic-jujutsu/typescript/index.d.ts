/**
 * TypeScript definitions for @agentic-flow/jujutsu
 * WASM-enabled Jujutsu VCS wrapper for AI agent collaboration
 */

/**
 * Configuration for JJWrapper
 */
export class JJConfig {
  /**
   * Create a new configuration with default values
   */
  constructor();

  /**
   * Create default configuration
   */
  static default(): JJConfig;

  /**
   * Set the path to the jj executable
   * @param path - Path to jj binary (default: "jj")
   */
  with_jj_path(path: string): JJConfig;

  /**
   * Set the repository path
   * @param path - Path to repository (default: ".")
   */
  with_repo_path(path: string): JJConfig;

  /**
   * Set operation timeout in milliseconds
   * @param timeout_ms - Timeout in milliseconds (default: 30000)
   */
  with_timeout(timeout_ms: number): JJConfig;

  /**
   * Enable verbose logging
   * @param verbose - Enable verbose mode
   */
  with_verbose(verbose: boolean): JJConfig;

  /**
   * Set maximum operation log entries to keep in memory
   * @param max - Maximum entries (default: 1000)
   */
  with_max_log_entries(max: number): JJConfig;

  /**
   * Enable AgentDB synchronization
   * @param enable - Enable AgentDB sync
   */
  with_agentdb_sync(enable: boolean): JJConfig;

  /** Path to jj executable */
  jj_path: string;

  /** Repository path */
  repo_path: string;

  /** Operation timeout in milliseconds */
  timeout_ms: number;

  /** Enable verbose logging */
  verbose: boolean;

  /** Maximum operation log entries */
  max_log_entries: number;

  /** Enable AgentDB sync */
  enable_agentdb_sync: boolean;
}

/**
 * Error types for Jujutsu operations
 */
export class JJError extends Error {
  /**
   * Get error message as string
   */
  message(): string;

  /**
   * Check if error is recoverable
   */
  is_recoverable(): boolean;
}

/**
 * Main wrapper for Jujutsu VCS operations
 */
export class JJWrapper {
  /**
   * Create a new JJWrapper instance
   * @param config - Optional configuration (uses defaults if not provided)
   */
  static new(config?: JJConfig): Promise<JJWrapper>;

  /**
   * Execute arbitrary jj command
   * @param args - Command arguments (e.g., ["status"], ["log", "-r", "@"])
   */
  execute(args: string[]): Promise<JJResult>;

  /**
   * Get operation log entries
   * @param limit - Maximum number of operations to retrieve
   */
  getOperations(limit: number): Promise<JJOperation[]>;

  /**
   * Get conflicts in a commit
   * @param commit - Optional commit ID (default: working copy)
   */
  getConflicts(commit?: string): Promise<JJConflict[]>;

  /**
   * Set commit description
   * @param message - Commit message
   */
  describe(message: string): Promise<JJOperation>;

  /**
   * Get repository status
   */
  status(): Promise<JJResult>;

  /**
   * Get diff between two commits
   * @param from - Source commit
   * @param to - Target commit
   */
  diff(from: string, to: string): Promise<JJDiff>;

  /**
   * Create a new commit with description
   * @param message - Commit message
   */
  commit(message: string): Promise<JJOperation>;

  /**
   * Create a new branch
   * @param name - Branch name
   */
  branch_create(name: string): Promise<JJResult>;

  /**
   * List all branches
   */
  branch_list(): Promise<JJBranch[]>;

  /**
   * Abandon a commit
   * @param commit - Commit ID to abandon
   */
  abandon(commit: string): Promise<JJOperation>;

  /**
   * Restore files from a commit
   * @param paths - File paths to restore
   * @param from - Source commit (default: parent)
   */
  restore(paths: string[], from?: string): Promise<JJResult>;
}

/**
 * Result of a jj command execution
 */
export interface JJResult {
  /** Exit code of the command */
  exit_code: number;

  /** Standard output */
  stdout: string;

  /** Standard error */
  stderr: string;

  /** Whether the command succeeded */
  success: boolean;
}

/**
 * Jujutsu operation log entry
 */
export interface JJOperation {
  /** Operation ID (hash) */
  id: string;

  /** Parent operation IDs */
  parents: string[];

  /** Operation timestamp (ISO 8601) */
  timestamp: string;

  /** User who performed the operation */
  user: string;

  /** Operation description */
  description: string;

  /** Operation type */
  operation_type: OperationType;

  /** Tags associated with this operation */
  tags: string[];

  /** Metadata key-value pairs */
  metadata: Record<string, string>;
}

/**
 * Type of Jujutsu operation
 */
export type OperationType =
  | "snapshot"
  | "describe"
  | "commit"
  | "abandon"
  | "edit"
  | "new"
  | "squash"
  | "unsquash"
  | "move"
  | "rebase"
  | "resolve"
  | "restore"
  | "duplicate"
  | "split"
  | "unknown";

/**
 * Conflict in a commit
 */
export interface JJConflict {
  /** File path with conflict */
  path: string;

  /** Number of conflict hunks */
  num_hunks: number;

  /** Sides involved in the conflict */
  sides: ConflictSide[];

  /** Whether the conflict is resolved */
  resolved: boolean;
}

/**
 * Side of a conflict (base, left, right, etc.)
 */
export interface ConflictSide {
  /** Content of this side */
  content: string;

  /** Label for this side (e.g., "base", "ours", "theirs") */
  label: string;
}

/**
 * Commit information
 */
export interface JJCommit {
  /** Commit ID (change ID) */
  change_id: string;

  /** Commit hash */
  commit_id: string;

  /** Parent commit IDs */
  parents: string[];

  /** Commit description (message) */
  description: string;

  /** Author name */
  author: string;

  /** Author email */
  author_email: string;

  /** Commit timestamp (ISO 8601) */
  timestamp: string;

  /** Whether this is the working copy */
  is_working_copy: boolean;

  /** Branches pointing to this commit */
  branches: string[];

  /** Tags on this commit */
  tags: string[];

  /** Conflict status */
  has_conflict: boolean;

  /** Whether this commit is empty */
  is_empty: boolean;
}

/**
 * Branch information
 */
export interface JJBranch {
  /** Branch name */
  name: string;

  /** Target commit ID */
  target: string;

  /** Whether this is a remote branch */
  remote: boolean;

  /** Remote name (if remote branch) */
  remote_name?: string;

  /** Whether branch is tracking a remote */
  tracking?: string;
}

/**
 * Diff between two commits
 */
export interface JJDiff {
  /** Source commit */
  from: string;

  /** Target commit */
  to: string;

  /** File changes */
  files: FileDiff[];

  /** Summary statistics */
  stats: DiffStats;
}

/**
 * Diff for a single file
 */
export interface FileDiff {
  /** File path */
  path: string;

  /** Change type */
  change_type: "added" | "modified" | "deleted" | "renamed" | "copied";

  /** Old path (for renames) */
  old_path?: string;

  /** Lines added */
  additions: number;

  /** Lines removed */
  deletions: number;

  /** Diff hunks */
  hunks: DiffHunk[];
}

/**
 * A hunk in a diff
 */
export interface DiffHunk {
  /** Old file line number */
  old_start: number;

  /** Old file line count */
  old_count: number;

  /** New file line number */
  new_start: number;

  /** New file line count */
  new_count: number;

  /** Lines in this hunk */
  lines: DiffLine[];
}

/**
 * A line in a diff
 */
export interface DiffLine {
  /** Line type: " " (context), "+" (added), "-" (removed) */
  type: " " | "+" | "-";

  /** Line content */
  content: string;
}

/**
 * Diff statistics
 */
export interface DiffStats {
  /** Total files changed */
  files_changed: number;

  /** Total insertions */
  insertions: number;

  /** Total deletions */
  deletions: number;
}

/**
 * Initialize WASM module (for web target)
 */
export function init(): void;

/**
 * Version of the agentic-jujutsu package
 */
export const VERSION: string;
