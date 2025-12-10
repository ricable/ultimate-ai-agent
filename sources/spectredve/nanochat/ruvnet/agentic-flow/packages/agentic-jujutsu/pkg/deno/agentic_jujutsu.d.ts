/* tslint:disable */
/* eslint-disable */
/**
 * Initialize WASM module
 */
export function wasm_init(): void;
/**
 * Type of jujutsu operation
 *
 * Represents the various operations that can be performed in a jujutsu repository.
 */
export enum OperationType {
  /**
   * Create a new commit
   */
  Commit = 0,
  /**
   * Snapshot operation (automatic)
   */
  Snapshot = 1,
  /**
   * Describe/amend commit message
   */
  Describe = 2,
  /**
   * New commit creation
   */
  New = 3,
  /**
   * Edit commit
   */
  Edit = 4,
  /**
   * Abandon a commit
   */
  Abandon = 5,
  /**
   * Rebase commits
   */
  Rebase = 6,
  /**
   * Squash commits
   */
  Squash = 7,
  /**
   * Resolve conflicts
   */
  Resolve = 8,
  /**
   * Branch operation
   */
  Branch = 9,
  /**
   * Delete a branch
   */
  BranchDelete = 10,
  /**
   * Bookmark operation
   */
  Bookmark = 11,
  /**
   * Create a tag
   */
  Tag = 12,
  /**
   * Checkout a commit
   */
  Checkout = 13,
  /**
   * Restore files
   */
  Restore = 14,
  /**
   * Split a commit
   */
  Split = 15,
  /**
   * Duplicate a commit
   */
  Duplicate = 16,
  /**
   * Undo last operation
   */
  Undo = 17,
  /**
   * Fetch from remote
   */
  Fetch = 18,
  /**
   * Git fetch
   */
  GitFetch = 19,
  /**
   * Push to remote
   */
  Push = 20,
  /**
   * Git push
   */
  GitPush = 21,
  /**
   * Clone repository
   */
  Clone = 22,
  /**
   * Initialize repository
   */
  Init = 23,
  /**
   * Git import
   */
  GitImport = 24,
  /**
   * Git export
   */
  GitExport = 25,
  /**
   * Move changes
   */
  Move = 26,
  /**
   * Diffedit
   */
  Diffedit = 27,
  /**
   * Merge branches
   */
  Merge = 28,
  /**
   * Unknown operation type
   */
  Unknown = 29,
}
/**
 * Branch information
 *
 * Represents a branch in the jujutsu repository.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::types::JJBranch;
 *
 * let branch = JJBranch::new("feature/new-api".to_string(), "def456".to_string(), false);
 * assert_eq!(branch.name, "feature/new-api");
 * ```
 */
export class JJBranch {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new branch
   */
  constructor(name: string, target: string, is_remote: boolean);
  /**
   * Set remote name
   */
  set_remote(remote: string): void;
  /**
   * Get full branch name (e.g., "origin/main")
   */
  full_name(): string;
  /**
   * Short target ID (first 12 characters)
   */
  short_target(): string;
  /**
   * Whether this is a remote branch
   */
  is_remote: boolean;
  /**
   * Whether this branch is tracking a remote
   */
  is_tracking: boolean;
  /**
   * Whether this is the current branch
   */
  is_current: boolean;
  /**
   * Get name (for WASM)
   */
  readonly name: string;
  /**
   * Get target (for WASM)
   */
  readonly target: string;
  /**
   * Get remote (for WASM)
   */
  readonly remote: string | undefined;
  /**
   * Get creation timestamp as ISO 8601 string (for WASM)
   */
  readonly created_at_iso: string;
}
/**
 * Commit metadata
 *
 * Represents a single commit in the jujutsu repository with all associated metadata.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::types::JJCommit;
 *
 * let commit = JJCommit::builder()
 *     .id("abc123".to_string())
 *     .message("Add new feature".to_string())
 *     .author("Bob".to_string())
 *     .build();
 *
 * assert_eq!(commit.message, "Add new feature");
 * ```
 */
export class JJCommit {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new commit with minimal fields
   */
  constructor(id: string, change_id: string, message: string, author: string, author_email: string);
  /**
   * Get timestamp as ISO 8601 string (for WASM)
   */
  timestamp_iso(): string;
  /**
   * Get parent count
   */
  parent_count(): number;
  /**
   * Short commit ID (first 12 characters)
   */
  short_id(): string;
  /**
   * Short change ID (first 12 characters)
   */
  short_change_id(): string;
  /**
   * Whether this is a merge commit
   */
  is_merge: boolean;
  /**
   * Whether this commit has conflicts
   */
  has_conflicts: boolean;
  /**
   * Whether this is an empty commit
   */
  is_empty: boolean;
  /**
   * Get ID (for WASM)
   */
  readonly id: string;
  /**
   * Get change ID (for WASM)
   */
  readonly change_id: string;
  /**
   * Get message (for WASM)
   */
  readonly message: string;
  /**
   * Get author (for WASM)
   */
  readonly author: string;
  /**
   * Get author email (for WASM)
   */
  readonly author_email: string;
  /**
   * Get parents as JSON string (for WASM)
   */
  readonly parents_json: string;
  /**
   * Get branches as JSON string (for WASM)
   */
  readonly branches_json: string;
}
/**
 * Configuration for JJWrapper
 */
export class JJConfig {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create new configuration with defaults
   */
  constructor();
  /**
   * Create default configuration
   */
  static default_config(): JJConfig;
  /**
   * Set jj executable path (builder pattern)
   */
  with_jj_path(path: string): JJConfig;
  /**
   * Set repository path (builder pattern)
   */
  with_repo_path(path: string): JJConfig;
  /**
   * Set operation timeout
   */
  with_timeout(timeout_ms: bigint): JJConfig;
  /**
   * Enable verbose logging
   */
  with_verbose(verbose: boolean): JJConfig;
  /**
   * Set max log entries
   */
  with_max_log_entries(max: number): JJConfig;
  /**
   * Enable AgentDB synchronization
   */
  with_agentdb_sync(enable: boolean): JJConfig;
  /**
   * Timeout for operations in milliseconds
   */
  timeout_ms: bigint;
  /**
   * Enable verbose logging
   */
  verbose: boolean;
  /**
   * Maximum operation log entries to keep in memory
   */
  max_log_entries: number;
  /**
   * Enable AgentDB sync
   */
  enable_agentdb_sync: boolean;
  /**
   * Get jj executable path
   */
  jj_path: string;
  /**
   * Get repository path
   */
  repo_path: string;
}
/**
 * Conflict representation
 *
 * Represents a merge conflict with detailed information about conflicting sides.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::types::JJConflict;
 *
 * let conflict = JJConflict::builder()
 *     .path("src/main.rs".to_string())
 *     .num_conflicts(1)
 *     .conflict_type("content".to_string())
 *     .build();
 * ```
 */
export class JJConflict {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new conflict
   */
  constructor(path: string, num_conflicts: number, conflict_type: string);
  /**
   * Add a side to the conflict
   */
  add_side(side: string): void;
  /**
   * Get the number of sides
   */
  num_sides(): number;
  /**
   * Number of conflict markers
   */
  num_conflicts: number;
  /**
   * Whether conflict is binary (non-text)
   */
  is_binary: boolean;
  /**
   * Whether conflict is resolved
   */
  is_resolved: boolean;
  /**
   * Get ID (for WASM)
   */
  readonly id: string;
  /**
   * Get path (for WASM)
   */
  readonly path: string;
  /**
   * Get conflict type (for WASM)
   */
  readonly conflict_type: string;
  /**
   * Get resolution strategy (for WASM)
   */
  readonly resolution_strategy: string | undefined;
  /**
   * Get sides as JSON string (for WASM)
   */
  readonly sides_json: string;
}
/**
 * Single jujutsu operation
 *
 * Represents a single operation in the jujutsu operation log with metadata.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::operations::{JJOperation, OperationType};
 *
 * let op = JJOperation::builder()
 *     .operation_type(OperationType::Commit)
 *     .command("jj commit".to_string())
 *     .user("alice".to_string())
 *     .build();
 * ```
 */
export class JJOperation {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new operation
   */
  constructor(operation_id: string, command: string, user: string, hostname: string);
  /**
   * Set operation type from string
   */
  set_operation_type(type_str: string): void;
  /**
   * Get timestamp as ISO 8601 string (for WASM)
   */
  timestamp_iso(): string;
  /**
   * Short operation ID (first 12 characters)
   */
  short_id(): string;
  /**
   * Check if operation is a snapshot
   */
  is_snapshot(): boolean;
  /**
   * Check if operation is user-initiated (not automatic snapshot)
   */
  is_user_initiated(): boolean;
  /**
   * Check if operation modifies history (for WASM)
   */
  modifies_history(): boolean;
  /**
   * Check if operation is remote (for WASM)
   */
  is_remote_operation(): boolean;
  /**
   * Duration in milliseconds
   */
  duration_ms: bigint;
  /**
   * Whether this operation was successful
   */
  success: boolean;
  /**
   * Get ID (for WASM)
   */
  readonly id: string;
  /**
   * Get operation ID (for WASM)
   */
  readonly operation_id: string;
  /**
   * Get command (for WASM)
   */
  readonly command: string;
  /**
   * Get user (for WASM)
   */
  readonly user: string;
  /**
   * Get hostname (for WASM)
   */
  readonly hostname: string;
  /**
   * Get parent ID (for WASM)
   */
  readonly parent_id: string | undefined;
  /**
   * Get error (for WASM)
   */
  readonly error: string | undefined;
  /**
   * Get operation type as string (for WASM)
   */
  readonly operation_type_str: string;
  /**
   * Get tags as JSON string (for WASM)
   */
  readonly tags_json: string;
  /**
   * Get metadata as JSON string (for WASM)
   */
  readonly metadata_json: string;
}
/**
 * Result wrapper for jujutsu operations
 *
 * This type extends the basic command result with metadata about execution,
 * warnings, and structured data.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::types::JJResult;
 *
 * let result = JJResult::new("output".to_string(), "".to_string(), 0, 100);
 * assert!(result.success());
 * ```
 */
export class JJResult {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new JJResult
   */
  constructor(stdout: string, stderr: string, exit_code: number, execution_time_ms: bigint);
  /**
   * Check if the command was successful
   */
  success(): boolean;
  /**
   * Get the output as a string (prefer stdout, fallback to stderr)
   */
  output(): string;
  /**
   * Exit code
   */
  exit_code: number;
  /**
   * Command execution time in milliseconds
   */
  execution_time_ms: bigint;
  /**
   * Get stdout (for WASM)
   */
  readonly stdout: string;
  /**
   * Get stderr (for WASM)
   */
  readonly stderr: string;
}
