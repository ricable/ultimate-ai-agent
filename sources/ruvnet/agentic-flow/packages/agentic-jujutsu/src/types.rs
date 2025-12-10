//! Core type system for agentic-jujutsu
//!
//! This module defines the fundamental data structures used throughout the crate,
//! with full N-API bindings for JavaScript/TypeScript interoperability.
//!
//! # Examples
//!
//! ```rust
//! use agentic_jujutsu::types::{JJCommit, JJBranch, JJConflict};
//! use chrono::Utc;
//!
//! // Create a commit using builder pattern
//! let commit = JJCommit::builder()
//!     .id("abc123".to_string())
//!     .change_id("xyz789".to_string())
//!     .message("Initial commit".to_string())
//!     .author("Alice".to_string())
//!     .author_email("alice@example.com".to_string())
//!     .build();
//!
//! // Create a branch
//! let branch = JJBranch::new("main".to_string(), "abc123".to_string(), false);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use napi_derive::napi;

use crate::error::{JJError, Result};

/// Result wrapper for jujutsu operations
///
/// This type extends the basic command result with metadata about execution,
/// warnings, and structured data.
///
/// # Examples
///
/// ```rust
/// use agentic_jujutsu::types::JJResult;
///
/// let result = JJResult::new("output".to_string(), "".to_string(), 0, 100);
/// assert!(result.success());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct JJResult {
    /// Standard output from the command
    pub stdout: String,

    /// Standard error from the command
    pub stderr: String,

    /// Exit code
    pub exit_code: i32,

    /// Command execution time in milliseconds
    pub execution_time_ms: u32,
}

impl JJResult {
    /// Create a new JJResult
    pub fn new(stdout: String, stderr: String, exit_code: i32, execution_time_ms: u64) -> Self {
        Self {
            stdout,
            stderr,
            exit_code,
            execution_time_ms: execution_time_ms as u32,
        }
    }

    /// Check if the command was successful
    #[inline]
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }

    /// Get the output as a string (prefer stdout, fallback to stderr)
    pub fn output(&self) -> String {
        if !self.stdout.is_empty() {
            self.stdout.clone()
        } else {
            self.stderr.clone()
        }
    }
}

impl JJResult {
    /// Convert to standard Result type
    pub fn to_result(&self) -> Result<String> {
        if self.success() {
            Ok(self.output())
        } else {
            Err(JJError::CommandFailed(self.stderr.clone()))
        }
    }
}

/// Commit metadata
///
/// Represents a single commit in the jujutsu repository with all associated metadata.
///
/// # Examples
///
/// ```rust
/// use agentic_jujutsu::types::JJCommit;
///
/// let commit = JJCommit::builder()
///     .id("abc123".to_string())
///     .message("Add new feature".to_string())
///     .author("Bob".to_string())
///     .build();
///
/// assert_eq!(commit.message, "Add new feature");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct JJCommit {
    /// Commit ID (revision hash)
    pub id: String,

    /// Change ID (unique identifier for the change)
    pub change_id: String,

    /// Commit message
    pub message: String,

    /// Author name
    pub author: String,

    /// Author email
    pub author_email: String,

    /// Timestamp (ISO 8601 format)
    pub timestamp: String,

    /// Parent commit IDs
    pub parents: Vec<String>,

    /// Child commit IDs
    pub children: Vec<String>,

    /// Branches pointing to this commit
    pub branches: Vec<String>,

    /// Tags associated with this commit
    pub tags: Vec<String>,

    /// Whether this is a merge commit
    pub is_merge: bool,

    /// Whether this commit has conflicts
    pub has_conflicts: bool,

    /// Whether this is an empty commit
    pub is_empty: bool,
}

impl JJCommit {
    /// Create a new commit with minimal fields
    pub fn new(
        id: String,
        change_id: String,
        message: String,
        author: String,
        author_email: String,
    ) -> Self {
        Self {
            id,
            change_id,
            message,
            author,
            author_email,
            timestamp: Utc::now().to_rfc3339(),
            parents: Vec::new(),
            children: Vec::new(),
            branches: Vec::new(),
            tags: Vec::new(),
            is_merge: false,
            has_conflicts: false,
            is_empty: false,
        }
    }

    /// Create a builder for constructing commits
    pub fn builder() -> JJCommitBuilder {
        JJCommitBuilder::default()
    }

    /// Get timestamp as ISO 8601 string
    pub fn timestamp_iso(&self) -> String {
        self.timestamp.clone()
    }

    /// Get parent count
    pub fn parent_count(&self) -> u32 {
        self.parents.len() as u32
    }

    /// Short commit ID (first 12 characters)
    #[inline]
    pub fn short_id(&self) -> String {
        self.id.chars().take(12).collect()
    }

    /// Short change ID (first 12 characters)
    #[inline]
    pub fn short_change_id(&self) -> String {
        self.change_id.chars().take(12).collect()
    }

    /// Add a parent commit ID
    pub fn add_parent(&mut self, parent_id: String) {
        self.parents.push(parent_id);
        if self.parents.len() > 1 {
            self.is_merge = true;
        }
    }

    /// Get parents
    pub fn parents(&self) -> &[String] {
        &self.parents
    }

    /// Get children
    pub fn children(&self) -> &[String] {
        &self.children
    }

    /// Add a child commit ID
    pub fn add_child(&mut self, child_id: String) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
        }
    }

    /// Add a branch
    pub fn add_branch(&mut self, branch: String) {
        if !self.branches.contains(&branch) {
            self.branches.push(branch);
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }
}

/// Builder for JJCommit
#[derive(Default)]
pub struct JJCommitBuilder {
    id: Option<String>,
    change_id: Option<String>,
    message: Option<String>,
    author: Option<String>,
    author_email: Option<String>,
    timestamp: Option<DateTime<Utc>>,
    parents: Vec<String>,
    children: Vec<String>,
    branches: Vec<String>,
    tags: Vec<String>,
    is_merge: bool,
    has_conflicts: bool,
    is_empty: bool,
}

impl JJCommitBuilder {
    /// Set commit ID
    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Set change ID
    pub fn change_id(mut self, change_id: String) -> Self {
        self.change_id = Some(change_id);
        self
    }

    /// Set commit message
    pub fn message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }

    /// Set author name
    pub fn author(mut self, author: String) -> Self {
        self.author = Some(author);
        self
    }

    /// Set author email
    pub fn author_email(mut self, email: String) -> Self {
        self.author_email = Some(email);
        self
    }

    /// Set timestamp
    pub fn timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Add a parent
    pub fn parent(mut self, parent: String) -> Self {
        self.parents.push(parent);
        if self.parents.len() > 1 {
            self.is_merge = true;
        }
        self
    }

    /// Add a child
    pub fn child(mut self, child: String) -> Self {
        self.children.push(child);
        self
    }

    /// Add a branch
    pub fn branch(mut self, branch: String) -> Self {
        self.branches.push(branch);
        self
    }

    /// Add a tag
    pub fn tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Mark as merge commit
    pub fn is_merge(mut self, is_merge: bool) -> Self {
        self.is_merge = is_merge;
        self
    }

    /// Mark as having conflicts
    pub fn has_conflicts(mut self, has_conflicts: bool) -> Self {
        self.has_conflicts = has_conflicts;
        self
    }

    /// Mark as empty commit
    pub fn is_empty(mut self, is_empty: bool) -> Self {
        self.is_empty = is_empty;
        self
    }

    /// Build the commit
    pub fn build(self) -> JJCommit {
        JJCommit {
            id: self.id.unwrap_or_default(),
            change_id: self.change_id.unwrap_or_default(),
            message: self.message.unwrap_or_default(),
            author: self.author.unwrap_or_default(),
            author_email: self.author_email.unwrap_or_default(),
            timestamp: self.timestamp.map(|t| t.to_rfc3339()).unwrap_or_else(|| Utc::now().to_rfc3339()),
            parents: self.parents,
            children: self.children,
            branches: self.branches,
            tags: self.tags,
            is_merge: self.is_merge,
            has_conflicts: self.has_conflicts,
            is_empty: self.is_empty,
        }
    }
}

/// Branch information
///
/// Represents a branch in the jujutsu repository.
///
/// # Examples
///
/// ```rust
/// use agentic_jujutsu::types::JJBranch;
///
/// let branch = JJBranch::new("feature/new-api".to_string(), "def456".to_string(), false);
/// assert_eq!(branch.name, "feature/new-api");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct JJBranch {
    /// Branch name
    pub name: String,

    /// Commit ID this branch points to
    pub target: String,

    /// Whether this is a remote branch
    pub is_remote: bool,

    /// Remote name (if remote branch)
    pub remote: Option<String>,

    /// Whether this branch is tracking a remote
    pub is_tracking: bool,

    /// Whether this is the current branch
    pub is_current: bool,

    /// Creation timestamp (ISO 8601 format)
    pub created_at: String,
}

impl JJBranch {
    /// Create a new branch
    pub fn new(name: String, target: String, is_remote: bool) -> Self {
        Self {
            name,
            target,
            is_remote,
            remote: None,
            is_tracking: false,
            is_current: false,
            created_at: Utc::now().to_rfc3339(),
        }
    }

    /// Set remote name
    pub fn set_remote(&mut self, remote: String) {
        self.remote = Some(remote);
    }

    /// Get full branch name (e.g., "origin/main")
    pub fn full_name(&self) -> String {
        if let Some(ref remote) = self.remote {
            format!("{}/{}", remote, self.name)
        } else {
            self.name.clone()
        }
    }

    /// Short target ID (first 12 characters)
    pub fn short_target(&self) -> String {
        self.target.chars().take(12).collect()
    }

    /// Get creation timestamp as ISO 8601 string
    pub fn created_at_iso(&self) -> String {
        self.created_at.clone()
    }
}

/// Conflict representation
///
/// Represents a merge conflict with detailed information about conflicting sides.
///
/// # Examples
///
/// ```rust
/// use agentic_jujutsu::types::JJConflict;
///
/// let conflict = JJConflict::builder()
///     .path("src/main.rs".to_string())
///     .num_conflicts(1)
///     .conflict_type("content".to_string())
///     .build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct JJConflict {
    /// Unique conflict identifier
    pub id: String,

    /// Path to the conflicted file
    pub path: String,

    /// Number of conflict markers
    pub num_conflicts: u32,

    /// Sides involved in the conflict
    pub sides: Vec<String>,

    /// Conflict type (e.g., "content", "modify/delete")
    pub conflict_type: String,

    /// Whether conflict is binary (non-text)
    pub is_binary: bool,

    /// Whether conflict is resolved
    pub is_resolved: bool,

    /// Resolution strategy used (if resolved)
    pub resolution_strategy: Option<String>,
}

impl JJConflict {
    /// Create a new conflict
    pub fn new(path: String, num_conflicts: u32, conflict_type: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            path,
            num_conflicts,
            sides: Vec::new(),
            conflict_type,
            is_binary: false,
            is_resolved: false,
            resolution_strategy: None,
        }
    }

    /// Add a side to the conflict
    pub fn add_side(&mut self, side: String) {
        self.sides.push(side);
    }

    /// Get the number of sides
    pub fn num_sides(&self) -> u32 {
        self.sides.len() as u32
    }
}

impl JJConflict {
    /// Create a builder for constructing conflicts
    pub fn builder() -> JJConflictBuilder {
        JJConflictBuilder::default()
    }
}

/// Builder for JJConflict
#[derive(Default)]
pub struct JJConflictBuilder {
    path: Option<String>,
    num_conflicts: u32,
    sides: Vec<String>,
    conflict_type: Option<String>,
    is_binary: bool,
    is_resolved: bool,
    resolution_strategy: Option<String>,
}

impl JJConflictBuilder {
    /// Set file path
    pub fn path(mut self, path: String) -> Self {
        self.path = Some(path);
        self
    }

    /// Set number of conflicts
    pub fn num_conflicts(mut self, num: u32) -> Self {
        self.num_conflicts = num;
        self
    }

    /// Add a side
    pub fn side(mut self, side: String) -> Self {
        self.sides.push(side);
        self
    }

    /// Set conflict type
    pub fn conflict_type(mut self, conflict_type: String) -> Self {
        self.conflict_type = Some(conflict_type);
        self
    }

    /// Mark as binary conflict
    pub fn is_binary(mut self, is_binary: bool) -> Self {
        self.is_binary = is_binary;
        self
    }

    /// Mark as resolved
    pub fn is_resolved(mut self, is_resolved: bool) -> Self {
        self.is_resolved = is_resolved;
        self
    }

    /// Set resolution strategy
    pub fn resolution_strategy(mut self, strategy: String) -> Self {
        self.resolution_strategy = Some(strategy);
        self
    }

    /// Build the conflict
    pub fn build(self) -> JJConflict {
        JJConflict {
            id: Uuid::new_v4().to_string(),
            path: self.path.unwrap_or_default(),
            num_conflicts: self.num_conflicts,
            sides: self.sides,
            conflict_type: self.conflict_type.unwrap_or_else(|| "content".to_string()),
            is_binary: self.is_binary,
            is_resolved: self.is_resolved,
            resolution_strategy: self.resolution_strategy,
        }
    }
}

/// Represents a diff between two commits
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct JJDiff {
    /// Files added
    pub added: Vec<String>,

    /// Files modified
    pub modified: Vec<String>,

    /// Files deleted
    pub deleted: Vec<String>,

    /// Files renamed (serialized as "old_path:new_path")
    pub renamed: Vec<String>,

    /// Total number of additions
    pub additions: u32,

    /// Total number of deletions
    pub deletions: u32,

    /// Diff content (unified diff format)
    pub content: String,
}

impl JJDiff {
    /// Create a new empty diff
    pub fn new() -> Self {
        Self {
            added: Vec::new(),
            modified: Vec::new(),
            deleted: Vec::new(),
            renamed: Vec::new(),
            additions: 0,
            deletions: 0,
            content: String::new(),
        }
    }

    /// Total number of files changed
    #[inline]
    pub fn total_files_changed(&self) -> u32 {
        (self.added.len() + self.modified.len() + self.deleted.len() + self.renamed.len()) as u32
    }

    /// Check if diff is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_files_changed() == 0
    }
}

impl Default for JJDiff {
    fn default() -> Self {
        Self::new()
    }
}

/// Working copy change
///
/// Represents a change in the working copy that hasn't been committed yet.
///
/// # Examples
///
/// ```rust
/// use agentic_jujutsu::types::{JJChange, ChangeStatus};
///
/// let change = JJChange::new("src/main.rs".to_string());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct JJChange {
    /// File path
    pub file_path: String,

    /// Change status
    pub status: ChangeStatus,

    /// Whether file is staged
    pub is_staged: bool,

    /// Size in bytes (if applicable) - using f64 for N-API compatibility with large numbers
    pub size_bytes: Option<f64>,
}

impl JJChange {
    /// Create a new change
    pub fn new(file_path: String) -> Self {
        Self {
            file_path,
            status: ChangeStatus::Modified,
            is_staged: false,
            size_bytes: None,
        }
    }

    /// Get status as string
    pub fn status_str(&self) -> String {
        format!("{:?}", self.status)
    }
}

/// Status of a file change
#[derive(Debug, Serialize, Deserialize)]
#[napi(string_enum)]
pub enum ChangeStatus {
    /// File added
    Added,
    /// File modified
    Modified,
    /// File deleted
    Deleted,
    /// File renamed (old path stored separately in JJChange)
    Renamed,
    /// File conflicted
    Conflicted,
    /// File type changed
    TypeChanged,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jj_result() {
        let result = JJResult::new("output".into(), "".into(), 0, 100);
        assert!(result.success());
        assert_eq!(result.output(), "output");
    }

    #[test]
    fn test_commit_builder() {
        let commit = JJCommit::builder()
            .id("abc123".to_string())
            .change_id("xyz789".to_string())
            .message("Test commit".to_string())
            .author("Test Author".to_string())
            .author_email("test@example.com".to_string())
            .parent("parent1".to_string())
            .parent("parent2".to_string())
            .is_merge(true)
            .build();

        assert_eq!(commit.id, "abc123");
        assert_eq!(commit.message, "Test commit");
        assert_eq!(commit.parents.len(), 2);
        assert!(commit.is_merge);
    }

    #[test]
    fn test_branch_creation() {
        let mut branch = JJBranch::new("main".to_string(), "commit123".to_string(), false);
        assert_eq!(branch.name, "main");
        assert_eq!(branch.target, "commit123");
        assert!(!branch.is_remote);

        branch.set_remote("origin".to_string());
        assert_eq!(branch.full_name(), "origin/main");
    }

    #[test]
    fn test_conflict_builder() {
        let conflict = JJConflict::builder()
            .path("test.rs".to_string())
            .num_conflicts(2)
            .side("ours".to_string())
            .side("theirs".to_string())
            .conflict_type("content".to_string())
            .build();

        assert_eq!(conflict.path, "test.rs");
        assert_eq!(conflict.sides.len(), 2);
        assert_eq!(conflict.num_conflicts, 2);
        assert!(!conflict.is_binary);
    }

    #[test]
    fn test_diff_creation() {
        let mut diff = JJDiff::new();
        assert!(diff.is_empty());

        diff.added.push("new.txt".into());
        diff.modified.push("changed.txt".into());
        assert_eq!(diff.total_files_changed(), 2);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_change_creation() {
        let change = JJChange::new("test.rs".to_string());
        assert_eq!(change.file_path, "test.rs");
        assert!(matches!(change.status, ChangeStatus::Modified));
        assert!(!change.is_staged);
    }

    #[test]
    fn test_commit_add_parent() {
        let mut commit = JJCommit::new(
            "abc123".into(),
            "xyz789".into(),
            "Initial commit".into(),
            "Alice".into(),
            "alice@example.com".into(),
        );

        commit.add_parent("parent1".into());
        assert_eq!(commit.parent_count(), 1);
        assert!(!commit.is_merge);

        commit.add_parent("parent2".into());
        assert!(commit.is_merge);
    }

    #[test]
    fn test_conflict_id_unique() {
        let conflict1 = JJConflict::new("file1.rs".to_string(), 1, "content".to_string());
        let conflict2 = JJConflict::new("file1.rs".to_string(), 1, "content".to_string());
        assert_ne!(conflict1.id, conflict2.id);
    }
}
