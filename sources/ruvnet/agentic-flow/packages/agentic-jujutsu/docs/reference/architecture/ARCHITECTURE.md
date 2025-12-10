# Agentic-Jujutsu Architecture

## Overview

Agentic-Jujutsu is a WASM-enabled Rust wrapper for the Jujutsu version control system, designed for AI agent collaboration and learning. This document outlines the core architecture, type system, and API contracts.

## Architecture Decisions

### ADR-001: Dual Compilation Strategy

**Status**: Accepted

**Context**: The library needs to support both native Rust environments and WebAssembly for browser/Node.js execution.

**Decision**: Use conditional compilation with `cfg` attributes to provide platform-specific implementations while maintaining a unified public API.

**Consequences**:
- All public types must derive `Serialize`, `Deserialize` for cross-platform data exchange
- WASM-exposed types use `#[wasm_bindgen]` macro
- Complex types (enums with data, collections) use JSON string accessors for WASM
- Native features are gated behind `#[cfg(feature = "native")]`

### ADR-002: Builder Pattern for Complex Types

**Status**: Accepted

**Context**: Types like `JJCommit`, `JJConflict`, and `JJOperation` have many optional fields.

**Decision**: Implement builder patterns for all complex types to provide fluent, ergonomic construction.

**Consequences**:
- Better API usability and discoverability
- Type-safe construction with compile-time validation
- Consistent API across all major types

### ADR-003: Thread-Safe Operation Log

**Status**: Accepted

**Context**: The operation log needs to be accessible from multiple threads and potentially shared across async contexts.

**Decision**: Use `Arc<Mutex<Vec<JJOperation>>>` for the operation log storage.

**Consequences**:
- Thread-safe access to operation history
- Automatic reference counting for shared ownership
- Mutex contention only on log modifications (rare compared to reads)

## Type Hierarchy

```
JJWrapper (main interface)
├── JJConfig (configuration)
├── JJOperationLog (operation tracking)
│   └── JJOperation (single operation)
│       └── OperationType (enum: 30+ variants)
└── Repository Operations
    ├── JJCommit (commit metadata)
    │   ├── JJCommitBuilder
    │   └── children, parents, branches, tags
    ├── JJBranch (branch information)
    ├── JJConflict (conflict representation)
    │   ├── JJConflictBuilder
    │   └── ConflictMarker, ConflictSide
    ├── JJDiff (file differences)
    │   └── DiffHunk, DiffLine
    ├── JJChange (working copy changes)
    │   └── ChangeStatus (enum)
    └── JJResult (operation results)
```

## Core Types

### 1. JJResult

**Purpose**: Wrapper for jujutsu command execution results

**Fields**:
- `stdout`: Command output
- `stderr`: Error output
- `exit_code`: Process exit code
- `execution_time_ms`: Performance tracking

**Methods**:
- `success()` → bool
- `output()` → String
- `to_result()` → Result<String>

**WASM Compatibility**: Full

### 2. JJCommit

**Purpose**: Represent commit metadata with full history tracking

**Key Fields**:
- `id`: Revision hash (commit ID)
- `change_id`: Unique change identifier
- `message`: Commit message
- `author`, `author_email`: Author information
- `timestamp`: When committed (DateTime<Utc>)
- `parents`, `children`: Graph relationships
- `branches`, `tags`: Reference lists
- `is_merge`, `has_conflicts`, `is_empty`: Status flags

**Builder Pattern**:
```rust
let commit = JJCommit::builder()
    .id("abc123".to_string())
    .change_id("xyz789".to_string())
    .message("Initial commit".to_string())
    .author("Alice".to_string())
    .author_email("alice@example.com".to_string())
    .parent("parent_id".to_string())
    .branch("main".to_string())
    .build();
```

**WASM Accessors**:
- `parents_json()` → String
- `branches_json()` → String
- `timestamp_iso()` → String
- `short_id()` → String (first 12 chars)

### 3. JJBranch

**Purpose**: Branch information with remote tracking

**Fields**:
- `name`: Branch name
- `target`: Commit ID it points to
- `is_remote`: Whether it's a remote branch
- `remote`: Remote name (if applicable)
- `is_tracking`: Tracks remote branch
- `is_current`: Currently checked out
- `created_at`: Creation timestamp

**Methods**:
- `full_name()` → String (e.g., "origin/main")
- `short_target()` → String (first 12 chars of commit ID)

### 4. JJConflict

**Purpose**: Detailed conflict representation for AI-assisted resolution

**Fields**:
- `id`: Unique identifier (UUID)
- `path`: File path
- `num_conflicts`: Number of conflict regions
- `sides`: Conflicting versions
- `conflict_type`: "content", "modify/delete", etc.
- `is_binary`: Binary file conflict
- `is_resolved`: Resolution status
- `resolution_strategy`: How it was resolved

**Builder Pattern**:
```rust
let conflict = JJConflict::builder()
    .path("src/main.rs".to_string())
    .num_conflicts(1)
    .side("ours".to_string())
    .side("theirs".to_string())
    .conflict_type("content".to_string())
    .build();
```

### 5. JJDiff

**Purpose**: File differences between commits

**Fields**:
- `added`, `modified`, `deleted`: File lists
- `renamed`: Tuple list of (old_path, new_path)
- `additions`, `deletions`: Line counts
- `content`: Unified diff format

**Methods**:
- `total_files_changed()` → usize
- `is_empty()` → bool

### 6. JJChange

**Purpose**: Working copy changes (unstaged/staged)

**Fields**:
- `file_path`: Changed file
- `status`: ChangeStatus enum
- `is_staged`: Staging status
- `size_bytes`: File size (optional)

**ChangeStatus Enum**:
- `Added`, `Modified`, `Deleted`
- `Renamed { old_path: String }`
- `Conflicted`, `TypeChanged`

## Operation Tracking

### OperationType Enum (30+ variants)

**History-Modifying Operations**:
- `Commit`, `Describe`, `Edit`, `Abandon`
- `Rebase`, `Squash`, `Split`, `Move`, `Merge`

**Remote Operations**:
- `Fetch`, `GitFetch`, `Push`, `GitPush`
- `Clone`, `GitImport`, `GitExport`

**Repository Operations**:
- `Branch`, `BranchDelete`, `Tag`, `Checkout`
- `Restore`, `Duplicate`, `Undo`
- `New` (create commit), `Bookmark`

**Automatic**:
- `Snapshot` (jj's automatic snapshots)

**Methods on OperationType**:
- `modifies_history()` → bool
- `is_remote_operation()` → bool
- `is_automatic()` → bool
- `from_string(s: &str)` → OperationType

### JJOperation

**Purpose**: Single operation with full metadata and performance tracking

**Fields**:
- `id`: Wrapper-generated UUID
- `operation_id`: jj's operation ID
- `operation_type`: OperationType
- `command`: Command string
- `user`, `hostname`: Execution context
- `timestamp`: When executed
- `tags`: Custom tags (Vec<String>)
- `metadata`: Additional key-value pairs
- `parent_id`: Previous operation
- `duration_ms`: Execution time
- `success`: Result status
- `error`: Error message (if failed)

**Builder Pattern**:
```rust
let op = JJOperation::builder()
    .operation_type(OperationType::Commit)
    .command("jj commit -m 'message'".to_string())
    .user("alice".to_string())
    .hostname("localhost".to_string())
    .add_metadata("files_changed", "3")
    .duration_ms(1500)
    .build();
```

### JJOperationLog

**Purpose**: Thread-safe operation history with query capabilities

**Storage**: `Arc<Mutex<Vec<JJOperation>>>`

**Configuration**:
- `max_entries`: Automatic trimming when exceeded

**Query Methods**:

```rust
// Retrieval
fn get_recent(limit: usize) → Vec<JJOperation>
fn get_all() → Vec<JJOperation>
fn find_by_id(id: &str) → Option<JJOperation>
fn get_operation(id: &str) → Result<JJOperation>

// Filtering
fn filter_by_type(op_type: OperationType) → Vec<JJOperation>
fn filter_by_user(user: &str) → Vec<JJOperation>
fn filter_by_date_range(start: DateTime<Utc>, end: DateTime<Utc>) → Vec<JJOperation>
fn recent_operations(hours: i64) → Vec<JJOperation>
fn search(query: &str) → Vec<JJOperation>

// Specialized queries
fn failed_operations() → Vec<JJOperation>
fn history_modifying_operations() → Vec<JJOperation>
fn remote_operations() → Vec<JJOperation>
fn get_user_operations(limit: usize) → Vec<JJOperation>

// Statistics
fn statistics() → OperationStatistics
fn avg_duration_ms() → f64
fn success_rate() → f64
fn count() → usize
fn is_empty() → bool
```

### OperationStatistics

**Purpose**: Aggregated analytics about operations

**Fields**:
- `total`: Total count
- `successful`, `failed`: Status breakdown
- `by_type`: HashMap<OperationType, usize>
- `total_duration_ms`, `avg_duration_ms`, `max_duration_ms`: Performance metrics

## API Contracts

### Async Patterns

**Native** (using tokio):
```rust
#[cfg(feature = "native")]
pub async fn execute_operation(&self, command: &str) -> Result<JJResult>
```

**WASM** (using wasm-bindgen-futures):
```rust
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn execute_operation(&self, command: String) -> Promise
```

### Error Handling

All operations return `Result<T, JJError>` where `JJError` provides:
- `message()` → String (WASM-accessible)
- `is_recoverable()` → bool (WASM-accessible)

Error types:
- `JJNotFound`: jj not installed
- `CommandFailed`: Execution error
- `ParseError`: Output parsing failed
- `OperationNotFound`: Operation ID not found
- `ConflictResolutionFailed`: Conflict resolution error
- `InvalidConfig`: Configuration error
- `IoError`, `SerializationError`, `Unknown`

### WASM Bindings Strategy

**Problem**: WASM can't export complex Rust types directly

**Solution**: JSON string accessors for collections and enums

**Pattern**:
```rust
#[wasm_bindgen]
pub struct JJCommit {
    #[wasm_bindgen(skip)]
    pub parents: Vec<String>,  // Not accessible from JS
}

#[wasm_bindgen]
impl JJCommit {
    #[wasm_bindgen(getter)]
    pub fn parents_json(&self) -> String {  // Accessible from JS
        serde_json::to_string(&self.parents).unwrap_or_else(|_| "[]".to_string())
    }
}
```

**JavaScript Usage**:
```javascript
const commit = new JJCommit("id", "change_id", "message", "author", "email");
const parents = JSON.parse(commit.parents_json);
console.log(parents); // ["parent1", "parent2"]
```

## Performance Considerations

### Memory Management

- **Operation Log**: Automatic LRU eviction when exceeding `max_entries`
- **String Cloning**: Minimized through `&str` parameters and `Arc` for shared data
- **Builder Patterns**: Zero-cost abstractions (compiled away)

### Concurrency

- **Thread Safety**: All public types are `Send + Sync`
- **Lock Contention**: Mutex only held briefly during log operations
- **Async Runtime**: Tokio for native, browser event loop for WASM

### WASM Optimizations

- **Code Size**: Strip symbols, LTO enabled in release profile
- **Startup Time**: Lazy initialization of heavy resources
- **JSON Serialization**: Cached where possible for frequently accessed data

## Testing Strategy

### Unit Tests

All types and operations have comprehensive unit tests:
- Type construction and validation
- Builder patterns
- Query methods
- Edge cases (empty collections, invalid data)

**Coverage Target**: 85%+ for all modules

### Integration Tests

(To be implemented by integration agent)
- Full jj workflow simulation
- Multi-operation scenarios
- Error handling paths

### WASM Tests

(To be implemented by WASM agent)
- wasm-pack test in headless browser
- JavaScript interop validation
- Performance benchmarks

## Usage Examples

### Basic Commit Tracking

```rust
use agentic_jujutsu::{JJWrapper, JJConfig};

let config = JJConfig::default()
    .with_repo_path("/path/to/repo".to_string())
    .with_max_log_entries(500);

let wrapper = JJWrapper::new(config)?;
let log = wrapper.operation_log();

// Execute operation
wrapper.execute("jj commit -m 'Initial commit'").await?;

// Query recent commits
let commits = log.filter_by_type(OperationType::Commit);
for commit in commits {
    println!("{}: {}", commit.short_id(), commit.command);
}
```

### Conflict Detection

```rust
let conflicts = wrapper.detect_conflicts().await?;
for conflict in conflicts {
    println!("Conflict in {} ({} sides)",
        conflict.path,
        conflict.num_sides());

    if !conflict.is_binary {
        // AI can analyze and suggest resolution
        let suggestion = ai_resolve_conflict(&conflict);
        wrapper.resolve_conflict(&conflict.id, suggestion).await?;
    }
}
```

### Performance Analytics

```rust
let stats = log.statistics();
println!("Total operations: {}", stats.total);
println!("Success rate: {:.2}%", log.success_rate() * 100.0);
println!("Avg duration: {}ms", stats.avg_duration_ms);

// Find slow operations
let slow_ops: Vec<_> = log.get_all()
    .into_iter()
    .filter(|op| op.duration_ms > 1000)
    .collect();
```

## Future Enhancements

### Planned Features

1. **AgentDB Integration**: Persist operations to vector database for learning
2. **Conflict Resolution AI**: Train models on successful conflict resolutions
3. **Operation Replay**: Undo/redo with full state management
4. **Diff Analysis**: AI-powered diff summarization and review
5. **Performance Predictions**: ML-based estimation of operation duration

### API Stability

**Current Status**: v0.1.0 (unstable)

**Stability Guarantees**:
- Public API may change until v1.0.0
- Breaking changes will increment minor version
- Deprecation warnings will be provided one version before removal

## Contributing

### Adding New Types

1. Define struct with full documentation
2. Derive: `Debug, Clone, Serialize, Deserialize`
3. Add `#[wasm_bindgen]` if WASM-exposed
4. Implement builder pattern if > 3 optional fields
5. Add JSON accessors for complex fields (WASM)
6. Write comprehensive unit tests
7. Update this architecture document

### Adding New Operations

1. Add variant to `OperationType` enum
2. Update `from_string()` parser
3. Update classification methods (`modifies_history`, etc.)
4. Add tests for the new operation type
5. Document in this file

## References

- [Jujutsu Documentation](https://github.com/martinvonz/jj)
- [wasm-bindgen Guide](https://rustwasm.github.io/docs/wasm-bindgen/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

---

**Last Updated**: 2025-11-07
**Authors**: Architecture Agent, Agentic Flow Team
