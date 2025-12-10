name: "Rust PRP Template - Modern Rust with Tokio and Serde"
description: |

## Purpose
Template optimized for AI agents to implement Rust features in the rust-env using modern Rust patterns, async/await with Tokio, and comprehensive error handling.

## Core Principles
1. **Memory Safety**: Leverage Rust's ownership system effectively
2. **Async-First**: Use Tokio for async operations, avoid blocking calls
3. **Error Handling**: Result<T, E> pattern with custom error types
4. **Type Safety**: Strong typing with serde for serialization
5. **Performance**: Zero-cost abstractions and efficient patterns

---

## Goal
[What needs to be built - be specific about the Rust feature and async integration]

## Why
- [Business value and performance requirements]
- [Integration with existing Rust services]
- [Problems this solves and for whom]

## What
[User-visible behavior and system requirements, performance characteristics]

### Success Criteria
- [ ] [Specific measurable outcomes for Rust implementation]
- [ ] All Clippy lints passing without warnings
- [ ] Comprehensive test coverage with cargo test (80%+)
- [ ] Memory-safe implementation using ownership properly

## All Needed Context

### Target Environment
```yaml
Environment: rust-env
Devbox_Config: rust-env/devbox.json
Dependencies: [List required crates in Cargo.toml]
Rust_Version: Latest stable (as specified in devbox.json)
Async_Runtime: Tokio
Serialization: Serde with JSON support
```

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://tokio.rs/tokio/tutorial
  why: Async patterns and Tokio runtime usage
  
- file: rust-env/src/[existing_similar_module].rs
  why: Existing patterns to follow
  
- file: rust-env/Cargo.toml
  why: Dependency management and features
  
- doc: https://serde.rs/
  section: Serialization and deserialization patterns
  critical: Use serde derive macros properly
  
- doc: https://doc.rust-lang.org/book/
  section: Error handling and ownership patterns
```

### Current Codebase tree
```bash
rust-env/
├── Cargo.toml          # Dependencies: tokio, serde, anyhow, thiserror
├── src/
│   ├── main.rs         # Binary entry point
│   ├── lib.rs          # Library root
│   ├── types/          # Type definitions and structs
│   ├── services/       # Business logic layer
│   ├── utils/          # Utility functions
│   └── error.rs        # Custom error types
├── tests/
│   ├── integration/    # Integration tests
│   └── fixtures/       # Test data
├── examples/           # Usage examples
└── README.md
```

### Desired Codebase tree with files to be added
```bash
rust-env/
├── src/
│   ├── types/
│   │   └── feature.rs       # Feature-related types and structs
│   ├── services/
│   │   └── feature_service.rs  # Business logic implementation
│   ├── utils/
│   │   └── feature_utils.rs    # Helper functions (if needed)
│   └── api/
│       └── feature_api.rs      # API layer (if needed)
├── tests/
│   ├── integration/
│   │   └── feature_tests.rs
│   └── unit/
│       ├── feature_service_tests.rs
│       └── feature_utils_tests.rs
```

### Known Gotchas of Rust Environment
```rust
// CRITICAL: Rust environment-specific gotchas
// Ownership: Don't clone unnecessarily - use references where possible
// Async: All async functions must be .await'ed properly
// Error handling: Use Result<T, E> consistently, don't panic in production
// Tokio: Use #[tokio::main] or tokio::spawn for async execution
// Serde: Use #[derive(Serialize, Deserialize)] for JSON types
// Clippy: Address all clippy warnings, they improve code quality

// Example patterns:
// ✅ async fn process_data(data: &Data) -> Result<Output, Error>
// ✅ #[derive(Debug, Clone, Serialize, Deserialize)]
// ✅ let result = service.method().await?;
// ✅ use thiserror::Error for custom error types
// ❌ data.clone() when &data would work
// ❌ .unwrap() in production code - use proper error handling
```

## Implementation Blueprint

### Environment Setup
```bash
# Activate Rust environment
cd rust-env && devbox shell

# Verify environment
rustc --version
cargo --version

# Build to check dependencies
cargo check

# Add dependencies if needed
cargo add [new-crate-name]
```

### Type Definitions with Serde
```rust
// src/types/feature.rs
use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureStatus {
    Active,
    Inactive,
    Pending,
}

impl Default for FeatureStatus {
    fn default() -> Self {
        Self::Pending
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub status: FeatureStatus,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateFeatureRequest {
    pub name: String,
    pub description: Option<String>,
    pub status: Option<FeatureStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateFeatureRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub status: Option<FeatureStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFilters {
    pub status: Option<FeatureStatus>,
    pub name_contains: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

impl Default for FeatureFilters {
    fn default() -> Self {
        Self {
            status: None,
            name_contains: None,
            limit: Some(100),
            offset: Some(0),
        }
    }
}

// Custom error types
#[derive(thiserror::Error, Debug)]
pub enum FeatureError {
    #[error("Feature not found with id: {id}")]
    NotFound { id: Uuid },
    
    #[error("Invalid feature name: {reason}")]
    InvalidName { reason: String },
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type FeatureResult<T> = Result<T, FeatureError>;
```

### List of tasks to be completed
```yaml
Task 1: Environment Setup
  COMMAND: cd rust-env && devbox shell
  VERIFY: rustc --version && cargo --version
  BUILD: cargo check

Task 2: Type Definitions
  CREATE: rust-env/src/types/feature.rs
  PATTERN: Follow existing type patterns with serde derives
  VALIDATE: cargo check

Task 3: Custom Error Types
  MODIFY: rust-env/src/error.rs or create new error module
  PATTERN: Use thiserror crate for custom errors
  DERIVE: Debug, Error traits for all error types

Task 4: Service Implementation
  CREATE: rust-env/src/services/feature_service.rs
  PATTERN: Async methods returning Result<T, E>
  DEPS: Use dependency injection pattern
  ERRORS: Comprehensive error handling

Task 5: Utility Functions
  CREATE: rust-env/src/utils/feature_utils.rs (if needed)
  PATTERN: Pure functions with proper error handling
  VALIDATION: Input validation and sanitization

Task 6: Integration Points
  MODIFY: rust-env/src/lib.rs for public exports
  PATTERN: pub use statements for clean API
  DOCS: Add rustdoc comments

Task 7: Comprehensive Testing
  CREATE: Unit and integration tests
  PATTERN: #[cfg(test)] modules and separate test files
  COVERAGE: Use cargo-tarpaulin for coverage
  ASYNC: Use tokio::test for async tests

Task 8: Documentation and Examples
  UPDATE: README.md with new functionality
  CREATE: Example usage in examples/ directory
  DOCS: Comprehensive rustdoc documentation
```

### Per task pseudocode

**Task 4: Service Implementation**
```rust
// src/services/feature_service.rs
use crate::types::feature::{
    Feature, CreateFeatureRequest, UpdateFeatureRequest, 
    FeatureFilters, FeatureError, FeatureResult
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct FeatureService {
    features: Arc<RwLock<HashMap<Uuid, Feature>>>,
}

impl FeatureService {
    pub fn new() -> Self {
        Self {
            features: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create(&self, request: CreateFeatureRequest) -> FeatureResult<Feature> {
        // Validate input
        if request.name.trim().is_empty() {
            return Err(FeatureError::InvalidName {
                reason: "Name cannot be empty".to_string(),
            });
        }

        let now = SystemTime::now();
        let feature = Feature {
            id: Uuid::new_v4(),
            name: request.name.trim().to_string(),
            description: request.description.map(|d| d.trim().to_string()),
            status: request.status.unwrap_or_default(),
            created_at: now,
            updated_at: now,
        };

        // Store feature
        {
            let mut features = self.features.write().await;
            features.insert(feature.id, feature.clone());
        }

        Ok(feature)
    }

    pub async fn find_by_id(&self, id: Uuid) -> FeatureResult<Feature> {
        let features = self.features.read().await;
        features
            .get(&id)
            .cloned()
            .ok_or(FeatureError::NotFound { id })
    }

    pub async fn find_many(&self, filters: FeatureFilters) -> FeatureResult<Vec<Feature>> {
        let features = self.features.read().await;
        let mut results: Vec<Feature> = features.values().cloned().collect();

        // Apply status filter
        if let Some(status) = filters.status {
            results.retain(|f| f.status == status);
        }

        // Apply name filter
        if let Some(name_contains) = filters.name_contains {
            let search_term = name_contains.to_lowercase();
            results.retain(|f| f.name.to_lowercase().contains(&search_term));
        }

        // Apply pagination
        let offset = filters.offset.unwrap_or(0);
        let limit = filters.limit.unwrap_or(100);
        
        results = results.into_iter().skip(offset).take(limit).collect();

        Ok(results)
    }

    pub async fn update(
        &self, 
        id: Uuid, 
        request: UpdateFeatureRequest
    ) -> FeatureResult<Feature> {
        let mut features = self.features.write().await;
        
        let feature = features
            .get_mut(&id)
            .ok_or(FeatureError::NotFound { id })?;

        // Update fields if provided
        if let Some(name) = request.name {
            if name.trim().is_empty() {
                return Err(FeatureError::InvalidName {
                    reason: "Name cannot be empty".to_string(),
                });
            }
            feature.name = name.trim().to_string();
        }

        if let Some(description) = request.description {
            feature.description = Some(description.trim().to_string());
        }

        if let Some(status) = request.status {
            feature.status = status;
        }

        feature.updated_at = SystemTime::now();

        Ok(feature.clone())
    }

    pub async fn delete(&self, id: Uuid) -> FeatureResult<()> {
        let mut features = self.features.write().await;
        features
            .remove(&id)
            .ok_or(FeatureError::NotFound { id })?;
        Ok(())
    }
}

impl Default for FeatureService {
    fn default() -> Self {
        Self::new()
    }
}

// Trait for dependency injection
#[async_trait::async_trait]
pub trait FeatureServiceTrait: Send + Sync {
    async fn create(&self, request: CreateFeatureRequest) -> FeatureResult<Feature>;
    async fn find_by_id(&self, id: Uuid) -> FeatureResult<Feature>;
    async fn find_many(&self, filters: FeatureFilters) -> FeatureResult<Vec<Feature>>;
    async fn update(&self, id: Uuid, request: UpdateFeatureRequest) -> FeatureResult<Feature>;
    async fn delete(&self, id: Uuid) -> FeatureResult<()>;
}

#[async_trait::async_trait]
impl FeatureServiceTrait for FeatureService {
    async fn create(&self, request: CreateFeatureRequest) -> FeatureResult<Feature> {
        self.create(request).await
    }

    async fn find_by_id(&self, id: Uuid) -> FeatureResult<Feature> {
        self.find_by_id(id).await
    }

    async fn find_many(&self, filters: FeatureFilters) -> FeatureResult<Vec<Feature>> {
        self.find_many(filters).await
    }

    async fn update(&self, id: Uuid, request: UpdateFeatureRequest) -> FeatureResult<Feature> {
        self.update(id, request).await
    }

    async fn delete(&self, id: Uuid) -> FeatureResult<()> {
        self.delete(id).await
    }
}
```

**Task 7: Comprehensive Testing**
```rust
// tests/unit/feature_service_tests.rs
use rust_env::services::feature_service::FeatureService;
use rust_env::types::feature::{CreateFeatureRequest, FeatureStatus};
use uuid::Uuid;

#[tokio::test]
async fn test_create_feature_success() {
    // Arrange
    let service = FeatureService::new();
    let request = CreateFeatureRequest {
        name: "Test Feature".to_string(),
        description: Some("Test description".to_string()),
        status: Some(FeatureStatus::Active),
    };

    // Act
    let result = service.create(request).await;

    // Assert
    assert!(result.is_ok());
    let feature = result.unwrap();
    assert_eq!(feature.name, "Test Feature");
    assert_eq!(feature.status, FeatureStatus::Active);
    assert!(feature.description.is_some());
}

#[tokio::test]
async fn test_create_feature_empty_name() {
    // Arrange
    let service = FeatureService::new();
    let request = CreateFeatureRequest {
        name: "".to_string(),
        description: None,
        status: None,
    };

    // Act
    let result = service.create(request).await;

    // Assert
    assert!(result.is_err());
    match result.unwrap_err() {
        rust_env::types::feature::FeatureError::InvalidName { reason } => {
            assert!(reason.contains("empty"));
        }
        _ => panic!("Expected InvalidName error"),
    }
}

#[tokio::test]
async fn test_find_by_id_not_found() {
    // Arrange
    let service = FeatureService::new();
    let non_existent_id = Uuid::new_v4();

    // Act
    let result = service.find_by_id(non_existent_id).await;

    // Assert
    assert!(result.is_err());
    match result.unwrap_err() {
        rust_env::types::feature::FeatureError::NotFound { id } => {
            assert_eq!(id, non_existent_id);
        }
        _ => panic!("Expected NotFound error"),
    }
}

#[tokio::test]
async fn test_full_crud_workflow() {
    // Arrange
    let service = FeatureService::new();
    
    // Create
    let create_request = CreateFeatureRequest {
        name: "CRUD Test".to_string(),
        description: Some("Testing CRUD operations".to_string()),
        status: Some(FeatureStatus::Active),
    };
    
    let created = service.create(create_request).await.unwrap();
    let feature_id = created.id;
    
    // Read
    let found = service.find_by_id(feature_id).await.unwrap();
    assert_eq!(found.name, "CRUD Test");
    
    // Update
    let update_request = rust_env::types::feature::UpdateFeatureRequest {
        name: Some("Updated CRUD Test".to_string()),
        description: None,
        status: Some(FeatureStatus::Inactive),
    };
    
    let updated = service.update(feature_id, update_request).await.unwrap();
    assert_eq!(updated.name, "Updated CRUD Test");
    assert_eq!(updated.status, FeatureStatus::Inactive);
    
    // Delete
    service.delete(feature_id).await.unwrap();
    
    // Verify deletion
    let not_found = service.find_by_id(feature_id).await;
    assert!(not_found.is_err());
}
```

### Integration Points
```yaml
LIB_EXPORTS:
  - modify: rust-env/src/lib.rs
  - pattern: "pub use services::feature_service::FeatureService;"
  
DEPENDENCIES:
  - add to: rust-env/Cargo.toml
  - pattern: Use cargo add for new dependencies
  
CONFIG:
  - add to: rust-env/src/config.rs
  - pattern: Environment configuration with serde
  
BINARY:
  - modify: rust-env/src/main.rs (if applicable)
  - pattern: Tokio main function setup
```

## Validation Loop

### Level 1: Rust Syntax & Style
```bash
cd rust-env && devbox shell

# Format code with rustfmt
cargo fmt

# Check with Clippy (Rust linter)
cargo clippy -- -D warnings

# Type and borrow checking
cargo check

# Expected: No errors or warnings. Fix all clippy suggestions.
```

### Level 2: Unit Tests with Cargo
```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run tests with coverage (if cargo-tarpaulin installed)
cargo tarpaulin --out Html

# Expected: All tests pass, 80%+ coverage
```

### Level 3: Build and Integration Tests
```bash
# Build release version
cargo build --release

# Run integration tests specifically
cargo test --test integration

# Run examples (if any)
cargo run --example feature_example

# Benchmark tests (if applicable)
cargo bench

# Expected: Clean build, integration tests pass
```

## Final Validation Checklist
- [ ] Environment setup successful: `devbox shell` works
- [ ] Dependencies resolved: `cargo check` successful
- [ ] Code formatted: `cargo fmt` clean
- [ ] Clippy lints passed: `cargo clippy` clean
- [ ] All tests pass: `cargo test`
- [ ] Test coverage 80%+: `cargo tarpaulin`
- [ ] Release build successful: `cargo build --release`
- [ ] No unsafe code unless absolutely necessary
- [ ] Proper error handling with Result<T, E>
- [ ] Memory safety verified (no memory leaks)
- [ ] Async functions properly await'ed
- [ ] Documentation updated if needed

---

## Rust-Specific Anti-Patterns to Avoid
- ❌ Don't use `.unwrap()` in production code - handle errors properly
- ❌ Don't clone unnecessarily - use references and borrowing
- ❌ Don't ignore Clippy warnings - they improve code quality
- ❌ Don't use `unsafe` unless absolutely necessary and well-documented
- ❌ Don't block async runtime with synchronous operations
- ❌ Don't use `panic!` for recoverable errors - use Result
- ❌ Don't ignore lifetime annotations when needed
- ❌ Don't use `String` when `&str` would suffice

## Rust Best Practices
- ✅ Leverage the ownership system for memory safety
- ✅ Use Result<T, E> for all fallible operations
- ✅ Prefer borrowing over cloning when possible
- ✅ Use async/await consistently for I/O operations
- ✅ Write comprehensive documentation with `///` comments
- ✅ Use derive macros for common traits (Debug, Clone, etc.)
- ✅ Implement proper error types with thiserror
- ✅ Use type-driven development - let the compiler guide you
- ✅ Write tests that cover both success and error cases