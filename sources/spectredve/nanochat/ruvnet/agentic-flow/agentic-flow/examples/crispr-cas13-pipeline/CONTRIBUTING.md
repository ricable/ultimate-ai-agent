# Contributing to CRISPR-Cas13 Pipeline

Thank you for your interest in contributing! This project was built using [SPARC methodology](https://github.com/ruvnet/agentic-flow) and [rUv.io](https://ruv.io) AI coordination.

## 🚀 Quick Start

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/agentic-flow.git
   cd agentic-flow/agentic-flow/examples/crispr-cas13-pipeline
   ```

2. **Set up development environment**
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Install development tools
   cargo install cargo-tarpaulin  # Code coverage
   cargo install cargo-audit      # Security auditing
   cargo install cargo-machete    # Unused dependency detection

   # Build and test
   cargo build
   cargo test --all
   ```

## 📋 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Write Tests First (TDD)

We follow Test-Driven Development:

```rust
// tests/my_feature_test.rs
#[test]
fn test_my_new_feature() {
    let result = my_new_feature();
    assert_eq!(result.status, "success");
}
```

### 3. Implement Your Feature

```rust
// crates/my-crate/src/lib.rs
pub fn my_new_feature() -> Result<Output> {
    // Implementation here
    Ok(Output { status: "success" })
}
```

### 4. Run Quality Checks

```bash
# Run all tests
cargo test --all

# Check formatting
cargo fmt --all -- --check

# Run linter
cargo clippy -- -D warnings

# Security audit
cargo audit

# Check for unused dependencies
cargo machete

# Test coverage (minimum 80%)
cargo tarpaulin --out Html
```

### 5. Update Documentation

```bash
# Generate and check documentation
cargo doc --no-deps --open

# Update relevant markdown files
# - README.md (if public API changed)
# - CHANGELOG.md (add your changes)
# - docs/ (if architecture/algorithms changed)
```

### 6. Commit Your Changes

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git add .
git commit -m "feat: add new off-target scoring algorithm"
# or
git commit -m "fix: correct DESeq2 normalization edge case"
# or
git commit -m "docs: update API documentation for immune analyzer"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding tests
- `chore`: Maintenance tasks

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## 🧪 Testing Guidelines

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_name() {
        // Arrange
        let input = setup_test_data();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result.expected_field, expected_value);
    }
}
```

### Integration Tests

```rust
// tests/integration_test.rs
use crispr_cas13_pipeline::prelude::*;

#[tokio::test]
async fn test_complete_workflow() {
    let target = create_test_target();
    let predictor = DefaultPredictor::new(Default::default()).unwrap();
    let prediction = predictor.predict(&target).await.unwrap();
    assert!(!prediction.off_targets.is_empty());
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_score_normalization(score in 0.0f64..1.0f64) {
        let normalized = normalize_score(score);
        assert!(normalized >= 0.0 && normalized <= 1.0);
    }
}
```

### Benchmark Tests

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn alignment_benchmark(c: &mut Criterion) {
    c.bench_function("align 1000 reads", |b| {
        let reads = generate_test_reads(1000);
        b.iter(|| align_reads(black_box(&reads)))
    });
}

criterion_group!(benches, alignment_benchmark);
criterion_main!(benches);
```

## 📝 Code Standards

### Style Guide

- Use `rustfmt` for formatting (enforced by CI)
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Maximum line length: 100 characters
- Use meaningful variable names (no single-letter variables except in math formulas)

### Documentation

```rust
/// Calculate CFD (Cutting Frequency Determination) score for off-target sites.
///
/// # Arguments
///
/// * `guide` - Guide RNA sequence (22-30nt)
/// * `target` - Target RNA sequence to score
///
/// # Returns
///
/// Returns a score between 0.0 (no match) and 1.0 (perfect match).
///
/// # Examples
///
/// ```
/// use crispr_cas13_pipeline::scoring::CfdScorer;
///
/// let score = CfdScorer::score("ACGUACGU", "ACGUACGU");
/// assert_eq!(score, 1.0);
/// ```
///
/// # References
///
/// Doench et al. (2016) Nature Biotechnology 34(2)
pub fn score(guide: &str, target: &str) -> f64 {
    // Implementation
}
```

### Error Handling

Use `thiserror` for custom errors:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PredictionError {
    #[error("Invalid guide RNA length: {0} (expected 22-30nt)")]
    InvalidGuideLength(usize),

    #[error("ML model not found at path: {0}")]
    ModelNotFound(String),

    #[error("Prediction failed: {0}")]
    PredictionFailed(#[from] anyhow::Error),
}
```

### Performance

- Use `rayon` for CPU-bound parallel work
- Use `tokio` for I/O-bound async work
- Minimize allocations (use `&str` over `String` when possible)
- Benchmark critical paths with `criterion`

## 🏗️ Architecture Guidelines

### Crate Organization

Each crate should have:
- Clear single responsibility
- Well-defined public API
- Comprehensive tests (>80% coverage)
- Examples in `examples/` directory

### Adding a New Crate

```bash
# Create new crate
cargo new --lib crates/my-new-crate

# Add to workspace in root Cargo.toml
[workspace]
members = [
    "crates/data-models",
    # ... existing crates
    "crates/my-new-crate",  # Add here
]
```

### Dependencies

- Minimize external dependencies
- Use well-maintained crates only
- Lock versions for stability
- Document why each dependency is needed

## 🤝 Code Review Process

### As a Contributor

- Respond to review comments promptly
- Make requested changes in new commits (don't force push)
- Mark conversations as resolved when addressed
- Be open to feedback and suggestions

### As a Reviewer

- Be respectful and constructive
- Focus on code quality, not personal preferences
- Provide specific suggestions for improvement
- Approve when all concerns are addressed

## 🐛 Bug Reports

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md):

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g. Ubuntu 22.04]
- Rust version: [e.g. 1.75.0]
- Package version: [e.g. 0.1.0]
```

## ✨ Feature Requests

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md):

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 📚 Resources

### Documentation
- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [SPARC Methodology](https://github.com/ruvnet/agentic-flow)
- [rUv.io Platform](https://ruv.io)

### Scientific References
- [CRISPR-Cas13 Overview](https://www.nature.com/articles/s41576-018-0059-1)
- [Off-Target Analysis](https://www.nature.com/articles/nbt.3437)
- [DESeq2 Algorithm](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8)

## 🎯 Current Priorities

See [GitHub Projects](https://github.com/ruvnet/agentic-flow/projects) for current roadmap.

High-priority areas:
- [ ] GPU acceleration for ML models
- [ ] Additional Cas13 variant support
- [ ] Frontend UI development
- [ ] Real-world validation studies

## 💬 Communication

- **GitHub Discussions**: For questions and general discussion
- **Discord**: [Join our community](https://discord.gg/ruv)
- **Email**: contribute@ruv.io

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CRISPR-Cas13 Pipeline! 🧬✨
