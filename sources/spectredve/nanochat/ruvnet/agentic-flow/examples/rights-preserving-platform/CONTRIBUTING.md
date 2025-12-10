# Contributing to Rights-Preserving Countermeasure Platform

Thank you for your interest in contributing to the Rights-Preserving Countermeasure Platform! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Security](#security)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

---

## Getting Started

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Git 2.30+
- Docker 24.0+ (for integration testing)
- PostgreSQL 15+ and Redis 7.0+ (for local development)

### Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/agentic-flow.git
cd agentic-flow/examples/rights-preserving-platform

# Add upstream remote
git remote add upstream https://github.com/ruvnet/agentic-flow.git
```

### Build and Test

```bash
# Build all modules
cargo build

# Run all tests
cargo test

# Run benchmarks
cargo bench

# Check code quality
cargo clippy -- -D warnings
cargo fmt --all -- --check
```

---

## Development Workflow

### 1. Create a Feature Branch

```bash
# Fetch latest changes
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write tests first (TDD approach)
- Implement your feature
- Ensure all tests pass
- Update documentation

### 3. Commit Your Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git add .
git commit -m "feat: add bias detection for intersectionality"

# Commit message format:
# <type>(<scope>): <subject>
#
# Types: feat, fix, docs, style, refactor, test, chore
# Scope: api, privacy, governance, audit, etc.
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
# Open Pull Request on GitHub
```

---

## Code Standards

### Rust Style Guide

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Organization

```rust
// Good: Clear separation of concerns
pub mod api {
    pub mod gateway;
    pub mod middleware;
}

pub mod privacy {
    pub mod differential;
    pub mod anonymization;
}

// Bad: Everything in one file
pub mod everything;
```

### Error Handling

```rust
// Good: Custom error types with context
#[derive(Debug, thiserror::Error)]
pub enum PrivacyError {
    #[error("Privacy budget exceeded: used {used}, limit {limit}")]
    BudgetExceeded { used: f64, limit: f64 },

    #[error("Invalid epsilon value: {0}")]
    InvalidEpsilon(f64),
}

// Bad: Generic errors
return Err("Something went wrong");
```

### Documentation

```rust
/// Computes differentially private mean with Laplace mechanism.
///
/// # Arguments
///
/// * `data` - Input data points
/// * `epsilon` - Privacy budget (ε)
/// * `sensitivity` - Query sensitivity
///
/// # Returns
///
/// Private mean with noise added
///
/// # Errors
///
/// Returns `PrivacyError::BudgetExceeded` if epsilon is too large.
///
/// # Examples
///
/// ```
/// use rights_preserving_platform::privacy::DifferentialPrivacy;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let private_mean = DifferentialPrivacy::compute_mean(&data, 1.0, 1.0)?;
/// ```
pub fn compute_mean(data: &[f64], epsilon: f64, sensitivity: f64) -> Result<f64> {
    // Implementation
}
```

---

## Testing Guidelines

### Test Coverage

- **Minimum 80% code coverage** required
- Unit tests for all public functions
- Integration tests for API endpoints
- Property-based tests for critical algorithms

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differential_privacy_laplace() {
        let config = DifferentialPrivacyConfig {
            epsilon: 1.0,
            delta: 1e-5,
            mechanism: Mechanism::Laplace,
            sensitivity: 1.0,
        };

        let privacy = PrivacyEngine::new(config).unwrap();
        let data = vec![100.0, 200.0, 150.0, 175.0];
        let private_mean = privacy.compute_mean(&data).unwrap();

        // Mean should be close to 156.25 (with noise)
        assert!((private_mean - 156.25).abs() < 50.0);
    }

    #[test]
    #[should_panic(expected = "BudgetExceeded")]
    fn test_privacy_budget_enforcement() {
        let config = DifferentialPrivacyConfig {
            epsilon: 10.0, // Too large!
            ..Default::default()
        };

        PrivacyEngine::new(config).unwrap();
    }
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_audit_chain_integrity(events in prop::collection::vec(any::<AuditEvent>(), 1..100)) {
        let logger = AuditLogger::new().unwrap();

        for event in &events {
            logger.log(event).await.unwrap();
        }

        // Property: Chain must always be valid
        let is_valid = logger.verify_chain(0, events.len()).await.unwrap();
        prop_assert!(is_valid);
    }
}
```

### Benchmarking

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_policy_evaluation(c: &mut Criterion) {
    let policy_engine = PolicyEngine::new("http://localhost:8181").unwrap();
    let input = PolicyInput { /* ... */ };

    c.bench_function("policy_evaluation", |b| {
        b.iter(|| {
            policy_engine.evaluate(black_box("data_access"), black_box(&input))
        });
    });
}

criterion_group!(benches, benchmark_policy_evaluation);
criterion_main!(benches);
```

---

## Documentation

### Documentation Requirements

- All public APIs must have rustdoc comments
- Examples in documentation must compile and run
- Architecture documents for major features
- Update README.md for user-facing changes

### Writing Documentation

```bash
# Generate and view documentation
cargo doc --open

# Test documentation examples
cargo test --doc
```

### Architecture Decision Records (ADRs)

For significant architectural changes, create an ADR:

```markdown
# ADR 001: Use SmartNoise for Differential Privacy

## Status

Accepted

## Context

We need a production-ready differential privacy library for the platform.

## Decision

Use Microsoft's SmartNoise framework with Rust bindings.

## Consequences

- Pros: Battle-tested, NIST competition winner, active development
- Cons: Requires Python runtime for some advanced features
```

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`cargo test`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation updated
- [ ] Changelog entry added (if applicable)
- [ ] Security audit clean (`cargo audit`)

### PR Template

```markdown
## Description

Brief description of changes

## Motivation

Why is this change needed?

## Changes

- List of changes made
- With bullet points

## Testing

How was this tested?

## Checklist

- [ ] Tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Security reviewed
```

### Review Process

1. Automated checks (CI/CD) must pass
2. At least one maintainer approval required
3. All conversations must be resolved
4. Squash and merge (maintainers will handle)

---

## Security

### Reporting Vulnerabilities

**DO NOT** open a public issue for security vulnerabilities.

Email security@ruv.io with:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Best Practices

- Never commit secrets (API keys, passwords)
- Use `cargo audit` before each commit
- Follow OWASP guidelines for web security
- Implement defense in depth
- Validate all user inputs
- Use strong cryptography (no custom crypto!)

### Security Review Checklist

- [ ] Input validation implemented
- [ ] Authentication and authorization checked
- [ ] Sensitive data encrypted
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] CSRF protection
- [ ] Rate limiting configured
- [ ] Audit logging enabled

---

## Community

### Getting Help

- **Discord**: [Join our Discord](https://discord.gg/ruv)
- **GitHub Discussions**: [Discussions](https://github.com/ruvnet/agentic-flow/discussions)
- **Email**: support@ruv.io

### Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- Release notes
- Project documentation

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Rights-Preserving Countermeasure Platform! 🛡️
