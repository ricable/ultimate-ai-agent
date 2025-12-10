# Testing Best Practices for Agentic-Jujutsu

## Philosophy

Testing is not just about catching bugs—it's about:
- **Documentation:** Tests document how the code should behave
- **Confidence:** Tests enable fearless refactoring
- **Design:** Test-driven development leads to better APIs
- **Regression Prevention:** Tests prevent old bugs from returning

## Test-Driven Development (TDD)

### The Red-Green-Refactor Cycle

1. **Red:** Write a failing test
2. **Green:** Write minimal code to pass the test
3. **Refactor:** Improve the code while keeping tests green

### Example TDD Workflow

```rust
// 1. RED: Write failing test
#[test]
fn test_operation_filter_by_type() {
    let mut log = JJOperationLog::new(10);
    let mut op = JJOperation::new("id".into(), "user".into(), "desc".into());
    op.op_type = OperationType::Commit;
    log.add_operation(op);

    let commits = log.filter_by_type(OperationType::Commit);
    assert_eq!(commits.len(), 1); // FAILS - method doesn't exist
}

// 2. GREEN: Implement minimal solution
impl JJOperationLog {
    pub fn filter_by_type(&self, op_type: OperationType) -> Vec<&JJOperation> {
        self.operations
            .iter()
            .filter(|op| op.op_type == op_type)
            .collect()
    }
}

// 3. REFACTOR: Optimize if needed
// Already good for this simple case
```

## Test Pyramid

```
         /\
        /E2E\      <- Few (5-10%)
       /------\    Integration tests
      /  Int.  \   <- Some (20-30%)
     /----------\
    /   Unit     \ <- Many (60-70%)
   /--------------\
```

### Unit Tests (60-70%)
- Fast (< 1ms)
- Isolated
- Test one thing
- No external dependencies

### Integration Tests (20-30%)
- Medium speed (10-100ms)
- Test component interactions
- May use real dependencies
- Test workflows

### End-to-End Tests (5-10%)
- Slow (>100ms)
- Test complete user scenarios
- Use real systems
- Critical paths only

## Test Patterns

### AAA Pattern (Arrange-Act-Assert)

```rust
#[test]
fn test_operation_log_max_entries() {
    // Arrange
    let mut log = JJOperationLog::new(3);

    // Act
    for i in 0..5 {
        log.add_operation(JJOperation::new(
            format!("id{}", i),
            "user".into(),
            "desc".into(),
        ));
    }

    // Assert
    assert_eq!(log.len(), 3);
    assert!(log.get_by_id("id0").is_none()); // Oldest removed
    assert!(log.get_by_id("id4").is_some()); // Newest kept
}
```

### Given-When-Then (BDD Style)

```rust
#[test]
fn test_conflict_resolution() {
    // Given a conflict exists
    let mut conflict = JJConflict::new(
        "file.rs".into(),
        "merge conflict".into(),
        2,
    );
    assert!(!conflict.is_resolved());

    // When we mark it as resolved
    conflict.mark_resolved();

    // Then it should be resolved
    assert!(conflict.is_resolved());
}
```

### Builder Pattern for Test Data

```rust
struct TestOperationBuilder {
    id: String,
    user: String,
    description: String,
    op_type: OperationType,
}

impl TestOperationBuilder {
    fn new() -> Self {
        Self {
            id: "test_id".into(),
            user: "test@example.com".into(),
            description: "test description".into(),
            op_type: OperationType::Unknown,
        }
    }

    fn with_type(mut self, op_type: OperationType) -> Self {
        self.op_type = op_type;
        self
    }

    fn build(self) -> JJOperation {
        let mut op = JJOperation::new(self.id, self.user, self.description);
        op.op_type = self.op_type;
        op
    }
}

#[test]
fn test_with_builder() {
    let op = TestOperationBuilder::new()
        .with_type(OperationType::Commit)
        .build();

    assert_eq!(op.op_type, OperationType::Commit);
}
```

## Property-Based Testing

### Why Property-Based Testing?

Traditional tests check specific examples. Property-based tests verify invariants hold for ALL inputs.

### Common Properties

1. **Serialization Round-Trip**
```rust
proptest! {
    #[test]
    fn test_roundtrip(value in strategy()) {
        let serialized = serialize(value);
        let deserialized = deserialize(serialized);
        assert_eq!(value, deserialized);
    }
}
```

2. **Idempotence**
```rust
proptest! {
    #[test]
    fn test_idempotent(input in strategy()) {
        let result1 = function(input);
        let result2 = function(input);
        assert_eq!(result1, result2);
    }
}
```

3. **Inverse Functions**
```rust
proptest! {
    #[test]
    fn test_inverse(value in strategy()) {
        assert_eq!(value, decode(encode(value)));
    }
}
```

4. **Invariants**
```rust
proptest! {
    #[test]
    fn test_bounded_size(max in 1usize..1000, count in 0usize..2000) {
        let mut log = JJOperationLog::new(max);
        for i in 0..count {
            log.add_operation(create_op(i));
        }
        assert!(log.len() <= max); // INVARIANT
    }
}
```

## WASM Testing

### Testing Strategy

1. **Unit Tests:** Test Rust logic (no WASM needed)
2. **WASM Bindings:** Test JS interop
3. **Browser Tests:** Test in actual browser environment

### WASM-Specific Concerns

```rust
#[wasm_bindgen_test]
fn test_wasm_memory_management() {
    let mut objects = vec![];

    // Create many objects
    for i in 0..1000 {
        objects.push(JJOperation::new(
            format!("id{}", i),
            "user".into(),
            "desc".into(),
        ));
    }

    // Drop them
    objects.clear();

    // Should not leak memory (verified by browser tools)
}
```

## Performance Testing

### Benchmarking Guidelines

1. **Use Criterion:** Provides statistical analysis
2. **Warm-up:** Run iterations before measuring
3. **Baseline:** Compare against previous runs
4. **Isolate:** Test one thing at a time

### Benchmark Example

```rust
fn benchmark_operation_creation(c: &mut Criterion) {
    c.bench_function("create_operation", |b| {
        b.iter(|| {
            JJOperation::new(
                black_box("id".to_string()),
                black_box("user".to_string()),
                black_box("desc".to_string()),
            )
        });
    });
}
```

### Performance Targets

| Operation | Target | Max |
|-----------|--------|-----|
| Operation creation | < 100ns | < 500ns |
| Log add | < 200ns | < 1μs |
| Log search (1000 items) | < 10μs | < 50μs |
| Serialization | < 1μs | < 10μs |

## Test Organization

### File Structure

```
tests/
├── unit/              # White-box testing
│   └── module_name.rs # One file per module
├── integration/       # Black-box testing
│   └── feature_name.rs # One file per feature
├── property/          # Property-based tests
│   └── type_properties.rs
├── wasm/              # WASM-specific
└── mocks/             # Shared test utilities
```

### Naming Conventions

```rust
// Good names (describe behavior)
test_operation_log_removes_oldest_when_full()
test_conflict_starts_unresolved()
test_serialization_preserves_all_fields()

// Bad names (implementation details)
test_function_1()
test_vec_push()
test_loop()
```

## Coverage Goals

### Module Targets

- **Core types:** 95%+
- **Configuration:** 95%+
- **Operations:** 90%+
- **Wrappers:** 85%+
- **WASM bindings:** 80%+

### What to Cover

✅ **Do cover:**
- Public APIs
- Error paths
- Edge cases
- Critical business logic

❌ **Don't stress about:**
- Trivial getters/setters
- Generated code
- Debug implementations
- Test utilities themselves

## Common Pitfalls

### 1. Testing Implementation Instead of Behavior

```rust
// BAD: Tests implementation details
#[test]
fn test_uses_vec_internally() {
    let log = JJOperationLog::new(10);
    assert_eq!(log.operations.capacity(), 10); // Implementation detail
}

// GOOD: Tests behavior
#[test]
fn test_respects_max_entries() {
    let mut log = JJOperationLog::new(10);
    for i in 0..20 {
        log.add_operation(create_op(i));
    }
    assert_eq!(log.len(), 10); // Observable behavior
}
```

### 2. Overly Complex Tests

```rust
// BAD: Too complex
#[test]
fn test_everything() {
    let mut log = JJOperationLog::new(10);
    log.add_operation(op1);
    assert_eq!(log.len(), 1);
    log.add_operation(op2);
    assert_eq!(log.len(), 2);
    let filtered = log.filter_by_type(Commit);
    assert_eq!(filtered.len(), 1);
    // ... 20 more assertions
}

// GOOD: One test, one concept
#[test]
fn test_add_increases_length() {
    let mut log = JJOperationLog::new(10);
    log.add_operation(create_op());
    assert_eq!(log.len(), 1);
}
```

### 3. Brittle Tests

```rust
// BAD: Depends on exact timestamp
#[test]
fn test_operation_timestamp() {
    let op = JJOperation::new(/*...*/);
    assert_eq!(op.timestamp, "2024-01-01T00:00:00Z"); // Brittle!
}

// GOOD: Tests property
#[test]
fn test_operation_has_valid_timestamp() {
    let op = JJOperation::new(/*...*/);
    let iso = op.timestamp_iso();
    assert!(iso.contains('T'));
    assert!(iso.len() > 20);
}
```

## Debugging Failed Tests

### 1. Use `--nocapture`
```bash
cargo test test_name -- --nocapture
```

### 2. Add Debug Output
```rust
#[test]
fn test_debug_example() {
    let result = complex_function();
    eprintln!("Result: {:?}", result); // Visible with --nocapture
    assert_eq!(result, expected);
}
```

### 3. Run Single-Threaded
```bash
cargo test -- --test-threads=1
```

### 4. Minimize Test Case
```rust
// Start with failing case
#[test]
fn test_failing() {
    // Large complex setup
    assert!(false);
}

// Reduce to minimal reproducer
#[test]
fn test_minimal() {
    let x = 1;
    assert_eq!(x, 2); // Still fails, but simpler
}
```

## Continuous Improvement

### Test Metrics to Track

1. **Coverage Percentage:** Aim for >90%
2. **Test Count:** Should grow with codebase
3. **Test Speed:** Keep unit tests fast (<1ms)
4. **Flakiness:** Zero flaky tests
5. **Property Test Cases:** Increase over time

### Refactoring Tests

Tests need refactoring too! Signs you need to refactor:
- Duplicated setup code → Extract helper functions
- Long test files → Split by feature
- Complex assertions → Extract assertion helpers
- Slow tests → Optimize or move to integration suite

## Resources

- [Rust Book - Testing](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Proptest Book](https://altsysrq.github.io/proptest-book/)
- [Criterion Guide](https://bheisler.github.io/criterion.rs/book/)
- [WASM Testing](https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/)

## Conclusion

Good tests are:
- **Fast** - Run quickly for tight feedback loops
- **Isolated** - Independent of each other
- **Repeatable** - Same result every time
- **Self-validating** - Pass/fail is clear
- **Thorough** - Cover edge cases and errors

Remember: **Tests are code too.** Give them the same care as production code.
