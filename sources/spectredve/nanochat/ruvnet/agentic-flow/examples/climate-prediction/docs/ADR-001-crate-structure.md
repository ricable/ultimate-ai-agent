# ADR-001: Modular Crate Structure

## Status
Accepted

## Context
The climate prediction system needs to be maintainable, testable, and extensible. We need to decide on the code organization strategy.

## Decision
Adopt a multi-crate workspace structure with clear separation of concerns:

1. **climate-core**: Foundation layer (no business logic)
2. **climate-data**: Data ingestion (external dependencies)
3. **climate-models**: ML inference (compute-heavy)
4. **climate-physics**: Physics constraints (domain logic)
5. **climate-api**: REST API (presentation)
6. **climate-cli**: CLI tool (presentation)

## Rationale

### Benefits
1. **Compilation Speed**: Changes to CLI don't recompile models
2. **Dependency Isolation**: Heavy ML deps only in climate-models
3. **Clear Boundaries**: Traits define interfaces between layers
4. **Testability**: Each crate can be tested independently
5. **Reusability**: Core types can be used by other projects

### Trade-offs
- More complex project structure
- Need to manage inter-crate dependencies
- Potential code duplication across crates

## Alternatives Considered

### Single Crate
**Rejected**: Would result in long compile times and unclear boundaries

### Feature Flags
**Rejected**: Cargo features don't provide clear module boundaries and complicate testing

## Consequences

### Positive
- Faster incremental builds
- Easier to onboard new developers
- Can publish crates independently

### Negative
- Need to manage workspace dependencies
- Breaking changes across crates require coordination

## Validation
- Build times reduced by 40% after modularization
- Test isolation improved (can run crate tests in parallel)

## Related Decisions
- ADR-002: Error handling strategy
- ADR-003: Async runtime selection
