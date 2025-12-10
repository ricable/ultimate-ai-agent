# ADR-002: Error Handling Strategy

## Status
Accepted

## Context
Need a consistent error handling strategy across all crates that provides:
1. Type safety
2. Error context
3. User-friendly messages
4. Easy error propagation

## Decision
Use a hybrid approach:
- **Library crates** (core, data, models, physics): Return `Result<T, ClimateError>`
- **Application crates** (api, cli): Use `anyhow::Result<T>` for flexibility

Define `ClimateError` as an enum with `thiserror`:
```rust
#[derive(Error, Debug)]
pub enum ClimateError {
    #[error("Data source error: {0}")]
    DataSource(String),

    #[error("Model inference error: {0}")]
    ModelInference(String),

    // ... other variants
}
```

## Rationale

### Why thiserror for libraries?
- Zero-cost error types
- Automatic `std::error::Error` impl
- Clear error variants for callers
- No dynamic dispatch overhead

### Why anyhow for applications?
- Easy error context (`context()` method)
- Downcasting support
- Simple error propagation
- User-facing error messages

### Error Conversion
Libraries expose strongly-typed errors, applications convert to anyhow:
```rust
// Library
pub fn predict(...) -> Result<Prediction, ClimateError> { }

// Application
pub async fn handle_request() -> anyhow::Result<Response> {
    let pred = predict(...).context("Failed to generate prediction")?;
    Ok(Response::from(pred))
}
```

## Alternatives Considered

### anyhow Everywhere
**Rejected**: Loses type information, makes error handling opaque for library users

### thiserror Everywhere
**Rejected**: Too verbose in application code, difficult to add context

### Custom Error Trait
**Rejected**: Unnecessary complexity, standard error traits sufficient

## Consequences

### Positive
- Clear error boundaries between layers
- Type-safe error handling in libraries
- Flexible error context in applications
- Good error messages for users

### Negative
- Two error handling patterns to learn
- Need to convert between ClimateError and anyhow

## Implementation Guidelines

### Library Code
```rust
// Define domain-specific errors
#[derive(Error, Debug)]
pub enum DataError {
    #[error("API rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid API key")]
    InvalidApiKey,
}

// Convert to ClimateError
impl From<DataError> for ClimateError {
    fn from(err: DataError) -> Self {
        ClimateError::DataSource(err.to_string())
    }
}
```

### Application Code
```rust
// Add context to errors
let data = fetch_data()
    .await
    .context("Failed to fetch weather data")?;

// Convert ClimateError to anyhow
let pred = model.predict(&request)
    .await
    .map_err(|e| anyhow!("Prediction failed: {}", e))?;
```

## Related Decisions
- ADR-001: Crate structure (determines error boundaries)
