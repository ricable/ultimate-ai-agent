# ADR-003: Async Runtime Selection

## Status
Accepted

## Context
The system requires async I/O for:
- HTTP requests to data sources
- API server request handling
- Concurrent model inference
- Database operations (future)

Need to choose between Tokio, async-std, or smol.

## Decision
Use **Tokio** as the async runtime for all crates.

## Rationale

### Why Tokio?

**Ecosystem Maturity**:
- Most popular async runtime (used by 70% of Rust projects)
- Best ecosystem support (reqwest, hyper, axum all use Tokio)
- Extensive documentation and examples

**Performance**:
- Excellent multi-threaded scheduler
- Work-stealing algorithm
- Low overhead for I/O operations

**Features**:
- `tokio::fs` for async file I/O
- `tokio::time` for timers
- `tokio::sync` for async primitives
- Console for runtime inspection

**Production Ready**:
- Used by major companies (AWS, Discord, Cloudflare)
- Stable API (1.0+ since 2020)
- Active maintenance

## Configuration

### Tokio Features
```toml
tokio = { version = "1.40", features = [
    "rt-multi-thread",  # Multi-threaded runtime
    "macros",           # #[tokio::main] and #[tokio::test]
    "io-util",          # I/O utilities
    "net",              # Networking (TCP, UDP)
    "time",             # Timers and delays
    "sync",             # Async primitives
    "fs",               # Async file operations
] }
```

### Runtime Configuration
```rust
// Default: multi-threaded runtime
#[tokio::main]
async fn main() {
    // Application code
}

// Custom configuration for API server
#[tokio::main(worker_threads = 8)]
async fn main() {
    // API server with 8 worker threads
}
```

## Alternatives Considered

### async-std
**Pros**:
- Mirrors std library API
- Simpler learning curve

**Cons**:
- Smaller ecosystem
- Less active development
- Fewer advanced features

**Rejected**: Ecosystem support not as strong

### smol
**Pros**:
- Lightweight (<1000 LOC)
- Simple implementation
- Good for embedded systems

**Cons**:
- Limited ecosystem
- Manual executor management
- Less production-tested

**Rejected**: Not mature enough for production

### Runtime Abstraction (async-trait)
**Pros**:
- Runtime agnostic
- Maximum flexibility

**Cons**:
- Adds complexity
- Performance overhead from trait objects
- No clear benefit for our use case

**Rejected**: Unnecessary abstraction

## Consequences

### Positive
- Best ecosystem support
- Excellent performance
- Rich feature set
- Production-proven

### Negative
- Larger binary size than smol
- Tokio-specific APIs (harder to switch later)
- Some learning curve for advanced features

## Implementation Guidelines

### Async Functions
```rust
// Use async-trait for trait methods
use async_trait::async_trait;

#[async_trait]
pub trait DataSource {
    async fn fetch(&self) -> Result<Data>;
}
```

### Error Handling
```rust
// Handle errors in async code
async fn fetch_with_retry() -> Result<Data> {
    for attempt in 1..=3 {
        match fetch_data().await {
            Ok(data) => return Ok(data),
            Err(e) if attempt < 3 => {
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}
```

### Concurrency
```rust
// Spawn concurrent tasks
let tasks = locations.into_iter()
    .map(|loc| tokio::spawn(predict_for_location(loc)))
    .collect::<Vec<_>>();

let results = futures::future::join_all(tasks).await;
```

### Timeouts
```rust
use tokio::time::{timeout, Duration};

// Add timeout to operations
let result = timeout(
    Duration::from_secs(30),
    fetch_data()
).await??;
```

## Performance Considerations

### Thread Pool Sizing
- Default: number of CPU cores
- I/O heavy: Can use more threads (2x cores)
- CPU heavy: Use thread pool (1x cores)

### Blocking Operations
```rust
// Use spawn_blocking for CPU-heavy work
let result = tokio::task::spawn_blocking(|| {
    expensive_computation()
}).await?;
```

## Related Decisions
- ADR-004: HTTP client selection (uses Tokio via reqwest)
- ADR-005: Web framework selection (Axum built on Tokio)
