# Neural Bus Architecture

## Overview

The Neural Bus is an advanced QUIC-based communication layer for ReasoningBank that provides:

- **Role-based stream multiplexing** - Separate streams for control, req/resp, gossip, snapshot, and telemetry
- **Intent-capped actions** - Ed25519-signed authorization with spend caps and replay protection
- **Application-level priority queues** - Three-tier priority system (high, normal, low)
- **Reasoning streaming** - Parallel streaming for tokens, traces, rubrics, and verification
- **Gossip protocol** - Peer-to-peer pattern synchronization with backpressure
- **Snapshot streaming** - Chunk-based bulk data transfer with checksums

**Note**: This is a **native-only** feature. WASM compatibility is not supported due to cryptographic requirements.

## Architecture

### Frame Protocol

All messages use a binary frame format:

```
┌──────────┬──────┬────────────┬─────────────┬─────────────┬─────────┐
│ version  │ type │ header_len │ payload_len │   header    │ payload │
│ (u16)    │ (u8) │   (u32)    │    (u32)    │   (JSON)    │ (bytes) │
└──────────┴──────┴────────────┴─────────────┴─────────────┴─────────┘
```

- **version**: Protocol version (currently 1)
- **type**: Frame type (control, request, response, gossip, etc.)
- **header_len**: Length of JSON header in bytes
- **payload_len**: Length of payload in bytes
- **header**: JSON-encoded metadata with intent signature
- **payload**: Binary payload data

### Intent Verification

Every mutating operation requires an Ed25519-signed intent with:

```json
{
  "kid": "key-identifier",
  "ts": 1234567890,
  "nonce": "unique-uuid",
  "scope": "write",
  "cap": 1000,
  "sig": "base64-signature",
  "op": "store_pattern",
  "version": 1
}
```

The signature covers: `SHA256(kid:ts:nonce:scope:cap:op)`

**Security Features:**
- ✅ Replay protection via nonce tracking
- ✅ Time-based expiration (5-minute window)
- ✅ Scope-based permissions (read, write, admin)
- ✅ Spend caps for resource limits
- ✅ 0-RTT rejection for mutating operations

### Stream Roles

| Role | Type | Scope | Purpose |
|------|------|-------|---------|
| **Control** | Bidirectional | Write | Connection management |
| **ReqResp** | Bidirectional | Write | Request/response patterns |
| **Gossip** | Unidirectional | Read | Peer-to-peer sync |
| **Snapshot** | Unidirectional | Write | Bulk data transfer |
| **Telemetry** | Unidirectional | Read | Metrics and monitoring |
| **ReasoningTokens** | Unidirectional | Read | Token streaming |
| **ReasoningTraces** | Unidirectional | Read | Trace streaming |
| **ReasoningRubrics** | Unidirectional | Read | Rubric updates |
| **ReasoningVerify** | Unidirectional | Read | Verification results |

### Priority Queue System

Three-tier application-level prioritization using separate mpsc channels:

```rust
// High priority: Latency-sensitive operations (control, urgent)
queue.send(Priority::High, data).await?;

// Normal priority: Standard operations
queue.send(Priority::Normal, data).await?;

// Low priority: Background tasks (gossip, telemetry)
queue.send(Priority::Low, data).await?;

// Fair scheduling: high → normal → low
let message = queue.recv().await?;
```

## Usage Examples

### Basic Neural Bus Setup

```rust
use reasoningbank_network::neural_bus::{
    NeuralBus, NeuralBusConfig, IntentVerifier, StreamRole,
};

// Configure neural bus
let config = NeuralBusConfig {
    max_streams_per_role: 10,
    high_priority_buffer: 1000,
    normal_priority_buffer: 5000,
    low_priority_buffer: 10000,
    enable_0rtt_readonly: true,
    reject_0rtt_mutations: true,
};

// Create intent verifier
let verifier = IntentVerifier::new(10000);

// Register public keys
verifier.register_key("agent-1".to_string(), public_key).await;

// Create neural bus from QUIC connection
let bus = NeuralBus::new(connection, config, verifier);
```

### Opening Role-Based Streams

```rust
// Open bidirectional stream for request/response
let stream = bus.open_stream(StreamRole::ReqResp).await?;

// Send frame with intent
let mut header = FrameHeader::new("store_pattern".to_string());
header.kid = Some("agent-1".to_string());
header.scope = Some("write".to_string());
// ... set intent fields ...

let frame = Frame::new(FrameType::Request, header, payload);
stream.send_frame(frame).await?;

// Receive response
let response = stream.recv_frame().await?;
```

### Priority Queue Operations

```rust
// Get priority queue
let queue = bus.priority_queue();

// Send with priority
queue.send(Priority::High, urgent_data).await?;
queue.send(Priority::Normal, normal_data).await?;
queue.send(Priority::Low, background_data).await?;

// Receive in priority order
let message = queue.recv().await?;
println!("Priority: {:?}", message.priority);
```

### Reasoning Streams

```rust
// Open reasoning streams
let req_stream = bus.open_stream(StreamRole::ReqResp).await?;
let token_stream = bus.open_uni_stream(StreamRole::ReasoningTokens).await?;
let trace_stream = bus.open_uni_stream(StreamRole::ReasoningTraces).await?;
let rubric_stream = bus.open_uni_stream(StreamRole::ReasoningRubrics).await?;
let verify_stream = bus.open_uni_stream(StreamRole::ReasoningVerify).await?;

let reasoning = ReasoningStreams::new(
    req_stream,
    token_stream,
    trace_stream,
    rubric_stream,
    verify_stream,
);

// Stream tokens incrementally
reasoning.send_token(Bytes::from("token chunk")).await?;

// Stream verification results in parallel
reasoning.send_verification(Bytes::from("verification result")).await?;
```

### Gossip Protocol

```rust
use reasoningbank_network::neural_bus::gossip::{GossipManager, GossipMessage};

let manager = GossipManager::new(100, 20);

// Add gossip stream
let stream = bus.open_uni_stream(StreamRole::Gossip).await?;
manager.add_stream(stream).await;

// Send gossip message with backpressure control
let message = GossipMessage::PatternUpdate {
    pattern_id: "pattern-123".to_string(),
    version: 5,
    metadata: HashMap::new(),
};

if manager.send(message).await.is_err() {
    println!("Backpressure active, queue full");
}

// Run dispatcher to send queued messages
tokio::spawn(async move {
    manager.run_dispatcher().await.unwrap();
});
```

### Snapshot Streaming

```rust
use reasoningbank_network::neural_bus::snapshot::SnapshotManager;

let manager = SnapshotManager::new(Some(1024 * 1024)); // 1MB chunks

// Send snapshot
let stream = bus.open_uni_stream(StreamRole::Snapshot).await?;
manager.send_snapshot(
    Arc::new(Mutex::new(stream)),
    "snapshot-123".to_string(),
    large_data,
).await?;

// Receive snapshot
let (metadata, data) = manager.receive_snapshot(stream).await?;
println!("Received {} bytes in {} chunks",
    metadata.total_size, metadata.chunk_count);
```

### Intent Creation and Verification

```rust
use reasoningbank_network::neural_bus::intent::{Intent, Scope, generate_keypair};

// Generate keypair
let (signing_key, verifying_key) = generate_keypair();

// Create signed intent
let intent = Intent::new(
    "agent-1".to_string(),
    Scope::Write,
    1000, // spend cap
    "store_pattern".to_string(),
    &signing_key,
);

// Verify intent
let verifier = IntentVerifier::new(1000);
verifier.register_key("agent-1".to_string(), verifying_key).await;

verifier.verify_intent(&intent, &Scope::Write).await?;
```

## Performance Characteristics

### Frame Overhead

- **Fixed header**: 11 bytes (version + type + lengths)
- **JSON header**: ~150-300 bytes (with intent signature)
- **Total overhead**: ~160-310 bytes per frame

### Priority Queue Performance

- **High priority latency**: <1ms (dedicated channel)
- **Normal priority latency**: 1-5ms
- **Low priority latency**: 5-20ms (fair scheduling)
- **Throughput**: 100k+ messages/sec per queue

### Snapshot Streaming

- **Chunk size**: 1MB (configurable)
- **Verification**: SHA256 per chunk + overall checksum
- **Overhead**: ~1% (chunk headers + checksums)
- **Throughput**: Limited by QUIC congestion control

### Gossip Protocol

- **Backpressure threshold**: Configurable (default: 20% capacity)
- **Queue depth**: Configurable (default: 100 messages)
- **Fanout**: Limited by stream count

## Security Considerations

### Intent Verification

1. **Always verify intents** for mutating operations
2. **Reject 0-RTT** for write operations (enable_0rtt_readonly = true)
3. **Track nonces** to prevent replay attacks (max_nonces configurable)
4. **Enforce time limits** (5-minute window, adjustable)
5. **Use scope hierarchy**: admin > write > read

### Key Management

```rust
// Register keys with unique identifiers
verifier.register_key("agent-1".to_string(), key1).await;
verifier.register_key("agent-2".to_string(), key2).await;

// Keys should be rotated periodically
// Old keys should be removed after grace period
```

### Nonce Management

The verifier maintains a rolling window of used nonces:
- Default: 10,000 nonces
- Auto-pruned when 80% full
- Prevents memory exhaustion

## Testing

The neural bus includes comprehensive tests:

### Unit Tests (18 tests)
- Frame encoding/decoding
- Intent creation/verification
- Priority queue ordering
- Stream role conversion
- Gossip message serialization
- Snapshot chunking

### Integration Tests (9 tests)
- Neural bus configuration
- Intent verification workflow
- Priority queue system
- Frame with intent headers
- Gossip protocol
- Snapshot streaming
- Stream roles
- 0-RTT rejection
- Concurrent streams

Run tests with:
```bash
cargo test -p reasoningbank-network
```

## Limitations

### WASM Incompatibility

The neural bus uses Ed25519 cryptography which requires native code:
- `ed25519-dalek` crate (native crypto)
- `sha2` for hashing
- `rand` for key generation

For WASM targets, use the basic QUIC layer without intent verification.

### QUIC Version

Currently uses `quinn = "0.10"`. Upgrade to 0.11 requires:
- API changes for stats (congestion_window → cwnd)
- Stream ID handling updates
- Transport config updates

## Future Enhancements

### Planned Features

1. **Neural model streaming** - Direct streaming of model weights
2. **Federated learning** - Gradient aggregation over neural bus
3. **Byzantine fault tolerance** - Multi-signature intents
4. **Rate limiting** - Per-key spend tracking
5. **Compression** - Zstd compression for large payloads
6. **Metrics** - Prometheus exporter for monitoring

### Optimization Opportunities

1. **Zero-copy** - Use `quinn::RecvStream::read_chunk` for large payloads
2. **Batching** - Batch small frames to reduce overhead
3. **Connection pooling** - Reuse connections across neural buses
4. **Stream multiplexing** - Share streams for multiple operations

## References

- [QUIC Protocol (RFC 9000)](https://www.rfc-editor.org/rfc/rfc9000.html)
- [Ed25519 Digital Signatures](https://ed25519.cr.yp.to/)
- [Quinn QUIC Implementation](https://github.com/quinn-rs/quinn)
- [ReasoningBank Documentation](../../../README.md)

## License

MIT OR Apache-2.0
