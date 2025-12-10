# Quantum Features Test Suite

Comprehensive test coverage for agentic-jujutsu quantum features.

## Test Files

### 1. `quantum-dag-integration.test.js`
Tests QuantumDAG coordination features:
- ✅ Vertex creation and management
- ✅ DAG tips tracking
- ✅ Agent registration in quantum space
- ✅ Parallel operation coordination
- ✅ Conflict detection and resolution
- ✅ Causality maintenance
- ✅ High-throughput benchmarks (>100 ops/sec)

**Performance Targets:**
- Vertex creation: <100ms
- Parallel coordination: Sub-linear scaling
- Concurrent agents: Support 10-20+ agents

### 2. `quantum-fingerprints.test.js`
Tests quantum-resistant fingerprint generation:
- ✅ SHA3-512 fingerprint generation
- ✅ Deterministic fingerprints
- ✅ Fingerprint verification
- ✅ Collision resistance (0% collision rate)
- ✅ Avalanche effect (>40% bit difference)
- ✅ Quantum attack resistance
- ✅ Large data handling (1MB+)

**Performance Targets:**
- Generation: <1ms (average)
- Verification: <1ms
- Throughput: >1000 fingerprints/sec

### 3. `ml-dsa-signing.test.js`
Tests ML-DSA (Dilithium) post-quantum signatures:
- ✅ Keypair generation (ML-DSA-65)
- ✅ Signature creation (~1.3ms target)
- ✅ Signature verification (~0.85ms target)
- ✅ Commit signing workflow
- ✅ Multi-agent signing
- ✅ Forgery resistance
- ✅ Quantum attack simulation

**Performance Targets:**
- Signing: ~1.3ms (practical: <10ms)
- Verification: ~0.85ms (practical: <5ms)
- Throughput: High-volume signing support

### 4. `quantum-full-workflow.test.js`
End-to-end integration tests:
- ✅ Complete single-agent workflow
- ✅ Multi-agent coordination
- ✅ Dependent operations (causal chains)
- ✅ File modification scenarios
- ✅ Merge conflict resolution
- ✅ Rollback capabilities
- ✅ End-to-end performance benchmarks

**Workflow Steps:**
1. Register agent → Quantum coordinates
2. Create operation → QuantumDAG vertex
3. Generate fingerprint → SHA3-512 hash
4. Sign commit → ML-DSA signature
5. Verify chain → Complete validation

## Running Tests

```bash
# Run all quantum tests
npm test tests/quantum/

# Run specific test file
npm test tests/quantum/quantum-dag-integration.test.js
npm test tests/quantum/quantum-fingerprints.test.js
npm test tests/quantum/ml-dsa-signing.test.js
npm test tests/quantum/quantum-full-workflow.test.js

# Run with coverage
npm run test:coverage tests/quantum/

# Run performance benchmarks only
npm test tests/quantum/ -- --grep "Performance Benchmarks"
```

## Test Coverage Goals

- **Line Coverage**: >90%
- **Branch Coverage**: >85%
- **Function Coverage**: >95%
- **Statement Coverage**: >90%

## Performance Benchmarks

### Expected Performance

| Operation | Target | Acceptable |
|-----------|--------|------------|
| Fingerprint generation | <1ms | <2ms |
| ML-DSA signing | ~1.3ms | <10ms |
| ML-DSA verification | ~0.85ms | <5ms |
| Vertex creation | <100ms | <200ms |
| DAG coordination | Sub-linear | Linear |

### Throughput Targets

| Component | Target |
|-----------|--------|
| Fingerprints/sec | >1,000 |
| Signatures/sec | >100 |
| Verifications/sec | >200 |
| Vertices/sec | >100 |

## Security Properties Tested

1. **Collision Resistance**: 0% collision rate in 1000+ samples
2. **Avalanche Effect**: >40% bit difference for similar inputs
3. **Forgery Resistance**: 0% success in quantum attack simulations
4. **Signature Uniqueness**: Unique signatures for unique messages
5. **Tamper Detection**: 100% detection of modified data

## Test Data Characteristics

- **Small data**: <1KB (typical operations)
- **Medium data**: 1KB-100KB (files)
- **Large data**: 1MB+ (stress testing)
- **Edge cases**: Empty, null, unicode, special characters

## CI/CD Integration

These tests are designed to run in:
- ✅ Local development environments
- ✅ GitHub Actions CI
- ✅ Docker containers
- ✅ Cross-platform (Linux, macOS, Windows)

## Debugging Tips

1. **Verbose logging**: Set `DEBUG=agentic-jujutsu:*`
2. **Performance profiling**: Use `--prof` flag
3. **Memory analysis**: Use `--expose-gc` for GC tests
4. **Isolate tests**: Use `.only` on specific test cases

## Contributing

When adding new quantum features:
1. Add corresponding test coverage
2. Include performance benchmarks
3. Test edge cases and error handling
4. Update this README with new test descriptions

## License

MIT - Part of agentic-jujutsu package
