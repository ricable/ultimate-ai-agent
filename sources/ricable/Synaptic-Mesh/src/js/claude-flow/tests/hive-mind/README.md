# Hive Mind Test Suite

Comprehensive test suite for the Hive Mind distributed AI coordination system.

## Test Structure

```
tests/hive-mind/
├── unit/                    # Unit tests for individual components
│   ├── hive-mind-cli.test.js      # CLI command tests
│   ├── sqlite-operations.test.js  # Database operation tests
│   ├── agent-spawning.test.js     # Agent lifecycle tests
│   ├── communication-protocol.test.js # Inter-agent messaging tests
│   └── memory-operations.test.js  # Shared memory tests
├── integration/             # Integration tests
│   ├── wizard-flow.test.js        # Setup wizard tests
│   └── swarm-scaling.test.js      # Scaling from 1-100 agents
├── performance/             # Performance benchmarks
│   └── benchmark-scenarios.test.js # Various workload tests
├── e2e/                     # End-to-end scenarios
│   └── test-scenarios.md          # Documented test scenarios
├── jest.config.js           # Jest configuration
├── setup.js                 # Test environment setup
└── README.md               # This file
```

## Running Tests

### Run All Tests
```bash
npm test -- tests/hive-mind/
```

### Run Specific Test Suites
```bash
# Unit tests only
npm test -- tests/hive-mind/unit/

# Integration tests only
npm test -- tests/hive-mind/integration/

# Performance tests only
npm test -- tests/hive-mind/performance/

# E2E tests only
npm test -- tests/hive-mind/e2e/
```

### Run with Coverage
```bash
npm test -- --coverage tests/hive-mind/
```

### Run in Watch Mode
```bash
npm test -- --watch tests/hive-mind/
```

## Docker Testing

### Build Test Container
```bash
docker build -f docker/Dockerfile.hive-mind --target test -t hive-mind:test .
```

### Run Tests in Docker
```bash
# All tests
./docker/run-tests.sh

# Unit tests only
./docker/run-tests.sh --unit-only

# Integration tests only
./docker/run-tests.sh --integration-only

# Performance tests only
./docker/run-tests.sh --performance-only
```

### Docker Compose Testing
```bash
# Run test suite
docker-compose -f docker/docker-compose.hive-mind.yml --profile test up

# Run with monitoring
docker-compose -f docker/docker-compose.hive-mind.yml --profile test --profile monitoring up
```

## Test Scenarios

### Unit Tests
- **CLI Commands**: Tests all hive-mind CLI commands and subcommands
- **SQLite Operations**: Database CRUD operations, transactions, indexes
- **Agent Spawning**: Agent creation, validation, lifecycle management
- **Communication**: Message passing, broadcasts, topic subscriptions
- **Memory Operations**: Shared memory, TTL, persistence, synchronization

### Integration Tests
- **Wizard Flow**: Complete setup wizard with various configurations
- **Swarm Scaling**: Tests scaling from 1 to 100+ agents
  - Small scale (1-10 agents)
  - Medium scale (10-50 agents)
  - Large scale (50-100 agents)
  - Stress testing (rapid scaling)

### Performance Tests
- **Agent Spawning**: Measures spawn time for various agent counts
- **Task Distribution**: Tests efficient task allocation
- **Communication**: Message throughput and latency
- **Concurrent Operations**: Mixed workload performance
- **Scalability Limits**: Maximum capacity testing

### E2E Scenarios
1. **Simple 5-Agent Refactoring**: Basic code refactoring task
2. **Medium 20-Agent Feature Dev**: Complete feature implementation
3. **Complex 100-Agent System Design**: Microservices architecture
4. **Stress Test 1000 Agents**: Large-scale data processing
5. **Failure Recovery**: Resilience and recovery testing
6. **Real-time Collaboration**: Human-AI interaction

## Performance Benchmarks

### Expected Metrics

| Metric | Small (5) | Medium (20) | Large (100) | Extreme (1000) |
|--------|-----------|-------------|-------------|----------------|
| Spawn Time | < 1s | < 5s | < 30s | < 60s |
| Memory/Agent | 10MB | 8MB | 5MB | 2MB |
| Message Latency | < 1ms | < 2ms | < 5ms | < 10ms |
| Task Distribution | < 100ms | < 500ms | < 2s | < 10s |

## Test Reports

Test results are saved to:
- **Coverage Report**: `coverage/hive-mind/lcov-report/index.html`
- **Test Results**: `test-results/hive-mind/`
  - `junit.xml` - JUnit format for CI
  - `index.html` - HTML report
  - `performance.json` - Performance metrics

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Hive Mind Tests
  run: |
    npm test -- --ci --coverage tests/hive-mind/
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage/hive-mind/lcov.info
    flags: hive-mind
```

### Jenkins
```groovy
stage('Hive Mind Tests') {
    steps {
        sh 'npm test -- --ci tests/hive-mind/'
        junit 'test-results/hive-mind/junit.xml'
        publishHTML([
            reportDir: 'coverage/hive-mind/lcov-report',
            reportFiles: 'index.html',
            reportName: 'Hive Mind Coverage'
        ])
    }
}
```

## Debugging Tests

### Enable Debug Logging
```bash
HIVE_LOG_LEVEL=debug npm test -- tests/hive-mind/
```

### Run Single Test
```bash
npm test -- tests/hive-mind/unit/agent-spawning.test.js -t "should spawn a single agent"
```

### Debug in VS Code
Add to `.vscode/launch.json`:
```json
{
  "type": "node",
  "request": "launch",
  "name": "Debug Hive Mind Tests",
  "program": "${workspaceFolder}/node_modules/.bin/jest",
  "args": [
    "--runInBand",
    "--no-cache",
    "tests/hive-mind/${fileBasenameNoExtension}"
  ],
  "cwd": "${workspaceFolder}",
  "console": "integratedTerminal"
}
```

## Contributing

When adding new tests:
1. Follow the existing structure
2. Use descriptive test names
3. Include both positive and negative test cases
4. Add performance assertions where applicable
5. Update this README with new scenarios

## Test Data

Test databases and files are automatically cleaned up after tests run. 
Temporary files are created in `tmp/test-*` directories.

## Troubleshooting

### Tests Timing Out
- Increase timeout in jest.config.js
- Check for blocking operations
- Ensure async operations are properly awaited

### Database Lock Errors
- Ensure previous test runs completed
- Check for zombie processes
- Use unique database names for parallel tests

### Memory Issues
- Limit parallel test execution
- Increase Node.js heap size: `NODE_OPTIONS=--max-old-space-size=4096`
- Profile memory usage with `--detect-leaks`

## Future Enhancements

- [ ] Add visual regression tests for UI
- [ ] Implement contract testing for agent protocols
- [ ] Add chaos engineering tests
- [ ] Create performance regression tracking
- [ ] Add security penetration tests
- [ ] Implement load testing with K6 or JMeter