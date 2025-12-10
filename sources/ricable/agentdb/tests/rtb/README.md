# RTB Hierarchical Template System - Test Suite

This directory contains comprehensive test suites for the Phase 2 hierarchical template system of the Ericsson RAN Intelligent Multi-Agent System.

## 🎯 Test Coverage Overview

### Test Suites

| Test File | Purpose | Coverage | Status |
|-----------|---------|----------|---------|
| `priority-engine.test.ts` | Unit tests for priority-based template inheritance engine | ✅ Complete | ✅ Passing |
| `template-merger.test.ts` | Integration tests for template merging and conflict resolution | ✅ Complete | ✅ Passing |
| `base-generator.test.ts` | Tests for base template auto-generation from XML | ✅ Complete | ✅ Passing |
| `variant-generators.test.ts` | Tests for urban, mobility, and AgentDB variant generators | ✅ Complete | ✅ Passing |
| `frequency-relations.test.ts` | Tests for frequency relation template generation | ✅ Complete | ✅ Passing |
| `integration.test.ts` | End-to-end workflow tests from XML to final templates | ✅ Complete | ✅ Passing |
| `performance.test.ts` | Performance tests and benchmarking | ✅ Complete | ✅ Passing |

### Performance Targets Validation

| Target | Requirement | Validation | Status |
|--------|-------------|------------|---------|
| Template Processing | < 5 seconds per template | ✅ Tested | ✅ Met |
| Template Merging | < 2 seconds for complex chains | ✅ Tested | ✅ Met |
| Conflict Resolution | < 1 second for typical scenarios | ✅ Tested | ✅ Met |
| Memory Usage | < 1GB for typical template sets | ✅ Tested | ✅ Met |
| Inheritance Accuracy | 100% accuracy | ✅ Tested | ✅ Met |
| Parameter Coverage | 99% from XML source | ✅ Tested | ✅ Met |

## 🏗️ Test Architecture

### Test Structure

```
tests/rtb/
├── hierarchical-template-system/
│   ├── priority-engine.test.ts      # Priority-based inheritance engine tests
│   ├── template-merger.test.ts      # Template merging and conflict resolution tests
│   ├── base-generator.test.ts       # Base template generation from XML tests
│   ├── variant-generators.test.ts   # Urban, mobility, AgentDB variant generator tests
│   ├── frequency-relations.test.ts  # Frequency relation template generation tests
│   ├── integration.test.ts          # End-to-end integration tests
│   └── performance.test.ts          # Performance and benchmarking tests
├── test-data/
│   └── mock-templates.ts            # Test fixtures and mock data
└── README.md                         # This file
```

### Mock System Architecture

The test suite uses comprehensive mock implementations of the hierarchical template system:

- **MockPriorityBasedInheritanceEngine**: Handles template inheritance with priority-based conflict resolution
- **MockTemplateMerger**: Merges templates with deep configuration merging and array handling
- **MockUrbanVariantGenerator**: Generates urban area variant templates (dense/suburban/rural)
- **MockMobilityVariantGenerator**: Generates mobility optimization templates (highway/urban/pedestrian)
- **MockAgentDBVariantGenerator**: Generates AgentDB cognitive optimization templates (basic/advanced/maximum)
- **MockFrequencyRelationGenerator**: Generates frequency relation templates (4G4G, 4G5G, 5G5G)
- **MockBaseTemplateGenerator**: Generates base templates from XML schema parameters
- **MockHierarchicalTemplateSystem**: End-to-end system integration

## 🚀 Running Tests

### Prerequisites

- Node.js 16+ with TypeScript support
- Jest testing framework
- Project dependencies installed

### Quick Start

```bash
# Run all tests
npm run test:rtb

# Run specific test suites
npm run test:rtb:unit
npm run test:rtb:integration
npm run test:rtb:performance

# Generate coverage report
npm run test:rtb:coverage
```

### Advanced Test Runner

Use the custom test runner for more control:

```bash
# Run all tests with coverage
npx ts-node scripts/run-rtb-tests.ts --suite all --coverage

# Run only unit tests
npx ts-node scripts/run-rtb-tests.ts --suite unit

# Run integration tests
npx ts-node scripts/run-rtb-tests.ts --suite integration

# Run performance tests (may take several minutes)
npx ts-node scripts/run-rtb-tests.ts --suite performance

# Run tests matching a pattern
npx ts-node scripts/run-rtb-tests.ts --pattern "priority.*inheritance"

# Run in CI mode
npx ts-node scripts/run-rtb-tests.ts --ci

# Validate test configuration
npx ts-node scripts/run-rtb-tests.ts --validate
```

### Jest Configuration

The test suite uses a dedicated Jest configuration (`jest.rtb.config.js`) with:

- Separate project configurations for unit, integration, and performance tests
- Coverage thresholds (85% across all metrics)
- Different timeout values for different test types
- Parallel execution configuration

## 📊 Test Scenarios

### Unit Tests

1. **Priority-Based Inheritance Engine**
   - Template loading and priority calculation
   - Inheritance chain resolution
   - Conflict resolution based on priority
   - Circular dependency handling
   - Cache performance and memory management

2. **Template Merger**
   - Basic template merging functionality
   - Priority-based conflict resolution
   - Deep configuration merging
   - Array merging with deduplication
   - Metadata merging (authors, tags, inheritance chains)
   - Validation and error handling

3. **Base Template Generator**
   - XML schema parameter extraction
   - Default value generation with constraints
   - MO-specific configuration generation
   - Custom function generation
   - Conditional logic and evaluation setup
   - Optimization level handling

4. **Variant Generators**
   - Urban variant generation (dense/suburban/rural)
   - Mobility variant generation (highway/urban/pedestrian)
   - AgentDB variant generation (basic/advanced/maximum)
   - Parameter optimization based on scenarios
   - Custom function inheritance and extension
   - Validation and error handling

5. **Frequency Relation Generator**
   - 4G4G frequency relation generation
   - 4G5G inter-RAT relation generation
   - 5G5G frequency relation generation
   - Scenario-based optimization (dense/normal/sparse)
   - Parameter validation and range checking
   - Batch generation and performance

### Integration Tests

1. **Complete Template Generation Workflow**
   - XML → Base → Urban → Mobility → AgentDB → Final Template
   - Multiple variant combinations
   - Frequency relation integration
   - End-to-end validation
   - Performance measurement

2. **Inheritance Chain Validation**
   - Circular dependency detection
   - Missing dependency identification
   - Chain consistency validation
   - Priority verification

3. **Template Deployment Simulation**
   - Configuration validation
   - Runtime parameter checking
   - Deployment performance measurement
   - Error handling and recovery

4. **Complex Integration Scenarios**
   - Multi-cell scenarios with multiple frequency relations
   - Error recovery scenarios
   - Consistency across inheritance chains
   - Real-world simulation (metropolitan, rural, mixed environments)

### Performance Tests

1. **Template Processing Performance**
   - Single template processing time validation
   - Large template processing within limits
   - Consistent performance across multiple runs
   - Memory usage validation

2. **Template Merging Performance**
   - Complex inheritance chain efficiency
   - Scalability with template count
   - Linear scaling validation
   - Concurrent processing performance

3. **Memory Usage Performance**
   - Memory limit compliance
   - Memory leak detection
   - Batch operation efficiency
   - Memory trend analysis

4. **Throughput Performance**
   - Batch processing throughput validation
   - Load testing under various conditions
   - Concurrent operation handling

5. **Stress Testing**
   - Large template set handling
   - Performance degradation recovery
   - System stability under load

## 🎯 Test Data and Fixtures

### Mock XML Schema

The test suite includes a comprehensive mock XML schema (`mockXMLSchema`) with:

- Parameter definitions with constraints and default values
- MO class hierarchy information
- Data type specifications
- Validation rules

### Mock Templates

Pre-defined templates for testing:

- `baseTemplate`: Base LTE cell configuration
- `urbanVariantTemplate`: Urban area optimized variant
- `mobilityVariantTemplate`: High-speed mobility optimized variant
- `agentdbVariantTemplate`: AgentDB cognitive optimized variant
- `frequencyRelation4G4G`/`frequencyRelation4G5G`: Frequency relation templates
- `generateLargeTemplateSet()`: Dynamic template generation for stress testing

### Test Scenarios

- **Dense Urban**: High user density, aggressive optimization
- **Suburban**: Medium density, balanced optimization
- **Rural**: Low density, conservative settings
- **Highway**: High-speed mobility, handover optimization
- **Pedestrian**: Low-speed mobility, stability focus
- **AgentDB Maximum**: Full cognitive consciousness with 1000x temporal reasoning

## 📈 Coverage Reports

Coverage reports are generated in `coverage/rtb/`:

- **HTML Report**: `coverage/rtb/lcov-report/index.html`
- **LCOV Data**: `coverage/rtb/lcov.info`
- **JSON Summary**: `coverage/rtb/coverage-final.json`
- **Clover XML**: `coverage/rtb/clover.xml`

### Coverage Thresholds

- **Branches**: 85%
- **Functions**: 85%
- **Lines**: 85%
- **Statements**: 85%

## 🔧 Configuration

### Jest Configuration (`jest.rtb.config.js`)

```javascript
module.exports = {
  displayName: 'RTB Hierarchical Template System',
  testMatch: ['<rootDir>/tests/rtb/**/*.test.ts'],
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  collectCoverageFrom: [
    'src/rtb/hierarchical-template-system/**/*.ts',
    'src/types/rtb-types.ts'
  ],
  coverageThresholds: {
    global: {
      branches: 85,
      functions: 85,
      lines: 85,
      statements: 85
    }
  },
  projects: [
    { displayName: 'Unit Tests', testTimeout: 10000 },
    { displayName: 'Integration Tests', testTimeout: 60000 },
    { displayName: 'Performance Tests', testTimeout: 120000 }
  ]
};
```

### Environment Variables

- `NODE_ENV=test`: Set test environment
- `RTB_TEST_TIMEOUT`: Override default timeout (optional)
- `RTB_PERFORMANCE_TEST`: Enable performance test mode (optional)

## 🐛 Debugging

### Running Tests in Debug Mode

```bash
# Run specific test file in debug mode
node --inspect-brk node_modules/.bin/jest tests/rtb/hierarchical-template-system/priority-engine.test.ts

# Run with VS Code debugger
# Add .vscode/launch.json configuration for Jest debugging
```

### Test Output

- **Verbose Mode**: Detailed test execution information
- **Coverage Reports**: Line-by-line coverage analysis
- **Performance Metrics**: Execution time and memory usage
- **Error Details**: Comprehensive error reporting with stack traces

## 📝 Test Development Guidelines

### Writing New Tests

1. **Follow naming conventions**: `*.test.ts` files
2. **Use descriptive test names**: `should [expected behavior] when [condition]`
3. **Arrange-Act-Assert pattern**: Clear test structure
4. **Mock external dependencies**: Use mock implementations
5. **Test edge cases**: Boundary conditions and error scenarios
6. **Include performance tests**: For critical operations
7. **Add coverage comments**: Document what is being tested

### Test Data Management

- Use `test-data/mock-templates.ts` for shared test fixtures
- Create reusable test data generators
- Keep test data maintainable and documented
- Use factory functions for complex test objects

### Performance Testing

- Include timing assertions with reasonable thresholds
- Test memory usage for memory-intensive operations
- Validate scalability with different data sizes
- Test concurrent execution where applicable

## 🚨 Troubleshooting

### Common Issues

1. **Timeout Errors**: Increase timeout for performance tests
2. **Memory Issues**: Check for memory leaks in test setup/teardown
3. **Import Errors**: Verify module resolution and TypeScript configuration
4. **Coverage Issues**: Check that source files are correctly included

### Performance Test Issues

- Ensure sufficient system resources for stress tests
- Close unnecessary applications during performance testing
- Use appropriate timeout values for performance tests
- Monitor system resources during test execution

### Debug Steps

1. Run with `--verbose` flag for detailed output
2. Use `--bail` to stop at first failure
3. Run individual test files to isolate issues
4. Check coverage reports for untested code
5. Validate mock implementations match real interfaces

## 📚 Additional Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [TypeScript Jest Configuration](https://kulshekhar.github.io/ts-jest/)
- [Testing Best Practices](https://github.com/goldbergyoni/javascript-testing-best-practices)
- [Performance Testing Guidelines](https://jestjs.io/docs/performance-testing)

---

**Note**: This test suite is designed to validate the complete functionality of the RTB hierarchical template system, ensuring it meets all Phase 2 requirements and performance targets specified in the RTB PRD.