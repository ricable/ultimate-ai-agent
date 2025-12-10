# /polyglot-tdd

Implements Test-Driven Development across all languages in the polyglot environment with intelligent test creation, minimal implementation, and cross-language validation.

## Usage
```
/polyglot-tdd <feature-description> [--env <environment>] [--integration] [--e2e]
```

## Features
- **Language-appropriate testing** using pytest, Jest, cargo test, go test, nu test
- **Red-Green-Refactor cycle** with automated validation
- **Cross-language test orchestration** for polyglot features
- **Test intelligence integration** with existing monitoring
- **Performance-aware testing** with execution time tracking
- **Mock and fixture generation** appropriate to each language
- **Integration test coordination** across environments

## TDD Workflow by Language

### Python (`python-env/`)
- **Framework**: pytest with fixtures and parameterization
- **Structure**: `tests/` directory with `test_*.py` pattern
- **Mocking**: pytest-mock and unittest.mock
- **Coverage**: pytest-cov integration
- **Async**: pytest-asyncio for async/await patterns

### TypeScript (`typescript-env/`)
- **Framework**: Jest with TypeScript support
- **Structure**: `__tests__/` or `*.test.ts` co-location
- **Mocking**: Jest mocks and spies
- **Coverage**: Built-in Jest coverage reporting
- **Types**: Strong typing for test assertions

### Rust (`rust-env/`)
- **Framework**: Built-in `cargo test` with `#[test]` attributes
- **Structure**: Tests in `src/` with `#[cfg(test)]` modules
- **Mocking**: mockall crate for complex mocking
- **Coverage**: tarpaulin for coverage analysis
- **Integration**: `tests/` directory for integration tests

### Go (`go-env/`)
- **Framework**: Built-in testing package with `*_test.go` files
- **Structure**: Tests alongside source files
- **Mocking**: testify/mock for interfaces
- **Coverage**: `go test -cover` built-in coverage
- **Benchmarks**: `Benchmark*` functions for performance

### Nushell (`nushell-env/`)
- **Framework**: `nu test` command with assert functions
- **Structure**: `tests/` directory with `test_*.nu` scripts
- **Mocking**: Function override and dependency injection
- **Coverage**: Command execution validation
- **Integration**: Cross-script testing capabilities

## TDD Implementation Process

### Phase 1: Test Creation (RED)
1. **Analyze feature requirements** and break into testable units
2. **Generate failing tests** appropriate to each language
3. **Create test structure** following language conventions
4. **Validate test execution** ensuring proper failure

### Phase 2: Minimal Implementation (GREEN)
1. **Implement minimal code** to make tests pass
2. **Focus on functionality** not optimization
3. **Maintain simplicity** avoiding over-engineering
4. **Validate test success** across all environments

### Phase 3: Refactoring (REFACTOR)
1. **Improve code quality** while maintaining test passes
2. **Optimize performance** using existing monitoring
3. **Enhance readability** and maintainability
4. **Cross-language consistency** validation

## Instructions
1. **Feature Analysis**:
   - Break down feature into language-specific components
   - Identify cross-language integration points
   - Determine appropriate test types (unit, integration, e2e)
   - Plan test data and mock requirements

2. **Test Generation**:
   ```bash
   # Python
   cd python-env && devbox shell
   # Create test_feature.py with pytest fixtures
   
   # TypeScript
   cd typescript-env && devbox shell  
   # Create feature.test.ts with Jest structure
   
   # Rust
   cd rust-env && devbox shell
   # Add test module with #[test] functions
   
   # Go  
   cd go-env && devbox shell
   # Create feature_test.go with testing.T
   
   # Nushell
   cd nushell-env && devbox shell
   # Create test_feature.nu with assert commands
   ```

3. **Red Phase Execution**:
   - Run tests to ensure they fail appropriately
   - Validate error messages are clear and helpful
   - Confirm test structure and assertions are correct
   - Document expected behavior in test descriptions

4. **Green Phase Implementation**:
   - Write minimal code to pass tests
   - Focus on core functionality only
   - Avoid premature optimization
   - Ensure all tests pass in their environments

5. **Refactor Phase Enhancement**:
   - Improve code quality and structure
   - Add performance optimizations if needed
   - Enhance error handling and edge cases
   - Maintain backward compatibility

6. **Cross-Environment Validation**:
   - Run comprehensive test suite
   - Validate integration between languages
   - Check performance impact using analytics
   - Ensure security and dependency health

## Test Templates by Language

### Python Test Template
```python
import pytest
from src.feature import FeatureClass


class TestFeature:
    @pytest.fixture
    def feature_instance(self):
        return FeatureClass()
    
    def test_feature_basic_functionality(self, feature_instance):
        # Arrange
        input_data = "test input"
        expected = "expected output"
        
        # Act  
        result = feature_instance.process(input_data)
        
        # Assert
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_feature_async_operation(self, feature_instance):
        result = await feature_instance.async_process()
        assert result is not None
```

### TypeScript Test Template
```typescript
import { FeatureClass } from '../src/feature';

describe('FeatureClass', () => {
    let feature: FeatureClass;
    
    beforeEach(() => {
        feature = new FeatureClass();
    });
    
    it('should process input correctly', () => {
        // Arrange
        const input = 'test input';
        const expected = 'expected output';
        
        // Act
        const result = feature.process(input);
        
        // Assert
        expect(result).toBe(expected);
    });
    
    it('should handle async operations', async () => {
        const result = await feature.asyncProcess();
        expect(result).toBeDefined();
    });
});
```

### Rust Test Template
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_basic_functionality() {
        // Arrange
        let feature = FeatureStruct::new();
        let input = "test input";
        let expected = "expected output";
        
        // Act
        let result = feature.process(input);
        
        // Assert
        assert_eq!(result, expected);
    }
    
    #[tokio::test]
    async fn test_feature_async_operation() {
        let feature = FeatureStruct::new();
        let result = feature.async_process().await;
        assert!(result.is_ok());
    }
}
```

### Go Test Template
```go
package feature

import (
    "testing"
    "github.com/stretchr/testify/assert"
)

func TestFeatureBasicFunctionality(t *testing.T) {
    // Arrange
    feature := NewFeature()
    input := "test input"
    expected := "expected output"
    
    // Act
    result := feature.Process(input)
    
    // Assert
    assert.Equal(t, expected, result)
}

func TestFeatureAsyncOperation(t *testing.T) {
    feature := NewFeature()
    result, err := feature.AsyncProcess()
    assert.NoError(t, err)
    assert.NotNil(t, result)
}
```

### Nushell Test Template
```nushell
use std assert

def test_feature_basic_functionality [] {
    # Arrange
    let input = "test input"
    let expected = "expected output"
    
    # Act
    let result = (feature process $input)
    
    # Assert
    assert equal $result $expected
}

def test_feature_async_operation [] {
    let result = (feature async-process)
    assert ($result != null)
}
```

## Integration Testing
- **Cross-language workflows**: Test features spanning multiple environments
- **API integration**: Validate service-to-service communication
- **Data consistency**: Ensure data integrity across language boundaries
- **Performance integration**: Monitor cross-environment performance impact

## Intelligence Integration
- **Test performance tracking** using existing analytics
- **Flaky test detection** with test intelligence scripts
- **Coverage analysis** across all environments
- **Regression detection** for performance and functionality

## Advanced Features
- **Property-based testing** where supported (Hypothesis, fast-check, proptest)
- **Mutation testing** for test quality validation
- **Contract testing** for API boundaries
- **Performance benchmarking** integrated with monitoring

## Error Handling and Recovery
- Clear error messages for test failures
- Suggestions for fixing common test issues
- Integration with `/polyglot-clean` for formatting fixes
- Rollback capabilities for failed implementations