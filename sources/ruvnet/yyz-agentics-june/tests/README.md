# Neural Network Test Suite

Comprehensive testing framework for neural network implementations with unit tests, integration tests, gradient checking, and performance benchmarks.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_layers.py      # Layer implementations
│   ├── test_activations.py # Activation functions
│   ├── test_loss_functions.py # Loss functions
│   ├── test_optimizers.py  # Optimization algorithms
│   └── test_initializers.py # Weight initialization
│
├── integration/            # Integration tests
│   └── test_neural_network.py # Full network testing
│
├── benchmarks/            # Performance benchmarks
│   └── test_performance.py # Speed and memory tests
│
├── utils/                 # Testing utilities
│   └── gradient_check.py  # Gradient checking tools
│
└── run_all_tests.py      # Main test runner

```

## Running Tests

### Run all tests:
```bash
python tests/run_all_tests.py
```

### Run specific test types:
```bash
# Unit tests only
python tests/run_all_tests.py --type unit

# Integration tests only
python tests/run_all_tests.py --type integration

# Benchmarks only
python tests/run_all_tests.py --type benchmark
```

### Additional options:
```bash
# Save test report
python tests/run_all_tests.py --save-report

# Change verbosity
python tests/run_all_tests.py --verbose 0  # Quiet
python tests/run_all_tests.py --verbose 1  # Normal
python tests/run_all_tests.py --verbose 2  # Detailed
```

## Test Coverage

### Unit Tests

1. **Layers** (`test_layers.py`)
   - Dense/Linear layers
   - Convolutional layers
   - Pooling layers
   - Normalization layers
   - Recurrent layers (RNN, LSTM, GRU)
   - Dropout
   - Embedding layers

2. **Activations** (`test_activations.py`)
   - ReLU, Leaky ReLU, ELU
   - Sigmoid, Tanh
   - Softmax
   - GELU, Swish
   - Numerical stability tests

3. **Loss Functions** (`test_loss_functions.py`)
   - MSE, MAE
   - Cross-entropy (binary and categorical)
   - Hinge loss
   - Focal loss
   - Custom losses

4. **Optimizers** (`test_optimizers.py`)
   - SGD (with/without momentum)
   - Adam, AdamW
   - RMSprop, Adagrad
   - Learning rate schedules
   - Gradient clipping

5. **Initializers** (`test_initializers.py`)
   - Xavier/Glorot initialization
   - He/Kaiming initialization
   - Orthogonal initialization
   - Statistical property validation

### Integration Tests

1. **Network Testing** (`test_neural_network.py`)
   - End-to-end forward/backward pass
   - Multi-layer networks
   - Skip connections
   - Training dynamics
   - Convergence properties

2. **Numerical Stability**
   - Large/small input handling
   - Gradient flow validation
   - Mixed precision testing

3. **Save/Load Functionality**
   - Model serialization
   - Optimizer state persistence
   - Training resumption

### Performance Benchmarks

1. **Speed Tests**
   - Layer operation timing
   - Scaling with batch size
   - Scaling with model size
   - Optimizer performance

2. **Memory Tests**
   - Memory usage profiling
   - Peak memory tracking
   - Memory leak detection

## Gradient Checking

The test suite includes comprehensive gradient checking utilities:

```python
from tests.utils.gradient_check import GradientChecker

checker = GradientChecker(epsilon=1e-5, tolerance=1e-7)
passed, error, details = checker.check_gradient(
    loss_func, grad_func, params
)
```

## Test Reports

Test reports are saved in JSON format with:
- Summary statistics
- Per-class breakdowns
- Failed test details
- Performance metrics
- Timing information

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:
- Exit code 0 on success
- Exit code 1 on failure
- Machine-readable JSON reports
- Configurable verbosity

## Adding New Tests

To add new tests:

1. Create test class inheriting from `unittest.TestCase`
2. Add test methods starting with `test_`
3. Import in `run_all_tests.py`
4. Add to appropriate test category

Example:
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup code
        pass
        
    def test_feature_behavior(self):
        # Test implementation
        self.assertEqual(expected, actual)
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Reproducibility**: Use fixed random seeds
3. **Coverage**: Test both normal and edge cases
4. **Performance**: Keep unit tests fast
5. **Documentation**: Document what each test validates

## Known Limitations

- Tests are designed for NumPy-based implementations
- GPU testing requires additional setup
- Some tests use placeholder assertions pending implementation
- Performance benchmarks are hardware-dependent