# Math Evaluation Tools

This package provides tools for evaluating mathematical computations and benchmarks.

## Components

### BenchmarkRequest
Request object containing:
- Problem statement
- Expected solution
- Evaluation criteria

### BenchmarkResponse  
Response object containing:
- Computed solution
- Solution steps
- Validation status

### MathValidator
Validates mathematical solutions using:
- Numerical comparison
- Symbolic equivalence
- Matrix operations

### MathBenchmarkEvaluator
Core evaluation engine that:
- Processes benchmark requests
- Coordinates validation
- Generates detailed reports

### MathAgent
LangChain agent that:
- Uses Chain-of-Thought reasoning
- Executes step-by-step problem solving
- Integrates with ReAct framework

## Usage

```python
from sparc_cli.tools.math import MathBenchmarkEvaluator, BenchmarkRequest

# Create a benchmark request
request = BenchmarkRequest(
    problem="2x + 5 = 13",
    expected_solution="x = 4",
    criteria={"method": "symbolic"}
)

# Initialize evaluator
evaluator = MathBenchmarkEvaluator()

# Run evaluation
result = evaluator.evaluate(request)
```

## Testing

Run tests with pytest:
```bash
pytest tests/sparc_cli/tools/math/
```
