# Math Evaluation in SPARC (v0.87.6)

SPARC includes a comprehensive math evaluation system that can handle various mathematical operations, from basic calculations to symbolic mathematics.

## Features

### 1. Equation Solving
- **Quadratic Equations**: Solves equations in the form ax² + bx + c = 0
  ```
  > solve x^2 + 5x + 6 = 0
  Result: x = -2 or x = -3
  ```

- **Linear Equations**: Handles equations in the form ax + b = c
  ```
  > solve 2x + 3 = 7
  Result: x = 2
  ```

### 2. Symbolic Mathematics
- **Expression Simplification**:
  ```
  > simplify (x + 1)(x - 1)
  Result: x^2 - 1
  ```

- **Factoring**:
  ```
  > factor x^2 - 2x + 1
  Result: (x - 1)^2
  ```

- **Trigonometric Identities**:
  ```
  > verify sin(x)^2 + cos(x)^2
  Result: 1
  ```

### 3. Basic Calculations
- Supports standard arithmetic operations:
  ```
  > calculate 2 ** 3 + 4 * 5
  Result: 28
  ```

### 4. Step-by-Step Solutions
Request detailed solution steps by prefixing with "solve step by step:":
```
> solve step by step: x^2 + 5x + 6 = 0
1. Identify coefficients: a=1, b=5, c=6
2. Use quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)
3. Calculate discriminant: b² - 4ac = 25 - 24 = 1
4. Solve: x = (-5 ± √1) / 2
5. Final solutions: x = -2 or x = -3
```

## Implementation Details

### Architecture
- **CalculatorTool**: Handles numerical calculations and basic equation solving
- **SymbolicSolverTool**: Manages symbolic mathematics using sympy
- **MathValidator**: Validates results with appropriate tolerances

### Validation Types
1. **Numerical**: Validates with configurable tolerance (default: 1e-6)
2. **Symbolic**: Checks expression equivalence
3. **Matrix**: Validates matrix operations using numpy

### Error Handling
- Robust error handling for invalid inputs
- Automatic recovery from parsing errors
- Fallback strategies for complex expressions

## Usage Examples

### 1. Basic Math
```
> calculate 3 * 4 + 5
Result: 17
```

### 2. Quadratic Equations
```
> solve x^2 - 4 = 0
Result: x = 2 or x = -2
```

### 3. Symbolic Math
```
> simplify (x + y)(x - y)
Result: x^2 - y^2
```

### 4. Complex Expressions
```
> solve 2x^2 - 3x + 1 = 0
Result: x = 1 or x = 0.5
```

## Advanced Features

### 1. Expression Forms
The system automatically chooses the most appropriate form:
- Simplified
- Factored
- Expanded

### 2. Solution Verification
For equations, the system can verify solutions:
```
> verify x^2 + 5x + 6 = 0, x = -2
Result: Verified (Left side = 0)
```

### 3. Precision Control
- Integer solutions are displayed as whole numbers
- Decimal solutions use appropriate precision
- Scientific notation for very large/small numbers

## Future Enhancements
1. Support for differential equations
2. Matrix operations and linear algebra
3. Graphing capabilities
4. Integration with computer algebra systems
5. Support for complex numbers
6. Additional mathematical domains (calculus, statistics)

## Technical Notes

### Dependencies
- sympy: Symbolic mathematics
- numpy: Numerical computations
- math: Standard mathematical functions

### Performance
- Automatic handling of token limits
- Efficient parsing of mathematical expressions
- Caching of intermediate results

### Integration
The math evaluation system is fully integrated with SPARC's:
- Chat mode
- Research capabilities
- Documentation generation
- Error handling system
