from evaluator import CalculatorTool, SymbolicSolverTool

def test_calculator():
    print("\nTesting Calculator Tool:")
    print("=======================")
    calc = CalculatorTool()
    
    # Test quadratic equations
    equations = [
        "x^2 + 5x + 6 = 0",
        "2x^2 - 3x + 1 = 0",
        "x^2 - 4 = 0"
    ]
    print("\nQuadratic Equations:")
    for eq in equations:
        print(f"\nInput: {eq}")
        try:
            result = calc.run(eq)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")

    # Test basic calculations
    calculations = [
        "2 + 2",
        "3 * 4",
        "10 / 2",
        "2 ** 3"
    ]
    print("\nBasic Calculations:")
    for calc_expr in calculations:
        print(f"\nInput: {calc_expr}")
        try:
            result = calc.run(calc_expr)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")

def test_symbolic():
    print("\nTesting Symbolic Solver Tool:")
    print("============================")
    solver = SymbolicSolverTool()
    
    # Test equations
    equations = [
        "x^2 + 5x + 6 = 0",
        "2*x + 3 = 7",
        "x^2 = 4"
    ]
    print("\nEquations:")
    for eq in equations:
        print(f"\nInput: {eq}")
        try:
            result = solver.run(eq)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")

    # Test expressions
    expressions = [
        "(x + 1)(x - 1)",
        "x^2 - 2*x + 1",
        "sin(x)^2 + cos(x)^2"
    ]
    print("\nExpressions:")
    for expr in expressions:
        print(f"\nInput: {expr}")
        try:
            result = solver.run(expr)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_calculator()
    test_symbolic()
