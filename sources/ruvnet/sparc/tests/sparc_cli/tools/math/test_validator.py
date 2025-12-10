import pytest
import numpy as np
from sparc_cli.tools.math.validator import MathValidator

@pytest.fixture
def validator():
    return MathValidator()

# Numerical Validation Tests
@pytest.mark.parametrize("predicted,expected,result", [
    ("2 + 2", "4", True),
    ("10 / 2", "5", True),
    ("3 * 4", "12", True),
    ("2.5 + 2.5", "5.0", True),
    ("1/3 + 1/3", "0.666666", True),
])
def test_validate_numerical_basic(validator, predicted, expected, result):
    is_valid, _ = MathValidator.validate_numerical(predicted, expected)
    assert is_valid == result

@pytest.mark.parametrize("predicted,expected,tolerance,result", [
    ("0.1 + 0.2", "0.3", 1e-6, True),
    ("1/3", "0.33333", 1e-3, True),
    ("1.0000001", "1.0", 1e-6, False),
])
def test_validate_numerical_tolerance(validator, predicted, expected, tolerance, result):
    is_valid, _ = MathValidator.validate_numerical(predicted, expected, tolerance)
    assert is_valid == result

def test_validate_numerical_errors():
    # Division by zero
    is_valid, error = MathValidator.validate_numerical("1/0", "1")
    assert not is_valid
    assert "error" in error.lower()

    # Invalid expression
    is_valid, error = MathValidator.validate_numerical("invalid", "1")
    assert not is_valid
    assert "error" in error.lower()

# Symbolic Validation Tests
@pytest.mark.parametrize("predicted,expected,result", [
    ("x + y", "y + x", True),
    ("(a + b)**2", "a**2 + 2*a*b + b**2", True),
    ("x**2 - y**2", "(x+y)*(x-y)", True),
    ("sin(x)**2 + cos(x)**2", "1", True),
])
def test_validate_symbolic_expressions(validator, predicted, expected, result):
    is_valid, _ = MathValidator.validate_symbolic(predicted, expected)
    assert is_valid == result

def test_validate_symbolic_errors():
    # Invalid expression
    is_valid, error = MathValidator.validate_symbolic("x +* y", "x + y")
    assert not is_valid
    assert "error" in error.lower()

# Matrix Validation Tests
def test_validate_matrix_numpy():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[1, 2], [3, 4]])
    is_valid, _ = MathValidator.validate_matrix(mat1, mat2)
    assert is_valid

def test_validate_matrix_shape_mismatch():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[1, 2, 3], [4, 5, 6]])
    is_valid, error = MathValidator.validate_matrix(mat1, mat2)
    assert not is_valid
    assert "shape" in error.lower()

def test_validate_matrix_close_values():
    mat1 = np.array([[1.0000001, 2], [3, 4]])
    mat2 = np.array([[1, 2], [3, 4]])
    is_valid, _ = MathValidator.validate_matrix(mat1, mat2)
    assert is_valid  # Should be True due to np.allclose default tolerance

def test_validate_matrix_string_input():
    is_valid, _ = MathValidator.validate_matrix("[[1, 2], [3, 4]]", "[[1, 2], [3, 4]]")
    assert is_valid

# Internal Helper Methods Tests
@pytest.mark.parametrize("expr,expected", [
    ("2 + 2", 4),
    ("3 * 4", 12),
    ("10 / 2", 5),
])
def test_safe_eval_valid(validator, expr, expected):
    result = MathValidator._safe_eval(expr)
    assert result == expected

def test_safe_eval_invalid():
    with pytest.raises(ValueError):
        MathValidator._safe_eval("import os")
    
    with pytest.raises(ValueError):
        MathValidator._safe_eval("__import__('os')")

def test_normalize_expression():
    expr1 = MathValidator._normalize_expression("x + y")
    expr2 = MathValidator._normalize_expression("y + x")
    assert expr1 - expr2 == 0

def test_compare_matrices():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[1, 2], [3, 4]])
    is_valid, _ = MathValidator._compare_matrices(mat1, mat2)
    assert is_valid

    # Test with different shapes
    mat3 = np.array([[1, 2]])
    is_valid, error = MathValidator._compare_matrices(mat1, mat3)
    assert not is_valid
    assert "shape" in error.lower()
"""
Tests for math validation functionality.
"""

import pytest
import numpy as np
from typing import Any, Dict
from unittest.mock import Mock

from sparc_cli.tools.math.validator import (
    validate_numerical,
    validate_symbolic,
    validate_matrix,
    ValidationResult
)

# Test fixtures
@pytest.fixture
def tolerance():
    """Default numerical comparison tolerance."""
    return 1e-6

@pytest.fixture
def mock_symbolic_engine():
    """Mock symbolic math engine."""
    engine = Mock()
    engine.simplify.return_value = "x + 1"
    return engine

# Numerical validation tests
@pytest.mark.parametrize("test_input,expected", [
    (4.0, 4.0),
    (3.14159, 3.14159),
    (-1.5, -1.5),
    (0, 0),
    (1e-10, 0),  # Test near-zero values
    (1e10, 1e10),  # Test large values
    (-0.0, 0.0),  # Test signed zero
])
def test_numerical_validation_exact(test_input: float, expected: float, tolerance: float):
    """Test exact numerical comparisons."""
    result = validate_numerical(test_input, expected, tolerance)
    assert result.valid
    assert result.confidence > 0.99

@pytest.mark.parametrize("test_input,expected,should_pass", [
    (4.0001, 4.0, True),
    (3.14, 3.14159, True),
    (5.0, 4.0, False),
    (-1.5, 1.5, False),
])
def test_numerical_validation_tolerance(
    test_input: float,
    expected: float,
    should_pass: bool,
    tolerance: float
):
    """Test numerical validation with tolerance."""
    result = validate_numerical(test_input, expected, tolerance)
    assert result.valid == should_pass

def test_numerical_validation_edge_cases():
    """Test numerical validation edge cases."""
    # Division by zero
    with pytest.raises(ValueError):
        validate_numerical(1/0, 0)
    
    # Infinity comparison
    result = validate_numerical(float('inf'), float('inf'))
    assert not result.valid
    
    # NaN comparison
    result = validate_numerical(float('nan'), 0)
    assert not result.valid

# Symbolic validation tests
@pytest.mark.parametrize("test_input,expected,should_match", [
    ("x + 1", "1 + x", True),
    ("2*x", "x*2", True),
    ("x**2 + 2*x + 1", "(x+1)**2", True),
    ("sin(x)", "cos(x)", False),
])
def test_symbolic_validation(
    test_input: str,
    expected: str,
    should_match: bool,
    mock_symbolic_engine
):
    """Test symbolic expression validation."""
    result = validate_symbolic(test_input, expected, engine=mock_symbolic_engine)
    assert result.valid == should_match

def test_symbolic_validation_errors():
    """Test symbolic validation error handling."""
    with pytest.raises(ValueError):
        validate_symbolic("invalid expr", "x + 1")

# Matrix validation tests
@pytest.mark.parametrize("test_input,expected", [
    (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
    (np.zeros((3, 3)), np.zeros((3, 3))),
    (np.eye(2), np.eye(2)),
])
def test_matrix_validation(test_input: np.ndarray, expected: np.ndarray):
    """Test matrix validation."""
    result = validate_matrix(test_input, expected)
    assert result.valid
    assert result.confidence > 0.99

def test_matrix_validation_shape_mismatch():
    """Test matrix validation with mismatched shapes."""
    result = validate_matrix(
        np.array([[1, 2], [3, 4]]),
        np.array([[1, 2, 3], [4, 5, 6]])
    )
    assert not result.valid
    assert "shape mismatch" in result.message.lower()

def test_matrix_validation_value_mismatch():
    """Test matrix validation with different values."""
    result = validate_matrix(
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]])
    )
    assert not result.valid
    assert result.confidence < 0.5

# Integration tests
def test_validation_result_structure():
    """Test ValidationResult structure and properties."""
    result = ValidationResult(
        valid=True,
        confidence=0.95,
        message="Test passed",
        metadata={"method": "numerical"}
    )
    assert result.valid
    assert 0 <= result.confidence <= 1
    assert isinstance(result.message, str)
    assert isinstance(result.metadata, dict)

@pytest.mark.parametrize("confidence", [-0.1, 1.1, float('nan')])
def test_validation_result_invalid_confidence(confidence: float):
    """Test ValidationResult with invalid confidence values."""
    with pytest.raises(ValueError):
        ValidationResult(valid=True, confidence=confidence)
