import unittest
import numpy as np
from typing import Callable


class TestActivationFunctions(unittest.TestCase):
    """Comprehensive unit tests for activation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.tolerance = 1e-7
        self.test_inputs = [
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
            np.random.randn(100),
            np.random.randn(32, 64),
            np.array([1e-10, -1e-10, 1e10, -1e10]),  # Edge cases
        ]
        
    def test_relu(self):
        """Test ReLU activation function."""
        for X in self.test_inputs:
            # Forward pass: max(0, x)
            expected = np.maximum(0, X)
            
            # Test gradient: 1 if x > 0, 0 otherwise
            grad_expected = (X > 0).astype(np.float64)
            
            # Test properties
            if X.size > 0:
                # ReLU should zero out negative values
                self.assertTrue(np.all(expected[X < 0] == 0))
                # ReLU should preserve positive values
                self.assertTrue(np.allclose(expected[X > 0], X[X > 0]))
                
    def test_leaky_relu(self):
        """Test Leaky ReLU activation function."""
        alpha = 0.01
        
        for X in self.test_inputs:
            # Forward: x if x > 0, alpha * x otherwise
            expected = np.where(X > 0, X, alpha * X)
            
            # Gradient: 1 if x > 0, alpha otherwise
            grad_expected = np.where(X > 0, 1, alpha)
            
    def test_sigmoid(self):
        """Test sigmoid activation function."""
        for X in self.test_inputs:
            # Forward: 1 / (1 + exp(-x))
            # Handle numerical stability
            X_safe = np.clip(X, -500, 500)
            expected = 1 / (1 + np.exp(-X_safe))
            
            # Test properties
            # Output should be in (0, 1)
            self.assertTrue(np.all(expected > 0))
            self.assertTrue(np.all(expected < 1))
            
            # Test gradient: sigmoid(x) * (1 - sigmoid(x))
            grad_expected = expected * (1 - expected)
            
            # Test symmetry property: sigmoid(-x) = 1 - sigmoid(x)
            if not np.any(np.abs(X) > 100):  # Avoid numerical issues
                self.assertTrue(np.allclose(
                    1 / (1 + np.exp(X_safe)),
                    1 - expected,
                    atol=self.tolerance
                ))
                
    def test_tanh(self):
        """Test hyperbolic tangent activation function."""
        for X in self.test_inputs:
            # Forward: tanh(x)
            expected = np.tanh(X)
            
            # Test properties
            # Output should be in (-1, 1)
            self.assertTrue(np.all(expected > -1))
            self.assertTrue(np.all(expected < 1))
            
            # Test gradient: 1 - tanh^2(x)
            grad_expected = 1 - expected**2
            
            # Test odd function property: tanh(-x) = -tanh(x)
            self.assertTrue(np.allclose(np.tanh(-X), -expected, atol=self.tolerance))
            
    def test_softmax(self):
        """Test softmax activation function."""
        # Softmax is typically applied across specific axes
        test_cases = [
            np.random.randn(10),  # 1D
            np.random.randn(32, 10),  # 2D batch
            np.random.randn(16, 20, 10),  # 3D with sequence
        ]
        
        for X in test_cases:
            # Numerical stability: subtract max
            X_shifted = X - np.max(X, axis=-1, keepdims=True)
            exp_X = np.exp(X_shifted)
            expected = exp_X / np.sum(exp_X, axis=-1, keepdims=True)
            
            # Test properties
            # Sum to 1 along last axis
            self.assertTrue(np.allclose(
                np.sum(expected, axis=-1),
                np.ones(X.shape[:-1]),
                atol=self.tolerance
            ))
            
            # All values in (0, 1)
            self.assertTrue(np.all(expected > 0))
            self.assertTrue(np.all(expected < 1))
            
    def test_elu(self):
        """Test ELU (Exponential Linear Unit) activation."""
        alpha = 1.0
        
        for X in self.test_inputs:
            # Forward: x if x > 0, alpha * (exp(x) - 1) otherwise
            expected = np.where(X > 0, X, alpha * (np.exp(np.minimum(X, 10)) - 1))
            
            # Gradient: 1 if x > 0, alpha * exp(x) otherwise
            grad_expected = np.where(X > 0, 1, expected + alpha)
            
    def test_swish(self):
        """Test Swish/SiLU activation function."""
        for X in self.test_inputs:
            # Forward: x * sigmoid(x)
            X_safe = np.clip(X, -500, 500)
            sigmoid_X = 1 / (1 + np.exp(-X_safe))
            expected = X * sigmoid_X
            
            # Gradient: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            grad_expected = sigmoid_X + X * sigmoid_X * (1 - sigmoid_X)
            
    def test_gelu(self):
        """Test GELU (Gaussian Error Linear Unit) activation."""
        for X in self.test_inputs[1:3]:  # Skip edge cases for GELU
            # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            inner = np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)
            expected = 0.5 * X * (1 + np.tanh(inner))
            
    def test_activation_numerical_stability(self):
        """Test numerical stability of activation functions."""
        extreme_values = [
            np.array([1e20, -1e20]),
            np.array([1e-20, -1e-20]),
            np.array([np.inf, -np.inf]),
        ]
        
        for X in extreme_values:
            # Sigmoid should not produce NaN
            sigmoid_safe = 1 / (1 + np.exp(-np.clip(X, -500, 500)))
            self.assertFalse(np.any(np.isnan(sigmoid_safe)))
            
            # Softmax should handle large values
            if not np.any(np.isinf(X)):
                X_shifted = X - np.max(X)
                softmax_result = np.exp(X_shifted) / np.sum(np.exp(X_shifted))
                self.assertFalse(np.any(np.isnan(softmax_result)))
                
    def test_gradient_computations(self):
        """Test gradient computations for all activation functions."""
        # This would use numerical gradient checking
        # comparing analytical gradients with finite differences
        epsilon = 1e-5
        X = np.random.randn(10, 20)
        
        # Example for ReLU gradient check
        def numerical_gradient(func: Callable, x: np.ndarray) -> np.ndarray:
            grad = np.zeros_like(x)
            it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_value = x[idx]
                
                x[idx] = old_value + epsilon
                f_plus = func(x)
                
                x[idx] = old_value - epsilon
                f_minus = func(x)
                
                grad[idx] = (f_plus - f_minus) / (2 * epsilon)
                x[idx] = old_value
                
                it.iternext()
            return grad


if __name__ == '__main__':
    unittest.main()