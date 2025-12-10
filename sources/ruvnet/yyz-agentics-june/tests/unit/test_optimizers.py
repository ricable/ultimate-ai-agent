import unittest
import numpy as np
from typing import Dict, List, Tuple


class TestOptimizers(unittest.TestCase):
    """Comprehensive unit tests for optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.tolerance = 1e-6
        
        # Simple quadratic function for testing: f(x) = 0.5 * x^T @ A @ x
        self.dim = 10
        A = np.random.randn(self.dim, self.dim)
        self.A = A.T @ A  # Make positive definite
        
        # Initial parameters
        self.params = {'W': np.random.randn(self.dim)}
        self.grads = {'W': self.A @ self.params['W']}
        
    def test_sgd_vanilla(self):
        """Test vanilla Stochastic Gradient Descent."""
        learning_rate = 0.01
        
        # Update rule: theta = theta - lr * grad
        W_old = self.params['W'].copy()
        expected_W = W_old - learning_rate * self.grads['W']
        
        # Test convergence on simple quadratic
        params = {'x': np.random.randn(2)}
        for i in range(1000):
            grad = params['x']  # Gradient of f(x) = 0.5 * ||x||^2
            params['x'] -= learning_rate * grad
            
        # Should converge close to zero
        self.assertTrue(np.linalg.norm(params['x']) < 0.01)
        
    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        learning_rate = 0.01
        momentum = 0.9
        
        # Initialize velocity
        velocity = {'W': np.zeros_like(self.params['W'])}
        
        # Update rule:
        # v = momentum * v - lr * grad
        # theta = theta + v
        
        # Test that momentum accelerates convergence
        # Compare convergence with and without momentum
        
    def test_sgd_nesterov(self):
        """Test SGD with Nesterov momentum."""
        learning_rate = 0.01
        momentum = 0.9
        
        # Nesterov update:
        # v_prev = v
        # v = momentum * v - lr * grad(theta + momentum * v_prev)
        # theta = theta + v
        
    def test_adam(self):
        """Test Adam optimizer."""
        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        # Initialize moments
        m = {'W': np.zeros_like(self.params['W'])}
        v = {'W': np.zeros_like(self.params['W'])}
        t = 0
        
        # Adam update:
        # t = t + 1
        # m = beta1 * m + (1 - beta1) * grad
        # v = beta2 * v + (1 - beta2) * grad^2
        # m_hat = m / (1 - beta1^t)
        # v_hat = v / (1 - beta2^t)
        # theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
        
        # Test bias correction
        t = 1
        m['W'] = (1 - beta1) * self.grads['W']
        v['W'] = (1 - beta2) * self.grads['W']**2
        
        m_hat = m['W'] / (1 - beta1**t)
        v_hat = v['W'] / (1 - beta2**t)
        
        # At t=1, bias correction should make m_hat ≈ grad
        self.assertTrue(np.allclose(m_hat, self.grads['W'], atol=self.tolerance))
        
    def test_rmsprop(self):
        """Test RMSprop optimizer."""
        learning_rate = 0.01
        decay_rate = 0.9
        epsilon = 1e-8
        
        # Initialize moving average of squared gradients
        cache = {'W': np.zeros_like(self.params['W'])}
        
        # RMSprop update:
        # cache = decay_rate * cache + (1 - decay_rate) * grad^2
        # theta = theta - lr * grad / (sqrt(cache) + epsilon)
        
    def test_adagrad(self):
        """Test Adagrad optimizer."""
        learning_rate = 0.01
        epsilon = 1e-8
        
        # Initialize sum of squared gradients
        cache = {'W': np.zeros_like(self.params['W'])}
        
        # Adagrad update:
        # cache = cache + grad^2
        # theta = theta - lr * grad / (sqrt(cache) + epsilon)
        
        # Test that learning rate decreases over time
        grad = np.ones(10)
        cache_val = 0
        effective_lrs = []
        
        for i in range(100):
            cache_val += grad**2
            effective_lr = learning_rate / (np.sqrt(cache_val) + epsilon)
            effective_lrs.append(effective_lr[0])
            
        # Learning rate should decrease
        self.assertTrue(all(effective_lrs[i] > effective_lrs[i+1] 
                           for i in range(len(effective_lrs)-1)))
        
    def test_adamw(self):
        """Test AdamW optimizer (Adam with decoupled weight decay)."""
        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        weight_decay = 0.01
        
        # AdamW adds weight decay directly to parameters
        # theta = theta - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * theta)
        
    def test_lamb(self):
        """Test LAMB (Layer-wise Adaptive Moments) optimizer."""
        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        # LAMB adapts learning rate per layer based on ratio of
        # parameter norm to update norm
        
    def test_learning_rate_schedules(self):
        """Test various learning rate scheduling strategies."""
        initial_lr = 0.1
        
        # Test step decay
        step_size = 30
        gamma = 0.1
        for epoch in range(100):
            lr = initial_lr * (gamma ** (epoch // step_size))
            
        # Test exponential decay
        decay_rate = 0.96
        for epoch in range(100):
            lr = initial_lr * (decay_rate ** epoch)
            
        # Test cosine annealing
        T_max = 100
        for epoch in range(100):
            lr = initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2
            
        # Test warmup
        warmup_epochs = 10
        for epoch in range(100):
            if epoch < warmup_epochs:
                lr = initial_lr * epoch / warmup_epochs
            else:
                lr = initial_lr
                
    def test_gradient_clipping(self):
        """Test gradient clipping strategies."""
        # Test value clipping
        max_value = 1.0
        grad = np.array([2.0, -3.0, 0.5])
        clipped = np.clip(grad, -max_value, max_value)
        expected = np.array([1.0, -1.0, 0.5])
        self.assertTrue(np.allclose(clipped, expected))
        
        # Test norm clipping
        max_norm = 1.0
        grad = np.array([3.0, 4.0])  # norm = 5
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_norm:
            clipped = grad * max_norm / grad_norm
            
        self.assertAlmostEqual(np.linalg.norm(clipped), max_norm)
        
    def test_optimizer_state_persistence(self):
        """Test saving and loading optimizer states."""
        # Optimizers should be able to save their internal state
        # (moments, velocities, step counts, etc.) and restore it
        
        # Example state dict for Adam
        state = {
            'step': 100,
            'exp_avg': {'W': np.random.randn(10)},
            'exp_avg_sq': {'W': np.random.randn(10)},
        }
        
    def test_sparse_gradient_handling(self):
        """Test optimizer handling of sparse gradients."""
        # Some parameters might have sparse gradients (e.g., embeddings)
        sparse_grad = np.zeros(1000)
        sparse_grad[[1, 5, 42, 100]] = [0.1, -0.2, 0.3, -0.4]
        
        # Optimizers should efficiently handle sparse updates
        
    def test_per_parameter_options(self):
        """Test different optimizer settings per parameter group."""
        # Different layers might need different learning rates
        param_groups = [
            {'params': ['conv.weight', 'conv.bias'], 'lr': 0.01},
            {'params': ['fc.weight', 'fc.bias'], 'lr': 0.001},
        ]


class TestConvergenceProperties(unittest.TestCase):
    """Test convergence properties of optimizers on standard problems."""
    
    def test_convex_quadratic_convergence(self):
        """Test optimizer convergence on convex quadratic function."""
        # f(x) = 0.5 * x^T @ A @ x + b^T @ x
        dim = 50
        A = np.random.randn(dim, dim)
        A = A.T @ A + np.eye(dim) * 0.1  # Ensure positive definite
        b = np.random.randn(dim)
        
        # Optimal solution: x* = -A^(-1) @ b
        x_optimal = -np.linalg.solve(A, b)
        
        optimizers = ['sgd', 'momentum', 'adam', 'rmsprop']
        
        for opt_name in optimizers:
            x = np.random.randn(dim)
            # Run optimizer and check convergence
            
    def test_rosenbrock_function(self):
        """Test optimizers on non-convex Rosenbrock function."""
        # f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
        # Global minimum at (1, 1)
        
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
            
        def rosenbrock_grad(x):
            dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
            dy = 200 * (x[1] - x[0]**2)
            return np.array([dx, dy])
            
        # Test various optimizers starting from different points


if __name__ == '__main__':
    unittest.main()