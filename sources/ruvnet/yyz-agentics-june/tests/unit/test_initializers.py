import unittest
import numpy as np
from typing import Tuple, Callable


class TestWeightInitializers(unittest.TestCase):
    """Comprehensive unit tests for weight initialization strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.tolerance = 1e-5
        self.num_samples = 10000  # For statistical tests
        
    def test_zeros_initializer(self):
        """Test zeros initialization."""
        shapes = [(10,), (10, 20), (5, 5, 3, 3)]
        
        for shape in shapes:
            weights = np.zeros(shape)
            
            # All values should be zero
            self.assertTrue(np.all(weights == 0))
            self.assertEqual(weights.shape, shape)
            
    def test_ones_initializer(self):
        """Test ones initialization."""
        shapes = [(10,), (10, 20), (5, 5, 3, 3)]
        
        for shape in shapes:
            weights = np.ones(shape)
            
            # All values should be one
            self.assertTrue(np.all(weights == 1))
            self.assertEqual(weights.shape, shape)
            
    def test_constant_initializer(self):
        """Test constant value initialization."""
        value = 0.1
        shapes = [(10,), (10, 20), (5, 5, 3, 3)]
        
        for shape in shapes:
            weights = np.full(shape, value)
            
            # All values should equal the constant
            self.assertTrue(np.all(weights == value))
            self.assertEqual(weights.shape, shape)
            
    def test_random_normal_initializer(self):
        """Test normal/Gaussian initialization."""
        mean = 0.0
        stddev = 1.0
        shape = (self.num_samples,)
        
        weights = np.random.normal(mean, stddev, shape)
        
        # Test statistical properties
        self.assertAlmostEqual(np.mean(weights), mean, places=1)
        self.assertAlmostEqual(np.std(weights), stddev, places=1)
        
        # Test different configurations
        configs = [
            {'mean': 0.0, 'stddev': 0.01},
            {'mean': 0.0, 'stddev': 0.1},
            {'mean': 0.0, 'stddev': 1.0},
        ]
        
        for config in configs:
            weights = np.random.normal(config['mean'], config['stddev'], (1000,))
            self.assertAlmostEqual(np.mean(weights), config['mean'], places=1)
            self.assertAlmostEqual(np.std(weights), config['stddev'], places=1)
            
    def test_random_uniform_initializer(self):
        """Test uniform initialization."""
        minval = -0.5
        maxval = 0.5
        shape = (self.num_samples,)
        
        weights = np.random.uniform(minval, maxval, shape)
        
        # Test bounds
        self.assertTrue(np.all(weights >= minval))
        self.assertTrue(np.all(weights <= maxval))
        
        # Test statistical properties
        expected_mean = (minval + maxval) / 2
        expected_var = (maxval - minval)**2 / 12
        
        self.assertAlmostEqual(np.mean(weights), expected_mean, places=1)
        self.assertAlmostEqual(np.var(weights), expected_var, places=1)
        
    def test_xavier_glorot_uniform(self):
        """Test Xavier/Glorot uniform initialization."""
        fan_in = 100
        fan_out = 200
        
        # Xavier uniform: U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        shape = (fan_in, fan_out)
        weights = np.random.uniform(-limit, limit, shape)
        
        # Test bounds
        self.assertTrue(np.all(weights >= -limit))
        self.assertTrue(np.all(weights <= limit))
        
        # Test variance
        expected_var = 2.0 / (fan_in + fan_out)
        self.assertAlmostEqual(np.var(weights), expected_var, places=2)
        
    def test_xavier_glorot_normal(self):
        """Test Xavier/Glorot normal initialization."""
        fan_in = 100
        fan_out = 200
        
        # Xavier normal: N(0, sqrt(2/(fan_in+fan_out)))
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        
        shape = (fan_in, fan_out)
        weights = np.random.normal(0, stddev, shape)
        
        # Test statistical properties
        self.assertAlmostEqual(np.mean(weights), 0.0, places=1)
        self.assertAlmostEqual(np.std(weights), stddev, places=2)
        
    def test_he_kaiming_uniform(self):
        """Test He/Kaiming uniform initialization."""
        fan_in = 100
        
        # He uniform: U[-sqrt(6/fan_in), sqrt(6/fan_in)]
        limit = np.sqrt(6.0 / fan_in)
        
        shape = (fan_in, 200)
        weights = np.random.uniform(-limit, limit, shape)
        
        # Test bounds
        self.assertTrue(np.all(weights >= -limit))
        self.assertTrue(np.all(weights <= limit))
        
        # Variance should be 2/fan_in
        expected_var = 2.0 / fan_in
        self.assertAlmostEqual(np.var(weights), expected_var, places=2)
        
    def test_he_kaiming_normal(self):
        """Test He/Kaiming normal initialization."""
        fan_in = 100
        
        # He normal: N(0, sqrt(2/fan_in))
        stddev = np.sqrt(2.0 / fan_in)
        
        shape = (fan_in, 200)
        weights = np.random.normal(0, stddev, shape)
        
        # Test statistical properties
        self.assertAlmostEqual(np.mean(weights), 0.0, places=1)
        self.assertAlmostEqual(np.std(weights), stddev, places=2)
        
    def test_lecun_initialization(self):
        """Test LeCun initialization."""
        fan_in = 100
        
        # LeCun: N(0, sqrt(1/fan_in))
        stddev = np.sqrt(1.0 / fan_in)
        
        shape = (fan_in, 200)
        weights = np.random.normal(0, stddev, shape)
        
        # Test statistical properties
        self.assertAlmostEqual(np.mean(weights), 0.0, places=1)
        self.assertAlmostEqual(np.std(weights), stddev, places=2)
        
    def test_orthogonal_initialization(self):
        """Test orthogonal initialization."""
        shape = (100, 100)
        
        # Generate random matrix and compute QR decomposition
        random_matrix = np.random.randn(*shape)
        Q, R = np.linalg.qr(random_matrix)
        
        # Fix signs
        d = np.diag(R)
        Q *= np.sign(d)
        
        weights = Q
        
        # Test orthogonality: W @ W.T should be identity
        product = weights @ weights.T
        identity = np.eye(shape[0])
        
        self.assertTrue(np.allclose(product, identity, atol=1e-6))
        
        # Test for non-square matrices
        shape = (100, 50)
        random_matrix = np.random.randn(*shape)
        Q, R = np.linalg.qr(random_matrix)
        weights = Q
        
        # Columns should be orthonormal
        product = weights.T @ weights
        identity = np.eye(shape[1])
        self.assertTrue(np.allclose(product, identity, atol=1e-6))
        
    def test_sparse_initialization(self):
        """Test sparse initialization."""
        sparsity = 0.1  # 10% non-zero
        shape = (1000, 1000)
        
        # Create sparse matrix
        weights = np.zeros(shape)
        num_nonzero = int(np.prod(shape) * sparsity)
        indices = np.random.choice(np.prod(shape), num_nonzero, replace=False)
        
        # Set non-zero values
        flat_weights = weights.flatten()
        flat_weights[indices] = np.random.normal(0, 0.01, num_nonzero)
        weights = flat_weights.reshape(shape)
        
        # Test sparsity
        actual_sparsity = np.sum(weights != 0) / np.prod(shape)
        self.assertAlmostEqual(actual_sparsity, sparsity, places=2)
        
    def test_identity_initialization(self):
        """Test identity matrix initialization (for RNNs)."""
        size = 100
        
        weights = np.eye(size)
        
        # Should be identity matrix
        self.assertTrue(np.allclose(weights, np.eye(size)))
        
        # With gain
        gain = 0.5
        weights = gain * np.eye(size)
        self.assertTrue(np.allclose(weights, gain * np.eye(size)))
        
    def test_initialization_for_different_activations(self):
        """Test appropriate initialization for different activation functions."""
        fan_in = 100
        fan_out = 200
        
        # ReLU: He initialization
        relu_std = np.sqrt(2.0 / fan_in)
        
        # Tanh: Xavier initialization  
        tanh_std = np.sqrt(2.0 / (fan_in + fan_out))
        
        # Sigmoid: Xavier initialization with adjusted factor
        sigmoid_std = np.sqrt(2.0 / (fan_in + fan_out))
        
        # SELU: Special initialization
        selu_std = np.sqrt(1.0 / fan_in)
        
        # Test that different activations use different scales
        self.assertNotAlmostEqual(relu_std, tanh_std)
        self.assertNotAlmostEqual(relu_std, selu_std)
        
    def test_fan_in_fan_out_computation(self):
        """Test fan-in/fan-out computation for different layer types."""
        # Dense layer
        dense_shape = (100, 200)
        fan_in = dense_shape[0]
        fan_out = dense_shape[1]
        self.assertEqual(fan_in, 100)
        self.assertEqual(fan_out, 200)
        
        # Conv2D layer
        conv_shape = (64, 32, 3, 3)  # (out_channels, in_channels, height, width)
        receptive_field_size = conv_shape[2] * conv_shape[3]
        fan_in = conv_shape[1] * receptive_field_size
        fan_out = conv_shape[0] * receptive_field_size
        self.assertEqual(fan_in, 32 * 9)
        self.assertEqual(fan_out, 64 * 9)
        
    def test_initialization_reproducibility(self):
        """Test that initialization is reproducible with fixed seed."""
        shape = (100, 200)
        
        # First run
        np.random.seed(42)
        weights1 = np.random.randn(*shape)
        
        # Second run with same seed
        np.random.seed(42)
        weights2 = np.random.randn(*shape)
        
        # Should be identical
        self.assertTrue(np.allclose(weights1, weights2))
        
        # Different seed should give different results
        np.random.seed(43)
        weights3 = np.random.randn(*shape)
        self.assertFalse(np.allclose(weights1, weights3))


class TestBiasInitializers(unittest.TestCase):
    """Test bias initialization strategies."""
    
    def test_zero_bias_initialization(self):
        """Test zero initialization for biases (most common)."""
        shapes = [(10,), (100,), (1000,)]
        
        for shape in shapes:
            bias = np.zeros(shape)
            self.assertTrue(np.all(bias == 0))
            
    def test_constant_bias_initialization(self):
        """Test constant initialization for biases."""
        value = 0.1
        shape = (100,)
        
        bias = np.full(shape, value)
        self.assertTrue(np.all(bias == value))
        
    def test_lstm_bias_initialization(self):
        """Test special LSTM bias initialization."""
        hidden_size = 128
        
        # LSTM has 4 gates, often forget gate bias is initialized to 1
        bias = np.zeros(4 * hidden_size)
        
        # Set forget gate bias to 1
        forget_gate_start = hidden_size
        forget_gate_end = 2 * hidden_size
        bias[forget_gate_start:forget_gate_end] = 1.0
        
        # Test
        self.assertTrue(np.all(bias[:hidden_size] == 0))  # Input gate
        self.assertTrue(np.all(bias[hidden_size:2*hidden_size] == 1))  # Forget gate
        self.assertTrue(np.all(bias[2*hidden_size:3*hidden_size] == 0))  # Cell gate
        self.assertTrue(np.all(bias[3*hidden_size:] == 0))  # Output gate


if __name__ == '__main__':
    unittest.main()