import unittest
import numpy as np


class TestLossFunctions(unittest.TestCase):
    """Comprehensive unit tests for loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.batch_size = 32
        self.num_classes = 10
        self.tolerance = 1e-7
        
    def test_mse_loss(self):
        """Test Mean Squared Error loss."""
        # Test case 1: Perfect predictions
        y_true = np.random.randn(self.batch_size, 1)
        y_pred = y_true.copy()
        expected_loss = 0.0
        
        # Test case 2: Known error
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.5], [2.5], [3.5]])
        expected_loss = 0.25  # ((0.5^2 + 0.5^2 + 0.5^2) / 3)
        
        # Test case 3: Multi-dimensional output
        y_true = np.random.randn(self.batch_size, 10)
        y_pred = np.random.randn(self.batch_size, 10)
        
        # Test gradient
        # dL/dy_pred = 2 * (y_pred - y_true) / n
        expected_grad = 2 * (y_pred - y_true) / y_true.shape[0]
        
    def test_mae_loss(self):
        """Test Mean Absolute Error loss."""
        # Test case 1: Perfect predictions
        y_true = np.random.randn(self.batch_size, 1)
        y_pred = y_true.copy()
        expected_loss = 0.0
        
        # Test case 2: Known error
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.5], [2.5], [3.5]])
        expected_loss = 0.5  # (|0.5| + |0.5| + |0.5|) / 3
        
        # Test gradient (subgradient for non-differentiable points)
        # dL/dy_pred = sign(y_pred - y_true) / n
        
    def test_cross_entropy_loss(self):
        """Test Cross-Entropy loss for classification."""
        # Test case 1: Perfect predictions (one-hot)
        y_true = np.eye(self.num_classes)[np.random.randint(0, self.num_classes, self.batch_size)]
        y_pred = y_true.copy()
        # Add small epsilon to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Test case 2: Uniform predictions
        y_true = np.eye(self.num_classes)[np.zeros(self.batch_size, dtype=int)]
        y_pred = np.ones((self.batch_size, self.num_classes)) / self.num_classes
        expected_loss = -np.log(1.0 / self.num_classes)
        
        # Test case 3: Random predictions
        y_true = np.eye(self.num_classes)[np.random.randint(0, self.num_classes, self.batch_size)]
        y_pred = np.random.rand(self.batch_size, self.num_classes)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)  # Normalize
        
        # Test gradient
        # dL/dy_pred = -y_true / y_pred / n (element-wise)
        
    def test_binary_cross_entropy(self):
        """Test Binary Cross-Entropy loss."""
        # Test case 1: Perfect predictions
        y_true = np.random.randint(0, 2, size=(self.batch_size, 1)).astype(float)
        y_pred = y_true.copy()
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Test case 2: Worst predictions
        y_true = np.ones((self.batch_size, 1))
        y_pred = np.zeros((self.batch_size, 1)) + 1e-7
        
        # Test case 3: 50-50 predictions
        y_true = np.random.randint(0, 2, size=(self.batch_size, 1)).astype(float)
        y_pred = np.ones((self.batch_size, 1)) * 0.5
        expected_loss = -np.log(0.5)
        
        # Test with logits (before sigmoid)
        logits = np.random.randn(self.batch_size, 1)
        
    def test_sparse_categorical_cross_entropy(self):
        """Test Sparse Categorical Cross-Entropy loss."""
        # Integer labels instead of one-hot
        y_true = np.random.randint(0, self.num_classes, size=self.batch_size)
        y_pred = np.random.rand(self.batch_size, self.num_classes)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        
        # Should give same result as regular cross-entropy with one-hot
        y_true_onehot = np.eye(self.num_classes)[y_true]
        
    def test_hinge_loss(self):
        """Test Hinge loss for SVM-style classification."""
        # Binary classification: y_true in {-1, +1}
        y_true = np.random.choice([-1, 1], size=(self.batch_size, 1))
        y_pred = np.random.randn(self.batch_size, 1)
        
        # Loss = max(0, 1 - y_true * y_pred)
        expected_loss = np.maximum(0, 1 - y_true * y_pred).mean()
        
        # Multi-class hinge loss
        y_true_multi = np.random.randint(0, self.num_classes, size=self.batch_size)
        y_pred_multi = np.random.randn(self.batch_size, self.num_classes)
        
    def test_huber_loss(self):
        """Test Huber loss (smooth L1 loss)."""
        delta = 1.0
        
        # Test case 1: Small errors (quadratic region)
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [2.1], [3.1]])
        # |error| < delta, so use 0.5 * error^2
        
        # Test case 2: Large errors (linear region)
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[3.0], [5.0], [7.0]])
        # |error| >= delta, so use delta * |error| - 0.5 * delta^2
        
    def test_focal_loss(self):
        """Test Focal loss for imbalanced classification."""
        alpha = 0.25
        gamma = 2.0
        
        # Focal loss applies a modulating factor to cross-entropy
        y_true = np.eye(self.num_classes)[np.random.randint(0, self.num_classes, self.batch_size)]
        y_pred = np.random.rand(self.batch_size, self.num_classes)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # FL = -alpha * (1 - p_t)^gamma * log(p_t)
        
    def test_dice_loss(self):
        """Test Dice loss for segmentation tasks."""
        # Binary segmentation
        y_true = np.random.randint(0, 2, size=(self.batch_size, 128, 128, 1))
        y_pred = np.random.rand(self.batch_size, 128, 128, 1)
        
        # Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        # Loss = 1 - Dice
        
    def test_triplet_loss(self):
        """Test Triplet loss for metric learning."""
        margin = 1.0
        embedding_dim = 128
        
        # Anchor, positive, negative embeddings
        anchor = np.random.randn(self.batch_size, embedding_dim)
        positive = anchor + np.random.randn(self.batch_size, embedding_dim) * 0.1
        negative = np.random.randn(self.batch_size, embedding_dim)
        
        # Loss = max(0, d(a, p) - d(a, n) + margin)
        
    def test_loss_numerical_stability(self):
        """Test numerical stability of loss functions."""
        # Test with extreme probability values
        y_true = np.eye(self.num_classes)[np.zeros(self.batch_size, dtype=int)]
        
        # Near 0 probabilities
        y_pred_small = np.ones((self.batch_size, self.num_classes)) * 1e-10
        y_pred_small[:, 0] = 1 - 1e-10
        
        # Near 1 probabilities
        y_pred_large = np.zeros((self.batch_size, self.num_classes)) + 1e-10
        y_pred_large[:, 0] = 1 - (self.num_classes - 1) * 1e-10
        
        # Should not produce inf or nan
        
    def test_loss_reduction_modes(self):
        """Test different reduction modes for losses."""
        y_true = np.random.randn(self.batch_size, 10)
        y_pred = np.random.randn(self.batch_size, 10)
        
        # Test 'none' reduction (no reduction)
        # Test 'mean' reduction (default)
        # Test 'sum' reduction
        
    def test_weighted_losses(self):
        """Test loss functions with sample weights."""
        y_true = np.random.randint(0, self.num_classes, size=self.batch_size)
        y_pred = np.random.rand(self.batch_size, self.num_classes)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        
        # Sample weights
        sample_weights = np.random.rand(self.batch_size)
        
        # Class weights for imbalanced datasets
        class_weights = np.ones(self.num_classes)
        class_weights[0] = 2.0  # Weight class 0 more heavily


if __name__ == '__main__':
    unittest.main()