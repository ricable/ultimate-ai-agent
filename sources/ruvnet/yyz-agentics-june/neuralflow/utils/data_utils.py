"""
Data utilities for NeuralFlow
"""
import numpy as np
from typing import Tuple, Optional, Union


def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2,
                    random_state: Optional[int] = None,
                    shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Parameters:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
        shuffle: Whether to shuffle before splitting
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    test_samples = int(n_samples * test_size)
    
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    return (X[train_indices], X[test_indices], 
            y[train_indices], y[test_indices])


def to_categorical(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert class labels to one-hot encoding.
    
    Parameters:
        y: Class labels (integers)
        num_classes: Total number of classes
    
    Returns:
        One-hot encoded labels
    """
    y = np.array(y, dtype='int')
    
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    n_samples = y.shape[0]
    categorical = np.zeros((n_samples, num_classes))
    categorical[np.arange(n_samples), y] = 1
    
    return categorical


def normalize(X: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize data to have zero mean and unit variance.
    
    Parameters:
        X: Data to normalize
        axis: Axis along which to normalize
    
    Returns:
        Normalized data
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    return (X - mean) / std


def minmax_scale(X: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Scale features to a given range.
    
    Parameters:
        X: Data to scale
        feature_range: Desired range of transformed data
    
    Returns:
        Scaled data
    """
    X_min = np.min(X, axis=0, keepdims=True)
    X_max = np.max(X, axis=0, keepdims=True)
    
    # Avoid division by zero
    X_range = X_max - X_min
    X_range = np.where(X_range == 0, 1, X_range)
    
    X_std = (X - X_min) / X_range
    X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return X_scaled


def shuffle_data(X: np.ndarray, y: np.ndarray, 
                random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle data and labels together.
    
    Parameters:
        X: Features
        y: Labels
        random_state: Random seed
    
    Returns:
        Shuffled X and y
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.permutation(X.shape[0])
    
    return X[indices], y[indices]


def batch_generator(X: np.ndarray, y: np.ndarray, 
                   batch_size: int = 32,
                   shuffle: bool = True):
    """
    Generate batches of data.
    
    Parameters:
        X: Features
        y: Labels
        batch_size: Size of each batch
        shuffle: Whether to shuffle data before each epoch
        
    Yields:
        Batches of (X_batch, y_batch)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield X[batch_indices], y[batch_indices]


def pad_sequences(sequences: list, maxlen: Optional[int] = None,
                 padding: str = 'pre', truncating: str = 'pre',
                 value: float = 0.0) -> np.ndarray:
    """
    Pad sequences to the same length.
    
    Parameters:
        sequences: List of sequences
        maxlen: Maximum length
        padding: 'pre' or 'post'
        truncating: 'pre' or 'post'
        value: Padding value
    
    Returns:
        Padded sequences
    """
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded = np.full((len(sequences), maxlen), value)
    
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]
        
        if padding == 'pre':
            padded[i, -len(seq):] = seq
        else:
            padded[i, :len(seq)] = seq
    
    return padded


def load_mnist_sample() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a small sample of MNIST-like data for demos.
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Generate synthetic MNIST-like data
    np.random.seed(42)
    
    # Training data
    n_train = 1000
    X_train = np.random.randn(n_train, 28, 28, 1) * 0.3
    y_train = np.random.randint(0, 10, n_train)
    
    # Add some pattern to make it learnable
    for i in range(n_train):
        label = y_train[i]
        # Add a simple pattern for each digit
        X_train[i, 10:18, 10:18, 0] += label * 0.1
        X_train[i, label:label+5, 5:10, 0] += 0.5
    
    # Test data
    n_test = 200
    X_test = np.random.randn(n_test, 28, 28, 1) * 0.3
    y_test = np.random.randint(0, 10, n_test)
    
    for i in range(n_test):
        label = y_test[i]
        X_test[i, 10:18, 10:18, 0] += label * 0.1
        X_test[i, label:label+5, 5:10, 0] += 0.5
    
    # Normalize
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    
    return X_train, y_train, X_test, y_test


def generate_spiral_data(n_points: int = 100, n_classes: int = 2, 
                        noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spiral dataset for classification demos.
    
    Parameters:
        n_points: Number of points per class
        n_classes: Number of classes
        noise: Noise level
    
    Returns:
        X, y
    """
    X = []
    y = []
    
    for class_idx in range(n_classes):
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_points) + \
            np.random.randn(n_points) * noise
        
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        
        X.append(np.c_[x1, x2])
        y.append(np.full(n_points, class_idx))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    
    return X[indices], y[indices]