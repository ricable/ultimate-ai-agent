"""
Weight initialization methods for neural networks.
Pure NumPy implementation.
"""

import numpy as np


class Initializer:
    """Base class for weight initializers."""
    
    def __call__(self, shape, dtype=np.float32):
        raise NotImplementedError


class RandomNormal(Initializer):
    """Random normal initialization."""
    
    def __init__(self, mean=0.0, std=0.05, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed
        
    def __call__(self, shape, dtype=np.float32):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(self.mean, self.std, shape).astype(dtype)


class RandomUniform(Initializer):
    """Random uniform initialization."""
    
    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        
    def __call__(self, shape, dtype=np.float32):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(self.minval, self.maxval, shape).astype(dtype)


class XavierNormal(Initializer):
    """Xavier/Glorot normal initialization."""
    
    def __init__(self, seed=None):
        self.seed = seed
        
    def __call__(self, shape, dtype=np.float32):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        fan_in, fan_out = self._compute_fans(shape)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, shape).astype(dtype)
    
    def _compute_fans(self, shape):
        if len(shape) == 2:  # Dense layer
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) == 4:  # Conv2D layer (out_channels, in_channels, height, width)
            receptive_field_size = shape[2] * shape[3]
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        else:
            fan_in = fan_out = np.prod(shape) // 2
        return fan_in, fan_out


class XavierUniform(Initializer):
    """Xavier/Glorot uniform initialization."""
    
    def __init__(self, seed=None):
        self.seed = seed
        
    def __call__(self, shape, dtype=np.float32):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        fan_in, fan_out = self._compute_fans(shape)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape).astype(dtype)
    
    def _compute_fans(self, shape):
        if len(shape) == 2:  # Dense layer
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) == 4:  # Conv2D layer
            receptive_field_size = shape[2] * shape[3]
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        else:
            fan_in = fan_out = np.prod(shape) // 2
        return fan_in, fan_out


class HeNormal(Initializer):
    """He normal initialization (for ReLU networks)."""
    
    def __init__(self, seed=None):
        self.seed = seed
        
    def __call__(self, shape, dtype=np.float32):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        fan_in = self._compute_fan_in(shape)
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape).astype(dtype)
    
    def _compute_fan_in(self, shape):
        if len(shape) == 2:  # Dense layer
            fan_in = shape[0]
        elif len(shape) == 4:  # Conv2D layer
            receptive_field_size = shape[2] * shape[3]
            fan_in = shape[1] * receptive_field_size
        else:
            fan_in = np.prod(shape[:-1])
        return fan_in


class HeUniform(Initializer):
    """He uniform initialization (for ReLU networks)."""
    
    def __init__(self, seed=None):
        self.seed = seed
        
    def __call__(self, shape, dtype=np.float32):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        fan_in = self._compute_fan_in(shape)
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, shape).astype(dtype)
    
    def _compute_fan_in(self, shape):
        if len(shape) == 2:  # Dense layer
            fan_in = shape[0]
        elif len(shape) == 4:  # Conv2D layer
            receptive_field_size = shape[2] * shape[3]
            fan_in = shape[1] * receptive_field_size
        else:
            fan_in = np.prod(shape[:-1])
        return fan_in


class Zeros(Initializer):
    """Initialize weights to zeros."""
    
    def __call__(self, shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)


class Ones(Initializer):
    """Initialize weights to ones."""
    
    def __call__(self, shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)


# Convenience functions
def get_initializer(name):
    """Get initializer by name."""
    initializers = {
        'random_normal': RandomNormal(),
        'random_uniform': RandomUniform(),
        'xavier_normal': XavierNormal(),
        'glorot_normal': XavierNormal(),  # Alias
        'xavier_uniform': XavierUniform(),
        'glorot_uniform': XavierUniform(),  # Alias
        'he_normal': HeNormal(),
        'he_uniform': HeUniform(),
        'zeros': Zeros(),
        'ones': Ones()
    }
    
    if name not in initializers:
        raise ValueError(f"Unknown initializer: {name}")
    
    return initializers[name]