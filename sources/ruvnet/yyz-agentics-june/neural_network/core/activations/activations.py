"""
Activation functions for neural networks.
Pure NumPy implementation with forward and backward propagation.
"""

import numpy as np


class Activation:
    """Base class for activation functions."""
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x, grad_output):
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x, grad_output):
        return grad_output * (x > 0)


class LeakyReLU(Activation):
    """Leaky Rectified Linear Unit activation function."""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x, grad_output):
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        return grad_output * dx


class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def forward(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, x, grad_output):
        sigmoid_x = self.forward(x)
        return grad_output * sigmoid_x * (1 - sigmoid_x)


class Tanh(Activation):
    """Hyperbolic tangent activation function."""
    
    def forward(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return np.tanh(x)
    
    def backward(self, x, grad_output):
        tanh_x = self.forward(x)
        return grad_output * (1 - tanh_x ** 2)


class Softmax(Activation):
    """Softmax activation function."""
    
    def __init__(self, axis=-1):
        self.axis = axis
    
    def forward(self, x):
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
    
    def backward(self, x, grad_output):
        # Softmax gradient is more complex
        # For simplicity, we'll compute the full Jacobian for each sample
        softmax_x = self.forward(x)
        
        # Handle different input shapes
        if x.ndim == 2:
            batch_size, num_classes = x.shape
            grad_input = np.zeros_like(x)
            
            for i in range(batch_size):
                s = softmax_x[i].reshape(-1, 1)
                jacobian = np.diagflat(s) - np.dot(s, s.T)
                grad_input[i] = np.dot(jacobian, grad_output[i])
            
            return grad_input
        else:
            # For other dimensions, reshape to 2D, compute, then reshape back
            original_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            grad_2d = self.backward(x_2d, grad_output.reshape(-1, grad_output.shape[-1]))
            return grad_2d.reshape(original_shape)


class ELU(Activation):
    """Exponential Linear Unit activation function."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x, grad_output):
        return grad_output * np.where(x > 0, 1, self.alpha * np.exp(x))


class Swish(Activation):
    """Swish activation function (x * sigmoid(x))."""
    
    def forward(self, x):
        x_clipped = np.clip(x, -500, 500)
        sigmoid_x = 1.0 / (1.0 + np.exp(-x_clipped))
        return x * sigmoid_x
    
    def backward(self, x, grad_output):
        x_clipped = np.clip(x, -500, 500)
        sigmoid_x = 1.0 / (1.0 + np.exp(-x_clipped))
        swish_x = x * sigmoid_x
        return grad_output * (swish_x + sigmoid_x * (1 - swish_x))


class GELU(Activation):
    """Gaussian Error Linear Unit activation function."""
    
    def forward(self, x):
        # Approximation of GELU
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def backward(self, x, grad_output):
        # Derivative of GELU approximation
        tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        tanh_val = np.tanh(tanh_arg)
        sech2_val = 1 - tanh_val**2
        
        derivative = 0.5 * (1 + tanh_val) + \
                     0.5 * x * sech2_val * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        
        return grad_output * derivative


# Convenience functions
def get_activation(name):
    """Get activation function by name."""
    activations = {
        'relu': ReLU(),
        'leaky_relu': LeakyReLU(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'softmax': Softmax(),
        'elu': ELU(),
        'swish': Swish(),
        'gelu': GELU()
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    
    return activations[name]