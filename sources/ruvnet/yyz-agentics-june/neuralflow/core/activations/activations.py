"""
Activation functions implementation
"""
import numpy as np
from typing import Optional, Union
from ..tensor import Tensor


# Functional API
def relu(x: Tensor) -> Tensor:
    """ReLU activation function."""
    out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
    
    def _backward(grad):
        if x.requires_grad:
            x.backward(grad * (x.data > 0))
            
    if out.requires_grad:
        out._grad_fn = _backward
        out._prev = {x}
        
    return out


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Leaky ReLU activation function."""
    out = Tensor(np.where(x.data > 0, x.data, alpha * x.data), requires_grad=x.requires_grad)
    
    def _backward(grad):
        if x.requires_grad:
            x.backward(grad * np.where(x.data > 0, 1, alpha))
            
    if out.requires_grad:
        out._grad_fn = _backward
        out._prev = {x}
        
    return out


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation function."""
    sig = 1 / (1 + np.exp(-x.data))
    out = Tensor(sig, requires_grad=x.requires_grad)
    
    def _backward(grad):
        if x.requires_grad:
            x.backward(grad * sig * (1 - sig))
            
    if out.requires_grad:
        out._grad_fn = _backward
        out._prev = {x}
        
    return out


def tanh(x: Tensor) -> Tensor:
    """Tanh activation function."""
    t = np.tanh(x.data)
    out = Tensor(t, requires_grad=x.requires_grad)
    
    def _backward(grad):
        if x.requires_grad:
            x.backward(grad * (1 - t**2))
            
    if out.requires_grad:
        out._grad_fn = _backward
        out._prev = {x}
        
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation function."""
    exp_x = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
    softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    out = Tensor(softmax_x, requires_grad=x.requires_grad)
    
    def _backward(grad):
        if x.requires_grad:
            # Jacobian of softmax
            s = softmax_x
            # For each sample in the batch
            grad_input = np.zeros_like(x.data)
            
            # Handle different tensor shapes
            if len(x.shape) == 2:  # (batch_size, features)
                for i in range(x.shape[0]):
                    s_i = s[i].reshape(-1, 1)
                    jacobian = np.diagflat(s[i]) - s_i @ s_i.T
                    grad_input[i] = jacobian @ grad[i]
            else:
                # For 1D or higher dimensional tensors
                s_flat = s.reshape(-1, 1)
                jacobian = np.diagflat(s.flatten()) - s_flat @ s_flat.T
                grad_input = (jacobian @ grad.flatten()).reshape(x.shape)
                
            x.backward(grad_input)
            
    if out.requires_grad:
        out._grad_fn = _backward
        out._prev = {x}
        
    return out


# Layer classes
class Activation:
    """Base class for activation layers."""
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ReLU(Activation):
    """ReLU activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class LeakyReLU(Activation):
    """Leaky ReLU activation layer."""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        return leaky_relu(x, self.alpha)


class Sigmoid(Activation):
    """Sigmoid activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(Activation):
    """Tanh activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class Softmax(Activation):
    """Softmax activation layer."""
    
    def __init__(self, axis: int = -1):
        self.axis = axis
    
    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, self.axis)


# Helper function to get activation by name
def get_activation(activation: Union[str, Activation, None]) -> Optional[Activation]:
    """Get activation function by name or return the activation if already instantiated."""
    if activation is None:
        return None
    
    if isinstance(activation, Activation):
        return activation
    
    activation_map = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax,
        'linear': None  # Linear activation is just identity (no activation)
    }
    
    if activation.lower() in activation_map:
        activation_class = activation_map[activation.lower()]
        return activation_class() if activation_class is not None else None
    else:
        raise ValueError(f"Unknown activation function: {activation}")