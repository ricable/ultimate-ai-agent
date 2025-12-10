"""
Normalization Layers
"""
import numpy as np
from typing import Optional, Tuple
from ..tensor import Tensor


class BatchNormalization:
    """
    Batch Normalization layer.
    
    Parameters:
        momentum: Momentum for moving average
        epsilon: Small constant for numerical stability
        center: If True, add learnable bias parameter
        scale: If True, add learnable scale parameter
    """
    
    def __init__(self,
                 momentum: float = 0.99,
                 epsilon: float = 1e-5,
                 center: bool = True,
                 scale: bool = True,
                 name: Optional[str] = None):
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.name = name or f"batchnorm_{id(self)}"
        self.training = True
        
        # Parameters
        self.gamma = None  # Scale
        self.beta = None   # Shift
        self.running_mean = None
        self.running_var = None
        
        self.built = False
        
    def build(self, input_shape: tuple):
        """Initialize parameters based on input shape."""
        # Get feature dimension (last dimension)
        feature_dim = input_shape[-1]
        
        # Initialize parameters
        if self.scale:
            self.gamma = Tensor.ones((feature_dim,), requires_grad=True)
        if self.center:
            self.beta = Tensor.zeros((feature_dim,), requires_grad=True)
        
        # Initialize running statistics
        self.running_mean = np.zeros((feature_dim,))
        self.running_var = np.ones((feature_dim,))
        
        self.built = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through batch normalization."""
        if not self.built:
            self.build(x.shape)
        
        if self.training:
            # Calculate batch statistics
            axes = tuple(range(len(x.shape) - 1))  # All axes except last
            mean = np.mean(x.data, axis=axes, keepdims=True)
            var = np.var(x.data, axis=axes, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
        else:
            # Use running statistics
            shape = [1] * len(x.shape)
            shape[-1] = -1
            mean = self.running_mean.reshape(shape)
            var = self.running_var.reshape(shape)
        
        # Normalize
        x_normalized = (x.data - mean) / np.sqrt(var + self.epsilon)
        
        # Scale and shift
        output_data = x_normalized
        if self.scale:
            output_data = output_data * self.gamma.data
        if self.center:
            output_data = output_data + self.beta.data
        
        output = Tensor(output_data, requires_grad=x.requires_grad or 
                       (self.scale and self.gamma.requires_grad) or 
                       (self.center and self.beta.requires_grad))
        
        # Cache for backward pass
        self._cache = (x, x_normalized, mean, var)
        
        def _backward(grad):
            x, x_normalized, mean, var = self._cache
            batch_size = x.shape[0]
            
            # Gradients w.r.t scale and shift
            if self.scale and self.gamma.requires_grad:
                axes = tuple(range(len(grad.shape) - 1))
                self.gamma.backward(np.sum(grad * x_normalized, axis=axes))
            
            if self.center and self.beta.requires_grad:
                axes = tuple(range(len(grad.shape) - 1))
                self.beta.backward(np.sum(grad, axis=axes))
            
            # Gradient w.r.t input
            if x.requires_grad:
                # Scale gradient by gamma
                grad_norm = grad
                if self.scale:
                    grad_norm = grad_norm * self.gamma.data
                
                # Gradient through normalization
                std = np.sqrt(var + self.epsilon)
                grad_var = np.sum(grad_norm * (x.data - mean) * -0.5 * (var + self.epsilon)**(-1.5), 
                                 axis=0, keepdims=True)
                grad_mean = np.sum(grad_norm * -1 / std, axis=0, keepdims=True) + \
                           grad_var * np.sum(-2 * (x.data - mean), axis=0, keepdims=True) / batch_size
                
                grad_input = grad_norm / std + \
                            grad_var * 2 * (x.data - mean) / batch_size + \
                            grad_mean / batch_size
                
                x.backward(grad_input)
        
        if output.requires_grad:
            output._grad_fn = _backward
            output._prev = {x} | ({self.gamma} if self.scale else set()) | ({self.beta} if self.center else set())
        
        return output
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def train(self):
        """Set layer to training mode."""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False
    
    def get_parameters(self) -> list:
        """Get trainable parameters."""
        params = []
        if self.scale:
            params.append(self.gamma)
        if self.center:
            params.append(self.beta)
        return params


class LayerNormalization:
    """
    Layer Normalization.
    
    Parameters:
        epsilon: Small constant for numerical stability
        center: If True, add learnable bias parameter
        scale: If True, add learnable scale parameter
    """
    
    def __init__(self,
                 epsilon: float = 1e-5,
                 center: bool = True,
                 scale: bool = True,
                 name: Optional[str] = None):
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.name = name or f"layernorm_{id(self)}"
        
        # Parameters
        self.gamma = None  # Scale
        self.beta = None   # Shift
        
        self.built = False
    
    def build(self, input_shape: tuple):
        """Initialize parameters based on input shape."""
        # Get normalized shape (typically last dimension or last few dimensions)
        normalized_shape = input_shape[-1:]
        
        # Initialize parameters
        if self.scale:
            self.gamma = Tensor.ones(normalized_shape, requires_grad=True)
        if self.center:
            self.beta = Tensor.zeros(normalized_shape, requires_grad=True)
        
        self.built = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through layer normalization."""
        if not self.built:
            self.build(x.shape)
        
        # Calculate mean and variance along last dimension(s)
        axes = tuple(range(-len(self.gamma.shape), 0))
        mean = np.mean(x.data, axis=axes, keepdims=True)
        var = np.var(x.data, axis=axes, keepdims=True)
        
        # Normalize
        x_normalized = (x.data - mean) / np.sqrt(var + self.epsilon)
        
        # Scale and shift
        output_data = x_normalized
        if self.scale:
            output_data = output_data * self.gamma.data
        if self.center:
            output_data = output_data + self.beta.data
        
        output = Tensor(output_data, requires_grad=x.requires_grad or 
                       (self.scale and self.gamma.requires_grad) or 
                       (self.center and self.beta.requires_grad))
        
        # Cache for backward pass
        self._cache = (x, x_normalized, mean, var, axes)
        
        def _backward(grad):
            x, x_normalized, mean, var, axes = self._cache
            
            # Gradients w.r.t scale and shift
            if self.scale and self.gamma.requires_grad:
                sum_axes = tuple(range(len(grad.shape) - len(self.gamma.shape)))
                self.gamma.backward(np.sum(grad * x_normalized, axis=sum_axes))
            
            if self.center and self.beta.requires_grad:
                sum_axes = tuple(range(len(grad.shape) - len(self.beta.shape)))
                self.beta.backward(np.sum(grad, axis=sum_axes))
            
            # Gradient w.r.t input
            if x.requires_grad:
                # Scale gradient by gamma
                grad_norm = grad
                if self.scale:
                    grad_norm = grad_norm * self.gamma.data
                
                # Gradient through normalization
                std = np.sqrt(var + self.epsilon)
                N = np.prod([x.shape[ax] for ax in axes])
                
                grad_var = np.sum(grad_norm * (x.data - mean) * -0.5 * (var + self.epsilon)**(-1.5), 
                                 axis=axes, keepdims=True)
                grad_mean = np.sum(grad_norm * -1 / std, axis=axes, keepdims=True)
                
                grad_input = grad_norm / std + \
                            grad_var * 2 * (x.data - mean) / N + \
                            grad_mean / N
                
                x.backward(grad_input)
        
        if output.requires_grad:
            output._grad_fn = _backward
            output._prev = {x} | ({self.gamma} if self.scale else set()) | ({self.beta} if self.center else set())
        
        return output
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def get_parameters(self) -> list:
        """Get trainable parameters."""
        params = []
        if self.scale:
            params.append(self.gamma)
        if self.center:
            params.append(self.beta)
        return params