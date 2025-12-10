"""
Dense (Fully Connected) Layer
"""
import numpy as np
from typing import Optional, Union
from ..tensor import Tensor
from ..activations import get_activation, Activation


class Dense:
    """
    Dense (fully connected) layer.
    
    Parameters:
        units: Number of output units
        activation: Activation function to use
        use_bias: Whether to use bias
        weight_initializer: Method for weight initialization
    """
    
    def __init__(self, 
                 units: int,
                 activation: Optional[Union[str, Activation]] = None,
                 use_bias: bool = True,
                 weight_initializer: str = 'glorot_uniform',
                 name: Optional[str] = None):
        self.units = units
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.name = name or f"dense_{id(self)}"
        
        # Parameters will be initialized when we know input shape
        self.weights = None
        self.bias = None
        self.input_shape = None
        
    def build(self, input_shape: tuple):
        """Initialize weights based on input shape."""
        self.input_shape = input_shape
        
        # Get the last dimension as input features
        if len(input_shape) > 1:
            input_features = input_shape[-1]
        else:
            input_features = input_shape[0]
        
        # Initialize weights
        if self.weight_initializer == 'glorot_uniform':
            limit = np.sqrt(6 / (input_features + self.units))
            self.weights = Tensor.uniform((input_features, self.units), -limit, limit, requires_grad=True)
        elif self.weight_initializer == 'he_normal':
            std = np.sqrt(2 / input_features)
            self.weights = Tensor(np.random.randn(input_features, self.units) * std, requires_grad=True)
        else:
            self.weights = Tensor.randn((input_features, self.units), requires_grad=True)
        
        # Initialize bias
        if self.use_bias:
            self.bias = Tensor.zeros((self.units,), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the layer."""
        # Build layer if not already built
        if self.weights is None:
            self.build(x.shape)
        
        # Linear transformation
        output = x @ self.weights
        
        if self.use_bias:
            output = output + self.bias
        
        # Apply activation if specified
        if self.activation:
            output = self.activation(output)
        
        return output
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def get_parameters(self) -> list:
        """Get all trainable parameters."""
        params = [self.weights]
        if self.use_bias:
            params.append(self.bias)
        return params
    
    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Calculate output shape given input shape."""
        return input_shape[:-1] + (self.units,)
    
    def __repr__(self):
        return f"Dense(units={self.units}, activation={self.activation.__class__.__name__ if self.activation else 'None'})"