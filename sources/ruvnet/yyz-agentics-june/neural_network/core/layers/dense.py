"""
Dense (Fully Connected) layer implementation.
Pure NumPy implementation with forward and backward propagation.
"""

import numpy as np
import sys
sys.path.append('/workspaces/claude-test')

from neural_network.core.base import Layer
from neural_network.core.initializers.initializers import get_initializer, XavierNormal, Zeros
from neural_network.core.activations.activations import get_activation


class Dense(Layer):
    """Dense/Fully Connected layer."""
    
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='xavier_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None):
        """
        Initialize Dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function (None or activation object)
            use_bias: Whether to use bias
            kernel_initializer: Weight initializer name or object
            bias_initializer: Bias initializer name or object
            kernel_regularizer: Weight regularizer (L1/L2)
            bias_regularizer: Bias regularizer (L1/L2)
        """
        super().__init__()
        self.units = units
        
        # Get activation function
        if isinstance(activation, str):
            self.activation = get_activation(activation)
        else:
            self.activation = activation
            
        self.use_bias = use_bias
        
        # Get initializers
        if isinstance(kernel_initializer, str):
            self.kernel_initializer = get_initializer(kernel_initializer)
        else:
            self.kernel_initializer = kernel_initializer
            
        if isinstance(bias_initializer, str):
            self.bias_initializer = get_initializer(bias_initializer)
        else:
            self.bias_initializer = bias_initializer
            
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        
        self.input_shape = None
        
    def build(self, input_shape):
        """Build the layer (initialize parameters)."""
        if len(input_shape) < 2:
            raise ValueError(f"Input shape must be at least 2D, got {input_shape}")
            
        self.input_shape = input_shape
        input_dim = input_shape[-1]
        
        # Initialize weights
        self.params['W'] = self.kernel_initializer((input_dim, self.units))
        self.grads['W'] = np.zeros_like(self.params['W'])
        
        if self.use_bias:
            self.params['b'] = self.bias_initializer((self.units,))
            self.grads['b'] = np.zeros_like(self.params['b'])
            
        self.built = True
        
    def forward(self, inputs, training=True):
        """
        Forward propagation.
        
        Args:
            inputs: Input tensor of shape (batch_size, ..., input_dim)
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, ..., units)
        """
        if not self.built:
            self.build(inputs.shape)
            
        # Store inputs for backward pass
        self.cache['inputs'] = inputs
        
        # Flatten all dimensions except the last one if needed
        input_shape = inputs.shape
        if len(input_shape) > 2:
            inputs_flat = inputs.reshape(-1, input_shape[-1])
        else:
            inputs_flat = inputs
            
        # Compute output: y = xW + b
        output = np.dot(inputs_flat, self.params['W'])
        
        if self.use_bias:
            output += self.params['b']
            
        # Reshape back to original shape (except last dimension)
        if len(input_shape) > 2:
            output_shape = input_shape[:-1] + (self.units,)
            output = output.reshape(output_shape)
            
        # Store pre-activation output
        self.cache['pre_activation'] = output
        
        # Apply activation if specified
        if self.activation is not None:
            output = self.activation.forward(output)
            
        self.cache['output'] = output
        return output
        
    def backward(self, grad_output):
        """
        Backward propagation.
        
        Args:
            grad_output: Gradient with respect to layer output
            
        Returns:
            Gradient with respect to layer input
        """
        # Apply activation backward if needed
        if self.activation is not None:
            grad_output = self.activation.backward(
                self.cache['pre_activation'], grad_output
            )
            
        inputs = self.cache['inputs']
        input_shape = inputs.shape
        
        # Flatten inputs and gradients if needed
        if len(input_shape) > 2:
            inputs_flat = inputs.reshape(-1, input_shape[-1])
            grad_output_flat = grad_output.reshape(-1, self.units)
        else:
            inputs_flat = inputs
            grad_output_flat = grad_output
            
        # Compute gradients
        # dL/dW = x^T @ dL/dy
        self.grads['W'] = np.dot(inputs_flat.T, grad_output_flat)
        
        if self.use_bias:
            # dL/db = sum(dL/dy, axis=0)
            self.grads['b'] = np.sum(grad_output_flat, axis=0)
            
        # Apply regularization to weight gradients
        if self.kernel_regularizer is not None:
            self.grads['W'] += self.kernel_regularizer.gradient(self.params['W'])
            
        if self.use_bias and self.bias_regularizer is not None:
            self.grads['b'] += self.bias_regularizer.gradient(self.params['b'])
            
        # Compute gradient with respect to input
        # dL/dx = dL/dy @ W^T
        grad_input = np.dot(grad_output_flat, self.params['W'].T)
        
        # Reshape back to original input shape
        if len(input_shape) > 2:
            grad_input = grad_input.reshape(input_shape)
            
        return grad_input
        
    def get_output_shape(self, input_shape):
        """Get output shape for given input shape."""
        return input_shape[:-1] + (self.units,)
        
    def get_config(self):
        """Get layer configuration."""
        config = {
            'units': self.units,
            'activation': self.activation.__class__.__name__ if self.activation else None,
            'use_bias': self.use_bias,
            'input_shape': self.input_shape
        }
        return config