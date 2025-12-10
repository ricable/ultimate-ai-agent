"""
Base Layer class for neural network components.
Pure NumPy implementation for CPU optimization.
"""

import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """Abstract base class for all neural network layers."""
    
    def __init__(self):
        self.trainable = True
        self.built = False
        self.params = {}
        self.grads = {}
        self.cache = {}
        
    @abstractmethod
    def forward(self, inputs, training=True):
        """
        Forward propagation through the layer.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def backward(self, grad_output):
        """
        Backward propagation through the layer.
        
        Args:
            grad_output: Gradient with respect to layer output
            
        Returns:
            Gradient with respect to layer input
        """
        pass
    
    def build(self, input_shape):
        """Build the layer (initialize parameters)."""
        self.built = True
    
    def get_params(self):
        """Get layer parameters."""
        return self.params
    
    def get_grads(self):
        """Get parameter gradients."""
        return self.grads
    
    def set_params(self, params):
        """Set layer parameters."""
        self.params = params