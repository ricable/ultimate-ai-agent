"""
Batch Normalization layer implementation.
Pure NumPy implementation with forward and backward propagation.
"""

import numpy as np
import sys
sys.path.append('/workspaces/claude-test')

from neural_network.core.base import Layer
from neural_network.core.initializers.initializers import Ones, Zeros


class BatchNormalization(Layer):
    """Batch Normalization layer."""
    
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3,
                 center=True, scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones'):
        """
        Initialize BatchNormalization layer.
        
        Args:
            axis: Axis along which to normalize (typically features axis)
            momentum: Momentum for moving average
            epsilon: Small constant for numerical stability
            center: Whether to use beta parameter
            scale: Whether to use gamma parameter
            beta_initializer: Initializer for beta (shift parameter)
            gamma_initializer: Initializer for gamma (scale parameter)
            moving_mean_initializer: Initializer for moving mean
            moving_variance_initializer: Initializer for moving variance
        """
        super().__init__()
        
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
        # Get initializers
        self.beta_initializer = Zeros() if beta_initializer == 'zeros' else beta_initializer
        self.gamma_initializer = Ones() if gamma_initializer == 'ones' else gamma_initializer
        self.moving_mean_initializer = Zeros() if moving_mean_initializer == 'zeros' else moving_mean_initializer
        self.moving_variance_initializer = Ones() if moving_variance_initializer == 'ones' else moving_variance_initializer
        
    def build(self, input_shape):
        """Build the layer."""
        # Get shape of features to normalize
        if self.axis < 0:
            axis = len(input_shape) + self.axis
        else:
            axis = self.axis
            
        self.axis = axis
        dim = input_shape[axis]
        
        # Shape for parameters
        param_shape = [1] * len(input_shape)
        param_shape[axis] = dim
        self.param_shape = tuple(param_shape)
        
        # Initialize trainable parameters
        if self.scale:
            self.params['gamma'] = self.gamma_initializer((dim,))
            self.grads['gamma'] = np.zeros((dim,))
        else:
            self.params['gamma'] = None
            
        if self.center:
            self.params['beta'] = self.beta_initializer((dim,))
            self.grads['beta'] = np.zeros((dim,))
        else:
            self.params['beta'] = None
            
        # Initialize moving statistics (not trainable)
        self.moving_mean = self.moving_mean_initializer((dim,))
        self.moving_variance = self.moving_variance_initializer((dim,))
        
        self.built = True
        
    def forward(self, inputs, training=True):
        """
        Forward propagation.
        
        During training: use batch statistics
        During inference: use moving statistics
        """
        if not self.built:
            self.build(inputs.shape)
            
        # Get axes to reduce over (all except the normalized axis)
        reduction_axes = list(range(len(inputs.shape)))
        del reduction_axes[self.axis]
        reduction_axes = tuple(reduction_axes)
        
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(inputs, axis=reduction_axes, keepdims=True)
            batch_var = np.var(inputs, axis=reduction_axes, keepdims=True)
            
            # Update moving statistics
            self.moving_mean = self.momentum * self.moving_mean + \
                              (1 - self.momentum) * batch_mean.squeeze()
            self.moving_variance = self.momentum * self.moving_variance + \
                                 (1 - self.momentum) * batch_var.squeeze()
            
            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Use moving statistics
            mean = self.moving_mean.reshape(self.param_shape)
            var = self.moving_variance.reshape(self.param_shape)
            
        # Normalize
        self.cache['mean'] = mean
        self.cache['var'] = var
        self.cache['inputs'] = inputs
        
        # Compute normalized input
        std = np.sqrt(var + self.epsilon)
        self.cache['std'] = std
        x_norm = (inputs - mean) / std
        self.cache['x_norm'] = x_norm
        
        # Scale and shift
        output = x_norm
        if self.scale:
            gamma = self.params['gamma'].reshape(self.param_shape)
            output = output * gamma
        if self.center:
            beta = self.params['beta'].reshape(self.param_shape)
            output = output + beta
            
        self.cache['training'] = training
        return output
        
    def backward(self, grad_output):
        """
        Backward propagation.
        
        Compute gradients with respect to inputs and parameters.
        """
        if not self.cache['training']:
            # During inference, simplified backward pass
            if self.scale:
                gamma = self.params['gamma'].reshape(self.param_shape)
                grad_output = grad_output * gamma
            return grad_output / self.cache['std']
            
        # Get cached values
        x_norm = self.cache['x_norm']
        std = self.cache['std']
        mean = self.cache['mean']
        var = self.cache['var']
        inputs = self.cache['inputs']
        
        # Get batch size
        reduction_axes = list(range(len(inputs.shape)))
        del reduction_axes[self.axis]
        m = np.prod([inputs.shape[i] for i in reduction_axes])
        
        # Gradients for scale and shift
        if self.center:
            self.grads['beta'] = np.sum(grad_output, axis=tuple(reduction_axes))
            
        if self.scale:
            self.grads['gamma'] = np.sum(grad_output * x_norm, axis=tuple(reduction_axes))
            gamma = self.params['gamma'].reshape(self.param_shape)
            grad_output = grad_output * gamma
            
        # Gradient with respect to normalized input
        grad_x_norm = grad_output
        
        # Gradient with respect to variance
        grad_var = np.sum(grad_x_norm * (inputs - mean) * -0.5 * (var + self.epsilon)**(-1.5),
                         axis=tuple(reduction_axes), keepdims=True)
        
        # Gradient with respect to mean
        grad_mean = np.sum(grad_x_norm * -1 / std, axis=tuple(reduction_axes), keepdims=True)
        grad_mean += grad_var * np.sum(-2 * (inputs - mean), axis=tuple(reduction_axes), keepdims=True) / m
        
        # Gradient with respect to input
        grad_input = grad_x_norm / std
        grad_input += grad_var * 2 * (inputs - mean) / m
        grad_input += grad_mean / m
        
        return grad_input
        
    def get_output_shape(self, input_shape):
        """Get output shape (same as input)."""
        return input_shape
        
    def get_config(self):
        """Get layer configuration."""
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        return config