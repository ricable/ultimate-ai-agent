"""
Dropout layer implementation.
Pure NumPy implementation with forward and backward propagation.
"""

import numpy as np
import sys
sys.path.append('/workspaces/claude-test')

from neural_network.core.base import Layer


class Dropout(Layer):
    """Dropout regularization layer."""
    
    def __init__(self, rate, seed=None):
        """
        Initialize Dropout layer.
        
        Args:
            rate: Dropout rate (fraction of units to drop)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
            
        self.rate = rate
        self.seed = seed
        self.trainable = False  # No trainable parameters
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
    def build(self, input_shape):
        """Build the layer."""
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True
        
    def forward(self, inputs, training=True):
        """
        Forward propagation.
        
        During training: randomly drop units
        During inference: scale outputs
        """
        if not self.built:
            self.build(inputs.shape)
            
        if training and self.rate > 0:
            # Generate dropout mask
            self.cache['mask'] = np.random.binomial(1, 1 - self.rate, inputs.shape)
            
            # Apply dropout and scale
            # Scale by 1/(1-rate) to maintain expected value
            output = inputs * self.cache['mask'] / (1 - self.rate)
        else:
            # During inference, just pass through
            output = inputs
            self.cache['mask'] = None
            
        self.cache['training'] = training
        return output
        
    def backward(self, grad_output):
        """
        Backward propagation.
        
        Apply the same mask used in forward pass.
        """
        if self.cache['training'] and self.rate > 0:
            # Apply the same mask and scaling
            return grad_output * self.cache['mask'] / (1 - self.rate)
        else:
            # During inference, just pass through
            return grad_output
            
    def get_output_shape(self, input_shape):
        """Get output shape (same as input)."""
        return input_shape


class SpatialDropout2D(Layer):
    """Spatial Dropout for 2D data (drops entire feature maps)."""
    
    def __init__(self, rate, seed=None):
        """
        Initialize SpatialDropout2D layer.
        
        Args:
            rate: Dropout rate (fraction of feature maps to drop)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
            
        self.rate = rate
        self.seed = seed
        self.trainable = False  # No trainable parameters
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
    def build(self, input_shape):
        """Build the layer."""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input for SpatialDropout2D, got {input_shape}")
            
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True
        
    def forward(self, inputs, training=True):
        """
        Forward propagation.
        
        Drops entire feature maps instead of individual units.
        """
        if not self.built:
            self.build(inputs.shape)
            
        if training and self.rate > 0:
            # Shape: (batch_size, height, width, channels)
            batch_size, h, w, channels = inputs.shape
            
            # Generate mask for channels only
            mask_shape = (batch_size, 1, 1, channels)
            self.cache['mask'] = np.random.binomial(1, 1 - self.rate, mask_shape)
            
            # Broadcast mask across spatial dimensions
            output = inputs * self.cache['mask'] / (1 - self.rate)
        else:
            # During inference, just pass through
            output = inputs
            self.cache['mask'] = None
            
        self.cache['training'] = training
        return output
        
    def backward(self, grad_output):
        """
        Backward propagation.
        
        Apply the same spatial mask used in forward pass.
        """
        if self.cache['training'] and self.rate > 0:
            # Apply the same mask and scaling
            return grad_output * self.cache['mask'] / (1 - self.rate)
        else:
            # During inference, just pass through
            return grad_output
            
    def get_output_shape(self, input_shape):
        """Get output shape (same as input)."""
        return input_shape


class AlphaDropout(Layer):
    """Alpha Dropout - maintains mean and variance (for SELU networks)."""
    
    def __init__(self, rate, seed=None):
        """
        Initialize AlphaDropout layer.
        
        Maintains self-normalizing property for SELU activation.
        
        Args:
            rate: Dropout rate
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
            
        self.rate = rate
        self.seed = seed
        self.trainable = False
        
        # Alpha dropout parameters for SELU
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        
        # Calculate affine transformation parameters
        self.a = ((1 - rate) * (1 + rate * self.alpha ** 2)) ** -0.5
        self.b = -self.a * rate * self.alpha
        
        if seed is not None:
            np.random.seed(seed)
            
    def build(self, input_shape):
        """Build the layer."""
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True
        
    def forward(self, inputs, training=True):
        """Forward propagation with alpha dropout."""
        if not self.built:
            self.build(inputs.shape)
            
        if training and self.rate > 0:
            # Generate dropout mask
            self.cache['mask'] = np.random.binomial(1, 1 - self.rate, inputs.shape)
            
            # Apply alpha dropout transformation
            output = inputs * self.cache['mask']
            output = self.a * (output + self.alpha * (1 - self.cache['mask'])) + self.b
        else:
            output = inputs
            self.cache['mask'] = None
            
        self.cache['training'] = training
        return output
        
    def backward(self, grad_output):
        """Backward propagation."""
        if self.cache['training'] and self.rate > 0:
            # Gradient only flows through kept units
            return self.a * grad_output * self.cache['mask']
        else:
            return grad_output
            
    def get_output_shape(self, input_shape):
        """Get output shape (same as input)."""
        return input_shape