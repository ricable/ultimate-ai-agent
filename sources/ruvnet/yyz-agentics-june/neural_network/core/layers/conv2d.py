"""
2D Convolutional layer implementation.
Pure NumPy implementation with forward and backward propagation.
"""

import numpy as np
import sys
sys.path.append('/workspaces/claude-test')

from neural_network.core.base import Layer
from neural_network.core.initializers.initializers import get_initializer, XavierNormal, Zeros
from neural_network.core.activations.activations import get_activation


class Conv2D(Layer):
    """2D Convolutional layer."""
    
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
                 activation=None, use_bias=True,
                 kernel_initializer='xavier_normal',
                 bias_initializer='zeros'):
        """
        Initialize Conv2D layer.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of convolution kernel (int or tuple)
            strides: Stride of convolution (int or tuple)
            padding: 'valid' or 'same'
            activation: Activation function
            use_bias: Whether to use bias
            kernel_initializer: Weight initializer
            bias_initializer: Bias initializer
        """
        super().__init__()
        self.filters = filters
        
        # Handle kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle strides
        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = strides
            
        self.padding = padding
        
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
            
    def build(self, input_shape):
        """Build the layer."""
        # Expected input shape: (batch_size, height, width, channels)
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input, got shape {input_shape}")
            
        self.input_shape = input_shape
        _, h, w, in_channels = input_shape
        
        # Initialize kernel: (height, width, in_channels, out_channels)
        kernel_shape = (*self.kernel_size, in_channels, self.filters)
        self.params['W'] = self.kernel_initializer(kernel_shape)
        self.grads['W'] = np.zeros_like(self.params['W'])
        
        if self.use_bias:
            self.params['b'] = self.bias_initializer((self.filters,))
            self.grads['b'] = np.zeros_like(self.params['b'])
            
        # Calculate output shape
        if self.padding == 'same':
            out_h = int(np.ceil(h / self.strides[0]))
            out_w = int(np.ceil(w / self.strides[1]))
        else:  # 'valid'
            out_h = int((h - self.kernel_size[0]) / self.strides[0]) + 1
            out_w = int((w - self.kernel_size[1]) / self.strides[1]) + 1
            
        self.output_shape = (input_shape[0], out_h, out_w, self.filters)
        self.built = True
        
    def _add_padding(self, x):
        """Add padding to input."""
        if self.padding == 'valid':
            return x
            
        # Calculate padding for 'same'
        h, w = x.shape[1:3]
        
        if h % self.strides[0] == 0:
            pad_h = max(self.kernel_size[0] - self.strides[0], 0)
        else:
            pad_h = max(self.kernel_size[0] - (h % self.strides[0]), 0)
            
        if w % self.strides[1] == 0:
            pad_w = max(self.kernel_size[1] - self.strides[1], 0)
        else:
            pad_w = max(self.kernel_size[1] - (w % self.strides[1]), 0)
            
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                     mode='constant', constant_values=0)
        
    def _im2col(self, x, kernel_h, kernel_w, stride_h, stride_w):
        """Transform input into column matrix for efficient convolution."""
        batch_size, h, w, c = x.shape
        
        # Calculate output dimensions
        out_h = (h - kernel_h) // stride_h + 1
        out_w = (w - kernel_w) // stride_w + 1
        
        # Create column matrix
        col = np.zeros((batch_size, out_h, out_w, kernel_h, kernel_w, c))
        
        for j in range(kernel_h):
            j_lim = j + stride_h * out_h
            for i in range(kernel_w):
                i_lim = i + stride_w * out_w
                col[:, :, :, j, i, :] = x[:, j:j_lim:stride_h, i:i_lim:stride_w, :]
                
        col = col.reshape(batch_size * out_h * out_w, -1)
        return col
        
    def _col2im(self, col, x_shape, kernel_h, kernel_w, stride_h, stride_w):
        """Transform column matrix back to image."""
        batch_size, h, w, c = x_shape
        out_h = (h - kernel_h) // stride_h + 1
        out_w = (w - kernel_w) // stride_w + 1
        
        col = col.reshape(batch_size, out_h, out_w, kernel_h, kernel_w, c)
        x = np.zeros(x_shape)
        
        for j in range(kernel_h):
            j_lim = j + stride_h * out_h
            for i in range(kernel_w):
                i_lim = i + stride_w * out_w
                x[:, j:j_lim:stride_h, i:i_lim:stride_w, :] += col[:, :, :, j, i, :]
                
        return x
        
    def forward(self, inputs, training=True):
        """Forward propagation."""
        if not self.built:
            self.build(inputs.shape)
            
        self.cache['inputs'] = inputs
        
        # Add padding if needed
        x_padded = self._add_padding(inputs)
        self.cache['x_padded'] = x_padded
        
        # Transform to column matrix
        x_col = self._im2col(x_padded, *self.kernel_size, *self.strides)
        
        # Reshape kernel for matrix multiplication
        W_col = self.params['W'].reshape(-1, self.filters)
        
        # Perform convolution as matrix multiplication
        out = np.dot(x_col, W_col)
        
        if self.use_bias:
            out += self.params['b']
            
        # Reshape output
        batch_size = inputs.shape[0]
        out_h = (x_padded.shape[1] - self.kernel_size[0]) // self.strides[0] + 1
        out_w = (x_padded.shape[2] - self.kernel_size[1]) // self.strides[1] + 1
        out = out.reshape(batch_size, out_h, out_w, self.filters)
        
        self.cache['pre_activation'] = out
        
        # Apply activation
        if self.activation is not None:
            out = self.activation.forward(out)
            
        self.cache['output'] = out
        return out
        
    def backward(self, grad_output):
        """Backward propagation."""
        # Apply activation backward
        if self.activation is not None:
            grad_output = self.activation.backward(
                self.cache['pre_activation'], grad_output
            )
            
        x_padded = self.cache['x_padded']
        batch_size = grad_output.shape[0]
        
        # Reshape grad_output for computation
        grad_output_reshaped = grad_output.reshape(-1, self.filters)
        
        # Compute weight gradient
        x_col = self._im2col(x_padded, *self.kernel_size, *self.strides)
        self.grads['W'] = np.dot(x_col.T, grad_output_reshaped).reshape(self.params['W'].shape)
        
        # Compute bias gradient
        if self.use_bias:
            self.grads['b'] = np.sum(grad_output_reshaped, axis=0)
            
        # Compute input gradient
        W_col = self.params['W'].reshape(-1, self.filters)
        grad_x_col = np.dot(grad_output_reshaped, W_col.T)
        
        # Transform back to image
        grad_x_padded = self._col2im(grad_x_col, x_padded.shape, 
                                    *self.kernel_size, *self.strides)
        
        # Remove padding
        if self.padding == 'same':
            h, w = self.cache['inputs'].shape[1:3]
            pad_h = x_padded.shape[1] - h
            pad_w = x_padded.shape[2] - w
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if pad_h > 0 or pad_w > 0:
                grad_x = grad_x_padded[:, pad_top:x_padded.shape[1]-pad_bottom,
                                     pad_left:x_padded.shape[2]-pad_right, :]
            else:
                grad_x = grad_x_padded
        else:
            grad_x = grad_x_padded
            
        return grad_x
        
    def get_output_shape(self, input_shape):
        """Get output shape for given input shape."""
        if not self.built:
            self.build(input_shape)
        return self.output_shape