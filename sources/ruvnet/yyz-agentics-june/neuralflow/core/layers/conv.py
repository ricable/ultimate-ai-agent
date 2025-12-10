"""
Convolutional Layers
"""
import numpy as np
from typing import Optional, Union, Tuple
from ..tensor import Tensor
from ..activations import get_activation, Activation


class Conv2D:
    """
    2D Convolutional Layer.
    
    Parameters:
        filters: Number of filters (output channels)
        kernel_size: Size of the convolutional kernel
        stride: Stride of the convolution
        padding: Padding type ('valid' or 'same')
        activation: Activation function
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: str = 'valid',
                 activation: Optional[Union[str, Activation]] = None,
                 use_bias: bool = True,
                 weight_initializer: str = 'glorot_uniform',
                 name: Optional[str] = None):
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.name = name or f"conv2d_{id(self)}"
        
        self.weights = None
        self.bias = None
        self.input_shape = None
        
    def build(self, input_shape: tuple):
        """Initialize weights based on input shape."""
        # input_shape: (batch, height, width, channels)
        self.input_shape = input_shape
        in_channels = input_shape[-1]
        
        # Initialize weights: (kernel_h, kernel_w, in_channels, out_channels)
        kernel_shape = (*self.kernel_size, in_channels, self.filters)
        
        if self.weight_initializer == 'glorot_uniform':
            fan_in = self.kernel_size[0] * self.kernel_size[1] * in_channels
            fan_out = self.kernel_size[0] * self.kernel_size[1] * self.filters
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights = Tensor.uniform(kernel_shape, -limit, limit, requires_grad=True)
        else:
            self.weights = Tensor.randn(kernel_shape, requires_grad=True)
        
        if self.use_bias:
            self.bias = Tensor.zeros((self.filters,), requires_grad=True)
    
    def _calculate_padding(self, input_shape):
        """Calculate padding for 'same' mode."""
        if self.padding == 'valid':
            return (0, 0)
        elif self.padding == 'same':
            pad_h = ((input_shape[1] - 1) * self.stride[0] + self.kernel_size[0] - input_shape[1]) // 2
            pad_w = ((input_shape[2] - 1) * self.stride[1] + self.kernel_size[1] - input_shape[2]) // 2
            return (pad_h, pad_w)
        else:
            raise ValueError(f"Unknown padding mode: {self.padding}")
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the convolutional layer."""
        if self.weights is None:
            self.build(x.shape)
        
        batch_size, in_height, in_width, in_channels = x.shape
        kernel_h, kernel_w = self.kernel_size
        
        # Calculate padding
        pad_h, pad_w = self._calculate_padding(x.shape)
        
        # Pad input if necessary
        if pad_h > 0 or pad_w > 0:
            padded_data = np.pad(x.data, 
                                ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 
                                mode='constant')
            padded_x = Tensor(padded_data, requires_grad=x.requires_grad)
        else:
            padded_x = x
        
        # Calculate output dimensions
        out_height = (padded_x.shape[1] - kernel_h) // self.stride[0] + 1
        out_width = (padded_x.shape[2] - kernel_w) // self.stride[1] + 1
        
        # Perform convolution using im2col approach for efficiency
        # Convert input to column matrix
        col = self._im2col(padded_x.data, kernel_h, kernel_w, self.stride)
        
        # Reshape weights for matrix multiplication
        weights_reshaped = self.weights.data.reshape(-1, self.filters)
        
        # Perform convolution as matrix multiplication
        output_data = col @ weights_reshaped
        
        # Reshape output
        output_data = output_data.reshape(batch_size, out_height, out_width, self.filters)
        output = Tensor(output_data, requires_grad=x.requires_grad or self.weights.requires_grad)
        
        # Add bias
        if self.use_bias:
            output = output + self.bias
        
        # Apply activation
        if self.activation:
            output = self.activation(output)
        
        # Store variables for backward pass
        self._cache = (x, padded_x, col, pad_h, pad_w)
        
        def _backward(grad):
            x, padded_x, col, pad_h, pad_w = self._cache
            
            # Gradient w.r.t weights
            if self.weights.requires_grad:
                grad_reshaped = grad.reshape(-1, self.filters)
                grad_weights = col.T @ grad_reshaped
                self.weights.backward(grad_weights.reshape(self.weights.shape))
            
            # Gradient w.r.t bias
            if self.use_bias and self.bias.requires_grad:
                self.bias.backward(np.sum(grad, axis=(0, 1, 2)))
            
            # Gradient w.r.t input
            if x.requires_grad:
                grad_col = grad_reshaped @ weights_reshaped.T
                grad_input = self._col2im(grad_col, padded_x.shape, kernel_h, kernel_w, self.stride)
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    grad_input = grad_input[:, pad_h:-pad_h, pad_w:-pad_w, :]
                
                x.backward(grad_input)
        
        if output.requires_grad:
            output._grad_fn = _backward
            output._prev = {x, self.weights} | ({self.bias} if self.use_bias else set())
        
        return output
    
    def _im2col(self, input_data, kernel_h, kernel_w, stride):
        """Convert input to column matrix for efficient convolution."""
        batch_size, height, width, channels = input_data.shape
        out_height = (height - kernel_h) // stride[0] + 1
        out_width = (width - kernel_w) // stride[1] + 1
        
        col = np.zeros((batch_size * out_height * out_width, kernel_h * kernel_w * channels))
        
        idx = 0
        for b in range(batch_size):
            for i in range(0, height - kernel_h + 1, stride[0]):
                for j in range(0, width - kernel_w + 1, stride[1]):
                    patch = input_data[b, i:i+kernel_h, j:j+kernel_w, :].flatten()
                    col[idx] = patch
                    idx += 1
        
        return col
    
    def _col2im(self, col, input_shape, kernel_h, kernel_w, stride):
        """Convert column matrix back to image format."""
        batch_size, height, width, channels = input_shape
        out_height = (height - kernel_h) // stride[0] + 1
        out_width = (width - kernel_w) // stride[1] + 1
        
        output = np.zeros(input_shape)
        
        idx = 0
        for b in range(batch_size):
            for i in range(0, height - kernel_h + 1, stride[0]):
                for j in range(0, width - kernel_w + 1, stride[1]):
                    patch = col[idx].reshape(kernel_h, kernel_w, channels)
                    output[b, i:i+kernel_h, j:j+kernel_w, :] += patch
                    idx += 1
        
        return output
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def get_parameters(self) -> list:
        """Get all trainable parameters."""
        params = [self.weights]
        if self.use_bias:
            params.append(self.bias)
        return params


class MaxPool2D:
    """
    2D Max Pooling Layer.
    
    Parameters:
        pool_size: Size of the pooling window
        stride: Stride of the pooling operation
    """
    
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]] = 2,
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 name: Optional[str] = None):
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = stride or self.pool_size
        self.stride = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        self.name = name or f"maxpool2d_{id(self)}"
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the max pooling layer."""
        batch_size, height, width, channels = x.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride
        
        # Calculate output dimensions
        out_height = (height - pool_h) // stride_h + 1
        out_width = (width - pool_w) // stride_w + 1
        
        # Perform max pooling
        output_data = np.zeros((batch_size, out_height, out_width, channels))
        mask = np.zeros_like(output_data, dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        pool_region = x.data[b, h_start:h_start+pool_h, w_start:w_start+pool_w, c]
                        output_data[b, i, j, c] = np.max(pool_region)
                        mask[b, i, j, c] = np.argmax(pool_region.flatten())
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        def _backward(grad):
            if x.requires_grad:
                grad_input = np.zeros_like(x.data)
                
                for b in range(batch_size):
                    for c in range(channels):
                        for i in range(out_height):
                            for j in range(out_width):
                                h_start = i * stride_h
                                w_start = j * stride_w
                                max_idx = mask[b, i, j, c]
                                h_idx = max_idx // pool_w
                                w_idx = max_idx % pool_w
                                grad_input[b, h_start+h_idx, w_start+w_idx, c] += grad[b, i, j, c]
                
                x.backward(grad_input)
        
        if output.requires_grad:
            output._grad_fn = _backward
            output._prev = {x}
        
        return output
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def get_parameters(self) -> list:
        """Max pooling has no trainable parameters."""
        return []