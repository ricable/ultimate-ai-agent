"""
Pooling layers implementation.
Pure NumPy implementation with forward and backward propagation.
"""

import numpy as np
import sys
sys.path.append('/workspaces/claude-test')

from neural_network.core.base import Layer


class MaxPool2D(Layer):
    """2D Max Pooling layer."""
    
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        """
        Initialize MaxPool2D layer.
        
        Args:
            pool_size: Size of pooling window (int or tuple)
            strides: Stride of pooling (defaults to pool_size)
            padding: 'valid' or 'same'
        """
        super().__init__()
        
        # Handle pool size
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        # Handle strides (default to pool_size)
        if strides is None:
            self.strides = self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = strides
            
        self.padding = padding
        self.trainable = False  # No trainable parameters
        
    def build(self, input_shape):
        """Build the layer."""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input, got shape {input_shape}")
            
        self.input_shape = input_shape
        _, h, w, c = input_shape
        
        # Calculate output shape
        if self.padding == 'same':
            out_h = int(np.ceil(h / self.strides[0]))
            out_w = int(np.ceil(w / self.strides[1]))
        else:  # 'valid'
            out_h = int((h - self.pool_size[0]) / self.strides[0]) + 1
            out_w = int((w - self.pool_size[1]) / self.strides[1]) + 1
            
        self.output_shape = (input_shape[0], out_h, out_w, c)
        self.built = True
        
    def _add_padding(self, x):
        """Add padding to input."""
        if self.padding == 'valid':
            return x
            
        # Calculate padding for 'same'
        h, w = x.shape[1:3]
        
        if h % self.strides[0] == 0:
            pad_h = max(self.pool_size[0] - self.strides[0], 0)
        else:
            pad_h = max(self.pool_size[0] - (h % self.strides[0]), 0)
            
        if w % self.strides[1] == 0:
            pad_w = max(self.pool_size[1] - self.strides[1], 0)
        else:
            pad_w = max(self.pool_size[1] - (w % self.strides[1]), 0)
            
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad with -inf for max pooling
        return np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                     mode='constant', constant_values=-np.inf)
        
    def forward(self, inputs, training=True):
        """Forward propagation."""
        if not self.built:
            self.build(inputs.shape)
            
        self.cache['inputs'] = inputs
        
        # Add padding if needed
        x_padded = self._add_padding(inputs)
        
        batch_size, h, w, c = x_padded.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.strides
        
        # Calculate output dimensions
        out_h = (h - pool_h) // stride_h + 1
        out_w = (w - pool_w) // stride_w + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_h, out_w, c))
        
        # Store max indices for backward pass
        self.cache['max_indices'] = np.zeros((batch_size, out_h, out_w, c, 2), dtype=np.int32)
        
        # Perform max pooling
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride_h
                    h_end = h_start + pool_h
                    w_start = j * stride_w
                    w_end = w_start + pool_w
                    
                    for ch in range(c):
                        pool_region = x_padded[b, h_start:h_end, w_start:w_end, ch]
                        max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        output[b, i, j, ch] = pool_region[max_idx]
                        
                        # Store absolute indices
                        self.cache['max_indices'][b, i, j, ch, 0] = h_start + max_idx[0]
                        self.cache['max_indices'][b, i, j, ch, 1] = w_start + max_idx[1]
                        
        self.cache['x_padded_shape'] = x_padded.shape
        return output
        
    def backward(self, grad_output):
        """Backward propagation."""
        batch_size, out_h, out_w, c = grad_output.shape
        grad_x_padded = np.zeros(self.cache['x_padded_shape'])
        
        # Distribute gradients to max locations
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    for ch in range(c):
                        h_idx = self.cache['max_indices'][b, i, j, ch, 0]
                        w_idx = self.cache['max_indices'][b, i, j, ch, 1]
                        grad_x_padded[b, h_idx, w_idx, ch] += grad_output[b, i, j, ch]
                        
        # Remove padding
        if self.padding == 'same':
            inputs_shape = self.cache['inputs'].shape
            h, w = inputs_shape[1:3]
            h_padded, w_padded = grad_x_padded.shape[1:3]
            
            pad_h = h_padded - h
            pad_w = w_padded - w
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if pad_h > 0 or pad_w > 0:
                grad_x = grad_x_padded[:, pad_top:h_padded-pad_bottom,
                                     pad_left:w_padded-pad_right, :]
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


class AveragePool2D(Layer):
    """2D Average Pooling layer."""
    
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        """
        Initialize AveragePool2D layer.
        
        Args:
            pool_size: Size of pooling window (int or tuple)
            strides: Stride of pooling (defaults to pool_size)
            padding: 'valid' or 'same'
        """
        super().__init__()
        
        # Handle pool size
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        # Handle strides (default to pool_size)
        if strides is None:
            self.strides = self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = strides
            
        self.padding = padding
        self.trainable = False  # No trainable parameters
        
    def build(self, input_shape):
        """Build the layer."""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input, got shape {input_shape}")
            
        self.input_shape = input_shape
        _, h, w, c = input_shape
        
        # Calculate output shape
        if self.padding == 'same':
            out_h = int(np.ceil(h / self.strides[0]))
            out_w = int(np.ceil(w / self.strides[1]))
        else:  # 'valid'
            out_h = int((h - self.pool_size[0]) / self.strides[0]) + 1
            out_w = int((w - self.pool_size[1]) / self.strides[1]) + 1
            
        self.output_shape = (input_shape[0], out_h, out_w, c)
        self.built = True
        
    def _add_padding(self, x):
        """Add padding to input."""
        if self.padding == 'valid':
            return x
            
        # Calculate padding for 'same'
        h, w = x.shape[1:3]
        
        if h % self.strides[0] == 0:
            pad_h = max(self.pool_size[0] - self.strides[0], 0)
        else:
            pad_h = max(self.pool_size[0] - (h % self.strides[0]), 0)
            
        if w % self.strides[1] == 0:
            pad_w = max(self.pool_size[1] - self.strides[1], 0)
        else:
            pad_w = max(self.pool_size[1] - (w % self.strides[1]), 0)
            
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                     mode='constant', constant_values=0)
        
    def forward(self, inputs, training=True):
        """Forward propagation."""
        if not self.built:
            self.build(inputs.shape)
            
        self.cache['inputs'] = inputs
        
        # Add padding if needed
        x_padded = self._add_padding(inputs)
        self.cache['x_padded_shape'] = x_padded.shape
        
        batch_size, h, w, c = x_padded.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.strides
        
        # Calculate output dimensions
        out_h = (h - pool_h) // stride_h + 1
        out_w = (w - pool_w) // stride_w + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_h, out_w, c))
        
        # Perform average pooling
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride_h
                    h_end = h_start + pool_h
                    w_start = j * stride_w
                    w_end = w_start + pool_w
                    
                    pool_region = x_padded[b, h_start:h_end, w_start:w_end, :]
                    output[b, i, j, :] = np.mean(pool_region, axis=(0, 1))
                    
        return output
        
    def backward(self, grad_output):
        """Backward propagation."""
        batch_size, out_h, out_w, c = grad_output.shape
        grad_x_padded = np.zeros(self.cache['x_padded_shape'])
        
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.strides
        pool_area = pool_h * pool_w
        
        # Distribute gradients uniformly
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride_h
                    h_end = h_start + pool_h
                    w_start = j * stride_w
                    w_end = w_start + pool_w
                    
                    # Average gradient over pool region
                    grad_x_padded[b, h_start:h_end, w_start:w_end, :] += \
                        grad_output[b, i, j, :] / pool_area
                        
        # Remove padding
        if self.padding == 'same':
            inputs_shape = self.cache['inputs'].shape
            h, w = inputs_shape[1:3]
            h_padded, w_padded = grad_x_padded.shape[1:3]
            
            pad_h = h_padded - h
            pad_w = w_padded - w
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if pad_h > 0 or pad_w > 0:
                grad_x = grad_x_padded[:, pad_top:h_padded-pad_bottom,
                                     pad_left:w_padded-pad_right, :]
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