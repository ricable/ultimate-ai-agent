"""
Base Tensor class for NeuralFlow
"""
import numpy as np
from typing import Optional, Union, List, Tuple


class Tensor:
    """
    Base tensor class with automatic differentiation support.
    """
    
    def __init__(self, data: Union[np.ndarray, List], requires_grad: bool = False):
        """Initialize a tensor."""
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._grad_fn = None
        self._prev = set()
        
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def T(self) -> 'Tensor':
        """Transpose of the tensor."""
        return self.transpose()
    
    def backward(self, grad: Optional[np.ndarray] = None):
        """Perform backpropagation."""
        if not self.requires_grad:
            return
            
        if grad is None:
            grad = np.ones_like(self.data)
            
        if self.grad is None:
            self.grad = grad
        else:
            # Handle shape mismatches by proper broadcasting/summing
            if self.grad.shape == grad.shape:
                self.grad = self.grad + grad
            else:
                # Try to broadcast - if shapes are compatible
                try:
                    self.grad = self.grad + grad
                except ValueError:
                    # Sum the incoming gradient to match existing gradient shape
                    grad_to_add = grad
                    # Sum across dimensions that don't match
                    if len(grad.shape) > len(self.grad.shape):
                        # Sum over extra leading dimensions
                        for _ in range(len(grad.shape) - len(self.grad.shape)):
                            grad_to_add = np.sum(grad_to_add, axis=0)
                    elif len(grad.shape) == len(self.grad.shape):
                        # Same number of dimensions but different sizes
                        for i in range(len(self.grad.shape)):
                            if self.grad.shape[i] != grad_to_add.shape[i]:
                                # Sum over the mismatched dimension
                                grad_to_add = np.sum(grad_to_add, axis=i, keepdims=True)
                    self.grad = self.grad + grad_to_add
        
        if self._grad_fn:
            self._grad_fn(grad)
    
    def zero_grad(self):
        """Zero out the gradient."""
        self.grad = None
        
    # Mathematical operations
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Addition operation."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad)
            if other.requires_grad:
                other.backward(grad)
                
        if out.requires_grad:
            out._grad_fn = _backward
            out._prev = {self, other}
            
        return out
    
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Subtraction operation."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad)
            if other.requires_grad:
                other.backward(-grad)
                
        if out.requires_grad:
            out._grad_fn = _backward
            out._prev = {self, other}
            
        return out
    
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad * other.data)
            if other.requires_grad:
                other.backward(grad * self.data)
                
        if out.requires_grad:
            out._grad_fn = _backward
            out._prev = {self, other}
            
        return out
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad @ other.data.T)
            if other.requires_grad:
                other.backward(self.data.T @ grad)
                
        if out.requires_grad:
            out._grad_fn = _backward
            out._prev = {self, other}
            
        return out
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Sum operation."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        
        def _backward(grad):
            if self.requires_grad:
                grad_expanded = grad
                if not keepdims and axis is not None:
                    # Expand dimensions that were summed
                    shape = list(self.data.shape)
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in sorted(axes):
                        grad_expanded = np.expand_dims(grad_expanded, axis=ax)
                        
                # Broadcast to original shape
                self.backward(np.ones_like(self.data) * grad_expanded)
                
        if out.requires_grad:
            out._grad_fn = _backward
            out._prev = {self}
            
        return out
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Mean operation."""
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        
        def _backward(grad):
            if self.requires_grad:
                grad_expanded = grad
                if not keepdims and axis is not None:
                    # Expand dimensions that were averaged
                    shape = list(self.data.shape)
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in sorted(axes):
                        grad_expanded = np.expand_dims(grad_expanded, axis=ax)
                
                # Calculate the number of elements that were averaged
                n_elements = self.data.size / out.data.size
                self.backward(np.ones_like(self.data) * grad_expanded / n_elements)
                
        if out.requires_grad:
            out._grad_fn = _backward
            out._prev = {self}
            
        return out
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Transpose operation."""
        if axes is None:
            transposed_data = self.data.T
            inverse_axes = None
        else:
            transposed_data = np.transpose(self.data, axes)
            # Calculate inverse permutation
            inverse_axes = list(range(len(axes)))
            for i, ax in enumerate(axes):
                inverse_axes[ax] = i
                
        out = Tensor(transposed_data, requires_grad=self.requires_grad)
        
        def _backward(grad):
            if self.requires_grad:
                if inverse_axes is None:
                    self.backward(grad.T)
                else:
                    self.backward(np.transpose(grad, inverse_axes))
                    
        if out.requires_grad:
            out._grad_fn = _backward
            out._prev = {self}
            
        return out
    
    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        """Reshape operation."""
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        
        def _backward(grad):
            if self.requires_grad:
                self.backward(grad.reshape(self.data.shape))
                
        if out.requires_grad:
            out._grad_fn = _backward
            out._prev = {self}
            
        return out
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of zeros."""
        return Tensor(np.zeros(shape), requires_grad=requires_grad)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of ones."""
        return Tensor(np.ones(shape), requires_grad=requires_grad)
    
    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor with random normal values."""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)
    
    @staticmethod
    def uniform(shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0, requires_grad: bool = False) -> 'Tensor':
        """Create a tensor with uniform random values."""
        return Tensor(np.random.uniform(low, high, shape), requires_grad=requires_grad)