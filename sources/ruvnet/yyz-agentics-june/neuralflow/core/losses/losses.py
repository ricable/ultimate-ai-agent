"""
Loss functions implementation
"""
import numpy as np
from typing import Union
from ..tensor import Tensor


# Functional API
def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Mean Squared Error loss."""
    diff = y_pred - y_true
    loss = (diff * diff).mean()
    return loss


def mean_absolute_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Mean Absolute Error loss."""
    diff = y_pred - y_true
    abs_diff = Tensor(np.abs(diff.data), requires_grad=diff.requires_grad)
    
    def _backward(grad):
        if diff.requires_grad:
            diff.backward(grad * np.sign(diff.data) / diff.data.size)
    
    if abs_diff.requires_grad:
        abs_diff._grad_fn = _backward
        abs_diff._prev = {diff}
    
    return abs_diff.mean()


def binary_crossentropy(y_true: Tensor, y_pred: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Binary Cross-Entropy loss."""
    # Clip predictions to prevent log(0)
    y_pred_clipped = Tensor(np.clip(y_pred.data, epsilon, 1 - epsilon), 
                            requires_grad=y_pred.requires_grad)
    
    # Calculate loss
    loss_data = -(y_true.data * np.log(y_pred_clipped.data) + 
                  (1 - y_true.data) * np.log(1 - y_pred_clipped.data))
    
    loss = Tensor(loss_data, requires_grad=y_pred.requires_grad)
    
    def _backward(grad):
        if y_pred.requires_grad:
            # Gradient of binary crossentropy
            grad_input = grad * (-(y_true.data / y_pred_clipped.data) + 
                                (1 - y_true.data) / (1 - y_pred_clipped.data))
            y_pred.backward(grad_input / y_pred.data.size)
    
    if loss.requires_grad:
        loss._grad_fn = _backward
        loss._prev = {y_pred}
    
    return loss.mean()


def categorical_crossentropy(y_true: Tensor, y_pred: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Categorical Cross-Entropy loss."""
    # Clip predictions to prevent log(0)
    y_pred_clipped = Tensor(np.clip(y_pred.data, epsilon, 1 - epsilon), 
                            requires_grad=y_pred.requires_grad)
    
    # Calculate loss
    loss_data = -np.sum(y_true.data * np.log(y_pred_clipped.data), axis=-1)
    
    loss = Tensor(loss_data, requires_grad=y_pred.requires_grad)
    
    def _backward(grad):
        if y_pred.requires_grad:
            # Gradient of categorical crossentropy
            grad_input = -y_true.data / y_pred_clipped.data
            # Expand grad to match dimensions
            if len(grad.shape) < len(grad_input.shape):
                grad = grad.reshape(grad.shape + (1,) * (len(grad_input.shape) - len(grad.shape)))
            y_pred.backward(grad * grad_input / y_pred.shape[0])
    
    if loss.requires_grad:
        loss._grad_fn = _backward
        loss._prev = {y_pred}
    
    return loss.mean()


# Loss classes
class Loss:
    """Base class for loss functions."""
    
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return self.forward(y_true, y_pred)
    
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error loss class."""
    
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return mean_squared_error(y_true, y_pred)


class MAE(Loss):
    """Mean Absolute Error loss class."""
    
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return mean_absolute_error(y_true, y_pred)


class BinaryCrossentropy(Loss):
    """Binary Cross-Entropy loss class."""
    
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
    
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return binary_crossentropy(y_true, y_pred, self.epsilon)


class CategoricalCrossentropy(Loss):
    """Categorical Cross-Entropy loss class."""
    
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
    
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return categorical_crossentropy(y_true, y_pred, self.epsilon)


# Helper function to get loss by name
def get_loss(loss: Union[str, Loss]) -> Loss:
    """Get loss function by name or return the loss if already instantiated."""
    if isinstance(loss, Loss):
        return loss
    
    loss_map = {
        'mse': MSE,
        'mean_squared_error': MSE,
        'mae': MAE,
        'mean_absolute_error': MAE,
        'binary_crossentropy': BinaryCrossentropy,
        'categorical_crossentropy': CategoricalCrossentropy
    }
    
    if loss.lower() in loss_map:
        return loss_map[loss.lower()]()
    else:
        raise ValueError(f"Unknown loss function: {loss}")