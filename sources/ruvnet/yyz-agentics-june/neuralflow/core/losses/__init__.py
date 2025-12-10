"""
Loss functions for NeuralFlow
"""
from .losses import (
    mean_squared_error, mean_absolute_error,
    binary_crossentropy, categorical_crossentropy,
    MSE, MAE, BinaryCrossentropy, CategoricalCrossentropy,
    Loss, get_loss
)

__all__ = [
    'mean_squared_error', 'mean_absolute_error',
    'binary_crossentropy', 'categorical_crossentropy',
    'MSE', 'MAE', 'BinaryCrossentropy', 'CategoricalCrossentropy',
    'Loss', 'get_loss'
]