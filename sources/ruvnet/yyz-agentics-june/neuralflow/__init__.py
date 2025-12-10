"""
NeuralFlow: A Comprehensive Neural Network Library
================================================

NeuralFlow is a flexible and easy-to-use neural network library designed for 
rapid prototyping and deployment of deep learning models.

Example:
    >>> import neuralflow as nf
    >>> model = nf.Sequential([
    ...     nf.layers.Dense(128, activation='relu'),
    ...     nf.layers.Dense(10, activation='softmax')
    ... ])
    >>> model.compile(optimizer='adam', loss='categorical_crossentropy')
    >>> model.fit(X_train, y_train, epochs=10)
"""

from .core import layers, activations, optimizers, losses
from .models import Sequential, Model
from .utils import data_utils, visualization

__version__ = "1.0.0"
__author__ = "NeuralFlow Team"

__all__ = [
    'layers',
    'activations', 
    'optimizers',
    'losses',
    'Sequential',
    'Model',
    'data_utils',
    'visualization'
]