"""
Machine Learning module for SAFLA Neural Embedding Engine.

This package provides the ML capabilities for the SAFLA system,
split from the original monolithic ml_neural_embedding_engine.py into modular components.
"""

from .base import (
    EmbeddingType,
    OptimizationStrategy,
    EmbeddingConfig,
    EmbeddingResult,
    EvaluationMetrics,
    ModelMetrics,
    OptimizationResult
)
from .models import TransformerEmbeddingModel
from .engine import NeuralEmbeddingEngine
from .optimizer import EmbeddingOptimizer
from .versioning import ModelVersionManager
from .transfer import TransferLearningManager

__all__ = [
    # Base types and configs
    'EmbeddingType',
    'OptimizationStrategy', 
    'EmbeddingConfig',
    'EmbeddingResult',
    'EvaluationMetrics',
    'ModelMetrics',
    'OptimizationResult',
    
    # Core components
    'TransformerEmbeddingModel',
    'NeuralEmbeddingEngine',
    'EmbeddingOptimizer',
    'ModelVersionManager',
    'TransferLearningManager'
]