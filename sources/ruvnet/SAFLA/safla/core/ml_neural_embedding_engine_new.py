"""
Neural Embedding Engine - Compatibility layer for the modular ML system.

This module serves as a compatibility layer that imports components from the new
modular ML package structure, maintaining backward compatibility while providing
the refactored architecture.

The actual implementation has been refactored into:
- safla/core/ml/base.py (90 lines) - Base types and configurations
- safla/core/ml/models.py (340 lines) - Transformer models and loss functions
- safla/core/ml/engine.py (300 lines) - Main embedding engine
- safla/core/ml/optimizer.py (320 lines) - Optimization strategies
- safla/core/ml/versioning.py (280 lines) - Model version management
- safla/core/ml/transfer.py (400 lines) - Transfer learning capabilities

Usage:
    from safla.core.ml_neural_embedding_engine import NeuralEmbeddingEngine
    # or
    from safla.core.ml import NeuralEmbeddingEngine
    
    config = EmbeddingConfig()
    engine = NeuralEmbeddingEngine(config)
    result = await engine.generate_embeddings("sample text")
"""

# Import all components from the new modular structure
from safla.core.ml import (
    # Base types and configs
    EmbeddingType,
    OptimizationStrategy,
    EmbeddingConfig,
    EmbeddingResult,
    EvaluationMetrics,
    ModelMetrics,
    OptimizationResult,
    
    # Core components
    TransformerEmbeddingModel,
    NeuralEmbeddingEngine,
    EmbeddingOptimizer,
    ModelVersionManager,
    TransferLearningManager
)

# Import additional model components
from safla.core.ml.models import (
    MultiModalProjector,
    ContrastiveLoss,
    TripletLoss
)

# Re-export for backward compatibility
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
    'TransferLearningManager',
    
    # Model utilities
    'MultiModalProjector',
    'ContrastiveLoss',
    'TripletLoss'
]

# Legacy aliases for backward compatibility
MLNeuralEmbeddingEngine = NeuralEmbeddingEngine  # Legacy alias