"""
Base types and configurations for the Neural Embedding Engine.

This module contains all the fundamental data structures, enums, and configurations
used throughout the ML embedding system.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import torch


class EmbeddingType(Enum):
    """Embedding type enumeration."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CODE = "code"
    STRUCTURED = "structured"


class OptimizationStrategy(Enum):
    """Optimization strategy enumeration."""
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    SUPERVISED = "supervised"
    SELF_SUPERVISED = "self_supervised"
    ADVERSARIAL = "adversarial"


@dataclass
class EmbeddingConfig:
    """Configuration for neural embedding engine."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_sequence_length: int = 512
    batch_size: int = 32
    device: str = "auto"
    cache_embeddings: bool = True
    normalize_embeddings: bool = True
    fine_tuning_enabled: bool = True
    fine_tune_enabled: bool = True  # Alias for compatibility
    transfer_learning: bool = True
    multi_modal_fusion: str = "concatenation"  # concatenation, attention, cross_modal
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.CONTRASTIVE
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    evaluation_frequency: int = 100
    clustering_algorithm: str = "kmeans"
    anomaly_detection_threshold: float = 0.8


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: torch.Tensor
    metadata: Dict[str, Any]
    processing_time: float
    model_version: str
    confidence_scores: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for embeddings."""
    silhouette_score: float
    intra_cluster_distance: float
    inter_cluster_distance: float
    embedding_quality_score: float
    retrieval_accuracy: float
    clustering_purity: float
    anomaly_detection_f1: float


@dataclass
class ModelMetrics:
    """Metrics for model performance tracking."""
    training_loss: float = 0.0
    validation_loss: float = 0.0
    embedding_quality: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    success: bool = False
    improvement: float = 0.0
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    strategy_used: str = ""
    iterations: int = 0
    convergence_time: float = 0.0
    
    @property
    def improvement_score(self) -> float:
        """Get improvement score (alias for improvement)."""
        return self.improvement
    
    @property
    def final_loss(self) -> float:
        """Get final training loss from metrics."""
        return self.metrics.training_loss if self.metrics else 0.0