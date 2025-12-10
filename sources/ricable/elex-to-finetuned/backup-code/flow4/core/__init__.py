"""Core functionality for document processing pipeline."""

from .converter import DocumentConverter
from .chunker import DocumentChunker
from .cleaner import HTMLCleaner, MarkdownCleaner
from .pipeline import DocumentPipeline

# Chat interfaces
from .chat_interface import DocumentChatInterface, ModelChatInterface

# Dataset generation and optimization
from .dataset_generator import (
    FineTuningDatasetGenerator,
    OptimizedFineTuneDatasetGenerator,
    ChunkOptimizer,
    DatasetConfig,
    DatasetComparator
)

# Optional MLX fine-tuning (only available if MLX is installed)
try:
    from .mlx_finetuner import MLXFineTuner
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False
    MLXFineTuner = None

# Optional Augmentoolkit (only available if Augmentoolkit is installed)
try:
    from .augmentoolkit_generator import AugmentoolkitGenerator, AugmentoolkitConfig
    _AUGMENTOOLKIT_AVAILABLE = True
except ImportError:
    _AUGMENTOOLKIT_AVAILABLE = False
    AugmentoolkitGenerator = None
    AugmentoolkitConfig = None

__all__ = [
    "DocumentConverter",
    "DocumentChunker", 
    "HTMLCleaner",
    "MarkdownCleaner",
    "DocumentPipeline",
    "DocumentChatInterface",
    "ModelChatInterface",
    "FineTuningDatasetGenerator",
    "OptimizedFineTuneDatasetGenerator",
    "ChunkOptimizer",
    "DatasetConfig",
    "DatasetComparator",
]

if _MLX_AVAILABLE:
    __all__.append("MLXFineTuner")

if _AUGMENTOOLKIT_AVAILABLE:
    __all__.extend(["AugmentoolkitGenerator", "AugmentoolkitConfig"])