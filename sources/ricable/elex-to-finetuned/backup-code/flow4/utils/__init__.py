"""Utility functions and configuration management."""

from .config import PipelineConfig, DoclingConfig, CLIConfig, MLXConfig
from .logging import setup_logging, get_logger

# Optional imports with graceful fallback
try:
    from .augmentoolkit_config import AugmentoolkitConfigManager
    HAS_AUGMENTOOLKIT_CONFIG = True
except ImportError:
    HAS_AUGMENTOOLKIT_CONFIG = False
    AugmentoolkitConfigManager = None

try:
    from .deduplication import deduplicate_instruction_dataset, deduplicate_rag_datasets
    HAS_DEDUPLICATION = True
except (ImportError, RuntimeError, UserWarning, Exception) as e:
    # Handle various import issues including NumPy compatibility problems
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    HAS_DEDUPLICATION = False
    deduplicate_instruction_dataset = None
    deduplicate_rag_datasets = None

__all__ = [
    "PipelineConfig",
    "DoclingConfig", 
    "CLIConfig",
    "MLXConfig",
    "setup_logging",
    "get_logger",
]

if HAS_AUGMENTOOLKIT_CONFIG:
    __all__.append("AugmentoolkitConfigManager")

if HAS_DEDUPLICATION:
    __all__.extend(["deduplicate_instruction_dataset", "deduplicate_rag_datasets"])