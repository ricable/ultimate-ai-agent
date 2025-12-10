"""
Flow2 Utilities
===============

Common utilities and helper functions for the Flow2 package.

Modules:
    utils: General utility functions for quantization and benchmarking
    model_manager: Model download and management utilities
    model_cli: Command-line interface for model operations
    verify_model: Model verification utilities
"""

from .utils import (
    calculate_file_hash, 
    verify_model_integrity,
    get_model_size,
    get_human_readable_size,
    get_system_ram_gb,
    get_apple_silicon_model,
    get_recommended_quantization,
    measure_inference_time,
    save_benchmark_results,
    load_benchmark_results,
    format_quantization_info,
    LLAMACPP_QUANT_METHODS,
    MLX_QUANT_METHODS
)

# Import model management utilities if available
try:
    from .model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False

try:
    from .model_cli import main as model_cli_main
    MODEL_CLI_AVAILABLE = True
except ImportError:
    MODEL_CLI_AVAILABLE = False

__all__ = [
    "calculate_file_hash", 
    "verify_model_integrity",
    "get_model_size",
    "get_human_readable_size",
    "get_system_ram_gb",
    "get_apple_silicon_model",
    "get_recommended_quantization",
    "measure_inference_time",
    "save_benchmark_results",
    "load_benchmark_results",
    "format_quantization_info",
    "LLAMACPP_QUANT_METHODS",
    "MLX_QUANT_METHODS",
    "MODEL_MANAGER_AVAILABLE",
    "MODEL_CLI_AVAILABLE",
]

if MODEL_MANAGER_AVAILABLE:
    __all__.append("ModelManager")

if MODEL_CLI_AVAILABLE:
    __all__.append("model_cli_main")