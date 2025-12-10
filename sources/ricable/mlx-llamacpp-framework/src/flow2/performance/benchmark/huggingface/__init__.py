"""
HuggingFace Framework Benchmarks
================================

Benchmarking suite for HuggingFace transformers with MPS acceleration.
"""

try:
    from .finetuning_test import (
        test_hf_lora_finetuning,
        test_hf_quantization,
        run_huggingface_benchmark
    )
    HF_FINETUNING_AVAILABLE = True
except ImportError:
    HF_FINETUNING_AVAILABLE = False

__all__ = []

if HF_FINETUNING_AVAILABLE:
    __all__.extend([
        'test_hf_lora_finetuning',
        'test_hf_quantization',
        'run_huggingface_benchmark'
    ])