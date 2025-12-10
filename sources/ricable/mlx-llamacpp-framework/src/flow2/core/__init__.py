"""
Core module for Flow2
=====================

Contains core functionality including Flash Attention implementation,
baseline benchmarks, and common utilities.
"""

# Conditional imports based on MLX availability
try:
    from .flash_attention import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    OptimizedMLXMultiHeadAttention = None
    FlashAttentionBenchmark = None

__all__ = [
    "FLASH_ATTENTION_AVAILABLE",
    "OptimizedMLXMultiHeadAttention",
    "FlashAttentionBenchmark",
]