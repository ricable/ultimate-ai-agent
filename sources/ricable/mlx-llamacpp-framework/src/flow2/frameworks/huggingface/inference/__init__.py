"""
HuggingFace Inference Module
===========================

High-performance inference utilities for HuggingFace transformers with:
- MPS (Metal Performance Shaders) optimization for Apple Silicon
- Streaming text generation
- Batch processing
- Chat completions with proper templating
- TGI (Text Generation Inference) integration
"""

from .inference import (
    load_hf_model,
    generate_completion,
    chat_completion,
    streaming_completion,
    batch_generate,
    TGIClient
)

__all__ = [
    "load_hf_model",
    "generate_completion", 
    "chat_completion",
    "streaming_completion",
    "batch_generate",
    "TGIClient"
]