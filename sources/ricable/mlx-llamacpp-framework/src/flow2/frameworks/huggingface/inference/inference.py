"""
HuggingFace Inference Implementation
===================================

High-performance inference with MPS, TGI, and streaming support.
"""

import os
import time
import torch
import warnings
from typing import Dict, List, Optional, Union, Iterator, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoConfig,
        TextStreamer,
        pipeline,
        GenerationConfig,
        BitsAndBytesConfig
    )
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("HuggingFace transformers not available")

try:
    from text_generation import Client
    TGI_AVAILABLE = True
except ImportError:
    TGI_AVAILABLE = False

from ..utils import setup_mps_device, get_optimal_dtype, optimize_for_inference, cleanup_memory

@dataclass
class GenerationParams:
    """Parameters for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transformers."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "use_cache": self.use_cache
        }

def load_hf_model(
    model_name: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    quantization_config: Optional[Dict[str, Any]] = None,
    use_fast_tokenizer: bool = True,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load HuggingFace model with optimizations.
    
    Args:
        model_name: Model name or path
        device: Target device (auto-detected if None)
        dtype: Data type (auto-detected if None)
        quantization_config: Quantization configuration
        use_fast_tokenizer: Use fast tokenizer
        trust_remote_code: Trust remote code
        cache_dir: Cache directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace transformers not available")
    
    print(f"Loading model: {model_name}")
    
    # Setup device and dtype
    if device is None:
        device = setup_mps_device()
    
    if dtype is None:
        dtype = get_optimal_dtype(device)
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Prepare model loading arguments
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "torch_dtype": dtype
    }
    
    # Add quantization config if provided
    if quantization_config:
        if "load_in_4bit" in quantization_config or "load_in_8bit" in quantization_config:
            # BitsAndBytes quantization
            bnb_config = BitsAndBytesConfig(**quantization_config)
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"  # Required for quantization
        else:
            # Regular quantization
            model_kwargs.update(quantization_config)
    else:
        # Regular loading
        model_kwargs["device_map"] = device if device.type != "cpu" else None
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Optimize for inference if not quantized  
        if not quantization_config:
            model = optimize_for_inference(model, device, dtype)
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Print model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Fallback to CPU if GPU loading fails
        if device.type != "cpu":
            print("Falling back to CPU...")
            model_kwargs["device_map"] = None
            model_kwargs.pop("quantization_config", None)  # Remove quantization for CPU
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            model = model.to("cpu")
            return model, tokenizer
        else:
            raise e

def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    generation_params: Optional[GenerationParams] = None,
    return_full_text: bool = False
) -> str:
    """
    Generate text completion.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        generation_params: Generation parameters
        return_full_text: Return full text including prompt
        
    Returns:
        Generated text
    """
    if generation_params is None:
        generation_params = GenerationParams()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set pad token ID if not set
    if generation_params.pad_token_id is None:
        generation_params.pad_token_id = tokenizer.pad_token_id
    
    if generation_params.eos_token_id is None:
        generation_params.eos_token_id = tokenizer.eos_token_id
    
    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_params.to_dict()
        )
    
    # Decode output
    if return_full_text:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # Only return new tokens
        new_tokens = outputs[0][len(inputs["input_ids"][0]):]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return generated_text.strip()

def streaming_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    generation_params: Optional[GenerationParams] = None,
    print_output: bool = True
) -> Iterator[str]:
    """
    Generate streaming text completion.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        generation_params: Generation parameters
        print_output: Print tokens as they're generated
        
    Yields:
        Generated tokens
    """
    if generation_params is None:
        generation_params = GenerationParams()
    
    # Setup streamer
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set pad token ID if not set
    if generation_params.pad_token_id is None:
        generation_params.pad_token_id = tokenizer.pad_token_id
    
    if generation_params.eos_token_id is None:
        generation_params.eos_token_id = tokenizer.eos_token_id
    
    # Generate with streaming
    generation_kwargs = generation_params.to_dict()
    generation_kwargs["streamer"] = streamer if print_output else None
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )
    
    # If not printing, yield tokens manually
    if not print_output:
        new_tokens = outputs[0][len(inputs["input_ids"][0]):]
        for token_id in new_tokens:
            token = tokenizer.decode([token_id], skip_special_tokens=True)
            if token:
                yield token

def chat_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    generation_params: Optional[GenerationParams] = None,
    system_message: Optional[str] = None
) -> str:
    """
    Generate chat completion with proper formatting.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        messages: Chat messages [{"role": "user", "content": "..."}]
        generation_params: Generation parameters
        system_message: System message to prepend
        
    Returns:
        Assistant response
    """
    # Prepare chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        # Use model's chat template
        if system_message:
            messages = [{"role": "system", "content": system_message}] + messages
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback template
        prompt_parts = []
        
        if system_message:
            prompt_parts.append(f"System: {system_message}")
        
        for message in messages:
            role = message["role"].title()
            content = message["content"]
            prompt_parts.append(f"{role}: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
    
    return generate_completion(model, tokenizer, prompt, generation_params)

def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generation_params: Optional[GenerationParams] = None,
    batch_size: int = 4
) -> List[str]:
    """
    Generate completions for multiple prompts in batches.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of input prompts
        generation_params: Generation parameters
        batch_size: Batch size for processing
        
    Returns:
        List of generated texts
    """
    if generation_params is None:
        generation_params = GenerationParams()
    
    results = []
    device = next(model.parameters()).device
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Set pad token ID if not set
        if generation_params.pad_token_id is None:
            generation_params.pad_token_id = tokenizer.pad_token_id
        
        if generation_params.eos_token_id is None:
            generation_params.eos_token_id = tokenizer.eos_token_id
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **generation_params.to_dict()
            )
        
        # Decode outputs
        for j, output in enumerate(outputs):
            # Only return new tokens
            input_length = len(inputs["input_ids"][j])
            new_tokens = output[input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(generated_text.strip())
    
    return results

class TGIClient:
    """Text Generation Inference (TGI) client wrapper."""
    
    def __init__(self, endpoint: str, timeout: int = 120):
        """
        Initialize TGI client.
        
        Args:
            endpoint: TGI server endpoint
            timeout: Request timeout in seconds
        """
        if not TGI_AVAILABLE:
            raise ImportError("Text Generation Inference client not available")
        
        self.client = Client(endpoint, timeout=timeout)
        self.endpoint = endpoint
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text completion via TGI."""
        response = self.client.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences or []
        )
        return response.generated_text
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> Iterator[str]:
        """Generate streaming text completion via TGI."""
        for response in self.client.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences or []
        ):
            if not response.token.special:
                yield response.token.text
    
    def health_check(self) -> bool:
        """Check if TGI server is healthy."""
        try:
            # Simple test generation
            response = self.client.generate("Hello", max_new_tokens=1)
            return True
        except Exception:
            return False

@contextmanager
def inference_context(cleanup_after: bool = True):
    """Context manager for inference with automatic cleanup."""
    try:
        yield
    finally:
        if cleanup_after:
            cleanup_memory()