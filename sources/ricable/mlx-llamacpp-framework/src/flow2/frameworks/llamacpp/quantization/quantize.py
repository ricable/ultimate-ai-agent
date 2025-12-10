#!/usr/bin/env python3
"""
Utilities for quantizing models using llama.cpp.
"""

import os
import sys
import subprocess
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from flow2.utils.utils import (
    get_model_size,
    get_human_readable_size,
    verify_model_integrity,
    measure_inference_time,
    LLAMACPP_QUANT_METHODS
)

logger = logging.getLogger('quantization_utils.llamacpp')

# Default paths
DEFAULT_LLAMA_CPP_PATH = os.path.expanduser("~/dev/ran/flow2/llama.cpp-setup")
DEFAULT_MODELS_DIR = "models"

def get_available_quant_types() -> List[str]:
    """Get a list of available quantization types for llama.cpp."""
    return list(LLAMACPP_QUANT_METHODS.keys())

def quantize_model(
    input_file: str, 
    output_file: str, 
    quant_type: str = "Q4_K",
    llama_cpp_path: Optional[str] = None
) -> Dict:
    """
    Quantize a GGUF model using llama.cpp's quantize tool.
    
    Args:
        input_file: Path to the input GGUF model
        output_file: Path to save the quantized model
        quant_type: Quantization type (e.g., Q4_K, Q8_0)
        llama_cpp_path: Path to the llama.cpp directory
        
    Returns:
        Dict containing information about the quantization process
    """
    if llama_cpp_path is None:
        llama_cpp_path = DEFAULT_LLAMA_CPP_PATH
    
    # Normalize quantization type
    quant_type = quant_type.upper()
    
    # Validate quantization type
    if quant_type not in LLAMACPP_QUANT_METHODS:
        raise ValueError(f"Unsupported quantization type: {quant_type}. "
                         f"Supported types: {', '.join(LLAMACPP_QUANT_METHODS.keys())}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input model file not found: {input_file}")
    
    # Check if quantize.py exists
    quantize_script = os.path.join(llama_cpp_path, "quantize.py")
    if not os.path.exists(quantize_script):
        raise FileNotFoundError(f"quantize.py not found at {quantize_script}. "
                               f"Please ensure llama.cpp is properly installed.")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get original model size
    original_size = get_model_size(input_file)
    
    # Quantize the model
    logger.info(f"Quantizing {input_file} to {quant_type}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [
                sys.executable, quantize_script,
                "--model", input_file,
                "--outfile", output_file,
                "--type", quant_type.lower()  # llama.cpp expects lowercase
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        end_time = time.time()
        quantization_time = end_time - start_time
        
        # Get quantized model size
        quantized_size = get_model_size(output_file)
        size_reduction = 1 - (quantized_size / original_size)
        
        # Verify the quantized model
        is_valid, _ = verify_model_integrity(output_file)
        
        return {
            "status": "success",
            "input_model": input_file,
            "output_model": output_file,
            "quantization_type": quant_type,
            "original_size": original_size,
            "original_size_human": get_human_readable_size(original_size),
            "quantized_size": quantized_size,
            "quantized_size_human": get_human_readable_size(quantized_size),
            "size_reduction_percent": f"{size_reduction:.2%}",
            "quantization_time_seconds": quantization_time,
            "is_valid": is_valid,
            "quantization_info": LLAMACPP_QUANT_METHODS.get(quant_type, {})
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error quantizing model: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Quantization failed: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error during quantization: {e}")
        raise

def convert_and_quantize_from_huggingface(
    hf_model_id: str,
    output_file: str,
    quant_type: str = "Q4_K",
    llama_cpp_path: Optional[str] = None,
    use_auth_token: Optional[str] = None
) -> Dict:
    """
    Convert a Hugging Face model to GGUF and quantize it in one step.
    
    Args:
        hf_model_id: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b")
        output_file: Path to save the quantized model
        quant_type: Quantization type (e.g., Q4_K, Q8_0)
        llama_cpp_path: Path to the llama.cpp directory
        use_auth_token: Hugging Face token for accessing gated models
        
    Returns:
        Dict containing information about the conversion and quantization process
    """
    if llama_cpp_path is None:
        llama_cpp_path = DEFAULT_LLAMA_CPP_PATH
    
    # Normalize quantization type
    quant_type = quant_type.upper()
    
    # Check if convert.py exists
    convert_script = os.path.join(llama_cpp_path, "convert.py")
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"convert.py not found at {convert_script}. "
                               f"Please ensure llama.cpp is properly installed.")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert and quantize in one step
    logger.info(f"Converting and quantizing {hf_model_id} to {quant_type}...")
    start_time = time.time()
    
    try:
        cmd = [
            sys.executable, convert_script,
            "--outtype", quant_type.lower(),  # llama.cpp expects lowercase
            "--outfile", output_file
        ]
        
        # Add auth token if provided
        if use_auth_token:
            cmd.extend(["--token", use_auth_token])
        
        # Add the model ID
        cmd.append(hf_model_id)
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        end_time = time.time()
        conversion_time = end_time - start_time
        
        # Get model size
        model_size = get_model_size(output_file)
        
        # Verify the model
        is_valid, _ = verify_model_integrity(output_file)
        
        return {
            "status": "success",
            "huggingface_model": hf_model_id,
            "output_model": output_file,
            "quantization_type": quant_type,
            "model_size": model_size,
            "model_size_human": get_human_readable_size(model_size),
            "conversion_time_seconds": conversion_time,
            "is_valid": is_valid,
            "quantization_info": LLAMACPP_QUANT_METHODS.get(quant_type, {})
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting and quantizing model: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Conversion and quantization failed: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error during conversion and quantization: {e}")
        raise

def batch_quantize_models(
    input_models: List[str],
    output_dir: str,
    quant_types: List[str] = ["Q4_K", "Q8_0"],
    llama_cpp_path: Optional[str] = None
) -> Dict:
    """
    Batch quantize multiple models to multiple quantization levels.
    
    Args:
        input_models: List of paths to input GGUF models
        output_dir: Directory to save the quantized models
        quant_types: List of quantization types to apply
        llama_cpp_path: Path to the llama.cpp directory
        
    Returns:
        Dict containing information about all quantization processes
    """
    results = {}
    
    for input_model in input_models:
        model_name = os.path.basename(input_model).split('.')[0]
        model_results = {}
        
        for quant_type in quant_types:
            output_file = os.path.join(output_dir, f"{model_name}.{quant_type.lower()}.gguf")
            
            try:
                result = quantize_model(
                    input_file=input_model,
                    output_file=output_file,
                    quant_type=quant_type,
                    llama_cpp_path=llama_cpp_path
                )
                model_results[quant_type] = result
            except Exception as e:
                logger.error(f"Failed to quantize {input_model} to {quant_type}: {e}")
                model_results[quant_type] = {"status": "failed", "error": str(e)}
        
        results[model_name] = model_results
    
    return results

def measure_inference_performance(
    model_path: str,
    prompt: str = "Once upon a time",
    n_tokens: int = 100,
    temp: float = 0.7,
    llama_cpp_path: Optional[str] = None,
    use_metal: bool = True
) -> Dict:
    """
    Measure inference performance of a model.
    
    Args:
        model_path: Path to the GGUF model
        prompt: Text prompt for inference
        n_tokens: Number of tokens to generate
        temp: Temperature for sampling
        llama_cpp_path: Path to the llama.cpp directory
        use_metal: Whether to use Metal acceleration on Mac
        
    Returns:
        Dict containing performance metrics
    """
    if llama_cpp_path is None:
        llama_cpp_path = DEFAULT_LLAMA_CPP_PATH
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check if main executable exists
    main_executable = os.path.join(llama_cpp_path, "build/bin/main")
    if not os.path.exists(main_executable):
        # Try alternate path
        main_executable = os.path.join(llama_cpp_path, "main")
        if not os.path.exists(main_executable):
            raise FileNotFoundError(f"llama.cpp main executable not found. "
                                  f"Please ensure llama.cpp is properly built.")
    
    # Prepare command
    cmd = [
        main_executable,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", str(temp),
        "-c", "2048",  # Context size
        "-b", "512"    # Batch size
    ]
    
    # Add Metal acceleration if requested and on Mac
    if use_metal and sys.platform == "darwin":
        cmd.extend(["--metal", "--metal-mmq"])
    
    # Measure inference time
    logger.info(f"Measuring inference performance for {model_path}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Parse tokens per second from output
        tokens_per_second = None
        for line in result.stderr.split('\n'):
            if "tok/s" in line:
                try:
                    # Extract tokens per second value
                    tokens_per_second = float(line.split("tok/s")[0].strip().split()[-1])
                except (ValueError, IndexError):
                    pass
        
        # Calculate tokens per second if not found in output
        if tokens_per_second is None:
            tokens_per_second = n_tokens / inference_time
        
        return {
            "status": "success",
            "model": model_path,
            "tokens_generated": n_tokens,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "temperature": temp,
            "metal_acceleration": use_metal,
            "output_text": result.stdout.strip(),
            "model_size": get_human_readable_size(get_model_size(model_path))
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during inference: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Inference failed: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        raise