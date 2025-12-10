#!/usr/bin/env python3
"""
Utilities for quantizing models using MLX.
"""

import os
import sys
import logging
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from flow2.utils.utils import (
    get_model_size,
    get_human_readable_size,
    verify_model_integrity,
    measure_inference_time,
    MLX_QUANT_METHODS
)

logger = logging.getLogger('quantization_utils.mlx')

# Default paths
DEFAULT_MLX_PATH = os.path.expanduser("~/dev/ran/flow2/mlx-setup")
DEFAULT_MODELS_DIR = "models"

def get_available_quant_types() -> List[str]:
    """Get a list of available quantization types for MLX."""
    return list(MLX_QUANT_METHODS.keys())

def quantize_model(
    input_dir: str, 
    output_dir: str, 
    quant_type: str = "INT4",
    group_size: int = 64,
    exclude_modules: Optional[List[str]] = None
) -> Dict:
    """
    Quantize an MLX model to the specified precision.
    
    Args:
        input_dir: Path to the input MLX model directory
        output_dir: Path to save the quantized model
        quant_type: Quantization type (e.g., INT4, INT8)
        group_size: Number of weights to group together for quantization
        exclude_modules: List of modules to exclude from quantization
        
    Returns:
        Dict containing information about the quantization process
    """
    # Normalize quantization type
    quant_type = quant_type.upper()
    
    # Validate quantization type
    if quant_type not in MLX_QUANT_METHODS:
        raise ValueError(f"Unsupported quantization type: {quant_type}. "
                         f"Supported types: {', '.join(MLX_QUANT_METHODS.keys())}")
    
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input model directory not found: {input_dir}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original model size
    original_size = get_model_size(input_dir)
    
    # Set bits based on quantization type
    if quant_type == "INT4":
        bits = 4
    elif quant_type == "INT8":
        bits = 8
    elif quant_type in ["F16", "BF16"]:
        # These don't actually need quantization, just copy the model
        logger.info(f"Copying model for {quant_type} precision (no quantization needed)...")
        shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
        
        return {
            "status": "success",
            "input_model": input_dir,
            "output_model": output_dir,
            "quantization_type": quant_type,
            "original_size": original_size,
            "original_size_human": get_human_readable_size(original_size),
            "quantized_size": original_size,
            "quantized_size_human": get_human_readable_size(original_size),
            "size_reduction_percent": "0.00%",
            "quantization_time_seconds": 0,
            "is_valid": True,
            "quantization_info": MLX_QUANT_METHODS.get(quant_type, {})
        }
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")
    
    # Create a temporary Python script for quantization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        script_path = temp_file.name
        
        # Write quantization script
        temp_file.write(f"""
import os
import shutil
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, save
from mlx_lm.utils import get_model_path

# Load model
model_path = "{input_dir}"
print(f"Loading model from {{model_path}}...")
model, tokenizer = load(model_path)

# Quantize model
print(f"Quantizing model to {quant_type.lower()} with group size {group_size}...")
from mlx_lm.quantize import quantize_model

# Set exclude modules
exclude_modules = {exclude_modules if exclude_modules else []}

# Perform quantization
start_time = time.time()
model = quantize_model(
    model, 
    nbits={bits}, 
    group_size={group_size},
    exclude=exclude_modules
)
end_time = time.time()
print(f"Quantization took {{end_time - start_time:.2f}} seconds")

# Save quantized model
output_path = "{output_dir}"
print(f"Saving quantized model to {{output_path}}...")
save(output_path, model, tokenizer)

# Copy other necessary files
for file in Path(model_path).glob("*.json"):
    if not (Path(output_path) / file.name).exists():
        shutil.copy(file, output_path)

print(f"Quantized model saved to {{output_path}}")
print(f"QUANTIZATION_TIME={{end_time - start_time}}")
""")
    
    # Run the quantization script
    logger.info(f"Quantizing model {input_dir} to {quant_type}...")
    start_time = time.time()
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        end_time = time.time()
        quantization_time = end_time - start_time
        
        # Extract quantization time from script output if available
        for line in result.stdout.split("\n"):
            if line.startswith("QUANTIZATION_TIME="):
                try:
                    quantization_time = float(line.split("=")[1])
                except (ValueError, IndexError):
                    pass
        
        # Get quantized model size
        quantized_size = get_model_size(output_dir)
        size_reduction = 1 - (quantized_size / original_size)
        
        return {
            "status": "success",
            "input_model": input_dir,
            "output_model": output_dir,
            "quantization_type": quant_type,
            "group_size": group_size,
            "exclude_modules": exclude_modules,
            "original_size": original_size,
            "original_size_human": get_human_readable_size(original_size),
            "quantized_size": quantized_size,
            "quantized_size_human": get_human_readable_size(quantized_size),
            "size_reduction_percent": f"{size_reduction:.2%}",
            "quantization_time_seconds": quantization_time,
            "is_valid": True,  # No integrity check for MLX models yet
            "quantization_info": MLX_QUANT_METHODS.get(quant_type, {})
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error quantizing model: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Quantization failed: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error during quantization: {e}")
        raise
    
    finally:
        # Clean up
        try:
            os.unlink(script_path)
        except:
            pass

def download_and_quantize_mlx_model(
    model_name: str,
    output_dir: str,
    quant_type: str = "INT4",
    group_size: int = 64,
    exclude_modules: Optional[List[str]] = None
) -> Dict:
    """
    Download an MLX model and quantize it.
    
    Args:
        model_name: MLX model name (e.g., "llama-2-7b")
        output_dir: Path to save the quantized model
        quant_type: Quantization type (e.g., INT4, INT8)
        group_size: Number of weights to group together for quantization
        exclude_modules: List of modules to exclude from quantization
        
    Returns:
        Dict containing information about the download and quantization process
    """
    # Create a temporary directory for the original model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary Python script for downloading and quantization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            script_path = temp_file.name
            
            # Write script to download and quantize
            temp_file.write(f"""
import os
import time
import mlx.core as mx
import shutil
from pathlib import Path
from mlx_lm import load, save
from mlx_lm.download import download_model

# Download model
print(f"Downloading model {model_name}...")
start_download = time.time()
model_path = download_model("{model_name}", "{temp_dir}")
end_download = time.time()
print(f"Download took {{end_download - start_download:.2f}} seconds")

# Load model
print(f"Loading model from {{model_path}}...")
model, tokenizer = load(model_path)

# Quantize model
print(f"Quantizing model to {quant_type.lower()} with group size {group_size}...")
from mlx_lm.quantize import quantize_model

# Set exclude modules
exclude_modules = {exclude_modules if exclude_modules else []}

# Perform quantization
start_quant = time.time()
model = quantize_model(
    model, 
    nbits={4 if quant_type.upper() == 'INT4' else 8}, 
    group_size={group_size},
    exclude=exclude_modules
)
end_quant = time.time()
print(f"Quantization took {{end_quant - start_quant:.2f}} seconds")

# Save quantized model
output_path = "{output_dir}"
print(f"Saving quantized model to {{output_path}}...")
save(output_path, model, tokenizer)

# Copy other necessary files
for file in Path(model_path).glob("*.json"):
    if not (Path(output_path) / file.name).exists():
        shutil.copy(file, output_path)

print(f"Quantized model saved to {{output_path}}")
print(f"DOWNLOAD_TIME={{end_download - start_download}}")
print(f"QUANTIZATION_TIME={{end_quant - start_quant}}")
""")
        
        # Run the script
        logger.info(f"Downloading and quantizing model {model_name} to {quant_type}...")
        start_time = time.time()
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Extract times from script output
            download_time = None
            quantization_time = None
            for line in result.stdout.split("\n"):
                if line.startswith("DOWNLOAD_TIME="):
                    try:
                        download_time = float(line.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("QUANTIZATION_TIME="):
                    try:
                        quantization_time = float(line.split("=")[1])
                    except (ValueError, IndexError):
                        pass
            
            # Use fallbacks if extraction failed
            if download_time is None:
                download_time = total_time / 2  # Rough estimate
            if quantization_time is None:
                quantization_time = total_time / 2  # Rough estimate
            
            # Get quantized model size
            quantized_size = get_model_size(output_dir)
            
            return {
                "status": "success",
                "model_name": model_name,
                "output_model": output_dir,
                "quantization_type": quant_type,
                "group_size": group_size,
                "exclude_modules": exclude_modules,
                "quantized_size": quantized_size,
                "quantized_size_human": get_human_readable_size(quantized_size),
                "download_time_seconds": download_time,
                "quantization_time_seconds": quantization_time,
                "total_time_seconds": total_time,
                "is_valid": True,  # No integrity check for MLX models yet
                "quantization_info": MLX_QUANT_METHODS.get(quant_type, {})
            }
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading and quantizing model: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Download and quantization failed: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error during download and quantization: {e}")
            raise
        
        finally:
            # Clean up
            try:
                os.unlink(script_path)
            except:
                pass

def convert_and_quantize_from_huggingface(
    hf_model_id: str,
    output_dir: str,
    quant_type: str = "INT4",
    group_size: int = 64,
    exclude_modules: Optional[List[str]] = None,
    use_auth_token: Optional[str] = None
) -> Dict:
    """
    Convert a Hugging Face model to MLX format and quantize it.
    
    Args:
        hf_model_id: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b")
        output_dir: Path to save the quantized model
        quant_type: Quantization type (e.g., INT4, INT8)
        group_size: Number of weights to group together for quantization
        exclude_modules: List of modules to exclude from quantization
        use_auth_token: Hugging Face token for accessing gated models
        
    Returns:
        Dict containing information about the conversion and quantization process
    """
    # Create a temporary directory for the original model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary Python script for conversion and quantization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            script_path = temp_file.name
            
            # Write script to convert and quantize
            temp_file.write(f"""
import os
import time
import mlx.core as mx
import shutil
from pathlib import Path
from mlx_lm import save
from mlx_lm.convert import convert_hf_model

# Set HF token if provided
{'os.environ["HF_TOKEN"] = "' + use_auth_token + '"' if use_auth_token else ''}

# Convert model
print(f"Converting HF model {hf_model_id} to MLX format...")
start_convert = time.time()
model, tokenizer = convert_hf_model("{hf_model_id}", "{temp_dir}")
end_convert = time.time()
print(f"Conversion took {{end_convert - start_convert:.2f}} seconds")

# Quantize model
print(f"Quantizing model to {quant_type.lower()} with group size {group_size}...")
from mlx_lm.quantize import quantize_model

# Set exclude modules
exclude_modules = {exclude_modules if exclude_modules else []}

# Perform quantization
start_quant = time.time()
model = quantize_model(
    model, 
    nbits={4 if quant_type.upper() == 'INT4' else 8}, 
    group_size={group_size},
    exclude=exclude_modules
)
end_quant = time.time()
print(f"Quantization took {{end_quant - start_quant:.2f}} seconds")

# Save quantized model
output_path = "{output_dir}"
print(f"Saving quantized model to {{output_path}}...")
save(output_path, model, tokenizer)

print(f"Quantized model saved to {{output_path}}")
print(f"CONVERSION_TIME={{end_convert - start_convert}}")
print(f"QUANTIZATION_TIME={{end_quant - start_quant}}")
""")
        
        # Run the script
        logger.info(f"Converting and quantizing model {hf_model_id} to {quant_type}...")
        start_time = time.time()
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Extract times from script output
            conversion_time = None
            quantization_time = None
            for line in result.stdout.split("\n"):
                if line.startswith("CONVERSION_TIME="):
                    try:
                        conversion_time = float(line.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("QUANTIZATION_TIME="):
                    try:
                        quantization_time = float(line.split("=")[1])
                    except (ValueError, IndexError):
                        pass
            
            # Use fallbacks if extraction failed
            if conversion_time is None:
                conversion_time = total_time / 2  # Rough estimate
            if quantization_time is None:
                quantization_time = total_time / 2  # Rough estimate
            
            # Get quantized model size
            quantized_size = get_model_size(output_dir)
            
            return {
                "status": "success",
                "huggingface_model": hf_model_id,
                "output_model": output_dir,
                "quantization_type": quant_type,
                "group_size": group_size,
                "exclude_modules": exclude_modules,
                "quantized_size": quantized_size,
                "quantized_size_human": get_human_readable_size(quantized_size),
                "conversion_time_seconds": conversion_time,
                "quantization_time_seconds": quantization_time,
                "total_time_seconds": total_time,
                "is_valid": True,  # No integrity check for MLX models yet
                "quantization_info": MLX_QUANT_METHODS.get(quant_type, {})
            }
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting and quantizing model: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Conversion and quantization failed: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error during conversion and quantization: {e}")
            raise
        
        finally:
            # Clean up
            try:
                os.unlink(script_path)
            except:
                pass

def batch_quantize_models(
    input_dirs: List[str],
    output_root_dir: str,
    quant_types: List[str] = ["INT4", "INT8"],
    group_size: int = 64,
    exclude_modules: Optional[List[str]] = None
) -> Dict:
    """
    Batch quantize multiple MLX models to multiple quantization levels.
    
    Args:
        input_dirs: List of paths to input MLX model directories
        output_root_dir: Root directory to save the quantized models
        quant_types: List of quantization types to apply
        group_size: Number of weights to group together for quantization
        exclude_modules: List of modules to exclude from quantization
        
    Returns:
        Dict containing information about all quantization processes
    """
    results = {}
    
    for input_dir in input_dirs:
        model_name = os.path.basename(input_dir)
        model_results = {}
        
        for quant_type in quant_types:
            output_dir = os.path.join(output_root_dir, f"{model_name}_{quant_type.lower()}")
            
            try:
                result = quantize_model(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    quant_type=quant_type,
                    group_size=group_size,
                    exclude_modules=exclude_modules
                )
                model_results[quant_type] = result
            except Exception as e:
                logger.error(f"Failed to quantize {input_dir} to {quant_type}: {e}")
                model_results[quant_type] = {"status": "failed", "error": str(e)}
        
        results[model_name] = model_results
    
    return results

def measure_inference_performance(
    model_dir: str,
    prompt: str = "Once upon a time",
    max_tokens: int = 100,
    temp: float = 0.7,
    quantization: Optional[str] = None
) -> Dict:
    """
    Measure inference performance of an MLX model.
    
    Args:
        model_dir: Path to the MLX model directory
        prompt: Text prompt for inference
        max_tokens: Maximum number of tokens to generate
        temp: Temperature for sampling
        quantization: Optional quantization to apply before measuring (e.g., "int4")
        
    Returns:
        Dict containing performance metrics
    """
    # Check if model directory exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Create a temporary script for measuring performance
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        script_path = temp_file.name
        
        # Write script for performance measurement
        temp_file.write(f"""
import time
import mlx.core as mx
from mlx_lm import load, generate

# Set the default device to GPU for optimal performance
mx.set_default_device(mx.gpu)

# Load model
print("Loading model...")
start_load = time.time()
model, tokenizer = load(
    "{model_dir}"{f', quantization="{quantization}"' if quantization else ''}
)
end_load = time.time()
print(f"Model loading took {{end_load - start_load:.2f}} seconds")

# Warmup run (first run often includes compilation time)
print("Warming up...")
_ = generate(model, tokenizer, "Hello", max_tokens=10)

# Measure inference time
print("Measuring inference performance...")
prompt = "{prompt}"
start_inference = time.time()
tokens = generate(
    model, 
    tokenizer, 
    prompt, 
    max_tokens={max_tokens}, 
    temp={temp}
)
end_inference = time.time()

inference_time = end_inference - start_inference
tokens_per_second = {max_tokens} / inference_time

print(f"Generated {{len(tokens)}} tokens in {{inference_time:.2f}} seconds")
print(f"Tokens per second: {{tokens_per_second:.2f}}")
print(f"Generated text: {{tokenizer.decode(tokens)}}")

print(f"LOAD_TIME={{end_load - start_load}}")
print(f"INFERENCE_TIME={{inference_time}}")
print(f"TOKENS_PER_SECOND={{tokens_per_second}}")
print(f"OUTPUT_TEXT={{tokenizer.decode(tokens)}}")
""")
    
    # Run the script
    logger.info(f"Measuring inference performance for {model_dir}...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Extract metrics from script output
        load_time = None
        inference_time = None
        tokens_per_second = None
        output_text = None
        
        for line in result.stdout.split("\n"):
            if line.startswith("LOAD_TIME="):
                try:
                    load_time = float(line.split("=")[1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("INFERENCE_TIME="):
                try:
                    inference_time = float(line.split("=")[1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("TOKENS_PER_SECOND="):
                try:
                    tokens_per_second = float(line.split("=")[1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("OUTPUT_TEXT="):
                output_text = line[len("OUTPUT_TEXT="):]
        
        return {
            "status": "success",
            "model": model_dir,
            "quantization": quantization,
            "tokens_generated": max_tokens,
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "temperature": temp,
            "output_text": output_text,
            "model_size": get_human_readable_size(get_model_size(model_dir))
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during inference performance measurement: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Inference performance measurement failed: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error during inference performance measurement: {e}")
        raise
    
    finally:
        # Clean up
        try:
            os.unlink(script_path)
        except:
            pass