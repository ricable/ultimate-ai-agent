#!/usr/bin/env python3
"""
Model Manager - Unified utilities for managing LLM models for llama.cpp and MLX

This module provides a comprehensive set of tools for downloading, converting,
verifying, and managing models for both llama.cpp and MLX frameworks on Apple Silicon.
"""

import os
import sys
import json
import hashlib
import requests
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_manager.log')
    ]
)
logger = logging.getLogger('model_manager')

# Default paths
DEFAULT_LLAMA_CPP_PATH = os.path.expanduser("~/dev/ran/flow2/llama.cpp-setup")
DEFAULT_MLX_PATH = os.path.expanduser("~/dev/ran/flow2/mlx-setup")
DEFAULT_MODELS_DIR = "models"

# Model registry - maps friendly names to Hugging Face repos and model files
MODEL_REGISTRY = {
    # llama.cpp compatible models (GGUF format)
    "llama2-7b-gguf": {
        "repo": "TheBloke/Llama-2-7B-GGUF",
        "files": {
            "q4_k_m": "llama-2-7b.Q4_K_M.gguf",
            "q5_k_m": "llama-2-7b.Q5_K_M.gguf",
            "q8_0": "llama-2-7b.Q8_0.gguf",
        },
        "framework": "llama.cpp",
        "description": "Meta's Llama 2 7B model in GGUF format with various quantization levels",
        "license": "llama2",
        "papers": ["https://arxiv.org/abs/2307.09288"],
        "sha256": {
            "q4_k_m": None,  # To be populated by verification
            "q5_k_m": None,
            "q8_0": None,
        }
    },
    "llama2-7b-chat-gguf": {
        "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "files": {
            "q4_k_m": "llama-2-7b-chat.Q4_K_M.gguf",
            "q5_k_m": "llama-2-7b-chat.Q5_K_M.gguf",
            "q8_0": "llama-2-7b-chat.Q8_0.gguf",
        },
        "framework": "llama.cpp",
        "description": "Meta's Llama 2 7B Chat model in GGUF format with various quantization levels",
        "license": "llama2",
        "papers": ["https://arxiv.org/abs/2307.09288"],
        "sha256": {
            "q4_k_m": None,
            "q5_k_m": None,
            "q8_0": None,
        }
    },
    "llama2-13b-gguf": {
        "repo": "TheBloke/Llama-2-13B-GGUF",
        "files": {
            "q4_k_m": "llama-2-13b.Q4_K_M.gguf",
            "q5_k_m": "llama-2-13b.Q5_K_M.gguf",
            "q8_0": "llama-2-13b.Q8_0.gguf",
        },
        "framework": "llama.cpp",
        "description": "Meta's Llama 2 13B model in GGUF format with various quantization levels",
        "license": "llama2",
        "papers": ["https://arxiv.org/abs/2307.09288"],
        "sha256": {
            "q4_k_m": None,
            "q5_k_m": None,
            "q8_0": None,
        }
    },
    "mistral-7b-gguf": {
        "repo": "TheBloke/Mistral-7B-v0.1-GGUF",
        "files": {
            "q4_k_m": "mistral-7b-v0.1.Q4_K_M.gguf",
            "q5_k_m": "mistral-7b-v0.1.Q5_K_M.gguf",
            "q8_0": "mistral-7b-v0.1.Q8_0.gguf",
        },
        "framework": "llama.cpp",
        "description": "Mistral AI's 7B model in GGUF format with various quantization levels",
        "license": "apache-2.0",
        "papers": ["https://arxiv.org/abs/2310.06825"],
        "sha256": {
            "q4_k_m": None,
            "q5_k_m": None,
            "q8_0": None,
        }
    },
    "mistral-7b-instruct-gguf": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "files": {
            "q4_k_m": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "q5_k_m": "mistral-7b-instruct-v0.1.Q5_K_M.gguf",
            "q8_0": "mistral-7b-instruct-v0.1.Q8_0.gguf",
        },
        "framework": "llama.cpp",
        "description": "Mistral AI's 7B Instruct model in GGUF format with various quantization levels",
        "license": "apache-2.0",
        "papers": ["https://arxiv.org/abs/2310.06825"],
        "sha256": {
            "q4_k_m": None,
            "q5_k_m": None,
            "q8_0": None,
        }
    },
    "phi-2-gguf": {
        "repo": "TheBloke/phi-2-GGUF",
        "files": {
            "q4_k_m": "phi-2.Q4_K_M.gguf",
            "q5_k_m": "phi-2.Q5_K_M.gguf",
            "q8_0": "phi-2.Q8_0.gguf",
        },
        "framework": "llama.cpp",
        "description": "Microsoft's Phi-2 2.7B model in GGUF format with various quantization levels",
        "license": "mit",
        "papers": ["https://arxiv.org/abs/2306.11644"],
        "sha256": {
            "q4_k_m": None,
            "q5_k_m": None,
            "q8_0": None,
        }
    },
    
    # MLX compatible models
    "llama2-7b-mlx": {
        "repo": "mlx-community/Llama-2-7B-mlx",
        "framework": "mlx",
        "description": "Meta's Llama 2 7B model converted for MLX",
        "license": "llama2",
        "papers": ["https://arxiv.org/abs/2307.09288"],
        "mlx_name": "llama-2-7b",
        "sha256": None
    },
    "llama2-7b-chat-mlx": {
        "repo": "mlx-community/Llama-2-7B-Chat-mlx",
        "framework": "mlx",
        "description": "Meta's Llama 2 7B Chat model converted for MLX",
        "license": "llama2",
        "papers": ["https://arxiv.org/abs/2307.09288"],
        "mlx_name": "llama-2-7b-chat",
        "sha256": None
    },
    "mistral-7b-mlx": {
        "repo": "mlx-community/Mistral-7B-v0.1-mlx",
        "framework": "mlx",
        "description": "Mistral AI's 7B model converted for MLX",
        "license": "apache-2.0",
        "papers": ["https://arxiv.org/abs/2310.06825"],
        "mlx_name": "mistral-7b-v0.1",
        "sha256": None
    },
    "mistral-7b-instruct-mlx": {
        "repo": "mlx-community/Mistral-7B-Instruct-v0.1-mlx",
        "framework": "mlx",
        "description": "Mistral AI's 7B Instruct model converted for MLX",
        "license": "apache-2.0",
        "papers": ["https://arxiv.org/abs/2310.06825"],
        "mlx_name": "mistral-7b-instruct-v0.1",
        "sha256": None
    },
    "phi-2-mlx": {
        "repo": "mlx-community/phi-2-mlx",
        "framework": "mlx",
        "description": "Microsoft's Phi-2 2.7B model converted for MLX",
        "license": "mit",
        "papers": ["https://arxiv.org/abs/2306.11644"],
        "mlx_name": "phi-2",
        "sha256": None
    },
    "gemma-2b-mlx": {
        "repo": "mlx-community/gemma-2b-mlx",
        "framework": "mlx",
        "description": "Google's Gemma 2B model converted for MLX",
        "license": "gemma",
        "papers": ["https://blog.google/technology/developers/gemma-open-models/"],
        "mlx_name": "gemma-2b",
        "sha256": None
    },
    "gemma-7b-mlx": {
        "repo": "mlx-community/gemma-7b-mlx",
        "framework": "mlx",
        "description": "Google's Gemma 7B model converted for MLX",
        "license": "gemma",
        "papers": ["https://blog.google/technology/developers/gemma-open-models/"],
        "mlx_name": "gemma-7b",
        "sha256": None
    },
}

# License information
LICENSE_INFO = {
    "llama2": {
        "name": "Llama 2 Community License",
        "url": "https://ai.meta.com/llama/license/",
        "restrictions": "Non-commercial use only for models >700M parameters unless you get approval from Meta.",
        "attribution": "Required",
        "registration": "Required through Hugging Face"
    },
    "apache-2.0": {
        "name": "Apache License 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
        "restrictions": "None for commercial or non-commercial use",
        "attribution": "Required",
        "registration": "Not required"
    },
    "mit": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
        "restrictions": "None for commercial or non-commercial use",
        "attribution": "Required",
        "registration": "Not required"
    },
    "gemma": {
        "name": "Gemma license",
        "url": "https://ai.google.dev/gemma/terms",
        "restrictions": "Commercial use allowed with some restrictions",
        "attribution": "Required",
        "registration": "Required through Kaggle or Hugging Face"
    }
}

def check_huggingface_cli():
    """Check if the Hugging Face CLI is installed and user is logged in."""
    try:
        # Check if huggingface_hub is installed
        subprocess.run(
            [sys.executable, "-c", "import huggingface_hub"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Check if user is logged in
        result = subprocess.run(
            ["huggingface-cli", "whoami"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if "not logged in" in result.stderr.lower():
            logger.warning("You are not logged in to Hugging Face. Some models may not be accessible.")
            logger.info("To log in, run: huggingface-cli login")
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Hugging Face CLI not found. Installing huggingface_hub...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface_hub"],
                check=True
            )
            logger.info("Hugging Face CLI installed. Please log in with: huggingface-cli login")
            return False
        except subprocess.CalledProcessError:
            logger.error("Failed to install huggingface_hub.")
            return False

def calculate_file_hash(file_path: str, algorithm="sha256") -> str:
    """Calculate the hash of a file."""
    hash_func = getattr(hashlib, algorithm)()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def verify_model_integrity(file_path: str, expected_hash: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verify the integrity of a model file by checking its hash.
    Returns (is_valid, actual_hash)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False, ""
    
    actual_hash = calculate_file_hash(file_path)
    
    if expected_hash is None:
        logger.warning(f"No expected hash provided for {file_path}. Calculated hash: {actual_hash}")
        return True, actual_hash
    
    if actual_hash == expected_hash:
        logger.info(f"Hash verification successful for {file_path}")
        return True, actual_hash
    else:
        logger.error(f"Hash verification failed for {file_path}")
        logger.error(f"Expected: {expected_hash}")
        logger.error(f"Actual:   {actual_hash}")
        return False, actual_hash

def download_file_with_progress(url: str, dest_path: str):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_model_huggingface(repo_id: str, filename: str, output_dir: str) -> str:
    """
    Download a model file from Hugging Face.
    Returns the path to the downloaded file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Check if file already exists
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path
    
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    try:
        logger.info(f"Downloading {filename} from {repo_id}...")
        download_file_with_progress(url, output_path)
        logger.info(f"Downloaded {filename} to {output_path}")
        return output_path
    except requests.RequestException as e:
        logger.error(f"Error downloading {filename} from {repo_id}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def download_llama_cpp_model(model_name: str, quant: str = "q4_k_m", output_dir: Optional[str] = None) -> str:
    """
    Download a model for llama.cpp.
    Returns the path to the downloaded model.
    """
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_LLAMA_CPP_PATH, DEFAULT_MODELS_DIR)
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(get_available_models('llama.cpp'))}")
    
    model_info = MODEL_REGISTRY[model_name]
    if model_info["framework"] != "llama.cpp":
        raise ValueError(f"Model {model_name} is not compatible with llama.cpp")
    
    if quant not in model_info["files"]:
        raise ValueError(f"Unknown quantization level: {quant}. Available: {', '.join(model_info['files'].keys())}")
    
    filename = model_info["files"][quant]
    repo_id = model_info["repo"]
    
    # Download the model
    try:
        output_path = download_model_huggingface(repo_id, filename, output_dir)
        
        # Verify the model if hash is available
        expected_hash = model_info["sha256"][quant]
        if expected_hash:
            is_valid, actual_hash = verify_model_integrity(output_path, expected_hash)
            if not is_valid:
                raise ValueError(f"Model integrity check failed. The file may be corrupted or incomplete.")
        else:
            # If no hash is available, calculate and save it
            _, actual_hash = verify_model_integrity(output_path)
            MODEL_REGISTRY[model_name]["sha256"][quant] = actual_hash
            save_model_registry()
        
        return output_path
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        raise

def download_mlx_model(model_name: str, output_dir: Optional[str] = None) -> str:
    """
    Download a model for MLX using mlx_lm.download.
    Returns the path to the downloaded model directory.
    """
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_MLX_PATH, DEFAULT_MODELS_DIR)
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(get_available_models('mlx'))}")
    
    model_info = MODEL_REGISTRY[model_name]
    if model_info["framework"] != "mlx":
        raise ValueError(f"Model {model_name} is not compatible with MLX")
    
    mlx_name = model_info["mlx_name"]
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Change to output directory
        original_dir = os.getcwd()
        os.chdir(output_dir)
        
        try:
            # Download the model using mlx_lm.download
            logger.info(f"Downloading MLX model: {mlx_name}")
            subprocess.run(
                [sys.executable, "-m", "mlx_lm.download", "--model", mlx_name],
                check=True
            )
            
            # The model is downloaded to a directory named after the model
            model_dir = os.path.join(output_dir, mlx_name)
            
            # Verify model directory exists
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
                
            logger.info(f"Downloaded MLX model to {model_dir}")
            return model_dir
        finally:
            # Change back to original directory
            os.chdir(original_dir)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading MLX model {mlx_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during MLX model download: {e}")
        raise

def convert_to_mlx(hf_model_id: str, output_dir: Optional[str] = None) -> str:
    """
    Convert a Hugging Face model to MLX format.
    Returns the path to the converted model directory.
    """
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_MLX_PATH, DEFAULT_MODELS_DIR)
    
    # Extract model name from Hugging Face ID
    model_name = hf_model_id.split('/')[-1].lower()
    output_path = os.path.join(output_dir, model_name)
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert the model
        logger.info(f"Converting {hf_model_id} to MLX format...")
        subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.convert", 
                "--hf-path", hf_model_id, 
                "--mlx-path", output_path
            ],
            check=True
        )
        
        logger.info(f"Converted model saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting model {hf_model_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model conversion: {e}")
        raise

def quantize_mlx_model(model_dir: str, quantization: str = "int4", output_dir: Optional[str] = None) -> str:
    """
    Quantize an MLX model to the specified precision.
    Returns the path to the quantized model directory.
    """
    if quantization not in ["int4", "int8"]:
        raise ValueError(f"Unsupported quantization: {quantization}. Use 'int4' or 'int8'.")
    
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if output_dir is None:
        # Create a directory next to the original model
        model_basename = os.path.basename(model_dir)
        parent_dir = os.path.dirname(model_dir)
        output_dir = os.path.join(parent_dir, f"{model_basename}_{quantization}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a Python script for quantization
    script_content = f"""
import os
import shutil
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, save
from mlx_lm.utils import get_model_path

# Load model
model_path = "{model_dir}"
model, tokenizer = load(model_path)

# Quantize model
print(f"Quantizing model to {quantization}...")
from mlx_lm.quantize import quantize_model
nbits = {4 if quantization == 'int4' else 8}
model = quantize_model(model, nbits=nbits, group_size=64)

# Save quantized model
output_path = "{output_dir}"
save(output_path, model, tokenizer)

# Copy other necessary files
for file in Path(model_path).glob("*.json"):
    if not (Path(output_path) / file.name).exists():
        shutil.copy(file, output_path)

print(f"Quantized model saved to {output_path}")
"""
    
    script_path = os.path.join(os.path.dirname(model_dir), "quantize_temp.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    try:
        # Run the quantization script
        logger.info(f"Quantizing model {model_dir} to {quantization}...")
        subprocess.run([sys.executable, script_path], check=True)
        logger.info(f"Quantized model saved to {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Error quantizing model: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)

def convert_to_gguf(hf_model_id: str, output_file: str, quant: str = "q4_k") -> str:
    """
    Convert a Hugging Face model to GGUF format for llama.cpp.
    Returns the path to the converted model file.
    """
    try:
        # Check if llama.cpp repo exists
        llama_cpp_dir = DEFAULT_LLAMA_CPP_PATH
        convert_script = os.path.join(llama_cpp_dir, "convert.py")
        
        if not os.path.exists(convert_script):
            raise FileNotFoundError(f"convert.py not found at {convert_script}. Please make sure llama.cpp is properly set up.")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert the model
        logger.info(f"Converting {hf_model_id} to GGUF format with {quant} quantization...")
        subprocess.run(
            [
                sys.executable, convert_script,
                "--outtype", quant,
                "--outfile", output_file,
                hf_model_id
            ],
            check=True
        )
        
        logger.info(f"Converted model saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting model {hf_model_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model conversion: {e}")
        raise

def quantize_gguf_model(input_file: str, output_file: str, quant: str = "q4_k") -> str:
    """
    Quantize a GGUF model for llama.cpp.
    Returns the path to the quantized model file.
    """
    try:
        # Check if llama.cpp repo exists
        llama_cpp_dir = DEFAULT_LLAMA_CPP_PATH
        quantize_script = os.path.join(llama_cpp_dir, "quantize.py")
        
        if not os.path.exists(quantize_script):
            raise FileNotFoundError(f"quantize.py not found at {quantize_script}. Please make sure llama.cpp is properly set up.")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Quantize the model
        logger.info(f"Quantizing {input_file} to {quant}...")
        subprocess.run(
            [
                sys.executable, quantize_script,
                "--model", input_file,
                "--outfile", output_file,
                "--type", quant
            ],
            check=True
        )
        
        logger.info(f"Quantized model saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error quantizing model: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model quantization: {e}")
        raise

def save_model_registry():
    """Save the model registry to a JSON file."""
    registry_path = os.path.join(os.path.dirname(__file__), "model_registry.json")
    with open(registry_path, "w") as f:
        json.dump(MODEL_REGISTRY, f, indent=2)

def load_model_registry():
    """Load the model registry from a JSON file if it exists."""
    global MODEL_REGISTRY
    registry_path = os.path.join(os.path.dirname(__file__), "model_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            MODEL_REGISTRY = json.load(f)

def get_available_models(framework: Optional[str] = None) -> List[str]:
    """Get a list of available models, optionally filtered by framework."""
    if framework is None:
        return list(MODEL_REGISTRY.keys())
    else:
        return [name for name, info in MODEL_REGISTRY.items() 
                if info["framework"] == framework]

def get_model_info(model_name: str) -> Dict:
    """Get information about a model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_REGISTRY[model_name].copy()
    
    # Add license information
    license_key = model_info.get("license")
    if license_key and license_key in LICENSE_INFO:
        model_info["license_info"] = LICENSE_INFO[license_key]
    
    return model_info

def list_installed_models(base_dir: Optional[str] = None, framework: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all installed models.
    Returns a dictionary mapping framework to a list of installed models.
    """
    installed_models = {
        "llama.cpp": [],
        "mlx": []
    }
    
    # Check llama.cpp models
    if framework is None or framework == "llama.cpp":
        llama_cpp_models_dir = os.path.join(DEFAULT_LLAMA_CPP_PATH, DEFAULT_MODELS_DIR) if base_dir is None else base_dir
        if os.path.exists(llama_cpp_models_dir):
            for file in os.listdir(llama_cpp_models_dir):
                if file.endswith(".gguf"):
                    installed_models["llama.cpp"].append(file)
    
    # Check MLX models
    if framework is None or framework == "mlx":
        mlx_models_dir = os.path.join(DEFAULT_MLX_PATH, DEFAULT_MODELS_DIR) if base_dir is None else base_dir
        if os.path.exists(mlx_models_dir):
            for dir_name in os.listdir(mlx_models_dir):
                dir_path = os.path.join(mlx_models_dir, dir_name)
                if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "weights.safetensors")):
                    installed_models["mlx"].append(dir_name)
    
    return installed_models

def get_model_license_info(model_name: str) -> Dict:
    """Get license information for a model."""
    model_info = get_model_info(model_name)
    license_key = model_info.get("license")
    
    if license_key and license_key in LICENSE_INFO:
        return LICENSE_INFO[license_key]
    else:
        return {"name": "Unknown", "url": "", "restrictions": "Unknown", "attribution": "Unknown"}

def get_recommended_models(system_ram_gb: int) -> Dict[str, List[str]]:
    """Get recommended models based on system RAM."""
    recommended = {
        "llama.cpp": [],
        "mlx": []
    }
    
    if system_ram_gb <= 8:
        # 8GB RAM or less
        recommended["llama.cpp"] = ["phi-2-gguf (q4_k_m)"]
        recommended["mlx"] = ["phi-2-mlx (int4)", "gemma-2b-mlx (int4)"]
    elif system_ram_gb <= 16:
        # 16GB RAM
        recommended["llama.cpp"] = ["llama2-7b-gguf (q4_k_m)", "mistral-7b-gguf (q4_k_m)"]
        recommended["mlx"] = ["llama2-7b-mlx (int8)", "mistral-7b-mlx (int8)", "phi-2-mlx (fp16)"]
    elif system_ram_gb <= 32:
        # 32GB RAM
        recommended["llama.cpp"] = ["llama2-7b-gguf (q8_0)", "llama2-13b-gguf (q4_k_m)"]
        recommended["mlx"] = ["llama2-7b-mlx (fp16)", "llama2-13b-mlx (int8)", "mistral-7b-mlx (fp16)"]
    else:
        # 64GB+ RAM
        recommended["llama.cpp"] = ["llama2-7b-gguf (q8_0)", "llama2-13b-gguf (q8_0)"]
        recommended["mlx"] = ["llama2-7b-mlx (fp16)", "llama2-13b-mlx (fp16)", "mistral-7b-mlx (fp16)"]
    
    return recommended

def get_system_ram_gb() -> int:
    """Get system RAM in GB."""
    try:
        if sys.platform == "darwin":  # macOS
            output = subprocess.check_output(["sysctl", "hw.memsize"]).decode().strip()
            mem_bytes = int(output.split()[1])
            return mem_bytes // (1024 ** 3)  # Convert to GB
        elif sys.platform == "linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        return mem_kb // (1024 ** 1)  # Convert KB to GB
        elif sys.platform == "win32":  # Windows
            output = subprocess.check_output(["wmic", "computersystem", "get", "totalphysicalmemory"]).decode()
            mem_bytes = int(output.split("\n")[1])
            return mem_bytes // (1024 ** 3)  # Convert to GB
    except:
        logger.warning("Could not determine system RAM. Assuming 16GB.")
        return 16
    
    logger.warning("Could not determine system RAM. Assuming 16GB.")
    return 16

def main():
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser(description="Model management utilities for llama.cpp and MLX")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available or installed models")
    list_parser.add_argument("--installed", action="store_true", help="List installed models instead of available ones")
    list_parser.add_argument("--framework", choices=["llama.cpp", "mlx"], help="Filter by framework")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model", help="Model name to download")
    download_parser.add_argument("--quant", default="q4_k_m", help="Quantization level for llama.cpp models")
    download_parser.add_argument("--output-dir", help="Output directory")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert a model between formats")
    convert_parser.add_argument("--hf", required=True, help="Hugging Face model ID")
    convert_parser.add_argument("--framework", choices=["llama.cpp", "mlx"], required=True, help="Target framework")
    convert_parser.add_argument("--output", required=True, help="Output file or directory")
    convert_parser.add_argument("--quant", default="q4_k", help="Quantization level for llama.cpp models")
    
    # Quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Quantize a model")
    quantize_parser.add_argument("--input", required=True, help="Input model file or directory")
    quantize_parser.add_argument("--output", required=True, help="Output file or directory")
    quantize_parser.add_argument("--framework", choices=["llama.cpp", "mlx"], required=True, help="Framework")
    quantize_parser.add_argument("--quant", default="q4_k", help="Quantization level (q4_k, q8_0 for llama.cpp; int4, int8 for MLX)")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify model integrity")
    verify_parser.add_argument("--model-path", required=True, help="Path to model file or directory")
    verify_parser.add_argument("--expected-hash", help="Expected hash (optional)")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about a model")
    info_parser.add_argument("model", help="Model name")
    
    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Get model recommendations based on system specs")
    recommend_parser.add_argument("--ram", type=int, help="System RAM in GB (detected automatically if not specified)")
    
    # License command
    license_parser = subparsers.add_parser("license", help="Get license information for a model")
    license_parser.add_argument("model", help="Model name")
    
    args = parser.parse_args()
    
    # Load model registry
    load_model_registry()
    
    # Execute command
    if args.command == "list":
        if args.installed:
            installed = list_installed_models(framework=args.framework)
            for fw, models in installed.items():
                if args.framework is None or fw == args.framework:
                    print(f"{fw} models:")
                    for model in models:
                        print(f"  - {model}")
        else:
            models = get_available_models(args.framework)
            if args.framework:
                print(f"Available {args.framework} models:")
            else:
                print("Available models:")
            for model in models:
                info = MODEL_REGISTRY[model]
                framework = info["framework"]
                description = info.get("description", "")
                print(f"  - {model} ({framework}): {description}")
    
    elif args.command == "download":
        try:
            check_huggingface_cli()
            model_info = get_model_info(args.model)
            framework = model_info["framework"]
            
            if framework == "llama.cpp":
                path = download_llama_cpp_model(args.model, args.quant, args.output_dir)
                print(f"Downloaded model to: {path}")
            elif framework == "mlx":
                path = download_mlx_model(args.model, args.output_dir)
                print(f"Downloaded model to: {path}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "convert":
        try:
            check_huggingface_cli()
            if args.framework == "llama.cpp":
                path = convert_to_gguf(args.hf, args.output, args.quant)
                print(f"Converted model to: {path}")
            elif args.framework == "mlx":
                path = convert_to_mlx(args.hf, args.output)
                print(f"Converted model to: {path}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "quantize":
        try:
            if args.framework == "llama.cpp":
                path = quantize_gguf_model(args.input, args.output, args.quant)
                print(f"Quantized model to: {path}")
            elif args.framework == "mlx":
                path = quantize_mlx_model(args.input, args.quant, args.output)
                print(f"Quantized model to: {path}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "verify":
        try:
            is_valid, actual_hash = verify_model_integrity(args.model_path, args.expected_hash)
            if is_valid:
                print(f"Model integrity verified: {args.model_path}")
                print(f"SHA-256: {actual_hash}")
            else:
                print(f"Model integrity verification failed: {args.model_path}")
                if args.expected_hash:
                    print(f"Expected: {args.expected_hash}")
                    print(f"Actual:   {actual_hash}")
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "info":
        try:
            info = get_model_info(args.model)
            print(f"Model: {args.model}")
            print(f"Framework: {info['framework']}")
            print(f"Description: {info.get('description', 'N/A')}")
            print(f"Repository: {info.get('repo', 'N/A')}")
            print(f"License: {info.get('license', 'N/A')}")
            if "papers" in info:
                print("Research Papers:")
                for paper in info["papers"]:
                    print(f"  - {paper}")
            if info["framework"] == "llama.cpp":
                print("Available quantization levels:")
                for quant in info["files"].keys():
                    print(f"  - {quant}")
            if "license_info" in info:
                license_info = info["license_info"]
                print("\nLicense Details:")
                print(f"  Name: {license_info['name']}")
                print(f"  URL: {license_info['url']}")
                print(f"  Restrictions: {license_info['restrictions']}")
                print(f"  Attribution required: {license_info['attribution']}")
                print(f"  Registration required: {license_info['registration']}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "recommend":
        ram_gb = args.ram if args.ram else get_system_ram_gb()
        print(f"System RAM: {ram_gb}GB")
        recommendations = get_recommended_models(ram_gb)
        
        print("\nRecommended models for your system:")
        for framework, models in recommendations.items():
            print(f"\n{framework.upper()}:")
            for model in models:
                print(f"  - {model}")
        
        print("\nTo download a recommended model:")
        if recommendations["llama.cpp"]:
            model_example = recommendations["llama.cpp"][0].split(" ")[0]
            quant_example = recommendations["llama.cpp"][0].split(" ")[1].strip("()")
            print(f"  python model_manager.py download {model_example} --quant {quant_example}")
        
        if recommendations["mlx"]:
            model_example = recommendations["mlx"][0].split(" ")[0]
            print(f"  python model_manager.py download {model_example}")
    
    elif args.command == "license":
        try:
            license_info = get_model_license_info(args.model)
            print(f"License information for {args.model}:")
            print(f"  Name: {license_info['name']}")
            print(f"  URL: {license_info['url']}")
            print(f"  Restrictions: {license_info['restrictions']}")
            print(f"  Attribution required: {license_info['attribution']}")
            print(f"  Registration required: {license_info['registration']}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()