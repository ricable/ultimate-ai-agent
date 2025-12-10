#!/usr/bin/env python3
"""
Common utilities for model quantization across frameworks.
"""

import os
import sys
import json
import hashlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantization.log')
    ]
)
logger = logging.getLogger('quantization_utils')

# Default paths (can be overridden via config)
DEFAULT_LLAMA_CPP_PATH = os.path.expanduser("~/dev/ran/flow2/llama.cpp-setup")
DEFAULT_MLX_PATH = os.path.expanduser("~/dev/ran/flow2/mlx-setup")
DEFAULT_MODELS_DIR = "models"

# Quantization methods map
LLAMACPP_QUANT_METHODS = {
    "Q2_K": {
        "description": "2-bit quantization with K-means",
        "size_reduction": "~87.5%", 
        "quality_impact": "Very high",
        "memory_usage": "Very low",
        "recommended_for": "Ultra-memory-constrained environments, where some quality loss is acceptable"
    },
    "Q3_K": {
        "description": "3-bit quantization with K-means",
        "size_reduction": "~81.25%", 
        "quality_impact": "High",
        "memory_usage": "Very low",
        "recommended_for": "Highly memory-constrained environments, where quality is somewhat important"
    },
    "Q4_0": {
        "description": "4-bit integer quantization",
        "size_reduction": "~75%", 
        "quality_impact": "Moderate-high",
        "memory_usage": "Low",
        "recommended_for": "Memory-constrained environments, with less concern for quality"
    },
    "Q4_K": {
        "description": "4-bit quantization with K-means",
        "size_reduction": "~75%", 
        "quality_impact": "Moderate",
        "memory_usage": "Low",
        "recommended_for": "Memory-constrained environments, balanced for quality and size"
    },
    "Q5_0": {
        "description": "5-bit integer quantization",
        "size_reduction": "~68.75%", 
        "quality_impact": "Low-moderate",
        "memory_usage": "Medium-low",
        "recommended_for": "Balanced option for moderately memory-constrained environments"
    },
    "Q5_K": {
        "description": "5-bit quantization with K-means",
        "size_reduction": "~68.75%", 
        "quality_impact": "Low",
        "memory_usage": "Medium-low",
        "recommended_for": "Good quality with significant memory savings"
    },
    "Q6_K": {
        "description": "6-bit quantization with K-means",
        "size_reduction": "~62.5%", 
        "quality_impact": "Very low",
        "memory_usage": "Medium",
        "recommended_for": "High quality with good memory savings"
    },
    "Q8_0": {
        "description": "8-bit integer quantization",
        "size_reduction": "~50%", 
        "quality_impact": "Minimal",
        "memory_usage": "Medium-high",
        "recommended_for": "Best quality with meaningful memory savings"
    },
    "F16": {
        "description": "16-bit floating point (no quantization)",
        "size_reduction": "0%", 
        "quality_impact": "None (reference)",
        "memory_usage": "High",
        "recommended_for": "Maximum quality, when memory is not a constraint"
    }
}

MLX_QUANT_METHODS = {
    "INT4": {
        "description": "4-bit integer quantization",
        "size_reduction": "~75%", 
        "quality_impact": "Moderate",
        "memory_usage": "Low",
        "recommended_for": "Memory-constrained environments, balanced for quality and size"
    },
    "INT8": {
        "description": "8-bit integer quantization",
        "size_reduction": "~50%", 
        "quality_impact": "Minimal",
        "memory_usage": "Medium",
        "recommended_for": "Best quality with meaningful memory savings"
    },
    "F16": {
        "description": "16-bit floating point (no quantization)",
        "size_reduction": "0%", 
        "quality_impact": "None (reference)",
        "memory_usage": "High",
        "recommended_for": "Maximum quality, when memory is not a constraint"
    },
    "BF16": {
        "description": "Brain floating point format",
        "size_reduction": "0%", 
        "quality_impact": "Negligible",
        "memory_usage": "High",
        "recommended_for": "Training and fine-tuning scenarios"
    }
}

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

def get_model_size(model_path: str) -> int:
    """Get the size of a model file or directory in bytes."""
    if os.path.isfile(model_path):
        return os.path.getsize(model_path)
    elif os.path.isdir(model_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
    else:
        logger.error(f"Path not found: {model_path}")
        return 0

def get_human_readable_size(size_in_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} PB"

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

def get_apple_silicon_model() -> str:
    """Get the Apple Silicon model if running on a Mac."""
    if sys.platform != "darwin":
        return "Non-Mac device"
    
    try:
        output = subprocess.check_output(["sysctl", "machdep.cpu.brand_string"]).decode().strip()
        cpu_brand = output.split(": ")[1]
        
        if "Apple M1" in cpu_brand:
            return "Apple M1"
        elif "Apple M1 Pro" in cpu_brand:
            return "Apple M1 Pro"
        elif "Apple M1 Max" in cpu_brand:
            return "Apple M1 Max"
        elif "Apple M1 Ultra" in cpu_brand:
            return "Apple M1 Ultra"
        elif "Apple M2" in cpu_brand:
            return "Apple M2"
        elif "Apple M2 Pro" in cpu_brand:
            return "Apple M2 Pro"
        elif "Apple M2 Max" in cpu_brand:
            return "Apple M2 Max"
        elif "Apple M2 Ultra" in cpu_brand:
            return "Apple M2 Ultra"
        elif "Apple M3" in cpu_brand:
            return "Apple M3"
        elif "Apple M3 Pro" in cpu_brand:
            return "Apple M3 Pro"
        elif "Apple M3 Max" in cpu_brand:
            return "Apple M3 Max"
        elif "Apple M3 Ultra" in cpu_brand:
            return "Apple M3 Ultra"
        else:
            return f"Unknown Apple Silicon: {cpu_brand}"
    except:
        return "Unknown Mac model"

def get_recommended_quantization(model_size_billions: float, system_ram_gb: int, 
                                 framework: str) -> List[str]:
    """Get recommended quantization methods based on model size and available RAM."""
    recommendations = []
    
    if framework.lower() == "llamacpp":
        # llama.cpp recommendations
        if system_ram_gb <= 8:
            # Very constrained memory
            if model_size_billions <= 7:
                recommendations = ["Q4_K", "Q3_K"]
            else:
                recommendations = ["Q2_K"]
        elif system_ram_gb <= 16:
            # Constrained memory
            if model_size_billions <= 7:
                recommendations = ["Q5_K", "Q4_K"]
            elif model_size_billions <= 13:
                recommendations = ["Q4_K", "Q3_K"]
            else:
                recommendations = ["Q2_K", "Q3_K"]
        elif system_ram_gb <= 32:
            # Moderate memory
            if model_size_billions <= 7:
                recommendations = ["Q8_0", "Q5_K"]
            elif model_size_billions <= 13:
                recommendations = ["Q5_K", "Q4_K"]
            elif model_size_billions <= 33:
                recommendations = ["Q4_K"]
            else:
                recommendations = ["Q3_K", "Q2_K"]
        else:
            # Plenty of memory
            if model_size_billions <= 13:
                recommendations = ["F16", "Q8_0"]
            elif model_size_billions <= 33:
                recommendations = ["Q8_0", "Q5_K"]
            elif model_size_billions <= 70:
                recommendations = ["Q4_K", "Q5_K"]
            else:
                recommendations = ["Q4_K", "Q3_K"]
    
    elif framework.lower() == "mlx":
        # MLX recommendations
        if system_ram_gb <= 8:
            # Very constrained memory
            if model_size_billions <= 7:
                recommendations = ["INT4"]
            else:
                recommendations = ["INT4"]  # Most aggressive option
        elif system_ram_gb <= 16:
            # Constrained memory
            if model_size_billions <= 7:
                recommendations = ["INT8", "INT4"]
            elif model_size_billions <= 13:
                recommendations = ["INT4"]
            else:
                recommendations = ["INT4"]
        elif system_ram_gb <= 32:
            # Moderate memory
            if model_size_billions <= 7:
                recommendations = ["F16", "INT8"]
            elif model_size_billions <= 13:
                recommendations = ["INT8"]
            elif model_size_billions <= 33:
                recommendations = ["INT4"]
            else:
                recommendations = ["INT4"]
        else:
            # Plenty of memory
            if model_size_billions <= 13:
                recommendations = ["F16"]
            elif model_size_billions <= 33:
                recommendations = ["INT8", "F16"]
            elif model_size_billions <= 70:
                recommendations = ["INT8", "INT4"]
            else:
                recommendations = ["INT4"]
    
    return recommendations

def measure_inference_time(func, *args, **kwargs) -> Tuple[float, any]:
    """
    Measure the inference time of a function.
    Returns (elapsed_time_seconds, function_result)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return elapsed_time, result

def save_benchmark_results(results: Dict, output_file: str):
    """Save benchmark results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_benchmark_results(input_file: str) -> Dict:
    """Load benchmark results from a JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)

def format_quantization_info(framework: str, method: str) -> Dict:
    """Format information about a quantization method."""
    if framework.lower() == "llamacpp":
        method = method.upper()  # Normalize method name
        if method in LLAMACPP_QUANT_METHODS:
            return LLAMACPP_QUANT_METHODS[method]
        else:
            return {
                "description": "Unknown quantization method",
                "size_reduction": "Unknown",
                "quality_impact": "Unknown",
                "memory_usage": "Unknown",
                "recommended_for": "Unknown"
            }
    elif framework.lower() == "mlx":
        method = method.upper()  # Normalize method name
        if method in MLX_QUANT_METHODS:
            return MLX_QUANT_METHODS[method]
        else:
            return {
                "description": "Unknown quantization method",
                "size_reduction": "Unknown",
                "quality_impact": "Unknown",
                "memory_usage": "Unknown",
                "recommended_for": "Unknown"
            }
    else:
        return {
            "description": "Unknown framework",
            "size_reduction": "Unknown",
            "quality_impact": "Unknown",
            "memory_usage": "Unknown",
            "recommended_for": "Unknown"
        }