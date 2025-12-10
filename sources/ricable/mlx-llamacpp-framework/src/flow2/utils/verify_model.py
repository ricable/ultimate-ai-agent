#!/usr/bin/env python3
"""
Model Verification Utility

This script verifies the integrity and correctness of downloaded LLM models
for both llama.cpp and MLX frameworks.
"""

import os
import sys
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path to import model_manager
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

try:
    from model_manager import (
        verify_model_integrity,
        DEFAULT_LLAMA_CPP_PATH,
        DEFAULT_MLX_PATH,
        DEFAULT_MODELS_DIR
    )
except ImportError:
    print("Error: model_manager.py not found. Make sure it's in the same directory as this script.")
    sys.exit(1)

def calculate_file_hash(file_path: str, algorithm="sha256") -> str:
    """Calculate the hash of a file."""
    hash_func = getattr(hashlib, algorithm)()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def verify_gguf_model(model_path: str) -> bool:
    """
    Verify a GGUF model file for llama.cpp.
    Checks:
    1. File exists
    2. File has correct extension
    3. File has reasonable size
    4. File passes basic structure check
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    if not model_path.endswith(".gguf"):
        print(f"Warning: Model file does not have .gguf extension: {model_path}")
    
    # Check file size (should be at least 1MB)
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
    if file_size < 1:
        print(f"Error: Model file is too small ({file_size:.2f} MB): {model_path}")
        return False
    
    print(f"Model file size: {file_size:.2f} MB")
    
    # Calculate hash
    file_hash = calculate_file_hash(model_path)
    print(f"Model SHA-256 hash: {file_hash}")
    
    # Try to read basic metadata using llama.cpp's info tool if available
    llama_info = os.path.join(DEFAULT_LLAMA_CPP_PATH, "llama-info")
    if os.path.exists(llama_info):
        try:
            print("\nRunning llama-info to verify model structure...")
            result = subprocess.run(
                [llama_info, model_path],
                capture_output=True,
                text=True,
                check=True
            )
            print("Model structure verified successfully!")
            
            # Print some key info from the output
            for line in result.stdout.splitlines():
                if any(key in line for key in ["model type", "vocab size", "context", "embedding", "parameter"]):
                    print(line.strip())
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error verifying model structure: {e}")
            print(e.stderr)
            return False
    else:
        print("Note: llama-info tool not found. Skipping detailed structure verification.")
        return True

def verify_mlx_model(model_dir: str) -> bool:
    """
    Verify an MLX model directory.
    Checks:
    1. Directory exists
    2. Required files are present
    3. Files have reasonable sizes
    4. Try to load the model with MLX
    """
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return False
    
    # Check required files
    required_files = ["weights.safetensors", "config.json", "tokenizer.json"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: The following required files are missing: {', '.join(missing_files)}")
        return False
    
    # Check file sizes
    weights_path = os.path.join(model_dir, "weights.safetensors")
    weights_size = os.path.getsize(weights_path) / (1024 * 1024)  # Convert to MB
    
    if weights_size < 1:
        print(f"Error: Weights file is too small ({weights_size:.2f} MB)")
        return False
    
    print(f"Weights file size: {weights_size:.2f} MB")
    
    # Calculate hash of weights file
    weights_hash = calculate_file_hash(weights_path)
    print(f"Weights SHA-256 hash: {weights_hash}")
    
    # Try to load the model with MLX
    try:
        # Create a temporary script to load the model
        temp_script = os.path.join(os.path.dirname(model_dir), "verify_temp.py")
        with open(temp_script, "w") as f:
            f.write(f"""
import sys
try:
    from mlx_lm import load
    print("Attempting to load model from {model_dir}...")
    model, tokenizer = load("{model_dir}")
    print("Model loaded successfully!")
    print(f"Model architecture: {{model.__class__.__name__}}")
    print(f"Tokenizer vocabulary size: {{tokenizer.vocab_size}}")
    print("Model verification complete.")
except Exception as e:
    print(f"Error loading model: {{e}}")
    sys.exit(1)
""")
        
        try:
            print("\nVerifying model with MLX...")
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error verifying model with MLX: {e}")
            print(e.stdout)
            print(e.stderr)
            return False
        finally:
            # Clean up
            if os.path.exists(temp_script):
                os.remove(temp_script)
    except Exception as e:
        print(f"Error during verification: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify LLM model files for llama.cpp and MLX")
    parser.add_argument("model_path", help="Path to model file or directory")
    parser.add_argument("--framework", choices=["llama.cpp", "mlx"], help="Framework (detected automatically if not specified)")
    parser.add_argument("--check-hash", action="store_true", help="Calculate and display hash values only")
    args = parser.parse_args()
    
    model_path = args.model_path
    framework = args.framework
    
    # Detect framework if not specified
    if framework is None:
        if os.path.isfile(model_path) and model_path.endswith(".gguf"):
            framework = "llama.cpp"
            print(f"Detected framework: {framework}")
        elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "weights.safetensors")):
            framework = "mlx"
            print(f"Detected framework: {framework}")
        else:
            print("Error: Could not detect framework. Please specify with --framework.")
            sys.exit(1)
    
    # If only checking hash
    if args.check_hash:
        if os.path.isfile(model_path):
            file_hash = calculate_file_hash(model_path)
            print(f"SHA-256 hash for {model_path}: {file_hash}")
        elif os.path.isdir(model_path) and framework == "mlx":
            weights_path = os.path.join(model_path, "weights.safetensors")
            if os.path.exists(weights_path):
                file_hash = calculate_file_hash(weights_path)
                print(f"SHA-256 hash for {weights_path}: {file_hash}")
            else:
                print(f"Error: weights.safetensors not found in {model_path}")
                sys.exit(1)
        else:
            print(f"Error: Cannot calculate hash for {model_path}")
            sys.exit(1)
        sys.exit(0)
    
    # Perform verification
    print(f"Verifying {framework} model: {model_path}")
    print("=" * 60)
    
    if framework == "llama.cpp":
        success = verify_gguf_model(model_path)
    elif framework == "mlx":
        success = verify_mlx_model(model_path)
    else:
        print(f"Error: Unknown framework: {framework}")
        sys.exit(1)
    
    print("=" * 60)
    if success:
        print(f"✅ Model verification PASSED for {model_path}")
        sys.exit(0)
    else:
        print(f"❌ Model verification FAILED for {model_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()