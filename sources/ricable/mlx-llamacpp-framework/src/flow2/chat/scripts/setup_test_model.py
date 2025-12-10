#!/usr/bin/env python3
"""
Setup Test Model Script

This script downloads and sets up small test models for both llama.cpp and MLX.
It's useful for quickly testing the chat interfaces without downloading large models.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import urllib.request
import zipfile
import tarfile

# Define ANSI colors for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}

# Define URLs for small test models
MODEL_URLS = {
    "llama.cpp": {
        "name": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        "description": "TinyLlama 1.1B Chat (Q4_0 quantization, ~600MB)",
    },
    "mlx": {
        "name": "phi-2.mlx",
        "url": "https://huggingface.co/mlx-community/Phi-2-MLP/resolve/main/phi-2.tar.gz",
        "description": "Phi-2 model for MLX (~1.7GB)",
    },
}


def download_file(url, dest_path, desc=None):
    """Download a file with progress reporting."""
    if desc:
        print(f"{COLORS['blue']}Downloading {desc}...{COLORS['reset']}")
    
    try:
        # Simple progress reporter
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))
            if total_size > 0:
                sys.stdout.write(f"\r{COLORS['blue']}Progress: {percent}% ({downloaded / 1024 / 1024:.1f}MB/{total_size / 1024 / 1024:.1f}MB){COLORS['reset']}")
                sys.stdout.flush()
        
        # Download the file
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print()  # New line after progress
        
        return True
    except Exception as e:
        print(f"{COLORS['red']}Error downloading file: {e}{COLORS['reset']}")
        return False


def setup_llamacpp_model():
    """Download and set up a small test model for llama.cpp."""
    model_info = MODEL_URLS["llama.cpp"]
    
    # Determine the models directory
    models_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../llama.cpp-setup/models'
    ))
    
    # Create the directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Download the model
    model_path = os.path.join(models_dir, model_info["name"])
    
    if os.path.exists(model_path):
        print(f"{COLORS['yellow']}Model already exists at {model_path}{COLORS['reset']}")
        return model_path
    
    if download_file(model_info["url"], model_path, model_info["description"]):
        print(f"{COLORS['green']}Successfully downloaded llama.cpp model to {model_path}{COLORS['reset']}")
        return model_path
    
    return None


def setup_mlx_model():
    """Download and set up a small test model for MLX."""
    model_info = MODEL_URLS["mlx"]
    
    # Determine the models directory
    models_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../mlx-setup/models'
    ))
    
    # Create the directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Define the final model directory
    model_dir = os.path.join(models_dir, model_info["name"].split('.')[0])  # Remove extension
    
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"{COLORS['yellow']}Model already exists at {model_dir}{COLORS['reset']}")
        return model_dir
    
    # Download to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
        temp_path = temp_file.name
    
    if download_file(model_info["url"], temp_path, model_info["description"]):
        print(f"{COLORS['blue']}Extracting MLX model...{COLORS['reset']}")
        
        try:
            # Create the model directory
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Extract the tar.gz file
            with tarfile.open(temp_path, "r:gz") as tar:
                tar.extractall(path=model_dir)
            
            print(f"{COLORS['green']}Successfully extracted MLX model to {model_dir}{COLORS['reset']}")
            
            # Check if the model has the required files
            if not os.path.exists(os.path.join(model_dir, "config.json")):
                print(f"{COLORS['red']}Error: Extracted model doesn't have the required files{COLORS['reset']}")
                return None
            
            return model_dir
            
        except Exception as e:
            print(f"{COLORS['red']}Error extracting model: {e}{COLORS['reset']}")
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    return None


def check_environment():
    """Check if the necessary frameworks are installed."""
    has_llamacpp = False
    has_mlx = False
    
    # Check llama.cpp
    llamacpp_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../llama.cpp-setup/build/main'
    ))
    
    if os.path.exists(llamacpp_path) and os.access(llamacpp_path, os.X_OK):
        has_llamacpp = True
    else:
        try:
            result = subprocess.run(
                ['which', 'main'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                has_llamacpp = True
        except:
            pass
    
    # Check MLX
    try:
        import mlx
        import mlx.core
        has_mlx = True
    except ImportError:
        pass
    
    return has_llamacpp, has_mlx


def main():
    """Main function to set up test models."""
    parser = argparse.ArgumentParser(description="Setup Test Models for Chat Interfaces")
    parser.add_argument(
        "--framework", "-f", 
        type=str, 
        choices=["llama.cpp", "mlx", "both"],
        default="both",
        help="Which framework to set up a test model for"
    )
    
    args = parser.parse_args()
    
    # Check environment
    has_llamacpp, has_mlx = check_environment()
    
    if args.framework in ["llama.cpp", "both"] and not has_llamacpp:
        print(f"{COLORS['red']}Warning: llama.cpp doesn't seem to be properly installed.{COLORS['reset']}")
        print(f"{COLORS['yellow']}You may need to build llama.cpp first.{COLORS['reset']}")
        if args.framework == "llama.cpp":
            return 1
    
    if args.framework in ["mlx", "both"] and not has_mlx:
        print(f"{COLORS['red']}Warning: MLX doesn't seem to be properly installed.{COLORS['reset']}")
        print(f"{COLORS['yellow']}You may need to install MLX first: pip install mlx mlx-lm{COLORS['reset']}")
        if args.framework == "mlx":
            return 1
    
    # Set up models
    llamacpp_model_path = None
    mlx_model_path = None
    
    if args.framework in ["llama.cpp", "both"] and has_llamacpp:
        print(f"{COLORS['bold']}{COLORS['cyan']}Setting up llama.cpp test model...{COLORS['reset']}")
        llamacpp_model_path = setup_llamacpp_model()
    
    if args.framework in ["mlx", "both"] and has_mlx:
        print(f"{COLORS['bold']}{COLORS['magenta']}Setting up MLX test model...{COLORS['reset']}")
        mlx_model_path = setup_mlx_model()
    
    # Print summary
    print(f"\n{COLORS['bold']}{COLORS['green']}Setup Summary:{COLORS['reset']}")
    
    if llamacpp_model_path:
        print(f"{COLORS['green']}llama.cpp model: {llamacpp_model_path}{COLORS['reset']}")
        print(f"{COLORS['yellow']}Try it with: python ../llama_cpp/cli/chat_cli.py --model {llamacpp_model_path}{COLORS['reset']}")
    elif args.framework in ["llama.cpp", "both"]:
        print(f"{COLORS['red']}Failed to set up llama.cpp model{COLORS['reset']}")
    
    if mlx_model_path:
        print(f"{COLORS['green']}MLX model: {mlx_model_path}{COLORS['reset']}")
        print(f"{COLORS['yellow']}Try it with: python ../mlx/cli/chat_cli.py --model {mlx_model_path}{COLORS['reset']}")
    elif args.framework in ["mlx", "both"]:
        print(f"{COLORS['red']}Failed to set up MLX model{COLORS['reset']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())