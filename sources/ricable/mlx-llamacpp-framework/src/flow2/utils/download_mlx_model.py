#!/usr/bin/env python3
"""
Download MLX Model Utility

This script simplifies downloading, converting, and quantizing models for MLX.
It provides a user-friendly interface to the model_manager module.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path to import model_manager
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

try:
    from model_manager import (
        download_mlx_model,
        convert_to_mlx,
        quantize_mlx_model,
        get_available_models,
        get_model_info,
        get_recommended_models,
        get_system_ram_gb,
        verify_model_integrity,
        load_model_registry
    )
except ImportError:
    print("Error: model_manager.py not found. Make sure it's in the same directory as this script.")
    sys.exit(1)

DEFAULT_MLX_PATH = os.path.expanduser("~/dev/ran/flow2/mlx-setup")
DEFAULT_MODELS_DIR = "models"

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download, convert and quantize models for MLX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main options
    parser.add_argument(
        "model", nargs="?", 
        help="Model to download (use 'list' to see available models, or specify a Hugging Face ID)"
    )
    
    parser.add_argument(
        "--list", action="store_true",
        help="List available pre-configured models"
    )
    
    parser.add_argument(
        "--recommend", action="store_true",
        help="Show recommended models for your system"
    )
    
    # Model source options
    model_source = parser.add_argument_group("Model source options")
    model_source.add_argument(
        "--hf", metavar="HF_ID",
        help="Download and convert from Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b')"
    )
    
    # Output options
    output_options = parser.add_argument_group("Output options")
    output_options.add_argument(
        "--output-dir", 
        default=os.path.join(DEFAULT_MLX_PATH, DEFAULT_MODELS_DIR),
        help="Directory to save the model"
    )
    
    output_options.add_argument(
        "--output-name",
        help="Directory name for the downloaded model (default: auto-generated based on model name)"
    )
    
    # Quantization options
    quant_options = parser.add_argument_group("Quantization options")
    quant_options.add_argument(
        "--quant", 
        choices=["int4", "int8", "none"],
        help="Quantize the model after downloading (default: no quantization)"
    )
    
    quant_options.add_argument(
        "--quantize-only", metavar="INPUT_DIR",
        help="Quantize an existing MLX model directory instead of downloading"
    )
    
    # Verification options
    verify_options = parser.add_argument_group("Verification options")
    verify_options.add_argument(
        "--verify", action="store_true",
        help="Verify model files exist and have the correct structure"
    )
    
    # Advanced options
    advanced = parser.add_argument_group("Advanced options")
    advanced.add_argument(
        "--info", metavar="MODEL_NAME",
        help="Show detailed information about a specific model"
    )
    
    return parser.parse_args()

def list_models():
    """List all available MLX models."""
    print("Available pre-configured models for MLX:")
    models = get_available_models("mlx")
    
    for model_name in models:
        model_info = get_model_info(model_name)
        description = model_info.get("description", "")
        mlx_name = model_info.get("mlx_name", "")
        print(f"  - {model_name}")
        print(f"    Description: {description}")
        print(f"    MLX name: {mlx_name}")
        print()

def show_recommendations():
    """Show recommended models based on system specs."""
    ram_gb = get_system_ram_gb()
    print(f"System RAM: {ram_gb}GB")
    
    recommendations = get_recommended_models(ram_gb)["mlx"]
    
    print("\nRecommended MLX models for your system:")
    for model in recommendations:
        model_name, quant = model.split(" ")
        quant = quant.strip("()")
        print(f"  - {model_name} with {quant} quantization")
    
    print("\nTo download a recommended model, run:")
    if recommendations:
        model_example = recommendations[0].split(" ")[0]
        quant_example = recommendations[0].split(" ")[1].strip("()")
        print(f"  {sys.argv[0]} {model_example} --quant {quant_example}")

def show_model_info(model_name):
    """Show detailed information about a specific model."""
    try:
        info = get_model_info(model_name)
        print(f"Model: {model_name}")
        print(f"Description: {info.get('description', 'N/A')}")
        print(f"Repository: {info.get('repo', 'N/A')}")
        print(f"MLX name: {info.get('mlx_name', 'N/A')}")
        print(f"License: {info.get('license', 'N/A')}")
        
        if "papers" in info:
            print("\nResearch Papers:")
            for paper in info["papers"]:
                print(f"  - {paper}")
        
        if "license_info" in info:
            license_info = info["license_info"]
            print("\nLicense Details:")
            print(f"  Name: {license_info['name']}")
            print(f"  URL: {license_info['url']}")
            print(f"  Restrictions: {license_info['restrictions']}")
            print(f"  Attribution required: {license_info['attribution']}")
            print(f"  Registration required: {license_info['registration']}")
        
        print("\nRecommended Quantization:")
        ram_gb = get_system_ram_gb()
        if ram_gb <= 8:
            print("  int4 (for your system with 8GB RAM)")
        elif ram_gb <= 16:
            print("  int8 (for your system with 16GB RAM)")
        else:
            print("  none/fp16 (for your system with 32GB+ RAM)")
        
        print("\nTypical Memory Usage:")
        model_size = "7B" if "7b" in model_name.lower() else "13B" if "13b" in model_name.lower() else "2B" if "2b" in model_name.lower() else "Unknown"
        if model_size == "7B":
            print("  FP16: ~14GB RAM")
            print("  INT8: ~7.5GB RAM")
            print("  INT4: ~4GB RAM")
        elif model_size == "13B":
            print("  FP16: ~26GB RAM")
            print("  INT8: ~13.5GB RAM")
            print("  INT4: ~7GB RAM")
        elif model_size == "2B":
            print("  FP16: ~4GB RAM")
            print("  INT8: ~2.2GB RAM")
            print("  INT4: ~1.3GB RAM")
        
    except Exception as e:
        print(f"Error: Could not get info for model '{model_name}': {e}")
        sys.exit(1)

def download_from_huggingface(hf_id, output_dir, output_name):
    """Download and convert a model from Hugging Face."""
    try:
        if not output_name:
            # Extract model name from HF ID
            output_name = hf_id.split("/")[-1].lower()
        
        output_path = os.path.join(output_dir, output_name)
        
        print(f"Converting {hf_id} to MLX format...")
        convert_to_mlx(hf_id, output_path)
        
        return output_path
    except Exception as e:
        print(f"Error converting model from Hugging Face: {e}")
        sys.exit(1)

def quantize_model_dir(input_dir, quant):
    """Quantize an existing MLX model directory."""
    try:
        if not os.path.isdir(input_dir):
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)
        
        # Generate output directory name
        output_dir = f"{input_dir}_{quant}"
        
        print(f"Quantizing model in {input_dir} to {quant}...")
        quantized_dir = quantize_mlx_model(input_dir, quant, output_dir)
        
        return quantized_dir
    except Exception as e:
        print(f"Error quantizing model: {e}")
        sys.exit(1)

def download_model(model_name, output_dir, output_name=None):
    """Download a pre-configured model."""
    try:
        print(f"Downloading {model_name}...")
        output_path = download_mlx_model(model_name, output_dir)
        
        return output_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def verify_mlx_model(model_dir):
    """Verify that the MLX model directory has the required files."""
    try:
        required_files = ["weights.safetensors", "config.json", "tokenizer.json"]
        missing_files = []
        
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"Warning: The following required files are missing: {', '.join(missing_files)}")
            return False
        else:
            print(f"Model directory structure verified: All required files present in {model_dir}")
            return True
    except Exception as e:
        print(f"Error verifying model: {e}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load model registry
    load_model_registry()
    
    # Handle informational commands
    if args.list:
        list_models()
        return
    
    if args.recommend:
        show_recommendations()
        return
    
    if args.info:
        show_model_info(args.info)
        return
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle model quantization
    if args.quantize_only:
        output_path = quantize_model_dir(args.quantize_only, args.quant)
        print(f"Model quantized and saved to: {output_path}")
        
        # Provide sample code to use the model
        print("\nSample code to use the quantized model:")
        print("```python")
        print("from mlx_lm import load, generate")
        print(f"model, tokenizer = load(\"{output_path}\")")
        print("output = generate(model, tokenizer, \"Your prompt here\", max_tokens=256)")
        print("print(tokenizer.decode(output))")
        print("```")
        return
    
    # Handle model download from Hugging Face
    if args.hf:
        output_path = download_from_huggingface(args.hf, args.output_dir, args.output_name)
        print(f"Model converted and saved to: {output_path}")
        
        # Quantize if requested
        if args.quant:
            quantized_path = quantize_model_dir(output_path, args.quant)
            print(f"Model quantized and saved to: {quantized_path}")
            output_path = quantized_path
        
        # Provide sample code to use the model
        print("\nSample code to use the model:")
        print("```python")
        print("from mlx_lm import load, generate")
        print(f"model, tokenizer = load(\"{output_path}\")")
        print("output = generate(model, tokenizer, \"Your prompt here\", max_tokens=256)")
        print("print(tokenizer.decode(output))")
        print("```")
        return
    
    # Handle pre-configured model download
    if args.model and args.model != "list":
        output_path = download_model(args.model, args.output_dir, args.output_name)
        print(f"Model downloaded and saved to: {output_path}")
        
        # Verify if requested
        if args.verify:
            verify_mlx_model(output_path)
        
        # Quantize if requested
        if args.quant:
            quantized_path = quantize_model_dir(output_path, args.quant)
            print(f"Model quantized and saved to: {quantized_path}")
            output_path = quantized_path
        
        # Provide sample code to use the model
        print("\nSample code to use the model:")
        print("```python")
        print("from mlx_lm import load, generate")
        model_info = get_model_info(args.model)
        mlx_name = model_info.get("mlx_name", "")
        
        if args.quant:
            print(f"model, tokenizer = load(\"{output_path}\")")
        else:
            print(f"model, tokenizer = load(\"{mlx_name}\")")
        
        print("output = generate(model, tokenizer, \"Your prompt here\", max_tokens=256)")
        print("print(tokenizer.decode(output))")
        print("```")
        
        # Provide chat example if it's a chat model
        if "chat" in args.model.lower() or "instruct" in args.model.lower():
            print("\nFor interactive chat:")
            print("```python")
            print("from mlx_lm import load")
            print("from mlx_lm.utils import chat")
            
            if args.quant:
                print(f"model, tokenizer = load(\"{output_path}\")")
            else:
                print(f"model, tokenizer = load(\"{mlx_name}\")")
            
            print("chat(model, tokenizer)")
            print("```")
        
        return
    
    # If no action specified, show help
    parse_arguments().__class__.print_help(parser)

if __name__ == "__main__":
    main()