#!/usr/bin/env python3
"""
Download Llama.cpp Model Utility

This script simplifies downloading and quantizing models for llama.cpp.
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
        download_llama_cpp_model,
        convert_to_gguf,
        quantize_gguf_model,
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

DEFAULT_LLAMA_CPP_PATH = os.path.expanduser("~/dev/ran/flow2/llama.cpp-setup")
DEFAULT_MODELS_DIR = "models"

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download, convert and quantize models for llama.cpp",
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
    
    model_source.add_argument(
        "--direct", metavar="URL",
        help="Download model directly from URL"
    )
    
    # Output options
    output_options = parser.add_argument_group("Output options")
    output_options.add_argument(
        "--output-dir", 
        default=os.path.join(DEFAULT_LLAMA_CPP_PATH, DEFAULT_MODELS_DIR),
        help="Directory to save the model"
    )
    
    output_options.add_argument(
        "--output-name",
        help="Filename for the downloaded model (default: auto-generated based on model name and quantization)"
    )
    
    # Quantization options
    quant_options = parser.add_argument_group("Quantization options")
    quant_options.add_argument(
        "--quant", default="q4_k_m",
        choices=["f16", "q2_k", "q3_k", "q4_0", "q4_k", "q4_k_m", "q5_0", "q5_k", "q5_k_m", "q6_k", "q8_0"],
        help="Quantization type"
    )
    
    quant_options.add_argument(
        "--quantize-only", metavar="INPUT_FILE",
        help="Quantize an existing GGUF model file instead of downloading"
    )
    
    # Verification options
    verify_options = parser.add_argument_group("Verification options")
    verify_options.add_argument(
        "--verify", action="store_true",
        help="Verify model integrity after download"
    )
    
    # Advanced options
    advanced = parser.add_argument_group("Advanced options")
    advanced.add_argument(
        "--no-cache", action="store_true",
        help="Force download even if model already exists"
    )
    
    advanced.add_argument(
        "--info", metavar="MODEL_NAME",
        help="Show detailed information about a specific model"
    )
    
    return parser.parse_args()

def list_models():
    """List all available llama.cpp models."""
    print("Available pre-configured models for llama.cpp:")
    models = get_available_models("llama.cpp")
    
    for model_name in models:
        model_info = get_model_info(model_name)
        description = model_info.get("description", "")
        quants = ", ".join(model_info["files"].keys())
        print(f"  - {model_name}")
        print(f"    Description: {description}")
        print(f"    Available quantizations: {quants}")
        print()

def show_recommendations():
    """Show recommended models based on system specs."""
    ram_gb = get_system_ram_gb()
    print(f"System RAM: {ram_gb}GB")
    
    recommendations = get_recommended_models(ram_gb)["llama.cpp"]
    
    print("\nRecommended llama.cpp models for your system:")
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
        print(f"License: {info.get('license', 'N/A')}")
        
        print("\nAvailable quantization levels:")
        for quant in info["files"].keys():
            print(f"  - {quant}")
        
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
    except Exception as e:
        print(f"Error: Could not get info for model '{model_name}': {e}")
        sys.exit(1)

def download_from_huggingface(hf_id, output_dir, output_name, quant):
    """Download and convert a model from Hugging Face."""
    try:
        if not output_name:
            # Extract model name from HF ID and add quantization suffix
            model_name = hf_id.split("/")[-1].lower()
            output_name = f"{model_name}.{quant}.gguf"
        
        output_path = os.path.join(output_dir, output_name)
        
        print(f"Converting {hf_id} to GGUF format with {quant} quantization...")
        convert_to_gguf(hf_id, output_path, quant)
        
        return output_path
    except Exception as e:
        print(f"Error converting model from Hugging Face: {e}")
        sys.exit(1)

def quantize_existing_model(input_file, output_dir, output_name, quant):
    """Quantize an existing GGUF model."""
    try:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)
        
        if not output_name:
            # Generate output filename based on input filename and quantization
            input_basename = os.path.basename(input_file)
            name_parts = input_basename.split(".")
            if len(name_parts) > 1 and name_parts[-2] in ["f16", "q2_k", "q3_k", "q4_0", "q4_k", "q4_k_m", "q5_0", "q5_k", "q5_k_m", "q6_k", "q8_0"]:
                # Replace existing quantization
                name_parts[-2] = quant
                output_name = ".".join(name_parts)
            else:
                # Add quantization suffix
                output_name = f"{input_basename.rsplit('.', 1)[0]}.{quant}.gguf"
        
        output_path = os.path.join(output_dir, output_name)
        
        print(f"Quantizing {input_file} to {quant}...")
        quantize_gguf_model(input_file, output_path, quant)
        
        return output_path
    except Exception as e:
        print(f"Error quantizing model: {e}")
        sys.exit(1)

def download_model(model_name, output_dir, output_name, quant, verify):
    """Download a pre-configured model."""
    try:
        print(f"Downloading {model_name} with {quant} quantization...")
        output_path = download_llama_cpp_model(model_name, quant, output_dir)
        
        if verify:
            print("Verifying model integrity...")
            is_valid, hash_value = verify_model_integrity(output_path)
            if is_valid:
                print(f"Model integrity verified. Hash: {hash_value}")
            else:
                print(f"Warning: Model integrity verification failed!")
        
        return output_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

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
        output_path = quantize_existing_model(
            args.quantize_only, 
            args.output_dir, 
            args.output_name, 
            args.quant
        )
        print(f"Model quantized and saved to: {output_path}")
        return
    
    # Handle model download from Hugging Face
    if args.hf:
        output_path = download_from_huggingface(
            args.hf, 
            args.output_dir, 
            args.output_name, 
            args.quant
        )
        print(f"Model converted and saved to: {output_path}")
        return
    
    # Handle direct URL download
    if args.direct:
        print("Direct URL download not implemented yet.")
        return
    
    # Handle pre-configured model download
    if args.model and args.model != "list":
        output_path = download_model(
            args.model, 
            args.output_dir, 
            args.output_name, 
            args.quant, 
            args.verify
        )
        print(f"Model downloaded and saved to: {output_path}")
        
        # Provide run example
        print("\nTo run the model:")
        run_script = os.path.join(DEFAULT_LLAMA_CPP_PATH, "run-model.sh")
        if os.path.exists(run_script):
            print(f"  {run_script} {output_path} \"Your prompt here\"")
        else:
            print(f"  cd {DEFAULT_LLAMA_CPP_PATH} && ./main -m {output_path} --color --ctx 2048 -n 256 -p \"Your prompt here\"")
        
        # Provide server example
        print("\nTo run the model as a server:")
        server_script = os.path.join(DEFAULT_LLAMA_CPP_PATH, "run-server.sh")
        if os.path.exists(server_script):
            print(f"  {server_script} {output_path} 8080")
        else:
            print(f"  cd {DEFAULT_LLAMA_CPP_PATH} && ./server -m {output_path} --ctx 2048 --port 8080")
        
        return
    
    # If no action specified, show help
    parse_arguments().__class__.print_help(parser)

if __name__ == "__main__":
    main()