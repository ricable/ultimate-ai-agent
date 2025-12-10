#!/usr/bin/env python3
"""
LLM Model Manager CLI

A unified command-line interface for managing models for llama.cpp and MLX frameworks.
This tool provides a simple way to download, convert, quantize, and manage models
for both frameworks on Apple Silicon Macs.
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
        download_mlx_model,
        convert_to_gguf,
        convert_to_mlx,
        quantize_gguf_model,
        quantize_mlx_model,
        get_available_models,
        get_model_info,
        get_recommended_models,
        get_system_ram_gb,
        verify_model_integrity,
        load_model_registry,
        list_installed_models,
        get_model_license_info,
        save_model_registry,
        DEFAULT_LLAMA_CPP_PATH,
        DEFAULT_MLX_PATH,
        DEFAULT_MODELS_DIR
    )
except ImportError:
    print("Error: model_manager.py not found. Make sure it's in the same directory as this script.")
    sys.exit(1)


def get_framework_from_path(path):
    """Detect framework from model path."""
    if os.path.isfile(path) and path.endswith(".gguf"):
        return "llama.cpp"
    
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "weights.safetensors")):
        return "mlx"
    
    print(f"Warning: Could not detect framework from path: {path}")
    return None

def show_run_examples(model_path, framework):
    """Show examples of how to run a model."""
    if framework == "llama.cpp":
        print(f"To run {model_path} with llama.cpp:")
        print("\nCommand-line interface:")
        print(f"  cd {DEFAULT_LLAMA_CPP_PATH} && ./main -m {model_path} --color --ctx 2048 -n 256 -p \"Your prompt here\"")
        
        print("\nInteractive chat mode:")
        print(f"  cd {DEFAULT_LLAMA_CPP_PATH} && ./main -m {model_path} --color --ctx 2048 --interactive -r \"User:\" -f prompts/chat-with-bob.txt")
        
        print("\nServer mode (for web UI or API access):")
        print(f"  cd {DEFAULT_LLAMA_CPP_PATH} && ./server -m {model_path} --ctx 2048 --host 0.0.0.0 --port 8080")
        
        print("\nPython integration:")
        print("```python")
        print("from llama_cpp import Llama")
        print(f"model = Llama(model_path=\"{model_path}\", n_ctx=2048, n_gpu_layers=1)")
        print("output = model.generate(\"Your prompt here\", max_tokens=256)")
        print("print(output['choices'][0]['text'])")
        print("```")
    
    elif framework == "mlx":
        # Determine if the model path is a directory name or a full path
        model_arg = os.path.basename(model_path) if os.path.dirname(model_path) == os.path.join(DEFAULT_MLX_PATH, DEFAULT_MODELS_DIR) else f"\"{model_path}\""
        
        print(f"To use {model_path} with MLX:")
        print("\nBasic generation:")
        print("```python")
        print("from mlx_lm import load, generate")
        print(f"model, tokenizer = load({model_arg})")
        print("output = generate(model, tokenizer, \"Your prompt here\", max_tokens=256)")
        print("print(tokenizer.decode(output))")
        print("```")
        
        print("\nInteractive chat:")
        print("```python")
        print("from mlx_lm import load")
        print("from mlx_lm.utils import chat")
        print(f"model, tokenizer = load({model_arg})")
        print("chat(model, tokenizer)")
        print("```")
        
        print("\nServer mode (requires Flask):")
        print("```python")
        print("from mlx_lm import load")
        print("from flask import Flask, request, jsonify")
        print(f"model, tokenizer = load({model_arg})")
        print("app = Flask(__name__)")
        print("@app.route('/generate', methods=['POST'])")
        print("def generate_text():")
        print("    data = request.json")
        print("    prompt = data.get('prompt', '')")
        print("    max_tokens = data.get('max_tokens', 256)")
        print("    from mlx_lm import generate")
        print("    output = generate(model, tokenizer, prompt, max_tokens=max_tokens)")
        print("    return jsonify({'text': tokenizer.decode(output)})")
        print("if __name__ == '__main__':")
        print("    app.run(host='0.0.0.0', port=8080)")
        print("```")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="LLM Model Manager - A unified CLI for managing models for llama.cpp and MLX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available or installed models")
    list_parser.add_argument(
        "--installed", action="store_true",
        help="List installed models instead of available ones"
    )
    list_parser.add_argument(
        "--framework", choices=["llama.cpp", "mlx"], 
        help="Filter by framework"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about a model")
    info_parser.add_argument("model", help="Model name")
    
    # License command
    license_parser = subparsers.add_parser("license", help="Get license information for a model")
    license_parser.add_argument("model", help="Model name")
    
    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Get model recommendations based on system specs")
    recommend_parser.add_argument(
        "--ram", type=int, 
        help="System RAM in GB (detected automatically if not specified)"
    )
    recommend_parser.add_argument(
        "--framework", choices=["llama.cpp", "mlx"], 
        help="Filter recommendations by framework"
    )
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model", help="Model name to download")
    download_parser.add_argument(
        "--framework", choices=["llama.cpp", "mlx"], required=True,
        help="Target framework"
    )
    download_parser.add_argument(
        "--quant", 
        help="Quantization level (q4_k_m, q8_0 for llama.cpp; int4, int8 for MLX)"
    )
    download_parser.add_argument(
        "--output", 
        help="Output directory (default: ./models/[framework])"
    )
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert a model from Hugging Face")
    convert_parser.add_argument("model_name", help="Hugging Face model name")
    convert_parser.add_argument(
        "--framework", choices=["llama.cpp", "mlx"], required=True,
        help="Target framework"
    )
    convert_parser.add_argument(
        "--output", 
        help="Output directory (default: ./models/[framework])"
    )
    
    # Quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Quantize a model")
    quantize_parser.add_argument("input", help="Input model path")
    quantize_parser.add_argument(
        "--framework", choices=["llama.cpp", "mlx"], required=True,
        help="Framework"
    )
    quantize_parser.add_argument(
        "--quant", required=True,
        help="Quantization level (q4_k_m, q8_0 for llama.cpp; int4, int8 for MLX)"
    )
    quantize_parser.add_argument(
        "--output", 
        help="Output path (default: input_path + quantization suffix)"
    )
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify model integrity")
    verify_parser.add_argument("model_path", help="Path to model file or directory")
    verify_parser.add_argument(
        "--framework", choices=["llama.cpp", "mlx"],
        help="Framework (detected automatically if not specified)"
    )
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Show examples of how to run a model")
    run_parser.add_argument("model_path", help="Path to model file or directory")
    run_parser.add_argument(
        "--framework", choices=["llama.cpp", "mlx"], required=True,
        help="Framework"
    )
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update model registry with new models")
    
    args = parser.parse_args()
    
    # Load model registry
    load_model_registry()
    
    if args.command == "list":
        if args.installed:
            installed = list_installed_models(framework=args.framework)
            for fw, models in installed.items():
                if args.framework is None or fw == args.framework:
                    print(f"{fw} models:")
                    for model in sorted(models):
                        print(f"  - {model}")
                    print()
        else:
            models = get_available_models(args.framework)
            if args.framework:
                print(f"Available {args.framework} models:")
            else:
                print("Available models:")
            
            for model in sorted(models):
                info = get_model_info(model)
                framework = info["framework"]
                description = info.get("description", "")
                print(f"  - {model} ({framework}): {description}")
    
    elif args.command == "info":
        try:
            info = get_model_info(args.model)
            print(f"Model: {args.model}")
            print(f"Framework: {info['framework']}")
            print(f"Description: {info.get('description', 'N/A')}")
            print(f"Repository: {info.get('repo', 'N/A')}")
            print(f"License: {info.get('license', 'N/A')}")
            
            if info["framework"] == "llama.cpp":
                print("\nAvailable quantization levels:")
                for quant in info["files"].keys():
                    print(f"  - {quant}")
            
            if info["framework"] == "mlx":
                print(f"MLX name: {info.get('mlx_name', 'N/A')}")
            
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
            print(f"Error: {e}")
            sys.exit(1)
    
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
    
    elif args.command == "recommend":
        ram_gb = args.ram if args.ram else get_system_ram_gb()
        print(f"System RAM: {ram_gb}GB")
        
        recommendations = get_recommended_models(ram_gb)
        
        for framework, models in recommendations.items():
            if args.framework is None or framework == args.framework:
                print(f"\n{framework.upper()} recommended models:")
                for model in models:
                    print(f"  - {model}")
                
                # Show download examples
                if models:
                    print(f"\nTo download a recommended {framework} model:")
                    model_example = models[0].split(" ")[0]
                    quant_example = models[0].split(" ")[1].strip("()")
                    print(f"  python model_cli.py download {model_example} --framework {framework} --quant {quant_example}")
    
    elif args.command == "download":
        try:
            if args.framework == "llama.cpp":
                quant = args.quant if args.quant else "q4_k_m"
                output_dir = args.output_dir if args.output_dir else os.path.join(DEFAULT_LLAMA_CPP_PATH, DEFAULT_MODELS_DIR)
                
                print(f"Downloading {args.model} for llama.cpp with {quant} quantization...")
                output_path = download_llama_cpp_model(args.model, quant, output_dir)
                
                print(f"Model downloaded to: {output_path}")
                show_run_examples(output_path, "llama.cpp")
            
            elif args.framework == "mlx":
                output_dir = args.output_dir if args.output_dir else os.path.join(DEFAULT_MLX_PATH, DEFAULT_MODELS_DIR)
                
                print(f"Downloading {args.model} for MLX...")
                output_path = download_mlx_model(args.model, output_dir)
                
                print(f"Model downloaded to: {output_path}")
                
                # Quantize if requested
                if args.quant:
                    if args.quant in ["int4", "int8"]:
                        quantized_path = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_{args.quant}")
                        print(f"Quantizing model to {args.quant}...")
                        quantized_path = quantize_mlx_model(output_path, args.quant, quantized_path)
                        print(f"Quantized model saved to: {quantized_path}")
                        output_path = quantized_path
                    else:
                        print(f"Warning: Invalid MLX quantization level: {args.quant}. Skipping quantization.")
                
                show_run_examples(output_path, "mlx")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "convert":
        try:
            if args.framework == "llama.cpp":
                quant = args.quant if args.quant else "q4_k"
                
                print(f"Converting {args.hf} to GGUF format with {quant} quantization...")
                output_path = convert_to_gguf(args.hf, args.output, quant)
                
                print(f"Converted model saved to: {output_path}")
                show_run_examples(output_path, "llama.cpp")
            
            elif args.framework == "mlx":
                print(f"Converting {args.hf} to MLX format...")
                output_path = convert_to_mlx(args.hf, args.output)
                
                print(f"Converted model saved to: {output_path}")
                
                # Quantize if requested
                if args.quant:
                    if args.quant in ["int4", "int8"]:
                        quantized_path = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_{args.quant}")
                        print(f"Quantizing model to {args.quant}...")
                        quantized_path = quantize_mlx_model(output_path, args.quant, quantized_path)
                        print(f"Quantized model saved to: {quantized_path}")
                        output_path = quantized_path
                    else:
                        print(f"Warning: Invalid MLX quantization level: {args.quant}. Skipping quantization.")
                
                show_run_examples(output_path, "mlx")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "quantize":
        try:
            if args.framework == "llama.cpp":
                print(f"Quantizing {args.input} to {args.quant}...")
                output_path = quantize_gguf_model(args.input, args.output, args.quant)
                
                print(f"Quantized model saved to: {output_path}")
                show_run_examples(output_path, "llama.cpp")
            
            elif args.framework == "mlx":
                if args.quant not in ["int4", "int8"]:
                    print(f"Error: Invalid MLX quantization level: {args.quant}. Use 'int4' or 'int8'.")
                    sys.exit(1)
                
                print(f"Quantizing {args.input} to {args.quant}...")
                output_path = quantize_mlx_model(args.input, args.quant, args.output)
                
                print(f"Quantized model saved to: {output_path}")
                show_run_examples(output_path, "mlx")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "verify":
        try:
            framework = args.framework
            if framework is None:
                framework = get_framework_from_path(args.model_path)
                if framework is None:
                    print("Error: Could not detect framework. Please specify with --framework.")
                    sys.exit(1)
            
            if framework == "llama.cpp":
                is_valid, actual_hash = verify_model_integrity(args.model_path)
                if is_valid:
                    print(f"Model integrity verified: {args.model_path}")
                    print(f"SHA-256: {actual_hash}")
                else:
                    print(f"Model integrity verification failed: {args.model_path}")
                    sys.exit(1)
            
            elif framework == "mlx":
                required_files = ["weights.safetensors", "config.json", "tokenizer.json"]
                missing_files = []
                
                for file in required_files:
                    file_path = os.path.join(args.model_path, file)
                    if not os.path.exists(file_path):
                        missing_files.append(file)
                
                if missing_files:
                    print(f"Warning: The following required files are missing: {', '.join(missing_files)}")
                    sys.exit(1)
                else:
                    print(f"Model directory structure verified: All required files present in {args.model_path}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "run":
        show_run_examples(args.model_path, args.framework)
    
    elif args.command == "update":
        print("Updating model registry...")
        
        # Here you could add code to fetch the latest model information
        # from online sources or update based on local discovery
        
        # For now, we'll just save the current registry
        save_model_registry()
        print("Model registry updated.")
    
    else:
        # If no command specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()