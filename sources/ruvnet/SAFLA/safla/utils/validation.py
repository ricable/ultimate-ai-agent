"""
SAFLA Validation Utilities

This module provides validation functions for configuration, environment,
and system requirements.
"""

import os
import sys
import importlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import platform

from safla.utils.config import SAFLAConfig


def validate_python_version(min_version: Tuple[int, int] = (3, 8)) -> List[str]:
    """
    Validate Python version meets minimum requirements.
    
    Args:
        min_version: Minimum required Python version as (major, minor)
        
    Returns:
        List of validation errors
    """
    errors = []
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        errors.append(
            f"Python {min_version[0]}.{min_version[1]}+ required, "
            f"but {current_version[0]}.{current_version[1]} found"
        )
    
    return errors


def validate_dependencies() -> List[str]:
    """
    Validate that required dependencies are available.
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Core dependencies
    required_packages = [
        "numpy",
        "scipy", 
        "sklearn",
        "aiohttp",
        "pydantic",
        "rich",
        "tqdm",
        "click",
        "dotenv",
        "psutil",
        "networkx"
    ]
    
    # Optional dependencies with fallbacks
    optional_packages = {
        "faiss": "Vector similarity search (CPU version)",
        "torch": "Neural network operations",
        "transformers": "Transformer models",
        "sentence_transformers": "Sentence embeddings"
    }
    
    # Check required packages
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            errors.append(f"Required package '{package}' not found")
    
    # Check optional packages (warnings, not errors)
    missing_optional = []
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
        except ImportError:
            missing_optional.append(f"Optional package '{package}' not found ({description})")
    
    if missing_optional:
        errors.extend([f"WARNING: {msg}" for msg in missing_optional])
    
    return errors


def validate_system_resources() -> List[str]:
    """
    Validate system resources meet minimum requirements.
    
    Returns:
        List of validation errors
    """
    errors = []
    
    try:
        import psutil
        
        # Check available memory (minimum 1GB)
        available_memory = psutil.virtual_memory().available
        min_memory = 1024 * 1024 * 1024  # 1GB
        
        if available_memory < min_memory:
            errors.append(
                f"Insufficient memory: {available_memory / 1024 / 1024:.0f}MB available, "
                f"minimum {min_memory / 1024 / 1024:.0f}MB required"
            )
        
        # Check disk space (minimum 1GB free)
        disk_usage = psutil.disk_usage('/')
        min_disk = 1024 * 1024 * 1024  # 1GB
        
        if disk_usage.free < min_disk:
            errors.append(
                f"Insufficient disk space: {disk_usage.free / 1024 / 1024:.0f}MB free, "
                f"minimum {min_disk / 1024 / 1024:.0f}MB required"
            )
        
        # Check CPU count (minimum 2 cores recommended)
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            errors.append(
                f"WARNING: Only {cpu_count} CPU core(s) detected, "
                "2+ cores recommended for optimal performance"
            )
    
    except ImportError:
        errors.append("WARNING: psutil not available for system resource validation")
    
    return errors


def validate_config(config: SAFLAConfig) -> List[str]:
    """
    Validate SAFLA configuration.
    
    Args:
        config: SAFLA configuration to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Use the config's built-in validation
    config_errors = config.validate()
    errors.extend(config_errors)
    
    # Additional validation
    try:
        # Validate data directory is writable
        data_path = Path(config.data_dir)
        if data_path.exists() and not os.access(data_path, os.W_OK):
            errors.append(f"Data directory not writable: {config.data_dir}")
        
        # Validate config directory is writable
        config_path = Path(config.config_dir)
        if config_path.exists() and not os.access(config_path, os.W_OK):
            errors.append(f"Config directory not writable: {config.config_dir}")
        
        # Validate vector dimensions are reasonable
        for dim in config.memory.vector_dimensions:
            if dim <= 0 or dim > 10000:
                errors.append(f"Invalid vector dimension: {dim} (must be 1-10000)")
        
        # Validate memory limits are reasonable
        if config.safety.memory_limit > 10 * 1024 * 1024 * 1024:  # 10GB
            errors.append("WARNING: Memory limit very high (>10GB)")
        
    except Exception as e:
        errors.append(f"Configuration validation error: {str(e)}")
    
    return errors


def validate_environment() -> List[str]:
    """
    Validate the complete environment for SAFLA.
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Validate Python version
    errors.extend(validate_python_version())
    
    # Validate dependencies
    errors.extend(validate_dependencies())
    
    # Validate system resources
    errors.extend(validate_system_resources())
    
    # Validate platform compatibility
    current_platform = platform.system()
    supported_platforms = ["Linux", "Darwin", "Windows"]
    
    if current_platform not in supported_platforms:
        errors.append(f"WARNING: Platform '{current_platform}' not officially supported")
    
    return errors


def validate_installation() -> Dict[str, Any]:
    """
    Comprehensive installation validation.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "system_info": {
            "platform": platform.system(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "architecture": platform.machine()
        }
    }
    
    # Run all validations
    all_errors = validate_environment()
    
    # Separate errors and warnings
    for error in all_errors:
        if error.startswith("WARNING:"):
            results["warnings"].append(error[9:])  # Remove "WARNING: " prefix
        else:
            results["errors"].append(error)
    
    # Set overall validity
    results["valid"] = len(results["errors"]) == 0
    
    # Add system resource info if psutil is available
    try:
        import psutil
        results["system_info"].update({
            "memory_total": f"{psutil.virtual_memory().total / 1024 / 1024:.0f}MB",
            "memory_available": f"{psutil.virtual_memory().available / 1024 / 1024:.0f}MB",
            "cpu_count": psutil.cpu_count(),
            "disk_free": f"{psutil.disk_usage('/').free / 1024 / 1024:.0f}MB"
        })
    except ImportError:
        pass
    
    return results


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check for GPU availability and CUDA support.
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "cuda_version": None
    }
    
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        
        if gpu_info["cuda_available"]:
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["gpu_names"] = [
                torch.cuda.get_device_name(i) 
                for i in range(gpu_info["gpu_count"])
            ]
            gpu_info["cuda_version"] = torch.version.cuda
    
    except ImportError:
        pass
    
    return gpu_info


def validate_mcp_config(mcp_config_path: Optional[str] = None) -> List[str]:
    """
    Validate MCP server configuration.
    
    Args:
        mcp_config_path: Path to MCP configuration file
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if mcp_config_path is None:
        mcp_config_path = ".roo/mcp.json"
    
    config_file = Path(mcp_config_path)
    
    if not config_file.exists():
        errors.append(f"MCP configuration file not found: {mcp_config_path}")
        return errors
    
    try:
        import json
        with open(config_file, 'r') as f:
            mcp_config = json.load(f)
        
        if "mcpServers" not in mcp_config:
            errors.append("MCP configuration missing 'mcpServers' section")
            return errors
        
        servers = mcp_config["mcpServers"]
        if not isinstance(servers, dict):
            errors.append("MCP 'mcpServers' must be a dictionary")
            return errors
        
        for server_name, server_config in servers.items():
            if not isinstance(server_config, dict):
                errors.append(f"MCP server '{server_name}' configuration must be a dictionary")
                continue
            
            if "command" not in server_config:
                errors.append(f"MCP server '{server_name}' missing 'command' field")
            
            if "args" in server_config and not isinstance(server_config["args"], list):
                errors.append(f"MCP server '{server_name}' 'args' must be a list")
    
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in MCP configuration: {str(e)}")
    except Exception as e:
        errors.append(f"Error validating MCP configuration: {str(e)}")
    
    return errors