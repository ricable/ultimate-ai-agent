# File: backend/processors/mlx_processor.py
# MLX Processor for Apple Silicon Local Inference
# High-performance local AI models using Apple's MLX framework

import asyncio
import json
import logging
import os
import platform
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Import configuration system
from ..config.mlx_config import get_mlx_config, get_mlx_config_manager

# Configure logging for MLX
logger = logging.getLogger(__name__)

class MLXProcessor:
    """High-performance MLX processor for Apple Silicon local inference.
    
    This class provides local AI inference capabilities using Apple's MLX framework,
    optimized for Apple Silicon hardware (M1, M2, M3, etc.).
    """
    
    def __init__(self):
        self.framework_name = "MLX"
        self.is_initialized = False
        self.is_apple_silicon = self._check_apple_silicon()
        self.status = "initializing"
        self.models = {}
        self.model_cache = {}
        self.error_count = 0
        self.last_error = None
        
        # Load configuration
        try:
            self.config = get_mlx_config()
            self.config_manager = get_mlx_config_manager()
            logger.info("MLX configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load MLX configuration: {e}. Using defaults.")
            self.config = None
            self.config_manager = None
        
        # Set configuration values (with fallbacks)
        if self.config:
            self.cache_dir = Path(self.config.cache.cache_dir)
            self.max_cache_size = self.config.performance.max_models
            self.default_model = self.config.default_model
            self.memory_limit = self.config.performance.memory_limit_gb
            self.batch_size = self.config.performance.batch_size
            self.model_configs = self._build_model_configs_from_config()
        else:
            # Fallback to environment variables and defaults
            self.cache_dir = Path(os.getenv("MLX_CACHE_DIR", "~/.cache/mlx")).expanduser()
            self.max_cache_size = int(os.getenv("MLX_MAX_CACHE_SIZE", "3"))
            self.default_model = os.getenv("MLX_DEFAULT_MODEL", "llama-3.2-1b")
            self.memory_limit = int(os.getenv("MLX_MEMORY_LIMIT", "8"))
            self.batch_size = int(os.getenv("MLX_BATCH_SIZE", "1"))
            self.model_configs = self._build_default_model_configs()
        
        logger.info(f"MLX processor initialized. Apple Silicon: {self.is_apple_silicon}, Config loaded: {self.config is not None}")
    
    def _check_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        try:
            return platform.system() == "Darwin" and platform.processor() == "arm"
        except Exception:
            return False
    
    async def initialize(self) -> None:
        """Initialize MLX framework and load default model."""
        try:
            logger.info("Initializing MLX framework...")
            
            if not self.is_apple_silicon:
                logger.warning("MLX is optimized for Apple Silicon. Performance may be limited.")
                self.status = "limited"
            
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to import MLX
            try:
                import mlx.core as mx
                import mlx.nn as nn
                self.mx = mx
                self.nn = nn
                logger.info("MLX framework imported successfully.")
            except ImportError as e:
                logger.error(f"MLX framework not available: {e}")
                self.status = "error"
                self.last_error = "MLX framework not installed"
                return
            
            # Initialize model loader
            await self._initialize_model_loader()
            
            # Load default model if specified
            if self.default_model:
                await self._load_model(self.default_model)
            
            self.is_initialized = True
            self.status = "active"
            logger.info("MLX framework initialization complete.")
            
        except Exception as e:
            self.status = "error"
            self.last_error = str(e)
            logger.error(f"MLX initialization failed: {e}")
            raise
    
    async def _initialize_model_loader(self) -> None:
        """Initialize the model loading system."""
        try:
            # Try to import MLX LM
            try:
                import mlx_lm
                self.mlx_lm = mlx_lm
                logger.info("MLX LM package imported successfully.")
            except ImportError:
                logger.warning("MLX LM package not available. Using fallback loader.")
                self.mlx_lm = None
            
            # Initialize model cache
            self._init_model_cache()
            
        except Exception as e:
            logger.error(f"Model loader initialization failed: {e}")
            raise
    
    def _init_model_cache(self) -> None:
        """Initialize the model cache system."""
        cache_info_file = self.cache_dir / "cache_info.json"
        
        try:
            if cache_info_file.exists():
                with open(cache_info_file, 'r') as f:
                    cache_info = json.load(f)
                    self.model_cache = cache_info.get("models", {})
                    logger.info(f"Loaded cache info for {len(self.model_cache)} models.")
            else:
                self.model_cache = {}
                logger.info("Initialized empty model cache.")
        except Exception as e:
            logger.warning(f"Failed to load cache info: {e}")
            self.model_cache = {}
    
    async def _load_model(self, model_identifier: str, force_reload: bool = False) -> bool:
        """Load a model into memory.
        
        Args:
            model_identifier: Model name or path
            force_reload: Force reload even if cached
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_identifier in self.models and not force_reload:
                logger.info(f"Model {model_identifier} already loaded.")
                return True
            
            logger.info(f"Loading MLX model: {model_identifier}")
            start_time = time.time()
            
            # Check if we have MLX LM available
            if self.mlx_lm:
                try:
                    # Use MLX LM for model loading
                    model, tokenizer = self.mlx_lm.load(model_identifier)
                    
                    self.models[model_identifier] = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "loaded_at": datetime.utcnow(),
                        "load_time": time.time() - start_time,
                        "usage_count": 0
                    }
                    
                    # Update cache info
                    self.model_cache[model_identifier] = {
                        "loaded_at": datetime.utcnow().isoformat(),
                        "load_time": time.time() - start_time,
                        "memory_usage": self._estimate_model_memory(model_identifier)
                    }
                    
                    logger.info(f"Model {model_identifier} loaded in {time.time() - start_time:.2f}s")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to load model with MLX LM: {e}")
                    return False
            else:
                # Fallback: simulate model loading
                logger.warning(f"MLX LM not available. Simulating model load for {model_identifier}")
                
                # Simulate loading time
                await asyncio.sleep(2)
                
                self.models[model_identifier] = {
                    "model": f"mock_model_{model_identifier}",
                    "tokenizer": f"mock_tokenizer_{model_identifier}",
                    "loaded_at": datetime.utcnow(),
                    "load_time": time.time() - start_time,
                    "usage_count": 0,
                    "mock": True
                }
                
                logger.info(f"Mock model {model_identifier} loaded.")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load model {model_identifier}: {e}")
            self.error_count += 1
            self.last_error = str(e)
            return False
    
    def _estimate_model_memory(self, model_identifier: str) -> int:
        """Estimate memory usage for a model in MB."""
        config = self.model_configs.get(model_identifier.split('/')[-1], {})
        return config.get("memory_mb", 2048)
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message using local MLX inference.
        
        Args:
            message: The user's input message
            context: Additional context for processing
            
        Returns:
            Dict containing 'content' and 'metadata' keys
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Determine which model to use
            model_name = context.get("model", self.default_model)
            
            # Load model if not already loaded
            if model_name not in self.models:
                loaded = await self._load_model(model_name)
                if not loaded:
                    return self._generate_error_response(f"Failed to load model: {model_name}")
            
            # Generate response
            start_time = time.time()
            response_content = await self._generate_response(message, model_name, context)
            inference_time = time.time() - start_time
            
            # Update usage stats
            self.models[model_name]["usage_count"] += 1
            
            return {
                "content": response_content,
                "metadata": {
                    "source": self.framework_name,
                    "model": model_name,
                    "inference_time": inference_time,
                    "apple_silicon": self.is_apple_silicon,
                    "local_inference": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "usage_count": self.models[model_name]["usage_count"]
                }
            }
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"MLX processing error: {e}")
            
            return self._generate_error_response(str(e))
    
    async def _generate_response(self, message: str, model_name: str, context: Dict[str, Any]) -> str:
        """Generate response using the specified model.
        
        Args:
            message: User message
            model_name: Model to use for inference
            context: Additional context
            
        Returns:
            Generated response
        """
        model_info = self.models.get(model_name)
        if not model_info:
            return "Model not available for inference."
        
        # Check if this is a mock model
        if model_info.get("mock", False):
            return self._generate_mock_response(message, model_name)
        
        try:
            # Real MLX inference
            if self.mlx_lm:
                model = model_info["model"]
                tokenizer = model_info["tokenizer"]
                
                # Prepare prompt
                prompt = self._prepare_prompt(message, context)
                
                # Generate response
                response = self.mlx_lm.generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                return response
            else:
                return self._generate_mock_response(message, model_name)
                
        except Exception as e:
            logger.error(f"MLX inference error: {e}")
            return self._generate_mock_response(message, model_name)
    
    def _prepare_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Prepare the prompt for the model."""
        system_prompt = context.get("system_prompt", 
            "You are a helpful AI assistant running locally on Apple Silicon using MLX. "
            "Provide clear, concise, and helpful responses.")
        
        # Format as instruction-following prompt
        return f"<|system|>\n{system_prompt}\n<|user|>\n{message}\n<|assistant|>\n"
    
    def _generate_mock_response(self, message: str, model_name: str) -> str:
        """Generate a mock response when MLX is not available."""
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ["code", "function", "programming"]):
            return f"I'd help you with coding tasks using the {model_name} model running locally on Apple Silicon. While I'm in mock mode, I can simulate responses for development purposes."
        
        elif "local" in message_lower or "mlx" in message_lower:
            return f"I'm running locally using Apple's MLX framework on Apple Silicon hardware. This provides fast, private inference without cloud dependencies. Model: {model_name}"
        
        elif any(keyword in message_lower for keyword in ["performance", "speed", "fast"]):
            return f"MLX provides optimized performance on Apple Silicon with unified memory architecture. The {model_name} model is designed for efficient local inference."
        
        else:
            return f"Hello! I'm responding using the {model_name} model via MLX on Apple Silicon. While in development mode, I can simulate local AI responses. What would you like to know?"
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "content": f"I encountered an error with local MLX inference: {error_message}. Please try again or check the system configuration.",
            "metadata": {
                "source": self.framework_name,
                "error": True,
                "error_message": error_message,
                "apple_silicon": self.is_apple_silicon,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the MLX processor."""
        return {
            "status": self.status,
            "framework": self.framework_name,
            "initialized": self.is_initialized,
            "apple_silicon": self.is_apple_silicon,
            "models_loaded": list(self.models.keys()),
            "model_count": len(self.models),
            "default_model": self.default_model,
            "cache_dir": str(self.cache_dir),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "capabilities": [
                "local_inference",
                "apple_silicon_optimization",
                "model_caching",
                "memory_management",
                "offline_processing"
            ],
            "available_models": list(self.model_configs.keys()),
            "memory_limit": self.memory_limit,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        for model_id, config in self.model_configs.items():
            models.append({
                "id": model_id,
                "path": config["model_path"],
                "max_tokens": config["max_tokens"],
                "memory_mb": config["memory_mb"],
                "description": config["description"],
                "loaded": config["model_path"] in self.models,
                "usage_count": self.models.get(config["model_path"], {}).get("usage_count", 0)
            })
        return models
    
    async def load_model(self, model_identifier: str) -> Dict[str, Any]:
        """Public method to load a model."""
        success = await self._load_model(model_identifier)
        return {
            "success": success,
            "model": model_identifier,
            "loaded_models": list(self.models.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def unload_model(self, model_identifier: str) -> Dict[str, Any]:
        """Unload a model from memory."""
        try:
            if model_identifier in self.models:
                del self.models[model_identifier]
                logger.info(f"Model {model_identifier} unloaded.")
                return {
                    "success": True,
                    "model": model_identifier,
                    "loaded_models": list(self.models.keys())
                }
            else:
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "model": model_identifier
                }
        except Exception as e:
            logger.error(f"Failed to unload model {model_identifier}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model_identifier
            }
    
    async def get_model_info(self, model_identifier: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_identifier in self.models:
            model_info = self.models[model_identifier]
            return {
                "model": model_identifier,
                "loaded": True,
                "loaded_at": model_info["loaded_at"].isoformat(),
                "load_time": model_info["load_time"],
                "usage_count": model_info["usage_count"],
                "mock": model_info.get("mock", False)
            }
        else:
            return {
                "model": model_identifier,
                "loaded": False,
                "available": model_identifier in [config["model_path"] for config in self.model_configs.values()]
            }
    
    async def cleanup(self) -> None:
        """Clean up MLX resources."""
        try:
            logger.info("Cleaning up MLX processor...")
            
            # Save cache info
            cache_info_file = self.cache_dir / "cache_info.json"
            try:
                with open(cache_info_file, 'w') as f:
                    json.dump({"models": self.model_cache}, f, indent=2)
                logger.info("Cache info saved.")
            except Exception as e:
                logger.warning(f"Failed to save cache info: {e}")
            
            # Clear models from memory
            self.models.clear()
            
            logger.info("MLX processor cleanup complete.")
            
        except Exception as e:
            logger.error(f"MLX cleanup error: {e}")

print("MLX processor module loaded.")