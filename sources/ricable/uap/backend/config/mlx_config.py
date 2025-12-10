# File: backend/config/mlx_config.py
# MLX Configuration System - Model Selection and Performance Tuning
# Provides centralized configuration for MLX Apple Silicon inference

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a specific MLX model."""
    model_id: str
    model_path: str
    max_tokens: int
    memory_mb: int
    temperature: float
    description: str
    priority: str = "normal"  # high, normal, low
    auto_load: bool = False
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class PerformanceConfig:
    """Performance tuning configuration for MLX."""
    memory_limit_gb: int
    max_models: int
    memory_threshold: float
    cleanup_interval: int
    batch_size: int
    enable_memory_mapping: bool
    enable_quantization: bool
    cpu_threads: int
    
@dataclass
class CacheConfig:
    """Caching configuration for MLX models."""
    cache_dir: str
    max_cache_size_gb: float
    cache_strategy: str  # lru, lfu, ttl
    ttl_hours: int
    enable_persistent_cache: bool
    compression_enabled: bool

@dataclass
class MLXConfig:
    """Complete MLX configuration."""
    enabled: bool
    default_model: str
    fallback_enabled: bool
    auto_cleanup: bool
    performance: PerformanceConfig
    cache: CacheConfig
    models: Dict[str, ModelConfig]
    api_settings: Dict[str, Any]

class MLXConfigManager:
    """Manager for MLX configuration with hot-reloading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.config: Optional[MLXConfig] = None
        self.last_modified = None
        
        # Environment variable overrides
        self.env_overrides = self._collect_env_overrides()
        
        logger.info(f"MLX config manager initialized with path: {self.config_path}")
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        # Try multiple locations
        possible_paths = [
            Path.cwd() / "backend" / "config" / "mlx_config.json",
            Path.cwd() / "config" / "mlx_config.json",
            Path.home() / ".uap" / "mlx_config.json",
            Path("/etc/uap/mlx_config.json")
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Default to first location for creation
        return possible_paths[0]
    
    def _collect_env_overrides(self) -> Dict[str, Any]:
        """Collect configuration overrides from environment variables."""
        overrides = {}
        
        # Model selection
        if os.getenv("MLX_DEFAULT_MODEL"):
            overrides["default_model"] = os.getenv("MLX_DEFAULT_MODEL")
        
        # Performance settings
        if os.getenv("MLX_MEMORY_LIMIT"):
            overrides["performance.memory_limit_gb"] = int(os.getenv("MLX_MEMORY_LIMIT"))
        
        if os.getenv("MLX_MAX_MODELS"):
            overrides["performance.max_models"] = int(os.getenv("MLX_MAX_MODELS"))
        
        if os.getenv("MLX_MEMORY_THRESHOLD"):
            overrides["performance.memory_threshold"] = float(os.getenv("MLX_MEMORY_THRESHOLD"))
        
        # Cache settings
        if os.getenv("MLX_CACHE_DIR"):
            overrides["cache.cache_dir"] = os.getenv("MLX_CACHE_DIR")
        
        if os.getenv("MLX_CACHE_SIZE"):
            overrides["cache.max_cache_size_gb"] = float(os.getenv("MLX_CACHE_SIZE"))
        
        # Feature flags
        if os.getenv("MLX_ENABLED"):
            overrides["enabled"] = os.getenv("MLX_ENABLED").lower() == "true"
        
        if os.getenv("MLX_FALLBACK_ENABLED"):
            overrides["fallback_enabled"] = os.getenv("MLX_FALLBACK_ENABLED").lower() == "true"
        
        return overrides
    
    def load_config(self, force_reload: bool = False) -> MLXConfig:
        """Load or reload the MLX configuration."""
        try:
            # Check if reload is needed
            if not force_reload and self.config and not self._config_changed():
                return self.config
            
            logger.info(f"Loading MLX configuration from {self.config_path}")
            
            if self.config_path.exists():
                # Load from file
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                self.last_modified = self.config_path.stat().st_mtime
                self.config = self._parse_config(config_data)
            else:
                # Create default configuration
                logger.info("Configuration file not found. Creating default configuration.")
                self.config = self._create_default_config()
                self.save_config()
            
            # Apply environment overrides
            self._apply_overrides()
            
            # Validate configuration
            self._validate_config()
            
            logger.info("MLX configuration loaded successfully.")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load MLX configuration: {e}")
            # Return minimal working config
            return self._create_minimal_config()
    
    def _config_changed(self) -> bool:
        """Check if configuration file has been modified."""
        if not self.config_path.exists():
            return False
        
        current_mtime = self.config_path.stat().st_mtime
        return current_mtime != self.last_modified
    
    def _parse_config(self, config_data: Dict[str, Any]) -> MLXConfig:
        """Parse configuration data into MLXConfig object."""
        # Parse performance config
        perf_data = config_data.get("performance", {})
        performance = PerformanceConfig(
            memory_limit_gb=perf_data.get("memory_limit_gb", 8),
            max_models=perf_data.get("max_models", 3),
            memory_threshold=perf_data.get("memory_threshold", 0.8),
            cleanup_interval=perf_data.get("cleanup_interval", 3600),
            batch_size=perf_data.get("batch_size", 1),
            enable_memory_mapping=perf_data.get("enable_memory_mapping", True),
            enable_quantization=perf_data.get("enable_quantization", True),
            cpu_threads=perf_data.get("cpu_threads", 0)  # 0 = auto
        )
        
        # Parse cache config
        cache_data = config_data.get("cache", {})
        cache = CacheConfig(
            cache_dir=cache_data.get("cache_dir", str(Path.home() / ".cache" / "uap" / "mlx")),
            max_cache_size_gb=cache_data.get("max_cache_size_gb", 10.0),
            cache_strategy=cache_data.get("cache_strategy", "lru"),
            ttl_hours=cache_data.get("ttl_hours", 168),  # 1 week
            enable_persistent_cache=cache_data.get("enable_persistent_cache", True),
            compression_enabled=cache_data.get("compression_enabled", False)
        )
        
        # Parse model configs
        models_data = config_data.get("models", {})
        models = {}
        for model_id, model_data in models_data.items():
            models[model_id] = ModelConfig(
                model_id=model_id,
                model_path=model_data.get("model_path", ""),
                max_tokens=model_data.get("max_tokens", 1000),
                memory_mb=model_data.get("memory_mb", 2048),
                temperature=model_data.get("temperature", 0.7),
                description=model_data.get("description", ""),
                priority=model_data.get("priority", "normal"),
                auto_load=model_data.get("auto_load", False),
                tags=model_data.get("tags", [])
            )
        
        return MLXConfig(
            enabled=config_data.get("enabled", True),
            default_model=config_data.get("default_model", "llama-3.2-1b"),
            fallback_enabled=config_data.get("fallback_enabled", True),
            auto_cleanup=config_data.get("auto_cleanup", True),
            performance=performance,
            cache=cache,
            models=models,
            api_settings=config_data.get("api_settings", {})
        )
    
    def _create_default_config(self) -> MLXConfig:
        """Create a default MLX configuration."""
        default_models = {
            "llama-3.2-1b": ModelConfig(
                model_id="llama-3.2-1b",
                model_path="mlx-community/Llama-3.2-1B-Instruct-4bit",
                max_tokens=1000,
                memory_mb=2048,
                temperature=0.7,
                description="Compact Llama 3.2 1B model for fast inference",
                priority="high",
                auto_load=True,
                tags=["small", "fast", "general"]
            ),
            "llama-3.2-3b": ModelConfig(
                model_id="llama-3.2-3b",
                model_path="mlx-community/Llama-3.2-3B-Instruct-4bit",
                max_tokens=2000,
                memory_mb=4096,
                temperature=0.7,
                description="Llama 3.2 3B model for balanced performance",
                priority="normal",
                auto_load=False,
                tags=["medium", "balanced", "general"]
            ),
            "phi-3-mini": ModelConfig(
                model_id="phi-3-mini",
                model_path="mlx-community/Phi-3-mini-4k-instruct-4bit",
                max_tokens=1000,
                memory_mb=2048,
                temperature=0.3,
                description="Microsoft Phi-3 mini model for coding tasks",
                priority="normal",
                auto_load=False,
                tags=["coding", "small", "microsoft"]
            ),
            "gemma-2b": ModelConfig(
                model_id="gemma-2b",
                model_path="mlx-community/gemma-2b-it-4bit",
                max_tokens=1000,
                memory_mb=2048,
                temperature=0.7,
                description="Google Gemma 2B instruction-tuned model",
                priority="low",
                auto_load=False,
                tags=["google", "small", "instruction"]
            )
        }
        
        performance = PerformanceConfig(
            memory_limit_gb=8,
            max_models=3,
            memory_threshold=0.8,
            cleanup_interval=3600,
            batch_size=1,
            enable_memory_mapping=True,
            enable_quantization=True,
            cpu_threads=0
        )
        
        cache = CacheConfig(
            cache_dir=str(Path.home() / ".cache" / "uap" / "mlx"),
            max_cache_size_gb=10.0,
            cache_strategy="lru",
            ttl_hours=168,
            enable_persistent_cache=True,
            compression_enabled=False
        )
        
        return MLXConfig(
            enabled=True,
            default_model="llama-3.2-1b",
            fallback_enabled=True,
            auto_cleanup=True,
            performance=performance,
            cache=cache,
            models=default_models,
            api_settings={
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "stream_responses": False
            }
        )
    
    def _create_minimal_config(self) -> MLXConfig:
        """Create minimal configuration for error recovery."""
        performance = PerformanceConfig(
            memory_limit_gb=4,
            max_models=1,
            memory_threshold=0.9,
            cleanup_interval=1800,
            batch_size=1,
            enable_memory_mapping=False,
            enable_quantization=True,
            cpu_threads=1
        )
        
        cache = CacheConfig(
            cache_dir="/tmp/uap_mlx_cache",
            max_cache_size_gb=2.0,
            cache_strategy="lru",
            ttl_hours=24,
            enable_persistent_cache=False,
            compression_enabled=False
        )
        
        return MLXConfig(
            enabled=False,  # Disabled by default in minimal config
            default_model="llama-3.2-1b",
            fallback_enabled=True,
            auto_cleanup=True,
            performance=performance,
            cache=cache,
            models={},
            api_settings={}
        )
    
    def _apply_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        for key, value in self.env_overrides.items():
            self._set_nested_value(self.config, key, value)
    
    def _set_nested_value(self, obj: Any, key: str, value: Any) -> None:
        """Set a nested value using dot notation."""
        keys = key.split('.')
        current = obj
        
        for k in keys[:-1]:
            if hasattr(current, k):
                current = getattr(current, k)
            else:
                return  # Skip if path doesn't exist
        
        if hasattr(current, keys[-1]):
            setattr(current, keys[-1], value)
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        if not self.config:
            raise ValueError("Configuration is None")
        
        # Validate performance settings
        if self.config.performance.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")
        
        if self.config.performance.max_models <= 0:
            raise ValueError("Max models must be positive")
        
        if not 0 < self.config.performance.memory_threshold <= 1:
            raise ValueError("Memory threshold must be between 0 and 1")
        
        # Validate cache settings
        cache_dir = Path(self.config.cache.cache_dir)
        if not cache_dir.parent.exists():
            logger.warning(f"Cache directory parent does not exist: {cache_dir.parent}")
        
        # Validate models
        if self.config.default_model not in self.config.models:
            logger.warning(f"Default model '{self.config.default_model}' not found in model configs")
        
        logger.info("Configuration validation passed.")
    
    def save_config(self) -> bool:
        """Save the current configuration to file."""
        try:
            if not self.config:
                logger.error("No configuration to save")
                return False
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict
            config_dict = self._config_to_dict()
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert MLXConfig to dictionary for serialization."""
        config_dict = {
            "enabled": self.config.enabled,
            "default_model": self.config.default_model,
            "fallback_enabled": self.config.fallback_enabled,
            "auto_cleanup": self.config.auto_cleanup,
            "performance": asdict(self.config.performance),
            "cache": asdict(self.config.cache),
            "models": {},
            "api_settings": self.config.api_settings,
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "generator": "UAP MLX Config Manager"
            }
        }
        
        # Convert models
        for model_id, model_config in self.config.models.items():
            config_dict["models"][model_id] = asdict(model_config)
        
        return config_dict
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        if not self.config:
            self.load_config()
        
        return self.config.models.get(model_id)
    
    def add_model_config(self, model_config: ModelConfig) -> bool:
        """Add a new model configuration."""
        try:
            if not self.config:
                self.load_config()
            
            self.config.models[model_config.model_id] = model_config
            logger.info(f"Added model configuration: {model_config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model configuration: {e}")
            return False
    
    def remove_model_config(self, model_id: str) -> bool:
        """Remove a model configuration."""
        try:
            if not self.config:
                self.load_config()
            
            if model_id in self.config.models:
                del self.config.models[model_id]
                logger.info(f"Removed model configuration: {model_id}")
                return True
            else:
                logger.warning(f"Model configuration not found: {model_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove model configuration: {e}")
            return False
    
    def update_performance_config(self, **kwargs) -> bool:
        """Update performance configuration."""
        try:
            if not self.config:
                self.load_config()
            
            for key, value in kwargs.items():
                if hasattr(self.config.performance, key):
                    setattr(self.config.performance, key, value)
                    logger.info(f"Updated performance config: {key} = {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update performance configuration: {e}")
            return False
    
    def get_models_by_tag(self, tag: str) -> List[ModelConfig]:
        """Get all models with a specific tag."""
        if not self.config:
            self.load_config()
        
        return [
            model for model in self.config.models.values()
            if tag in model.tags
        ]
    
    def get_auto_load_models(self) -> List[str]:
        """Get list of models that should be auto-loaded."""
        if not self.config:
            self.load_config()
        
        return [
            model_id for model_id, model in self.config.models.items()
            if model.auto_load
        ]

# Global configuration manager instance
_config_manager = None

def get_mlx_config_manager() -> MLXConfigManager:
    """Get the global MLX configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = MLXConfigManager()
    return _config_manager

def get_mlx_config() -> MLXConfig:
    """Get the current MLX configuration."""
    return get_mlx_config_manager().load_config()

print("MLX configuration system loaded.")