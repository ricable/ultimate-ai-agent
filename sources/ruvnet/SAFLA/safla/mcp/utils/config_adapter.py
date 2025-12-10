"""
Configuration adapter for smooth migration from dataclass to Pydantic config.

This module provides a compatibility layer to help transition the codebase
from the legacy dataclass configuration to the Pydantic-based configuration.
"""

from typing import Any, Union
import logging
from pathlib import Path

from safla.utils.config import SAFLAConfig as DataclassConfig
from safla.utils.pydantic_config import SAFLAConfig as PydanticConfig
from safla.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ConfigAdapter:
    """
    Adapter to provide a unified interface for both configuration systems.
    
    This allows gradual migration from dataclass to Pydantic configuration
    without breaking existing code.
    """
    
    def __init__(self, use_pydantic: bool = True, config_path: Optional[str] = None):
        """
        Initialize the configuration adapter.
        
        Args:
            use_pydantic: Whether to use Pydantic configuration (recommended)
            config_path: Optional path to configuration file
        """
        self.use_pydantic = use_pydantic
        self._config = None
        self._loader = None
        
        if use_pydantic:
            self._initialize_pydantic_config(config_path)
        else:
            self._initialize_dataclass_config()
    
    def _initialize_pydantic_config(self, config_path: Optional[str]) -> None:
        """Initialize Pydantic-based configuration."""
        try:
            if config_path:
                # Use ConfigLoader for file-based config
                self._loader = ConfigLoader(config_path)
                config_dict = self._loader.load_config()
                self._config = PydanticConfig(**config_dict)
            else:
                # Use default Pydantic config with env vars
                self._config = PydanticConfig()
            
            logger.info("Initialized Pydantic configuration")
        except Exception as e:
            logger.error(f"Failed to initialize Pydantic config: {str(e)}")
            logger.warning("Falling back to dataclass configuration")
            self._initialize_dataclass_config()
    
    def _initialize_dataclass_config(self) -> None:
        """Initialize legacy dataclass configuration."""
        from safla.utils.config import get_config
        self._config = get_config(use_pydantic=False)
        logger.warning("Using legacy dataclass configuration (deprecated)")
    
    @property
    def config(self) -> Union[DataclassConfig, PydanticConfig]:
        """Get the underlying configuration object."""
        return self._config
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.
        
        Args:
            path: Dot-separated path (e.g., "memory.vector_store.dimension")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        try:
            value = self._config
            for part in path.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        except Exception:
            return default
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        if self.use_pydantic and hasattr(self._config, 'dict'):
            return self._config.dict()
        elif hasattr(self._config, '__dict__'):
            return self._serialize_dataclass(self._config)
        else:
            return {}
    
    def _serialize_dataclass(self, obj: Any) -> dict:
        """Recursively serialize dataclass to dict."""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._serialize_dataclass(value)
                elif isinstance(value, list):
                    result[key] = [
                        self._serialize_dataclass(item) if hasattr(item, '__dict__') else item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    result[key] = {
                        k: self._serialize_dataclass(v) if hasattr(v, '__dict__') else v
                        for k, v in value.items()
                    }
                else:
                    result[key] = value
            return result
        return obj
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid
        """
        if self.use_pydantic:
            # Pydantic validates on instantiation
            return self._config is not None
        else:
            # Basic validation for dataclass
            from safla.utils.validation import validate_config
            try:
                validate_config(self._config)
                return True
            except Exception:
                return False
    
    def reload(self) -> None:
        """Reload configuration from source."""
        if self._loader:
            config_dict = self._loader.load_config()
            self._config = PydanticConfig(**config_dict)
        else:
            # Re-initialize from environment
            if self.use_pydantic:
                self._config = PydanticConfig()
            else:
                from safla.utils.config import get_config
                self._config = get_config(use_pydantic=False)
    
    @classmethod
    def create_default(cls) -> 'ConfigAdapter':
        """Create adapter with default settings (Pydantic-based)."""
        return cls(use_pydantic=True)
    
    @classmethod
    def create_legacy(cls) -> 'ConfigAdapter':
        """Create adapter with legacy dataclass config."""
        return cls(use_pydantic=False)


def get_unified_config(prefer_pydantic: bool = True, 
                      config_path: Optional[str] = None) -> ConfigAdapter:
    """
    Get a unified configuration adapter.
    
    This is the recommended way to get configuration in the SAFLA system
    during the migration period.
    
    Args:
        prefer_pydantic: Whether to prefer Pydantic config (default: True)
        config_path: Optional configuration file path
        
    Returns:
        ConfigAdapter instance
    """
    return ConfigAdapter(use_pydantic=prefer_pydantic, config_path=config_path)


# Compatibility exports
__all__ = [
    'ConfigAdapter',
    'get_unified_config'
]