"""
Application settings and configuration management.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class AppSettings(BaseSettings):
    """Application configuration settings."""
    
    # Environment
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # HuggingFace
    hf_token: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    hf_model_cache_dir: str = Field(default="/tmp/hf_cache", env="HF_CACHE_DIR")
    
    # SAFLA Configuration
    safla_config_path: Optional[str] = Field(default=None, env="SAFLA_CONFIG_PATH")
    safla_memory_size: int = Field(default=1000, env="SAFLA_MEMORY_SIZE")
    safla_vector_dim: int = Field(default=768, env="SAFLA_VECTOR_DIM")
    
    # Performance
    max_concurrent_users: int = Field(default=10, env="MAX_CONCURRENT_USERS")
    cache_timeout: int = Field(default=300, env="CACHE_TIMEOUT")
    
    # Monitoring
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")
    
    # Application paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    assets_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "assets")
    
    @field_validator("safla_vector_dim")
    def validate_vector_dim(cls, v):
        """Validate vector dimensions are supported."""
        valid_dims = [512, 768, 1024, 1536]
        if v not in valid_dims:
            raise ValueError(f"Vector dimension must be one of {valid_dims}")
        return v
    
    @field_validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = ["development", "test", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    def get_safla_config(self) -> Dict[str, Any]:
        """Get SAFLA-specific configuration."""
        return {
            "memory_size": self.safla_memory_size,
            "vector_dimensions": self.safla_vector_dim,
            "config_path": self.safla_config_path,
            "environment": self.environment
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            "max_concurrent_users": self.max_concurrent_users,
            "cache_timeout": self.cache_timeout,
            "rate_limit": self.rate_limit_per_minute
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug or self.environment in ["development", "test"]
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow"
    )


# Global settings instance
settings = AppSettings()


# Performance optimization settings
PERFORMANCE_TARGETS = {
    "initial_load_time": 3.0,      # seconds
    "memory_search_time": 0.1,     # seconds
    "safety_validation_time": 0.05, # seconds
    "ui_response_time": 0.05,      # seconds
    "max_memory_usage": 512,       # MB
    "min_concurrent_users": 10,    # users
}


# UI Configuration
UI_CONFIG = {
    "theme": {
        "primary_color": "#3B82F6",
        "secondary_color": "#8B5CF6",
        "success_color": "#10B981",
        "warning_color": "#F59E0B",
        "error_color": "#EF4444",
        "font_family": "Inter, system-ui, sans-serif"
    },
    "layout": {
        "max_width": "1200px",
        "default_tab": "demo",
        "show_footer": True
    }
}