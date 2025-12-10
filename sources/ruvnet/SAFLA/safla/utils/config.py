"""
SAFLA Configuration Management

This module provides configuration management for the SAFLA system,
including environment variable handling, default values, and validation.

This module maintains backward compatibility with the original dataclass-based
configuration while providing access to the new Pydantic-based system.
"""

import os
import json
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Import the new Pydantic-based configuration system
try:
    from .pydantic_config import SAFLAConfig as PydanticSAFLAConfig
    from .config_loader import ConfigLoader
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    warnings.warn(
        "Pydantic configuration system not available. "
        "Install pydantic and pydantic-settings for enhanced features.",
        ImportWarning
    )

# Note: Environment variables are now loaded explicitly by ConfigLoader when needed
# Removed global load_dotenv() to prevent automatic environment variable loading


@dataclass
class MemoryConfig:
    """Configuration for memory components (Legacy - use PydanticSAFLAConfig for new features)."""
    vector_dimensions: List[int] = field(default_factory=lambda: [512, 768, 1024, 1536])
    max_memories: int = 10000
    similarity_threshold: float = 0.8
    consolidation_interval: int = 3600  # seconds
    cleanup_threshold: float = 0.1


@dataclass
class SafetyConfig:
    """Configuration for safety validation (Legacy - use PydanticSAFLAConfig for new features)."""
    memory_limit: int = 1000000000  # 1GB in bytes
    cpu_limit: float = 0.9
    monitoring_interval: float = 1.0  # seconds
    emergency_stop_threshold: float = 0.95
    rollback_enabled: bool = True


@dataclass
class MCPConfig:
    """Configuration for MCP orchestration (Legacy - use PydanticSAFLAConfig for new features)."""
    timeout: int = 30  # seconds
    max_retries: int = 3
    health_check_interval: int = 60  # seconds
    connection_pool_size: int = 10
    circuit_breaker_threshold: int = 5


@dataclass
class MetaCognitiveConfig:
    """Configuration for meta-cognitive engine (Legacy - use PydanticSAFLAConfig for new features)."""
    adaptation_rate: float = 0.1
    goal_priority_threshold: float = 0.5
    strategy_selection_timeout: int = 10  # seconds
    performance_window_size: int = 100
    self_awareness_interval: float = 5.0  # seconds


@dataclass
class SAFLAConfig:
    """Main SAFLA configuration class (Legacy - use PydanticSAFLAConfig for new features)."""
    
    # Component configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    metacognitive: MetaCognitiveConfig = field(default_factory=MetaCognitiveConfig)
    
    # General settings
    debug: bool = False
    log_level: str = "INFO"
    enable_monitoring: bool = True
    data_dir: str = "./data"
    config_dir: str = "./.roo"
    
    def __post_init__(self):
        """Issue deprecation warning for legacy configuration."""
        warnings.warn(
            "The dataclass-based SAFLAConfig is deprecated. "
            "Use safla.utils.pydantic_config.SAFLAConfig for enhanced features, "
            "validation, and security.",
            DeprecationWarning,
            stacklevel=2
        )
    
    @classmethod
    def from_env(cls) -> "SAFLAConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Memory configuration
        if vector_dims := os.getenv("SAFLA_VECTOR_DIMENSIONS"):
            config.memory.vector_dimensions = [int(d) for d in vector_dims.split(",")]
        
        config.memory.max_memories = int(os.getenv("SAFLA_MAX_MEMORIES", config.memory.max_memories))
        config.memory.similarity_threshold = float(os.getenv("SAFLA_SIMILARITY_THRESHOLD", config.memory.similarity_threshold))
        
        # Safety configuration
        config.safety.memory_limit = int(os.getenv("SAFLA_MEMORY_LIMIT", config.safety.memory_limit))
        config.safety.cpu_limit = float(os.getenv("SAFLA_CPU_LIMIT", config.safety.cpu_limit))
        config.safety.monitoring_interval = float(os.getenv("SAFLA_SAFETY_MONITORING_INTERVAL", config.safety.monitoring_interval))
        
        # MCP configuration
        config.mcp.timeout = int(os.getenv("SAFLA_MCP_TIMEOUT", config.mcp.timeout))
        config.mcp.max_retries = int(os.getenv("SAFLA_MCP_MAX_RETRIES", config.mcp.max_retries))
        config.mcp.health_check_interval = int(os.getenv("SAFLA_MCP_HEALTH_CHECK_INTERVAL", config.mcp.health_check_interval))
        
        # General settings
        config.debug = os.getenv("SAFLA_DEBUG", "false").lower() == "true"
        config.log_level = os.getenv("SAFLA_LOG_LEVEL", config.log_level)
        config.enable_monitoring = os.getenv("SAFLA_ENABLE_MONITORING", "true").lower() == "true"
        config.data_dir = os.getenv("SAFLA_DATA_DIR", config.data_dir)
        config.config_dir = os.getenv("SAFLA_CONFIG_DIR", config.config_dir)
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> "SAFLAConfig":
        """Load configuration from a JSON file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAFLAConfig":
        """Create configuration from a dictionary."""
        config = cls()
        
        # Update memory config
        if "memory" in config_dict:
            memory_data = config_dict["memory"]
            for key, value in memory_data.items():
                if hasattr(config.memory, key):
                    setattr(config.memory, key, value)
        
        # Update safety config
        if "safety" in config_dict:
            safety_data = config_dict["safety"]
            for key, value in safety_data.items():
                if hasattr(config.safety, key):
                    setattr(config.safety, key, value)
        
        # Update MCP config
        if "mcp" in config_dict:
            mcp_data = config_dict["mcp"]
            for key, value in mcp_data.items():
                if hasattr(config.mcp, key):
                    setattr(config.mcp, key, value)
        
        # Update meta-cognitive config
        if "metacognitive" in config_dict:
            meta_data = config_dict["metacognitive"]
            for key, value in meta_data.items():
                if hasattr(config.metacognitive, key):
                    setattr(config.metacognitive, key, value)
        
        # Update general settings
        for key in ["debug", "log_level", "enable_monitoring", "data_dir", "config_dir"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "memory": {
                "vector_dimensions": self.memory.vector_dimensions,
                "max_memories": self.memory.max_memories,
                "similarity_threshold": self.memory.similarity_threshold,
                "consolidation_interval": self.memory.consolidation_interval,
                "cleanup_threshold": self.memory.cleanup_threshold
            },
            "safety": {
                "memory_limit": self.safety.memory_limit,
                "cpu_limit": self.safety.cpu_limit,
                "monitoring_interval": self.safety.monitoring_interval,
                "emergency_stop_threshold": self.safety.emergency_stop_threshold,
                "rollback_enabled": self.safety.rollback_enabled
            },
            "mcp": {
                "timeout": self.mcp.timeout,
                "max_retries": self.mcp.max_retries,
                "health_check_interval": self.mcp.health_check_interval,
                "connection_pool_size": self.mcp.connection_pool_size,
                "circuit_breaker_threshold": self.mcp.circuit_breaker_threshold
            },
            "metacognitive": {
                "adaptation_rate": self.metacognitive.adaptation_rate,
                "goal_priority_threshold": self.metacognitive.goal_priority_threshold,
                "strategy_selection_timeout": self.metacognitive.strategy_selection_timeout,
                "performance_window_size": self.metacognitive.performance_window_size,
                "self_awareness_interval": self.metacognitive.self_awareness_interval
            },
            "debug": self.debug,
            "log_level": self.log_level,
            "enable_monitoring": self.enable_monitoring,
            "data_dir": self.data_dir,
            "config_dir": self.config_dir
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to a JSON file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate memory configuration
        if not self.memory.vector_dimensions:
            errors.append("Memory vector_dimensions cannot be empty")
        
        if self.memory.max_memories <= 0:
            errors.append("Memory max_memories must be positive")
        
        if not 0 <= self.memory.similarity_threshold <= 1:
            errors.append("Memory similarity_threshold must be between 0 and 1")
        
        # Validate safety configuration
        if self.safety.memory_limit <= 0:
            errors.append("Safety memory_limit must be positive")
        
        if not 0 <= self.safety.cpu_limit <= 1:
            errors.append("Safety cpu_limit must be between 0 and 1")
        
        if self.safety.monitoring_interval <= 0:
            errors.append("Safety monitoring_interval must be positive")
        
        # Validate MCP configuration
        if self.mcp.timeout <= 0:
            errors.append("MCP timeout must be positive")
        
        if self.mcp.max_retries < 0:
            errors.append("MCP max_retries cannot be negative")
        
        # Validate meta-cognitive configuration
        if not 0 <= self.metacognitive.adaptation_rate <= 1:
            errors.append("MetaCognitive adaptation_rate must be between 0 and 1")
        
        return errors
    
    def create_directories(self) -> None:
        """Create necessary directories for SAFLA operation."""
        directories = [
            self.data_dir,
            self.config_dir,
            f"{self.data_dir}/memory",
            f"{self.data_dir}/checkpoints",
            f"{self.data_dir}/logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_global_config: Optional[SAFLAConfig] = None
_global_pydantic_config: Optional["PydanticSAFLAConfig"] = None


def get_config(use_pydantic: bool = True) -> SAFLAConfig:
    """Get the global SAFLA configuration.
    
    Args:
        use_pydantic: If True and available, use the Pydantic-based configuration.
                     If False, use the legacy dataclass-based configuration.
    
    Returns:
        SAFLAConfig: The global configuration instance.
    """
    global _global_config, _global_pydantic_config
    
    if use_pydantic and PYDANTIC_AVAILABLE:
        if _global_pydantic_config is None:
            _global_pydantic_config = PydanticSAFLAConfig()
        return _global_pydantic_config
    else:
        if _global_config is None:
            _global_config = SAFLAConfig.from_env()
        return _global_config


def set_config(config: SAFLAConfig) -> None:
    """Set the global SAFLA configuration."""
    global _global_config, _global_pydantic_config
    
    if PYDANTIC_AVAILABLE and isinstance(config, PydanticSAFLAConfig):
        _global_pydantic_config = config
    else:
        _global_config = config


def reset_config() -> None:
    """Reset the global configuration to default."""
    global _global_config, _global_pydantic_config
    _global_config = None
    _global_pydantic_config = None


def get_pydantic_config() -> "PydanticSAFLAConfig":
    """Get the Pydantic-based configuration.
    
    Returns:
        PydanticSAFLAConfig: The Pydantic configuration instance.
        
    Raises:
        ImportError: If Pydantic is not available.
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic configuration system not available. "
            "Install pydantic and pydantic-settings: pip install pydantic pydantic-settings"
        )
    
    global _global_pydantic_config
    if _global_pydantic_config is None:
        _global_pydantic_config = PydanticSAFLAConfig()
    return _global_pydantic_config


def load_config_from_file(config_path: str, use_pydantic: bool = True) -> SAFLAConfig:
    """Load configuration from file.
    
    Args:
        config_path: Path to the configuration file.
        use_pydantic: Whether to use the Pydantic-based configuration system.
        
    Returns:
        SAFLAConfig: The loaded configuration.
    """
    if use_pydantic and PYDANTIC_AVAILABLE:
        loader = ConfigLoader()
        config = loader.load_config(config_file=config_path)
        set_config(config)
        return config
    else:
        config = SAFLAConfig.from_file(config_path)
        set_config(config)
        return config


# Convenience functions for migration
def migrate_to_pydantic() -> "PydanticSAFLAConfig":
    """Migrate from legacy configuration to Pydantic-based configuration.
    
    Returns:
        PydanticSAFLAConfig: The migrated configuration.
        
    Raises:
        ImportError: If Pydantic is not available.
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic configuration system not available. "
            "Install pydantic and pydantic-settings: pip install pydantic pydantic-settings"
        )
    
    # Get current legacy config
    legacy_config = get_config(use_pydantic=False)
    
    # Convert to dict and create Pydantic config
    config_dict = legacy_config.to_dict()
    pydantic_config = PydanticSAFLAConfig(**config_dict)
    
    # Set as global config
    set_config(pydantic_config)
    
    return pydantic_config


# Export the Pydantic configuration classes if available
if PYDANTIC_AVAILABLE:
    from .pydantic_config import (
        SAFLAConfig as PydanticSAFLAConfig,
        PerformanceConfig,
        IntegrationConfig,
        SecurityConfig,
        NetworkingConfig,
        DevelopmentConfig,
        ExperimentalConfig,
        MonitoringConfig,
    )
    from .config_loader import ConfigLoader, ConfigurationError
    
    __all__ = [
        "SAFLAConfig", "MemoryConfig", "SafetyConfig", "MCPConfig", "MetaCognitiveConfig",
        "PydanticSAFLAConfig", "PerformanceConfig", "IntegrationConfig", "SecurityConfig",
        "NetworkingConfig", "DevelopmentConfig", "ExperimentalConfig", "MonitoringConfig",
        "ConfigLoader", "ConfigurationError",
        "get_config", "set_config", "reset_config", "get_pydantic_config",
        "load_config_from_file", "migrate_to_pydantic"
    ]
else:
    __all__ = [
        "SAFLAConfig", "MemoryConfig", "SafetyConfig", "MCPConfig", "MetaCognitiveConfig",
        "get_config", "set_config", "reset_config", "load_config_from_file"
    ]