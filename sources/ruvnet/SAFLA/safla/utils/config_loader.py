"""
SAFLA Configuration Loader

This module provides utilities for loading and managing SAFLA configuration
from various sources including environment variables, files, and defaults.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dotenv import load_dotenv

from .pydantic_config import SAFLAConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an error in configuration loading or validation."""
    pass


class EnvironmentMapper:
    """Maps environment variables to nested configuration structure."""
    
    # Mapping of environment variables to nested config paths
    ENV_MAPPING = {
        # Top-level settings
        "SAFLA_DEBUG": "debug",
        "SAFLA_LOG_LEVEL": "log_level",
        "SAFLA_ENABLE_MONITORING": "enable_monitoring",
        "SAFLA_DATA_DIR": "data_dir",
        "SAFLA_CONFIG_DIR": "config_dir",
        
        # Performance settings
        "SAFLA_WORKER_THREADS": "performance.worker_threads",
        "SAFLA_BATCH_SIZE": "performance.batch_size",
        "SAFLA_MAX_CONCURRENT_OPS": "performance.max_concurrent_ops",
        "SAFLA_MEMORY_POOL_SIZE": "performance.memory_pool_size",
        "SAFLA_CACHE_SIZE": "performance.cache_size",
        "SAFLA_ENABLE_OPTIMIZATIONS": "performance.enable_optimizations",
        
        # Memory settings
        "SAFLA_VECTOR_DIMENSIONS": "memory.vector_dimensions",
        "SAFLA_MAX_MEMORIES": "memory.max_memories",
        "SAFLA_SIMILARITY_THRESHOLD": "memory.similarity_threshold",
        "SAFLA_MEMORY_CONSOLIDATION_INTERVAL": "memory.consolidation_interval",
        "SAFLA_MEMORY_CLEANUP_THRESHOLD": "memory.cleanup_threshold",
        "SAFLA_MEMORY_COMPRESSION": "memory.compression",
        "SAFLA_MEMORY_BACKUP_INTERVAL": "memory.backup_interval",
        
        # Safety settings
        "SAFLA_MEMORY_LIMIT": "safety.memory_limit",
        "SAFLA_CPU_LIMIT": "safety.cpu_limit",
        "SAFLA_SAFETY_MONITORING_INTERVAL": "safety.monitoring_interval",
        "SAFLA_EMERGENCY_STOP_THRESHOLD": "safety.emergency_stop_threshold",
        "SAFLA_ROLLBACK_ENABLED": "safety.rollback_enabled",
        "SAFLA_MAX_EXECUTION_TIME": "safety.max_execution_time",
        "SAFLA_ENABLE_RESOURCE_QUOTAS": "safety.enable_resource_quotas",
        
        # MCP settings
        "SAFLA_MCP_TIMEOUT": "mcp.timeout",
        "SAFLA_MCP_MAX_RETRIES": "mcp.max_retries",
        "SAFLA_MCP_HEALTH_CHECK_INTERVAL": "mcp.health_check_interval",
        "SAFLA_MCP_CONNECTION_POOL_SIZE": "mcp.connection_pool_size",
        "SAFLA_MCP_CIRCUIT_BREAKER_THRESHOLD": "mcp.circuit_breaker_threshold",
        "SAFLA_MCP_LOAD_BALANCING": "mcp.load_balancing",
        "SAFLA_MCP_RATE_LIMIT": "mcp.rate_limit",
        
        # Meta-cognitive settings
        "SAFLA_METACOGNITIVE_ADAPTATION_RATE": "metacognitive.adaptation_rate",
        "SAFLA_METACOGNITIVE_GOAL_PRIORITY_THRESHOLD": "metacognitive.goal_priority_threshold",
        "SAFLA_METACOGNITIVE_STRATEGY_TIMEOUT": "metacognitive.strategy_selection_timeout",
        "SAFLA_METACOGNITIVE_PERFORMANCE_WINDOW": "metacognitive.performance_window_size",
        "SAFLA_METACOGNITIVE_SELF_AWARENESS_INTERVAL": "metacognitive.self_awareness_interval",
        "SAFLA_METACOGNITIVE_META_LEARNING": "metacognitive.meta_learning",
        
        # Integration settings
        "REQUESTY_API_KEY": "integration.requesty_api_key",
        "OPENAI_API_KEY": "integration.openai_api_key",
        "OPENAI_API_BASE": "integration.openai_api_base",
        "OPENAI_MODEL": "integration.openai_model",
        "ANTHROPIC_API_KEY": "integration.anthropic_api_key",
        "ANTHROPIC_MODEL": "integration.anthropic_model",
        "DATABASE_URL": "integration.database_url",
        "DATABASE_POOL_SIZE": "integration.database_pool_size",
        "DATABASE_MAX_OVERFLOW": "integration.database_max_overflow",
        "REDIS_URL": "integration.redis_url",
        "REDIS_PASSWORD": "integration.redis_password",
        "REDIS_DB": "integration.redis_db",
        "RABBITMQ_URL": "integration.rabbitmq_url",
        "CELERY_BROKER_URL": "integration.celery_broker_url",
        
        # Security settings
        "SAFLA_ENCRYPTION_KEY": "security.encryption_key",
        "SAFLA_ENCRYPT_DATA": "security.encrypt_data",
        "JWT_SECRET_KEY": "security.jwt_secret_key",
        "JWT_EXPIRATION_TIME": "security.jwt_expiration_time",
        "SAFLA_ENABLE_RATE_LIMITING": "security.enable_rate_limiting",
        "SAFLA_API_RATE_LIMIT": "security.api_rate_limit",
        "SAFLA_ENABLE_CORS": "security.enable_cors",
        "SAFLA_CORS_ORIGINS": "security.cors_origins",
        "SAFLA_ENABLE_SSL": "security.enable_ssl",
        "SAFLA_SSL_CERT_PATH": "security.ssl_cert_path",
        "SAFLA_SSL_KEY_PATH": "security.ssl_key_path",
        
        # Networking settings
        "SAFLA_HOST": "networking.host",
        "SAFLA_PORT": "networking.port",
        "SAFLA_ENABLE_IPV6": "networking.enable_ipv6",
        "SAFLA_CONNECTION_TIMEOUT": "networking.connection_timeout",
        "SAFLA_READ_TIMEOUT": "networking.read_timeout",
        "SAFLA_MAX_REQUEST_SIZE": "networking.max_request_size",
        
        # Development settings
        "SAFLA_DEV_MODE": "development.dev_mode",
        "SAFLA_HOT_RELOAD": "development.hot_reload",
        "SAFLA_ENABLE_PROFILING": "development.enable_profiling",
        "SAFLA_PROFILING_OUTPUT_DIR": "development.profiling_output_dir",
        "SAFLA_TEST_MODE": "development.test_mode",
        "TEST_DATABASE_URL": "development.test_database_url",
        
        # Experimental settings
        "SAFLA_ENABLE_EXPERIMENTAL": "experimental.enable_experimental",
        "SAFLA_ENABLE_ADVANCED_ML": "experimental.enable_advanced_ml",
        "SAFLA_ENABLE_DISTRIBUTED": "experimental.enable_distributed",
        "SAFLA_CLUSTER_NODES": "experimental.cluster_nodes",
        
        # Monitoring settings
        "SAFLA_ENABLE_METRICS": "monitoring.enable_metrics",
        "SAFLA_METRICS_INTERVAL": "monitoring.metrics_interval",
        "SAFLA_PROMETHEUS_ENDPOINT": "monitoring.prometheus_endpoint",
        "SAFLA_ENABLE_TRACING": "monitoring.enable_tracing",
        "JAEGER_ENDPOINT": "monitoring.jaeger_endpoint",
        "SAFLA_ENABLE_HEALTH_CHECKS": "monitoring.enable_health_checks",
        "SAFLA_HEALTH_ENDPOINT": "monitoring.health_endpoint",
    }
    
    @classmethod
    def map_env_to_config(cls) -> Dict[str, Any]:
        """Map environment variables to nested configuration dictionary."""
        config_dict = {}
        
        for env_var, config_path in cls.ENV_MAPPING.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Extract field name from config_path for type inference
                field_name = config_path.split('.')[-1] if '.' in config_path else config_path
                cls._set_nested_value(config_dict, config_path, cls._convert_value(env_value, field_name))
        
        return config_dict
    
    @classmethod
    def _set_nested_value(cls, config_dict: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in the configuration dictionary."""
        keys = path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    @classmethod
    def _convert_value(cls, value: str, field_name: str = None) -> Union[str, int, float, bool, List[str], List[int]]:
        """Convert string environment variable to appropriate type."""
        # Handle empty strings - for list fields, return empty list; otherwise empty string
        if not value.strip():
            # Check if this is likely a list field based on field name
            if field_name and ('nodes' in field_name.lower() or 'list' in field_name.lower() or field_name.endswith('s')):
                return []
            return ""
        # Handle boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Handle comma-separated lists
        if ',' in value:
            items = [item.strip() for item in value.split(',') if item.strip()]
            # Try to convert to integers if possible
            try:
                return [int(item) for item in items]
            except ValueError:
                return items
        
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value


class ConfigLoader:
    """Main configuration loader with support for multiple sources."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize the configuration loader.
        
        Args:
            env_file: Path to the .env file (defaults to .env in current directory)
        """
        self.env_file = env_file or ".env"
        self.logger = logging.getLogger(__name__)
    
    def load_config(
        self,
        config_file: Optional[str] = None,
        env_override: bool = True,
        validate: bool = True,
        create_dirs: bool = True
    ) -> SAFLAConfig:
        """Load configuration from multiple sources.
        
        Args:
            config_file: Path to JSON configuration file
            env_override: Whether environment variables should override file settings
            validate: Whether to validate the configuration
            create_dirs: Whether to create necessary directories
            
        Returns:
            SAFLAConfig: The loaded and validated configuration
            
        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        try:
            # Load environment variables from .env file only if env_override is True
            if env_override and Path(self.env_file).exists():
                # Don't let .env file override existing env vars
                load_dotenv(self.env_file, override=False)
                self.logger.info(f"Loaded environment variables from {self.env_file}")
            
            # Start with default configuration
            config_dict = {}
            
            # Load from JSON file if provided
            if config_file:
                if not Path(config_file).exists():
                    raise ConfigurationError(f"Configuration file not found: {config_file}")
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config_dict.update(file_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            # Override with environment variables if enabled
            if env_override:
                env_config = EnvironmentMapper.map_env_to_config()
                config_dict = self._deep_merge(config_dict, env_config)
                self.logger.debug("Applied environment variable overrides")
            
            # Create SAFLAConfig instance
            if config_dict:
                # Create config with file data first
                config = SAFLAConfig(_apply_env_overrides=env_override, **config_dict)
            else:
                config = SAFLAConfig(_apply_env_overrides=env_override)
                # Only apply environment overrides if env_override is True
                if env_override:
                    config.apply_environment_overrides()
            
            # Validate configuration if requested
            if validate:
                self._validate_config(config)
            
            # Create directories if requested
            if create_dirs:
                config.create_directories()
                self.logger.info("Created necessary directories")
            
            self.logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_file_config_selectively(self, config: SAFLAConfig, file_config: Dict[str, Any]) -> None:
        """Apply file configuration only for fields not set by environment variables."""
        import os
        
        # Check which environment variables are set
        env_vars = set(os.environ.keys())
        
        # Apply top-level config values if no corresponding env var is set
        for key, value in file_config.items():
            env_key = f"SAFLA_{key.upper()}"
            if env_key not in env_vars and hasattr(config, key):
                if isinstance(value, dict):
                    # Handle nested configuration
                    section_obj = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        nested_env_key = f"SAFLA_{nested_key.upper()}"
                        if nested_env_key not in env_vars and hasattr(section_obj, nested_key):
                            setattr(section_obj, nested_key, nested_value)
                else:
                    setattr(config, key, value)
    
    def _validate_config(self, config: SAFLAConfig) -> None:
        """Validate the configuration and log warnings."""
        # Check security warnings
        security_warnings = config.validate_security()
        for warning in security_warnings:
            self.logger.warning(f"Security configuration warning: {warning}")
        
        # Validate directory paths
        self._validate_paths(config)
        
        # Validate network settings
        self._validate_network_config(config)
        
        # Validate resource limits
        self._validate_resource_limits(config)
    
    def _validate_paths(self, config: SAFLAConfig) -> None:
        """Validate directory and file paths."""
        # Check if SSL files exist when SSL is enabled
        if config.security.enable_ssl:
            if config.security.ssl_cert_path and not Path(config.security.ssl_cert_path).exists():
                self.logger.warning(f"SSL certificate file not found: {config.security.ssl_cert_path}")
            
            if config.security.ssl_key_path and not Path(config.security.ssl_key_path).exists():
                self.logger.warning(f"SSL key file not found: {config.security.ssl_key_path}")
    
    def _validate_network_config(self, config: SAFLAConfig) -> None:
        """Validate network configuration."""
        # Check port availability (basic validation)
        if config.networking.port < 1024 and os.getuid() != 0:
            self.logger.warning(f"Port {config.networking.port} requires root privileges")
    
    def _validate_resource_limits(self, config: SAFLAConfig) -> None:
        """Validate resource limit settings."""
        # Check if memory limits are reasonable
        if config.safety.memory_limit > 8 * 1024 * 1024 * 1024:  # 8GB
            self.logger.warning("Memory limit is set very high (>8GB)")
        
        # Check CPU limit
        if config.safety.cpu_limit > 0.95:
            self.logger.warning("CPU limit is set very high (>95%)")
    
    def save_config(
        self,
        config: SAFLAConfig,
        output_file: str,
        exclude_secrets: bool = True,
        format: str = "json"
    ) -> None:
        """Save configuration to file.
        
        Args:
            config: The configuration to save
            output_file: Path to output file
            exclude_secrets: Whether to exclude secret values
            format: Output format ('json' or 'env')
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                config.save_to_json_file(output_file, exclude_secrets=exclude_secrets)
            elif format.lower() == "env":
                self._save_as_env_file(config, output_file, exclude_secrets=exclude_secrets)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration saved to {output_file}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}") from e
    
    def _save_as_env_file(self, config: SAFLAConfig, output_file: str, exclude_secrets: bool = True) -> None:
        """Save configuration as environment file."""
        config_dict = config.dict()
        
        if exclude_secrets:
            config_dict = config._exclude_secrets(config_dict)
        
        with open(output_file, 'w') as f:
            f.write("# SAFLA Configuration Environment File\n")
            f.write("# Generated automatically - modify with care\n\n")
            
            for env_var, config_path in EnvironmentMapper.ENV_MAPPING.items():
                value = self._get_nested_value(config_dict, config_path)
                if value is not None:
                    if isinstance(value, list):
                        value = ','.join(map(str, value))
                    f.write(f"{env_var}={value}\n")
    
    def _get_nested_value(self, config_dict: Dict[str, Any], path: str) -> Any:
        """Get a nested value from the configuration dictionary."""
        keys = path.split('.')
        current = config_dict
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


def create_default_config_file(output_path: str = "config.json") -> None:
    """Create a default configuration file with all options documented.
    
    Args:
        output_path: Path where to save the default configuration
    """
    config = SAFLAConfig()
    loader = ConfigLoader()
    loader.save_config(config, output_path, exclude_secrets=True, format="json")
    
    print(f"Default configuration file created at: {output_path}")
    print("Edit this file to customize your SAFLA configuration.")


def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file.
    
    Args:
        config_path: Path to the configuration file to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        loader = ConfigLoader()
        config = loader.load_config(config_file=config_path, env_override=False, validate=True, create_dirs=False)
        print(f"Configuration file {config_path} is valid!")
        return True
    except Exception as e:
        print(f"Configuration validation failed: {str(e)}")
        return False


# Convenience functions for common use cases
def load_config_from_env(env_file: str = ".env") -> SAFLAConfig:
    """Load configuration from environment file."""
    loader = ConfigLoader(env_file=env_file)
    return loader.load_config()


def load_config_from_json(config_file: str) -> SAFLAConfig:
    """Load configuration from JSON file."""
    loader = ConfigLoader()
    return loader.load_config(config_file=config_file, env_override=False)


def load_config_with_overrides(config_file: str, env_file: str = ".env") -> SAFLAConfig:
    """Load configuration from JSON file with environment overrides."""
    loader = ConfigLoader(env_file=env_file)
    return loader.load_config(config_file=config_file, env_override=True)