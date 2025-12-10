"""
SAFLA Pydantic Configuration Management

This module provides a comprehensive configuration management system using Pydantic
for validation, type safety, and secure handling of sensitive information.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, SecretStr, AnyUrl
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging

# Note: Environment variables are loaded explicitly by ConfigLoader when needed
# Removed global load_dotenv() to prevent automatic environment variable loading

logger = logging.getLogger(__name__)


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    
    worker_threads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker threads for parallel processing"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1000,
        description="Batch size for processing operations"
    )
    max_concurrent_ops: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent operations"
    )
    memory_pool_size: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Memory pool size in MB"
    )
    cache_size: int = Field(
        default=256,
        ge=32,
        le=4096,
        description="Cache size in MB"
    )
    enable_optimizations: bool = Field(
        default=True,
        description="Enable performance optimizations"
    )


class MemoryConfig(BaseModel):
    """Configuration for memory components."""
    
    vector_dimensions: List[int] = Field(
        default=[512, 768, 1024, 1536],
        description="Vector dimensions for embeddings"
    )
    max_memories: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum number of memories to store"
    )
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for memory matching"
    )
    consolidation_interval: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Memory consolidation interval in seconds"
    )
    cleanup_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Memory cleanup threshold"
    )
    compression: bool = Field(
        default=True,
        description="Enable memory compression"
    )
    backup_interval: int = Field(
        default=7200,
        ge=300,
        le=86400,
        description="Memory backup interval in seconds"
    )

    @field_validator('vector_dimensions')
    @classmethod
    def validate_vector_dimensions(cls, v):
        if not v:
            raise ValueError("Vector dimensions cannot be empty")
        if any(dim <= 0 for dim in v):
            raise ValueError("All vector dimensions must be positive")
        return v


class SafetyConfig(BaseModel):
    """Configuration for safety validation."""
    
    memory_limit: int = Field(
        default=1000000000,  # 1GB
        ge=100000000,  # 100MB minimum
        description="Memory limit in bytes"
    )
    cpu_limit: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="CPU usage limit (0.0-1.0)"
    )
    monitoring_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Safety monitoring interval in seconds"
    )
    emergency_stop_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Emergency stop threshold (0.0-1.0)"
    )
    rollback_enabled: bool = Field(
        default=True,
        description="Enable automatic rollback on failures"
    )
    max_execution_time: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Maximum execution time in seconds"
    )
    enable_resource_quotas: bool = Field(
        default=True,
        description="Enable resource quotas"
    )


class MCPConfig(BaseModel):
    """Configuration for MCP orchestration."""
    
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="MCP operation timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed operations"
    )
    health_check_interval: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Health check interval in seconds"
    )
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Connection pool size"
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Circuit breaker failure threshold"
    )
    load_balancing: bool = Field(
        default=True,
        description="Enable MCP load balancing"
    )
    rate_limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="MCP request rate limit (requests per second)"
    )


class MetaCognitiveConfig(BaseModel):
    """Configuration for meta-cognitive engine."""
    
    adaptation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Adaptation rate for learning"
    )
    goal_priority_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Goal priority threshold"
    )
    strategy_selection_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Strategy selection timeout in seconds"
    )
    performance_window_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Performance window size for analysis"
    )
    self_awareness_interval: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Self-awareness check interval in seconds"
    )
    meta_learning: bool = Field(
        default=True,
        description="Enable meta-learning"
    )


class IntegrationConfig(BaseModel):
    """Configuration for external integrations."""
    
    # API Keys (using SecretStr for security)
    requesty_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Requesty API key"
    )
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    # API Configuration
    openai_api_base: AnyUrl = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )
    openai_model: str = Field(
        default="gpt-4",
        description="Default OpenAI model"
    )
    anthropic_model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Default Anthropic model"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./data/safla.db",
        description="Database connection URL"
    )
    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Database max overflow connections"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
        alias="REDIS_URL"
    )
    redis_password: Optional[SecretStr] = Field(
        default=None,
        description="Redis password"
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )
    
    # Message Queue Configuration
    rabbitmq_url: str = Field(
        default="amqp://localhost:5672",
        description="RabbitMQ connection URL"
    )
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL"
    )


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    
    encryption_key: Optional[SecretStr] = Field(
        default=None,
        description="Encryption key for sensitive data"
    )
    encrypt_data: bool = Field(
        default=True,
        description="Enable data encryption at rest",
        alias="SAFLA_ENCRYPT_DATA"
    )
    jwt_secret_key: Optional[SecretStr] = Field(
        default=None,
        description="JWT secret for authentication"
    )
    jwt_expiration_time: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="JWT token expiration time in seconds"
    )
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable API rate limiting"
    )
    api_rate_limit: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="API rate limit (requests per minute)"
    )
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    enable_ssl: bool = Field(
        default=False,
        description="Enable SSL/TLS",
        alias="SAFLA_ENABLE_SSL"
    )
    ssl_cert_path: Optional[str] = Field(
        default=None,
        description="SSL certificate path"
    )
    ssl_key_path: Optional[str] = Field(
        default=None,
        description="SSL private key path"
    )


class NetworkingConfig(BaseModel):
    """Configuration for networking settings."""
    
    host: str = Field(
        default="localhost",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Server port"
    )
    enable_ipv6: bool = Field(
        default=False,
        description="Enable IPv6"
    )
    connection_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Connection timeout in seconds"
    )
    read_timeout: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Read timeout in seconds"
    )
    max_request_size: int = Field(
        default=16777216,  # 16MB
        ge=1024,
        le=104857600,  # 100MB
        description="Maximum request size in bytes"
    )


class DevelopmentConfig(BaseModel):
    """Configuration for development settings."""
    
    dev_mode: bool = Field(
        default=False,
        description="Enable development mode features",
        alias="SAFLA_DEV_MODE"
    )
    hot_reload: bool = Field(
        default=False,
        description="Enable hot reloading",
        alias="SAFLA_HOT_RELOAD"
    )
    enable_profiling: bool = Field(
        default=False,
        description="Enable profiling",
        alias="SAFLA_ENABLE_PROFILING"
    )
    profiling_output_dir: str = Field(
        default="./profiling",
        description="Profiling output directory"
    )
    test_mode: bool = Field(
        default=False,
        description="Enable test mode"
    )
    test_database_url: str = Field(
        default="sqlite:///./test.db",
        description="Test database URL"
    )


class ExperimentalConfig(BaseModel):
    """Configuration for experimental features."""
    
    enable_experimental: bool = Field(
        default=False,
        description="Enable experimental features"
    )
    enable_advanced_ml: bool = Field(
        default=False,
        description="Enable advanced ML features"
    )
    enable_distributed: bool = Field(
        default=False,
        description="Enable distributed processing"
    )
    cluster_nodes: List[str] = Field(
        default=[],
        description="Distributed cluster nodes"
    )


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and observability."""
    
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_interval: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Metrics export interval in seconds"
    )
    prometheus_endpoint: str = Field(
        default="/metrics",
        description="Prometheus metrics endpoint"
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    jaeger_endpoint: Optional[str] = Field(
        default=None,
        description="Jaeger tracing endpoint"
    )
    enable_health_checks: bool = Field(
        default=True,
        description="Enable health checks"
    )
    health_endpoint: str = Field(
        default="/health",
        description="Health check endpoint"
    )


class SAFLAConfig(BaseSettings):
    """Main SAFLA configuration class using Pydantic BaseSettings."""
    
    model_config = {
        "env_prefix": "SAFLA_",
        "env_file": None,  # Disable automatic .env loading - controlled by ConfigLoader
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_nested_delimiter": "_",
        "extra": "ignore",
        "validate_assignment": True
    }
    
    def __init__(self, _apply_env_overrides: bool = True, **data):
        """Initialize SAFLAConfig with optional data.
        
        Args:
            _apply_env_overrides: Whether to apply environment overrides in model_post_init
            **data: Configuration data
        """
        self._apply_env_overrides = _apply_env_overrides
        
        # If env overrides are disabled, temporarily clear environment variables
        if not _apply_env_overrides:
            import os
            # Save current environment variables
            saved_env = {}
            env_keys_to_clear = []
            for key in os.environ:
                if key.startswith('SAFLA_') or key in ['DATABASE_URL', 'REDIS_URL', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'JWT_SECRET_KEY']:
                    saved_env[key] = os.environ[key]
                    env_keys_to_clear.append(key)
            
            # Temporarily clear environment variables
            for key in env_keys_to_clear:
                del os.environ[key]
            
            try:
                super().__init__(**data)
            finally:
                # Restore environment variables
                for key, value in saved_env.items():
                    os.environ[key] = value
        else:
            super().__init__(**data)
    
    def model_post_init(self, __context) -> None:
        """Post-initialization hook."""
        # Apply environment overrides for direct instantiation
        # This allows SAFLAConfig() to read environment variables
        # but ConfigLoader controls this explicitly
        if getattr(self, '_apply_env_overrides', True):
            self.apply_environment_overrides()
    
    def apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        import os
        
        # Handle nested environment variable mappings
        nested_env_mappings = {
            # Performance settings
            'SAFLA_WORKER_THREADS': ('performance', 'worker_threads'),
            'SAFLA_BATCH_SIZE': ('performance', 'batch_size'),
            'SAFLA_MAX_CONCURRENT_OPS': ('performance', 'max_concurrent_ops'),
            'SAFLA_MEMORY_POOL_SIZE': ('performance', 'memory_pool_size'),
            'SAFLA_CACHE_SIZE': ('performance', 'cache_size'),
            'SAFLA_ENABLE_OPTIMIZATIONS': ('performance', 'enable_optimizations'),
            
            # Memory settings
            'SAFLA_MAX_MEMORIES': ('memory', 'max_memories'),
            'SAFLA_SIMILARITY_THRESHOLD': ('memory', 'similarity_threshold'),
            'SAFLA_CONSOLIDATION_INTERVAL': ('memory', 'consolidation_interval'),
            'SAFLA_CLEANUP_THRESHOLD': ('memory', 'cleanup_threshold'),
            'SAFLA_COMPRESSION': ('memory', 'compression'),
            'SAFLA_BACKUP_INTERVAL': ('memory', 'backup_interval'),
            
            # Development settings
            'SAFLA_DEV_MODE': ('development', 'dev_mode'),
            'SAFLA_HOT_RELOAD': ('development', 'hot_reload'),
            'SAFLA_ENABLE_PROFILING': ('development', 'enable_profiling'),
            'SAFLA_PROFILING_OUTPUT_DIR': ('development', 'profiling_output_dir'),
            'SAFLA_TEST_MODE': ('development', 'test_mode'),
            'SAFLA_TEST_DATABASE_URL': ('development', 'test_database_url'),
            
            # Security settings
            'SAFLA_ENCRYPT_DATA': ('security', 'encrypt_data'),
            'SAFLA_ENCRYPTION_KEY': ('security', 'encryption_key'),
            'SAFLA_JWT_SECRET_KEY': ('security', 'jwt_secret_key'),
            'SAFLA_ENABLE_SSL': ('security', 'enable_ssl'),
            'SAFLA_SSL_CERT_PATH': ('security', 'ssl_cert_path'),
            'SAFLA_SSL_KEY_PATH': ('security', 'ssl_key_path'),
            'SAFLA_RATE_LIMIT_ENABLED': ('security', 'rate_limit_enabled'),
            'SAFLA_RATE_LIMIT_REQUESTS': ('security', 'rate_limit_requests'),
            'SAFLA_RATE_LIMIT_WINDOW': ('security', 'rate_limit_window'),
            
            # Integration settings
            'DATABASE_URL': ('integration', 'database_url'),
            'REDIS_URL': ('integration', 'redis_url'),
            'SAFLA_DATABASE_URL': ('integration', 'database_url'),
            'SAFLA_REDIS_URL': ('integration', 'redis_url'),
        }
        
        for env_var, (section, field) in nested_env_mappings.items():
            env_val = os.getenv(env_var)
            if env_val is not None:
                section_obj = getattr(self, section)
                current_value = getattr(section_obj, field)
                
                # Get the default value for this field by creating a fresh instance of the section
                section_class = type(section_obj)
                default_section_obj = section_class()
                default_value = getattr(default_section_obj, field)
                
                # Only override if the current value is still the default value
                if current_value == default_value:
                    # Convert value based on field type
                    if field in ['worker_threads', 'batch_size', 'max_concurrent_ops', 'memory_pool_size', 'cache_size',
                               'max_memories', 'consolidation_interval', 'backup_interval', 'rate_limit_requests', 'rate_limit_window']:
                        try:
                            setattr(section_obj, field, int(env_val))
                        except ValueError:
                            continue
                    elif field in ['similarity_threshold', 'cleanup_threshold']:
                        try:
                            setattr(section_obj, field, float(env_val))
                        except ValueError:
                            continue
                    elif field in ['enable_optimizations', 'compression', 'dev_mode', 'hot_reload', 'enable_profiling',
                                 'test_mode', 'encrypt_data', 'enable_ssl', 'rate_limit_enabled']:
                        setattr(section_obj, field, env_val.lower() in ('true', '1', 'yes', 'on'))
                    else:
                        setattr(section_obj, field, env_val)
    
    # Component configurations
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    metacognitive: MetaCognitiveConfig = Field(default_factory=MetaCognitiveConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    networking: NetworkingConfig = Field(default_factory=NetworkingConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # General settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    enable_monitoring: bool = Field(
        default=True,
        description="Enable system monitoring"
    )
    data_dir: str = Field(
        default="./data",
        description="Data directory path"
    )
    config_dir: str = Field(
        default="./.roo",
        description="Configuration directory path"
    )

    @classmethod
    def from_env_file(cls, env_file: str = ".env") -> "SAFLAConfig":
        """Create configuration from environment file."""
        if Path(env_file).exists():
            load_dotenv(env_file, override=True)
        return cls()

    @classmethod
    def from_json_file(cls, config_path: str) -> "SAFLAConfig":
        """Load configuration from a JSON file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)

    def save_to_json_file(self, config_path: str, exclude_secrets: bool = True) -> None:
        """Save configuration to a JSON file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and optionally exclude secrets
        config_dict = self.model_dump()
        
        if exclude_secrets:
            config_dict = self._exclude_secrets(config_dict)
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def _exclude_secrets(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove secret values from configuration dictionary."""
        def _remove_secrets(obj):
            if isinstance(obj, dict):
                return {
                    k: _remove_secrets(v) if not k.endswith('_key') and 'password' not in k.lower() 
                    else "***REDACTED***" 
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [_remove_secrets(item) for item in obj]
            else:
                return obj
        
        return _remove_secrets(config_dict)

    def create_directories(self) -> None:
        """Create necessary directories for SAFLA operation."""
        directories = [
            self.data_dir,
            self.config_dir,
            f"{self.data_dir}/memory",
            f"{self.data_dir}/checkpoints",
            f"{self.data_dir}/logs",
            f"{self.data_dir}/backups",
            self.development.profiling_output_dir if self.development.enable_profiling else None
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")

    def validate_security(self) -> List[str]:
        """Validate security configuration and return warnings."""
        warnings = []
        
        if self.security.encrypt_data and not self.security.encryption_key:
            warnings.append("Data encryption is enabled but no encryption key is provided")
        
        if not self.security.jwt_secret_key:
            warnings.append("No JWT secret key provided - authentication may not work")
        
        if self.security.enable_ssl and (not self.security.ssl_cert_path or not self.security.ssl_key_path):
            warnings.append("SSL is enabled but certificate or key path is missing")
        
        return warnings

    def get_database_url(self, test_mode: bool = False) -> str:
        """Get the appropriate database URL based on mode."""
        if test_mode or self.development.test_mode:
            return self.development.test_database_url
        return self.integration.database_url

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not (self.debug or self.development.dev_mode or self.development.test_mode)

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary, merging with existing values."""
        def merge_dicts(base: dict, update: dict) -> dict:
            """Recursively merge two dictionaries."""
            result = base.copy()
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        # Get current config as dict
        current_dict = self.model_dump()
        
        # Merge with new config
        merged_dict = merge_dicts(current_dict, config_dict)
        
        # Update all fields
        for field_name, field_info in self.model_fields.items():
            if field_name in merged_dict:
                setattr(self, field_name, merged_dict[field_name])


# Global configuration instance
_global_config: Optional[SAFLAConfig] = None


def get_config() -> SAFLAConfig:
    """Get the global SAFLA configuration."""
    global _global_config
    if _global_config is None:
        _global_config = SAFLAConfig()
        
        # Validate security settings and log warnings
        security_warnings = _global_config.validate_security()
        for warning in security_warnings:
            logger.warning(f"Security configuration warning: {warning}")
    
    return _global_config


def set_config(config: SAFLAConfig) -> None:
    """Set the global SAFLA configuration."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to default."""
    global _global_config
    _global_config = None


def load_config_from_file(config_path: str) -> SAFLAConfig:
    """Load configuration from file and set as global."""
    config = SAFLAConfig.from_json_file(config_path)
    set_config(config)
    return config


def reload_config() -> SAFLAConfig:
    """Reload configuration from environment."""
    global _global_config
    _global_config = SAFLAConfig()
    return _global_config