# SAFLA Configuration System

The SAFLA Configuration System provides a comprehensive, secure, and flexible way to manage configuration settings for the SAFLA package. It supports multiple configuration sources, validation, type safety, and secure handling of sensitive information.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration Sources](#configuration-sources)
4. [Configuration Structure](#configuration-structure)
5. [Environment Variables](#environment-variables)
6. [Validation and Type Safety](#validation-and-type-safety)
7. [Security Features](#security-features)
8. [CLI Tools](#cli-tools)
9. [Migration Guide](#migration-guide)
10. [Best Practices](#best-practices)
11. [Examples](#examples)
12. [API Reference](#api-reference)

## Overview

The SAFLA configuration system consists of several components:

- **Legacy Configuration**: Dataclass-based configuration for backward compatibility
- **Pydantic Configuration**: Modern configuration with validation and type safety
- **Configuration Loader**: Utility for loading from multiple sources
- **CLI Tools**: Command-line interface for configuration management
- **Environment Integration**: Seamless environment variable support

### Key Features

- ✅ **Type Safety**: Full type hints and validation using Pydantic
- ✅ **Multiple Sources**: Load from environment variables, JSON files, or .env files
- ✅ **Security**: Secure handling of API keys and sensitive data
- ✅ **Validation**: Comprehensive validation with helpful error messages
- ✅ **CLI Tools**: Command-line interface for configuration management
- ✅ **Backward Compatibility**: Seamless migration from legacy configuration
- ✅ **Production Ready**: Production vs development configuration patterns

## Quick Start

### Basic Usage

```python
from safla.utils.config import get_config

# Get configuration (automatically uses best available system)
config = get_config()

print(f"Debug mode: {config.debug}")
print(f"Worker threads: {config.performance.worker_threads}")
print(f"Memory limit: {config.safety.memory_limit}")
```

### Using Pydantic Configuration Directly

```python
from safla.utils.pydantic_config import SAFLAConfig

# Create configuration with validation
config = SAFLAConfig(
    debug=True,
    performance={"worker_threads": 8},
    memory={"max_memories": 50000}
)

# Validate security settings
warnings = config.validate_security()
if warnings:
    print("Security warnings:", warnings)
```

### Loading from Environment File

```python
from safla.utils.config_loader import load_config_from_env

# Load from .env file
config = load_config_from_env(".env")
print(f"Loaded configuration: {config.debug}")
```

## Configuration Sources

The configuration system supports loading from multiple sources in order of precedence:

1. **Environment Variables** (highest precedence)
2. **JSON Configuration Files**
3. **Environment (.env) Files**
4. **Default Values** (lowest precedence)

### Environment Variables

Set environment variables with the `SAFLA_` prefix:

```bash
export SAFLA_DEBUG=true
export SAFLA_LOG_LEVEL=DEBUG
export SAFLA_WORKER_THREADS=8
export SAFLA_MAX_MEMORIES=50000
```

### JSON Configuration

Create a JSON configuration file:

```json
{
  "debug": true,
  "log_level": "INFO",
  "performance": {
    "worker_threads": 8,
    "batch_size": 64
  },
  "memory": {
    "max_memories": 50000,
    "similarity_threshold": 0.8
  }
}
```

Load it:

```python
from safla.utils.config import load_config_from_file

config = load_config_from_file("config.json")
```

### Environment (.env) Files

Create a `.env` file:

```bash
# Basic Configuration
SAFLA_DEBUG=true
SAFLA_LOG_LEVEL=INFO
SAFLA_DATA_DIR=./data

# Performance Settings
SAFLA_WORKER_THREADS=8
SAFLA_BATCH_SIZE=64

# Memory Management
SAFLA_MAX_MEMORIES=50000
SAFLA_SIMILARITY_THRESHOLD=0.8
```

## Configuration Structure

The configuration is organized into logical sections:

### Basic Settings
- `debug`: Enable debug mode
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `data_dir`: Data directory path
- `config_file`: Configuration file path

### Performance Configuration
- `worker_threads`: Number of worker threads
- `batch_size`: Processing batch size
- `enable_optimizations`: Enable performance optimizations
- `cache_size`: Cache size in MB

### Memory Management
- `max_memories`: Maximum number of memories to store
- `similarity_threshold`: Similarity threshold for memory matching
- `vector_dimensions`: Supported vector dimensions
- `compression`: Enable memory compression
- `cleanup_interval`: Memory cleanup interval in seconds

### Safety Configuration
- `memory_limit`: Maximum memory usage in bytes
- `cpu_limit`: Maximum CPU usage (0.0-1.0)
- `timeout`: Operation timeout in seconds
- `rollback_enabled`: Enable automatic rollback on errors
- `max_retries`: Maximum number of retries

### MCP Orchestration
- `timeout`: MCP operation timeout
- `max_connections`: Maximum concurrent connections
- `retry_attempts`: Number of retry attempts
- `enable_logging`: Enable MCP logging

### Security Configuration
- `enable_ssl`: Enable SSL/TLS
- `encrypt_data`: Encrypt stored data
- `api_rate_limit`: API rate limit per minute
- `enable_rate_limiting`: Enable rate limiting
- `allowed_hosts`: List of allowed hosts
- `encryption_key`: Data encryption key (SecretStr)
- `jwt_secret_key`: JWT secret key (SecretStr)

### Integration Configuration
- `openai_api_key`: OpenAI API key (SecretStr)
- `anthropic_api_key`: Anthropic API key (SecretStr)
- `database_url`: Database connection URL (SecretStr)
- `redis_url`: Redis connection URL (SecretStr)

### Development Configuration
- `dev_mode`: Enable development mode
- `hot_reload`: Enable hot reloading
- `enable_profiling`: Enable performance profiling
- `mock_external_apis`: Mock external API calls

### Experimental Features
- `enable_experimental`: Enable experimental features
- `enable_advanced_ml`: Enable advanced ML features
- `enable_distributed`: Enable distributed processing
- `cluster_nodes`: List of cluster node addresses

### Monitoring Configuration
- `enable_metrics`: Enable metrics collection
- `enable_tracing`: Enable distributed tracing
- `metrics_interval`: Metrics collection interval
- `log_performance`: Log performance metrics

## Environment Variables

All configuration options can be set via environment variables using the `SAFLA_` prefix. Nested configuration uses underscores:

| Configuration Path | Environment Variable |
|-------------------|---------------------|
| `debug` | `SAFLA_DEBUG` |
| `performance.worker_threads` | `SAFLA_WORKER_THREADS` |
| `memory.max_memories` | `SAFLA_MAX_MEMORIES` |
| `safety.memory_limit` | `SAFLA_MEMORY_LIMIT` |
| `security.enable_ssl` | `SAFLA_ENABLE_SSL` |
| `integration.openai_api_key` | `SAFLA_OPENAI_API_KEY` |

### Type Conversion

Environment variables are automatically converted to appropriate types:

- `"true"/"false"` → `bool`
- `"123"` → `int`
- `"3.14"` → `float`
- `"item1,item2,item3"` → `List[str]`
- `"[1,2,3]"` → `List[int]` (JSON format)

## Validation and Type Safety

The Pydantic-based configuration provides comprehensive validation:

### Field Validation

```python
from safla.utils.pydantic_config import SAFLAConfig
from pydantic import ValidationError

try:
    config = SAFLAConfig(
        performance={"worker_threads": 100},  # Too high
        memory={"similarity_threshold": 1.5},  # Out of range
    )
except ValidationError as e:
    print(f"Validation errors: {e}")
```

### Custom Validators

The configuration includes custom validators for:

- **Worker threads**: Must be between 1 and 32
- **Similarity threshold**: Must be between 0.0 and 1.0
- **CPU limit**: Must be between 0.0 and 1.0
- **Memory limit**: Must be positive
- **Vector dimensions**: Must be powers of 2

### Security Validation

```python
config = SAFLAConfig()
warnings = config.validate_security()

if warnings:
    for warning in warnings:
        print(f"Security warning: {warning}")
```

## Security Features

### Secure Secret Handling

Sensitive information is handled using Pydantic's `SecretStr`:

```python
from safla.utils.pydantic_config import SAFLAConfig

config = SAFLAConfig(
    integration={
        "openai_api_key": "sk-your-secret-key",
        "database_url": "postgresql://user:pass@host/db"
    }
)

# Secrets are automatically wrapped in SecretStr
print(type(config.integration.openai_api_key))  # <class 'pydantic.types.SecretStr'>

# Get the actual value when needed
api_key = config.integration.openai_api_key.get_secret_value()
```

### Excluding Secrets from Output

```python
# Save configuration without secrets
config.save_to_json_file("config.json", exclude_secrets=True)

# Convert to dict without secrets
safe_dict = config.model_dump(exclude_secrets=True)
```

### Production Security Checks

```python
config = SAFLAConfig()

# Check if configuration is production-ready
if config.is_production():
    print("Production configuration detected")
    
    # Validate security settings
    warnings = config.validate_security()
    if warnings:
        print("Security issues found:", warnings)
```

## CLI Tools

The configuration system includes a comprehensive CLI tool:

### Installation

```bash
# The CLI is included with SAFLA
python -m safla.utils.config_cli --help
```

### Commands

#### Create Configuration

```bash
# Create a new .env file with all options
python -m safla.utils.config_cli create --output .env

# Create a JSON configuration file
python -m safla.utils.config_cli create --output config.json --format json
```

#### Validate Configuration

```bash
# Validate current configuration
python -m safla.utils.config_cli validate

# Validate specific file
python -m safla.utils.config_cli validate --config config.json
```

#### Show Configuration

```bash
# Show current configuration
python -m safla.utils.config_cli show

# Show configuration from file
python -m safla.utils.config_cli show --config config.json

# Show only specific section
python -m safla.utils.config_cli show --section performance
```

#### Production Readiness Check

```bash
# Check if configuration is production-ready
python -m safla.utils.config_cli check

# Check specific configuration file
python -m safla.utils.config_cli check --config config.json
```

#### Convert Configuration

```bash
# Convert .env to JSON
python -m safla.utils.config_cli convert .env config.json

# Convert JSON to .env
python -m safla.utils.config_cli convert config.json .env
```

#### Set Configuration Values

```bash
# Set a configuration value
python -m safla.utils.config_cli set debug true
python -m safla.utils.config_cli set performance.worker_threads 8
python -m safla.utils.config_cli set memory.max_memories 50000
```

#### Get Configuration Values

```bash
# Get a configuration value
python -m safla.utils.config_cli get debug
python -m safla.utils.config_cli get performance.worker_threads
```

#### Setup Development Environment

```bash
# Setup development environment
python -m safla.utils.config_cli setup --env development

# Setup production environment
python -m safla.utils.config_cli setup --env production
```

## Migration Guide

### From Legacy to Pydantic Configuration

The configuration system provides seamless migration:

```python
from safla.utils.config import migrate_to_pydantic, get_config

# Automatic migration
pydantic_config = migrate_to_pydantic()

# Or use the unified interface
config = get_config()  # Automatically uses best available system
```

### Backward Compatibility

Existing code continues to work without changes:

```python
# This still works with both legacy and Pydantic configurations
from safla.utils.config import get_config

config = get_config()
print(config.debug)  # Works with both systems
```

### Migration Steps

1. **Install Pydantic** (if not already installed):
   ```bash
   pip install pydantic pydantic-settings
   ```

2. **Update imports** (optional):
   ```python
   # Old way (still works)
   from safla.utils.config import get_config
   
   # New way (recommended for new code)
   from safla.utils.pydantic_config import SAFLAConfig
   ```

3. **Update configuration creation**:
   ```python
   # Old way
   from safla.utils.config import SAFLAConfig as LegacyConfig
   config = LegacyConfig(debug=True)
   
   # New way
   from safla.utils.pydantic_config import SAFLAConfig
   config = SAFLAConfig(debug=True)
   ```

4. **Use new features**:
   ```python
   # Validation
   warnings = config.validate_security()
   
   # Production detection
   is_prod = config.is_production()
   
   # Directory creation
   config.create_directories()
   ```

## Best Practices

### 1. Environment-Specific Configuration

Use different configurations for different environments:

```python
# Development
dev_config = SAFLAConfig(
    debug=True,
    log_level="DEBUG",
    development={"dev_mode": True, "hot_reload": True},
    security={"enable_ssl": False}
)

# Production
prod_config = SAFLAConfig(
    debug=False,
    log_level="WARNING",
    development={"dev_mode": False},
    security={"enable_ssl": True, "encrypt_data": True}
)
```

### 2. Secret Management

Never store secrets in configuration files:

```bash
# Good: Use environment variables
export SAFLA_OPENAI_API_KEY="sk-your-secret-key"
export SAFLA_DATABASE_URL="postgresql://user:pass@host/db"

# Bad: Don't put secrets in config files
# config.json: {"openai_api_key": "sk-your-secret-key"}  # DON'T DO THIS
```

### 3. Validation Before Deployment

Always validate configuration before deployment:

```python
config = SAFLAConfig.from_env()

# Validate configuration
try:
    config.model_validate(config.model_dump())
    print("✅ Configuration is valid")
except ValidationError as e:
    print(f"❌ Configuration errors: {e}")
    exit(1)

# Check security
warnings = config.validate_security()
if warnings and config.is_production():
    print(f"❌ Security issues in production: {warnings}")
    exit(1)
```

### 4. Configuration Documentation

Document your configuration:

```python
# config.py
"""
Application Configuration

This module defines the configuration for the SAFLA application.
Environment variables can be used to override any setting.

Example:
    export SAFLA_DEBUG=true
    export SAFLA_WORKER_THREADS=8
    export SAFLA_OPENAI_API_KEY="sk-your-key"
"""

config = SAFLAConfig()
```

### 5. Testing Configuration

Test your configuration:

```python
import pytest
from safla.utils.pydantic_config import SAFLAConfig

def test_valid_configuration():
    config = SAFLAConfig(
        performance={"worker_threads": 4},
        memory={"max_memories": 10000}
    )
    assert config.performance.worker_threads == 4
    assert config.memory.max_memories == 10000

def test_invalid_configuration():
    with pytest.raises(ValidationError):
        SAFLAConfig(performance={"worker_threads": 100})  # Too high
```

## Examples

### Complete Example Application

```python
"""
Example SAFLA application with proper configuration management.
"""

import os
import logging
from safla.utils.config import get_config
from safla.utils.pydantic_config import SAFLAConfig

def setup_logging(config):
    """Setup logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Load configuration
    config = get_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    if hasattr(config, 'create_directories'):
        config.create_directories()
    
    # Validate security in production
    if hasattr(config, 'is_production') and config.is_production():
        warnings = config.validate_security()
        if warnings:
            logger.warning(f"Security warnings: {warnings}")
    
    # Use configuration
    logger.info(f"Starting SAFLA with {config.performance.worker_threads} workers")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Debug mode: {config.debug}")
    
    # Your application logic here
    print("SAFLA application started successfully!")

if __name__ == "__main__":
    main()
```

### Configuration with Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Set environment variables
ENV SAFLA_DEBUG=false
ENV SAFLA_LOG_LEVEL=INFO
ENV SAFLA_WORKER_THREADS=8

# Run application
CMD ["python", "main.py"]
```

```bash
# docker-compose.yml
version: '3.8'
services:
  safla:
    build: .
    environment:
      - SAFLA_DEBUG=false
      - SAFLA_LOG_LEVEL=WARNING
      - SAFLA_WORKER_THREADS=16
      - SAFLA_OPENAI_API_KEY=${OPENAI_API_KEY}
      - SAFLA_DATABASE_URL=postgresql://user:pass@db:5432/safla
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=safla
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
```

## API Reference

### Core Functions

#### `get_config(use_pydantic=None, **kwargs)`

Get the current configuration instance.

**Parameters:**
- `use_pydantic` (bool, optional): Force use of Pydantic configuration
- `**kwargs`: Additional configuration parameters

**Returns:**
- Configuration instance (legacy or Pydantic)

#### `set_config(config)`

Set the global configuration instance.

**Parameters:**
- `config`: Configuration instance to set

#### `reset_config()`

Reset the global configuration to defaults.

#### `load_config_from_file(file_path, **kwargs)`

Load configuration from a file.

**Parameters:**
- `file_path` (str): Path to configuration file
- `**kwargs`: Additional parameters

**Returns:**
- Configuration instance

#### `migrate_to_pydantic()`

Migrate from legacy to Pydantic configuration.

**Returns:**
- Pydantic configuration instance

### Pydantic Configuration Classes

#### `SAFLAConfig`

Main configuration class with all settings.

**Methods:**
- `validate_security()`: Validate security settings
- `is_production()`: Check if running in production
- `create_directories()`: Create necessary directories
- `get_database_url(test_mode=False)`: Get database URL
- `save_to_json_file(file_path, exclude_secrets=False)`: Save to JSON file

#### Configuration Sections

- `PerformanceConfig`: Performance-related settings
- `MemoryConfig`: Memory management settings
- `SafetyConfig`: Safety and limits settings
- `MCPConfig`: MCP orchestration settings
- `MetaCognitiveConfig`: Meta-cognitive engine settings
- `IntegrationConfig`: External service integration
- `SecurityConfig`: Security settings
- `NetworkingConfig`: Network configuration
- `DevelopmentConfig`: Development settings
- `ExperimentalConfig`: Experimental features
- `MonitoringConfig`: Monitoring and observability

### Configuration Loader

#### `ConfigLoader`

Utility class for loading configuration from multiple sources.

**Constructor:**
- `ConfigLoader(env_file=None, json_file=None, **kwargs)`

**Methods:**
- `load_config(create_dirs=True)`: Load configuration
- `save_config(config, file_path, format='auto')`: Save configuration
- `validate_config(config)`: Validate configuration

#### `load_config_from_env(env_file='.env', **kwargs)`

Load configuration from environment file.

**Parameters:**
- `env_file` (str): Path to .env file
- `**kwargs`: Additional parameters

**Returns:**
- Configuration instance

### CLI Commands

The CLI tool provides the following commands:

- `create`: Create new configuration file
- `validate`: Validate configuration
- `show`: Display configuration
- `check`: Check production readiness
- `convert`: Convert between formats
- `set`: Set configuration value
- `get`: Get configuration value
- `setup`: Setup environment

For detailed CLI usage, run:
```bash
python -m safla.utils.config_cli --help
```

## Troubleshooting

### Common Issues

#### 1. Pydantic Not Available

**Error:** `Pydantic not available. Install with: pip install pydantic pydantic-settings`

**Solution:**
```bash
pip install pydantic pydantic-settings
```

#### 2. Validation Errors

**Error:** `ValidationError: 1 validation error for SAFLAConfig`

**Solution:** Check the error message for specific field issues and fix the configuration values.

#### 3. Environment Variable Not Recognized

**Error:** Environment variable not being loaded

**Solution:** Ensure the variable name follows the `SAFLA_` prefix pattern and matches the configuration structure.

#### 4. Secret Values Not Loading

**Error:** Secret values showing as `SecretStr` objects

**Solution:** Use `.get_secret_value()` to access the actual secret:
```python
api_key = config.integration.openai_api_key.get_secret_value()
```

### Debug Mode

Enable debug mode to see detailed configuration loading:

```bash
export SAFLA_DEBUG=true
python your_app.py
```

### Logging

Enable configuration logging:

```python
import logging
logging.getLogger('safla.utils.config').setLevel(logging.DEBUG)
```

## Contributing

To contribute to the configuration system:

1. **Add new configuration options** in the appropriate config class
2. **Update environment variable mapping** in the ConfigLoader
3. **Add validation** for new fields
4. **Update documentation** and examples
5. **Add tests** for new functionality

### Adding New Configuration Options

1. Add to the appropriate config class:
```python
class PerformanceConfig(BaseModel):
    # Existing fields...
    new_option: int = Field(default=10, ge=1, le=100)
```

2. Update environment mapping:
```python
# In config_loader.py
"NEW_OPTION": "performance.new_option"
```

3. Add to .env template:
```bash
# New option description
SAFLA_NEW_OPTION=10
```

4. Add tests:
```python
def test_new_option():
    config = SAFLAConfig(performance={"new_option": 50})
    assert config.performance.new_option == 50
```

For more information, see the [SAFLA documentation](https://safla.readthedocs.io/).