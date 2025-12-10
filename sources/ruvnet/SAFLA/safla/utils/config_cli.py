"""
SAFLA Configuration CLI

Command-line interface for managing SAFLA configuration files,
validation, and environment setup.
"""

import os
import sys
import json
import click
import logging
from pathlib import Path
from typing import Optional

from .pydantic_config import SAFLAConfig
from .config_loader import (
    ConfigLoader,
    ConfigurationError,
    create_default_config_file,
    validate_config_file,
    load_config_from_env,
    load_config_from_json,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def config_cli(ctx, verbose):
    """SAFLA Configuration Management CLI."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@config_cli.command()
@click.option('--output', '-o', default='config.json', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'env']), default='json', help='Output format')
@click.pass_context
def create(ctx, output, format):
    """Create a default configuration file."""
    try:
        if format == 'json':
            create_default_config_file(output)
            click.echo(f"✅ Default configuration created: {output}")
        else:
            # Create default config and save as env file
            config = SAFLAConfig()
            loader = ConfigLoader()
            loader.save_config(config, output, format='env')
            click.echo(f"✅ Default environment file created: {output}")
            
    except Exception as e:
        click.echo(f"❌ Error creating configuration: {str(e)}", err=True)
        sys.exit(1)


@config_cli.command()
@click.argument('config_file')
@click.pass_context
def validate(ctx, config_file):
    """Validate a configuration file."""
    try:
        if validate_config_file(config_file):
            click.echo(f"✅ Configuration file is valid: {config_file}")
        else:
            click.echo(f"❌ Configuration file is invalid: {config_file}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error validating configuration: {str(e)}", err=True)
        sys.exit(1)


@config_cli.command()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--env-file', '-e', default='.env', help='Environment file path')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml', 'table']), default='table', help='Output format')
@click.option('--show-secrets', is_flag=True, help='Show secret values (use with caution)')
@click.pass_context
def show(ctx, config, env_file, format, show_secrets):
    """Show current configuration."""
    try:
        # Load configuration
        if config:
            config_obj = load_config_from_json(config)
        else:
            config_obj = load_config_from_env(env_file)
        
        # Display configuration
        if format == 'json':
            config_dict = config_obj.dict()
            if not show_secrets:
                config_dict = config_obj._exclude_secrets(config_dict)
            click.echo(json.dumps(config_dict, indent=2, default=str))
            
        elif format == 'yaml':
            try:
                import yaml
                config_dict = config_obj.dict()
                if not show_secrets:
                    config_dict = config_obj._exclude_secrets(config_dict)
                click.echo(yaml.dump(config_dict, default_flow_style=False))
            except ImportError:
                click.echo("❌ PyYAML not installed. Use 'pip install pyyaml'", err=True)
                sys.exit(1)
                
        else:  # table format
            _display_config_table(config_obj, show_secrets)
            
    except Exception as e:
        click.echo(f"❌ Error showing configuration: {str(e)}", err=True)
        sys.exit(1)


@config_cli.command()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--env-file', '-e', default='.env', help='Environment file path')
@click.pass_context
def check(ctx, config, env_file):
    """Check configuration for issues and provide recommendations."""
    try:
        # Load configuration
        if config:
            config_obj = load_config_from_json(config)
        else:
            config_obj = load_config_from_env(env_file)
        
        click.echo("🔍 Checking configuration...")
        
        # Check security
        security_warnings = config_obj.validate_security()
        if security_warnings:
            click.echo("\n⚠️  Security Warnings:")
            for warning in security_warnings:
                click.echo(f"  • {warning}")
        else:
            click.echo("\n✅ No security issues found")
        
        # Check production readiness
        if config_obj.is_production():
            click.echo("\n🏭 Production Mode Detected")
            _check_production_readiness(config_obj)
        else:
            click.echo("\n🛠️  Development Mode Detected")
            _check_development_setup(config_obj)
        
        # Check resource limits
        _check_resource_limits(config_obj)
        
        # Check directory structure
        _check_directories(config_obj)
        
        click.echo("\n✅ Configuration check complete")
        
    except Exception as e:
        click.echo(f"❌ Error checking configuration: {str(e)}", err=True)
        sys.exit(1)


@config_cli.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--format', '-f', type=click.Choice(['json', 'env']), required=True, help='Output format')
@click.option('--exclude-secrets', is_flag=True, default=True, help='Exclude secret values')
@click.pass_context
def convert(ctx, input_file, output_file, format, exclude_secrets):
    """Convert configuration between formats."""
    try:
        # Load configuration
        if input_file.endswith('.json'):
            config = load_config_from_json(input_file)
        else:
            # Assume it's an env file
            loader = ConfigLoader(env_file=input_file)
            config = loader.load_config(create_dirs=False)
        
        # Save in new format
        loader = ConfigLoader()
        loader.save_config(config, output_file, exclude_secrets=exclude_secrets, format=format)
        
        click.echo(f"✅ Configuration converted: {input_file} → {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Error converting configuration: {str(e)}", err=True)
        sys.exit(1)


@config_cli.command()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--env-file', '-e', default='.env', help='Environment file path')
@click.pass_context
def setup(ctx, config, env_file):
    """Setup SAFLA environment based on configuration."""
    try:
        # Load configuration
        if config:
            config_obj = load_config_from_json(config)
        else:
            config_obj = load_config_from_env(env_file)
        
        click.echo("🚀 Setting up SAFLA environment...")
        
        # Create directories
        config_obj.create_directories()
        click.echo("✅ Created necessary directories")
        
        # Check and install dependencies
        _check_dependencies(config_obj)
        
        # Setup database if needed
        _setup_database(config_obj)
        
        # Setup monitoring if enabled
        _setup_monitoring(config_obj)
        
        click.echo("✅ SAFLA environment setup complete")
        
    except Exception as e:
        click.echo(f"❌ Error setting up environment: {str(e)}", err=True)
        sys.exit(1)


@config_cli.command()
@click.option('--key', '-k', required=True, help='Configuration key (dot notation)')
@click.option('--value', '-v', required=True, help='New value')
@click.option('--config', '-c', help='Configuration file path')
@click.option('--env-file', '-e', default='.env', help='Environment file path')
@click.pass_context
def set(ctx, key, value, config, env_file):
    """Set a configuration value."""
    try:
        # Load configuration
        if config:
            config_obj = load_config_from_json(config)
            output_file = config
            output_format = 'json'
        else:
            config_obj = load_config_from_env(env_file)
            output_file = env_file
            output_format = 'env'
        
        # Convert value to appropriate type
        converted_value = _convert_string_value(value)
        
        # Set the value
        _set_nested_config_value(config_obj, key, converted_value)
        
        # Save configuration
        loader = ConfigLoader()
        loader.save_config(config_obj, output_file, format=output_format)
        
        click.echo(f"✅ Set {key} = {converted_value}")
        
    except Exception as e:
        click.echo(f"❌ Error setting configuration value: {str(e)}", err=True)
        sys.exit(1)


@config_cli.command()
@click.option('--key', '-k', required=True, help='Configuration key (dot notation)')
@click.option('--config', '-c', help='Configuration file path')
@click.option('--env-file', '-e', default='.env', help='Environment file path')
@click.pass_context
def get(ctx, key, config, env_file):
    """Get a configuration value."""
    try:
        # Load configuration
        if config:
            config_obj = load_config_from_json(config)
        else:
            config_obj = load_config_from_env(env_file)
        
        # Get the value
        value = _get_nested_config_value(config_obj, key)
        
        if value is not None:
            click.echo(f"{key} = {value}")
        else:
            click.echo(f"❌ Configuration key not found: {key}", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ Error getting configuration value: {str(e)}", err=True)
        sys.exit(1)


def _display_config_table(config: SAFLAConfig, show_secrets: bool = False):
    """Display configuration in table format."""
    click.echo("📋 SAFLA Configuration")
    click.echo("=" * 50)
    
    # General settings
    click.echo("\n🔧 General Settings:")
    click.echo(f"  Debug Mode: {config.debug}")
    click.echo(f"  Log Level: {config.log_level}")
    click.echo(f"  Monitoring: {config.enable_monitoring}")
    click.echo(f"  Data Directory: {config.data_dir}")
    click.echo(f"  Config Directory: {config.config_dir}")
    
    # Performance settings
    click.echo("\n⚡ Performance Settings:")
    click.echo(f"  Worker Threads: {config.performance.worker_threads}")
    click.echo(f"  Batch Size: {config.performance.batch_size}")
    click.echo(f"  Max Concurrent Ops: {config.performance.max_concurrent_ops}")
    click.echo(f"  Memory Pool Size: {config.performance.memory_pool_size}MB")
    click.echo(f"  Cache Size: {config.performance.cache_size}MB")
    
    # Memory settings
    click.echo("\n🧠 Memory Settings:")
    click.echo(f"  Vector Dimensions: {config.memory.vector_dimensions}")
    click.echo(f"  Max Memories: {config.memory.max_memories}")
    click.echo(f"  Similarity Threshold: {config.memory.similarity_threshold}")
    click.echo(f"  Consolidation Interval: {config.memory.consolidation_interval}s")
    
    # Safety settings
    click.echo("\n🛡️  Safety Settings:")
    click.echo(f"  Memory Limit: {config.safety.memory_limit / 1024 / 1024:.0f}MB")
    click.echo(f"  CPU Limit: {config.safety.cpu_limit * 100:.0f}%")
    click.echo(f"  Monitoring Interval: {config.safety.monitoring_interval}s")
    click.echo(f"  Rollback Enabled: {config.safety.rollback_enabled}")
    
    # Security settings
    click.echo("\n🔒 Security Settings:")
    click.echo(f"  Data Encryption: {config.security.encrypt_data}")
    click.echo(f"  Rate Limiting: {config.security.enable_rate_limiting}")
    click.echo(f"  CORS Enabled: {config.security.enable_cors}")
    click.echo(f"  SSL Enabled: {config.security.enable_ssl}")
    
    if show_secrets:
        click.echo("\n🔑 API Keys:")
        if config.integration.openai_api_key:
            click.echo(f"  OpenAI: {config.integration.openai_api_key}")
        if config.integration.anthropic_api_key:
            click.echo(f"  Anthropic: {config.integration.anthropic_api_key}")


def _check_production_readiness(config: SAFLAConfig):
    """Check if configuration is ready for production."""
    issues = []
    
    if config.debug:
        issues.append("Debug mode is enabled")
    
    if config.log_level == "DEBUG":
        issues.append("Log level is set to DEBUG")
    
    if not config.security.encrypt_data:
        issues.append("Data encryption is disabled")
    
    if not config.security.enable_rate_limiting:
        issues.append("Rate limiting is disabled")
    
    if not config.security.enable_ssl:
        issues.append("SSL is disabled")
    
    if issues:
        click.echo("  ⚠️  Production Issues:")
        for issue in issues:
            click.echo(f"    • {issue}")
    else:
        click.echo("  ✅ Configuration appears production-ready")


def _check_development_setup(config: SAFLAConfig):
    """Check development configuration."""
    recommendations = []
    
    if not config.debug:
        recommendations.append("Consider enabling debug mode for development")
    
    if config.log_level not in ["DEBUG", "INFO"]:
        recommendations.append("Consider using DEBUG or INFO log level for development")
    
    if config.security.enable_ssl:
        recommendations.append("SSL might not be necessary for local development")
    
    if recommendations:
        click.echo("  💡 Development Recommendations:")
        for rec in recommendations:
            click.echo(f"    • {rec}")


def _check_resource_limits(config: SAFLAConfig):
    """Check resource limit configuration."""
    click.echo("\n💾 Resource Limits:")
    
    memory_gb = config.safety.memory_limit / 1024 / 1024 / 1024
    click.echo(f"  Memory Limit: {memory_gb:.1f}GB")
    
    if memory_gb > 8:
        click.echo("    ⚠️  Very high memory limit (>8GB)")
    
    cpu_percent = config.safety.cpu_limit * 100
    click.echo(f"  CPU Limit: {cpu_percent:.0f}%")
    
    if cpu_percent > 90:
        click.echo("    ⚠️  Very high CPU limit (>90%)")


def _check_directories(config: SAFLAConfig):
    """Check directory structure."""
    click.echo("\n📁 Directory Structure:")
    
    directories = [
        config.data_dir,
        config.config_dir,
        f"{config.data_dir}/memory",
        f"{config.data_dir}/checkpoints",
        f"{config.data_dir}/logs"
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            click.echo(f"  ✅ {directory}")
        else:
            click.echo(f"  ❌ {directory} (missing)")


def _check_dependencies(config: SAFLAConfig):
    """Check for required dependencies."""
    click.echo("📦 Checking dependencies...")
    
    # Check for optional dependencies based on configuration
    if config.integration.redis_url.startswith("redis://"):
        try:
            import redis
            click.echo("  ✅ Redis client available")
        except ImportError:
            click.echo("  ⚠️  Redis client not installed (pip install redis)")
    
    if "postgresql://" in config.integration.database_url:
        try:
            import psycopg2
            click.echo("  ✅ PostgreSQL client available")
        except ImportError:
            click.echo("  ⚠️  PostgreSQL client not installed (pip install psycopg2-binary)")


def _setup_database(config: SAFLAConfig):
    """Setup database if needed."""
    if config.integration.database_url.startswith("sqlite://"):
        db_path = config.integration.database_url.replace("sqlite:///", "")
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"  ✅ SQLite database directory ready: {db_dir}")


def _setup_monitoring(config: SAFLAConfig):
    """Setup monitoring if enabled."""
    if config.monitoring.enable_metrics:
        click.echo("  📊 Metrics collection enabled")
    
    if config.monitoring.enable_health_checks:
        click.echo("  🏥 Health checks enabled")


def _convert_string_value(value: str):
    """Convert string value to appropriate type."""
    # Boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # List
    if ',' in value:
        items = [item.strip() for item in value.split(',')]
        try:
            return [int(item) for item in items]
        except ValueError:
            return items
    
    # Number
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value


def _set_nested_config_value(config: SAFLAConfig, key: str, value):
    """Set a nested configuration value."""
    keys = key.split('.')
    obj = config
    
    for k in keys[:-1]:
        obj = getattr(obj, k)
    
    setattr(obj, keys[-1], value)


def _get_nested_config_value(config: SAFLAConfig, key: str):
    """Get a nested configuration value."""
    keys = key.split('.')
    obj = config
    
    try:
        for k in keys:
            obj = getattr(obj, k)
        return obj
    except AttributeError:
        return None


if __name__ == '__main__':
    config_cli()