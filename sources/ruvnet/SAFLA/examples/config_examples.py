"""
SAFLA Configuration System Examples

This script demonstrates various ways to use the SAFLA configuration system,
including loading from different sources, validation, and best practices.
"""

import os
import json
import tempfile
from pathlib import Path

# Import both legacy and new configuration systems
from safla.utils.config import (
    get_config,
    set_config,
    reset_config,
    get_pydantic_config,
    migrate_to_pydantic,
    load_config_from_file,
    PYDANTIC_AVAILABLE
)

if PYDANTIC_AVAILABLE:
    from safla.utils.pydantic_config import SAFLAConfig
    from safla.utils.config_loader import ConfigLoader, load_config_from_env


def example_basic_usage():
    """Demonstrate basic configuration usage."""
    print("=" * 60)
    print("BASIC CONFIGURATION USAGE")
    print("=" * 60)
    
    # Get default configuration
    config = get_config()
    print(f"Debug mode: {config.debug}")
    print(f"Log level: {config.log_level}")
    print(f"Data directory: {config.data_dir}")
    
    # Access nested configuration
    print(f"Worker threads: {config.performance.worker_threads}")
    print(f"Memory limit: {config.safety.memory_limit / 1024 / 1024:.0f}MB")
    print(f"MCP timeout: {config.mcp.timeout}s")


def example_environment_variables():
    """Demonstrate loading configuration from environment variables."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLE CONFIGURATION")
    print("=" * 60)
    
    # Set some environment variables
    os.environ.update({
        "SAFLA_DEBUG": "true",
        "SAFLA_LOG_LEVEL": "DEBUG",
        "SAFLA_WORKER_THREADS": "8",
        "SAFLA_MAX_MEMORIES": "20000",
        "SAFLA_CPU_LIMIT": "0.8"
    })
    
    # Reset and reload configuration
    reset_config()
    config = get_config()
    
    print(f"Debug mode (from env): {config.debug}")
    print(f"Log level (from env): {config.log_level}")
    print(f"Worker threads (from env): {config.performance.worker_threads}")
    print(f"Max memories (from env): {config.memory.max_memories}")
    print(f"CPU limit (from env): {config.safety.cpu_limit}")
    
    # Clean up environment variables
    for key in ["SAFLA_DEBUG", "SAFLA_LOG_LEVEL", "SAFLA_WORKER_THREADS", 
                "SAFLA_MAX_MEMORIES", "SAFLA_CPU_LIMIT"]:
        os.environ.pop(key, None)


def example_json_configuration():
    """Demonstrate loading configuration from JSON file."""
    print("\n" + "=" * 60)
    print("JSON FILE CONFIGURATION")
    print("=" * 60)
    
    # Create a sample configuration file
    config_data = {
        "debug": True,
        "log_level": "INFO",
        "data_dir": "./custom_data",
        "performance": {
            "worker_threads": 6,
            "batch_size": 64,
            "enable_optimizations": True
        },
        "memory": {
            "max_memories": 15000,
            "similarity_threshold": 0.85,
            "compression": True
        },
        "safety": {
            "memory_limit": 2000000000,  # 2GB
            "cpu_limit": 0.85,
            "rollback_enabled": True
        },
        "security": {
            "encrypt_data": True,
            "enable_rate_limiting": True,
            "api_rate_limit": 500
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, indent=2)
        config_file = f.name
    
    try:
        # Load configuration from file
        config = load_config_from_file(config_file)
        
        print(f"Loaded from: {config_file}")
        print(f"Debug mode: {config.debug}")
        print(f"Data directory: {config.data_dir}")
        print(f"Worker threads: {config.performance.worker_threads}")
        print(f"Batch size: {config.performance.batch_size}")
        print(f"Max memories: {config.memory.max_memories}")
        print(f"Memory limit: {config.safety.memory_limit / 1024 / 1024:.0f}MB")
        print(f"Data encryption: {config.security.encrypt_data}")
        
    finally:
        # Clean up
        os.unlink(config_file)


def example_validation():
    """Demonstrate configuration validation."""
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)
    
    if not PYDANTIC_AVAILABLE:
        print("Pydantic not available - skipping validation example")
        return
    
    try:
        # Create configuration with valid values
        config = SAFLAConfig(
            debug=True,
            performance={"worker_threads": 4, "batch_size": 32},
            memory={"similarity_threshold": 0.8, "max_memories": 10000},
            safety={"cpu_limit": 0.9, "memory_limit": 1000000000}
        )
        print("✅ Valid configuration created successfully")
        
        # Try to create configuration with invalid values
        try:
            invalid_config = SAFLAConfig(
                performance={"worker_threads": 100},  # Too high
                memory={"similarity_threshold": 1.5},  # Out of range
                safety={"cpu_limit": 1.5}  # Out of range
            )
        except Exception as e:
            print(f"❌ Invalid configuration rejected: {type(e).__name__}")
            print(f"   Error details: {str(e)}")
        
        # Check security validation
        security_warnings = config.validate_security()
        if security_warnings:
            print("\n⚠️  Security warnings:")
            for warning in security_warnings:
                print(f"   • {warning}")
        
    except Exception as e:
        print(f"❌ Validation example failed: {str(e)}")


def example_secret_handling():
    """Demonstrate secure handling of sensitive information."""
    print("\n" + "=" * 60)
    print("SECRET HANDLING")
    print("=" * 60)
    
    if not PYDANTIC_AVAILABLE:
        print("Pydantic not available - skipping secret handling example")
        return
    
    # Create configuration with secrets
    config = SAFLAConfig(
        integration={
            "openai_api_key": "sk-test-key-123",
            "anthropic_api_key": "claude-key-456",
            "database_url": "postgresql://user:password@localhost/db"
        },
        security={
            "encryption_key": "super-secret-key",
            "jwt_secret_key": "jwt-secret-123"
        }
    )
    
    print("Configuration with secrets created")
    
    # Show how secrets are handled
    print(f"OpenAI API Key type: {type(config.integration.openai_api_key)}")
    print(f"OpenAI API Key value: {config.integration.openai_api_key}")
    
    # Save configuration excluding secrets
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_file = f.name
    
    try:
        config.save_to_json_file(config_file, exclude_secrets=True)
        
        # Read and show the saved file
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        print("\nSaved configuration (secrets excluded):")
        print(f"OpenAI API Key: {saved_config['integration'].get('openai_api_key', 'NOT FOUND')}")
        print(f"Encryption Key: {saved_config['security'].get('encryption_key', 'NOT FOUND')}")
        
    finally:
        os.unlink(config_file)


def example_production_vs_development():
    """Demonstrate production vs development configuration patterns."""
    print("\n" + "=" * 60)
    print("PRODUCTION VS DEVELOPMENT CONFIGURATION")
    print("=" * 60)
    
    if not PYDANTIC_AVAILABLE:
        print("Pydantic not available - skipping production/development example")
        return
    
    # Development configuration
    dev_config = SAFLAConfig(
        debug=True,
        log_level="DEBUG",
        development={
            "dev_mode": True,
            "hot_reload": True,
            "enable_profiling": True
        },
        performance={
            "worker_threads": 2,
            "batch_size": 16
        },
        security={
            "enable_ssl": False,
            "encrypt_data": False
        }
    )
    
    print("Development Configuration:")
    print(f"  Debug: {dev_config.debug}")
    print(f"  Log Level: {dev_config.log_level}")
    print(f"  Dev Mode: {dev_config.development.dev_mode}")
    print(f"  SSL Enabled: {dev_config.security.enable_ssl}")
    print(f"  Is Production: {dev_config.is_production()}")
    
    # Production configuration
    prod_config = SAFLAConfig(
        debug=False,
        log_level="WARNING",
        development={
            "dev_mode": False,
            "hot_reload": False,
            "enable_profiling": False
        },
        performance={
            "worker_threads": 16,
            "batch_size": 128
        },
        security={
            "enable_ssl": True,
            "encrypt_data": True,
            "enable_rate_limiting": True
        }
    )
    
    print("\nProduction Configuration:")
    print(f"  Debug: {prod_config.debug}")
    print(f"  Log Level: {prod_config.log_level}")
    print(f"  Dev Mode: {prod_config.development.dev_mode}")
    print(f"  SSL Enabled: {prod_config.security.enable_ssl}")
    print(f"  Is Production: {prod_config.is_production()}")


def example_migration():
    """Demonstrate migrating from legacy to Pydantic configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION MIGRATION")
    print("=" * 60)
    
    if not PYDANTIC_AVAILABLE:
        print("Pydantic not available - skipping migration example")
        return
    
    # Start with legacy configuration
    reset_config()
    legacy_config = get_config(use_pydantic=False)
    print(f"Legacy config type: {type(legacy_config).__name__}")
    print(f"Legacy debug mode: {legacy_config.debug}")
    
    # Migrate to Pydantic
    try:
        pydantic_config = migrate_to_pydantic()
        print(f"Migrated config type: {type(pydantic_config).__name__}")
        print(f"Migrated debug mode: {pydantic_config.debug}")
        print("✅ Migration successful")
        
        # Show additional features available in Pydantic config
        print(f"Security validation available: {hasattr(pydantic_config, 'validate_security')}")
        print(f"Production detection available: {hasattr(pydantic_config, 'is_production')}")
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")


def example_advanced_features():
    """Demonstrate advanced configuration features."""
    print("\n" + "=" * 60)
    print("ADVANCED CONFIGURATION FEATURES")
    print("=" * 60)
    
    if not PYDANTIC_AVAILABLE:
        print("Pydantic not available - skipping advanced features example")
        return
    
    # Create configuration with all features
    config = SAFLAConfig(
        performance={
            "worker_threads": 8,
            "batch_size": 64,
            "enable_optimizations": True
        },
        memory={
            "vector_dimensions": [256, 512, 1024, 2048],
            "max_memories": 50000,
            "compression": True
        },
        experimental={
            "enable_experimental": True,
            "enable_advanced_ml": True,
            "enable_distributed": True,
            "cluster_nodes": ["node1:8000", "node2:8000", "node3:8000"]
        },
        monitoring={
            "enable_metrics": True,
            "enable_tracing": True,
            "metrics_interval": 30
        }
    )
    
    # Directory creation
    print("Creating directories...")
    config.create_directories()
    print(f"✅ Directories created in: {config.data_dir}")
    
    # Database URL selection
    print(f"Production database: {config.get_database_url()}")
    print(f"Test database: {config.get_database_url(test_mode=True)}")
    
    # Feature flags
    print(f"Experimental features: {config.experimental.enable_experimental}")
    print(f"Advanced ML: {config.experimental.enable_advanced_ml}")
    print(f"Distributed processing: {config.experimental.enable_distributed}")
    print(f"Cluster nodes: {len(config.experimental.cluster_nodes)}")
    
    # Monitoring
    print(f"Metrics enabled: {config.monitoring.enable_metrics}")
    print(f"Tracing enabled: {config.monitoring.enable_tracing}")


def example_config_loader():
    """Demonstrate the ConfigLoader utility."""
    print("\n" + "=" * 60)
    print("CONFIG LOADER UTILITY")
    print("=" * 60)
    
    if not PYDANTIC_AVAILABLE:
        print("Pydantic not available - skipping ConfigLoader example")
        return
    
    # Create sample .env file
    env_content = """
# Sample SAFLA configuration
SAFLA_DEBUG=true
SAFLA_LOG_LEVEL=DEBUG
SAFLA_WORKER_THREADS=6
SAFLA_MAX_MEMORIES=25000
SAFLA_ENABLE_OPTIMIZATIONS=true
DATABASE_URL=sqlite:///./example.db
REDIS_URL=redis://localhost:6379/1
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(env_content)
        env_file = f.name
    
    try:
        # Use ConfigLoader
        loader = ConfigLoader(env_file=env_file)
        config = loader.load_config(create_dirs=False)
        
        print(f"Loaded from env file: {env_file}")
        print(f"Debug: {config.debug}")
        print(f"Log level: {config.log_level}")
        print(f"Worker threads: {config.performance.worker_threads}")
        print(f"Max memories: {config.memory.max_memories}")
        print(f"Database URL: {config.integration.database_url}")
        print(f"Redis URL: {config.integration.redis_url}")
        
        # Save configuration in different format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            loader.save_config(config, output_file, format='json')
            print(f"✅ Configuration saved to: {output_file}")
            
            # Verify saved file
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            print(f"Saved config has {len(saved_data)} top-level keys")
            
        finally:
            os.unlink(output_file)
        
    finally:
        os.unlink(env_file)


def main():
    """Run all configuration examples."""
    print("SAFLA Configuration System Examples")
    print("This script demonstrates various configuration features and usage patterns.")
    
    try:
        example_basic_usage()
        example_environment_variables()
        example_json_configuration()
        example_validation()
        example_secret_handling()
        example_production_vs_development()
        example_migration()
        example_advanced_features()
        example_config_loader()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        if PYDANTIC_AVAILABLE:
            print("\n💡 Tips for using the configuration system:")
            print("  • Use the Pydantic-based SAFLAConfig for new projects")
            print("  • Store secrets in environment variables, not config files")
            print("  • Use different configurations for development and production")
            print("  • Validate your configuration before deployment")
            print("  • Use the CLI tool: python -m safla.utils.config_cli --help")
        else:
            print("\n💡 Install Pydantic for enhanced features:")
            print("  pip install pydantic pydantic-settings")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()