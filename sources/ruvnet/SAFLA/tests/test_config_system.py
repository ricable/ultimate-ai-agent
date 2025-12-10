"""
Tests for SAFLA Configuration System

This module contains comprehensive tests for the Pydantic-based configuration
system including validation, loading, and security features.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from safla.utils.pydantic_config import (
    SAFLAConfig,
    PerformanceConfig,
    MemoryConfig,
    SafetyConfig,
    MCPConfig,
    MetaCognitiveConfig,
    IntegrationConfig,
    SecurityConfig,
    NetworkingConfig,
    DevelopmentConfig,
    ExperimentalConfig,
    MonitoringConfig,
    get_config,
    set_config,
    reset_config,
)
from safla.utils.config_loader import (
    ConfigLoader,
    EnvironmentMapper,
    ConfigurationError,
    create_default_config_file,
    validate_config_file,
    load_config_from_env,
    load_config_from_json,
)


class TestPydanticConfig:
    """Test cases for Pydantic configuration classes."""
    
    def test_default_config_creation(self):
        """Test creating configuration with default values."""
        config = SAFLAConfig()
        
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.enable_monitoring is True
        assert config.data_dir == "./data"
        assert config.config_dir == "./.roo"
        
        # Test nested configurations
        assert config.performance.worker_threads == 4
        assert config.memory.max_memories == 10000
        assert config.safety.memory_limit == 1000000000
        assert config.mcp.timeout == 30
        assert config.metacognitive.adaptation_rate == 0.1
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = SAFLAConfig(
            performance=PerformanceConfig(worker_threads=8),
            memory=MemoryConfig(similarity_threshold=0.9),
            safety=SafetyConfig(cpu_limit=0.8)
        )
        assert config.performance.worker_threads == 8
        assert config.memory.similarity_threshold == 0.9
        assert config.safety.cpu_limit == 0.8
    
    def test_invalid_config_validation(self):
        """Test that invalid configurations raise validation errors."""
        # Test invalid worker threads (too high)
        with pytest.raises(ValueError):
            PerformanceConfig(worker_threads=100)
        
        # Test invalid similarity threshold (out of range)
        with pytest.raises(ValueError):
            MemoryConfig(similarity_threshold=1.5)
        
        # Test invalid CPU limit (out of range)
        with pytest.raises(ValueError):
            SafetyConfig(cpu_limit=1.5)
        
        # Test invalid log level
        with pytest.raises(ValueError):
            SAFLAConfig(log_level="INVALID")
    
    def test_memory_config_validation(self):
        """Test memory configuration specific validation."""
        # Test empty vector dimensions
        with pytest.raises(ValueError, match="Vector dimensions cannot be empty"):
            MemoryConfig(vector_dimensions=[])
        
        # Test negative vector dimensions
        with pytest.raises(ValueError, match="All vector dimensions must be positive"):
            MemoryConfig(vector_dimensions=[512, -768, 1024])
    
    def test_security_validation(self):
        """Test security configuration validation."""
        config = SAFLAConfig()
        warnings = config.validate_security()
        
        # Should have warnings about missing keys
        assert len(warnings) > 0
        assert any("encryption key" in warning.lower() for warning in warnings)
        assert any("jwt secret" in warning.lower() for warning in warnings)
    
    def test_config_serialization(self):
        """Test configuration serialization to/from dict."""
        config = SAFLAConfig(
            debug=True,
            log_level="DEBUG",
            performance=PerformanceConfig(worker_threads=8),
            memory=MemoryConfig(max_memories=5000)
        )
        
        # Test to dict
        config_dict = config.model_dump()
        assert config_dict["debug"] is True
        assert config_dict["log_level"] == "DEBUG"
        assert config_dict["performance"]["worker_threads"] == 8
        assert config_dict["memory"]["max_memories"] == 5000
        
        # Test from dict
        new_config = SAFLAConfig(**config_dict)
        assert new_config.debug is True
        assert new_config.log_level == "DEBUG"
        assert new_config.performance.worker_threads == 8
        assert new_config.memory.max_memories == 5000
    
    def test_secret_handling(self):
        """Test that secrets are properly handled."""
        config = SAFLAConfig(
            integration=IntegrationConfig(
                openai_api_key="secret-key-123",
                anthropic_api_key="another-secret"
            ),
            security=SecurityConfig(
                encryption_key="encryption-secret",
                jwt_secret_key="jwt-secret"
            )
        )
        
        # Test that secrets are SecretStr objects
        assert config.integration.openai_api_key.get_secret_value() == "secret-key-123"
        assert config.security.encryption_key.get_secret_value() == "encryption-secret"
    
    def test_database_url_selection(self):
        """Test database URL selection based on mode."""
        config = SAFLAConfig(
            integration=IntegrationConfig(database_url="sqlite:///prod.db"),
            development=DevelopmentConfig(
                test_mode=False,
                test_database_url="sqlite:///test.db"
            )
        )
        
        # Test production mode
        assert config.get_database_url() == "sqlite:///prod.db"
        
        # Test test mode
        assert config.get_database_url(test_mode=True) == "sqlite:///test.db"
        
        # Test when test_mode is enabled in config
        config.development.test_mode = True
        assert config.get_database_url() == "sqlite:///test.db"
    
    def test_production_mode_detection(self):
        """Test production mode detection."""
        # Default config should be production
        config = SAFLAConfig()
        assert config.is_production() is True
        
        # Debug mode should not be production
        config.debug = True
        assert config.is_production() is False
        
        # Dev mode should not be production
        config.debug = False
        config.development.dev_mode = True
        assert config.is_production() is False
        
        # Test mode should not be production
        config.development.dev_mode = False
        config.development.test_mode = True
        assert config.is_production() is False


class TestEnvironmentMapper:
    """Test cases for environment variable mapping."""
    
    def test_env_mapping(self):
        """Test environment variable to config mapping."""
        with patch.dict(os.environ, {
            "SAFLA_WORKER_THREADS": "8",
            "SAFLA_DEBUG": "true",
            "SAFLA_SIMILARITY_THRESHOLD": "0.9",
            "SAFLA_VECTOR_DIMENSIONS": "256,512,1024"
        }):
            config_dict = EnvironmentMapper.map_env_to_config()
            
            assert config_dict["performance"]["worker_threads"] == 8
            assert config_dict["memory"]["similarity_threshold"] == 0.9
            assert config_dict["memory"]["vector_dimensions"] == [256, 512, 1024]
    
    def test_value_conversion(self):
        """Test environment variable value conversion."""
        # Test boolean conversion
        assert EnvironmentMapper._convert_value("true") is True
        assert EnvironmentMapper._convert_value("false") is False
        assert EnvironmentMapper._convert_value("True") is True
        assert EnvironmentMapper._convert_value("FALSE") is False
        
        # Test integer conversion
        assert EnvironmentMapper._convert_value("42") == 42
        assert EnvironmentMapper._convert_value("0") == 0
        
        # Test float conversion
        assert EnvironmentMapper._convert_value("3.14") == 3.14
        assert EnvironmentMapper._convert_value("0.5") == 0.5
        
        # Test list conversion
        assert EnvironmentMapper._convert_value("1,2,3") == [1, 2, 3]
        assert EnvironmentMapper._convert_value("a,b,c") == ["a", "b", "c"]
        
        # Test string (no conversion)
        assert EnvironmentMapper._convert_value("hello") == "hello"
    
    def test_nested_value_setting(self):
        """Test setting nested values in configuration dictionary."""
        config_dict = {}
        EnvironmentMapper._set_nested_value(config_dict, "performance.worker_threads", 8)
        EnvironmentMapper._set_nested_value(config_dict, "memory.max_memories", 5000)
        
        assert config_dict["performance"]["worker_threads"] == 8
        assert config_dict["memory"]["max_memories"] == 5000


class TestConfigLoader:
    """Test cases for configuration loader."""
    
    def test_config_loader_initialization(self):
        """Test config loader initialization."""
        loader = ConfigLoader()
        assert loader.env_file == ".env"
        
        loader = ConfigLoader(env_file="custom.env")
        assert loader.env_file == "custom.env"
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        loader = ConfigLoader()
        config = loader.load_config(create_dirs=False)
        
        assert isinstance(config, SAFLAConfig)
        assert config.debug is False
        assert config.log_level == "INFO"
    
    def test_load_config_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "debug": True,
            "log_level": "DEBUG",
            "performance": {
                "worker_threads": 8,
                "batch_size": 64
            },
            "memory": {
                "max_memories": 5000,
                "similarity_threshold": 0.9
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            loader = ConfigLoader()
            config = loader.load_config(config_file=config_file, env_override=False, create_dirs=False)
            
            assert config.debug is True
            assert config.log_level == "DEBUG"
            assert config.performance.worker_threads == 8
            assert config.performance.batch_size == 64
            assert config.memory.max_memories == 5000
            assert config.memory.similarity_threshold == 0.9
        finally:
            os.unlink(config_file)
    
    def test_env_override(self):
        """Test environment variable override functionality."""
        config_data = {
            "debug": False,
            "performance": {
                "worker_threads": 4
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            with patch.dict(os.environ, {
                "SAFLA_DEBUG": "true",
                "SAFLA_WORKER_THREADS": "8"
            }):
                loader = ConfigLoader()
                config = loader.load_config(
                    config_file=config_file,
                    env_override=True,
                    create_dirs=False
                )
                
                # Environment should override file values
                assert config.debug is True
                assert config.performance.worker_threads == 8
        finally:
            os.unlink(config_file)
    
    def test_deep_merge(self):
        """Test deep merging of configuration dictionaries."""
        loader = ConfigLoader()
        
        base = {
            "debug": False,
            "performance": {
                "worker_threads": 4,
                "batch_size": 32
            },
            "memory": {
                "max_memories": 10000
            }
        }
        
        override = {
            "debug": True,
            "performance": {
                "worker_threads": 8
            },
            "safety": {
                "cpu_limit": 0.8
            }
        }
        
        result = loader._deep_merge(base, override)
        
        assert result["debug"] is True
        assert result["performance"]["worker_threads"] == 8
        assert result["performance"]["batch_size"] == 32  # Preserved from base
        assert result["memory"]["max_memories"] == 10000  # Preserved from base
        assert result["safety"]["cpu_limit"] == 0.8  # Added from override
    
    def test_save_config_json(self):
        """Test saving configuration to JSON file."""
        config = SAFLAConfig(
            debug=True,
            log_level="DEBUG",
            performance=PerformanceConfig(worker_threads=8)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            loader = ConfigLoader()
            loader.save_config(config, output_file, format="json")
            
            # Verify file was created and contains correct data
            assert Path(output_file).exists()
            
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["debug"] is True
            assert saved_data["log_level"] == "DEBUG"
            assert saved_data["performance"]["worker_threads"] == 8
        finally:
            if Path(output_file).exists():
                os.unlink(output_file)
    
    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        loader = ConfigLoader()
        
        # Test loading non-existent JSON file
        with pytest.raises(ConfigurationError):
            loader.load_config(config_file="non_existent.json", create_dirs=False)
        
        # Test saving to invalid path
        config = SAFLAConfig()
        with pytest.raises(ConfigurationError):
            loader.save_config(config, "/invalid/path/config.json")


class TestGlobalConfigManagement:
    """Test cases for global configuration management."""
    
    def test_get_set_reset_config(self):
        """Test global configuration management functions."""
        # Reset to ensure clean state
        reset_config()
        
        # Test get_config creates default
        config1 = get_config()
        assert isinstance(config1, SAFLAConfig)
        
        # Test that subsequent calls return same instance
        config2 = get_config()
        assert config1 is config2
        
        # Test set_config
        new_config = SAFLAConfig(debug=True)
        set_config(new_config)
        
        config3 = get_config()
        assert config3 is new_config
        assert config3.debug is True
        
        # Test reset_config
        reset_config()
        config4 = get_config()
        assert config4 is not new_config
        assert config4.debug is False  # Back to default


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_create_default_config_file(self):
        """Test creating default configuration file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            create_default_config_file(output_file)
            
            # Verify file was created
            assert Path(output_file).exists()
            
            # Verify it contains valid configuration
            with open(output_file, 'r') as f:
                config_data = json.load(f)
            
            assert "debug" in config_data
            assert "performance" in config_data
            assert "memory" in config_data
        finally:
            if Path(output_file).exists():
                os.unlink(output_file)
    
    def test_validate_config_file(self):
        """Test configuration file validation."""
        # Test valid configuration
        valid_config = {
            "debug": True,
            "log_level": "DEBUG",
            "performance": {
                "worker_threads": 4
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config, f)
            valid_file = f.name
        
        # Test invalid configuration
        invalid_config = {
            "debug": True,
            "log_level": "INVALID_LEVEL",
            "performance": {
                "worker_threads": 100  # Too high
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            invalid_file = f.name
        
        try:
            # Valid config should pass
            assert validate_config_file(valid_file) is True
            
            # Invalid config should fail
            assert validate_config_file(invalid_file) is False
        finally:
            os.unlink(valid_file)
            os.unlink(invalid_file)
    
    def test_load_config_convenience_functions(self):
        """Test convenience functions for loading configuration."""
        # Test load_config_from_env
        with patch.dict(os.environ, {"SAFLA_DEBUG": "true"}):
            config = load_config_from_env()
            assert config.debug is True
        
        # Test load_config_from_json
        config_data = {"debug": True, "log_level": "DEBUG"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = load_config_from_json(config_file)
            assert config.debug is True
            assert config.log_level == "DEBUG"
        finally:
            os.unlink(config_file)


class TestIntegrationScenarios:
    """Test cases for real-world integration scenarios."""
    
    def test_production_config_scenario(self):
        """Test a typical production configuration scenario."""
        with patch.dict(os.environ, {
            "SAFLA_DEBUG": "false",
            "SAFLA_LOG_LEVEL": "WARNING",
            "SAFLA_WORKER_THREADS": "16",
            "SAFLA_MAX_MEMORIES": "50000",
            "SAFLA_ENABLE_SSL": "true",
            "SAFLA_ENCRYPT_DATA": "true",
            "DATABASE_URL": "postgresql://user:pass@localhost/safla",
            "REDIS_URL": "redis://localhost:6379/0"
        }):
            config = SAFLAConfig()
            
            assert config.debug is False
            assert config.log_level == "WARNING"
            assert config.performance.worker_threads == 16
            assert config.memory.max_memories == 50000
            assert config.security.enable_ssl is True
            assert config.security.encrypt_data is True
            assert config.integration.database_url == "postgresql://user:pass@localhost/safla"
            assert config.integration.redis_url == "redis://localhost:6379/0"
            
            # Should be detected as production
            assert config.is_production() is True
    
    def test_development_config_scenario(self):
        """Test a typical development configuration scenario."""
        with patch.dict(os.environ, {
            "SAFLA_DEBUG": "true",
            "SAFLA_LOG_LEVEL": "DEBUG",
            "SAFLA_DEV_MODE": "true",
            "SAFLA_HOT_RELOAD": "true",
            "SAFLA_ENABLE_PROFILING": "true",
            "SAFLA_WORKER_THREADS": "2",
            "DATABASE_URL": "sqlite:///./dev.db"
        }):
            config = SAFLAConfig()
            
            assert config.debug is True
            assert config.log_level == "DEBUG"
            assert config.development.dev_mode is True
            assert config.development.hot_reload is True
            assert config.development.enable_profiling is True
            assert config.performance.worker_threads == 2
            assert config.integration.database_url == "sqlite:///./dev.db"
            
            # Should not be detected as production
            assert config.is_production() is False
    
    def test_config_file_with_env_overrides(self):
        """Test loading from config file with environment overrides."""
        # Base configuration file
        base_config = {
            "debug": False,
            "log_level": "INFO",
            "performance": {
                "worker_threads": 4,
                "batch_size": 32
            },
            "memory": {
                "max_memories": 10000,
                "similarity_threshold": 0.8
            },
            "integration": {
                "database_url": "sqlite:///./default.db"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(base_config, f)
            config_file = f.name
        
        try:
            # Environment overrides
            with patch.dict(os.environ, {
                "SAFLA_DEBUG": "true",
                "SAFLA_WORKER_THREADS": "8",
                "DATABASE_URL": "postgresql://localhost/override"
            }):
                loader = ConfigLoader()
                config = loader.load_config(config_file=config_file, create_dirs=False)
                
                # Environment should override file values
                assert config.debug is True  # Overridden
                assert config.performance.worker_threads == 8  # Overridden
                assert config.integration.database_url == "postgresql://localhost/override"  # Overridden
                
                # File values should be preserved where not overridden
                assert config.log_level == "INFO"  # From file
                assert config.performance.batch_size == 32  # From file
                assert config.memory.max_memories == 10000  # From file
                assert config.memory.similarity_threshold == 0.8  # From file
        finally:
            os.unlink(config_file)


if __name__ == "__main__":
    pytest.main([__file__])