"""
Unit tests for configuration management.
"""

import pytest
import os
from unittest.mock import patch
from src.config.settings import AppSettings, PERFORMANCE_TARGETS, UI_CONFIG


class TestAppSettings:
    """Test application settings configuration."""
    
    def test_default_settings_initialization(self):
        """Test that default settings initialize correctly."""
        settings = AppSettings()
        
        assert settings.environment in ["development", "test", "staging", "production"]
        assert isinstance(settings.debug, bool)
        assert settings.safla_memory_size == 1000
        assert settings.safla_vector_dim in [512, 768, 1024, 1536]
        assert settings.max_concurrent_users > 0
        assert settings.cache_timeout > 0
    
    def test_environment_variable_loading(self):
        """Test loading settings from environment variables."""
        test_env = {
            "ENVIRONMENT": "test",
            "DEBUG": "true",
            "SAFLA_MEMORY_SIZE": "500",
            "SAFLA_VECTOR_DIM": "512",
            "MAX_CONCURRENT_USERS": "5"
        }
        
        with patch.dict(os.environ, test_env):
            settings = AppSettings()
            
            assert settings.environment == "test"
            assert settings.debug is True
            assert settings.safla_memory_size == 500
            assert settings.safla_vector_dim == 512
            assert settings.max_concurrent_users == 5
    
    def test_vector_dim_validation(self):
        """Test vector dimension validation."""
        # Valid dimensions
        for dim in [512, 768, 1024, 1536]:
            settings = AppSettings(safla_vector_dim=dim)
            assert settings.safla_vector_dim == dim
        
        # Invalid dimension
        with pytest.raises(ValueError, match="Vector dimension must be one of"):
            AppSettings(safla_vector_dim=256)
    
    def test_environment_validation(self):
        """Test environment setting validation."""
        # Valid environments
        for env in ["development", "test", "staging", "production"]:
            settings = AppSettings(environment=env)
            assert settings.environment == env
        
        # Invalid environment
        with pytest.raises(ValueError, match="Environment must be one of"):
            AppSettings(environment="invalid")
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = AppSettings(log_level=level)
            assert settings.log_level == level
        
        # Case insensitive
        settings = AppSettings(log_level="info")
        assert settings.log_level == "INFO"
        
        # Invalid log level
        with pytest.raises(ValueError, match="Log level must be one of"):
            AppSettings(log_level="INVALID")
    
    def test_get_safla_config(self):
        """Test SAFLA configuration retrieval."""
        settings = AppSettings(
            safla_memory_size=2000,
            safla_vector_dim=1024,
            environment="production"
        )
        
        config = settings.get_safla_config()
        
        assert config["memory_size"] == 2000
        assert config["vector_dimensions"] == 1024
        assert config["environment"] == "production"
        assert "config_path" in config
    
    def test_get_performance_config(self):
        """Test performance configuration retrieval."""
        settings = AppSettings(
            max_concurrent_users=20,
            cache_timeout=600,
            rate_limit_per_minute=120
        )
        
        config = settings.get_performance_config()
        
        assert config["max_concurrent_users"] == 20
        assert config["cache_timeout"] == 600
        assert config["rate_limit"] == 120
    
    def test_is_production(self):
        """Test production environment detection."""
        prod_settings = AppSettings(environment="production")
        assert prod_settings.is_production() is True
        
        dev_settings = AppSettings(environment="development")
        assert dev_settings.is_production() is False
    
    def test_is_debug(self):
        """Test debug mode detection."""
        # Explicit debug
        debug_settings = AppSettings(debug=True, environment="production")
        assert debug_settings.is_debug() is True
        
        # Debug via environment
        dev_settings = AppSettings(debug=False, environment="development")
        assert dev_settings.is_debug() is True
        
        test_settings = AppSettings(debug=False, environment="test")
        assert test_settings.is_debug() is True
        
        # No debug in production
        prod_settings = AppSettings(debug=False, environment="production")
        assert prod_settings.is_debug() is False
    
    def test_project_paths(self):
        """Test project path configuration."""
        settings = AppSettings()
        
        assert settings.project_root.exists()
        assert settings.project_root.is_dir()
        assert settings.assets_dir == settings.project_root / "assets"


class TestPerformanceTargets:
    """Test performance target configuration."""
    
    def test_performance_targets_defined(self):
        """Test that all performance targets are defined."""
        required_targets = [
            "initial_load_time",
            "memory_search_time",
            "safety_validation_time",
            "ui_response_time",
            "max_memory_usage",
            "min_concurrent_users"
        ]
        
        for target in required_targets:
            assert target in PERFORMANCE_TARGETS
            assert isinstance(PERFORMANCE_TARGETS[target], (int, float))
            assert PERFORMANCE_TARGETS[target] > 0
    
    def test_performance_targets_reasonable(self):
        """Test that performance targets are reasonable."""
        assert PERFORMANCE_TARGETS["initial_load_time"] <= 5.0  # 5 seconds max
        assert PERFORMANCE_TARGETS["memory_search_time"] <= 0.5  # 500ms max
        assert PERFORMANCE_TARGETS["safety_validation_time"] <= 0.1  # 100ms max
        assert PERFORMANCE_TARGETS["ui_response_time"] <= 0.1  # 100ms max
        assert PERFORMANCE_TARGETS["max_memory_usage"] <= 1024  # 1GB max
        assert PERFORMANCE_TARGETS["min_concurrent_users"] >= 5  # At least 5 users


class TestUIConfig:
    """Test UI configuration."""
    
    def test_ui_theme_config(self):
        """Test UI theme configuration."""
        theme = UI_CONFIG.get("theme", {})
        
        required_colors = [
            "primary_color",
            "secondary_color",
            "success_color",
            "warning_color",
            "error_color"
        ]
        
        for color in required_colors:
            assert color in theme
            assert theme[color].startswith("#")  # Valid hex color
            assert len(theme[color]) == 7  # #RRGGBB format
        
        assert "font_family" in theme
        assert isinstance(theme["font_family"], str)
    
    def test_ui_layout_config(self):
        """Test UI layout configuration."""
        layout = UI_CONFIG.get("layout", {})
        
        assert "max_width" in layout
        assert "default_tab" in layout
        assert "show_footer" in layout
        
        assert layout["default_tab"] in ["demo", "settings", "benchmarks", "docs"]
        assert isinstance(layout["show_footer"], bool)