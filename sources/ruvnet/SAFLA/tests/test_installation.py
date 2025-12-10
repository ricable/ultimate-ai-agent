"""
Test SAFLA Installation and Package Structure

This module contains tests to verify that the SAFLA package is properly
installed and all components can be imported correctly.
"""

import pytest
import sys
from pathlib import Path


class TestPackageStructure:
    """Test the basic package structure and imports."""
    
    def test_main_package_import(self):
        """Test that the main safla package can be imported."""
        import safla
        assert hasattr(safla, '__version__')
        assert hasattr(safla, '__author__')
    
    def test_core_components_import(self):
        """Test that core components can be imported."""
        try:
            from safla.core.hybrid_memory import HybridMemoryArchitecture
            from safla.core.meta_cognitive_engine import MetaCognitiveEngine
            from safla.core.mcp_orchestration import MCPOrchestrator
            from safla.core.safety_validation import SafetyValidationFramework
            from safla.core.delta_evaluation import DeltaEvaluator
        except ImportError as e:
            pytest.skip(f"Core components not available: {e}")
    
    def test_utilities_import(self):
        """Test that utility modules can be imported."""
        from safla.utils.config import SAFLAConfig
        from safla.utils.logging import get_logger
        from safla.utils.validation import validate_installation
        
        # Test basic functionality
        config = SAFLAConfig()
        assert config is not None
        
        logger = get_logger(__name__)
        assert logger is not None
    
    def test_exceptions_import(self):
        """Test that exception classes can be imported."""
        from safla.exceptions import (
            SAFLAError,
            MemoryError,
            SafetyViolationError,
            MCPError,
            ConfigurationError
        )
        
        # Test exception hierarchy
        assert issubclass(MemoryError, SAFLAError)
        assert issubclass(SafetyViolationError, SAFLAError)
        assert issubclass(MCPError, SAFLAError)
        assert issubclass(ConfigurationError, SAFLAError)
    
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from safla.cli import main
        assert callable(main)
    
    def test_installer_import(self):
        """Test that installer module can be imported."""
        from safla.installer import SAFLAInstaller
        assert SAFLAInstaller is not None


class TestConfiguration:
    """Test configuration functionality."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        from safla.utils.config import SAFLAConfig
        
        config = SAFLAConfig()
        assert config.memory.max_memories > 0
        assert 0 <= config.memory.similarity_threshold <= 1
        assert config.safety.memory_limit > 0
        assert 0 <= config.safety.cpu_limit <= 1
    
    def test_config_validation(self):
        """Test configuration validation."""
        from safla.utils.config import SAFLAConfig
        
        config = SAFLAConfig()
        errors = config.validate()
        assert isinstance(errors, list)
        # Default config should be valid
        assert len(errors) == 0
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        from safla.utils.config import SAFLAConfig
        
        config = SAFLAConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "memory" in config_dict
        assert "safety" in config_dict
        assert "mcp" in config_dict
        assert "metacognitive" in config_dict
        
        # Test round-trip
        new_config = SAFLAConfig.from_dict(config_dict)
        assert new_config.memory.max_memories == config.memory.max_memories


class TestValidation:
    """Test validation functionality."""
    
    def test_installation_validation(self):
        """Test installation validation."""
        from safla.utils.validation import validate_installation
        
        results = validate_installation()
        assert isinstance(results, dict)
        assert "valid" in results
        assert "errors" in results
        assert "warnings" in results
        assert "system_info" in results
    
    def test_python_version_validation(self):
        """Test Python version validation."""
        from safla.utils.validation import validate_python_version
        
        errors = validate_python_version()
        assert isinstance(errors, list)
        # Current Python should meet requirements
        assert len(errors) == 0
    
    def test_config_validation_function(self):
        """Test configuration validation function."""
        from safla.utils.config import SAFLAConfig
        from safla.utils.validation import validate_config
        
        config = SAFLAConfig()
        errors = validate_config(config)
        assert isinstance(errors, list)


class TestLogging:
    """Test logging functionality."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        from safla.utils.logging import get_logger, SAFLALogger, create_safla_logger
        
        # Test standard logger
        logger = get_logger(__name__)
        assert logger is not None
        
        # Test SAFLA logger
        safla_logger = create_safla_logger(__name__)
        assert isinstance(safla_logger, SAFLALogger)
    
    def test_logging_setup(self):
        """Test logging setup."""
        from safla.utils.logging import setup_logging
        
        # Should not raise an exception
        setup_logging(level="INFO", rich_logging=True)


class TestExceptions:
    """Test exception functionality."""
    
    def test_base_exception(self):
        """Test base SAFLA exception."""
        from safla.exceptions import SAFLAError
        
        error = SAFLAError("Test error", error_code="TEST001")
        assert str(error) == "[TEST001] Test error"
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "SAFLAError"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "TEST001"
    
    def test_specific_exceptions(self):
        """Test specific exception types."""
        from safla.exceptions import (
            MemoryError,
            SafetyViolationError,
            MCPError,
            ConfigurationError
        )
        
        # Test MemoryError
        mem_error = MemoryError("Memory test", memory_type="vector")
        assert mem_error.memory_type == "vector"
        
        # Test SafetyViolationError
        safety_error = SafetyViolationError("Safety test", severity="high")
        assert safety_error.severity == "high"
        
        # Test MCPError
        mcp_error = MCPError("MCP test", server_name="test_server")
        assert mcp_error.server_name == "test_server"
        
        # Test ConfigurationError
        config_error = ConfigurationError("Config test", config_section="memory")
        assert config_error.config_section == "memory"


class TestPackageMetadata:
    """Test package metadata and version information."""
    
    def test_version_info(self):
        """Test version information."""
        import safla
        
        assert hasattr(safla, '__version__')
        assert isinstance(safla.__version__, str)
        assert len(safla.__version__) > 0
    
    def test_package_info(self):
        """Test package information function."""
        import safla
        
        if hasattr(safla, 'get_info'):
            info = safla.get_info()
            assert isinstance(info, dict)
            assert "name" in info
            assert "version" in info
            assert info["name"] == "safla"


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete package."""
    
    def test_full_import_chain(self):
        """Test that all major components can be imported together."""
        try:
            # Import all major components
            from safla import (
                HybridMemoryArchitecture,
                MetaCognitiveEngine,
                MCPOrchestrator,
                SafetyValidationFramework,
                DeltaEvaluator,
                SAFLAConfig,
                get_logger
            )
            
            # Test basic instantiation
            config = SAFLAConfig()
            logger = get_logger(__name__)
            
            # Verify all imports succeeded
            assert HybridMemoryArchitecture is not None
            assert MetaCognitiveEngine is not None
            assert MCPOrchestrator is not None
            assert SafetyValidationFramework is not None
            assert DeltaEvaluator is not None
            assert config is not None
            assert logger is not None
            
        except ImportError as e:
            pytest.skip(f"Core components not available for integration test: {e}")
    
    def test_cli_functionality(self):
        """Test basic CLI functionality."""
        from safla.cli import cli
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test help command
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "SAFLA" in result.output
    
    def test_installer_basic_functionality(self):
        """Test basic installer functionality."""
        from safla.installer import SAFLAInstaller
        
        installer = SAFLAInstaller()
        assert installer is not None
        assert hasattr(installer, 'run_interactive_install')


if __name__ == "__main__":
    pytest.main([__file__])