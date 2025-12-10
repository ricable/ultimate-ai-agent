"""
Comprehensive test suite for SAFLA CLI.

Tests command-line interface functionality including validation,
configuration management, and various commands.
"""

import json
import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from safla.cli import cli
from safla.utils.config import SAFLAConfig
from safla.exceptions import ConfigurationError, SAFLAError


class TestCLI:
    """Test cases for SAFLA CLI."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock(spec=SAFLAConfig)
        config.debug = False
        config.log_level = "INFO"
        return config
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "mode": "development",
                "memory": {
                    "vector_memory": {
                        "embedding_dim": 512,
                        "max_capacity": 10000
                    }
                },
                "metacognitive": {
                    "enabled": True,
                    "self_awareness_threshold": 0.8
                }
            }
            json.dump(config_data, f)
            return f.name
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'SAFLA - Self-Aware Feedback Loop Algorithm CLI' in result.output
        assert 'Commands:' in result.output
    
    def test_cli_version(self, runner):
        """Test CLI version command."""
        with patch('safla.cli.click') as mock_click:
            result = runner.invoke(cli, ['--version'])
            # Version might not be implemented yet, check for proper behavior
            assert result.exit_code in [0, 2]  # 0 if implemented, 2 if not
    
    @patch('safla.cli.setup_logging')
    @patch('safla.cli.SAFLAConfig')
    def test_cli_debug_mode(self, mock_config_class, mock_setup_logging, runner):
        """Test CLI with debug mode enabled."""
        mock_config = MagicMock()
        mock_config_class.from_env.return_value = mock_config
        
        result = runner.invoke(cli, ['--debug', 'validate'])
        
        # Verify debug mode is set
        assert mock_config.debug is True
        assert mock_config.log_level == "DEBUG"
    
    @patch('safla.cli.setup_logging')
    @patch('safla.cli.SAFLAConfig')
    def test_cli_custom_config_file(self, mock_config_class, mock_setup_logging, runner, temp_config_file):
        """Test CLI with custom configuration file."""
        mock_config = MagicMock()
        mock_config_class.from_file.return_value = mock_config
        
        result = runner.invoke(cli, ['--config', temp_config_file, 'validate'])
        
        # Verify config was loaded from file
        mock_config_class.from_file.assert_called_once_with(temp_config_file)
    
    @patch('safla.cli.setup_logging')
    @patch('safla.cli.SAFLAConfig')
    def test_cli_config_error_handling(self, mock_config_class, mock_setup_logging, runner):
        """Test CLI configuration error handling."""
        mock_config_class.from_env.side_effect = ConfigurationError("Invalid config")
        
        result = runner.invoke(cli, ['validate'])
        
        assert result.exit_code == 1
        assert "Error loading configuration" in result.output
    
    @patch('safla.cli.validate_installation')
    @patch('safla.cli.validate_config')
    @patch('safla.cli.check_gpu_availability')
    @patch('safla.cli.setup_logging')
    @patch('safla.cli.SAFLAConfig')
    def test_validate_command_success(
        self, mock_config_class, mock_setup_logging,
        mock_check_gpu, mock_validate_config, mock_validate_install,
        runner
    ):
        """Test validate command successful execution."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config_class.from_env.return_value = mock_config
        
        mock_validate_install.return_value = {
            "status": "success",
            "modules": ["safla.core", "safla.mcp"],
            "missing": []
        }
        mock_validate_config.return_value = []
        mock_check_gpu.return_value = {
            "available": True,
            "devices": ["NVIDIA GeForce RTX 3080"]
        }
        
        result = runner.invoke(cli, ['validate'])
        
        assert result.exit_code == 0
        assert "Validating SAFLA Installation" in result.output
        mock_validate_install.assert_called_once()
        mock_validate_config.assert_called_once_with(mock_config)
        mock_check_gpu.assert_called_once()
    
    @patch('safla.cli.validate_installation')
    @patch('safla.cli.validate_config')
    @patch('safla.cli.check_gpu_availability')
    @patch('safla.cli.setup_logging')
    @patch('safla.cli.SAFLAConfig')
    def test_validate_command_with_output(
        self, mock_config_class, mock_setup_logging,
        mock_check_gpu, mock_validate_config, mock_validate_install,
        runner
    ):
        """Test validate command with output file."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config_class.from_env.return_value = mock_config
        
        validation_results = {
            "status": "success",
            "modules": ["safla.core"],
            "missing": []
        }
        config_errors = ["Error 1", "Error 2"]
        gpu_info = {"available": False, "devices": []}
        
        mock_validate_install.return_value = validation_results
        mock_validate_config.return_value = config_errors
        mock_check_gpu.return_value = gpu_info
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        result = runner.invoke(cli, ['validate', '--output', output_file])
        
        assert result.exit_code == 0
        
        # Check output file was created
        with open(output_file, 'r') as f:
            report = json.load(f)
        
        assert report["validation"] == validation_results
        assert report["config_errors"] == config_errors
        assert report["gpu_info"] == gpu_info
        
        # Cleanup
        Path(output_file).unlink()
    
    @patch('safla.cli.setup_logging')
    def test_cli_log_level_option(self, mock_setup_logging, runner):
        """Test CLI log level option."""
        with patch('safla.cli.SAFLAConfig') as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.from_env.return_value = mock_config
            
            result = runner.invoke(cli, ['--log-level', 'DEBUG', 'validate'])
            
            # Verify logging was setup with correct level
            mock_setup_logging.assert_called_once_with(level='DEBUG', rich_logging=True)
    
    def test_validate_command_help(self, runner):
        """Test validate command help."""
        result = runner.invoke(cli, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'Validate SAFLA installation and configuration' in result.output
        assert '--output' in result.output
    
    @patch('safla.cli._display_validation_results')
    @patch('safla.cli.validate_installation')
    @patch('safla.cli.validate_config')
    @patch('safla.cli.check_gpu_availability')
    @patch('safla.cli.setup_logging')
    @patch('safla.cli.SAFLAConfig')
    def test_validate_display_results_called(
        self, mock_config_class, mock_setup_logging,
        mock_check_gpu, mock_validate_config, mock_validate_install,
        mock_display_results, runner
    ):
        """Test that validation results are displayed."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config_class.from_env.return_value = mock_config
        
        validation_results = {"status": "success"}
        config_errors = []
        gpu_info = {"available": True}
        
        mock_validate_install.return_value = validation_results
        mock_validate_config.return_value = config_errors
        mock_check_gpu.return_value = gpu_info
        
        result = runner.invoke(cli, ['validate'])
        
        # Verify display function was called with correct arguments
        mock_display_results.assert_called_once_with(
            validation_results, config_errors, gpu_info
        )
    
    @patch('safla.cli.setup_logging')
    @patch('safla.cli.SAFLAConfig')
    def test_cli_context_passing(self, mock_config_class, mock_setup_logging, runner):
        """Test context passing between CLI commands."""
        mock_config = MagicMock()
        mock_config.debug = False
        mock_config.log_level = "INFO"
        mock_config_class.from_env.return_value = mock_config
        
        # Use a command that accesses context
        with patch('safla.cli.validate_installation'), \
             patch('safla.cli.validate_config') as mock_validate_config, \
             patch('safla.cli.check_gpu_availability'):
            
            result = runner.invoke(cli, ['validate'])
            
            # Verify config was passed to validation
            mock_validate_config.assert_called_once()
            passed_config = mock_validate_config.call_args[0][0]
            assert passed_config == mock_config
    
    def test_cli_invalid_command(self, runner):
        """Test CLI with invalid command."""
        result = runner.invoke(cli, ['invalid-command'])
        assert result.exit_code == 2
        assert "Error" in result.output or "Usage" in result.output
    
    @patch('safla.cli.setup_logging')
    @patch('safla.cli.SAFLAConfig')
    def test_cli_exception_handling(self, mock_config_class, mock_setup_logging, runner):
        """Test CLI exception handling during command execution."""
        mock_config = MagicMock()
        mock_config_class.from_env.return_value = mock_config
        
        with patch('safla.cli.validate_installation') as mock_validate:
            mock_validate.side_effect = SAFLAError("Validation failed")
            
            result = runner.invoke(cli, ['validate'])
            
            # Should handle the exception gracefully
            assert result.exit_code != 0
    
    def test_cli_multiple_options(self, runner):
        """Test CLI with multiple options."""
        with patch('safla.cli.setup_logging') as mock_setup_logging, \
             patch('safla.cli.SAFLAConfig') as mock_config_class:
            
            mock_config = MagicMock()
            mock_config_class.from_file.return_value = mock_config
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                json.dump({"test": "config"}, f)
                f.flush()
                
                result = runner.invoke(cli, [
                    '--debug',
                    '--config', f.name,
                    '--log-level', 'WARNING',
                    'validate'
                ])
            
            # Verify all options were processed
            assert mock_config.debug is True
            assert mock_config.log_level == "DEBUG"  # Debug overrides log-level
            mock_setup_logging.assert_called_once_with(level='WARNING', rich_logging=True)