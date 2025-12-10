"""
Comprehensive test suite for SAFLA CLI Management System.

Tests all CLI commands, interactive components, and system management
functionality.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from safla.cli_manager import cli, SaflaCliManager
from safla.cli_implementations import *
from safla.cli_interactive import *


class TestCLIManager:
    """Test suite for the main CLI manager."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("SAFLA_DEBUG=true\nSAFLA_LOG_LEVEL=INFO\n")
            return f.name
    
    @pytest.fixture
    def mock_manager(self):
        """Create mock CLI manager."""
        manager = Mock(spec=SaflaCliManager)
        manager.config = Mock()
        manager.config.get = Mock(return_value='default_value')
        return manager
    
    def test_cli_help(self, runner):
        """Test main CLI help."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'SAFLA CLI Management System' in result.output
        assert 'system' in result.output
        assert 'config' in result.output
        assert 'monitor' in result.output
    
    def test_cli_with_debug_flag(self, runner, temp_config):
        """Test CLI with debug flag."""
        with patch('safla.cli_manager.setup_logging') as mock_logging:
            result = runner.invoke(cli, ['--debug', '--config', temp_config, 'version'])
            mock_logging.assert_called_once()
            args, kwargs = mock_logging.call_args
            assert kwargs['level'] == 'DEBUG'
    
    def test_cli_with_quiet_flag(self, runner, temp_config):
        """Test CLI with quiet flag."""
        with patch('safla.cli_manager.setup_logging') as mock_logging:
            result = runner.invoke(cli, ['--quiet', '--config', temp_config, 'version'])
            mock_logging.assert_called_once()
            args, kwargs = mock_logging.call_args
            assert kwargs['level'] == 'WARNING'


class TestSystemCommands:
    """Test system management commands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def mock_manager(self):
        manager = Mock(spec=SaflaCliManager)
        manager.config = Mock()
        return manager
    
    def test_system_status_command(self, runner):
        """Test system status command."""
        with patch('safla.cli_implementations._get_system_status') as mock_status:
            mock_status.return_value = {
                'health': 'healthy',
                'uptime': '2h 30m',
                'components': {
                    'memory': {'status': 'running', 'health': 'healthy'},
                    'mcp': {'status': 'running', 'health': 'healthy'}
                }
            }
            
            result = runner.invoke(cli, ['system', 'status'])
            assert result.exit_code == 0
            mock_status.assert_called_once()
    
    def test_system_status_json_format(self, runner):
        """Test system status with JSON output."""
        with patch('safla.cli_implementations._get_system_status') as mock_status:
            mock_status.return_value = {'health': 'healthy'}
            
            result = runner.invoke(cli, ['system', 'status', '--format', 'json'])
            assert result.exit_code == 0
            # Should contain JSON output
            assert '{' in result.output and '}' in result.output
    
    def test_system_start_command(self, runner):
        """Test system start command."""
        with patch('safla.cli_implementations._start_full_system') as mock_start:
            mock_start.return_value = True
            
            result = runner.invoke(cli, ['system', 'start'])
            assert result.exit_code == 0
            mock_start.assert_called_once()
    
    def test_system_start_component(self, runner):
        """Test starting specific component."""
        with patch('safla.cli_implementations._start_component') as mock_start:
            mock_start.return_value = True
            
            result = runner.invoke(cli, ['system', 'start', '--component', 'memory'])
            assert result.exit_code == 0
            mock_start.assert_called_once()
    
    def test_system_validate_command(self, runner):
        """Test system validation command."""
        with patch('safla.cli_manager.validate_installation') as mock_validate, \
             patch('safla.cli_manager.validate_config') as mock_config, \
             patch('safla.cli_manager.check_gpu_availability') as mock_gpu, \
             patch('safla.cli_implementations._run_health_checks') as mock_health:
            
            mock_validate.return_value = {'status': 'success'}
            mock_config.return_value = []
            mock_gpu.return_value = {'available': True}
            mock_health.return_value = {'overall_health': 'good'}
            
            result = runner.invoke(cli, ['system', 'validate'])
            assert result.exit_code == 0


class TestConfigCommands:
    """Test configuration management commands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_config_show_command(self, runner):
        """Test config show command."""
        with patch('safla.cli_implementations._display_config') as mock_display:
            result = runner.invoke(cli, ['config', 'show'])
            assert result.exit_code == 0
            mock_display.assert_called_once()
    
    def test_config_show_specific_key(self, runner):
        """Test showing specific configuration key."""
        with patch('safla.cli_implementations._get_config_value') as mock_get:
            mock_get.return_value = 'test_value'
            
            result = runner.invoke(cli, ['config', 'show', '--key', 'debug'])
            assert result.exit_code == 0
            assert 'test_value' in result.output
    
    def test_config_set_command(self, runner):
        """Test config set command."""
        with patch('safla.cli_implementations._set_config_value') as mock_set:
            result = runner.invoke(cli, ['config', 'set', 'test_key', 'test_value'])
            assert result.exit_code == 0
            mock_set.assert_called_once()
    
    def test_config_backup_command(self, runner):
        """Test config backup command."""
        with patch('safla.cli_implementations._backup_config') as mock_backup:
            mock_backup.return_value = '/tmp/backup.json'
            
            result = runner.invoke(cli, ['config', 'backup'])
            assert result.exit_code == 0
            assert 'backup.json' in result.output


class TestMonitorCommands:
    """Test monitoring commands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_monitor_metrics_command(self, runner):
        """Test monitor metrics command."""
        with patch('safla.cli_implementations._get_system_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'system': {'cpu_percent': 50.0, 'memory_percent': 60.0},
                'components': {}
            }
            
            result = runner.invoke(cli, ['monitor', 'metrics'])
            assert result.exit_code == 0
            mock_metrics.assert_called_once()
    
    def test_monitor_logs_command(self, runner):
        """Test monitor logs command."""
        with patch('safla.cli_implementations._view_logs') as mock_logs:
            result = runner.invoke(cli, ['monitor', 'logs'])
            assert result.exit_code == 0
            mock_logs.assert_called_once()


class TestOptimizeCommands:
    """Test optimization commands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_optimize_analyze_command(self, runner):
        """Test optimization analysis."""
        with patch('safla.cli_implementations._analyze_optimizations') as mock_analyze:
            mock_analyze.return_value = [
                {
                    'component': 'memory',
                    'type': 'cache_optimization',
                    'description': 'Increase cache size',
                    'impact': 'high',
                    'effort': 'low'
                }
            ]
            
            result = runner.invoke(cli, ['optimize', 'analyze'])
            assert result.exit_code == 0
            mock_analyze.assert_called_once()
    
    def test_optimize_memory_command(self, runner):
        """Test memory optimization."""
        with patch('safla.cli_implementations._optimize_memory') as mock_optimize:
            mock_optimize.return_value = 'Memory optimized: 15% improvement'
            
            result = runner.invoke(cli, ['optimize', 'memory'])
            assert result.exit_code == 0
            assert 'Memory optimization complete' in result.output


class TestBenchmarkCommands:
    """Test benchmark commands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_benchmark_run_command(self, runner):
        """Test benchmark run command."""
        with patch('safla.cli_implementations._run_benchmark_suite') as mock_benchmark:
            mock_benchmark.return_value = {
                'suite': 'standard',
                'benchmarks': {'memory_search': {'status': 'passed'}},
                'summary': {'total_benchmarks': 1, 'passed_benchmarks': 1}
            }
            
            result = runner.invoke(cli, ['benchmark', 'run'])
            assert result.exit_code == 0
            mock_benchmark.assert_called_once()
    
    def test_benchmark_stress_command(self, runner):
        """Test stress test command."""
        with patch('safla.cli_implementations._run_stress_test') as mock_stress:
            mock_stress.return_value = {
                'duration': 300,
                'load_level': 0.8,
                'status': 'passed'
            }
            
            result = runner.invoke(cli, ['benchmark', 'stress'])
            assert result.exit_code == 0


class TestAgentCommands:
    """Test agent management commands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_agents_list_command(self, runner):
        """Test agents list command."""
        with patch('safla.cli_implementations._get_deployed_agents') as mock_agents:
            mock_agents.return_value = [
                {
                    'name': 'test-agent',
                    'type': 'optimization',
                    'status': 'running',
                    'replicas': 2,
                    'cpu_usage': 15.0,
                    'memory_usage': 128.0,
                    'uptime': '1h 30m'
                }
            ]
            
            result = runner.invoke(cli, ['agents', 'list'])
            assert result.exit_code == 0
            assert 'test-agent' in result.output
    
    def test_agents_deploy_command(self, runner):
        """Test agent deployment."""
        with patch('safla.cli_implementations._deploy_agent') as mock_deploy:
            result = runner.invoke(cli, ['agents', 'deploy', 'test-agent'])
            assert result.exit_code == 0
            mock_deploy.assert_called_once()


class TestInteractiveComponents:
    """Test interactive CLI components."""
    
    def test_setup_wizard_basic_flow(self):
        """Test setup wizard basic flow."""
        mock_manager = Mock()
        
        with patch('safla.cli_interactive.Confirm.ask', return_value=True), \
             patch('safla.cli_interactive.Prompt.ask', return_value='INFO'), \
             patch('safla.cli_interactive.IntPrompt.ask', return_value=4), \
             patch('safla.cli_interactive.FloatPrompt.ask', return_value=0.8), \
             patch('safla.cli_interactive._save_wizard_config'), \
             patch('safla.cli_interactive._start_full_system'):
            
            # Should not raise any exceptions
            _run_setup_wizard(mock_manager)
    
    def test_config_summary_display(self):
        """Test configuration summary display."""
        config_updates = {
            'SAFLA_DEBUG': 'true',
            'SAFLA_LOG_LEVEL': 'INFO',
            'SAFLA_WORKER_THREADS': '4'
        }
        
        # Should not raise any exceptions
        _display_config_summary(config_updates)
    
    def test_save_wizard_config(self):
        """Test saving wizard configuration."""
        config_updates = {
            'SAFLA_DEBUG': 'true',
            'SAFLA_LOG_LEVEL': 'INFO'
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / '.env'
            
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                _save_wizard_config(config_updates)
                
                # Verify file was created and contains expected content
                assert env_file.exists()
                content = env_file.read_text()
                assert 'SAFLA_DEBUG=true' in content
                assert 'SAFLA_LOG_LEVEL=INFO' in content
                
            finally:
                os.chdir(original_cwd)
    
    def test_simple_dashboard_components(self):
        """Test simple dashboard component creation."""
        mock_manager = Mock()
        
        # Test status display
        status_data = {
            'health': 'healthy',
            'uptime': '2h 30m',
            'components': {
                'memory': {'status': 'running', 'health': 'healthy'}
            }
        }
        
        result = _create_status_display(status_data)
        assert 'Overall Health: Healthy' in result
        assert 'memory' in result.lower()
        
        # Test metrics display
        metrics_data = {
            'system': {
                'cpu_percent': 45.0,
                'memory_percent': 65.0,
                'disk_percent': 30.0
            },
            'components': {
                'memory': {'cpu_usage_percent': 10.5}
            }
        }
        
        result = _create_metrics_display(metrics_data)
        assert 'CPU: 45.0%' in result
        assert 'Memory: 65.0%' in result
        
        # Test agents display
        agents_data = [
            {
                'name': 'test-agent',
                'type': 'optimization',
                'status': 'running',
                'replicas': 2
            }
        ]
        
        result = _create_agents_display(agents_data)
        assert 'test-agent' in result
        assert 'optimization' in result


class TestCLIHelpers:
    """Test CLI helper functions."""
    
    def test_search_cli_help(self):
        """Test CLI help search functionality."""
        with patch('safla.cli_implementations.console') as mock_console:
            _search_cli_help('status')
            
            # Should have made calls to print search results
            assert mock_console.print.called
    
    def test_health_checks(self):
        """Test health check functions."""
        mock_manager = Mock()
        mock_manager.config.get.return_value = 'localhost'
        
        # Test system resource check
        with patch('safla.cli_implementations._get_resource_usage') as mock_resources:
            mock_resources.return_value = {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_percent': 30.0
            }
            
            result = _check_system_resources(mock_manager)
            assert result['status'] == 'pass'
            assert 'System resources OK' in result['message']
    
    def test_performance_baseline_check(self):
        """Test performance baseline check."""
        mock_manager = Mock()
        
        result = _check_performance_baseline(mock_manager)
        assert 'status' in result
        assert 'message' in result
        assert 'details' in result
    
    def test_version_info(self):
        """Test version information retrieval."""
        version_info = _get_version_info()
        
        assert 'safla_version' in version_info
        assert 'python_version' in version_info
        assert 'components' in version_info
        assert version_info['safla_version'] == '2.0.0'
    
    def test_config_operations(self):
        """Test configuration operations."""
        mock_config = Mock()
        mock_config.debug = False
        
        # Test getting config value
        with patch('safla.cli_implementations.hasattr', return_value=True):
            with patch('safla.cli_implementations.getattr', return_value='test_value'):
                result = _get_config_value(mock_config, 'debug')
                assert result == 'test_value'
        
        # Test setting config value
        with patch('safla.cli_implementations.setattr') as mock_setattr:
            _set_config_value(mock_config, 'debug', 'true')
            mock_setattr.assert_called()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_full_command_chain(self, runner):
        """Test a full chain of CLI commands."""
        with patch('safla.cli_implementations._get_system_status') as mock_status, \
             patch('safla.cli_implementations._get_system_metrics') as mock_metrics, \
             patch('safla.cli_implementations._analyze_optimizations') as mock_optimize:
            
            mock_status.return_value = {'health': 'healthy', 'components': {}}
            mock_metrics.return_value = {'system': {}, 'components': {}}
            mock_optimize.return_value = []
            
            # Test status -> metrics -> optimize chain
            result1 = runner.invoke(cli, ['system', 'status'])
            assert result1.exit_code == 0
            
            result2 = runner.invoke(cli, ['monitor', 'metrics'])
            assert result2.exit_code == 0
            
            result3 = runner.invoke(cli, ['optimize', 'analyze'])
            assert result3.exit_code == 0
    
    def test_error_handling(self, runner):
        """Test CLI error handling."""
        with patch('safla.cli_implementations._get_system_status', side_effect=Exception('Test error')):
            result = runner.invoke(cli, ['system', 'status'])
            # Should handle the error gracefully
            assert result.exit_code != 0 or 'error' in result.output.lower()
    
    def test_output_formats(self, runner):
        """Test different output formats."""
        with patch('safla.cli_implementations._get_system_status') as mock_status:
            mock_status.return_value = {'health': 'healthy'}
            
            # Test table format (default)
            result1 = runner.invoke(cli, ['system', 'status'])
            assert result1.exit_code == 0
            
            # Test JSON format
            result2 = runner.invoke(cli, ['system', 'status', '--format', 'json'])
            assert result2.exit_code == 0
            
            # Test YAML format
            result3 = runner.invoke(cli, ['system', 'status', '--format', 'yaml'])
            assert result3.exit_code == 0


class TestCLIPerformance:
    """Performance tests for CLI operations."""
    
    def test_status_command_performance(self):
        """Test that status command executes quickly."""
        import time
        
        mock_manager = Mock()
        
        with patch('safla.cli_implementations._check_component_status', return_value={'health': 'healthy'}):
            start_time = time.time()
            _get_system_status(mock_manager, detailed=False)
            execution_time = time.time() - start_time
            
            # Should complete in under 1 second
            assert execution_time < 1.0
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance."""
        import time
        
        mock_manager = Mock()
        
        start_time = time.time()
        _get_system_metrics(mock_manager, detailed=True)
        execution_time = time.time() - start_time
        
        # Should complete quickly
        assert execution_time < 0.5


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])