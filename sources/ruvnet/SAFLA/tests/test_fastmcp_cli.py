"""
Unit tests for FastMCP CLI Integration

This module contains comprehensive tests for the FastMCP CLI commands,
including client, server, discovery, testing, and endpoint management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from click.testing import CliRunner
from typing import Dict, Any, List

from safla.cli import cli
from safla.integrations.fastmcp_adapter import FastMCPConfig, FastMCPEndpoint
from safla.integrations.fastmcp_client import FastMCPClient
from safla.integrations.fastmcp_server import FastMCPServer
from safla.exceptions import SAFLAError


class TestFastMCPCLI:
    """Test FastMCP CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_config(self):
        """Create mock FastMCP configuration."""
        return FastMCPConfig(
            endpoints={
                "test_endpoint": FastMCPEndpoint(
                    name="test_endpoint",
                    url="http://localhost:8001",
                    transport="sse"
                )
            }
        )
    
    def test_fastmcp_group_help(self, runner):
        """Test FastMCP command group help."""
        result = runner.invoke(cli, ['fastmcp', '--help'])
        
        assert result.exit_code == 0
        assert 'FastMCP integration commands' in result.output
        assert 'client' in result.output
        assert 'server' in result.output
        assert 'discover' in result.output
        assert 'test' in result.output
        assert 'list-endpoints' in result.output


class TestFastMCPClientCLI:
    """Test FastMCP client CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_client_call_tool_success(self, mock_client_class, mock_adapter_class, runner):
        """Test successful tool call via CLI."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.call_tool = AsyncMock(return_value={"result": "success", "value": 42})
        mock_client_class.return_value = mock_client
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'client', 'call-tool',
                '--endpoint', 'test_endpoint',
                '--tool-name', 'test_tool',
                '--arguments', '{"arg1": "value1"}'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_client_call_tool_invalid_json(self, mock_client_class, mock_adapter_class, runner):
        """Test tool call with invalid JSON arguments."""
        result = runner.invoke(cli, [
            'fastmcp', 'client', 'call-tool',
            '--endpoint', 'test_endpoint',
            '--tool-name', 'test_tool',
            '--arguments', 'invalid_json'
        ])
        
        assert result.exit_code != 0
        assert 'Invalid JSON' in result.output
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_client_read_resource_success(self, mock_client_class, mock_adapter_class, runner):
        """Test successful resource read via CLI."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.read_resource = AsyncMock(return_value={"content": "resource data"})
        mock_client_class.return_value = mock_client
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'client', 'read-resource',
                '--endpoint', 'test_endpoint',
                '--resource-uri', 'resource://test'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_client_list_tools_success(self, mock_client_class, mock_adapter_class, runner):
        """Test successful tool listing via CLI."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.list_tools = AsyncMock(return_value=[
            {"name": "tool1", "description": "Test tool 1"},
            {"name": "tool2", "description": "Test tool 2"}
        ])
        mock_client_class.return_value = mock_client
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'client', 'list-tools',
                '--endpoint', 'test_endpoint'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_client_list_resources_success(self, mock_client_class, mock_adapter_class, runner):
        """Test successful resource listing via CLI."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.list_resources = AsyncMock(return_value=[
            {"uri": "resource://test1", "name": "Test Resource 1"},
            {"uri": "resource://test2", "name": "Test Resource 2"}
        ])
        mock_client_class.return_value = mock_client
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'client', 'list-resources',
                '--endpoint', 'test_endpoint'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_client_batch_operations_success(self, mock_client_class, mock_adapter_class, runner):
        """Test successful batch operations via CLI."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.execute = AsyncMock(return_value=[
            {"success": True, "result": "tool_result"},
            {"success": True, "result": "resource_result"}
        ])
        mock_client.batch_context = Mock()
        mock_client.batch_context.return_value.__aenter__ = AsyncMock(return_value=mock_batch)
        mock_client.batch_context.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client
        
        # Create temporary batch file
        batch_operations = [
            {"type": "call_tool", "tool_name": "tool1", "arguments": {}},
            {"type": "read_resource", "resource_uri": "resource://test"}
        ]
        
        with runner.isolated_filesystem():
            with open('batch.json', 'w') as f:
                import json
                json.dump(batch_operations, f)
            
            with patch('safla.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                
                result = runner.invoke(cli, [
                    'fastmcp', 'client', 'batch',
                    '--endpoint', 'test_endpoint',
                    '--batch-file', 'batch.json'
                ])
                
                assert result.exit_code == 0
                mock_run.assert_called_once()
    
    def test_client_batch_operations_invalid_file(self, runner):
        """Test batch operations with invalid file."""
        result = runner.invoke(cli, [
            'fastmcp', 'client', 'batch',
            '--endpoint', 'test_endpoint',
            '--batch-file', 'nonexistent.json'
        ])
        
        assert result.exit_code != 0
        assert 'Error reading batch file' in result.output


class TestFastMCPServerCLI:
    """Test FastMCP server CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.fastmcp_server.FastMCPServer')
    def test_server_start_success(self, mock_server_class, runner):
        """Test successful server start via CLI."""
        # Setup mock
        mock_server = Mock()
        mock_server.start = AsyncMock()
        mock_server_class.return_value = mock_server
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'server', 'start',
                '--name', 'test_server',
                '--host', 'localhost',
                '--port', '8080'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_server.FastMCPServer')
    def test_server_start_with_tools_file(self, mock_server_class, runner):
        """Test server start with tools file."""
        # Setup mock
        mock_server = Mock()
        mock_server.register_tool = Mock()
        mock_server.start = AsyncMock()
        mock_server_class.return_value = mock_server
        
        # Create temporary tools file
        tools_config = {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "function": "test_module.test_function"
                }
            ]
        }
        
        with runner.isolated_filesystem():
            with open('tools.json', 'w') as f:
                import json
                json.dump(tools_config, f)
            
            with patch('safla.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                
                result = runner.invoke(cli, [
                    'fastmcp', 'server', 'start',
                    '--name', 'test_server',
                    '--tools-file', 'tools.json'
                ])
                
                assert result.exit_code == 0
                mock_run.assert_called_once()
    
    def test_server_start_invalid_tools_file(self, runner):
        """Test server start with invalid tools file."""
        result = runner.invoke(cli, [
            'fastmcp', 'server', 'start',
            '--name', 'test_server',
            '--tools-file', 'nonexistent.json'
        ])
        
        assert result.exit_code != 0
        assert 'Error reading tools file' in result.output


class TestFastMCPDiscoveryCLI:
    """Test FastMCP discovery CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.fastmcp_helpers.discover_fastmcp_endpoints')
    def test_discover_endpoints_success(self, mock_discover, runner):
        """Test successful endpoint discovery via CLI."""
        mock_discover.return_value = [
            {
                "name": "endpoint1",
                "url": "http://localhost:8001",
                "transport": "sse",
                "status": "healthy"
            },
            {
                "name": "endpoint2",
                "url": "http://localhost:8002", 
                "transport": "streamable_http",
                "status": "healthy"
            }
        ]
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'discover',
                '--discovery-url', 'http://discovery.example.com'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_helpers.discover_fastmcp_endpoints')
    def test_discover_endpoints_with_filter(self, mock_discover, runner):
        """Test endpoint discovery with status filter via CLI."""
        mock_discover.return_value = [
            {
                "name": "healthy_endpoint",
                "url": "http://localhost:8001",
                "status": "healthy"
            }
        ]
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'discover',
                '--discovery-url', 'http://discovery.example.com',
                '--filter-status', 'healthy'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_helpers.discover_fastmcp_endpoints')
    def test_discover_endpoints_save_config(self, mock_discover, runner):
        """Test endpoint discovery with config save via CLI."""
        mock_discover.return_value = [
            {
                "name": "endpoint1",
                "url": "http://localhost:8001",
                "transport": "sse"
            }
        ]
        
        with runner.isolated_filesystem():
            with patch('safla.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                
                result = runner.invoke(cli, [
                    'fastmcp', 'discover',
                    '--discovery-url', 'http://discovery.example.com',
                    '--save-config', 'discovered_config.json'
                ])
                
                assert result.exit_code == 0
                mock_run.assert_called_once()


class TestFastMCPTestCLI:
    """Test FastMCP testing CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.fastmcp_helpers.check_fastmcp_endpoint')
    def test_test_endpoint_success(self, mock_test, runner):
        """Test successful endpoint testing via CLI."""
        mock_test.return_value = {
            "endpoint_name": "test_endpoint",
            "status": "healthy",
            "tool_count": 2,
            "resource_count": 1,
            "basic_tests": {
                "passed": True,
                "results": []
            }
        }
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'test',
                '--endpoint-name', 'test_endpoint',
                '--endpoint-url', 'http://localhost:8001',
                '--transport', 'sse'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_helpers.check_fastmcp_endpoint')
    def test_test_endpoint_with_basic_tests(self, mock_test, runner):
        """Test endpoint testing with basic tests via CLI."""
        mock_test.return_value = {
            "endpoint_name": "test_endpoint",
            "status": "healthy",
            "basic_tests": {
                "passed": True,
                "results": [
                    {"test": "list_tools", "passed": True},
                    {"test": "list_resources", "passed": True}
                ]
            }
        }
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'test',
                '--endpoint-name', 'test_endpoint',
                '--endpoint-url', 'http://localhost:8001',
                '--transport', 'sse',
                '--run-basic-tests'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_helpers.check_fastmcp_endpoint')
    def test_test_endpoint_save_report(self, mock_test, runner):
        """Test endpoint testing with report save via CLI."""
        mock_test.return_value = {
            "endpoint_name": "test_endpoint",
            "status": "healthy",
            "timestamp": "2023-01-01T00:00:00Z"
        }
        
        with runner.isolated_filesystem():
            with patch('safla.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                
                result = runner.invoke(cli, [
                    'fastmcp', 'test',
                    '--endpoint-name', 'test_endpoint',
                    '--endpoint-url', 'http://localhost:8001',
                    '--transport', 'sse',
                    '--save-report', 'test_report.json'
                ])
                
                assert result.exit_code == 0
                mock_run.assert_called_once()


class TestFastMCPListEndpointsCLI:
    """Test FastMCP list endpoints CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_list_endpoints_success(self, mock_client_class, mock_adapter_class, runner):
        """Test successful endpoint listing via CLI."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.list_endpoints = AsyncMock(return_value=[
            {
                "endpoint_name": "endpoint1",
                "status": "healthy",
                "is_default": True,
                "tool_count": 2,
                "resource_count": 1
            },
            {
                "endpoint_name": "endpoint2",
                "status": "unhealthy",
                "is_default": False,
                "tool_count": 0,
                "resource_count": 0,
                "error": "Connection failed"
            }
        ])
        mock_client_class.return_value = mock_client
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'list-endpoints'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_list_endpoints_with_details(self, mock_client_class, mock_adapter_class, runner):
        """Test endpoint listing with details via CLI."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.list_endpoints = AsyncMock(return_value=[
            {
                "endpoint_name": "endpoint1",
                "status": "healthy",
                "is_default": True,
                "tool_count": 2,
                "resource_count": 1,
                "tools": [
                    {"name": "tool1", "description": "Test tool 1"},
                    {"name": "tool2", "description": "Test tool 2"}
                ],
                "resources": [
                    {"uri": "resource://test1", "name": "Test Resource 1"}
                ]
            }
        ])
        mock_client_class.return_value = mock_client
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'list-endpoints',
                '--detailed'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_list_endpoints_save_output(self, mock_client_class, mock_adapter_class, runner):
        """Test endpoint listing with output save via CLI."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.list_endpoints = AsyncMock(return_value=[
            {
                "endpoint_name": "endpoint1",
                "status": "healthy"
            }
        ])
        mock_client_class.return_value = mock_client
        
        with runner.isolated_filesystem():
            with patch('safla.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                
                result = runner.invoke(cli, [
                    'fastmcp', 'list-endpoints',
                    '--output', 'endpoints.json'
                ])
                
                assert result.exit_code == 0
                mock_run.assert_called_once()


class TestFastMCPStartIntegration:
    """Test FastMCP integration with main start command."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.cli.SAFLAOrchestrator')
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    def test_start_with_fastmcp_component(self, mock_adapter_class, mock_orchestrator_class, runner):
        """Test start command with FastMCP component."""
        # Setup mocks
        mock_orchestrator = Mock()
        mock_orchestrator.start = AsyncMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'start',
                '--component', 'fastmcp'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.cli.SAFLAOrchestrator')
    def test_start_with_fastmcp_config_file(self, mock_orchestrator_class, runner):
        """Test start command with FastMCP config file."""
        # Setup mock
        mock_orchestrator = Mock()
        mock_orchestrator.start = AsyncMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Create temporary config file
        fastmcp_config = {
            "endpoints": {
                "test_endpoint": {
                    "name": "test_endpoint",
                    "url": "http://localhost:8001",
                    "transport": "sse"
                }
            }
        }
        
        with runner.isolated_filesystem():
            with open('fastmcp_config.json', 'w') as f:
                import json
                json.dump(fastmcp_config, f)
            
            with patch('safla.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                
                result = runner.invoke(cli, [
                    'start',
                    '--fastmcp-config', 'fastmcp_config.json'
                ])
                
                assert result.exit_code == 0
                mock_run.assert_called_once()


class TestFastMCPCLIErrorHandling:
    """Test FastMCP CLI error handling."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_fastmcp_not_available_error(self, runner):
        """Test CLI behavior when FastMCP is not available."""
        with patch('****', False):
            result = runner.invoke(cli, [
                'fastmcp', 'client', 'list-tools',
                '--endpoint', 'test_endpoint'
            ])
            
            assert result.exit_code != 0
            assert 'FastMCP is not available' in result.output
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    def test_configuration_error(self, mock_adapter_class, runner):
        """Test CLI behavior with configuration errors."""
        mock_adapter_class.side_effect = SAFLAError("Configuration error")
        
        result = runner.invoke(cli, [
            'fastmcp', 'client', 'list-tools',
            '--endpoint', 'test_endpoint'
        ])
        
        assert result.exit_code != 0
        assert 'Configuration error' in result.output
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter')
    @patch('safla.integrations.fastmcp_client.FastMCPClient')
    def test_runtime_error_handling(self, mock_client_class, mock_adapter_class, runner):
        """Test CLI behavior with runtime errors."""
        # Setup mocks
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.call_tool = AsyncMock(side_effect=Exception("Runtime error"))
        mock_client_class.return_value = mock_client
        
        with patch('safla.cli.asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Runtime error")
            
            result = runner.invoke(cli, [
                'fastmcp', 'client', 'call-tool',
                '--endpoint', 'test_endpoint',
                '--tool-name', 'test_tool'
            ])
            
            assert result.exit_code != 0
            assert 'Runtime error' in result.output