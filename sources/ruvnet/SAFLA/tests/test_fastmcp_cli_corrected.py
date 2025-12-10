"""
Unit tests for FastMCP CLI Integration - Corrected Version

This module contains comprehensive tests for the FastMCP CLI commands,
matching the actual CLI implementation structure.
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
                    transport_type="sse"
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
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter', create=True)
    @patch('safla.integrations.fastmcp_client.FastMCPClient', create=True)
    @patch('safla.integrations.create_fastmcp_client_from_config', create=True)
    def test_client_start_with_config_file(self, mock_create_client, mock_client_class, mock_adapter_class, runner):
        """Test starting FastMCP client with config file."""
        # Setup mocks
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        with runner.isolated_filesystem():
            # Create a test config file
            with open('test_config.json', 'w') as f:
                f.write('{"endpoints": {}}')
            
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = mock_client
                
                result = runner.invoke(cli, [
                    'fastmcp', 'client',
                    '--config-file', 'test_config.json',
                    '--endpoint', 'test_endpoint'
                ])
                
                assert result.exit_code == 0
                assert 'FastMCP client started successfully' in result.output
                mock_run.assert_called_once()
    
    @patch('safla.integrations.FastMCPAdapter', create=True)
    @patch('safla.integrations.FastMCPClient', create=True)
    @patch('safla.integrations.FastMCPConfig', create=True)
    def test_client_start_without_config(self, mock_config_class, mock_client_class, mock_adapter_class, runner):
        """Test starting FastMCP client without config file."""
        # Setup mocks
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_adapter = Mock()
        mock_adapter.start = AsyncMock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, ['fastmcp', 'client'])
            
            assert result.exit_code == 0
            assert 'FastMCP client started successfully' in result.output
            mock_run.assert_called_once()
    
    @patch('safla.integrations.create_fastmcp_client_from_config', create=True)
    def test_client_start_config_error(self, mock_create_client, runner):
        """Test client start with configuration error."""
        mock_create_client.side_effect = Exception("Configuration error")
        
        with runner.isolated_filesystem():
            with open('test_config.json', 'w') as f:
                f.write('{"endpoints": {}}')
            
            with patch('asyncio.run') as mock_run:
                mock_run.side_effect = Exception("Configuration error")
                
                result = runner.invoke(cli, [
                    'fastmcp', 'client',
                    '--config-file', 'test_config.json'
                ])
                
                assert result.exit_code == 1
                assert 'Error starting FastMCP client' in result.output


class TestFastMCPServerCLI:
    """Test FastMCP server CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.FastMCPServer', create=True)
    def test_server_start_success(self, mock_server_class, runner):
        """Test successful server start."""
        # Setup mocks
        mock_server = Mock()
        mock_server.tool = Mock()
        mock_server.start = AsyncMock()
        mock_server_class.return_value = mock_server
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'server',
                '--name', 'test_server',
                '--port', '8001',
                '--host', 'localhost'
            ])
            
            assert result.exit_code == 0
            mock_server_class.assert_called_once()
            mock_run.assert_called_once()
    
    @patch('safla.integrations.FastMCPServer', create=True)
    def test_server_start_with_defaults(self, mock_server_class, runner):
        """Test server start with default port and host."""
        # Setup mocks
        mock_server = Mock()
        mock_server.tool = Mock()
        mock_server.start = AsyncMock()
        mock_server_class.return_value = mock_server
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'fastmcp', 'server',
                '--name', 'test_server'
            ])
            
            assert result.exit_code == 0
            mock_server_class.assert_called_once()
    
    @patch('safla.integrations.fastmcp_server.FastMCPServer', create=True)
    def test_server_start_error(self, mock_server_class, runner):
        """Test server start with error."""
        mock_server_class.side_effect = Exception("Server error")
        
        result = runner.invoke(cli, [
            'fastmcp', 'server',
            '--name', 'test_server'
        ])
        
        assert result.exit_code == 1
        assert 'Error starting FastMCP server' in result.output
    
    def test_server_missing_name(self, runner):
        """Test server command without required name."""
        result = runner.invoke(cli, ['fastmcp', 'server'])
        
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'Error' in result.output


class TestFastMCPDiscoveryCLI:
    """Test FastMCP discovery CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.discover_fastmcp_endpoints', create=True)
    def test_discover_endpoints_success(self, mock_discover, runner):
        """Test successful endpoint discovery."""
        mock_discover.return_value = [
            {
                "name": "endpoint1",
                "base_url": "http://localhost:8001",
                "version": "1.0.0",
                "capabilities": ["tools", "resources"]
            }
        ]
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = mock_discover.return_value
            
            result = runner.invoke(cli, [
                'fastmcp', 'discover',
                '--urls', 'http://localhost:8001',
                '--timeout', '5.0'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.discover_fastmcp_endpoints', create=True)
    def test_discover_endpoints_with_output(self, mock_discover, runner):
        """Test endpoint discovery with output file."""
        mock_discover.return_value = [
            {
                "name": "endpoint1",
                "base_url": "http://localhost:8001",
                "version": "1.0.0",
                "capabilities": []
            }
        ]
        
        with runner.isolated_filesystem():
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = mock_discover.return_value
                
                result = runner.invoke(cli, [
                    'fastmcp', 'discover',
                    '--urls', 'http://localhost:8001',
                    '--output', 'endpoints.json'
                ])
                
                assert result.exit_code == 0
                # Check if output file was created
                import os
                assert os.path.exists('endpoints.json')
    
    def test_discover_no_urls(self, runner):
        """Test discovery command without URLs."""
        result = runner.invoke(cli, ['fastmcp', 'discover'])
        
        assert result.exit_code == 1
        assert 'Please provide at least one URL' in result.output
    
    @patch('safla.integrations.discover_fastmcp_endpoints', create=True)
    def test_discover_endpoints_error(self, mock_discover, runner):
        """Test discovery with error."""
        mock_discover.side_effect = Exception("Discovery error")
        
        with patch('asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Discovery error")
            
            result = runner.invoke(cli, [
                'fastmcp', 'discover',
                '--urls', 'http://localhost:8001'
            ])
            
            assert result.exit_code == 1
            assert 'Error during discovery: Discovery error' in result.output


class TestFastMCPTestCLI:
    """Test FastMCP test CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.check_fastmcp_endpoint', create=True)
    def test_test_endpoint_success(self, mock_test, runner):
        """Test successful endpoint testing."""
        mock_test.return_value = {
            "endpoint_url": "http://localhost:8001",
            "connectivity": True,
            "capabilities": {
                "tools": 2,
                "resources": 1,
                "tool_names": ["ping", "status"],
                "resource_uris": ["file://test"]
            },
            "performance": {
                "connection_time": 0.1,
                "response_time": 0.1
            },
            "errors": []
        }
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = mock_test.return_value
            
            result = runner.invoke(cli, [
                'fastmcp', 'test',
                '--endpoint-url', 'http://localhost:8001',
                '--timeout', '5.0'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.check_fastmcp_endpoint', create=True)
    def test_test_endpoint_with_auth(self, mock_test, runner):
        """Test endpoint testing with authentication."""
        mock_test.return_value = {
            "endpoint_url": "http://localhost:8001",
            "connectivity": True,
            "capabilities": {},
            "performance": {},
            "errors": []
        }
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = mock_test.return_value
            
            result = runner.invoke(cli, [
                'fastmcp', 'test',
                '--endpoint-url', 'http://localhost:8001',
                '--auth-token', 'test_token'
            ])
            
            assert result.exit_code == 0
    
    def test_test_endpoint_missing_url(self, runner):
        """Test test command without required URL."""
        result = runner.invoke(cli, ['fastmcp', 'test'])
        
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'Error' in result.output
    
    @patch('safla.integrations.check_fastmcp_endpoint', create=True)
    def test_test_endpoint_error(self, mock_test, runner):
        """Test endpoint testing with error."""
        mock_test.side_effect = Exception("Test error")
        
        with patch('asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Test error")
            
            result = runner.invoke(cli, [
                'fastmcp', 'test',
                '--endpoint-url', 'http://localhost:8001'
            ])
            
            assert result.exit_code == 1
            assert 'Error testing endpoint: Test error' in result.output


class TestFastMCPListEndpointsCLI:
    """Test FastMCP list-endpoints CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.FastMCPAdapter', create=True)
    @patch('safla.integrations.FastMCPClient', create=True)
    @patch('safla.integrations.create_fastmcp_client_from_config', create=True)
    def test_list_endpoints_with_config(self, mock_create_client, mock_client_class, mock_adapter_class, runner):
        """Test listing endpoints with config file."""
        # Setup mocks
        mock_client = Mock()
        mock_client.list_endpoints = AsyncMock(return_value=[
            {
                "name": "endpoint1",
                "url": "http://localhost:8001",
                "status": "healthy"
            }
        ])
        mock_create_client.return_value = mock_client
        
        with runner.isolated_filesystem():
            with open('test_config.json', 'w') as f:
                f.write('{"endpoints": {}}')

            with patch('asyncio.run') as mock_run:
                # First call returns the client, second call returns the endpoints list
                mock_run.side_effect = [
                    mock_client,  # create_fastmcp_client_from_config result
                    [{"name": "endpoint1", "status": "healthy", "tool_count": 0, "resource_count": 0}]  # client.list_endpoints result
                ]

                result = runner.invoke(cli, [
                    'fastmcp', 'list-endpoints',
                    '--config-file', 'test_config.json'
                ])

                assert result.exit_code == 0
                assert mock_run.call_count == 2  # Called twice: create client and list endpoints
    
    @patch('safla.integrations.FastMCPAdapter', create=True)
    @patch('safla.integrations.FastMCPClient', create=True)
    @patch('safla.integrations.FastMCPConfig', create=True)
    def test_list_endpoints_without_config(self, mock_config_class, mock_client_class, mock_adapter_class, runner):
        """Test listing endpoints without config file."""
        # Setup mocks
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_adapter = Mock()
        mock_adapter.start = AsyncMock()
        mock_adapter_class.return_value = mock_adapter
        
        mock_client = Mock()
        mock_client.list_endpoints = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client
        
        with patch('asyncio.run') as mock_run:
            # Mock both asyncio.run calls (adapter.start and client.list_endpoints)
            mock_run.side_effect = [None, []]  # First call returns None, second returns empty list
            
            result = runner.invoke(cli, ['fastmcp', 'list-endpoints'])
            
            assert result.exit_code == 0
            assert mock_run.call_count == 2  # Called twice: adapter.start() and client.list_endpoints()
    
    @patch('safla.integrations.create_fastmcp_client_from_config', create=True)
    def test_list_endpoints_error(self, mock_create_client, runner):
        """Test list endpoints with error."""
        mock_create_client.side_effect = Exception("List error")
        
        with runner.isolated_filesystem():
            with open('test_config.json', 'w') as f:
                f.write('{"endpoints": {}}')
            
            with patch('asyncio.run') as mock_run:
                mock_run.side_effect = Exception("List error")
                
                result = runner.invoke(cli, [
                    'fastmcp', 'list-endpoints',
                    '--config-file', 'test_config.json'
                ])
                
                assert result.exit_code == 1
                assert 'Error listing endpoints: List error' in result.output


class TestFastMCPStartIntegration:
    """Test FastMCP start integration commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('safla.integrations.FastMCPAdapter', create=True)
    @patch('safla.integrations.FastMCPConfig', create=True)
    def test_start_with_fastmcp_component(self, mock_config_class, mock_adapter_class, runner):
        """Test starting SAFLA with FastMCP component."""
        # Setup mocks
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_adapter = Mock()
        mock_adapter.start = AsyncMock()
        mock_adapter.shutdown = AsyncMock()
        mock_adapter_class.return_value = mock_adapter
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = runner.invoke(cli, [
                'start',
                '--component', 'fastmcp'
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter', create=True)
    def test_start_fastmcp_error(self, mock_adapter_class, runner):
        """Test FastMCP component start with error."""
        mock_adapter_class.side_effect = Exception("FastMCP error")
        
        result = runner.invoke(cli, [
            'start',
            '--component', 'fastmcp'
        ])
        
        assert result.exit_code == 1
        assert 'Error starting FastMCP component' in result.output


class TestFastMCPCLIErrorHandling:
    """Test FastMCP CLI error handling."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_fastmcp_help_available(self, runner):
        """Test that FastMCP help is available even when FastMCP is not installed."""
        result = runner.invoke(cli, ['fastmcp', '--help'])
        
        assert result.exit_code == 0
        assert 'FastMCP integration commands' in result.output
    
    @patch('safla.integrations.fastmcp_adapter.FastMCPAdapter', create=True)
    def test_configuration_error_handling(self, mock_adapter_class, runner):
        """Test CLI behavior with configuration errors."""
        mock_adapter_class.side_effect = SAFLAError("Configuration error")
        
        result = runner.invoke(cli, [
            'fastmcp', 'client'
        ])
        
        assert result.exit_code == 1
        assert 'Error starting FastMCP client' in result.output
    
    @patch('safla.integrations.fastmcp_server.FastMCPServer', create=True)
    def test_server_runtime_error_handling(self, mock_server_class, runner):
        """Test CLI behavior with server runtime errors."""
        mock_server = Mock()
        mock_server.tool = Mock()
        mock_server.start = AsyncMock(side_effect=Exception("Runtime error"))
        mock_server_class.return_value = mock_server
        
        with patch('asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Runtime error")
            
            result = runner.invoke(cli, [
                'fastmcp', 'server',
                '--name', 'test_server'
            ])
            
            assert result.exit_code == 1
            assert 'Error starting FastMCP server' in result.output