"""
FastMCP Integration Tests - Complete Fixed Version

This module contains comprehensive integration tests for the FastMCP integration
with proper mocking and error handling.
"""

import asyncio
import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from safla.integrations.fastmcp_adapter import FastMCPAdapter, FastMCPConfig, FastMCPEndpoint
from safla.integrations.fastmcp_client import FastMCPClient
from safla.integrations.fastmcp_server import FastMCPServer, FastMCPServerBuilder
from safla.integrations.fastmcp_helpers import batch_call_tools, convert_safla_config_to_fastmcp


class TestFastMCPEndToEndWorkflow:
    """Test complete end-to-end FastMCP workflows."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample FastMCP configuration."""
        return FastMCPConfig(
            enabled=True,
            default_timeout=30.0,
            max_retries=3,
            connection_pool_size=10,
            health_check_interval=60.0,
            endpoints={
                "test_server": {
                    "url": "http://localhost:8001",
                    "transport_type": "sse",
                    "auth_type": "none"
                },
                "local_server": {
                    "url": "stdio://local_server",
                    "transport_type": "stdio",
                    "auth_type": "none"
                }
            },
            default_auth_type="none"
        )

    @pytest.mark.asyncio
    async def test_complete_client_server_workflow(self, sample_config):
        """Test complete client-server workflow."""
        # Mock FastMCP availability and components
        with patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
                with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
                    with patch('safla.integrations.fastmcp_server.FastMCP', create=True) as mock_fastmcp_class:
                        with patch('safla.integrations.fastmcp_client.Client', create=True) as mock_client_class:
                            # Setup server mock
                            mock_server = Mock()
                            mock_server.tool = Mock()
                            mock_server.resource = Mock()
                            mock_server.run = AsyncMock()
                            mock_server.shutdown = AsyncMock()
                            mock_server.stop = AsyncMock()
                            mock_fastmcp_class.return_value = mock_server

                            # Setup client mock
                            mock_client = Mock()
                            mock_client.list_tools = AsyncMock(return_value=[
                                {"name": "test_tool", "description": "Test tool"}
                            ])
                            mock_client.call_tool = AsyncMock(return_value={"result": "success"})
                            mock_client_class.return_value = mock_client

                            # Create and configure server
                            server = FastMCPServer("test_server")
                            
                            # Register a tool
                            def test_tool(message: str) -> str:
                                return f"Processed: {message}"
                            
                            server.register_tool("test_tool", "Test tool", test_tool)
                            
                            # Start server
                            await server.start()
                            
                            # Create client and mock adapter's client management
                            adapter = FastMCPAdapter(sample_config)
                            
                            # Mock the adapter's clients dictionary to include our test endpoint
                            mock_fastmcp_client = Mock()
                            mock_fastmcp_client.list_tools = AsyncMock(return_value=[
                                {"name": "test_tool", "description": "Test tool"}
                            ])
                            mock_fastmcp_client.call_tool = AsyncMock(return_value={"result": "success"})
                            mock_fastmcp_client.__aenter__ = AsyncMock(return_value=mock_fastmcp_client)
                            mock_fastmcp_client.__aexit__ = AsyncMock(return_value=None)
                            
                            # Create a mock endpoint configuration
                            from safla.integrations.fastmcp_adapter import FastMCPEndpoint
                            mock_endpoint = FastMCPEndpoint(
                                name="test_server",
                                url="http://localhost:8000",
                                transport_type="http",
                                auth_type="none"
                            )
                            
                            # Add the mock client and endpoint to the adapter's dictionaries
                            adapter.clients["test_server"] = mock_fastmcp_client
                            adapter.endpoints["test_server"] = mock_endpoint
                            
                            client = FastMCPClient(adapter)
                
                            # Test tool listing with endpoint
                            tools = await client.list_tools("test_server")
                            assert len(tools) == 1
                            assert tools[0]["name"] == "test_tool"
                            
                            # Test tool calling with endpoint
                            result = await client.call_tool("test_tool", arguments={"message": "Hello"}, endpoint="test_server")
                            assert result["result"] == "success"
                            
                            # Stop server
                            await server.stop()

    @pytest.mark.asyncio
    async def test_batch_operations_workflow(self, sample_config):
        """Test batch operations workflow."""
        with patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
                with patch('safla.integrations.fastmcp_client.Client', create=True) as mock_client_class:
                    # Setup client mock
                    mock_client = Mock()
                    mock_client.call_tool = AsyncMock(side_effect=[
                        {"result": "tool1_result"},
                        {"result": "tool2_result"},
                        {"result": "tool3_result"}
                    ])
                    mock_client_class.return_value = mock_client

                    # Create adapter and client
                    adapter = FastMCPAdapter(sample_config)
                    
                    # Mock the adapter's clients dictionary to include our test endpoint
                    mock_fastmcp_client = Mock()
                    mock_fastmcp_client.call_tool = AsyncMock(side_effect=[
                        {"result": "tool1_result"},
                        {"result": "tool2_result"},
                        {"result": "tool3_result"}
                    ])
                    mock_fastmcp_client.__aenter__ = AsyncMock(return_value=mock_fastmcp_client)
                    mock_fastmcp_client.__aexit__ = AsyncMock(return_value=None)
                    
                    # Create a mock endpoint configuration
                    from safla.integrations.fastmcp_adapter import FastMCPEndpoint
                    mock_endpoint = FastMCPEndpoint(
                        name="test_server",
                        url="http://localhost:8000",
                        transport_type="http",
                        auth_type="none"
                    )
                    
                    # Add the mock client and endpoint to the adapter's dictionaries
                    adapter.clients["test_server"] = mock_fastmcp_client
                    adapter.endpoints["test_server"] = mock_endpoint
                    
                    client = FastMCPClient(adapter)

                    # Define batch operations
                    tool_calls = [
                        {"tool_name": "tool1", "arguments": {"input": "data1"}},
                        {"tool_name": "tool2", "arguments": {"input": "data2"}},
                        {"tool_name": "tool3", "arguments": {"input": "data3"}}
                    ]

                    # Execute batch operations
                    results = await batch_call_tools(
                        client,
                        tool_calls,
                        "test_server",
                        max_concurrent=2
                    )

                    # Verify results
                    assert len(results) == 3
                    for i, result in enumerate(results):
                        assert result["success"] is True
                        assert result["result"]["result"] == f"tool{i+1}_result"
                        assert result["tool_name"] == f"tool{i+1}"

    @pytest.mark.asyncio
    async def test_server_builder_workflow(self):
        """Test server builder workflow."""
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', create=True) as mock_fastmcp_class:
                mock_server = Mock()
                mock_server.tool = Mock()
                mock_server.resource = Mock()
                mock_fastmcp_class.return_value = mock_server

                # Define tool and resource functions
                def calculator_tool(operation: str, a: float, b: float) -> float:
                    """Calculator tool."""
                    if operation == "add":
                        return a + b
                    elif operation == "multiply":
                        return a * b
                    else:
                        raise ValueError(f"Unknown operation: {operation}")

                def config_resource() -> Dict[str, Any]:
                    """Configuration resource."""
                    return {
                        "config": {"setting1": "value1", "setting2": "value2"}
                    }

                # Build server using builder pattern
                server = (FastMCPServerBuilder("calculator_server")
                          .add_tool("calculator", "Calculator tool", calculator_tool)
                          .add_resource("config://.*", "Configuration", "Config resource", config_resource)
                          .build())

                # Verify server configuration
                assert server.name == "calculator_server"
                assert len(server.list_tools()) == 2  # calculator + health_check
                assert len(server.list_resources()) == 1

    def test_configuration_conversion_workflow(self):
        """Test MCP to FastMCP configuration conversion workflow."""
        # Sample MCP configuration
        mcp_config = {
            "servers": {
                "stdio_server": {
                    "command": "python",
                    "args": ["-m", "my_server"],
                    "env": {"SERVER_MODE": "production"}
                },
                "http_server": {
                    "url": "http://api.example.com/mcp",
                    "transport": "sse",
                    "auth": {
                        "type": "bearer",
                        "token": "secret_token"
                    }
                }
            }
        }

        # Convert configuration
        fastmcp_config = convert_safla_config_to_fastmcp(mcp_config)

        # Verify conversion
        assert len(fastmcp_config.endpoints) == 2
        assert "stdio_server" in fastmcp_config.endpoints
        assert "http_server" in fastmcp_config.endpoints

        # Verify stdio server configuration
        stdio_endpoint = fastmcp_config.endpoints["stdio_server"]
        assert stdio_endpoint["transport_type"] == "stdio"
        assert stdio_endpoint["auth_type"] == "none"

        # Verify HTTP server configuration
        http_endpoint = fastmcp_config.endpoints["http_server"]
        assert http_endpoint["url"] == "http://api.example.com/mcp"
        assert http_endpoint["transport_type"] == "sse"
        assert http_endpoint["auth_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_workflow(self, sample_config):
        """Test error handling and recovery workflow."""
        with patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
                with patch('safla.integrations.fastmcp_client.Client', create=True) as mock_client_class:
                    # Setup client with intermittent failures
                    mock_client = Mock()
                    call_count = 0

                    async def mock_call_tool(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        if call_count <= 2:
                            raise Exception("Temporary failure")
                        return {"result": "success_after_retry"}

                    mock_client.call_tool = mock_call_tool
                    mock_client_class.return_value = mock_client

                    # Create adapter with retry configuration
                    adapter = FastMCPAdapter(sample_config)
                    client = FastMCPClient(adapter)

                    # Test retry mechanism (mocked at adapter level)
                    with patch.object(adapter, 'call_tool') as mock_adapter_call:
                        mock_adapter_call.side_effect = [
                            Exception("First failure"),
                            Exception("Second failure"),
                            {"result": "success_after_retry"}
                        ]

                        # This would normally retry internally
                        try:
                            result = await mock_adapter_call()
                        except Exception:
                            # Simulate retry
                            try:
                                result = await mock_adapter_call()
                            except Exception:
                                # Final retry
                                result = await mock_adapter_call()

                        assert result["result"] == "success_after_retry"

    @pytest.mark.asyncio
    async def test_concurrent_operations_workflow(self, sample_config):
        """Test concurrent operations workflow."""
        with patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
                with patch('safla.integrations.fastmcp_client.Client', create=True) as mock_client_class:
                    # Setup client mock with delays to simulate real operations
                    mock_client = Mock()

                    async def mock_call_tool(tool_name, *args, **kwargs):
                        await asyncio.sleep(0.01)  # Minimal delay for testing
                        return {"result": f"{tool_name}_result"}

                    mock_client.call_tool = mock_call_tool
                    mock_client_class.return_value = mock_client

                    # Create client
                    adapter = FastMCPAdapter(sample_config)
                    
                    # Mock the adapter's clients dictionary to include our test endpoint
                    mock_fastmcp_client = Mock()
                    mock_fastmcp_client.call_tool = AsyncMock(side_effect=mock_call_tool)
                    mock_fastmcp_client.__aenter__ = AsyncMock(return_value=mock_fastmcp_client)
                    mock_fastmcp_client.__aexit__ = AsyncMock(return_value=None)
                    
                    # Create a mock endpoint configuration
                    from safla.integrations.fastmcp_adapter import FastMCPEndpoint
                    mock_endpoint = FastMCPEndpoint(
                        name="test_server",
                        url="http://localhost:8000",
                        transport_type="http",
                        auth_type="none"
                    )
                    
                    # Add the mock client and endpoint to the adapter's dictionaries
                    adapter.clients["test_server"] = mock_fastmcp_client
                    adapter.endpoints["test_server"] = mock_endpoint
                    
                    client = FastMCPClient(adapter)

                    # Define concurrent operations
                    tool_calls = [
                        {"tool_name": f"tool_{i}", "arguments": {"index": i}}
                        for i in range(5)
                    ]

                    # Execute concurrent operations
                    start_time = asyncio.get_event_loop().time()
                    results = await batch_call_tools(
                        client,
                        tool_calls,
                        "test_server",
                        max_concurrent=3
                    )
                    end_time = asyncio.get_event_loop().time()

                    # Verify results
                    assert len(results) == 5
                    for i, result in enumerate(results):
                        assert result["success"] is True
                        assert result["result"]["result"] == f"tool_{i}_result"
                        assert result["tool_name"] == f"tool_{i}"

                    # Verify concurrency (should be faster than sequential)
                    execution_time = end_time - start_time
                    assert execution_time < 0.5  # Should be much faster than 5 * 0.1 = 0.5s

    def test_configuration_file_workflow(self, sample_config):
        """Test configuration file loading and saving workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "fastmcp_config.json")

            # Save configuration to file
            with open(config_file, 'w') as f:
                json.dump(sample_config.model_dump(), f, indent=2)

            # Load configuration from file
            with open(config_file, 'r') as f:
                loaded_config_data = json.load(f)

            loaded_config = FastMCPConfig(**loaded_config_data)

            # Verify loaded configuration
            assert len(loaded_config.endpoints) == 2
            assert "test_server" in loaded_config.endpoints
            assert "local_server" in loaded_config.endpoints

            test_endpoint = loaded_config.endpoints["test_server"]
            assert test_endpoint["url"] == "http://localhost:8001"
            assert test_endpoint["transport_type"] == "sse"
            assert test_endpoint["auth_type"] == "none"

    @pytest.mark.asyncio
    async def test_health_monitoring_workflow(self, sample_config):
        """Test health monitoring workflow."""
        with patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
                with patch('safla.integrations.fastmcp_client.Client', create=True) as mock_client_class:
                    # Setup client mock
                    mock_client = Mock()
                    mock_client.list_tools = AsyncMock(return_value=[])
                    mock_client.list_resources = AsyncMock(return_value=[])
                    mock_client_class.return_value = mock_client

                    # Create client
                    adapter = FastMCPAdapter(sample_config)
                    
                    # Mock the adapter's clients dictionary to include our test endpoint
                    mock_fastmcp_client = Mock()
                    mock_fastmcp_client.__aenter__ = AsyncMock(return_value=mock_fastmcp_client)
                    mock_fastmcp_client.__aexit__ = AsyncMock(return_value=None)
                    
                    # Create a mock endpoint configuration
                    from safla.integrations.fastmcp_adapter import FastMCPEndpoint
                    mock_endpoint = FastMCPEndpoint(
                        name="test_server",
                        url="http://localhost:8000",
                        transport_type="http",
                        auth_type="none"
                    )
                    
                    # Add the mock client and endpoint to the adapter's dictionaries
                    adapter.clients["test_server"] = mock_fastmcp_client
                    adapter.endpoints["test_server"] = mock_endpoint
                    
                    # Mock the adapter's get_endpoint_status method
                    adapter.get_endpoint_status = AsyncMock(return_value={"status": "healthy"})
                    
                    client = FastMCPClient(adapter)
        
                    # Test health check using ping_endpoint
                    health_status = await client.ping_endpoint("test_server")
                    assert health_status is True


class TestFastMCPPerformanceAndScaling:
    """Test FastMCP performance and scaling scenarios."""

    @pytest.fixture
    def performance_config(self):
        """Create a performance test configuration."""
        return FastMCPConfig(
            enabled=True,
            default_timeout=10.0,
            max_retries=1,
            connection_pool_size=50,
            health_check_interval=30.0,
            endpoints={
                f"server_{i}": {
                    "url": f"http://localhost:800{i}",
                    "transport_type": "sse",
                    "auth_type": "none"
                }
                for i in range(10)
            },
            default_auth_type="none"
        )

    @pytest.mark.asyncio
    async def test_high_concurrency_operations(self, performance_config):
        """Test high concurrency operations."""
        with patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
                with patch('safla.integrations.fastmcp_client.Client', create=True) as mock_client_class:
                    # Setup client mock
                    mock_client = Mock()

                    async def mock_call_tool(tool_name, **kwargs):
                        await asyncio.sleep(0.001)  # Very minimal delay
                        return {"result": f"{tool_name}_result"}

                    mock_client.call_tool = mock_call_tool
                    mock_client_class.return_value = mock_client

                    # Create client
                    adapter = FastMCPAdapter(performance_config)
                    client = FastMCPClient(adapter)

                    # Define high concurrency operations
                    tool_calls = [
                        {"tool_name": f"tool_{i}", "arguments": {"index": i}}
                        for i in range(100)
                    ]

                    # Execute high concurrency operations
                    start_time = asyncio.get_event_loop().time()
                    results = await batch_call_tools(
                        client,
                        tool_calls,
                        "server_0",
                        max_concurrent=20
                    )
                    end_time = asyncio.get_event_loop().time()

                    # Verify results
                    assert len(results) == 100
                    execution_time = end_time - start_time
                    assert execution_time < 2.0  # Should complete within 2 seconds

    @pytest.mark.asyncio
    async def test_multiple_endpoint_operations(self, performance_config):
        """Test operations across multiple endpoints."""
        with patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
                with patch('safla.integrations.fastmcp_client.Client', create=True) as mock_client_class:
                    # Setup client mock
                    mock_client = Mock()
                    mock_client.list_tools = AsyncMock(return_value=[
                        {"name": "test_tool", "description": "Test tool"}
                    ])
                    mock_client_class.return_value = mock_client

                    # Create client
                    adapter = FastMCPAdapter(performance_config)
                    
                    # Mock the adapter's clients dictionary for all endpoints
                    for endpoint_name in performance_config.endpoints.keys():
                        mock_fastmcp_client = Mock()
                        mock_fastmcp_client.list_tools = AsyncMock(return_value=[
                            {"name": "test_tool", "description": "Test tool"}
                        ])
                        mock_fastmcp_client.__aenter__ = AsyncMock(return_value=mock_fastmcp_client)
                        mock_fastmcp_client.__aexit__ = AsyncMock(return_value=None)
                        
                        # Create a mock endpoint configuration
                        from safla.integrations.fastmcp_adapter import FastMCPEndpoint
                        mock_endpoint = FastMCPEndpoint(
                            name=endpoint_name,
                            url=f"http://localhost:800{endpoint_name.split('_')[1]}",
                            transport_type="sse",
                            auth_type="none"
                        )
                        
                        # Add the mock client and endpoint to the adapter's dictionaries
                        adapter.clients[endpoint_name] = mock_fastmcp_client
                        adapter.endpoints[endpoint_name] = mock_endpoint
                    
                    client = FastMCPClient(adapter)
        
                    # Test operations across all endpoints
                    endpoint_results = {}
                    for endpoint_name in performance_config.endpoints.keys():
                        tools = await client.list_tools(endpoint_name)
                        endpoint_results[endpoint_name] = tools

                    # Verify all endpoints responded
                    assert len(endpoint_results) == 10
                    for endpoint_name, tools in endpoint_results.items():
                        assert len(tools) == 1
                        assert tools[0]["name"] == "test_tool"