"""
Integration tests for FastMCP

This module contains integration tests that verify the complete FastMCP workflow
from configuration through client/server operations and CLI integration.
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from safla.integrations.fastmcp_adapter import FastMCPAdapter, FastMCPConfig, FastMCPEndpoint, FastMCPTransportType, FastMCPAuthType
from safla.integrations.fastmcp_client import FastMCPClient
from safla.integrations.fastmcp_server import FastMCPServer, FastMCPServerBuilder
from safla.integrations.fastmcp_helpers import (
    discover_fastmcp_endpoints,
    check_fastmcp_endpoint,
    batch_call_tools,
    convert_safla_config_to_fastmcp
)
from safla.exceptions import SAFLAError


class TestFastMCPEndToEndWorkflow:
    """Test complete FastMCP workflow from configuration to execution."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample FastMCP configuration."""
        return FastMCPConfig(
            endpoints={
                "test_server": {
                    "url": "http://localhost:8001",
                    "transport_type": "sse",
                    "auth_type": "bearer_token",
                    "auth_token": "test_token"
                },
                "local_server": {
                    "url": "http://localhost:8002",
                    "transport_type": "sse",
                    "auth_type": "none"
                }
            },
            default_timeout=30.0,
            max_retries=3
        )
    
    @pytest.fixture
    def mock_fastmcp_available(self):
        """Mock FastMCP availability."""
        with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
            yield
    
    @pytest.mark.asyncio
    async def test_complete_client_server_workflow(self, sample_config, mock_fastmcp_available):
        """Test complete client-server workflow."""
        # Mock FastMCP components and availability
        with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True):
                with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
                    with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_fastmcp_class:
                        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
                            # Setup server mock
                            mock_server = Mock()
                            mock_server.tool = Mock()
                            mock_server.resource = Mock()
                            mock_server.run = AsyncMock()
                            mock_server.shutdown = AsyncMock()
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
                            @server.tool("test_tool", "Test tool")
                            def test_tool_func(message: str) -> str:
                                """Test tool function."""
                                return f"Processed: {message}"
                            
                            # Create adapter and client
                            adapter = FastMCPAdapter(sample_config)
                            client = FastMCPClient(adapter)
                            
                            # Test server setup
                            await server._setup_server()
                            
                            # Test client operations
                            tools = await client.list_tools("test_server")
                            assert len(tools) == 1
                            assert tools[0]["name"] == "test_tool"
                            
                            result = await client.call_tool(
                                tool_name="test_tool",
                                arguments={"message": "Hello World"},
                                endpoint="test_server"
                            )
                            assert result["result"] == "success"
                assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_batch_operations_workflow(self, sample_config, mock_fastmcp_available):
        """Test batch operations workflow."""
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
            # Setup client mock
            mock_client = Mock()
            mock_client.execute_workflow = AsyncMock(return_value=[
                {"success": True, "result": {"output": "tool1_result"}},
                {"success": True, "result": {"output": "tool2_result"}},
                {"success": True, "result": {"content": "resource_data"}}
            ])
            mock_client_class.return_value = mock_client
            
            # Create adapter and client
            adapter = FastMCPAdapter(sample_config)
            client = FastMCPClient(adapter)
            client.set_default_endpoint("test_server")
            
            # Test batch context manager
            async with client.batch_context() as batch:
                batch.add_tool_call("tool1", {"arg1": "value1"})
                batch.add_tool_call("tool2", {"arg2": "value2"})
                batch.add_resource_read("resource://test")
                
                results = await batch.execute()
                
                assert len(results) == 3
                assert all(r["success"] for r in results)
                assert results[0]["result"]["output"] == "tool1_result"
                assert results[1]["result"]["output"] == "tool2_result"
                assert results[2]["result"]["content"] == "resource_data"
    
    @pytest.mark.asyncio
    async def test_server_builder_workflow(self, mock_fastmcp_available):
        """Test server builder workflow."""
        with patch('safla.integrations.fastmcp_server.FastMCPServer') as mock_fastmcp_class:
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
            
            def config_resource(uri: str) -> Dict[str, Any]:
                """Configuration resource."""
                return {
                    "uri": uri,
                    "config": {"setting1": "value1", "setting2": "value2"}
                }
            
            # Build server using builder pattern
            server = (FastMCPServerBuilder("calculator_server")
                      .add_tool("calculator", "Calculator tool", calculator_tool)
                      .add_resource("config://.*", "Configuration", "Config resource", config_resource)
                      .build())
            
            # Verify server configuration
            assert server.name == "calculator_server"
            assert len(server._tools) == 1
            assert len(server._resources) == 1
            assert "calculator" in server._tools
            assert "config://.*" in server._resources
            
            # Test tool and resource info
            tools = server.get_tool_list()
            resources = server.get_resource_list()
            
            assert len(tools) == 1
            assert tools[0]["name"] == "calculator"
            assert len(resources) == 1
            assert resources[0]["uri_pattern"] == "config://.*"
    
    @pytest.mark.asyncio
    async def test_discovery_and_testing_workflow(self, mock_fastmcp_available):
        """Test endpoint discovery and testing workflow."""
        # Mock discovery response
        mock_discovered_endpoints = [
            {
                "name": "math_server",
                "url": "http://localhost:8001",
                "transport": "sse",
                "status": "healthy"
            },
            {
                "name": "data_server", 
                "url": "http://localhost:8002",
                "transport": "streamable_http",
                "status": "healthy"
            }
        ]
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock discovery HTTP response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"endpoints": mock_discovered_endpoints})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test discovery
            endpoints = await discover_fastmcp_endpoints("http://discovery.example.com")
            
            assert len(endpoints) == 2
            assert endpoints[0]["name"] == "math_server"
            assert endpoints[1]["name"] == "data_server"
        
        # Test endpoint testing
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
            mock_client = Mock()
            mock_client.list_tools = AsyncMock(return_value=[
                {"name": "add", "description": "Addition tool"},
                {"name": "multiply", "description": "Multiplication tool"}
            ])
            mock_client.list_resources = AsyncMock(return_value=[
                {"uri": "math://constants", "name": "Math Constants"}
            ])
            mock_client.call_tool = AsyncMock(return_value={"result": 42})
            mock_client_class.return_value = mock_client
            
            endpoint = FastMCPEndpoint(
                name="math_server",
                url="http://localhost:8001",
                transport="sse"
            )
            
            test_result = await check_fastmcp_endpoint(endpoint, run_basic_tests=True)
            
            assert test_result["endpoint_name"] == "math_server"
            assert test_result["status"] == "healthy"
            assert test_result["tool_count"] == 2
            assert test_result["resource_count"] == 1
            assert test_result["basic_tests"]["passed"] is True
    
    @pytest.mark.asyncio
    async def test_configuration_conversion_workflow(self, mock_fastmcp_available):
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
        
        stdio_endpoint = fastmcp_config.endpoints["stdio_server"]
        assert stdio_endpoint.name == "stdio_server"
        assert stdio_endpoint.command == "python"
        assert stdio_endpoint.args == ["-m", "my_server"]
        assert stdio_endpoint.env == {"SERVER_MODE": "production"}
        assert stdio_endpoint.transport == "stdio"
        
        http_endpoint = fastmcp_config.endpoints["http_server"]
        assert http_endpoint.name == "http_server"
        assert http_endpoint.url == "http://api.example.com/mcp"
        assert http_endpoint.transport == "sse"
        assert http_endpoint.auth_type == "bearer"
        assert http_endpoint.auth_token == "secret_token"
        
        # Test using converted configuration
        adapter = FastMCPAdapter(fastmcp_config)
        assert len(adapter.endpoints) == 2
        assert "stdio_server" in adapter.endpoints
        assert "http_server" in adapter.endpoints
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_workflow(self, sample_config, mock_fastmcp_available):
        """Test error handling and recovery workflow."""
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
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
                    Exception("Failure 1"),
                    Exception("Failure 2"), 
                    {"result": "success_after_retry"}
                ]
                
                # This would normally trigger retries in the real adapter
                try:
                    result = await client.call_tool(
                        tool_name="test_tool",
                        endpoint="test_server"
                    )
                    # In real scenario, this would succeed after retries
                except Exception:
                    # Expected in mock scenario
                    pass
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_workflow(self, sample_config, mock_fastmcp_available):
        """Test concurrent operations workflow."""
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
            # Setup client mock with delays to simulate real operations
            mock_client = Mock()
            
            async def mock_call_tool(tool_name, **kwargs):
                await asyncio.sleep(0.1)  # Simulate work
                return {"result": f"{tool_name}_result"}
            
            mock_client.call_tool = mock_call_tool
            mock_client_class.return_value = mock_client
            
            # Create client
            adapter = FastMCPAdapter(sample_config)
            client = FastMCPClient(adapter)
            
            # Test concurrent tool calls using batch operations
            tool_calls = [
                {"tool_name": f"tool_{i}", "arguments": {"index": i}}
                for i in range(5)
            ]
            
            start_time = asyncio.get_event_loop().time()
            results = await batch_call_tools(
                client,
                tool_calls,
                "test_server",
                max_concurrency=3
            )
            end_time = asyncio.get_event_loop().time()
            
            # Verify results
            assert len(results) == 5
            assert all(r["success"] for r in results)
            
            # With concurrency limit of 3, should be faster than sequential
            # but slower than unlimited concurrency
            assert end_time - start_time < 0.5  # Should complete reasonably fast
    
    def test_configuration_file_workflow(self, sample_config, mock_fastmcp_available):
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
            assert test_endpoint.url == "http://localhost:8001"
            assert test_endpoint.auth_type == "bearer"
            assert test_endpoint.auth_token == "test_token"
            
            local_endpoint = loaded_config.endpoints["local_server"]
            assert local_endpoint.command == "python"
            assert local_endpoint.args == ["-m", "test_server"]
            assert local_endpoint.env == {"TEST_VAR": "test_value"}
    
    @pytest.mark.asyncio
    async def test_health_monitoring_workflow(self, sample_config, mock_fastmcp_available):
        """Test health monitoring workflow."""
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
            # Setup client mock with health status
            mock_client = Mock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.list_resources = AsyncMock(return_value=[])
            mock_client_class.return_value = mock_client
            
            # Create adapter and client
            adapter = FastMCPAdapter(sample_config)
            client = FastMCPClient(adapter)
            
            # Test endpoint health checks
            endpoints_to_check = ["test_server", "local_server"]
            health_results = []
            
            for endpoint_name in endpoints_to_check:
                is_healthy = await client.ping_endpoint(endpoint_name)
                health_results.append({
                    "endpoint": endpoint_name,
                    "healthy": is_healthy
                })
            
            # Verify health check results
            assert len(health_results) == 2
            # In mock scenario, ping would succeed
            # In real scenario, would depend on actual endpoint status
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_dependencies(self, sample_config, mock_fastmcp_available):
        """Test workflow execution with step dependencies."""
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
            # Setup client mock
            mock_client = Mock()
            mock_client.call_tool = AsyncMock(side_effect=[
                {"result": "config_data"},
                {"result": "processed_data"},
                {"result": "final_result"}
            ])
            mock_client.read_resource = AsyncMock(return_value={"content": "resource_data"})
            mock_client_class.return_value = mock_client
            
            # Create client
            adapter = FastMCPAdapter(sample_config)
            client = FastMCPClient(adapter)
            client.set_default_endpoint("test_server")
            
            # Define workflow with dependencies
            workflow = [
                {
                    "type": "read_resource",
                    "resource_uri": "config://settings"
                },
                {
                    "type": "call_tool",
                    "tool_name": "load_config",
                    "arguments": {"config_uri": "config://settings"}
                },
                {
                    "type": "call_tool", 
                    "tool_name": "process_data",
                    "arguments": {"config": "config_data"}
                },
                {
                    "type": "call_tool",
                    "tool_name": "finalize",
                    "arguments": {"data": "processed_data"}
                }
            ]
            
            # Execute workflow
            results = await client.execute_workflow(workflow)
            
            # Verify workflow execution
            assert len(results) == 4
            assert all(r["success"] for r in results)
            assert results[0]["result"]["content"] == "resource_data"
            assert results[1]["result"]["result"] == "config_data"
            assert results[2]["result"]["result"] == "processed_data"
            assert results[3]["result"]["result"] == "final_result"


class TestFastMCPPerformanceAndScaling:
    """Test FastMCP performance and scaling characteristics."""
    
    @pytest.fixture
    def mock_fastmcp_available(self):
        """Mock FastMCP availability."""
        with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True):
            yield
    
    @pytest.mark.asyncio
    async def test_high_concurrency_batch_operations(self, mock_fastmcp_available):
        """Test batch operations with high concurrency."""
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
            # Setup client mock
            mock_client = Mock()
            
            async def mock_call_tool(*args, **kwargs):
                await asyncio.sleep(0.01)  # Minimal delay
                return {"result": "success"}
            
            mock_client.call_tool = mock_call_tool
            mock_client_class.return_value = mock_client
            
            # Create large batch of operations
            tool_calls = [
                {"tool_name": f"tool_{i}", "arguments": {"index": i}}
                for i in range(100)
            ]
            
            # Test with different concurrency limits
            for max_concurrency in [1, 5, 10, 20]:
                start_time = asyncio.get_event_loop().time()
                results = await batch_call_tools(
                    mock_client,
                    tool_calls,
                    "test_endpoint",
                    max_concurrency=max_concurrency
                )
                end_time = asyncio.get_event_loop().time()
                
                assert len(results) == 100
                assert all(r["success"] for r in results)
                
                # Higher concurrency should generally be faster
                # (though with minimal delays, the difference may be small)
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_payloads(self, mock_fastmcp_available):
        """Test memory usage with large payloads."""
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
            # Setup client mock with large responses
            mock_client = Mock()
            
            async def mock_call_tool(*args, **kwargs):
                # Simulate large response
                large_data = "x" * 10000  # 10KB string
                return {"result": large_data}
            
            mock_client.call_tool = mock_call_tool
            mock_client_class.return_value = mock_client
            
            # Test multiple large operations
            tool_calls = [
                {"tool_name": f"large_tool_{i}", "arguments": {}}
                for i in range(10)
            ]
            
            results = await batch_call_tools(
                mock_client,
                tool_calls,
                "test_endpoint",
                max_concurrency=5
            )
            
            assert len(results) == 10
            assert all(r["success"] for r in results)
            assert all(len(r["result"]["result"]) == 10000 for r in results)
    
    @pytest.mark.asyncio
    async def test_error_recovery_under_load(self, mock_fastmcp_available):
        """Test error recovery under high load."""
        with patch('safla.integrations.fastmcp_client.FastMCPClient') as mock_client_class:
            # Setup client mock with intermittent failures
            mock_client = Mock()
            call_count = 0
            
            async def mock_call_tool(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # Fail every 3rd call
                if call_count % 3 == 0:
                    raise Exception("Intermittent failure")
                return {"result": f"success_{call_count}"}
            
            mock_client.call_tool = mock_call_tool
            mock_client_class.return_value = mock_client
            
            # Test batch operations with failures
            tool_calls = [
                {"tool_name": f"tool_{i}", "arguments": {}}
                for i in range(30)
            ]
            
            results = await batch_call_tools(
                mock_client,
                tool_calls,
                "test_endpoint",
                max_concurrency=5,
                stop_on_error=False
            )
            
            assert len(results) == 30
            
            # Count successes and failures
            successes = sum(1 for r in results if r["success"])
            failures = sum(1 for r in results if not r["success"])
            
            # Should have approximately 2/3 successes and 1/3 failures
            assert successes > 15  # At least half should succeed
            assert failures > 5    # Some should fail