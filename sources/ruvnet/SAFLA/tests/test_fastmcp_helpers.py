"""
Unit tests for FastMCP Helper Functions

This module contains comprehensive tests for the FastMCP helper functions,
including endpoint discovery, testing, configuration conversion, and utilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional

from safla.integrations.fastmcp_helpers import (
    discover_fastmcp_endpoints,
    check_fastmcp_endpoint,
    batch_call_tools,
    create_fastmcp_proxy_server,
    convert_safla_config_to_fastmcp,
    convert_fastmcp_config_to_safla,
    create_fastmcp_client_from_config,
    create_fastmcp_server_from_modules,
    fastmcp_tool,
    fastmcp_resource
)
from safla.integrations.fastmcp_adapter import FastMCPConfig, FastMCPEndpoint, FastMCPTransportType, FastMCPAuthType
from safla.exceptions import SAFLAError


class TestEndpointDiscovery:
    """Test endpoint discovery functionality."""
    
    @pytest.mark.asyncio
    async def test_discover_fastmcp_endpoints_success(self):
        """Test successful endpoint discovery."""
        mock_endpoints = [
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
        
        # Skip test if aiohttp not available
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not available")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"endpoints": mock_endpoints})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            endpoints = await discover_fastmcp_endpoints("http://discovery.example.com")
            
            assert len(endpoints) == 2
            assert endpoints[0]["name"] == "endpoint1"
            assert endpoints[1]["name"] == "endpoint2"
    
    @pytest.mark.asyncio
    async def test_discover_fastmcp_endpoints_with_filter(self):
        """Test endpoint discovery with status filter."""
        mock_endpoints = [
            {"name": "healthy1", "status": "healthy"},
            {"name": "unhealthy1", "status": "unhealthy"},
            {"name": "healthy2", "status": "healthy"}
        ]
        
        # Skip test if aiohttp not available
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not available")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"endpoints": mock_endpoints})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            endpoints = await discover_fastmcp_endpoints(
                "http://discovery.example.com",
                filter_status="healthy"
            )
            
            assert len(endpoints) == 2
            assert all(ep["status"] == "healthy" for ep in endpoints)
    
    @pytest.mark.asyncio
    async def test_discover_fastmcp_endpoints_http_error(self):
        """Test endpoint discovery with HTTP error."""
        # Skip test if aiohttp not available
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not available")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 404
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(SAFLAError, match="Failed to discover endpoints"):
                await discover_fastmcp_endpoints("http://discovery.example.com")
    
    @pytest.mark.asyncio
    async def test_discover_fastmcp_endpoints_connection_error(self):
        """Test endpoint discovery with connection error."""
        # Skip test if aiohttp not available
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not available")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            with pytest.raises(SAFLAError, match="Failed to discover endpoints"):
                await discover_fastmcp_endpoints("http://discovery.example.com")


class TestEndpointTesting:
    """Test endpoint testing functionality."""
    
    @pytest.mark.asyncio
    async def test_check_fastmcp_endpoint_success(self):
        """Test successful endpoint testing."""
        # Skip test if aiohttp not available
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not available")
            
        endpoint_url = "http://localhost:8001"
        
        with patch('safla.integrations.fastmcp_helpers.FastMCPAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.start = AsyncMock()
            mock_adapter.stop = AsyncMock()
            mock_adapter.get_endpoint_status = AsyncMock(return_value={"status": "healthy"})
            mock_adapter.list_tools = AsyncMock(return_value=[
                {"name": "tool1", "description": "Test tool 1"}
            ])
            mock_adapter.list_resources = AsyncMock(return_value=[
                {"uri": "resource://test1", "name": "Test Resource 1"}
            ])
            mock_adapter_class.return_value = mock_adapter
            
            result = await check_fastmcp_endpoint(endpoint_url)
            
            assert result["endpoint_url"] == endpoint_url
            assert result["connectivity"] is True
            assert "capabilities" in result
    
    @pytest.mark.asyncio
    async def test_check_fastmcp_endpoint_connection_failure(self):
        """Test endpoint testing with connection failure."""
        # Skip test if aiohttp not available
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not available")
            
        endpoint_url = "http://localhost:8001"
        
        with patch('safla.integrations.fastmcp_helpers.FastMCPAdapter') as mock_adapter_class:
            mock_adapter_class.side_effect = Exception("Connection failed")
            
            result = await check_fastmcp_endpoint(endpoint_url)
            
            assert result["endpoint_url"] == endpoint_url
            assert result["connectivity"] is False
            assert "Connection failed" in str(result["errors"])
    
    @pytest.mark.asyncio
    async def test_check_fastmcp_endpoint_tool_failure(self):
        """Test endpoint testing with tool call failure."""
        # Skip test if aiohttp not available
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not available")
            
        endpoint_url = "http://localhost:8001"
        
        with patch('safla.integrations.fastmcp_helpers.FastMCPAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.start = AsyncMock()
            mock_adapter.stop = AsyncMock()
            mock_adapter.get_endpoint_status = AsyncMock(return_value={"status": "healthy"})
            mock_adapter.list_tools = AsyncMock(side_effect=Exception("Tool failed"))
            mock_adapter.list_resources = AsyncMock(return_value=[])
            mock_adapter_class.return_value = mock_adapter
            
            result = await check_fastmcp_endpoint(endpoint_url)
            
            assert result["connectivity"] is True  # Connection works
            assert "Tool failed" in str(result["errors"])
            assert "Tool failed" in result["basic_tests"]["error"]


class TestBatchOperations:
    """Test batch operation helper functions."""
    
    @pytest.mark.asyncio
    async def test_batch_call_tools_success(self):
        """Test successful batch tool calls."""
        mock_client = Mock()
        mock_client.call_tool = AsyncMock(side_effect=[
            {"result": "tool1_result"},
            {"result": "tool2_result"}
        ])
        
        tool_calls = [
            {"tool_name": "tool1", "arguments": {"arg1": "value1"}},
            {"tool_name": "tool2", "arguments": {"arg2": "value2"}}
        ]
        
        results = await batch_call_tools(mock_client, tool_calls, "test_endpoint")
        
        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[0]["result"] == {"result": "tool1_result"}
        assert results[1]["success"] is True
        assert results[1]["result"] == {"result": "tool2_result"}
    
    @pytest.mark.asyncio
    async def test_batch_call_tools_with_errors(self):
        """Test batch tool calls with some errors."""
        mock_client = Mock()
        mock_client.call_tool = AsyncMock(side_effect=[
            {"result": "tool1_result"},
            Exception("Tool2 failed")
        ])
        
        tool_calls = [
            {"tool_name": "tool1", "arguments": {}},
            {"tool_name": "tool2", "arguments": {}}
        ]
        
        results = await batch_call_tools(
            mock_client, 
            tool_calls, 
            "test_endpoint",
            stop_on_error=False
        )
        
        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "Tool2 failed" in results[1]["error"]
    
    @pytest.mark.asyncio
    async def test_batch_call_tools_stop_on_error(self):
        """Test batch tool calls with stop on error."""
        mock_client = Mock()
        mock_client.call_tool = AsyncMock(side_effect=[
            Exception("Tool1 failed"),
            {"result": "tool2_result"}  # Should not be called
        ])
        
        tool_calls = [
            {"tool_name": "tool1", "arguments": {}},
            {"tool_name": "tool2", "arguments": {}}
        ]
        
        # Should raise exception when stop_on_error=True
        with pytest.raises(SAFLAError, match="Batch execution failed"):
            await batch_call_tools(
                mock_client,
                tool_calls,
                "test_endpoint",
                stop_on_error=True
            )
    
    @pytest.mark.asyncio
    async def test_batch_call_tools_with_concurrency_limit(self):
        """Test batch tool calls with concurrency limit."""
        mock_client = Mock()
        call_times = []
        
        async def mock_call_tool(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": "success"}
        
        mock_client.call_tool = mock_call_tool
        
        tool_calls = [
            {"tool_name": f"tool{i}", "arguments": {}} for i in range(5)
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await batch_call_tools(
            mock_client,
            tool_calls,
            "test_endpoint",
            max_concurrent=2
        )
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == 5
        assert all(r["success"] for r in results)
        
        # With concurrency limit of 2, should take longer than if all parallel
        assert end_time - start_time >= 0.25  # At least 3 batches * 0.1s
    


class TestProxyServer:
    """Test proxy server creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_fastmcp_proxy_server(self):
        """Test creating FastMCP proxy server."""
        target_endpoints = [
            "http://localhost:8001",
            "http://localhost:8002"
        ]
        
        with patch('safla.integrations.fastmcp_helpers.FastMCPServer') as mock_server_class:
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            
            proxy = create_fastmcp_proxy_server(
                "proxy_server",
                target_endpoints,
                load_balancing="round_robin"
            )
            
            assert proxy is not None
            mock_server_class.assert_called_once()
    
    def test_create_fastmcp_proxy_server_no_endpoints(self):
        """Test creating proxy server with no endpoints."""
        # Since FastMCP is not available, this will raise a different error
        with pytest.raises(SAFLAError):
            create_fastmcp_proxy_server("proxy_server", [])


class TestConfigurationConversion:
    """Test configuration conversion functionality."""
    
    def test_convert_mcp_to_fastmcp_config_basic(self):
        """Test basic MCP to FastMCP config conversion."""
        safla_config = {
            "fastmcp": {
                "endpoints": {
                    "test_server": {
                        "name": "test_server",
                        "url": "http://localhost:8001",
                        "transport_type": "sse"
                    }
                }
            }
        }
        
        fastmcp_config = convert_safla_config_to_fastmcp(safla_config)
        
        assert "test_server" in fastmcp_config.endpoints
        endpoint = fastmcp_config.endpoints["test_server"]
        assert endpoint["name"] == "test_server"
        assert endpoint["url"] == "http://localhost:8001"
        assert endpoint["transport_type"] == "sse"
    
    def test_convert_mcp_to_fastmcp_config_with_url(self):
        """Test MCP to FastMCP config conversion with URL."""
        safla_config = {
            "fastmcp": {
                "endpoints": {
                    "remote_server": {
                        "name": "remote_server",
                        "url": "http://localhost:8001/mcp",
                        "transport_type": "sse"
                    }
                }
            }
        }
        
        fastmcp_config = convert_safla_config_to_fastmcp(safla_config)
        
        endpoint = fastmcp_config.endpoints["remote_server"]
        assert endpoint["url"] == "http://localhost:8001/mcp"
        assert endpoint["transport_type"] == "sse"
    
    def test_convert_mcp_to_fastmcp_config_with_auth(self):
        """Test MCP to FastMCP config conversion with authentication."""
        safla_config = {
            "fastmcp": {
                "endpoints": {
                    "auth_server": {
                        "name": "auth_server",
                        "url": "http://localhost:8001/mcp",
                        "auth_type": "bearer_token",
                        "bearer_token": "test_token"
                    }
                }
            }
        }
        
        fastmcp_config = convert_safla_config_to_fastmcp(safla_config)
        
        endpoint = fastmcp_config.endpoints["auth_server"]
        assert endpoint["auth_type"] == "bearer_token"
        assert endpoint["auth_token"] == "test_token"  # bearer_token converted to auth_token
    
    def test_convert_mcp_to_fastmcp_config_empty(self):
        """Test converting empty SAFLA config."""
        safla_config = {"fastmcp": {"endpoints": {}}}
        
        fastmcp_config = convert_safla_config_to_fastmcp(safla_config)
        
        assert len(fastmcp_config.endpoints) == 0

class TestDecorators:
    """Test decorator functionality."""
    
    def test_fastmcp_tool_decorator(self):
        """Test FastMCP tool decorator."""
        @fastmcp_tool("test_tool", "Test tool description")
        def test_function(arg1: str, arg2: int = 10) -> str:
            """Test function."""
            return f"{arg1}_{arg2}"
        
        # Check that function is decorated with metadata
        assert hasattr(test_function, '_fastmcp_tool')
        assert test_function._fastmcp_tool['name'] == "test_tool"
        assert test_function._fastmcp_tool['description'] == "Test tool description"
        
        # Function should still work normally
        result = test_function("hello", 5)
        assert result == "hello_5"
    
    def test_fastmcp_resource_decorator(self):
        """Test FastMCP resource decorator."""
        @fastmcp_resource("test://.*", "Test Resource", "Test resource description")
        def test_resource_handler(uri: str) -> Dict[str, Any]:
            """Test resource handler."""
            return {"uri": uri, "content": "test data"}
        
        # Check that function is decorated with metadata
        assert hasattr(test_resource_handler, '_fastmcp_resource')
        assert test_resource_handler._fastmcp_resource['uri'] == "test://.*"
        assert test_resource_handler._fastmcp_resource['name'] == "Test Resource"
        assert test_resource_handler._fastmcp_resource['description'] == "Test resource description"
        
        # Function should still work normally
        result = test_resource_handler("test://example")
        assert result["uri"] == "test://example"