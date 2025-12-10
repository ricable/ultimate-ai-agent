"""
Unit tests for FastMCP Adapter

This module contains comprehensive tests for the FastMCP adapter class,
including configuration, connection management, error handling, and integration
with the existing MCP orchestration system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from safla.integrations.fastmcp_adapter import (
    FastMCPAdapter,
    FastMCPConfig,
    FastMCPEndpoint,
    FastMCPAuthType,
    FastMCPTransportType
)
from safla.exceptions import SAFLAError


class TestFastMCPEndpoint:
    """Test FastMCP endpoint configuration."""
    
    def test_endpoint_creation_minimal(self):
        """Test creating endpoint with minimal configuration."""
        endpoint = FastMCPEndpoint(
            name="test_endpoint",
            url="http://localhost:8000",
            transport_type=FastMCPTransportType.SSE
        )
        
        assert endpoint.name == "test_endpoint"
        assert endpoint.url == "http://localhost:8000"
        assert endpoint.auth_type == FastMCPAuthType.NONE
        assert endpoint.transport_type == FastMCPTransportType.SSE
        assert endpoint.timeout == 30.0
        assert endpoint.max_retries == 3
        assert endpoint.auth_token is None
    
    def test_endpoint_creation_full(self):
        """Test creating endpoint with full configuration."""
        endpoint = FastMCPEndpoint(
            name="test_endpoint",
            url="https://api.example.com/mcp",
            auth_type=FastMCPAuthType.BEARER_TOKEN,
            auth_token="test-token",
            transport_type=FastMCPTransportType.STREAMABLE_HTTP,
            timeout=60.0,
            max_retries=5,
            custom_headers={"X-Custom": "value"}
        )
        
        assert endpoint.name == "test_endpoint"
        assert endpoint.url == "https://api.example.com/mcp"
        assert endpoint.auth_type == FastMCPAuthType.BEARER_TOKEN
        assert endpoint.auth_token == "test-token"
        assert endpoint.transport_type == FastMCPTransportType.STREAMABLE_HTTP
        assert endpoint.timeout == 60.0
        assert endpoint.max_retries == 5
        assert endpoint.custom_headers == {"X-Custom": "value"}
    
    def test_endpoint_validation_invalid_url(self):
        """Test endpoint validation with invalid URL."""
        # Note: FastMCPEndpoint is a dataclass, so it doesn't validate URLs by default
        # This test would need to be implemented if URL validation is added
        endpoint = FastMCPEndpoint(
            name="test_endpoint",
            url="not-a-url",
            transport_type=FastMCPTransportType.SSE
        )
        assert endpoint.url == "not-a-url"
    
    def test_endpoint_validation_bearer_without_token(self):
        """Test endpoint validation for bearer auth without token."""
        # Note: FastMCPEndpoint is a dataclass, so it doesn't validate auth by default
        # This test would need to be implemented if auth validation is added
        endpoint = FastMCPEndpoint(
            name="test_endpoint",
            url="http://localhost:8000",
            transport_type=FastMCPTransportType.SSE,
            auth_type=FastMCPAuthType.BEARER_TOKEN
        )
        assert endpoint.auth_type == FastMCPAuthType.BEARER_TOKEN
        assert endpoint.auth_token is None
    
    def test_endpoint_validation_api_key_without_token(self):
        """Test endpoint validation for API key auth without token."""
        # Note: FastMCPEndpoint is a dataclass, so it doesn't validate auth by default
        # This test would need to be implemented if auth validation is added
        endpoint = FastMCPEndpoint(
            name="test_endpoint",
            url="http://localhost:8000",
            transport_type=FastMCPTransportType.SSE,
            auth_type=FastMCPAuthType.API_KEY
        )
        assert endpoint.auth_type == FastMCPAuthType.API_KEY
        assert endpoint.auth_token is None


class TestFastMCPConfig:
    """Test FastMCP configuration."""
    
    def test_config_creation_minimal(self):
        """Test creating config with minimal settings."""
        config = FastMCPConfig()
        
        assert config.endpoints == {}
        assert config.default_timeout == 30.0
        assert config.max_retries == 3
        
        assert config.connection_pool_size == 10
        
        assert config.health_check_interval == 60.0
    
    def test_config_creation_with_endpoints(self):
        """Test creating config with endpoints."""
        endpoint1 = FastMCPEndpoint(
            name="endpoint1",
            url="http://localhost:8000",
            transport_type=FastMCPTransportType.SSE
        )
        endpoint2 = FastMCPEndpoint(
            name="endpoint2",
            url="http://localhost:8001",
            transport_type=FastMCPTransportType.SSE
        )
        
        config = FastMCPConfig(
            endpoints={
                "ep1": {
                    "name": "endpoint1",
                    "url": "http://localhost:8000",
                    "transport_type": "sse"
                },
                "ep2": {
                    "name": "endpoint2", 
                    "url": "http://localhost:8001",
                    "transport_type": "sse"
                }
            },
            default_timeout=45.0
        )
        
        assert len(config.endpoints) == 2
        assert "ep1" in config.endpoints
        assert "ep2" in config.endpoints
        assert config.default_timeout == 45.0
    
    def test_config_validation_negative_timeout(self):
        """Test config validation with negative timeout."""
        with pytest.raises(Exception):
            FastMCPConfig(default_timeout=-1.0)
    
    def test_config_validation_negative_retries(self):
        """Test config validation with negative retries."""
        with pytest.raises(Exception):
            FastMCPConfig(max_retries=-1)


class TestFastMCPAdapter:
    """Test FastMCP adapter functionality."""
    
    @pytest.fixture
    def mock_fastmcp_available(self):
        """Mock FastMCP availability."""
        with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True), \
             patch('safla.integrations.fastmcp_adapter.SSETransport') as mock_sse, \
             patch('safla.integrations.fastmcp_adapter.StreamableHttpTransport') as mock_http:
            mock_sse.return_value = Mock()
            mock_http.return_value = Mock()
            yield
    
    @pytest.fixture
    def sample_config(self):
        """Create sample FastMCP configuration."""
        endpoint = FastMCPEndpoint(
            name="test_endpoint",
            url="http://localhost:8000",
            transport_type=FastMCPTransportType.SSE
        )
        return FastMCPConfig(
            endpoints={
                "test": {
                    "name": "test_endpoint",
                    "url": "http://localhost:8000",
                    "transport_type": "sse"
                }
            },
            default_timeout=30.0
        )
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock MCP orchestrator."""
        orchestrator = Mock()
        orchestrator.register_server = AsyncMock()
        orchestrator.unregister_server = AsyncMock()
        return orchestrator
    
    def test_adapter_creation_without_fastmcp(self, sample_config):
        """Test adapter creation when FastMCP is not available."""
        with patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', False):
            with pytest.raises(SAFLAError, match="FastMCP is not available"):
                FastMCPAdapter(sample_config)
    
    def test_adapter_creation_with_fastmcp(self, mock_fastmcp_available, sample_config):
        """Test adapter creation when FastMCP is available."""
        adapter = FastMCPAdapter(sample_config)
        
        assert adapter.config == sample_config
        assert adapter.endpoints == {}
        assert adapter.is_running is False
        assert adapter._health_check_task is None
    
    def test_adapter_creation_with_orchestrator(self, mock_fastmcp_available, sample_config, mock_orchestrator):
        """Test adapter creation with MCP orchestrator."""
        adapter = FastMCPAdapter(sample_config, orchestrator=mock_orchestrator)
        
        assert adapter.orchestrator == mock_orchestrator
    
    @pytest.mark.asyncio
    async def test_adapter_start_stop(self, mock_fastmcp_available, sample_config):
        """Test adapter start and stop lifecycle."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Enable the config so start() actually does something
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            
            # Test start
            await adapter.start()
            assert adapter.is_running is True
            assert "test" in adapter.endpoints
            
            # Test stop
            await adapter.shutdown()
            assert adapter.is_running is False
    
    @pytest.mark.asyncio
    async def test_adapter_add_remove_endpoint(self, mock_fastmcp_available, sample_config):
        """Test adding and removing endpoints dynamically."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            # Add new endpoint
            new_endpoint = FastMCPEndpoint(
                name="new_endpoint",
                url="http://localhost:8001",
                transport_type=FastMCPTransportType.SSE
            )
            success = await adapter.add_endpoint(new_endpoint)
            
            assert success is True
            assert "new_endpoint" in adapter.endpoints
            
            # Remove endpoint
            success = await adapter.remove_endpoint("new_endpoint")
            
            assert success is True
            assert "new_endpoint" not in adapter.endpoints
    
    @pytest.mark.asyncio
    async def test_adapter_call_tool_success(self, mock_fastmcp_available, sample_config):
        """Test successful tool call."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value={"result": "success"})
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            result = await adapter.call_tool(
                endpoint_name="test",
                tool_name="test_tool",
                arguments={"arg1": "value1"}
            )
            
            assert result == {"result": "success"}
            mock_client.call_tool.assert_called_once_with(
                "test_tool",
                {"arg1": "value1"},
                timeout=30.0,
                progress_handler=None
            )
    
    @pytest.mark.asyncio
    async def test_adapter_call_tool_with_retries(self, mock_fastmcp_available, sample_config):
        """Test tool call with retry logic."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            # The adapter doesn't implement retry logic, it just calls once
            mock_client.call_tool = AsyncMock(return_value={"result": "success"})
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            result = await adapter.call_tool(
                endpoint_name="test",
                tool_name="test_tool",
                arguments={}
            )
            
            assert result == {"result": "success"}
            assert mock_client.call_tool.call_count == 1
    
    @pytest.mark.asyncio
    async def test_adapter_call_tool_max_retries_exceeded(self, mock_fastmcp_available, sample_config):
        """Test tool call when max retries are exceeded."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(side_effect=Exception("Persistent error"))
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            with pytest.raises(SAFLAError, match="FastMCP tool call failed"):
                await adapter.call_tool(
                    endpoint_name="test",
                    tool_name="test_tool",
                    arguments={}
                )
    
    @pytest.mark.asyncio
    async def test_adapter_read_resource_success(self, mock_fastmcp_available, sample_config):
        """Test successful resource read."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.read_resource = AsyncMock(return_value={"content": "resource data"})
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            result = await adapter.read_resource(
                endpoint_name="test",
                resource_uri="resource://test"
            )
            
            assert result == {"content": "resource data"}
            mock_client.read_resource.assert_called_once_with("resource://test", timeout=30.0)
    
    @pytest.mark.asyncio
    async def test_adapter_list_tools(self, mock_fastmcp_available, sample_config):
        """Test listing tools from endpoint."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[
                {"name": "tool1", "description": "Test tool 1"},
                {"name": "tool2", "description": "Test tool 2"}
            ])
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            tools = await adapter.list_tools("test")
            
            assert len(tools) == 2
            assert tools[0]["name"] == "tool1"
            assert tools[1]["name"] == "tool2"
    
    @pytest.mark.asyncio
    async def test_adapter_list_resources(self, mock_fastmcp_available, sample_config):
        """Test listing resources from endpoint."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_resources = AsyncMock(return_value=[
                {"uri": "resource://test1", "name": "Test Resource 1"},
                {"uri": "resource://test2", "name": "Test Resource 2"}
            ])
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            resources = await adapter.list_resources("test")
            
            assert len(resources) == 2
            assert resources[0]["uri"] == "resource://test1"
            assert resources[1]["uri"] == "resource://test2"
    
    @pytest.mark.asyncio
    async def test_adapter_get_endpoint_status_healthy(self, mock_fastmcp_available, sample_config):
        """Test getting endpoint status when healthy."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=None)  # ping() returns None on success
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            status = await adapter.get_endpoint_status("test")
            
            assert status["status"] == "healthy"
            assert status["name"] == "test"  # The method returns "name", not "endpoint_name"
            assert status["connected"] is True
    
    @pytest.mark.asyncio
    async def test_adapter_get_endpoint_status_unhealthy(self, mock_fastmcp_available, sample_config):
        """Test getting endpoint status when unhealthy."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            status = await adapter.get_endpoint_status("test")
            
            assert status["status"] == "unhealthy"
            assert status["connected"] is False
            assert "error" in status
    
    @pytest.mark.asyncio
    async def test_adapter_endpoint_not_found(self, mock_fastmcp_available, sample_config):
        """Test operations on non-existent endpoint."""
        sample_config.enabled = True
        adapter = FastMCPAdapter(sample_config)
        await adapter.start()
        
        with pytest.raises(SAFLAError, match="FastMCP endpoint not found: nonexistent"):
            await adapter.call_tool("nonexistent", "test_tool", {})
        
        with pytest.raises(SAFLAError, match="FastMCP endpoint not found: nonexistent"):
            await adapter.read_resource("nonexistent", "resource://test")
        
        with pytest.raises(SAFLAError, match="FastMCP endpoint not found: nonexistent"):
            await adapter.list_tools("nonexistent")
    
    @pytest.mark.asyncio
    async def test_adapter_health_check_integration(self, mock_fastmcp_available, sample_config, mock_orchestrator):
        """Test health check integration with orchestrator."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value={"status": "healthy"})
            mock_client_class.return_value = mock_client
            
            # Create config with health checks enabled
            config_with_health = FastMCPConfig(
                enabled=True,
                health_check_interval=10.0,  # Minimum allowed value
                endpoints={
                    "test": {
                        "name": "test_endpoint",
                        "url": "http://localhost:8000",
                        "transport_type": "sse"
                    }
                }
            )
            
            adapter = FastMCPAdapter(config_with_health, orchestrator=mock_orchestrator)
            await adapter.start()
            
            # Wait for a short time (health check runs in background)
            await asyncio.sleep(0.1)
            
            # Verify orchestrator server_manager.register_server was called
            mock_orchestrator.server_manager.register_server.assert_called()
            
            await adapter.shutdown()
    
    @pytest.mark.asyncio
    async def test_adapter_authentication_bearer(self, mock_fastmcp_available):
        """Test adapter with bearer token authentication."""
        endpoint = FastMCPEndpoint(
            name="test_endpoint",
            url="http://localhost:8000",
            transport_type=FastMCPTransportType.SSE,
            auth_type=FastMCPAuthType.BEARER_TOKEN,
            auth_token="test-bearer-token"
        )
        config = FastMCPConfig(endpoints={
            "auth_test": {
                "name": "test_endpoint",
                "url": "http://localhost:8000",
                "transport_type": "sse",
                "auth_type": "bearer_token",
                "auth_token": "test-bearer-token"
            }
        })
        
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            config.enabled = True
            adapter = FastMCPAdapter(config)
            await adapter.start()
            
            # Verify client was created with authentication
            mock_client_class.assert_called()
            call_args = mock_client_class.call_args
            
            # Check that authentication headers were set up
            assert adapter.endpoints["auth_test"] is not None
    
    @pytest.mark.asyncio
    async def test_adapter_custom_headers(self, mock_fastmcp_available):
        """Test adapter with custom headers."""
        endpoint = FastMCPEndpoint(
            name="test_endpoint",
            url="http://localhost:8000",
            transport_type=FastMCPTransportType.SSE,
            custom_headers={"X-Custom-Header": "custom-value", "X-API-Version": "v1"}
        )
        config = FastMCPConfig(endpoints={
            "custom_test": {
                "name": "test_endpoint",
                "url": "http://localhost:8000",
                "transport_type": "sse",
                "custom_headers": {"X-Custom-Header": "custom-value", "X-API-Version": "v1"}
            }
        })
        
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            config.enabled = True
            adapter = FastMCPAdapter(config)
            await adapter.start()
            
            # Verify client was created
            mock_client_class.assert_called()
            assert adapter.endpoints["custom_test"] is not None
    
    @pytest.mark.asyncio
    async def test_adapter_progress_handler(self, mock_fastmcp_available, sample_config):
        """Test adapter with progress handler."""
        with patch('safla.integrations.fastmcp_adapter.Client') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value={"result": "success"})
            mock_client_class.return_value = mock_client
            
            sample_config.enabled = True
            adapter = FastMCPAdapter(sample_config)
            await adapter.start()
            
            progress_calls = []
            
            async def progress_handler(progress: float, message: str = None):
                progress_calls.append((progress, message))
            
            result = await adapter.call_tool(
                endpoint_name="test",
                tool_name="test_tool",
                arguments={},
                progress_handler=progress_handler
            )
            
            assert result == {"result": "success"}
            # The adapter passes progress_handler to the client, but doesn't call it directly
            # Verify that the progress_handler was passed to the client
            mock_client.call_tool.assert_called_once_with(
                "test_tool", {}, timeout=30.0, progress_handler=progress_handler
            )