"""
Tests for FastMCP Server integration - Final Working Version

This module contains comprehensive tests for the FastMCP server functionality,
using proper mocking techniques to bypass availability checks.
"""

import pytest
import asyncio
import sys
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, Optional

from safla.integrations.fastmcp_adapter import FastMCPConfig
from safla.exceptions import SAFLAError


class TestFastMCPServer:
    """Test FastMCP server functionality."""

    @pytest.fixture
    def mock_fastmcp_server(self):
        """Create a mock FastMCP server."""
        mock = Mock()
        mock.tool = Mock(return_value=lambda func: func)
        mock.resource = Mock(return_value=lambda func: func)
        mock.start = AsyncMock()
        mock.stop = AsyncMock()
        mock.run = AsyncMock()
        return mock

    @pytest.fixture
    def mock_fastmcp_class(self):
        """Create a mock FastMCP class."""
        mock_class = Mock()
        mock_instance = Mock()
        mock_instance.tool = Mock(return_value=lambda func: func)
        mock_instance.resource = Mock(return_value=lambda func: func)
        mock_instance.start = AsyncMock()
        mock_instance.stop = AsyncMock()
        mock_instance.run = AsyncMock()
        mock_class.return_value = mock_instance
        return mock_class

    def test_server_creation_without_fastmcp(self):
        """Test server creation when FastMCP is not available."""
        # Import here to ensure we get the actual module state
        from safla.integrations.fastmcp_server import FastMCPServer
        
        with pytest.raises(SAFLAError):
            server = FastMCPServer("test_server")

    def test_tool_definition_creation(self):
        """Test tool definition creation."""
        from safla.integrations.fastmcp_server import ToolDefinition
        
        def test_tool(arg: str) -> str:
            return arg

        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            handler=test_tool,
            parameters={"arg": {"type": "str", "required": True, "name": "arg"}},
            return_type=str,
            tags=[]
        )
        
        assert tool_def.name == "test_tool"
        assert tool_def.description == "Test tool"
        assert tool_def.handler == test_tool

    def test_resource_definition_creation(self):
        """Test resource definition creation."""
        from safla.integrations.fastmcp_server import ResourceDefinition
        
        async def test_handler() -> Dict[str, Any]:
            return {"data": "test"}

        resource_def = ResourceDefinition(
            uri="test://resource",
            name="Test Resource",
            description="Test resource",
            handler=test_handler,
            mime_type=None,
            tags=[]
        )
        
        assert resource_def.uri == "test://resource"
        assert resource_def.name == "Test Resource"
        assert resource_def.description == "Test resource"
        assert resource_def.handler == test_handler

    def test_register_tool_without_fastmcp(self):
        """Test tool registration when FastMCP is not available."""
        from safla.integrations.fastmcp_server import FastMCPServer
        
        with pytest.raises(SAFLAError):
            server = FastMCPServer("test_server")

    def test_server_creation_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test server creation with mocked FastMCP availability."""
        # Mock the availability check and FastMCP import
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                # Import after patching
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                assert server.name == "test_server"
                assert server.server == mock_fastmcp_class.return_value

    def test_register_tool_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test tool registration with mocked FastMCP."""
        def test_tool(arg: str) -> str:
            return arg

        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                server.register_tool("test_tool", "Test tool", test_tool)
                
                assert "test_tool" in server.tools
                assert server.tools["test_tool"].name == "test_tool"
                assert server.tools["test_tool"].handler == test_tool

    def test_register_resource_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test resource registration with mocked FastMCP."""
        async def test_handler() -> Dict[str, Any]:
            return {"data": "test"}

        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                server.register_resource("test://resource", "Test Resource", "Test resource", test_handler)
                
                assert "test://resource" in server.resources
                assert server.resources["test://resource"].uri == "test://resource"
                assert server.resources["test://resource"].handler == test_handler

    @pytest.mark.asyncio
    async def test_server_start_stop_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test server start/stop with mocked FastMCP."""
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                
                # Test start
                await server.start()
                assert server._running is True
                
                # Test stop
                await server.stop()
                assert server._running is False

    def test_list_tools_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test listing tools with mocked FastMCP."""
        def test_tool(arg: str) -> str:
            return arg

        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                server.register_tool("test_tool", "Test tool", test_tool)
                
                tools = server.list_tools()
                # tools is a list of dictionaries, find the test_tool
                tool_names = [tool['name'] for tool in tools]
                assert "test_tool" in tool_names
                
                test_tool_info = next(tool for tool in tools if tool['name'] == "test_tool")
                assert test_tool_info["name"] == "test_tool"
                assert test_tool_info["description"] == "Test tool"

    def test_list_resources_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test listing resources with mocked FastMCP."""
        async def test_handler() -> Dict[str, Any]:
            return {"data": "test"}

        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                server.register_resource("test://resource", "Test Resource", "Test resource", test_handler)
                
                resources = server.list_resources()
                # resources is a list of dictionaries, find the test resource
                resource_uris = [resource['uri'] for resource in resources]
                assert "test://resource" in resource_uris
                
                test_resource_info = next(resource for resource in resources if resource['uri'] == "test://resource")
                assert test_resource_info["name"] == "Test Resource"
                assert test_resource_info["description"] == "Test resource"

    def test_get_tool_info_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test getting tool info with mocked FastMCP."""
        def test_tool(arg: str) -> str:
            return arg

        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                server.register_tool("test_tool", "Test tool", test_tool)
                
                info = server.get_tool_info("test_tool")
                assert info["name"] == "test_tool"
                assert info["description"] == "Test tool"

    def test_get_resource_info_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test getting resource info with mocked FastMCP."""
        async def test_handler() -> Dict[str, Any]:
            return {"data": "test"}

        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                server.register_resource("test://resource", "Test Resource", "Test resource", test_handler)
                
                info = server.get_resource_info("test://resource")
                assert info["name"] == "Test Resource"
                assert info["description"] == "Test resource"

    def test_get_nonexistent_tool_info_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test getting info for nonexistent tool."""
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                
                # Should return None for nonexistent tool
                info = server.get_tool_info("nonexistent")
                assert info is None

    def test_get_nonexistent_resource_info_with_mocked_fastmcp(self, mock_fastmcp_class):
        """Test getting info for nonexistent resource."""
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = FastMCPServer("test_server")
                
                # Should return None for nonexistent resource
                info = server.get_resource_info("nonexistent")
                assert info is None


class TestFastMCPServerBuilder:
    """Test FastMCP server builder functionality."""

    @pytest.fixture
    def builder(self):
        """Create a FastMCP server builder."""
        from safla.integrations.fastmcp_server import FastMCPServerBuilder
        return FastMCPServerBuilder("test_builder")

    @pytest.fixture
    def mock_fastmcp_class(self):
        """Create a mock FastMCP class."""
        mock_class = Mock()
        mock_instance = Mock()
        mock_instance.tool = Mock(return_value=lambda func: func)
        mock_instance.resource = Mock(return_value=lambda func: func)
        mock_instance.start = AsyncMock()
        mock_instance.stop = AsyncMock()
        mock_instance.run = AsyncMock()
        mock_class.return_value = mock_instance
        return mock_class

    def test_builder_creation(self, builder):
        """Test builder creation."""
        assert builder.name == "test_builder"
        assert builder.description is None
        assert builder.version == "1.0.0"
        assert len(builder._tools) == 0
        assert len(builder._resources) == 0

    def test_add_tool(self, builder):
        """Test adding tool to builder."""
        def test_tool(arg: str) -> str:
            return arg

        result = builder.add_tool("test_tool", "Test tool", test_tool)
        
        assert result == builder  # Should return self for chaining
        assert len(builder._tools) == 1
        assert builder._tools[0]["name"] == "test_tool"
        assert builder._tools[0]["description"] == "Test tool"
        assert builder._tools[0]["handler"] == test_tool

    def test_add_resource(self, builder):
        """Test adding resource to builder."""
        async def test_handler() -> Dict[str, Any]:
            return {"data": "test"}

        result = builder.add_resource("test://resource", "Test Resource", "Test resource", test_handler)
        
        assert result == builder  # Should return self for chaining
        assert len(builder._resources) == 1
        assert builder._resources[0]["uri"] == "test://resource"
        assert builder._resources[0]["name"] == "Test Resource"
        assert builder._resources[0]["description"] == "Test resource"
        assert builder._resources[0]["handler"] == test_handler

    def test_build_server_with_mocked_fastmcp(self, builder, mock_fastmcp_class):
        """Test building server from builder with mocked FastMCP."""
        def test_tool(arg: str) -> str:
            return arg

        async def test_handler() -> Dict[str, Any]:
            return {"data": "test"}

        builder.add_tool("test_tool", "Test tool", test_tool)
        builder.add_resource("test://resource", "Test Resource", "Test resource", test_handler)

        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = builder.build()
                
                assert isinstance(server, FastMCPServer)
                assert server.name == "test_builder"
                assert "test_tool" in server.tools
                assert "test://resource" in server.resources

    def test_build_server_with_no_tools_or_resources_mocked_fastmcp(self, builder, mock_fastmcp_class):
        """Test building server with no tools or resources with mocked FastMCP."""
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('safla.integrations.fastmcp_server.FastMCP', mock_fastmcp_class):
                from safla.integrations.fastmcp_server import FastMCPServer
                
                server = builder.build()
                
                assert isinstance(server, FastMCPServer)
                assert server.name == "test_builder"
                assert len(server.tools) == 1  # Only health_check tool
                assert len(server.resources) == 0

    def test_builder_chaining(self, builder):
        """Test builder method chaining."""
        def test_tool(arg: str) -> str:
            return arg

        async def test_handler() -> Dict[str, Any]:
            return {"data": "test"}

        result = (builder
                 .add_tool("test_tool", "Test tool", test_tool)
                 .add_resource("test://resource", "Test Resource", "Test resource", test_handler))
        
        assert result == builder
        assert len(builder._tools) == 1
        assert len(builder._resources) == 1

    def test_builder_without_fastmcp(self, builder):
        """Test builder when FastMCP is not available."""
        def test_tool(arg: str) -> str:
            return arg

        builder.add_tool("test_tool", "Test tool", test_tool)
        
        with pytest.raises(SAFLAError):
            builder.build()