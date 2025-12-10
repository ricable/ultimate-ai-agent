"""
Unit tests for FastMCP Server

This module contains comprehensive tests for the FastMCP server wrapper,
including tool/resource registration, decorators, and builder pattern.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional

from safla.integrations.fastmcp_server import (
    FastMCPServer, FastMCPServerBuilder, ToolDefinition, ResourceDefinition
)
from safla.exceptions import SAFLAError


class TestToolDefinition:
    """Test tool definition functionality."""
    
    def test_tool_definition_creation(self):
        """Test creating tool definition."""
        def test_func(arg1: str, arg2: int = 10) -> str:
            """Test function."""
            return f"{arg1}_{arg2}"
        
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            handler=test_func  # Fixed: use 'handler' not 'function'
        )
        
        assert tool_def.name == "test_tool"
        assert tool_def.description == "Test tool"
        assert tool_def.handler == test_func  # Fixed: use 'handler'
        assert tool_def.parameters is None
    
    def test_tool_definition_with_parameters(self):
        """Test creating tool definition with custom parameters."""
        def test_func(arg1: str) -> str:
            return arg1
        
        parameters = {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "Test argument"}
            },
            "required": ["arg1"]
        }
        
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            handler=test_func,  # Fixed: use 'handler'
            parameters=parameters
        )
        
        assert tool_def.parameters == parameters


class TestResourceDefinition:
    """Test resource definition functionality."""
    
    def test_resource_definition_creation(self):
        """Test creating resource definition."""
        async def test_handler(uri: str) -> Dict[str, Any]:
            """Test resource handler."""
            return {"uri": uri, "data": "test_data"}
        
        resource_def = ResourceDefinition(
            uri="test://resource",
            name="Test Resource",
            description="Test resource",
            handler=test_handler
        )
        
        assert resource_def.uri == "test://resource"
        assert resource_def.name == "Test Resource"
        assert resource_def.description == "Test resource"
        assert resource_def.handler == test_handler
        assert resource_def.mime_type is None
    
    def test_resource_definition_with_mime_type(self):
        """Test creating resource definition with MIME type."""
        async def test_handler(uri: str) -> Dict[str, Any]:
            return {"data": "test"}
        
        resource_def = ResourceDefinition(
            uri="test://resource",
            name="Test Resource",
            description="Test resource",
            handler=test_handler,
            mime_type="application/json"
        )
        
        assert resource_def.mime_type == "application/json"


class TestFastMCPServer:
    """Test FastMCP server wrapper."""
    
    @pytest.fixture
    def mock_fastmcp_server(self):
        """Create a mock FastMCP server."""
        server = Mock()
        server.tool = Mock()
        server.resource = Mock()
        server.run = AsyncMock()
        server.stop = AsyncMock()
        return server
    
    @pytest.fixture
    def server(self, mock_fastmcp_server):
        """Create FastMCP server wrapper."""
        # Fixed: use correct patch target
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('fastmcp.FastMCP', return_value=mock_fastmcp_server):
                return FastMCPServer("test_server")
    
    def test_server_creation_without_fastmcp(self):
        """Test server creation when FastMCP is not available."""
        # Fixed: use correct patch target
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', False):
            with pytest.raises(SAFLAError, match="FastMCP is not available"):
                FastMCPServer("test_server")
    
    def test_server_creation_with_fastmcp(self, mock_fastmcp_server):
        """Test server creation when FastMCP is available."""
        # Fixed: use correct patch target
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            with patch('fastmcp.FastMCP', return_value=mock_fastmcp_server):
                server = FastMCPServer("test_server")
                assert server.name == "test_server"
                assert server.server == mock_fastmcp_server  # Fixed: use 'server' not '_server'
                assert server.tools == {}  # Fixed: use 'tools' not '_tools'
                assert server.resources == {}  # Fixed: use 'resources' not '_resources'
    
    def test_register_tool_function(self, server):
        """Test registering a tool function."""
        def test_tool(arg1: str, arg2: int = 10) -> str:
            """Test tool function."""
            return f"{arg1}_{arg2}"
        
        server.register_tool("test_tool", "Test tool", test_tool)
        
        assert "test_tool" in server.tools  # Fixed: use 'tools' not '_tools'
        tool_def = server.tools["test_tool"]  # Fixed: use 'tools'
        assert tool_def.name == "test_tool"
        assert tool_def.description == "Test tool"
        assert tool_def.handler == test_tool  # Fixed: use 'handler' not 'function'
    
    def test_register_tool_with_parameters(self, server):
        """Test registering a tool with custom parameters."""
        def test_tool(arg1: str) -> str:
            return arg1
        
        parameters = {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "Test argument"}
            },
            "required": ["arg1"]
        }
        
        server.register_tool("test_tool", "Test tool", test_tool, parameters)
        
        tool_def = server.tools["test_tool"]  # Fixed: use 'tools'
        assert tool_def.parameters == parameters
    
    def test_register_resource_handler(self, server):
        """Test registering a resource handler."""
        async def test_handler(uri: str) -> Dict[str, Any]:
            return {"uri": uri, "data": "test_data"}
        
        server.register_resource("test://resource", "Test Resource", "Test resource", test_handler)
        
        assert "test://resource" in server.resources  # Fixed: use 'resources'
        resource_def = server.resources["test://resource"]  # Fixed: use 'resources'
        assert resource_def.uri == "test://resource"
        assert resource_def.name == "Test Resource"
        assert resource_def.description == "Test resource"
        assert resource_def.handler == test_handler
    
    def test_register_resource_with_mime_type(self, server):
        """Test registering a resource with MIME type."""
        async def test_handler(uri: str) -> Dict[str, Any]:
            return {"data": "test"}
        
        server.register_resource(
            "test://resource", 
            "Test Resource", 
            "Test resource", 
            test_handler,
            mime_type="application/json"
        )
        
        resource_def = server.resources["test://resource"]  # Fixed: use 'resources'
        assert resource_def.mime_type == "application/json"
    
    def test_tool_decorator(self, server):
        """Test tool decorator functionality."""
        @server.tool("decorated_tool", "Decorated tool")
        def decorated_tool(message: str) -> str:
            """A decorated tool."""
            return f"Decorated: {message}"
        
        assert "decorated_tool" in server.tools  # Fixed: use 'tools'
        tool_def = server.tools["decorated_tool"]  # Fixed: use 'tools'
        assert tool_def.name == "decorated_tool"
        assert tool_def.description == "Decorated tool"
        assert tool_def.handler == decorated_tool  # Fixed: use 'handler'
    
    def test_resource_decorator(self, server):
        """Test resource decorator functionality."""
        @server.resource("test://decorated", "Decorated Resource", "Decorated resource")
        async def decorated_resource(uri: str) -> Dict[str, Any]:
            """A decorated resource."""
            return {"uri": uri, "decorated": True}
        
        assert "test://decorated" in server.resources  # Fixed: use 'resources'
        resource_def = server.resources["test://decorated"]  # Fixed: use 'resources'
        assert resource_def.uri == "test://decorated"
        assert resource_def.name == "Decorated Resource"
        assert resource_def.handler == decorated_resource
    
    @pytest.mark.asyncio
    async def test_start_server(self, server, mock_fastmcp_server):
        """Test starting the server."""
        # Fixed: use 'start' method instead of 'run'
        await server.start(host="localhost", port=8000)
        
        # Verify that tools and resources were registered
        assert mock_fastmcp_server.tool.called
        assert mock_fastmcp_server.resource.called
        
        # Verify server was started
        mock_fastmcp_server.run.assert_called_once_with(host="localhost", port=8000)
    
    @pytest.mark.asyncio
    async def test_stop_server(self, server, mock_fastmcp_server):
        """Test stopping the server."""
        # Fixed: use 'stop' method instead of 'shutdown'
        await server.stop()
        mock_fastmcp_server.stop.assert_called_once()
    
    def test_list_tools(self, server):
        """Test listing registered tools."""
        def tool1(arg: str) -> str:
            return arg
        
        def tool2(arg: int) -> int:
            return arg
        
        server.register_tool("tool1", "Tool 1", tool1)
        server.register_tool("tool2", "Tool 2", tool2)
        
        tools = server.list_tools()
        assert len(tools) == 3  # Including default health_check tool
        tool_names = [tool["name"] for tool in tools]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
        assert "health_check" in tool_names  # Default tool
    
    def test_list_resources(self, server):
        """Test listing registered resources."""
        async def handler1(uri: str) -> Dict[str, Any]:
            return {"data": "1"}
        
        async def handler2(uri: str) -> Dict[str, Any]:
            return {"data": "2"}
        
        server.register_resource("test://resource1", "Resource 1", "Resource 1", handler1)
        server.register_resource("test://resource2", "Resource 2", "Resource 2", handler2)
        
        resources = server.list_resources()
        assert len(resources) == 2
        resource_uris = [resource["uri"] for resource in resources]
        assert "test://resource1" in resource_uris
        assert "test://resource2" in resource_uris
    
    def test_get_tool_info(self, server):
        """Test getting tool information."""
        def test_tool(arg1: str, arg2: int = 10) -> str:
            """Test tool function."""
            return f"{arg1}_{arg2}"
        
        server.register_tool("test_tool", "Test tool", test_tool)
        
        info = server.get_tool_info("test_tool")
        assert info["name"] == "test_tool"
        assert info["description"] == "Test tool"
        # Fixed: info doesn't contain 'function', it contains tool definition data
        assert "parameters" in info
    
    def test_get_resource_info(self, server):
        """Test getting resource information."""
        async def test_handler(uri: str) -> Dict[str, Any]:
            return {"data": "test"}
        
        server.register_resource("test://resource", "Test Resource", "Test resource", test_handler)
        
        info = server.get_resource_info("test://resource")
        assert info["uri"] == "test://resource"
        assert info["name"] == "Test Resource"
        assert info["description"] == "Test resource"
        # Fixed: info doesn't contain 'handler', it contains resource definition data
        assert "mime_type" in info


class TestFastMCPServerBuilder:
    """Test FastMCP server builder."""
    
    @pytest.fixture
    def builder(self):
        """Create FastMCP server builder."""
        # Fixed: use correct patch target
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            return FastMCPServerBuilder("test_server")
    
    def test_builder_creation_without_fastmcp(self):
        """Test builder creation when FastMCP is not available."""
        # Fixed: Builder doesn't check FASTMCP_AVAILABLE, only server creation does
        # So this test should actually work fine
        builder = FastMCPServerBuilder("test_server")
        assert builder.name == "test_server"
    
    def test_builder_creation_with_fastmcp(self):
        """Test builder creation when FastMCP is available."""
        # Fixed: use correct patch target
        with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
            builder = FastMCPServerBuilder("test_server")
            assert builder.name == "test_server"
            assert builder._tools == []  # Fixed: builder uses lists, not dicts
            assert builder._resources == []  # Fixed: builder uses lists, not dicts
    
    def test_add_tool_to_builder(self, builder):
        """Test adding tool to builder."""
        def test_tool(arg: str) -> str:
            return arg
        
        result = builder.add_tool("test_tool", "Test tool", test_tool)
        
        assert result == builder  # Should return self for chaining
        assert len(builder._tools) == 1
        assert builder._tools[0]["name"] == "test_tool"  # Fixed: access dict, not object
    
    def test_add_resource_to_builder(self, builder):
        """Test adding resource to builder."""
        async def test_handler(uri: str) -> Dict[str, Any]:
            return {"data": "test"}
        
        result = builder.add_resource("test://resource", "Test Resource", "Test resource", test_handler)
        
        assert result == builder  # Should return self for chaining
        assert len(builder._resources) == 1
        assert builder._resources[0]["uri"] == "test://resource"  # Fixed: access dict, not object
    
    def test_builder_chaining(self, builder):
        """Test method chaining in builder."""
        def tool1(arg: str) -> str:
            return arg
        
        def tool2(arg: int) -> int:
            return arg
        
        async def handler(uri: str) -> Dict[str, Any]:
            return {"data": "test"}
        
        result = (builder
                 .add_tool("tool1", "Tool 1", tool1)
                 .add_tool("tool2", "Tool 2", tool2)
                 .add_resource("test://resource", "Resource", "Resource", handler))
        
        assert result == builder
        assert len(builder._tools) == 2
        assert len(builder._resources) == 1
    
    def test_build_server(self, builder):
        """Test building server from builder."""
        def test_tool(arg: str) -> str:
            return arg
        
        async def test_handler(uri: str) -> Dict[str, Any]:
            return {"data": "test"}
        
        builder.add_tool("test_tool", "Test tool", test_tool)
        builder.add_resource("test://resource", "Test Resource", "Test resource", test_handler)
        
        with patch('fastmcp.FastMCP') as mock_fastmcp:
            mock_server_instance = Mock()
            mock_fastmcp.return_value = mock_server_instance
            
            # Fixed: patch FASTMCP_AVAILABLE for server creation
            with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
                server = builder.build()
                
                assert isinstance(server, FastMCPServer)
                assert server.name == "test_server"
                # Fixed: check actual tools count (includes default health_check)
                assert len(server.tools) == 2  # test_tool + health_check
                assert len(server.resources) == 1
    
    def test_build_server_with_no_tools_or_resources(self, builder):
        """Test building server with no tools or resources."""
        with patch('fastmcp.FastMCP') as mock_fastmcp:
            mock_server_instance = Mock()
            mock_fastmcp.return_value = mock_server_instance
            
            # Fixed: patch FASTMCP_AVAILABLE for server creation
            with patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True):
                server = builder.build()
                
                assert isinstance(server, FastMCPServer)
                # Fixed: check actual tools count (includes default health_check)
                assert len(server.tools) == 1  # Only health_check
                assert len(server.resources) == 0