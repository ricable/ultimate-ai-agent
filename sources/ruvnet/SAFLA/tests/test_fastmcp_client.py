"""Working tests for FastMCP client integration."""

import pytest
from unittest.mock import Mock, AsyncMock

from safla.integrations.fastmcp_client import FastMCPClient, BatchOperations
from safla.exceptions import SAFLAError


class TestFastMCPClientWorking:
    """Working test cases for FastMCPClient."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a properly configured mock FastMCP adapter."""
        adapter = Mock()
        # Mock the endpoints attribute as a dictionary
        adapter.endpoints = {"endpoint1": Mock(), "endpoint2": Mock()}
        
        # Configure async methods with proper signatures
        adapter.call_tool = AsyncMock(return_value={"result": "success"})
        adapter.read_resource = AsyncMock(return_value={"content": "test"})
        adapter.list_tools = AsyncMock(return_value=[{"name": "tool1"}])
        adapter.list_resources = AsyncMock(return_value=[{"uri": "test://resource"}])
        adapter.get_endpoint_status = AsyncMock(return_value={"status": "healthy"})
        adapter.ping_endpoint = AsyncMock(return_value=True)
        return adapter
    
    @pytest.fixture
    def client(self, mock_adapter):
        """Create a FastMCP client with mocked adapter."""
        # Skip the availability check by directly creating the client
        client = object.__new__(FastMCPClient)
        client.adapter = mock_adapter
        client._default_endpoint = None
        return client
    
    def test_set_default_endpoint_success(self, client, mock_adapter):
        """Test setting default endpoint successfully."""
        client.set_default_endpoint("endpoint1")
        assert client._default_endpoint == "endpoint1"
    
    def test_set_default_endpoint_not_found(self, client, mock_adapter):
        """Test setting default endpoint that doesn't exist."""
        with pytest.raises(SAFLAError, match="Endpoint not found: nonexistent"):
            client.set_default_endpoint("nonexistent")
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self, client, mock_adapter):
        """Test successful tool execution."""
        client._default_endpoint = "endpoint1"
        
        result = await client.call_tool("test_tool", {"param": "value"})
        
        assert result == {"result": "success"}
        mock_adapter.call_tool.assert_called_once_with(
            endpoint_name="endpoint1", 
            tool_name="test_tool", 
            arguments={"param": "value"}, 
            timeout=None,
            progress_handler=None
        )
    
    @pytest.mark.asyncio
    async def test_call_tool_with_specific_endpoint(self, client, mock_adapter):
        """Test tool execution with specific endpoint."""
        # Add the specific endpoint to the mock
        mock_adapter.endpoints["specific_endpoint"] = Mock()
        
        result = await client.call_tool(
            "test_tool", 
            {"param": "value"}, 
            endpoint="specific_endpoint"
        )
        
        assert result == {"result": "success"}
        mock_adapter.call_tool.assert_called_once_with(
            endpoint_name="specific_endpoint", 
            tool_name="test_tool", 
            arguments={"param": "value"}, 
            timeout=None,
            progress_handler=None
        )
    
    @pytest.mark.asyncio
    async def test_call_tool_no_endpoint_error(self, client, mock_adapter):
        """Test tool execution with no endpoint specified."""
        with pytest.raises(SAFLAError, match="No endpoint specified and no default endpoint set"):
            await client.call_tool("test_tool", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_read_resource_success(self, client, mock_adapter):
        """Test successful resource reading."""
        client._default_endpoint = "endpoint1"
        
        result = await client.read_resource("test://resource")
        
        assert result == {"content": "test"}
        mock_adapter.read_resource.assert_called_once_with(
            endpoint_name="endpoint1", 
            resource_uri="test://resource", 
            timeout=None
        )
    
    @pytest.mark.asyncio
    async def test_list_tools(self, client, mock_adapter):
        """Test listing available tools."""
        client._default_endpoint = "endpoint1"
        
        result = await client.list_tools()
        
        assert result == [{"name": "tool1"}]
        mock_adapter.list_tools.assert_called_once_with("endpoint1")
    
    @pytest.mark.asyncio
    async def test_list_resources(self, client, mock_adapter):
        """Test listing available resources."""
        client._default_endpoint = "endpoint1"
        
        result = await client.list_resources()
        
        assert result == [{"uri": "test://resource"}]
        mock_adapter.list_resources.assert_called_once_with("endpoint1")
    
    @pytest.mark.asyncio
    async def test_get_endpoint_info(self, client, mock_adapter):
        """Test getting endpoint information."""
        client._default_endpoint = "endpoint1"
        
        result = await client.get_endpoint_info()
        
        assert result["status"] == "healthy"
        assert result["is_default"] is True
        assert result["tool_count"] == 1
        assert result["resource_count"] == 1
    
    @pytest.mark.asyncio
    async def test_list_endpoints(self, client, mock_adapter):
        """Test listing all endpoints."""
        result = await client.list_endpoints()
        
        # Should return a list with endpoint info
        assert len(result) == 2
        # The actual implementation builds endpoint info from adapter.endpoints
    
    @pytest.mark.asyncio
    async def test_ping_endpoint_success(self, client, mock_adapter):
        """Test pinging an endpoint successfully."""
        client._default_endpoint = "endpoint1"
        
        result = await client.ping_endpoint()
        
        assert result is True
        # Note: ping_endpoint calls get_endpoint_status, not ping_endpoint directly
        mock_adapter.get_endpoint_status.assert_called_once_with("endpoint1")
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, client, mock_adapter):
        """Test executing a workflow."""
        client._default_endpoint = "endpoint1"
        
        workflow = [
            {"type": "call_tool", "tool_name": "test_tool", "arguments": {"param": "value"}},
            {"type": "read_resource", "resource_uri": "test://resource"}
        ]
        
        results = await client.execute_workflow(workflow)
        
        assert len(results) == 2
        # The workflow execution wraps results in operation metadata
        assert results[0]["success"] is True
        assert results[0]["result"] == {"result": "success"}
        assert results[1]["success"] is True
        assert results[1]["result"] == {"content": "test"}
    @pytest.mark.asyncio
    async def test_batch_context(self, client, mock_adapter):
        """Test batch operations context manager."""
        client._default_endpoint = "endpoint1"
        
        async with client.batch_context() as batch:
            batch.add_tool_call("test_tool", {"param": "value"})
            batch.add_resource_read("test://resource")
            
            # Verify operations were added
            assert len(batch.operations) == 2
            assert batch.operations[0]["type"] == "call_tool"
            assert batch.operations[1]["type"] == "read_resource"
            
            # Execute the batch operations
            results = await batch.execute()
            
            # Verify results
            assert len(results) == 2
        
        # Verify the adapter methods were called during execution
        mock_adapter.call_tool.assert_called()
        mock_adapter.read_resource.assert_called()
        mock_adapter.read_resource.assert_called()


class TestBatchOperationsWorking:
    """Working test cases for BatchOperations."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock FastMCP client."""
        return Mock(spec=FastMCPClient)
    
    def test_batch_operations_creation(self, mock_client):
        """Test creating batch operations."""
        batch = BatchOperations(mock_client, "test_endpoint")
        assert batch.client == mock_client
        assert batch.endpoint_name == "test_endpoint"
        assert batch.operations == []
    
    def test_add_tool_call(self, mock_client):
        """Test adding tool call to batch."""
        batch = BatchOperations(mock_client, "test_endpoint")
        
        result = batch.add_tool_call("test_tool", {"param": "value"})
        
        assert result == batch  # Should return self for chaining
        assert len(batch.operations) == 1
        assert batch.operations[0] == {
            "type": "call_tool",
            "tool_name": "test_tool",
            "arguments": {"param": "value"},
            "timeout": None
        }
    
    def test_add_resource_read(self, mock_client):
        """Test adding resource read to batch."""
        batch = BatchOperations(mock_client, "test_endpoint")
        
        result = batch.add_resource_read("test://resource")
        
        assert result == batch  # Should return self for chaining
        assert len(batch.operations) == 1
        # The actual implementation includes timeout parameter
        expected = {
            "type": "read_resource",
            "resource_uri": "test://resource",
            "timeout": None
        }
        assert batch.operations[0] == expected
    
    def test_method_chaining(self, mock_client):
        """Test method chaining for batch operations."""
        batch = BatchOperations(mock_client, "test_endpoint")
        
        result = (batch
                 .add_tool_call("tool1", {"param1": "value1"})
                 .add_resource_read("test://resource1")
                 .add_tool_call("tool2", {"param2": "value2"}))
        
        assert result == batch
        assert len(batch.operations) == 3
        assert batch.operations[0]["tool_name"] == "tool1"
        assert batch.operations[1]["resource_uri"] == "test://resource1"
        assert batch.operations[2]["tool_name"] == "tool2"
    
    def test_clear_batch(self, mock_client):
        """Test clearing batch operations."""
        batch = BatchOperations(mock_client, "test_endpoint")
        batch.add_tool_call("test_tool", {"param": "value"})
        
        result = batch.clear()
        
        assert result == batch  # Should return self for chaining
        assert len(batch.operations) == 0