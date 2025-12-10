"""
Test suite for MCP Orchestration Infrastructure.

This module contains comprehensive tests for the MCP orchestration system,
following TDD principles with Red-Green-Refactor methodology.

Test Coverage:
- MCP Server Management (discovery, registration, lifecycle)
- Context Sharing (vector embeddings, similarity routing)
- Agent Coordination (orchestration, conflict resolution)
- Resource Management (load balancing, allocation)
- Error Handling (recovery, failover mechanisms)
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import the modules we'll implement
from safla.core.mcp_orchestration import (
    MCPServer,
    MCPServerStatus,
    MCPServerManager,
    ContextVector,
    ContextSharingEngine,
    Agent,
    AgentStatus,
    AgentCoordinator,
    ResourcePool,
    ResourceManager,
    ErrorRecoveryStrategy,
    ErrorHandler,
    MCPOrchestrator
)


class TestMCPServerStatus:
    """Test the MCPServerStatus enum."""
    
    def test_server_status_values(self):
        """Test that all required server status values exist."""
        assert MCPServerStatus.INITIALIZING
        assert MCPServerStatus.ACTIVE
        assert MCPServerStatus.INACTIVE
        assert MCPServerStatus.ERROR
        assert MCPServerStatus.SHUTTING_DOWN


class TestMCPServer:
    """Test the MCPServer data structure."""
    
    def test_mcp_server_creation(self):
        """Test creating an MCP server instance."""
        server = MCPServer(
            server_id="test-server-1",
            name="Test Server",
            endpoint="http://localhost:8080",
            capabilities=["memory", "search"],
            status=MCPServerStatus.INITIALIZING
        )
        
        assert server.server_id == "test-server-1"
        assert server.name == "Test Server"
        assert server.endpoint == "http://localhost:8080"
        assert server.capabilities == ["memory", "search"]
        assert server.status == MCPServerStatus.INITIALIZING
        assert server.health_score == 1.0  # Default health score
        assert server.load_factor == 0.0   # Default load factor
    
    def test_mcp_server_with_custom_metrics(self):
        """Test creating an MCP server with custom health and load metrics."""
        server = MCPServer(
            server_id="test-server-2",
            name="Test Server 2",
            endpoint="http://localhost:8081",
            capabilities=["analysis"],
            status=MCPServerStatus.ACTIVE,
            health_score=0.85,
            load_factor=0.6
        )
        
        assert server.health_score == 0.85
        assert server.load_factor == 0.6
    
    def test_mcp_server_validation(self):
        """Test MCP server input validation."""
        with pytest.raises(ValueError, match="Server ID cannot be empty"):
            MCPServer(
                server_id="",
                name="Test",
                endpoint="http://localhost:8080",
                capabilities=[],
                status=MCPServerStatus.INITIALIZING
            )
        
        with pytest.raises(ValueError, match="Invalid health score"):
            MCPServer(
                server_id="test",
                name="Test",
                endpoint="http://localhost:8080",
                capabilities=[],
                status=MCPServerStatus.INITIALIZING,
                health_score=1.5  # Invalid: > 1.0
            )


class TestMCPServerManager:
    """Test the MCP Server Management functionality."""
    
    @pytest.fixture
    def server_manager(self):
        """Create a server manager instance for testing."""
        return MCPServerManager()
    
    @pytest.fixture
    def sample_server(self):
        """Create a sample MCP server for testing."""
        return MCPServer(
            server_id="test-server",
            name="Test Server",
            endpoint="http://localhost:8080",
            capabilities=["memory", "search"],
            status=MCPServerStatus.INITIALIZING
        )
    
    def test_server_manager_initialization(self, server_manager):
        """Test server manager initialization."""
        assert len(server_manager.servers) == 0
        assert server_manager.discovery_enabled is True
        assert server_manager.health_check_interval == 30.0
    
    def test_register_server(self, server_manager, sample_server):
        """Test registering a new MCP server."""
        result = server_manager.register_server(sample_server)
        
        assert result is True
        assert len(server_manager.servers) == 1
        assert server_manager.servers["test-server"] == sample_server
    
    def test_register_duplicate_server(self, server_manager, sample_server):
        """Test registering a server with duplicate ID fails."""
        server_manager.register_server(sample_server)
        
        duplicate_server = MCPServer(
            server_id="test-server",  # Same ID
            name="Duplicate Server",
            endpoint="http://localhost:8081",
            capabilities=["analysis"],
            status=MCPServerStatus.INITIALIZING
        )
        
        result = server_manager.register_server(duplicate_server)
        assert result is False
        assert len(server_manager.servers) == 1
    
    def test_unregister_server(self, server_manager, sample_server):
        """Test unregistering an MCP server."""
        server_manager.register_server(sample_server)
        
        result = server_manager.unregister_server("test-server")
        assert result is True
        assert len(server_manager.servers) == 0
    
    def test_unregister_nonexistent_server(self, server_manager):
        """Test unregistering a non-existent server."""
        result = server_manager.unregister_server("nonexistent-server")
        assert result is False
    
    def test_get_server_by_id(self, server_manager, sample_server):
        """Test retrieving a server by ID."""
        server_manager.register_server(sample_server)
        
        retrieved_server = server_manager.get_server("test-server")
        assert retrieved_server == sample_server
        
        nonexistent_server = server_manager.get_server("nonexistent")
        assert nonexistent_server is None
    
    def test_get_servers_by_capability(self, server_manager):
        """Test retrieving servers by capability."""
        server1 = MCPServer(
            server_id="server1",
            name="Memory Server",
            endpoint="http://localhost:8080",
            capabilities=["memory", "search"],
            status=MCPServerStatus.ACTIVE
        )
        
        server2 = MCPServer(
            server_id="server2",
            name="Analysis Server",
            endpoint="http://localhost:8081",
            capabilities=["analysis", "search"],
            status=MCPServerStatus.ACTIVE
        )
        
        server_manager.register_server(server1)
        server_manager.register_server(server2)
        
        memory_servers = server_manager.get_servers_by_capability("memory")
        assert len(memory_servers) == 1
        assert memory_servers[0] == server1
        
        search_servers = server_manager.get_servers_by_capability("search")
        assert len(search_servers) == 2
        
        nonexistent_servers = server_manager.get_servers_by_capability("nonexistent")
        assert len(nonexistent_servers) == 0
    
    @pytest.mark.asyncio
    async def test_discover_servers(self, server_manager):
        """Test automatic server discovery."""
        # Mock the discovery process
        with patch.object(server_manager, '_scan_network_for_servers') as mock_scan:
            mock_scan.return_value = [
                {
                    'server_id': 'discovered-1',
                    'name': 'Discovered Server 1',
                    'endpoint': 'http://localhost:9001',
                    'capabilities': ['memory']
                }
            ]
            
            discovered_servers = await server_manager.discover_servers()
            
            assert len(discovered_servers) == 1
            assert discovered_servers[0].server_id == 'discovered-1'
            assert len(server_manager.servers) == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, server_manager, sample_server):
        """Test server health checking."""
        server_manager.register_server(sample_server)
        
        # Mock successful health check
        with patch.object(server_manager, '_check_server_health') as mock_health:
            mock_health.return_value = (True, 0.9, 0.3)  # healthy, health_score, load_factor
            
            await server_manager.perform_health_check("test-server")
            
            updated_server = server_manager.get_server("test-server")
            assert updated_server.status == MCPServerStatus.ACTIVE
            assert updated_server.health_score == 0.9
            assert updated_server.load_factor == 0.3
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, server_manager, sample_server):
        """Test server health check failure handling."""
        server_manager.register_server(sample_server)
        
        # Mock failed health check
        with patch.object(server_manager, '_check_server_health') as mock_health:
            mock_health.return_value = (False, 0.0, 1.0)
            
            await server_manager.perform_health_check("test-server")
            
            updated_server = server_manager.get_server("test-server")
            assert updated_server.status == MCPServerStatus.ERROR


class TestContextVector:
    """Test the ContextVector data structure."""
    
    def test_context_vector_creation(self):
        """Test creating a context vector."""
        vector = ContextVector(
            context_id="ctx-1",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"source": "agent-1", "timestamp": 1234567890},
            ttl=3600
        )
        
        assert vector.context_id == "ctx-1"
        assert vector.embedding == [0.1, 0.2, 0.3, 0.4]
        assert vector.metadata["source"] == "agent-1"
        assert vector.ttl == 3600
    
    def test_context_vector_validation(self):
        """Test context vector validation."""
        with pytest.raises(ValueError, match="Context ID cannot be empty"):
            ContextVector(
                context_id="",
                embedding=[0.1, 0.2],
                metadata={},
                ttl=3600
            )
        
        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            ContextVector(
                context_id="ctx-1",
                embedding=[],
                metadata={},
                ttl=3600
            )


class TestContextSharingEngine:
    """Test the Context Sharing Engine functionality."""
    
    @pytest.fixture
    def context_engine(self):
        """Create a context sharing engine for testing."""
        return ContextSharingEngine(embedding_dimension=4)
    
    @pytest.fixture
    def sample_context_vector(self):
        """Create a sample context vector for testing."""
        return ContextVector(
            context_id="ctx-1",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"source": "agent-1"},
            ttl=3600
        )
    
    def test_context_engine_initialization(self, context_engine):
        """Test context sharing engine initialization."""
        assert context_engine.embedding_dimension == 4
        assert len(context_engine.context_store) == 0
        assert context_engine.similarity_threshold == 0.8
    
    def test_store_context(self, context_engine, sample_context_vector):
        """Test storing a context vector."""
        result = context_engine.store_context(sample_context_vector)
        
        assert result is True
        assert len(context_engine.context_store) == 1
        assert context_engine.context_store["ctx-1"] == sample_context_vector
    
    def test_store_context_dimension_mismatch(self, context_engine):
        """Test storing context with wrong embedding dimension."""
        wrong_dimension_vector = ContextVector(
            context_id="ctx-wrong",
            embedding=[0.1, 0.2],  # Wrong dimension (2 instead of 4)
            metadata={},
            ttl=3600
        )
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            context_engine.store_context(wrong_dimension_vector)
    
    def test_retrieve_context(self, context_engine, sample_context_vector):
        """Test retrieving a context vector by ID."""
        context_engine.store_context(sample_context_vector)
        
        retrieved = context_engine.retrieve_context("ctx-1")
        assert retrieved == sample_context_vector
        
        nonexistent = context_engine.retrieve_context("nonexistent")
        assert nonexistent is None
    
    def test_find_similar_contexts(self, context_engine):
        """Test finding similar contexts using vector similarity."""
        # Store multiple context vectors
        contexts = [
            ContextVector("ctx-1", [1.0, 0.0, 0.0, 0.0], {"type": "A"}, 3600),
            ContextVector("ctx-2", [0.9, 0.1, 0.0, 0.0], {"type": "A"}, 3600),  # Similar to ctx-1
            ContextVector("ctx-3", [0.0, 0.0, 1.0, 0.0], {"type": "B"}, 3600),  # Different
        ]
        
        for ctx in contexts:
            context_engine.store_context(ctx)
        
        # Query with vector similar to ctx-1
        query_vector = [0.95, 0.05, 0.0, 0.0]
        similar_contexts = context_engine.find_similar_contexts(
            query_vector, 
            max_results=2,
            similarity_threshold=0.8
        )
        
        assert len(similar_contexts) >= 1
        # Should find ctx-1 and ctx-2 as they're similar to the query
        similar_ids = [ctx.context_id for ctx in similar_contexts]
        assert "ctx-1" in similar_ids
    
    def test_cleanup_expired_contexts(self, context_engine):
        """Test cleanup of expired context vectors."""
        # Create contexts with different TTLs
        current_time = time.time()
        
        expired_context = ContextVector(
            "ctx-expired",
            [0.1, 0.2, 0.3, 0.4],
            {"created_at": current_time - 7200},  # 2 hours ago
            ttl=3600  # 1 hour TTL - should be expired
        )
        
        valid_context = ContextVector(
            "ctx-valid",
            [0.5, 0.6, 0.7, 0.8],
            {"created_at": current_time - 1800},  # 30 minutes ago
            ttl=3600  # 1 hour TTL - should be valid
        )
        
        context_engine.store_context(expired_context)
        context_engine.store_context(valid_context)
        
        # Perform cleanup
        cleaned_count = context_engine.cleanup_expired_contexts()
        
        assert cleaned_count == 1
        assert len(context_engine.context_store) == 1
        assert "ctx-valid" in context_engine.context_store
        assert "ctx-expired" not in context_engine.context_store


class TestAgent:
    """Test the Agent data structure."""
    
    def test_agent_creation(self):
        """Test creating an agent instance."""
        agent = Agent(
            agent_id="agent-1",
            name="Memory Agent",
            agent_type="memory_manager",
            capabilities=["store", "retrieve", "search"],
            status=AgentStatus.INITIALIZING,
            server_id="server-1"
        )
        
        assert agent.agent_id == "agent-1"
        assert agent.name == "Memory Agent"
        assert agent.agent_type == "memory_manager"
        assert agent.capabilities == ["store", "retrieve", "search"]
        assert agent.status == AgentStatus.INITIALIZING
        assert agent.server_id == "server-1"
        assert agent.priority == 1  # Default priority
        assert agent.resource_allocation == {}  # Default empty allocation


class TestAgentCoordinator:
    """Test the Agent Coordination functionality."""
    
    @pytest.fixture
    def coordinator(self):
        """Create an agent coordinator for testing."""
        return AgentCoordinator()
    
    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing."""
        return Agent(
            agent_id="agent-1",
            name="Test Agent",
            agent_type="test",
            capabilities=["test"],
            status=AgentStatus.ACTIVE,
            server_id="server-1"
        )
    
    def test_coordinator_initialization(self, coordinator):
        """Test agent coordinator initialization."""
        assert len(coordinator.agents) == 0
        assert len(coordinator.task_queue) == 0
        assert coordinator.conflict_resolution_strategy == "priority"
    
    def test_register_agent(self, coordinator, sample_agent):
        """Test registering an agent."""
        result = coordinator.register_agent(sample_agent)
        
        assert result is True
        assert len(coordinator.agents) == 1
        assert coordinator.agents["agent-1"] == sample_agent
    
    def test_register_duplicate_agent(self, coordinator, sample_agent):
        """Test registering an agent with duplicate ID fails."""
        coordinator.register_agent(sample_agent)
        
        duplicate_agent = Agent(
            agent_id="agent-1",  # Same ID
            name="Duplicate Agent",
            agent_type="duplicate",
            capabilities=["duplicate"],
            status=AgentStatus.ACTIVE,
            server_id="server-2"
        )
        
        result = coordinator.register_agent(duplicate_agent)
        assert result is False
        assert len(coordinator.agents) == 1
    
    def test_assign_task_to_agent(self, coordinator, sample_agent):
        """Test assigning a task to a specific agent."""
        coordinator.register_agent(sample_agent)
        
        task = {
            "task_id": "task-1",
            "type": "test_task",
            "data": {"input": "test"},
            "priority": 1
        }
        
        result = coordinator.assign_task("agent-1", task)
        assert result is True
        assert len(coordinator.task_queue) == 1
    
    def test_assign_task_to_nonexistent_agent(self, coordinator):
        """Test assigning a task to a non-existent agent fails."""
        task = {
            "task_id": "task-1",
            "type": "test_task",
            "data": {"input": "test"},
            "priority": 1
        }
        
        result = coordinator.assign_task("nonexistent-agent", task)
        assert result is False
        assert len(coordinator.task_queue) == 0
    
    def test_find_capable_agents(self, coordinator):
        """Test finding agents by capability."""
        agent1 = Agent(
            agent_id="agent-1",
            name="Memory Agent",
            agent_type="memory",
            capabilities=["store", "retrieve"],
            status=AgentStatus.ACTIVE,
            server_id="server-1"
        )
        
        agent2 = Agent(
            agent_id="agent-2",
            name="Analysis Agent",
            agent_type="analysis",
            capabilities=["analyze", "retrieve"],
            status=AgentStatus.ACTIVE,
            server_id="server-2"
        )
        
        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)
        
        retrieve_agents = coordinator.find_capable_agents("retrieve")
        assert len(retrieve_agents) == 2
        
        store_agents = coordinator.find_capable_agents("store")
        assert len(store_agents) == 1
        assert store_agents[0] == agent1
    
    def test_resolve_capability_conflict(self, coordinator):
        """Test conflict resolution when multiple agents can handle a task."""
        # Create agents with different priorities
        high_priority_agent = Agent(
            agent_id="agent-high",
            name="High Priority Agent",
            agent_type="test",
            capabilities=["test"],
            status=AgentStatus.ACTIVE,
            server_id="server-1",
            priority=10
        )
        
        low_priority_agent = Agent(
            agent_id="agent-low",
            name="Low Priority Agent",
            agent_type="test",
            capabilities=["test"],
            status=AgentStatus.ACTIVE,
            server_id="server-2",
            priority=1
        )
        
        coordinator.register_agent(high_priority_agent)
        coordinator.register_agent(low_priority_agent)
        
        # Find best agent for capability
        best_agent = coordinator.resolve_capability_conflict("test")
        
        # Should select the high priority agent
        assert best_agent == high_priority_agent


class TestResourceManager:
    """Test the Resource Management functionality."""
    
    @pytest.fixture
    def resource_manager(self):
        """Create a resource manager for testing."""
        return ResourceManager()
    
    def test_resource_manager_initialization(self, resource_manager):
        """Test resource manager initialization."""
        assert len(resource_manager.resource_pools) == 0
        assert resource_manager.load_balancing_strategy == "round_robin"
    
    def test_create_resource_pool(self, resource_manager):
        """Test creating a resource pool."""
        pool = ResourcePool(
            pool_id="cpu-pool",
            resource_type="cpu",
            total_capacity=100.0,
            available_capacity=100.0
        )
        
        result = resource_manager.create_pool(pool)
        assert result is True
        assert len(resource_manager.resource_pools) == 1
        assert resource_manager.resource_pools["cpu-pool"] == pool
    
    def test_allocate_resources(self, resource_manager):
        """Test allocating resources from a pool."""
        pool = ResourcePool(
            pool_id="memory-pool",
            resource_type="memory",
            total_capacity=1000.0,
            available_capacity=1000.0
        )
        resource_manager.create_pool(pool)
        
        allocation_result = resource_manager.allocate_resources(
            pool_id="memory-pool",
            amount=200.0,
            requester_id="agent-1"
        )
        
        assert allocation_result is True
        updated_pool = resource_manager.resource_pools["memory-pool"]
        assert updated_pool.available_capacity == 800.0
        assert "agent-1" in updated_pool.allocations
        assert updated_pool.allocations["agent-1"] == 200.0
    
    def test_allocate_insufficient_resources(self, resource_manager):
        """Test allocating more resources than available fails."""
        pool = ResourcePool(
            pool_id="disk-pool",
            resource_type="disk",
            total_capacity=100.0,
            available_capacity=50.0
        )
        resource_manager.create_pool(pool)
        
        allocation_result = resource_manager.allocate_resources(
            pool_id="disk-pool",
            amount=75.0,  # More than available
            requester_id="agent-1"
        )
        
        assert allocation_result is False
        # Pool should remain unchanged
        updated_pool = resource_manager.resource_pools["disk-pool"]
        assert updated_pool.available_capacity == 50.0
        assert len(updated_pool.allocations) == 0
    
    def test_deallocate_resources(self, resource_manager):
        """Test deallocating resources back to a pool."""
        pool = ResourcePool(
            pool_id="network-pool",
            resource_type="network",
            total_capacity=1000.0,
            available_capacity=800.0
        )
        pool.allocations["agent-1"] = 200.0
        resource_manager.create_pool(pool)
        
        deallocation_result = resource_manager.deallocate_resources(
            pool_id="network-pool",
            requester_id="agent-1"
        )
        
        assert deallocation_result is True
        updated_pool = resource_manager.resource_pools["network-pool"]
        assert updated_pool.available_capacity == 1000.0
        assert "agent-1" not in updated_pool.allocations
    
    def test_get_load_distribution(self, resource_manager):
        """Test getting load distribution across resource pools."""
        # Create multiple pools with different utilization
        pool1 = ResourcePool("pool1", "cpu", 100.0, 60.0)  # 40% utilized
        pool2 = ResourcePool("pool2", "cpu", 100.0, 20.0)  # 80% utilized
        
        resource_manager.create_pool(pool1)
        resource_manager.create_pool(pool2)
        
        distribution = resource_manager.get_load_distribution("cpu")
        
        assert len(distribution) == 2
        assert distribution["pool1"] == 0.4  # 40% load
        assert distribution["pool2"] == 0.8  # 80% load


class TestErrorHandler:
    """Test the Error Handling functionality."""
    
    @pytest.fixture
    def error_handler(self):
        """Create an error handler for testing."""
        return ErrorHandler()
    
    def test_error_handler_initialization(self, error_handler):
        """Test error handler initialization."""
        assert len(error_handler.recovery_strategies) > 0
        assert ErrorRecoveryStrategy.RETRY in error_handler.recovery_strategies
        assert ErrorRecoveryStrategy.FAILOVER in error_handler.recovery_strategies
        assert error_handler.max_retry_attempts == 3
    
    @pytest.mark.asyncio
    async def test_handle_server_error_with_retry(self, error_handler):
        """Test handling server error with retry strategy."""
        error_context = {
            "server_id": "server-1",
            "error_type": "connection_timeout",
            "error_message": "Connection timed out",
            "timestamp": time.time()
        }
        
        # Mock successful retry
        with patch.object(error_handler, '_attempt_server_recovery') as mock_recovery:
            mock_recovery.return_value = True
            
            recovery_result = await error_handler.handle_server_error(
                error_context,
                strategy=ErrorRecoveryStrategy.RETRY
            )
            
            assert recovery_result is True
            mock_recovery.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_server_error_with_failover(self, error_handler):
        """Test handling server error with failover strategy."""
        error_context = {
            "server_id": "server-1",
            "error_type": "server_crash",
            "error_message": "Server crashed",
            "timestamp": time.time()
        }
        
        # Mock successful failover
        with patch.object(error_handler, '_perform_failover') as mock_failover:
            mock_failover.return_value = "backup-server-1"
            
            recovery_result = await error_handler.handle_server_error(
                error_context,
                strategy=ErrorRecoveryStrategy.FAILOVER
            )
            
            assert recovery_result == "backup-server-1"
            mock_failover.assert_called_once()
    
    def test_determine_recovery_strategy(self, error_handler):
        """Test automatic determination of recovery strategy based on error type."""
        # Transient errors should use retry
        transient_error = {
            "error_type": "connection_timeout",
            "retry_count": 0
        }
        strategy = error_handler.determine_recovery_strategy(transient_error)
        assert strategy == ErrorRecoveryStrategy.RETRY
        
        # Critical errors should use failover
        critical_error = {
            "error_type": "server_crash",
            "retry_count": 0
        }
        strategy = error_handler.determine_recovery_strategy(critical_error)
        assert strategy == ErrorRecoveryStrategy.FAILOVER
        
        # Too many retries should escalate to failover
        exhausted_retries = {
            "error_type": "connection_timeout",
            "retry_count": 5
        }
        strategy = error_handler.determine_recovery_strategy(exhausted_retries)
        assert strategy == ErrorRecoveryStrategy.FAILOVER


class TestMCPOrchestrator:
    """Test the main MCP Orchestrator integration."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create an MCP orchestrator for testing."""
        return MCPOrchestrator()
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test MCP orchestrator initialization."""
        assert orchestrator.server_manager is not None
        assert orchestrator.context_engine is not None
        assert orchestrator.agent_coordinator is not None
        assert orchestrator.resource_manager is not None
        assert orchestrator.error_handler is not None
        assert orchestrator.is_running is False
    
    @pytest.mark.asyncio
    async def test_orchestrator_startup(self, orchestrator):
        """Test orchestrator startup process."""
        with patch.object(orchestrator.server_manager, 'discover_servers') as mock_discover:
            mock_discover.return_value = []
            
            await orchestrator.start()
            
            assert orchestrator.is_running is True
            mock_discover.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self, orchestrator):
        """Test orchestrator shutdown process."""
        orchestrator.is_running = True
        
        await orchestrator.shutdown()
        
        assert orchestrator.is_running is False
    
    @pytest.mark.asyncio
    async def test_process_request_end_to_end(self, orchestrator):
        """Test end-to-end request processing."""
        # Setup test environment
        await orchestrator.start()
        
        # Register a test server and agent
        test_server = MCPServer(
            server_id="test-server",
            name="Test Server",
            endpoint="http://localhost:8080",
            capabilities=["test"],
            status=MCPServerStatus.ACTIVE
        )
        orchestrator.server_manager.register_server(test_server)
        
        test_agent = Agent(
            agent_id="test-agent",
            name="Test Agent",
            agent_type="test",
            capabilities=["test"],
            status=AgentStatus.ACTIVE,
            server_id="test-server"
        )
        orchestrator.agent_coordinator.register_agent(test_agent)
        
        # Process a test request
        request = {
            "request_id": "req-1",
            "capability": "test",
            "data": {"input": "test data"},
            "priority": 1
        }
        
        with patch.object(orchestrator, '_execute_request') as mock_execute:
            mock_execute.return_value = {"result": "success"}
            
            result = await orchestrator.process_request(request)
            
            assert result is not None
            assert result["result"] == "success"
            mock_execute.assert_called_once()
    
    def test_get_system_status(self, orchestrator):
        """Test getting comprehensive system status."""
        status = orchestrator.get_system_status()
        
        assert "servers" in status
        assert "agents" in status
        assert "resources" in status
        assert "context_store" in status
        assert "is_running" in status
        
        assert status["is_running"] == orchestrator.is_running
        assert isinstance(status["servers"], dict)
        assert isinstance(status["agents"], dict)
        assert isinstance(status["resources"], dict)
        assert isinstance(status["context_store"], dict)