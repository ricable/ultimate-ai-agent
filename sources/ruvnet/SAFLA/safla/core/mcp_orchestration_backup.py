"""
MCP Orchestration Infrastructure for SAFLA.

This module provides the core infrastructure for orchestrating Model Context Protocol (MCP)
servers, managing context sharing between agents, coordinating agent activities, managing
resources, and handling errors with recovery mechanisms.

The MCP Orchestration Infrastructure is a Priority 1 component that enables:

1. **MCP Server Management**: Dynamic server discovery, registration, and lifecycle management
   - Automatic discovery of available MCP servers on the network
   - Health monitoring and status tracking
   - Load balancing across multiple servers

2. **Context Sharing**: Efficient context propagation between agents using vector embeddings
   - Vector similarity search for context routing
   - TTL-based context expiration and cleanup
   - Configurable similarity thresholds

3. **Agent Coordination**: Orchestration of multiple specialized agents with conflict resolution
   - Priority-based agent selection
   - Task assignment and queue management
   - Capability-based agent discovery

4. **Resource Management**: Load balancing and resource allocation across MCP servers
   - Resource pool creation and management
   - Dynamic allocation and deallocation
   - Load distribution monitoring

5. **Error Handling**: Robust error recovery and failover mechanisms
   - Multiple recovery strategies (retry, failover, circuit breaker)
   - Automatic strategy selection based on error type
   - Comprehensive error history tracking

This implementation follows the research findings from the integrated model and provides
a solid foundation for the SAFLA self-aware feedback loop algorithm.
"""

import asyncio
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MCPServerStatus(Enum):
    """Enumeration of possible MCP server statuses."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"


class AgentStatus(Enum):
    """Enumeration of possible agent statuses."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    BUSY = "busy"


class ErrorRecoveryStrategy(Enum):
    """Enumeration of error recovery strategies."""
    RETRY = "retry"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class MCPServer:
    """Data structure representing an MCP server."""
    server_id: str
    name: str
    endpoint: str
    capabilities: List[str]
    status: MCPServerStatus
    health_score: float = 1.0
    load_factor: float = 0.0
    last_health_check: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate server data after initialization."""
        if not self.server_id:
            raise ValueError("Server ID cannot be empty")
        if not (0.0 <= self.health_score <= 1.0):
            raise ValueError("Invalid health score: must be between 0.0 and 1.0")
        if not (0.0 <= self.load_factor <= 1.0):
            raise ValueError("Invalid load factor: must be between 0.0 and 1.0")


@dataclass
class ContextVector:
    """Data structure representing a context vector for sharing between agents."""
    context_id: str
    embedding: List[float]
    metadata: Dict[str, Any]
    ttl: int  # Time to live in seconds
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate context vector data after initialization."""
        if not self.context_id:
            raise ValueError("Context ID cannot be empty")
        if not self.embedding:
            raise ValueError("Embedding cannot be empty")
        if self.ttl <= 0:
            raise ValueError("TTL must be positive")


@dataclass
class Agent:
    """Data structure representing an agent in the system."""
    agent_id: str
    name: str
    agent_type: str
    capabilities: List[str]
    status: AgentStatus
    server_id: str
    priority: int = 1
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate agent data after initialization."""
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty")
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        if self.priority < 1:
            raise ValueError("Priority must be at least 1")


@dataclass
class ResourcePool:
    """Data structure representing a resource pool."""
    pool_id: str
    resource_type: str
    total_capacity: float
    available_capacity: float
    allocations: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate resource pool data after initialization."""
        if not self.pool_id:
            raise ValueError("Pool ID cannot be empty")
        if self.total_capacity < 0:
            raise ValueError("Total capacity cannot be negative")
        if self.available_capacity < 0:
            raise ValueError("Available capacity cannot be negative")
        if self.available_capacity > self.total_capacity:
            raise ValueError("Available capacity cannot exceed total capacity")


class MCPServerManager:
    """Manages MCP server discovery, registration, and lifecycle."""
    
    def __init__(self):
        """Initialize the MCP server manager."""
        self.servers: Dict[str, MCPServer] = {}
        self.discovery_enabled: bool = True
        self.health_check_interval: float = 30.0
        self._health_check_task: Optional[asyncio.Task] = None
    
    def register_server(self, server: MCPServer) -> bool:
        """
        Register a new MCP server.
        
        Args:
            server: The MCP server to register
            
        Returns:
            True if registration successful, False if server ID already exists
        """
        if server.server_id in self.servers:
            logger.warning(f"Server {server.server_id} already registered")
            return False
        
        self.servers[server.server_id] = server
        logger.info(f"Registered server {server.server_id}")
        return True
    
    def unregister_server(self, server_id: str) -> bool:
        """
        Unregister an MCP server.
        
        Args:
            server_id: ID of the server to unregister
            
        Returns:
            True if unregistration successful, False if server not found
        """
        if server_id not in self.servers:
            logger.warning(f"Server {server_id} not found for unregistration")
            return False
        
        del self.servers[server_id]
        logger.info(f"Unregistered server {server_id}")
        return True
    
    def get_server(self, server_id: str) -> Optional[MCPServer]:
        """
        Get a server by ID.
        
        Args:
            server_id: ID of the server to retrieve
            
        Returns:
            The server if found, None otherwise
        """
        return self.servers.get(server_id)
    
    def get_servers_by_capability(self, capability: str) -> List[MCPServer]:
        """
        Get all servers that have a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of servers with the specified capability
        """
        return [
            server for server in self.servers.values()
            if capability in server.capabilities and server.status == MCPServerStatus.ACTIVE
        ]
    
    async def discover_servers(self) -> List[MCPServer]:
        """
        Discover available MCP servers on the network.
        
        Returns:
            List of discovered servers
        """
        if not self.discovery_enabled:
            return []
        
        discovered_servers = []
        
        # Simulate network discovery
        discovered_configs = await self._scan_network_for_servers()
        
        for config in discovered_configs:
            server = MCPServer(
                server_id=config['server_id'],
                name=config['name'],
                endpoint=config['endpoint'],
                capabilities=config['capabilities'],
                status=MCPServerStatus.INITIALIZING
            )
            
            if self.register_server(server):
                discovered_servers.append(server)
        
        return discovered_servers
    
    async def _scan_network_for_servers(self) -> List[Dict[str, Any]]:
        """
        Scan the network for available MCP servers.
        
        Returns:
            List of server configuration dictionaries
        """
        # This would implement actual network discovery
        # For now, return empty list as this is a mock implementation
        return []
    
    async def perform_health_check(self, server_id: str) -> bool:
        """
        Perform a health check on a specific server.
        
        Args:
            server_id: ID of the server to check
            
        Returns:
            True if health check successful, False otherwise
        """
        server = self.get_server(server_id)
        if not server:
            return False
        
        try:
            # Perform actual health check
            is_healthy, health_score, load_factor = await self._check_server_health(server)
            
            # Update server status
            server.health_score = health_score
            server.load_factor = load_factor
            server.last_health_check = time.time()
            
            if is_healthy:
                server.status = MCPServerStatus.ACTIVE
            else:
                server.status = MCPServerStatus.ERROR
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for server {server_id}: {e}")
            server.status = MCPServerStatus.ERROR
            server.health_score = 0.0
            return False
    
    async def _check_server_health(self, server: MCPServer) -> Tuple[bool, float, float]:
        """
        Check the health of a specific server.
        
        Args:
            server: The server to check
            
        Returns:
            Tuple of (is_healthy, health_score, load_factor)
        """
        # This would implement actual health checking logic
        # For now, return mock values
        return True, 1.0, 0.0


class ContextSharingEngine:
    """Manages context sharing between agents using vector embeddings."""
    
    def __init__(self, embedding_dimension: int = 512):
        """
        Initialize the context sharing engine.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors
        """
        self.embedding_dimension = embedding_dimension
        self.context_store: Dict[str, ContextVector] = {}
        self.similarity_threshold = 0.8
    
    def store_context(self, context: ContextVector) -> bool:
        """
        Store a context vector.
        
        Args:
            context: The context vector to store
            
        Returns:
            True if storage successful
        """
        if len(context.embedding) != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(context.embedding)}"
            )
        
        self.context_store[context.context_id] = context
        logger.debug(f"Stored context {context.context_id}")
        return True
    
    def retrieve_context(self, context_id: str) -> Optional[ContextVector]:
        """
        Retrieve a context vector by ID.
        
        Args:
            context_id: ID of the context to retrieve
            
        Returns:
            The context vector if found, None otherwise
        """
        return self.context_store.get(context_id)
    
    def find_similar_contexts(
        self, 
        query_vector: List[float], 
        max_results: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[ContextVector]:
        """
        Find contexts similar to the query vector.
        
        Args:
            query_vector: The query embedding vector
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar context vectors
        """
        if len(query_vector) != self.embedding_dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_vector)}"
            )
        
        threshold = similarity_threshold or self.similarity_threshold
        similarities = []
        
        for context in self.context_store.values():
            similarity = self._calculate_cosine_similarity(query_vector, context.embedding)
            if similarity >= threshold:
                similarities.append((similarity, context))
        
        # Sort by similarity (descending) and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [context for _, context in similarities[:max_results]]
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def cleanup_expired_contexts(self) -> int:
        """
        Remove expired context vectors.
        
        Returns:
            Number of contexts removed
        """
        current_time = time.time()
        expired_contexts = []
        
        for context_id, context in self.context_store.items():
            # Check if context has expired based on created_at + ttl
            context_created_at = context.metadata.get('created_at', context.created_at)
            if current_time - context_created_at > context.ttl:
                expired_contexts.append(context_id)
        
        for context_id in expired_contexts:
            del self.context_store[context_id]
        
        logger.debug(f"Cleaned up {len(expired_contexts)} expired contexts")
        return len(expired_contexts)


class AgentCoordinator:
    """Coordinates multiple agents and handles task assignment."""
    
    def __init__(self):
        """Initialize the agent coordinator."""
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.conflict_resolution_strategy = "priority"
    
    def register_agent(self, agent: Agent) -> bool:
        """
        Register a new agent.
        
        Args:
            agent: The agent to register
            
        Returns:
            True if registration successful, False if agent ID already exists
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} already registered")
            return False
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id}")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if unregistration successful, False if agent not found
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found for unregistration")
            return False
        
        del self.agents[agent_id]
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def assign_task(self, agent_id: str, task: Dict[str, Any]) -> bool:
        """
        Assign a task to a specific agent.
        
        Args:
            agent_id: ID of the agent to assign the task to
            task: The task to assign
            
        Returns:
            True if assignment successful, False if agent not found
        """
        if agent_id not in self.agents:
            logger.warning(f"Cannot assign task to unknown agent {agent_id}")
            return False
        
        task["assigned_agent"] = agent_id
        task["assigned_at"] = time.time()
        self.task_queue.append(task)
        
        logger.info(f"Assigned task {task.get('task_id')} to agent {agent_id}")
        return True
    
    def find_capable_agents(self, capability: str) -> List[Agent]:
        """
        Find all agents that have a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of agents with the specified capability
        """
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities and agent.status == AgentStatus.ACTIVE
        ]
    
    def resolve_capability_conflict(self, capability: str) -> Optional[Agent]:
        """
        Resolve conflicts when multiple agents can handle a capability.
        
        Args:
            capability: The capability to resolve conflicts for
            
        Returns:
            The best agent for the capability, or None if no capable agents
        """
        capable_agents = self.find_capable_agents(capability)
        
        if not capable_agents:
            return None
        
        if self.conflict_resolution_strategy == "priority":
            # Return agent with highest priority
            return max(capable_agents, key=lambda agent: agent.priority)
        elif self.conflict_resolution_strategy == "load_balancing":
            # Return agent with lowest current load (simplified)
            return min(capable_agents, key=lambda agent: len(agent.resource_allocation))
        else:
            # Default: return first available agent
            return capable_agents[0]


class ResourceManager:
    """Manages resource allocation and load balancing."""
    
    def __init__(self):
        """Initialize the resource manager."""
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.load_balancing_strategy = "round_robin"
    
    def create_pool(self, pool: ResourcePool) -> bool:
        """
        Create a new resource pool.
        
        Args:
            pool: The resource pool to create
            
        Returns:
            True if creation successful, False if pool ID already exists
        """
        if pool.pool_id in self.resource_pools:
            logger.warning(f"Resource pool {pool.pool_id} already exists")
            return False
        
        self.resource_pools[pool.pool_id] = pool
        logger.info(f"Created resource pool {pool.pool_id}")
        return True
    
    def allocate_resources(
        self, 
        pool_id: str, 
        amount: float, 
        requester_id: str
    ) -> bool:
        """
        Allocate resources from a pool.
        
        Args:
            pool_id: ID of the resource pool
            amount: Amount of resources to allocate
            requester_id: ID of the entity requesting resources
            
        Returns:
            True if allocation successful, False otherwise
        """
        pool = self.resource_pools.get(pool_id)
        if not pool:
            logger.error(f"Resource pool {pool_id} not found")
            return False
        
        if pool.available_capacity < amount:
            logger.warning(
                f"Insufficient resources in pool {pool_id}: "
                f"requested {amount}, available {pool.available_capacity}"
            )
            return False
        
        pool.available_capacity -= amount
        pool.allocations[requester_id] = pool.allocations.get(requester_id, 0) + amount
        
        logger.info(f"Allocated {amount} resources from pool {pool_id} to {requester_id}")
        return True
    
    def deallocate_resources(self, pool_id: str, requester_id: str) -> bool:
        """
        Deallocate all resources for a requester from a pool.
        
        Args:
            pool_id: ID of the resource pool
            requester_id: ID of the entity to deallocate resources from
            
        Returns:
            True if deallocation successful, False otherwise
        """
        pool = self.resource_pools.get(pool_id)
        if not pool:
            logger.error(f"Resource pool {pool_id} not found")
            return False
        
        if requester_id not in pool.allocations:
            logger.warning(f"No allocation found for {requester_id} in pool {pool_id}")
            return False
        
        amount = pool.allocations[requester_id]
        pool.available_capacity += amount
        del pool.allocations[requester_id]
        
        logger.info(f"Deallocated {amount} resources from pool {pool_id} for {requester_id}")
        return True
    
    def get_load_distribution(self, resource_type: str) -> Dict[str, float]:
        """
        Get load distribution across pools of a specific resource type.
        
        Args:
            resource_type: Type of resource to analyze
            
        Returns:
            Dictionary mapping pool IDs to load factors (0.0 to 1.0)
        """
        distribution = {}
        
        for pool_id, pool in self.resource_pools.items():
            if pool.resource_type == resource_type:
                if pool.total_capacity > 0:
                    load_factor = (pool.total_capacity - pool.available_capacity) / pool.total_capacity
                    distribution[pool_id] = load_factor
                else:
                    distribution[pool_id] = 0.0
        
        return distribution


class ErrorHandler:
    """Handles errors and implements recovery strategies."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.recovery_strategies = [
            ErrorRecoveryStrategy.RETRY,
            ErrorRecoveryStrategy.FAILOVER,
            ErrorRecoveryStrategy.CIRCUIT_BREAKER,
            ErrorRecoveryStrategy.GRACEFUL_DEGRADATION
        ]
        self.max_retry_attempts = 3
        self.circuit_breaker_threshold = 5
        self.error_history: List[Dict[str, Any]] = []
    
    async def handle_server_error(
        self, 
        error_context: Dict[str, Any], 
        strategy: Optional[ErrorRecoveryStrategy] = None
    ) -> Union[bool, str]:
        """
        Handle a server error using the specified or determined strategy.
        
        Args:
            error_context: Context information about the error
            strategy: Recovery strategy to use (auto-determined if None)
            
        Returns:
            Recovery result (True/False for success, or new server ID for failover)
        """
        if strategy is None:
            strategy = self.determine_recovery_strategy(error_context)
        
        self.error_history.append({
            **error_context,
            "strategy": strategy.value,
            "handled_at": time.time()
        })
        
        if strategy == ErrorRecoveryStrategy.RETRY:
            return await self._attempt_server_recovery(error_context)
        elif strategy == ErrorRecoveryStrategy.FAILOVER:
            return await self._perform_failover(error_context)
        elif strategy == ErrorRecoveryStrategy.CIRCUIT_BREAKER:
            return await self._activate_circuit_breaker(error_context)
        elif strategy == ErrorRecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._enable_graceful_degradation(error_context)
        else:
            logger.error(f"Unknown recovery strategy: {strategy}")
            return False
    
    def determine_recovery_strategy(self, error_context: Dict[str, Any]) -> ErrorRecoveryStrategy:
        """
        Determine the best recovery strategy for an error.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            The recommended recovery strategy
        """
        error_type = error_context.get("error_type", "unknown")
        retry_count = error_context.get("retry_count", 0)
        
        # If too many retries, escalate to failover
        if retry_count >= self.max_retry_attempts:
            return ErrorRecoveryStrategy.FAILOVER
        
        # Transient errors should be retried
        if error_type in ["connection_timeout", "temporary_unavailable", "rate_limit"]:
            return ErrorRecoveryStrategy.RETRY
        
        # Critical errors should trigger failover
        if error_type in ["server_crash", "authentication_failure", "critical_error"]:
            return ErrorRecoveryStrategy.FAILOVER
        
        # Default to retry for unknown errors
        return ErrorRecoveryStrategy.RETRY
    
    async def _attempt_server_recovery(self, error_context: Dict[str, Any]) -> bool:
        """
        Attempt to recover a server through retry.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            True if recovery successful, False otherwise
        """
        # This would implement actual server recovery logic
        # For now, return True to simulate successful recovery
        logger.info(f"Attempting server recovery for {error_context.get('server_id')}")
        return True
    
    async def _perform_failover(self, error_context: Dict[str, Any]) -> str:
        """
        Perform failover to a backup server.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            ID of the backup server
        """
        # This would implement actual failover logic
        # For now, return a mock backup server ID
        backup_server_id = f"backup-{error_context.get('server_id', 'unknown')}"
        logger.info(f"Performing failover to {backup_server_id}")
        return backup_server_id
    
    async def _activate_circuit_breaker(self, error_context: Dict[str, Any]) -> bool:
        """
        Activate circuit breaker for a failing component.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            True if circuit breaker activated successfully
        """
        logger.info(f"Activating circuit breaker for {error_context.get('server_id')}")
        return True
    
    async def _enable_graceful_degradation(self, error_context: Dict[str, Any]) -> bool:
        """
        Enable graceful degradation mode.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            True if graceful degradation enabled successfully
        """
        logger.info(f"Enabling graceful degradation for {error_context.get('server_id')}")
        return True


class MCPOrchestrator:
    """Main orchestrator that coordinates all MCP infrastructure components."""
    
    def __init__(self):
        """Initialize the MCP orchestrator."""
        self.server_manager = MCPServerManager()
        self.context_engine = ContextSharingEngine()
        self.agent_coordinator = AgentCoordinator()
        self.resource_manager = ResourceManager()
        self.error_handler = ErrorHandler()
        self.is_running = False
        self._orchestration_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the MCP orchestrator."""
        if self.is_running:
            logger.warning("Orchestrator is already running")
            return
        
        logger.info("Starting MCP orchestrator")
        
        # Discover available servers
        await self.server_manager.discover_servers()
        
        # Start background tasks
        self._orchestration_task = asyncio.create_task(self._orchestration_loop())
        
        self.is_running = True
        logger.info("MCP orchestrator started successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the MCP orchestrator."""
        if not self.is_running:
            logger.warning("Orchestrator is not running")
            return
        
        logger.info("Shutting down MCP orchestrator")
        
        self.is_running = False
        
        # Cancel background tasks
        if self._orchestration_task:
            self._orchestration_task.cancel()
            try:
                await self._orchestration_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MCP orchestrator shut down successfully")
    
    async def process_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a request through the orchestration system.
        
        Args:
            request: The request to process
            
        Returns:
            The response from processing the request
        """
        if not self.is_running:
            logger.error("Cannot process request: orchestrator not running")
            return None
        
        request_id = request.get("request_id", "unknown")
        capability = request.get("capability")
        
        logger.info(f"Processing request {request_id} for capability {capability}")
        
        try:
            # Find capable agents
            capable_agents = self.agent_coordinator.find_capable_agents(capability)
            if not capable_agents:
                logger.error(f"No capable agents found for capability {capability}")
                return {"error": "No capable agents available"}
            
            # Select best agent
            selected_agent = self.agent_coordinator.resolve_capability_conflict(capability)
            if not selected_agent:
                logger.error(f"Failed to select agent for capability {capability}")
                return {"error": "Agent selection failed"}
            
            # Execute the request
            result = await self._execute_request(request, selected_agent)
            
            logger.info(f"Successfully processed request {request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            return {"error": str(e)}
    
    async def _execute_request(
        self, 
        request: Dict[str, Any], 
        agent: Agent
    ) -> Dict[str, Any]:
        """
        Execute a request using the selected agent.
        
        Args:
            request: The request to execute
            agent: The agent to execute the request
            
        Returns:
            The result of the request execution
        """
        # This would implement actual request execution logic
        # For now, return a mock result
        return {
            "request_id": request.get("request_id"),
            "agent_id": agent.agent_id,
            "result": "success",
            "executed_at": time.time()
        }
    
    async def _orchestration_loop(self) -> None:
        """Main orchestration loop for background tasks."""
        while self.is_running:
            try:
                # Perform periodic health checks
                for server_id in list(self.server_manager.servers.keys()):
                    await self.server_manager.perform_health_check(server_id)
                
                # Clean up expired contexts
                self.context_engine.cleanup_expired_contexts()
                
                # Sleep before next iteration
                await asyncio.sleep(30)  # 30 second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(5)  # Short sleep on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary containing system status information
        """
        return {
            "is_running": self.is_running,
            "servers": {
                "total": len(self.server_manager.servers),
                "active": len([
                    s for s in self.server_manager.servers.values()
                    if s.status == MCPServerStatus.ACTIVE
                ]),
                "details": {
                    server_id: {
                        "status": server.status.value,
                        "health_score": server.health_score,
                        "load_factor": server.load_factor
                    }
                    for server_id, server in self.server_manager.servers.items()
                }
            },
            "agents": {
                "total": len(self.agent_coordinator.agents),
                "active": len([
                    a for a in self.agent_coordinator.agents.values()
                    if a.status == AgentStatus.ACTIVE
                ]),
                "details": {
                    agent_id: {
                        "status": agent.status.value,
                        "capabilities": agent.capabilities,
                        "server_id": agent.server_id
                    }
                    for agent_id, agent in self.agent_coordinator.agents.items()
                }
            },
            "resources": {
                "pools": len(self.resource_manager.resource_pools),
                "details": {
                    pool_id: {
                        "resource_type": pool.resource_type,
                        "utilization": (
                            (pool.total_capacity - pool.available_capacity) / pool.total_capacity
                            if pool.total_capacity > 0 else 0.0
                        ),
                        "allocations": len(pool.allocations)
                    }
                    for pool_id, pool in self.resource_manager.resource_pools.items()
                }
            },
            "context_store": {
                "total_contexts": len(self.context_engine.context_store),
                "embedding_dimension": self.context_engine.embedding_dimension
            }
        }