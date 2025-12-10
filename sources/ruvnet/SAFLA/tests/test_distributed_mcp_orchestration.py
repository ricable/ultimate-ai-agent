"""
Test suite for Distributed MCP Orchestration - Multi-node coordination and deployment.

This module tests the distributed capabilities for SAFLA:
- Multi-node MCP coordination
- Distributed task scheduling and execution
- Load balancing and fault tolerance
- Inter-node communication and synchronization
- Scalable deployment management

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass, field
from enum import Enum

# Import the classes we'll implement
from safla.core.distributed_mcp_orchestration import (
    DistributedMCPOrchestrator,
    MCPNode,
    TaskScheduler,
    LoadBalancer,
    FaultTolerance,
    InterNodeCommunication,
    DistributedTaskManager,
    NodeRegistry,
    ClusterConfig,
    NodeState,
    NodeStatus,
    TaskStatus,
    DistributedTask,
    NodeMetrics,
    ClusterMetrics,
    SynchronizationManager,
    ConsensusProtocol
)


@dataclass
class MockNodeInfo:
    """Mock node information for testing."""
    node_id: str
    host: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    current_load: float = 0.0
    state: NodeState = NodeState.ACTIVE
    last_heartbeat: datetime = field(default_factory=datetime.now)


class TestDistributedMCPOrchestrator:
    """Test suite for Distributed MCP Orchestrator functionality."""
    
    @pytest.fixture
    def cluster_config(self):
        """Create cluster configuration for testing."""
        return ClusterConfig(
            cluster_name="test_cluster",
            min_nodes=2,
            max_nodes=10,
            heartbeat_interval=5.0,
            task_timeout=300.0,
            replication_factor=2,
            consensus_threshold=0.6,
            load_balance_strategy="round_robin",
            fault_tolerance_enabled=True
        )
    
    @pytest.fixture
    def orchestrator(self, cluster_config):
        """Create DistributedMCPOrchestrator instance for testing."""
        return DistributedMCPOrchestrator(config=cluster_config)
    
    @pytest.fixture
    def mock_nodes(self):
        """Create mock nodes for testing."""
        return [
            MockNodeInfo(
                node_id=f"node_{i}",
                host=f"192.168.1.{10+i}",
                port=8000 + i,
                capabilities=["compute", "storage"] if i % 2 == 0 else ["compute"],
                max_concurrent_tasks=5,
                current_load=0.2 * i
            )
            for i in range(5)
        ]
    
    def test_orchestrator_initialization(self, orchestrator, cluster_config):
        """Test DistributedMCPOrchestrator initialization."""
        assert orchestrator.config == cluster_config
        assert orchestrator.node_registry is not None
        assert orchestrator.task_scheduler is not None
        assert orchestrator.load_balancer is not None
        assert orchestrator.fault_tolerance is not None
        assert orchestrator.communication is not None
        assert orchestrator.cluster_id is not None
        assert orchestrator.is_leader is False  # Initially not leader
    
    @pytest.mark.asyncio
    async def test_cluster_initialization(self, orchestrator, mock_nodes):
        """Test cluster initialization with multiple nodes."""
        # Mock node discovery
        with patch.object(orchestrator, '_discover_nodes') as mock_discover:
            mock_discover.return_value = mock_nodes[:3]  # Start with 3 nodes
            
            await orchestrator.initialize_cluster()
            
            assert orchestrator.is_initialized
            assert len(orchestrator.get_active_nodes()) == 3
            assert orchestrator.cluster_size >= orchestrator.config.min_nodes
    
    @pytest.mark.asyncio
    async def test_node_registration_and_deregistration(self, orchestrator):
        """Test node registration and deregistration."""
        node_info = MockNodeInfo(
            node_id="test_node",
            host="192.168.1.100",
            port=8080,
            capabilities=["compute", "storage"]
        )
        
        # Register node
        success = await orchestrator.register_node(node_info)
        assert success
        assert orchestrator.node_registry.has_node("test_node")
        assert "test_node" in orchestrator.get_active_nodes()
        
        # Deregister node
        success = await orchestrator.deregister_node("test_node")
        assert success
        assert not orchestrator.node_registry.has_node("test_node")
        assert "test_node" not in orchestrator.get_active_nodes()
    
    @pytest.mark.asyncio
    async def test_leader_election(self, orchestrator, mock_nodes):
        """Test leader election process."""
        # Initialize cluster with multiple nodes
        for node in mock_nodes[:3]:
            await orchestrator.register_node(node)
        
        # Trigger leader election
        with patch.object(orchestrator.consensus_protocol, 'elect_leader') as mock_elect:
            mock_elect.return_value = mock_nodes[0].node_id
            
            leader_id = await orchestrator.elect_leader()
            
            assert leader_id == mock_nodes[0].node_id
            mock_elect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_distributed_task_submission(self, orchestrator, mock_nodes):
        """Test distributed task submission and scheduling."""
        # Setup cluster
        for node in mock_nodes[:3]:
            await orchestrator.register_node(node)
        
        # Create test task
        task = DistributedTask(
            task_id="test_task_001",
            task_type="compute",
            payload={"operation": "matrix_multiply", "size": 1000},
            requirements={"cpu_cores": 2, "memory_gb": 4},
            priority=1,
            timeout=300.0
        )
        
        # Submit task
        with patch.object(orchestrator.task_scheduler, 'schedule_task') as mock_schedule:
            mock_schedule.return_value = mock_nodes[0].node_id
            
            assigned_node = await orchestrator.submit_task(task)
            
            assert assigned_node == mock_nodes[0].node_id
            mock_schedule.assert_called_once()
            # Verify the task was passed as first argument
            args, kwargs = mock_schedule.call_args
            assert args[0] == task
            assert len(args) == 2  # task and available_nodes
    
    @pytest.mark.asyncio
    async def test_load_balancing_strategies(self, orchestrator, mock_nodes):
        """Test different load balancing strategies."""
        # Setup cluster with varying loads
        for i, node in enumerate(mock_nodes[:4]):
            node.current_load = 0.1 * (i + 1)  # Increasing load
            await orchestrator.register_node(node)
        
        load_balancer = orchestrator.load_balancer
        
        # Test round-robin strategy
        load_balancer.set_strategy("round_robin")
        assignments = []
        for i in range(8):
            task = DistributedTask(
                task_id=f"task_{i}",
                task_type="compute",
                payload={},
                requirements={}
            )
            available_nodes = [
                orchestrator.node_registry.get_node(node_id)
                for node_id in orchestrator.get_active_nodes()
            ]
            available_nodes = [node for node in available_nodes if node is not None]
            node_id = load_balancer.select_node(task, available_nodes)
            assignments.append(node_id)
        
        # Should cycle through nodes
        unique_assignments = set(assignments)
        assert len(unique_assignments) == 4
        
        # Test least-loaded strategy
        load_balancer.set_strategy("least_loaded")
        task = DistributedTask(
            task_id="load_test",
            task_type="compute",
            payload={},
            requirements={}
        )
        available_nodes = [
            orchestrator.node_registry.get_node(node_id)
            for node_id in orchestrator.get_active_nodes()
        ]
        available_nodes = [node for node in available_nodes if node is not None]
        selected_node = load_balancer.select_node(task, available_nodes)
        
        # Should select node with lowest load (node_0)
        assert selected_node == mock_nodes[0].node_id
    
    @pytest.mark.asyncio
    async def test_fault_tolerance_and_recovery(self, orchestrator, mock_nodes):
        """Test fault tolerance and node failure recovery."""
        # Setup cluster
        for node in mock_nodes[:3]:
            await orchestrator.register_node(node)
        
        fault_tolerance = orchestrator.fault_tolerance
        
        # Simulate node failure
        failed_node_id = mock_nodes[1].node_id
        
        with patch.object(fault_tolerance, '_detect_node_failure') as mock_detect:
            mock_detect.return_value = True
            
            # Trigger failure detection
            await fault_tolerance.handle_node_failure(failed_node_id)
            
            # Node should be marked as failed
            node_status = orchestrator.node_registry.get_node_status(failed_node_id)
            assert node_status == NodeState.FAILED
            
            # Active nodes should exclude failed node
            active_nodes = orchestrator.get_active_nodes()
            assert failed_node_id not in active_nodes
    
    @pytest.mark.asyncio
    async def test_task_migration_on_failure(self, orchestrator, mock_nodes):
        """Test task migration when node fails."""
        # Setup cluster
        for node in mock_nodes[:3]:
            await orchestrator.register_node(node)
        
        # Create and assign task
        task = DistributedTask(
            task_id="migration_test",
            task_type="compute",
            payload={"data": "test_data"},
            requirements={}
        )
        
        original_node = mock_nodes[0].node_id
        
        with patch.object(orchestrator.task_scheduler, 'schedule_task') as mock_schedule:
            mock_schedule.return_value = original_node
            await orchestrator.submit_task(task)
        
        # Simulate node failure and task migration
        with patch.object(orchestrator, '_migrate_tasks') as mock_migrate:
            mock_migrate.return_value = [mock_nodes[1].node_id]
            
            migrated_nodes = await orchestrator.handle_task_migration(original_node)
            
            assert mock_nodes[1].node_id in migrated_nodes
            mock_migrate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_inter_node_communication(self, orchestrator, mock_nodes):
        """Test inter-node communication and message passing."""
        # Setup cluster
        for node in mock_nodes[:3]:
            await orchestrator.register_node(node)
        
        communication = orchestrator.communication
        
        # Test broadcast message
        message = {
            "type": "cluster_update",
            "data": {"new_config": "test_value"},
            "timestamp": datetime.now().isoformat()
        }
        
        with patch.object(communication, '_send_message') as mock_send:
            mock_send.return_value = True
            
            success = await communication.broadcast_message(message)
            
            assert success
            # Should send to all nodes except self
            assert mock_send.call_count == len(mock_nodes[:3])
        
        # Test direct message
        target_node = mock_nodes[1].node_id
        direct_message = {"type": "task_assignment", "task_id": "test_task"}
        
        with patch.object(communication, '_send_direct_message') as mock_direct:
            mock_direct.return_value = True
            
            success = await communication.send_message(target_node, direct_message)
            
            assert success
            mock_direct.assert_called_once_with(target_node, direct_message)
    
    @pytest.mark.asyncio
    async def test_heartbeat_monitoring(self, orchestrator, mock_nodes):
        """Test heartbeat monitoring and node health tracking."""
        # Setup cluster
        for node in mock_nodes[:3]:
            await orchestrator.register_node(node)
        
        # Start heartbeat monitoring
        with patch.object(orchestrator, '_process_heartbeat') as mock_process:
            mock_process.return_value = True
            
            await orchestrator.start_heartbeat_monitoring()
            
            # Simulate heartbeat from node
            heartbeat_data = {
                "node_id": mock_nodes[0].node_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "cpu_usage": 0.3,
                    "memory_usage": 0.5,
                    "active_tasks": 2
                }
            }
            
            await orchestrator.process_heartbeat(heartbeat_data)
            
            # Node should be marked as healthy
            node_metrics = orchestrator.get_node_metrics(mock_nodes[0].node_id)
            assert node_metrics is not None
            assert node_metrics.cpu_usage == 0.3
    
    def test_cluster_metrics_collection(self, orchestrator, mock_nodes):
        """Test cluster-wide metrics collection and aggregation."""
        # Setup cluster with mock metrics
        for i, node in enumerate(mock_nodes[:3]):
            orchestrator.node_registry.register_node(node)
            
            # Mock node metrics
            metrics = NodeMetrics(
                node_id=node.node_id,
                cpu_usage=0.2 + (i * 0.1),
                memory_usage=0.3 + (i * 0.1),
                disk_usage=0.4 + (i * 0.1),
                network_io=1000 + (i * 500),
                active_tasks=i + 1,
                completed_tasks=i * 10,
                failed_tasks=i,
                timestamp=datetime.now()
            )
            orchestrator._update_node_metrics(node.node_id, metrics)
        
        # Get cluster metrics
        cluster_metrics = orchestrator.get_cluster_metrics()
        
        assert cluster_metrics.total_nodes == 3
        assert cluster_metrics.active_nodes == 3
        assert cluster_metrics.total_tasks == 6  # 1+2+3
        assert 0.2 <= cluster_metrics.average_cpu_usage <= 0.4
        assert 0.3 <= cluster_metrics.average_memory_usage <= 0.5
    
    @pytest.mark.asyncio
    async def test_dynamic_scaling(self, orchestrator, mock_nodes):
        """Test dynamic cluster scaling based on load."""
        # Setup initial cluster
        for node in mock_nodes[:2]:
            await orchestrator.register_node(node)
        
        # Simulate high load requiring scale-up
        with patch.object(orchestrator, '_should_scale_up') as mock_scale_up:
            mock_scale_up.return_value = True
            
            with patch.object(orchestrator, '_provision_new_node') as mock_provision:
                mock_provision.return_value = mock_nodes[2]
                
                scaled_up = await orchestrator.auto_scale()
                
                assert scaled_up
                mock_provision.assert_called_once()
        
        # Simulate low load requiring scale-down
        with patch.object(orchestrator, '_should_scale_down') as mock_scale_down:
            mock_scale_down.return_value = True
            
            with patch.object(orchestrator, '_decommission_node') as mock_decommission:
                mock_decommission.return_value = True
                
                scaled_down = await orchestrator.auto_scale()
                
                assert scaled_down
                mock_decommission.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consensus_protocol(self, orchestrator, mock_nodes):
        """Test consensus protocol for distributed decisions."""
        # Setup cluster
        for node in mock_nodes[:5]:
            await orchestrator.register_node(node)
        
        consensus = orchestrator.consensus_protocol
        
        # Test consensus on cluster configuration change
        proposal = {
            "type": "config_change",
            "changes": {"max_concurrent_tasks": 10},
            "proposer": mock_nodes[0].node_id
        }
        
        with patch.object(consensus, '_collect_votes') as mock_votes:
            # Simulate majority agreement (3 out of 5 nodes)
            mock_votes.return_value = {
                mock_nodes[0].node_id: True,
                mock_nodes[1].node_id: True,
                mock_nodes[2].node_id: True,
                mock_nodes[3].node_id: False,
                mock_nodes[4].node_id: False
            }
            
            result = await consensus.propose_and_vote(proposal)
            
            assert result.approved
            assert result.vote_count == 3
            assert result.total_nodes == 5
    
    @pytest.mark.asyncio
    async def test_data_synchronization(self, orchestrator, mock_nodes):
        """Test data synchronization across nodes."""
        # Setup cluster
        for node in mock_nodes[:3]:
            await orchestrator.register_node(node)
        
        sync_manager = orchestrator.synchronization_manager
        
        # Test state synchronization
        state_update = {
            "type": "memory_update",
            "data": {"key": "value", "timestamp": datetime.now().isoformat()},
            "version": 1
        }
        
        with patch.object(sync_manager, '_replicate_state') as mock_replicate:
            mock_replicate.return_value = True
            
            success = await sync_manager.synchronize_state(state_update)
            
            assert success
            mock_replicate.assert_called_once()
        
        # Test conflict resolution
        conflicting_updates = [
            {"key": "value1", "timestamp": "2023-01-01T10:00:00", "node": "node_0"},
            {"key": "value2", "timestamp": "2023-01-01T10:01:00", "node": "node_1"}
        ]
        
        resolved_state = sync_manager.resolve_conflicts(conflicting_updates)
        
        # Should keep the most recent update
        assert resolved_state["key"] == "value2"
        assert resolved_state["node"] == "node_1"


class TestMCPNode:
    """Test suite for individual MCP Node functionality."""
    
    @pytest.fixture
    def mcp_node(self):
        """Create MCPNode instance for testing."""
        node_config = {
            "node_id": "test_node",
            "host": "localhost",
            "port": 8080,
            "capabilities": ["compute", "storage"],
            "max_concurrent_tasks": 5
        }
        return MCPNode(config=node_config)
    
    def test_mcp_node_initialization(self, mcp_node):
        """Test MCPNode initialization."""
        assert mcp_node.node_id == "test_node"
        assert mcp_node.host == "localhost"
        assert mcp_node.port == 8080
        assert "compute" in mcp_node.capabilities
        assert "storage" in mcp_node.capabilities
        assert mcp_node.max_concurrent_tasks == 5
        assert mcp_node.current_tasks == 0
        assert mcp_node.state == NodeState.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_node_startup_and_shutdown(self, mcp_node):
        """Test node startup and shutdown procedures."""
        # Test startup
        with patch.object(mcp_node, '_initialize_services') as mock_init:
            mock_init.return_value = True
            
            await mcp_node.start()
            
            assert mcp_node.state == NodeState.ACTIVE
            assert mcp_node.is_running
            mock_init.assert_called_once()
        
        # Test shutdown
        with patch.object(mcp_node, '_cleanup_services') as mock_cleanup:
            mock_cleanup.return_value = True
            
            await mcp_node.stop()
            
            assert mcp_node.state == NodeState.DISCONNECTED
            assert not mcp_node.is_running
            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_execution(self, mcp_node):
        """Test task execution on node."""
        await mcp_node.start()
        
        task = DistributedTask(
            task_id="node_test_task",
            task_type="compute",
            payload={"operation": "add", "a": 5, "b": 3},
            requirements={}
        )
        
        with patch.object(mcp_node, '_execute_task') as mock_execute:
            mock_execute.return_value = {"result": 8, "status": "completed"}
            
            result = await mcp_node.execute_task(task)
            
            assert result["result"] == 8
            assert result["status"] == "completed"
            mock_execute.assert_called_once_with(task)
    
    def test_resource_monitoring(self, mcp_node):
        """Test node resource monitoring."""
        # Mock system metrics
        with patch.object(mcp_node, '_get_system_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "cpu_usage": 0.45,
                "memory_usage": 0.67,
                "disk_usage": 0.23,
                "network_io": 1500
            }
            
            metrics = mcp_node.get_resource_metrics()
            
            assert metrics["cpu_usage"] == 0.45
            assert metrics["memory_usage"] == 0.67
            assert metrics["disk_usage"] == 0.23
            assert metrics["network_io"] == 1500
    
    @pytest.mark.asyncio
    async def test_health_check(self, mcp_node):
        """Test node health check functionality."""
        await mcp_node.start()
        
        # Test healthy node
        health_status = await mcp_node.health_check()
        
        assert health_status["status"] == "healthy"
        assert "uptime" in health_status
        assert "resource_usage" in health_status
        
        # Test unhealthy node (high resource usage)
        with patch.object(mcp_node, 'get_resource_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "cpu_usage": 0.95,  # Very high CPU usage
                "memory_usage": 0.98,  # Very high memory usage
                "disk_usage": 0.23,
                "network_io": 1500
            }
            
            health_status = await mcp_node.health_check()
            
            assert health_status["status"] == "unhealthy"
            assert "high_cpu_usage" in health_status["issues"]
            assert "high_memory_usage" in health_status["issues"]


class TestTaskScheduler:
    """Test suite for Task Scheduler functionality."""
    
    @pytest.fixture
    def task_scheduler(self):
        """Create TaskScheduler instance for testing."""
        return TaskScheduler()
    
    @pytest.fixture
    def mock_nodes(self):
        """Create mock nodes for testing."""
        return [
            MockNodeInfo(
                node_id=f"node_{i}",
                host=f"192.168.1.{10+i}",
                port=8000 + i,
                capabilities=["compute", "storage"] if i % 2 == 0 else ["compute"],
                max_concurrent_tasks=5,
                current_load=0.2 * i
            )
            for i in range(5)
        ]
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            DistributedTask(
                task_id=f"task_{i}",
                task_type="compute",
                payload={"size": i * 100},
                requirements={"cpu_cores": i + 1, "memory_gb": (i + 1) * 2},
                priority=i % 3,  # Priorities 0, 1, 2
                timeout=300.0
            )
            for i in range(5)
        ]
    
    def test_task_queue_management(self, task_scheduler, sample_tasks):
        """Test task queue management and prioritization."""
        # Add tasks to queue
        for task in sample_tasks:
            task_scheduler.enqueue_task(task)
        
        assert task_scheduler.queue_size() == 5
        
        # Tasks should be ordered by priority (higher priority first)
        next_task = task_scheduler.get_next_task()
        assert next_task.priority == 2  # Highest priority
        
        # Remove task from queue
        task_scheduler.dequeue_task(next_task.task_id)
        assert task_scheduler.queue_size() == 4
    
    def test_task_scheduling_algorithms(self, task_scheduler, sample_tasks, mock_nodes):
        """Test different task scheduling algorithms."""
        # Setup available nodes
        available_nodes = mock_nodes[:3]
        
        # Test FIFO scheduling
        task_scheduler.set_algorithm("fifo")
        scheduled_tasks = []
        
        for task in sample_tasks:
            node_id = task_scheduler.schedule_task(task, available_nodes)
            scheduled_tasks.append((task.task_id, node_id))
        
        assert len(scheduled_tasks) == 5
        
        # Test priority-based scheduling
        task_scheduler.set_algorithm("priority")
        task_scheduler.clear_queue()
        
        for task in sample_tasks:
            task_scheduler.enqueue_task(task)
        
        # Get tasks in priority order
        priority_order = []
        while not task_scheduler.is_empty():
            task = task_scheduler.get_next_task()
            priority_order.append(task.priority)
            task_scheduler.dequeue_task(task.task_id)
        
        # Should be in descending priority order
        assert priority_order == sorted(priority_order, reverse=True)
    
    def test_resource_aware_scheduling(self, task_scheduler, mock_nodes):
        """Test resource-aware task scheduling."""
        # Create task with specific resource requirements
        resource_intensive_task = DistributedTask(
            task_id="resource_task",
            task_type="compute",
            payload={},
            requirements={"cpu_cores": 8, "memory_gb": 16, "gpu": True},
            priority=1
        )
        
        # Mock node capabilities
        for i, node in enumerate(mock_nodes):
            node.capabilities = ["compute"]
            if i == 2:  # Only node_2 has GPU
                node.capabilities.append("gpu")
        
        # Schedule task
        selected_node = task_scheduler.schedule_task(resource_intensive_task, mock_nodes)
        
        # Should select node with GPU capability
        assert selected_node == mock_nodes[2].node_id
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, task_scheduler):
        """Test task timeout detection and handling."""
        # Create task with short timeout
        timeout_task = DistributedTask(
            task_id="timeout_task",
            task_type="compute",
            payload={},
            requirements={},
            timeout=0.1  # 100ms timeout
        )
        
        # Start task
        task_scheduler.start_task(timeout_task, "test_node")
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Check if task timed out
        timed_out_tasks = task_scheduler.get_timed_out_tasks()
        assert timeout_task.task_id in [task.task_id for task in timed_out_tasks]
    
    def test_task_dependency_management(self, task_scheduler):
        """Test task dependency resolution and scheduling."""
        # Create tasks with dependencies
        task_a = DistributedTask(
            task_id="task_a",
            task_type="compute",
            payload={},
            requirements={},
            dependencies=[]
        )
        
        task_b = DistributedTask(
            task_id="task_b", 
            task_type="compute",
            payload={},
            requirements={},
            dependencies=["task_a"]
        )
        
        task_c = DistributedTask(
            task_id="task_c",
            task_type="compute", 
            payload={},
            requirements={},
            dependencies=["task_a", "task_b"]
        )
        
        # Add tasks to scheduler
        task_scheduler.add_task_with_dependencies(task_c)
        task_scheduler.add_task_with_dependencies(task_b)
        task_scheduler.add_task_with_dependencies(task_a)
        
        # Get execution order
        execution_order = task_scheduler.get_execution_order()
        
        # task_a should come first, then task_b, then task_c
        assert execution_order.index("task_a") < execution_order.index("task_b")
        assert execution_order.index("task_b") < execution_order.index("task_c")


class TestLoadBalancer:
    """Test suite for Load Balancer functionality."""
    
    @pytest.fixture
    def load_balancer(self):
        """Create LoadBalancer instance for testing."""
        return LoadBalancer()
    
    @pytest.fixture
    def mock_nodes(self):
        """Create mock nodes for testing."""
        return [
            MockNodeInfo(
                node_id=f"node_{i}",
                host=f"192.168.1.{10+i}",
                port=8000 + i,
                capabilities=["compute", "storage"] if i % 2 == 0 else ["compute"],
                max_concurrent_tasks=5,
                current_load=0.2 * i
            )
            for i in range(5)
        ]
    
    def test_round_robin_balancing(self, load_balancer, mock_nodes):
        """Test round-robin load balancing strategy."""
        load_balancer.set_strategy("round_robin")
        
        # Make multiple selections
        selections = []
        for i in range(10):
            task = DistributedTask(
                task_id=f"rr_task_{i}",
                task_type="compute",
                payload={},
                requirements={}
            )
            selected_node = load_balancer.select_node(task, mock_nodes[:3])
            selections.append(selected_node)
        
        # Should cycle through nodes
        expected_pattern = [node.node_id for node in mock_nodes[:3]] * 4  # 3 nodes * 3 cycles + 1
        assert selections == expected_pattern[:10]
    
    def test_least_loaded_balancing(self, load_balancer, mock_nodes):
        """Test least-loaded load balancing strategy."""
        load_balancer.set_strategy("least_loaded")
        
        # Set different loads on nodes
        for i, node in enumerate(mock_nodes[:3]):
            node.current_load = 0.1 * (i + 1)  # 0.1, 0.2, 0.3
        
        task = DistributedTask(
            task_id="ll_task",
            task_type="compute",
            payload={},
            requirements={}
        )
        
        selected_node = load_balancer.select_node(task, mock_nodes[:3])
        
        # Should select node with lowest load (node_0)
        assert selected_node == mock_nodes[0].node_id
    
    def test_weighted_balancing(self, load_balancer, mock_nodes):
        """Test weighted load balancing strategy."""
        load_balancer.set_strategy("weighted")
        
        # Set weights for nodes
        weights = {
            mock_nodes[0].node_id: 1.0,
            mock_nodes[1].node_id: 2.0,  # Higher weight
            mock_nodes[2].node_id: 1.0
        }
        load_balancer.set_node_weights(weights)
        
        # Make many selections to test distribution
        selections = []
        for i in range(100):
            task = DistributedTask(
                task_id=f"weighted_task_{i}",
                task_type="compute",
                payload={},
                requirements={}
            )
            selected_node = load_balancer.select_node(task, mock_nodes[:3])
            selections.append(selected_node)
        
        # Count selections per node
        selection_counts = {}
        for node_id in selections:
            selection_counts[node_id] = selection_counts.get(node_id, 0) + 1
        
        # Node 1 should have approximately twice as many selections
        node_1_ratio = selection_counts[mock_nodes[1].node_id] / len(selections)
        assert 0.4 <= node_1_ratio <= 0.6  # Should be around 50%
    
    def test_load_balancer_metrics(self, load_balancer, mock_nodes):
        """Test load balancer metrics and statistics."""
        load_balancer.set_strategy("round_robin")
        
        # Simulate task assignments
        for i in range(20):
            task = DistributedTask(
                task_id=f"metrics_task_{i}",
                task_type="compute",
                payload={},
                requirements={}
            )
            load_balancer.select_node(task, mock_nodes[:3])
        
        metrics = load_balancer.get_metrics()
        
        assert "total_assignments" in metrics
        assert "assignments_per_node" in metrics
        assert "load_distribution" in metrics
        
        assert metrics["total_assignments"] == 20
        
        # Each node should have approximately equal assignments (round-robin)
        for node in mock_nodes[:3]:
            node_assignments = metrics["assignments_per_node"].get(node.node_id, 0)
            assert 6 <= node_assignments <= 7  # Should be around 20/3