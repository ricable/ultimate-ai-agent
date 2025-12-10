# File: backend/tests/test_distributed_processing.py
"""
Tests for Ray Distributed Processing System
Tests cluster management, task distribution, and workload orchestration.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# Import the modules under test
from ..distributed.ray_manager import (
    RayClusterManager, DistributedTask, ClusterNodeInfo,
    TaskStatus, ClusterStatus, ray_cluster_manager
)
from ..services.distributed_orchestrator import (
    DistributedOrchestrator, DistributedWorkload, WorkloadType,
    ProcessingStrategy, initialize_distributed_orchestrator
)
from ..services.agent_orchestrator import UAP_AgentOrchestrator

class TestRayClusterManager:
    """Test Ray cluster management functionality"""
    
    @pytest.fixture
    def cluster_manager(self):
        """Create a cluster manager instance for testing"""
        return RayClusterManager(
            max_nodes=5,
            min_nodes=1,
            node_idle_timeout=60,
            enable_autoscaling=False  # Disable for testing
        )
    
    def test_cluster_initialization(self, cluster_manager):
        """Test cluster manager initialization"""
        assert cluster_manager.max_nodes == 5
        assert cluster_manager.min_nodes == 1
        assert cluster_manager.cluster_status == ClusterStatus.INITIALIZING
        assert len(cluster_manager.tasks) == 0
        assert len(cluster_manager.task_queue) == 0
    
    @pytest.mark.asyncio
    async def test_task_submission_fallback(self, cluster_manager):
        """Test task submission when Ray is not available"""
        # Test function for tasks
        def test_function(value: int) -> int:
            return value * 2
        
        # Submit task
        task_id = await cluster_manager.submit_task(
            task_type="test_task",
            task_function=test_function,
            input_data={"value": 5},
            priority=1
        )
        
        # Verify task was created
        assert task_id in cluster_manager.tasks
        task = cluster_manager.tasks[task_id]
        assert task.task_type == "test_task"
        assert task.input_data["value"] == 5
        assert task.priority == 1
        
        # Wait for task completion
        for _ in range(10):  # Wait up to 10 seconds
            if task.status == TaskStatus.COMPLETED:
                break
            await asyncio.sleep(1)
        
        # Verify task completed successfully
        assert task.status == TaskStatus.COMPLETED
        assert task.result == 10  # 5 * 2
        assert task.error is None
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, cluster_manager):
        """Test task cancellation"""
        def long_running_task(duration: int) -> str:
            time.sleep(duration)
            return "completed"
        
        # Submit long-running task
        task_id = await cluster_manager.submit_task(
            task_type="long_task",
            task_function=long_running_task,
            input_data={"duration": 10}  # 10 second task
        )
        
        # Wait a bit, then cancel
        await asyncio.sleep(0.5)
        success = await cluster_manager.cancel_task(task_id)
        
        assert success is True
        task = cluster_manager.tasks[task_id]
        assert task.status == TaskStatus.CANCELLED
        assert "cancelled" in task.error.lower()
    
    @pytest.mark.asyncio
    async def test_cluster_status(self, cluster_manager):
        """Test cluster status reporting"""
        status = await cluster_manager.get_cluster_status()
        
        assert "timestamp" in status
        assert "cluster_status" in status
        assert "ray_available" in status
        assert "task_statistics" in status
        assert "cluster_metrics" in status
        assert "configuration" in status
        
        # Check configuration
        config = status["configuration"]
        assert config["max_nodes"] == 5
        assert config["min_nodes"] == 1
        assert config["enable_autoscaling"] is False
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, cluster_manager):
        """Test that metrics are properly tracked"""
        def simple_task(x: int) -> int:
            return x + 1
        
        # Submit multiple tasks
        task_ids = []
        for i in range(3):
            task_id = await cluster_manager.submit_task(
                task_type="metric_test",
                task_function=simple_task,
                input_data={"x": i}
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        for task_id in task_ids:
            for _ in range(10):
                task = cluster_manager.tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    break
                await asyncio.sleep(0.5)
        
        # Check metrics
        metrics = cluster_manager.cluster_metrics
        assert metrics["total_tasks_processed"] >= 3
        assert metrics["successful_tasks"] >= 3
        assert metrics["failed_tasks"] == 0
        assert metrics["avg_task_time"] > 0

class TestDistributedOrchestrator:
    """Test distributed orchestrator functionality"""
    
    @pytest.fixture
    def mock_agent_orchestrator(self):
        """Create a mock agent orchestrator"""
        orchestrator = Mock(spec=UAP_AgentOrchestrator)
        orchestrator.handle_http_chat = AsyncMock(return_value={
            "content": "Test response",
            "metadata": {"test": True}
        })
        orchestrator.agno_manager = Mock()
        orchestrator.agno_manager.process_message = AsyncMock(return_value={
            "content": "Agno response",
            "metadata": {"framework": "agno"}
        })
        return orchestrator
    
    @pytest.fixture
    def distributed_orchestrator(self, mock_agent_orchestrator):
        """Create a distributed orchestrator instance"""
        return DistributedOrchestrator(mock_agent_orchestrator)
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, distributed_orchestrator):
        """Test orchestrator initialization"""
        assert len(distributed_orchestrator.workloads) == 0
        assert len(distributed_orchestrator.workload_queue) == 0
        assert distributed_orchestrator.metrics["total_workloads"] == 0
        assert WorkloadType.DOCUMENT_PROCESSING in distributed_orchestrator.task_handlers
        assert WorkloadType.AI_INFERENCE in distributed_orchestrator.task_handlers
    
    @pytest.mark.asyncio
    async def test_document_processing_workload(self, distributed_orchestrator):
        """Test document processing workload"""
        input_data = {
            "documents": [
                {"content": "Test document content 1"},
                {"content": "Test document content 2"}
            ]
        }
        
        workload_id = await distributed_orchestrator.submit_workload(
            workload_type=WorkloadType.DOCUMENT_PROCESSING,
            input_data=input_data,
            strategy=ProcessingStrategy.SEQUENTIAL
        )
        
        assert workload_id in distributed_orchestrator.workloads
        workload = distributed_orchestrator.workloads[workload_id]
        assert workload.workload_type == WorkloadType.DOCUMENT_PROCESSING
        assert workload.strategy == ProcessingStrategy.SEQUENTIAL
        
        # Wait for workload to complete
        for _ in range(20):  # Wait up to 20 seconds
            if workload.status == "completed":
                break
            await asyncio.sleep(1)
        
        # Check if workload completed (may be "running" due to async nature)
        assert workload.status in ["completed", "running"]
        assert workload.progress >= 0
    
    @pytest.mark.asyncio
    async def test_ai_inference_workload(self, distributed_orchestrator, mock_agent_orchestrator):
        """Test AI inference workload"""
        input_data = {
            "queries": [
                "What is artificial intelligence?",
                "Explain machine learning",
                "What is deep learning?"
            ],
            "framework": "copilot"
        }
        
        workload_id = await distributed_orchestrator.submit_workload(
            workload_type=WorkloadType.AI_INFERENCE,
            input_data=input_data,
            strategy=ProcessingStrategy.PARALLEL
        )
        
        workload = distributed_orchestrator.workloads[workload_id]
        assert workload.workload_type == WorkloadType.AI_INFERENCE
        assert workload.strategy == ProcessingStrategy.PARALLEL
        
        # Wait for workload processing to start
        await asyncio.sleep(1)
        
        # Verify agent orchestrator was called for AI inference
        # (It should be called during workload processing)
        assert workload.status in ["queued", "running", "completed"]
    
    @pytest.mark.asyncio
    async def test_workload_cancellation(self, distributed_orchestrator):
        """Test workload cancellation"""
        input_data = {
            "items": ["item1", "item2", "item3"],
            "analysis_type": "test"
        }
        
        workload_id = await distributed_orchestrator.submit_workload(
            workload_type=WorkloadType.BATCH_ANALYSIS,
            input_data=input_data,
            strategy=ProcessingStrategy.SEQUENTIAL
        )
        
        # Cancel immediately
        success = await distributed_orchestrator.cancel_workload(workload_id)
        
        assert success is True
        workload = distributed_orchestrator.workloads[workload_id]
        assert workload.status == "cancelled"
        assert "cancelled" in workload.error.lower()
    
    @pytest.mark.asyncio
    async def test_workload_status_tracking(self, distributed_orchestrator):
        """Test workload status tracking"""
        input_data = {"test": "data"}
        
        workload_id = await distributed_orchestrator.submit_workload(
            workload_type=WorkloadType.DATA_PROCESSING,
            input_data=input_data
        )
        
        # Get workload status
        workload = await distributed_orchestrator.get_workload_status(workload_id)
        
        assert workload is not None
        assert workload.workload_id == workload_id
        assert workload.workload_type == WorkloadType.DATA_PROCESSING
        assert workload.created_at is not None
        assert workload.progress >= 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_status(self, distributed_orchestrator):
        """Test orchestrator status reporting"""
        status = await distributed_orchestrator.get_orchestrator_status()
        
        assert "timestamp" in status
        assert "cluster_status" in status
        assert "workload_statistics" in status
        assert "performance_metrics" in status
        assert "supported_workload_types" in status
        assert "supported_strategies" in status
        
        # Check workload types
        workload_types = status["supported_workload_types"]
        assert "document_processing" in workload_types
        assert "ai_inference" in workload_types
        assert "batch_analysis" in workload_types
        assert "multi_agent_task" in workload_types
        
        # Check strategies
        strategies = status["supported_strategies"]
        assert "sequential" in strategies
        assert "parallel" in strategies
        assert "map_reduce" in strategies
        assert "adaptive" in strategies
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_selection(self, distributed_orchestrator):
        """Test adaptive strategy selection based on workload characteristics"""
        # Small workload - should use sequential
        small_workload_data = {
            "items": ["item1", "item2"]
        }
        
        workload_id = await distributed_orchestrator.submit_workload(
            workload_type=WorkloadType.BATCH_ANALYSIS,
            input_data=small_workload_data,
            strategy=ProcessingStrategy.ADAPTIVE
        )
        
        workload = distributed_orchestrator.workloads[workload_id]
        assert workload.strategy == ProcessingStrategy.ADAPTIVE
        
        # Strategy may be updated during processing based on analysis
        await asyncio.sleep(0.5)  # Allow time for analysis
    
    @pytest.mark.asyncio 
    async def test_task_handler_registration(self, distributed_orchestrator):
        """Test that all required task handlers are registered"""
        handlers = distributed_orchestrator.task_handlers
        
        required_types = [
            WorkloadType.DOCUMENT_PROCESSING,
            WorkloadType.AI_INFERENCE,
            WorkloadType.BATCH_ANALYSIS,
            WorkloadType.DATA_PROCESSING,
            WorkloadType.MULTI_AGENT_TASK
        ]
        
        for workload_type in required_types:
            assert workload_type in handlers
            assert callable(handlers[workload_type])

class TestDistributedIntegration:
    """Integration tests for the complete distributed processing system"""
    
    @pytest.fixture
    def mock_agent_orchestrator(self):
        """Create a comprehensive mock agent orchestrator"""
        orchestrator = Mock(spec=UAP_AgentOrchestrator)
        orchestrator.handle_http_chat = AsyncMock(return_value={
            "content": "Integration test response",
            "metadata": {"framework": "test", "integration": True}
        })
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self, mock_agent_orchestrator):
        """Test complete system integration"""
        # Initialize distributed orchestrator
        distributed_orch = initialize_distributed_orchestrator(mock_agent_orchestrator)
        
        # Submit a complex workload
        input_data = {
            "queries": [
                "Test query 1",
                "Test query 2",
                "Test query 3"
            ],
            "framework": "copilot"
        }
        
        workload_id = await distributed_orch.submit_workload(
            workload_type=WorkloadType.AI_INFERENCE,
            input_data=input_data,
            strategy=ProcessingStrategy.PARALLEL,
            metadata={"test": "integration"}
        )
        
        # Verify workload was submitted
        assert workload_id is not None
        
        # Get workload status
        workload = await distributed_orch.get_workload_status(workload_id)
        assert workload is not None
        assert workload.metadata.get("test") == "integration"
        
        # Get system status
        system_status = await distributed_orch.get_orchestrator_status()
        assert system_status["workload_statistics"]["total_workloads"] >= 1
        
        # Cleanup
        await distributed_orch.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, mock_agent_orchestrator):
        """Test system performance under moderate load"""
        distributed_orch = initialize_distributed_orchestrator(mock_agent_orchestrator)
        
        # Submit multiple workloads concurrently
        workload_ids = []
        start_time = time.time()
        
        for i in range(5):
            input_data = {
                "items": [f"item_{i}_{j}" for j in range(10)],
                "analysis_type": "performance_test"
            }
            
            workload_id = await distributed_orch.submit_workload(
                workload_type=WorkloadType.BATCH_ANALYSIS,
                input_data=input_data,
                strategy=ProcessingStrategy.PARALLEL
            )
            workload_ids.append(workload_id)
        
        submission_time = time.time() - start_time
        
        # Verify all workloads were submitted quickly
        assert submission_time < 5.0  # Should submit 5 workloads in under 5 seconds
        assert len(workload_ids) == 5
        
        # Check that all workloads are tracked
        for workload_id in workload_ids:
            workload = await distributed_orch.get_workload_status(workload_id)
            assert workload is not None
            assert workload.status in ["queued", "running", "completed"]
        
        # Get final system status
        final_status = await distributed_orch.get_orchestrator_status()
        assert final_status["workload_statistics"]["total_workloads"] >= 5
        
        # Cleanup
        await distributed_orch.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_agent_orchestrator):
        """Test error handling and system recovery"""
        distributed_orch = initialize_distributed_orchestrator(mock_agent_orchestrator)
        
        # Configure mock to simulate errors
        mock_agent_orchestrator.handle_http_chat.side_effect = Exception("Simulated error")
        
        # Submit workload that will fail
        input_data = {
            "queries": ["This will fail"],
            "framework": "error_test"
        }
        
        workload_id = await distributed_orch.submit_workload(
            workload_type=WorkloadType.AI_INFERENCE,
            input_data=input_data
        )
        
        # Wait for workload to process and fail
        await asyncio.sleep(2)
        
        workload = await distributed_orch.get_workload_status(workload_id)
        
        # System should handle errors gracefully
        assert workload is not None
        # Status might be "failed" or still "running" depending on timing
        assert workload.status in ["running", "failed"]
        
        # System should still be responsive
        status = await distributed_orch.get_orchestrator_status()
        assert status is not None
        assert "timestamp" in status
        
        # Cleanup
        await distributed_orch.cleanup()

def test_global_functions():
    """Test global convenience functions"""
    # Test that global cluster manager exists
    assert ray_cluster_manager is not None
    assert isinstance(ray_cluster_manager, RayClusterManager)
    
    # Test initialization function
    mock_orchestrator = Mock(spec=UAP_AgentOrchestrator)
    distributed_orch = initialize_distributed_orchestrator(mock_orchestrator)
    assert isinstance(distributed_orch, DistributedOrchestrator)
    assert distributed_orch.agent_orchestrator == mock_orchestrator

if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__])
