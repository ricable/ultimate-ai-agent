# File: backend/distributed/ray_manager.py
"""
Ray Cluster Manager for UAP Distributed Processing
Provides auto-scaling, cluster management, and distributed ML workload orchestration.
"""

import asyncio
import os
import time
import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import psutil
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# Ray imports with fallback
try:
    import ray
    from ray import serve
    from ray.serve import Application
    from ray.cluster_utils import Cluster
    from ray.exceptions import RaySystemError, RayTaskError
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available. Distributed processing will use fallback mode.")

# Import monitoring components
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.performance import performance_monitor
from ..monitoring.metrics.prometheus_metrics import (
    record_agent_request, record_ray_task, update_ray_cluster_metrics
)

class ClusterStatus(Enum):
    """Ray cluster status states"""
    INITIALIZING = "initializing"
    READY = "ready"
    SCALING = "scaling"
    ERROR = "error"
    TERMINATED = "terminated"

class TaskStatus(Enum):
    """Distributed task status states"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ClusterNodeInfo:
    """Information about a Ray cluster node"""
    node_id: str
    node_ip: str
    resources: Dict[str, float]
    alive: bool
    last_heartbeat: datetime
    cpu_percent: float
    memory_percent: float
    gpu_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data

@dataclass
class DistributedTask:
    """Distributed task information"""
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    node_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

class RayClusterManager:
    """
    Ray Cluster Manager for auto-scaling distributed ML workloads.
    Manages cluster lifecycle, resource allocation, and job distribution.
    """
    
    def __init__(self, 
                 max_nodes: int = 10,
                 min_nodes: int = 1,
                 node_idle_timeout: int = 300,
                 task_timeout: int = 3600,
                 enable_autoscaling: bool = True):
        """
        Initialize Ray cluster manager.
        
        Args:
            max_nodes: Maximum number of nodes in cluster
            min_nodes: Minimum number of nodes in cluster
            node_idle_timeout: Seconds before idle nodes are terminated
            task_timeout: Maximum seconds for task execution
            enable_autoscaling: Whether to enable automatic scaling
        """
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.node_idle_timeout = node_idle_timeout
        self.task_timeout = task_timeout
        self.enable_autoscaling = enable_autoscaling
        
        self.logger = logging.getLogger(__name__)
        self.cluster_status = ClusterStatus.INITIALIZING
        self.nodes: Dict[str, ClusterNodeInfo] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue: List[str] = []  # Task IDs in execution order
        
        # Performance tracking
        self.cluster_metrics = {
            'total_tasks_processed': 0,
            'total_task_time': 0.0,
            'avg_task_time': 0.0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'cluster_uptime': 0.0
        }
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.monitoring_active = False
        self.cluster_start_time = None
        
        # Initialize Ray if available
        self.ray_initialized = False
        if RAY_AVAILABLE:
            self._initialize_ray_cluster()
    
    def _initialize_ray_cluster(self) -> bool:
        """
        Initialize Ray cluster with appropriate configuration.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check if Ray is already initialized
            if ray.is_initialized():
                self.logger.info("Ray cluster already initialized")
                self.ray_initialized = True
                self.cluster_status = ClusterStatus.READY
                self.cluster_start_time = datetime.utcnow()
                return True
            
            # Initialize Ray with cluster configuration
            ray_config = {
                'num_cpus': psutil.cpu_count(),
                'include_dashboard': True,
                'dashboard_host': '0.0.0.0',
                'dashboard_port': 8265,
                'log_to_driver': True,
                'ignore_reinit_error': True
            }
            
            # Add GPU support if available
            try:
                import GPUtil
                gpu_count = len(GPUtil.getGPUs())
                if gpu_count > 0:
                    ray_config['num_gpus'] = gpu_count
                    self.logger.info(f"Found {gpu_count} GPUs for Ray cluster")
            except ImportError:
                pass
            
            # Initialize Ray
            ray.init(**ray_config)
            
            self.ray_initialized = True
            self.cluster_status = ClusterStatus.READY
            self.cluster_start_time = datetime.utcnow()
            
            self.logger.info("Ray cluster initialized successfully")
            
            # Start monitoring
            asyncio.create_task(self._start_cluster_monitoring())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ray cluster: {str(e)}")
            self.cluster_status = ClusterStatus.ERROR
            return False
    
    async def submit_task(self, 
                         task_type: str, 
                         task_function: Callable,
                         input_data: Dict[str, Any],
                         priority: int = 0,
                         timeout: Optional[int] = None) -> str:
        """
        Submit a distributed task to the cluster.
        
        Args:
            task_type: Type/category of the task
            task_function: Function to execute
            input_data: Input data for the task
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        # Create task record
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.QUEUED,
            created_at=datetime.utcnow(),
            input_data=input_data,
            priority=priority
        )
        
        self.tasks[task_id] = task
        
        # Add to queue (sorted by priority)
        self.task_queue.append(task_id)
        self.task_queue.sort(key=lambda tid: self.tasks[tid].priority, reverse=True)
        
        # Log task submission
        uap_logger.log_event(
            LogLevel.INFO,
            f"Distributed task submitted: {task_type}",
            EventType.AGENT,
            {
                "task_id": task_id,
                "task_type": task_type,
                "priority": priority,
                "queue_size": len(self.task_queue)
            },
            "ray_cluster"
        )
        
        # Execute task if Ray is available, otherwise use fallback
        if self.ray_initialized and RAY_AVAILABLE:
            await self._execute_ray_task(task_id, task_function, timeout)
        else:
            await self._execute_fallback_task(task_id, task_function)
        
        return task_id
    
    async def _execute_ray_task(self, task_id: str, task_function: Callable, timeout: Optional[int]):
        """
        Execute task using Ray distributed processing.
        
        Args:
            task_id: Task ID
            task_function: Function to execute
            timeout: Task timeout in seconds
        """
        task = self.tasks[task_id]
        
        try:
            # Mark task as running
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Create Ray remote function
            @ray.remote
            def remote_task_wrapper(input_data):
                return task_function(**input_data)
            
            # Submit to Ray cluster
            future = remote_task_wrapper.remote(task.input_data)
            
            # Wait for completion with timeout
            timeout_seconds = timeout or self.task_timeout
            try:
                result = ray.get(future, timeout=timeout_seconds)
                
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.result = result
                
                # Update metrics
                self.cluster_metrics['successful_tasks'] += 1
                self.cluster_metrics['total_tasks_processed'] += 1
                
                execution_time = (task.completed_at - task.started_at).total_seconds()
                self.cluster_metrics['total_task_time'] += execution_time
                self.cluster_metrics['avg_task_time'] = (
                    self.cluster_metrics['total_task_time'] / 
                    self.cluster_metrics['total_tasks_processed']
                )
                
                # Record Prometheus metrics
                record_ray_task(task.task_type, "completed", execution_time)
                
                self.logger.info(f"Ray task {task_id} completed successfully in {execution_time:.2f}s")
                
            except ray.exceptions.GetTimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"Task timeout after {timeout_seconds} seconds"
                task.completed_at = datetime.utcnow()
                self.cluster_metrics['failed_tasks'] += 1
                
                # Record Prometheus metrics
                record_ray_task(task.task_type, "timeout")
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            self.cluster_metrics['failed_tasks'] += 1
            
            # Record Prometheus metrics
            record_ray_task(task.task_type, "failed")
            
            self.logger.error(f"Ray task {task_id} failed: {str(e)}")
        
        finally:
            # Remove from queue
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
    
    async def _execute_fallback_task(self, task_id: str, task_function: Callable):
        """
        Execute task using local processing (fallback when Ray not available).
        
        Args:
            task_id: Task ID
            task_function: Function to execute
        """
        task = self.tasks[task_id]
        
        try:
            # Mark task as running
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Execute task in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: task_function(**task.input_data)
            )
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result
            
            # Update metrics
            self.cluster_metrics['successful_tasks'] += 1
            self.cluster_metrics['total_tasks_processed'] += 1
            
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.cluster_metrics['total_task_time'] += execution_time
            self.cluster_metrics['avg_task_time'] = (
                self.cluster_metrics['total_task_time'] / 
                self.cluster_metrics['total_tasks_processed']
            )
            
            self.logger.info(f"Fallback task {task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            self.cluster_metrics['failed_tasks'] += 1
            
            self.logger.error(f"Fallback task {task_id} failed: {str(e)}")
        
        finally:
            # Remove from queue
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """
        Get status of a specific task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            DistributedTask object or None if not found
        """
        return self.tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued or running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        task.error = "Task cancelled by user"
        
        # Record Prometheus metrics
        record_ray_task(task.task_type, "cancelled")
        
        # Remove from queue
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        
        self.logger.info(f"Task {task_id} cancelled")
        return True
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get comprehensive cluster status information.
        
        Returns:
            Cluster status dictionary
        """
        # Update cluster uptime
        if self.cluster_start_time:
            self.cluster_metrics['cluster_uptime'] = (
                datetime.utcnow() - self.cluster_start_time
            ).total_seconds()
        
        # Get Ray cluster info if available
        ray_info = {}
        if self.ray_initialized and RAY_AVAILABLE:
            try:
                cluster_resources = ray.cluster_resources()
                available_resources = ray.available_resources()
                
                ray_info = {
                    'total_resources': cluster_resources,
                    'available_resources': available_resources,
                    'nodes': len(ray.nodes()),
                    'dashboard_url': 'http://localhost:8265' if cluster_resources else None
                }
            except Exception as e:
                ray_info = {'error': str(e)}
        
        # Task statistics
        task_stats = {
            'total_tasks': len(self.tasks),
            'queued_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.QUEUED]),
            'running_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
            'completed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
            'queue_length': len(self.task_queue)
        }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cluster_status': self.cluster_status.value,
            'ray_available': RAY_AVAILABLE,
            'ray_initialized': self.ray_initialized,
            'ray_info': ray_info,
            'task_statistics': task_stats,
            'cluster_metrics': self.cluster_metrics,
            'configuration': {
                'max_nodes': self.max_nodes,
                'min_nodes': self.min_nodes,
                'node_idle_timeout': self.node_idle_timeout,
                'task_timeout': self.task_timeout,
                'enable_autoscaling': self.enable_autoscaling
            }
        }
    
    async def _start_cluster_monitoring(self):
        """
        Start background cluster monitoring.
        """
        self.monitoring_active = True
        
        while self.monitoring_active and self.ray_initialized:
            try:
                await self._update_cluster_info()
                await self._check_cluster_health()
                await self._cleanup_completed_tasks()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in cluster monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _update_cluster_info(self):
        """
        Update cluster node information.
        """
        if not (self.ray_initialized and RAY_AVAILABLE):
            return
        
        try:
            # Get current nodes from Ray
            ray_nodes = ray.nodes()
            current_node_ids = set()
            
            for node in ray_nodes:
                node_id = node['NodeID']
                current_node_ids.add(node_id)
                
                # Update or create node info
                self.nodes[node_id] = ClusterNodeInfo(
                    node_id=node_id,
                    node_ip=node.get('NodeManagerAddress', 'unknown'),
                    resources=node.get('Resources', {}),
                    alive=node.get('Alive', False),
                    last_heartbeat=datetime.utcnow(),
                    cpu_percent=0.0,  # Would need additional monitoring for actual values
                    memory_percent=0.0,
                    gpu_count=int(node.get('Resources', {}).get('GPU', 0))
                )
            
            # Remove nodes that are no longer present
            for node_id in list(self.nodes.keys()):
                if node_id not in current_node_ids:
                    del self.nodes[node_id]
            
            # Update Prometheus metrics
            try:
                cluster_resources = ray.cluster_resources()
                available_cluster_resources = ray.available_resources()
                
                # Convert to format expected by metrics
                total_resources = {}
                available_resources = {}
                
                for resource, value in cluster_resources.items():
                    total_resources[resource] = float(value)
                
                for resource, value in available_cluster_resources.items():
                    available_resources[resource] = float(value)
                
                update_ray_cluster_metrics(
                    nodes_count=len(current_node_ids),
                    total_resources=total_resources,
                    available_resources=available_resources,
                    queue_size=len(self.task_queue)
                )
            except Exception as metrics_error:
                self.logger.warning(f"Failed to update Ray cluster metrics: {metrics_error}")
                    
        except Exception as e:
            self.logger.error(f"Error updating cluster info: {str(e)}")
    
    async def _check_cluster_health(self):
        """
        Check cluster health and perform auto-scaling if needed.
        """
        if not self.enable_autoscaling:
            return
        
        try:
            # Check queue length and resource utilization
            queue_length = len(self.task_queue)
            running_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
            
            # Simple auto-scaling logic
            if queue_length > 10 and len(self.nodes) < self.max_nodes:
                self.logger.info(f"High queue length ({queue_length}), considering scale-up")
                # Note: Actual scaling would require integration with cloud providers
                
            elif running_tasks == 0 and len(self.nodes) > self.min_nodes:
                self.logger.info("No running tasks, considering scale-down")
                # Note: Actual scaling would require integration with cloud providers
                
        except Exception as e:
            self.logger.error(f"Error in cluster health check: {str(e)}")
    
    async def _cleanup_completed_tasks(self, max_age_hours: int = 24):
        """
        Clean up old completed tasks to free memory.
        
        Args:
            max_age_hours: Maximum age for completed tasks in hours
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.completed_at and task.completed_at < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        if tasks_to_remove:
            self.logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
    
    async def shutdown(self):
        """
        Shutdown the cluster manager and clean up resources.
        """
        self.monitoring_active = False
        
        # Cancel all queued tasks
        for task_id in list(self.task_queue):
            await self.cancel_task(task_id)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Shutdown Ray if we initialized it
        if self.ray_initialized and RAY_AVAILABLE:
            try:
                ray.shutdown()
                self.logger.info("Ray cluster shutdown complete")
            except Exception as e:
                self.logger.error(f"Error shutting down Ray: {str(e)}")
        
        self.cluster_status = ClusterStatus.TERMINATED
        self.logger.info("Ray cluster manager shutdown complete")

# Global cluster manager instance
ray_cluster_manager = RayClusterManager()

# Convenience functions
async def submit_distributed_task(task_type: str, task_function: Callable, 
                                 input_data: Dict[str, Any], priority: int = 0) -> str:
    """Submit a distributed task to the Ray cluster"""
    return await ray_cluster_manager.submit_task(task_type, task_function, input_data, priority)

async def get_distributed_task_status(task_id: str) -> Optional[DistributedTask]:
    """Get status of a distributed task"""
    return await ray_cluster_manager.get_task_status(task_id)

async def get_cluster_health() -> Dict[str, Any]:
    """Get cluster health information"""
    return await ray_cluster_manager.get_cluster_status()
