# File: backend/monitoring/metrics/prometheus_metrics.py
"""
Prometheus metrics integration for UAP platform.
Exposes standardized metrics for monitoring and alerting.
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from typing import Dict, Any
import time
from datetime import datetime

class UAP_PrometheusMetrics:
    """Prometheus metrics collector for UAP platform"""
    
    def __init__(self):
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Agent interaction metrics
        self.agent_requests_total = Counter(
            'uap_agent_requests_total',
            'Total number of agent requests',
            ['agent_id', 'framework', 'status'],
            registry=self.registry
        )
        
        self.agent_response_time = Histogram(
            'uap_agent_response_time_seconds',
            'Agent response time in seconds',
            ['agent_id', 'framework'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        self.agent_response_size = Histogram(
            'uap_agent_response_size_bytes',
            'Agent response size in bytes',
            ['agent_id', 'framework'],
            buckets=(100, 1000, 10000, 100000, 1000000),
            registry=self.registry
        )
        
        # WebSocket metrics
        self.websocket_connections_active = Gauge(
            'uap_websocket_connections_active',
            'Number of active WebSocket connections',
            ['agent_id'],
            registry=self.registry
        )
        
        self.websocket_messages_total = Counter(
            'uap_websocket_messages_total',
            'Total WebSocket messages',
            ['agent_id', 'direction', 'message_type'],
            registry=self.registry
        )
        
        self.websocket_connection_duration = Histogram(
            'uap_websocket_connection_duration_seconds',
            'WebSocket connection duration in seconds',
            ['agent_id'],
            buckets=(1, 10, 60, 300, 1800, 3600, 7200),
            registry=self.registry
        )
        
        # API metrics
        self.http_requests_total = Counter(
            'uap_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'uap_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'uap_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'uap_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_memory_used_bytes = Gauge(
            'uap_system_memory_used_bytes',
            'System memory used in bytes',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'uap_system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        self.system_process_count = Gauge(
            'uap_system_process_count',
            'Number of system processes',
            registry=self.registry
        )
        
        # Framework status metrics
        self.framework_status = Enum(
            'uap_framework_status',
            'Framework status',
            ['framework'],
            states=['healthy', 'degraded', 'unhealthy', 'unknown'],
            registry=self.registry
        )
        
        self.framework_agents_active = Gauge(
            'uap_framework_agents_active',
            'Number of active agents per framework',
            ['framework'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'uap_errors_total',
            'Total number of errors',
            ['component', 'error_type', 'severity'],
            registry=self.registry
        )
        
        # Performance threshold violations
        self.threshold_violations_total = Counter(
            'uap_threshold_violations_total',
            'Total threshold violations',
            ['metric', 'threshold_type'],
            registry=self.registry
        )
        
        # Distributed processing metrics
        self.ray_cluster_nodes = Gauge(
            'uap_ray_cluster_nodes_total',
            'Total number of Ray cluster nodes',
            registry=self.registry
        )
        
        self.ray_cluster_resources_total = Gauge(
            'uap_ray_cluster_resources_total',
            'Total Ray cluster resources',
            ['resource_type'],
            registry=self.registry
        )
        
        self.ray_cluster_resources_available = Gauge(
            'uap_ray_cluster_resources_available',
            'Available Ray cluster resources',
            ['resource_type'],
            registry=self.registry
        )
        
        self.ray_tasks_total = Counter(
            'uap_ray_tasks_total',
            'Total Ray tasks submitted',
            ['task_type', 'status'],
            registry=self.registry
        )
        
        self.ray_task_duration = Histogram(
            'uap_ray_task_duration_seconds',
            'Ray task execution duration in seconds',
            ['task_type'],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
            registry=self.registry
        )
        
        self.ray_queue_size = Gauge(
            'uap_ray_queue_size',
            'Number of tasks in Ray queue',
            registry=self.registry
        )
        
        self.distributed_workloads_total = Counter(
            'uap_distributed_workloads_total',
            'Total distributed workloads submitted',
            ['workload_type', 'strategy', 'status'],
            registry=self.registry
        )
        
        self.distributed_workload_duration = Histogram(
            'uap_distributed_workload_duration_seconds',
            'Distributed workload execution duration in seconds',
            ['workload_type', 'strategy'],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0),
            registry=self.registry
        )
        
        self.distributed_workload_progress = Gauge(
            'uap_distributed_workload_progress_percent',
            'Distributed workload progress percentage',
            ['workload_id'],
            registry=self.registry
        )
        
        self.distributed_queue_size = Gauge(
            'uap_distributed_queue_size',
            'Number of workloads in distributed queue',
            registry=self.registry
        )
        
        self.distributed_task_count = Gauge(
            'uap_distributed_task_count',
            'Number of tasks in distributed workloads',
            ['workload_type'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'uap_application_info',
            'UAP application information',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': '3.0.0',
            'component': 'uap-backend',
            'environment': 'development'  # This would be configured
        })
        
        # Track active connections by agent
        self.active_connections_by_agent: Dict[str, int] = {}
    
    def record_agent_request(self, agent_id: str, framework: str, 
                           response_time_seconds: float, success: bool = True,
                           response_size_bytes: int = None):
        """Record agent request metrics"""
        status = 'success' if success else 'error'
        
        # Increment request counter
        self.agent_requests_total.labels(
            agent_id=agent_id,
            framework=framework,
            status=status
        ).inc()
        
        # Record response time
        self.agent_response_time.labels(
            agent_id=agent_id,
            framework=framework
        ).observe(response_time_seconds)
        
        # Record response size if provided
        if response_size_bytes is not None:
            self.agent_response_size.labels(
                agent_id=agent_id,
                framework=framework
            ).observe(response_size_bytes)
    
    def record_websocket_connection_opened(self, agent_id: str, connection_id: str):
        """Record WebSocket connection opened"""
        if agent_id not in self.active_connections_by_agent:
            self.active_connections_by_agent[agent_id] = 0
        
        self.active_connections_by_agent[agent_id] += 1
        
        self.websocket_connections_active.labels(
            agent_id=agent_id
        ).set(self.active_connections_by_agent[agent_id])
    
    def record_websocket_connection_closed(self, agent_id: str, connection_id: str,
                                         duration_seconds: float):
        """Record WebSocket connection closed"""
        if agent_id in self.active_connections_by_agent:
            self.active_connections_by_agent[agent_id] = max(0, 
                self.active_connections_by_agent[agent_id] - 1)
            
            self.websocket_connections_active.labels(
                agent_id=agent_id
            ).set(self.active_connections_by_agent[agent_id])
        
        # Record connection duration
        self.websocket_connection_duration.labels(
            agent_id=agent_id
        ).observe(duration_seconds)
    
    def record_websocket_message(self, agent_id: str, direction: str, 
                                message_type: str):
        """Record WebSocket message"""
        self.websocket_messages_total.labels(
            agent_id=agent_id,
            direction=direction,
            message_type=message_type
        ).inc()
    
    def record_http_request(self, method: str, endpoint: str, status_code: int,
                          duration_seconds: float):
        """Record HTTP request metrics"""
        # Normalize endpoint to avoid high cardinality
        normalized_endpoint = self._normalize_endpoint(endpoint)
        
        self.http_requests_total.labels(
            method=method,
            endpoint=normalized_endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=normalized_endpoint
        ).observe(duration_seconds)
    
    def update_system_metrics(self, cpu_percent: float, memory_percent: float,
                            memory_used_bytes: int, disk_usage_percent: float,
                            process_count: int):
        """Update system resource metrics"""
        self.system_cpu_usage.set(cpu_percent)
        self.system_memory_usage.set(memory_percent)
        self.system_memory_used_bytes.set(memory_used_bytes)
        self.system_disk_usage.set(disk_usage_percent)
        self.system_process_count.set(process_count)
    
    def update_framework_status(self, framework: str, status: str, active_agents: int):
        """Update framework status metrics"""
        self.framework_status.labels(framework=framework).state(status)
        self.framework_agents_active.labels(framework=framework).set(active_agents)
    
    def record_error(self, component: str, error_type: str, severity: str = 'error'):
        """Record error occurrence"""
        self.errors_total.labels(
            component=component,
            error_type=error_type,
            severity=severity
        ).inc()
    
    def record_threshold_violation(self, metric: str, threshold_type: str):
        """Record threshold violation"""
        self.threshold_violations_total.labels(
            metric=metric,
            threshold_type=threshold_type
        ).inc()
    
    def update_ray_cluster_metrics(self, nodes_count: int, total_resources: Dict[str, float],
                                 available_resources: Dict[str, float], queue_size: int):
        """Update Ray cluster metrics"""
        self.ray_cluster_nodes.set(nodes_count)
        self.ray_queue_size.set(queue_size)
        
        # Update resource metrics
        for resource_type, total in total_resources.items():
            self.ray_cluster_resources_total.labels(resource_type=resource_type).set(total)
        
        for resource_type, available in available_resources.items():
            self.ray_cluster_resources_available.labels(resource_type=resource_type).set(available)
    
    def record_ray_task(self, task_type: str, status: str, duration_seconds: float = None):
        """Record Ray task metrics"""
        self.ray_tasks_total.labels(task_type=task_type, status=status).inc()
        
        if duration_seconds is not None:
            self.ray_task_duration.labels(task_type=task_type).observe(duration_seconds)
    
    def record_distributed_workload(self, workload_type: str, strategy: str, status: str,
                                   duration_seconds: float = None):
        """Record distributed workload metrics"""
        self.distributed_workloads_total.labels(
            workload_type=workload_type,
            strategy=strategy,
            status=status
        ).inc()
        
        if duration_seconds is not None:
            self.distributed_workload_duration.labels(
                workload_type=workload_type,
                strategy=strategy
            ).observe(duration_seconds)
    
    def update_distributed_workload_progress(self, workload_id: str, progress_percent: float):
        """Update distributed workload progress"""
        self.distributed_workload_progress.labels(workload_id=workload_id).set(progress_percent)
    
    def update_distributed_queue_metrics(self, queue_size: int, task_counts: Dict[str, int]):
        """Update distributed queue metrics"""
        self.distributed_queue_size.set(queue_size)
        
        for workload_type, task_count in task_counts.items():
            self.distributed_task_count.labels(workload_type=workload_type).set(task_count)
    
    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint to reduce cardinality"""
        # Replace UUIDs and IDs with placeholders
        import re
        
        # Replace UUID patterns
        endpoint = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 
                         '/{id}', endpoint)
        
        # Replace numeric IDs
        endpoint = re.sub(r'/\d+', '/{id}', endpoint)
        
        # Replace agent IDs (common pattern)
        endpoint = re.sub(r'/agents/[^/]+', '/agents/{agent_id}', endpoint)
        
        return endpoint
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get Prometheus content type"""
        return CONTENT_TYPE_LATEST

# Global metrics instance
prometheus_metrics = UAP_PrometheusMetrics()

# Convenience functions for recording metrics
def record_agent_request(agent_id: str, framework: str, response_time_seconds: float,
                        success: bool = True, response_size_bytes: int = None):
    """Record agent request metrics"""
    prometheus_metrics.record_agent_request(
        agent_id, framework, response_time_seconds, success, response_size_bytes
    )

def record_websocket_connection_opened(agent_id: str, connection_id: str):
    """Record WebSocket connection opened"""
    prometheus_metrics.record_websocket_connection_opened(agent_id, connection_id)

def record_websocket_connection_closed(agent_id: str, connection_id: str, duration_seconds: float):
    """Record WebSocket connection closed"""
    prometheus_metrics.record_websocket_connection_closed(agent_id, connection_id, duration_seconds)

def record_websocket_message(agent_id: str, direction: str, message_type: str):
    """Record WebSocket message"""
    prometheus_metrics.record_websocket_message(agent_id, direction, message_type)

def record_http_request(method: str, endpoint: str, status_code: int, duration_seconds: float):
    """Record HTTP request"""
    prometheus_metrics.record_http_request(method, endpoint, status_code, duration_seconds)

def update_system_metrics(cpu_percent: float, memory_percent: float, memory_used_bytes: int,
                         disk_usage_percent: float, process_count: int):
    """Update system metrics"""
    prometheus_metrics.update_system_metrics(
        cpu_percent, memory_percent, memory_used_bytes, disk_usage_percent, process_count
    )

def update_framework_status(framework: str, status: str, active_agents: int):
    """Update framework status"""
    prometheus_metrics.update_framework_status(framework, status, active_agents)

def record_error(component: str, error_type: str, severity: str = 'error'):
    """Record error"""
    prometheus_metrics.record_error(component, error_type, severity)

def record_threshold_violation(metric: str, threshold_type: str):
    """Record threshold violation"""
    prometheus_metrics.record_threshold_violation(metric, threshold_type)

def update_ray_cluster_metrics(nodes_count: int, total_resources: Dict[str, float],
                              available_resources: Dict[str, float], queue_size: int):
    """Update Ray cluster metrics"""
    prometheus_metrics.update_ray_cluster_metrics(nodes_count, total_resources, available_resources, queue_size)

def record_ray_task(task_type: str, status: str, duration_seconds: float = None):
    """Record Ray task metrics"""
    prometheus_metrics.record_ray_task(task_type, status, duration_seconds)

def record_distributed_workload(workload_type: str, strategy: str, status: str,
                               duration_seconds: float = None):
    """Record distributed workload metrics"""
    prometheus_metrics.record_distributed_workload(workload_type, strategy, status, duration_seconds)

def update_distributed_workload_progress(workload_id: str, progress_percent: float):
    """Update distributed workload progress"""
    prometheus_metrics.update_distributed_workload_progress(workload_id, progress_percent)

def update_distributed_queue_metrics(queue_size: int, task_counts: Dict[str, int]):
    """Update distributed queue metrics"""
    prometheus_metrics.update_distributed_queue_metrics(queue_size, task_counts)