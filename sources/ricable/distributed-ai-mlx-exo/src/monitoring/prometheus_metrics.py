"""
Prometheus Metrics Integration for MLX-Exo Distributed Cluster
Provides comprehensive metrics collection, custom exporters, and alert rules
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import threading

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info, Enum,
        start_http_server, generate_latest, CollectorRegistry,
        multiprocess, values
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes to avoid NameError
    class CollectorRegistry:
        pass
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    class Enum:
        def __init__(self, *args, **kwargs): pass
        def state(self, *args, **kwargs): pass
    def start_http_server(*args, **kwargs): pass
    def generate_latest(*args, **kwargs): return b""
    logging.warning("prometheus_client not available - metrics collection disabled")

# Import health monitor for integration
try:
    from .health_monitor import ClusterHealthMonitor, HealthStatus, ComponentHealth
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False
    logging.warning("Health monitor not available")

logger = logging.getLogger(__name__)

class MetricsRegistry:
    """Central registry for all cluster metrics"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize all metric collectors"""
        
        # API Server Metrics
        self.api_requests_total = Counter(
            'mlx_cluster_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code', 'model'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'mlx_cluster_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint', 'model'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.api_request_size = Histogram(
            'mlx_cluster_api_request_size_bytes',
            'API request size in bytes',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=self.registry
        )
        
        self.api_response_size = Histogram(
            'mlx_cluster_api_response_size_bytes', 
            'API response size in bytes',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
            registry=self.registry
        )
        
        # Inference Metrics
        self.inference_requests_total = Counter(
            'mlx_cluster_inference_requests_total',
            'Total number of inference requests',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.inference_duration = Histogram(
            'mlx_cluster_inference_duration_seconds',
            'Inference duration in seconds', 
            ['model', 'node'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.tokens_generated_total = Counter(
            'mlx_cluster_tokens_generated_total',
            'Total number of tokens generated',
            ['model', 'node'],
            registry=self.registry
        )
        
        self.tokens_per_second = Gauge(
            'mlx_cluster_tokens_per_second',
            'Current tokens per second rate',
            ['model', 'node'],
            registry=self.registry
        )
        
        self.time_to_first_token = Histogram(
            'mlx_cluster_time_to_first_token_seconds',
            'Time to first token in seconds',
            ['model', 'node'],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # Node Health Metrics
        self.node_status = Enum(
            'mlx_cluster_node_status',
            'Current status of cluster nodes',
            ['node_id'],
            states=['healthy', 'degraded', 'critical', 'failed', 'unknown'],
            registry=self.registry
        )
        
        self.node_cpu_usage = Gauge(
            'mlx_cluster_node_cpu_usage_percent',
            'CPU usage percentage by node',
            ['node_id'],
            registry=self.registry
        )
        
        self.node_memory_usage = Gauge(
            'mlx_cluster_node_memory_usage_percent', 
            'Memory usage percentage by node',
            ['node_id'],
            registry=self.registry
        )
        
        self.node_memory_total = Gauge(
            'mlx_cluster_node_memory_total_bytes',
            'Total memory in bytes by node',
            ['node_id'],
            registry=self.registry
        )
        
        self.node_disk_usage = Gauge(
            'mlx_cluster_node_disk_usage_percent',
            'Disk usage percentage by node', 
            ['node_id'],
            registry=self.registry
        )
        
        self.node_network_latency = Gauge(
            'mlx_cluster_node_network_latency_milliseconds',
            'Network latency to node in milliseconds',
            ['node_id'],
            registry=self.registry
        )
        
        # Model Metrics
        self.models_loaded = Gauge(
            'mlx_cluster_models_loaded',
            'Number of models currently loaded',
            ['node_id'],
            registry=self.registry
        )
        
        self.model_load_duration = Histogram(
            'mlx_cluster_model_load_duration_seconds',
            'Model loading duration in seconds',
            ['model', 'node_id'],
            buckets=[1, 5, 10, 30, 60, 120, 300],
            registry=self.registry
        )
        
        self.model_memory_usage = Gauge(
            'mlx_cluster_model_memory_usage_bytes',
            'Memory usage by loaded models',
            ['model', 'node_id'],
            registry=self.registry
        )
        
        # Cluster Metrics
        self.cluster_nodes_total = Gauge(
            'mlx_cluster_nodes_total',
            'Total number of nodes in cluster',
            registry=self.registry
        )
        
        self.cluster_nodes_healthy = Gauge(
            'mlx_cluster_nodes_healthy',
            'Number of healthy nodes in cluster',
            registry=self.registry
        )
        
        self.cluster_requests_queued = Gauge(
            'mlx_cluster_requests_queued',
            'Number of requests currently queued',
            registry=self.registry
        )
        
        self.cluster_throughput = Gauge(
            'mlx_cluster_throughput_requests_per_second',
            'Current cluster throughput in requests per second',
            registry=self.registry
        )
        
        # Error Metrics
        self.errors_total = Counter(
            'mlx_cluster_errors_total',
            'Total number of errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        self.failovers_total = Counter(
            'mlx_cluster_failovers_total',
            'Total number of failovers',
            ['node_id', 'reason'],
            registry=self.registry
        )

class MetricsCollector:
    """Collects and exports metrics for the MLX-Exo cluster"""
    
    def __init__(self, metrics_registry: MetricsRegistry, 
                 health_monitor: Optional[ClusterHealthMonitor] = None,
                 collection_interval: float = 10.0):
        self.metrics = metrics_registry
        self.health_monitor = health_monitor
        self.collection_interval = collection_interval
        self.collecting = False
        self._performance_history = {}
        
    async def start_collection(self):
        """Start metrics collection loop"""
        self.collecting = True
        logger.info("Starting metrics collection")
        
        while self.collecting:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.collecting = False
        logger.info("Stopping metrics collection")
    
    async def _collect_metrics(self):
        """Collect all metrics"""
        await asyncio.gather(
            self._collect_system_metrics(),
            self._collect_health_metrics(),
            self._collect_performance_metrics(),
            return_exceptions=True
        )
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics.node_cpu_usage.labels(node_id='localhost').set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.node_memory_usage.labels(node_id='localhost').set(memory.percent)
            self.metrics.node_memory_total.labels(node_id='localhost').set(memory.total)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics.node_disk_usage.labels(node_id='localhost').set(disk_percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_health_metrics(self):
        """Collect health monitoring metrics"""
        if not self.health_monitor:
            return
            
        try:
            cluster_health = self.health_monitor.get_cluster_health()
            
            # Update cluster-level metrics
            summary = cluster_health.get('summary', {})
            self.metrics.cluster_nodes_total.set(summary.get('total', 0))
            self.metrics.cluster_nodes_healthy.set(summary.get('healthy', 0))
            
            # Update per-node metrics
            for component_id, component_health in cluster_health.get('components', {}).items():
                status = component_health.get('status', 'unknown')
                self.metrics.node_status.labels(node_id=component_id).state(status)
                
                # Extract metrics from health data
                for metric in component_health.get('metrics', []):
                    if metric['name'] == 'network_latency':
                        self.metrics.node_network_latency.labels(
                            node_id=component_id
                        ).set(metric['value'])
                    elif metric['name'] == 'cpu_usage':
                        self.metrics.node_cpu_usage.labels(
                            node_id=component_id
                        ).set(metric['value'])
                    elif metric['name'] == 'memory_usage':
                        self.metrics.node_memory_usage.labels(
                            node_id=component_id
                        ).set(metric['value'])
                    elif metric['name'] == 'disk_usage':
                        self.metrics.node_disk_usage.labels(
                            node_id=component_id
                        ).set(metric['value'])
                        
        except Exception as e:
            logger.error(f"Error collecting health metrics: {e}")
    
    async def _collect_performance_metrics(self):
        """Collect performance-related metrics"""
        try:
            # Calculate throughput from request history
            current_time = time.time()
            
            # This would integrate with actual request tracking
            # For now, we'll maintain a simple counter
            
            pass  # Placeholder for actual performance collection
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float, model: str = 'unknown',
                          request_size: int = 0, response_size: int = 0):
        """Record an API request"""
        self.metrics.api_requests_total.labels(
            method=method, endpoint=endpoint, 
            status_code=str(status_code), model=model
        ).inc()
        
        self.metrics.api_request_duration.labels(
            method=method, endpoint=endpoint, model=model
        ).observe(duration)
        
        if request_size > 0:
            self.metrics.api_request_size.labels(
                method=method, endpoint=endpoint
            ).observe(request_size)
            
        if response_size > 0:
            self.metrics.api_response_size.labels(
                method=method, endpoint=endpoint
            ).observe(response_size)
    
    def record_inference(self, model: str, node: str, duration: float, 
                        tokens_generated: int, time_to_first_token: float,
                        status: str = 'success'):
        """Record an inference operation"""
        self.metrics.inference_requests_total.labels(
            model=model, status=status
        ).inc()
        
        self.metrics.inference_duration.labels(
            model=model, node=node
        ).observe(duration)
        
        self.metrics.tokens_generated_total.labels(
            model=model, node=node
        ).inc(tokens_generated)
        
        if duration > 0:
            tokens_per_sec = tokens_generated / duration
            self.metrics.tokens_per_second.labels(
                model=model, node=node
            ).set(tokens_per_sec)
        
        self.metrics.time_to_first_token.labels(
            model=model, node=node
        ).observe(time_to_first_token)
    
    def record_model_load(self, model: str, node_id: str, duration: float, memory_usage: int):
        """Record model loading operation"""
        self.metrics.model_load_duration.labels(
            model=model, node_id=node_id
        ).observe(duration)
        
        self.metrics.model_memory_usage.labels(
            model=model, node_id=node_id
        ).set(memory_usage)
    
    def record_error(self, component: str, error_type: str):
        """Record an error"""
        self.metrics.errors_total.labels(
            component=component, error_type=error_type
        ).inc()
    
    def record_failover(self, node_id: str, reason: str):
        """Record a failover event"""
        self.metrics.failovers_total.labels(
            node_id=node_id, reason=reason
        ).inc()

class PrometheusExporter:
    """Prometheus metrics exporter with HTTP server"""
    
    def __init__(self, port: int = 8000, host: str = '0.0.0.0',
                 registry: Optional[CollectorRegistry] = None):
        self.port = port
        self.host = host
        self.registry = registry or CollectorRegistry()
        self.server_thread = None
        
    def start_server(self):
        """Start Prometheus HTTP server"""
        try:
            start_http_server(self.port, self.host, self.registry)
            logger.info(f"Prometheus metrics server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            raise
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

class AlertRulesGenerator:
    """Generates Prometheus alert rules for cluster monitoring"""
    
    def __init__(self):
        self.rules = []
    
    def generate_rules(self) -> Dict[str, Any]:
        """Generate complete alert rules configuration"""
        return {
            'groups': [
                self._node_health_alerts(),
                self._performance_alerts(), 
                self._inference_alerts(),
                self._cluster_alerts()
            ]
        }
    
    def _node_health_alerts(self) -> Dict[str, Any]:
        """Node health alert rules"""
        return {
            'name': 'mlx_cluster_node_health',
            'rules': [
                {
                    'alert': 'NodeDown',
                    'expr': 'mlx_cluster_node_status != 0',
                    'for': '30s',
                    'labels': {'severity': 'critical'},
                    'annotations': {
                        'summary': 'Node {{ $labels.node_id }} is down',
                        'description': 'Node {{ $labels.node_id }} has been down for more than 30 seconds.'
                    }
                },
                {
                    'alert': 'HighCPUUsage',
                    'expr': 'mlx_cluster_node_cpu_usage_percent > 90',
                    'for': '5m',
                    'labels': {'severity': 'warning'},
                    'annotations': {
                        'summary': 'High CPU usage on node {{ $labels.node_id }}',
                        'description': 'CPU usage is {{ $value }}% on node {{ $labels.node_id }}'
                    }
                },
                {
                    'alert': 'HighMemoryUsage',
                    'expr': 'mlx_cluster_node_memory_usage_percent > 95',
                    'for': '2m',
                    'labels': {'severity': 'critical'},
                    'annotations': {
                        'summary': 'High memory usage on node {{ $labels.node_id }}',
                        'description': 'Memory usage is {{ $value }}% on node {{ $labels.node_id }}'
                    }
                },
                {
                    'alert': 'HighDiskUsage',
                    'expr': 'mlx_cluster_node_disk_usage_percent > 85',
                    'for': '10m',
                    'labels': {'severity': 'warning'},
                    'annotations': {
                        'summary': 'High disk usage on node {{ $labels.node_id }}',
                        'description': 'Disk usage is {{ $value }}% on node {{ $labels.node_id }}'
                    }
                }
            ]
        }
    
    def _performance_alerts(self) -> Dict[str, Any]:
        """Performance alert rules"""
        return {
            'name': 'mlx_cluster_performance',
            'rules': [
                {
                    'alert': 'HighAPILatency',
                    'expr': 'histogram_quantile(0.95, mlx_cluster_api_request_duration_seconds) > 5',
                    'for': '2m',
                    'labels': {'severity': 'warning'},
                    'annotations': {
                        'summary': 'High API latency',
                        'description': '95th percentile API latency is {{ $value }}s'
                    }
                },
                {
                    'alert': 'LowTokensPerSecond',
                    'expr': 'mlx_cluster_tokens_per_second < 5',
                    'for': '5m',
                    'labels': {'severity': 'warning'},
                    'annotations': {
                        'summary': 'Low token generation rate',
                        'description': 'Token generation rate is {{ $value }} tokens/sec on {{ $labels.node }}'
                    }
                }
            ]
        }
    
    def _inference_alerts(self) -> Dict[str, Any]:
        """Inference-specific alert rules"""
        return {
            'name': 'mlx_cluster_inference',
            'rules': [
                {
                    'alert': 'InferenceErrors',
                    'expr': 'rate(mlx_cluster_inference_requests_total{status="error"}[5m]) > 0.1',
                    'for': '1m',
                    'labels': {'severity': 'warning'},
                    'annotations': {
                        'summary': 'High inference error rate',
                        'description': 'Inference error rate is {{ $value }} errors/sec for model {{ $labels.model }}'
                    }
                },
                {
                    'alert': 'SlowInference',
                    'expr': 'histogram_quantile(0.95, mlx_cluster_inference_duration_seconds) > 30',
                    'for': '5m',
                    'labels': {'severity': 'warning'},
                    'annotations': {
                        'summary': 'Slow inference performance',
                        'description': '95th percentile inference time is {{ $value }}s for model {{ $labels.model }}'
                    }
                }
            ]
        }
    
    def _cluster_alerts(self) -> Dict[str, Any]:
        """Cluster-level alert rules"""
        return {
            'name': 'mlx_cluster_health',
            'rules': [
                {
                    'alert': 'ClusterDegraded',
                    'expr': 'mlx_cluster_nodes_healthy / mlx_cluster_nodes_total < 0.75',
                    'for': '1m',
                    'labels': {'severity': 'warning'},
                    'annotations': {
                        'summary': 'Cluster is degraded',
                        'description': 'Only {{ $value }} of cluster nodes are healthy'
                    }
                },
                {
                    'alert': 'ClusterCritical',
                    'expr': 'mlx_cluster_nodes_healthy / mlx_cluster_nodes_total < 0.5',
                    'for': '30s',
                    'labels': {'severity': 'critical'},
                    'annotations': {
                        'summary': 'Cluster is in critical state',
                        'description': 'Only {{ $value }} of cluster nodes are healthy'
                    }
                }
            ]
        }
    
    def save_rules(self, output_path: str):
        """Save alert rules to YAML file"""
        import yaml
        
        rules = self.generate_rules()
        with open(output_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False)
        
        logger.info(f"Alert rules saved to {output_path}")

def create_metrics_system(health_monitor: Optional[ClusterHealthMonitor] = None,
                         prometheus_port: int = 8000) -> Tuple[MetricsRegistry, MetricsCollector, PrometheusExporter]:
    """Factory function to create a complete metrics system"""
    
    if not PROMETHEUS_AVAILABLE:
        logger.error("Prometheus client not available - cannot create metrics system")
        return None, None, None
    
    # Create registry and metrics
    registry = CollectorRegistry()
    metrics_registry = MetricsRegistry(registry)
    
    # Create collector
    collector = MetricsCollector(metrics_registry, health_monitor)
    
    # Create exporter
    exporter = PrometheusExporter(prometheus_port, registry=registry)
    
    return metrics_registry, collector, exporter

# Usage example
if __name__ == "__main__":
    import yaml
    
    async def main():
        # Create metrics system
        metrics, collector, exporter = create_metrics_system()
        
        if not metrics:
            print("Prometheus not available")
            return
        
        # Start Prometheus server
        exporter.start_server()
        print(f"Metrics available at http://localhost:8000/metrics")
        
        # Start collection
        collection_task = asyncio.create_task(collector.start_collection())
        
        # Simulate some metrics
        collector.record_api_request('POST', '/v1/chat/completions', 200, 1.5, 'llama-70b')
        collector.record_inference('llama-70b', 'node1', 2.0, 150, 0.1)
        
        # Let it run briefly
        await asyncio.sleep(15)
        
        # Get current metrics
        print("\nCurrent metrics:")
        print(exporter.get_metrics())
        
        # Generate alert rules
        alert_gen = AlertRulesGenerator()
        rules = alert_gen.generate_rules()
        print("\nAlert rules generated:")
        print(yaml.dump(rules, default_flow_style=False))
        
        # Stop collection
        collector.stop_collection()
        await collection_task
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())