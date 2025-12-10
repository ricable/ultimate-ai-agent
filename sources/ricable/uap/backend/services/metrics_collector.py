# File: backend/services/metrics_collector.py
"""
Real-time Metrics Collection Service for UAP Platform

Collects and processes metrics from various sources including:
- System performance metrics
- Agent request/response metrics  
- User activity metrics
- Business intelligence metrics
- Custom application metrics
"""

import asyncio
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import logging

from .analytics_service import analytics_service, RealTimeMetric
from ..cache.redis_cache import get_redis_client
from ..database.service import get_database_service
from ..models.analytics import SystemMetrics, AgentUsage, MetricType
from ..monitoring.metrics.prometheus_metrics import prometheus_metrics

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MetricDefinition:
    """Definition of a collectible metric"""
    name: str
    description: str
    unit: str
    category: str
    collection_interval: int  # seconds
    collector_func: Callable
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

class MetricsCollector:
    """
    Advanced metrics collection service that gathers performance,
    business, and operational metrics for real-time analytics.
    """
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.db_service = get_database_service()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Metric definitions registry
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.collection_tasks = {}
        self.is_collecting = False
        
        # Metric storage
        self.metrics_buffer = defaultdict(list)
        self.buffer_size = 1000
        self.flush_interval = 30  # seconds
        
        # Performance tracking
        self.collection_stats = {
            "metrics_collected": 0,
            "collection_errors": 0,
            "last_collection": None,
            "active_collectors": 0
        }
        
        # Initialize core metrics
        self._register_core_metrics()
    
    def _register_core_metrics(self):
        """Register core system and application metrics"""
        
        # System metrics
        self.register_metric(MetricDefinition(
            name="system_cpu_percent",
            description="CPU usage percentage",
            unit="percent",
            category="system",
            collection_interval=10,
            collector_func=self._collect_cpu_usage
        ))
        
        self.register_metric(MetricDefinition(
            name="system_memory_percent",
            description="Memory usage percentage",
            unit="percent",
            category="system",
            collection_interval=10,
            collector_func=self._collect_memory_usage
        ))
        
        self.register_metric(MetricDefinition(
            name="system_disk_usage",
            description="Disk usage percentage",
            unit="percent",
            category="system",
            collection_interval=30,
            collector_func=self._collect_disk_usage
        ))
        
        self.register_metric(MetricDefinition(
            name="system_network_io",
            description="Network I/O statistics",
            unit="bytes",
            category="system",
            collection_interval=10,
            collector_func=self._collect_network_io
        ))
        
        # Application metrics
        self.register_metric(MetricDefinition(
            name="active_connections",
            description="Number of active WebSocket connections",
            unit="count",
            category="application",
            collection_interval=5,
            collector_func=self._collect_active_connections
        ))
        
        self.register_metric(MetricDefinition(
            name="request_rate",
            description="HTTP requests per second",
            unit="requests/sec",
            category="application",
            collection_interval=5,
            collector_func=self._collect_request_rate
        ))
        
        self.register_metric(MetricDefinition(
            name="agent_response_time",
            description="Average agent response time",
            unit="milliseconds",
            category="agents",
            collection_interval=10,
            collector_func=self._collect_agent_response_time
        ))
        
        self.register_metric(MetricDefinition(
            name="active_user_sessions",
            description="Number of active user sessions",
            unit="count",
            category="users",
            collection_interval=15,
            collector_func=self._collect_active_sessions
        ))
        
        # Business metrics
        self.register_metric(MetricDefinition(
            name="hourly_requests",
            description="Requests processed in the last hour",
            unit="count",
            category="business",
            collection_interval=60,
            collector_func=self._collect_hourly_requests
        ))
        
        self.register_metric(MetricDefinition(
            name="cost_per_hour",
            description="Estimated cost per hour",
            unit="dollars",
            category="business",
            collection_interval=60,
            collector_func=self._collect_hourly_cost
        ))
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric for collection"""
        self.metric_definitions[metric_def.name] = metric_def
        logger.info(f"Registered metric: {metric_def.name}")
    
    async def start_collection(self):
        """Start metrics collection for all registered metrics"""
        if self.is_collecting:
            logger.warning("Metrics collection already started")
            return
        
        self.is_collecting = True
        logger.info("Starting metrics collection...")
        
        # Start collection task for each metric
        for metric_name, metric_def in self.metric_definitions.items():
            task = asyncio.create_task(
                self._collection_loop(metric_name, metric_def)
            )
            self.collection_tasks[metric_name] = task
        
        # Start buffer flush task
        flush_task = asyncio.create_task(self._flush_metrics_loop())
        self.collection_tasks["flush"] = flush_task
        
        # Start stats update task
        stats_task = asyncio.create_task(self._update_stats_loop())
        self.collection_tasks["stats"] = stats_task
        
        logger.info(f"Started collection for {len(self.metric_definitions)} metrics")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        logger.info("Stopping metrics collection...")
        
        # Cancel all collection tasks
        for task_name, task in self.collection_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.collection_tasks.clear()
        
        # Flush remaining metrics
        await self._flush_metrics()
        
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self, metric_name: str, metric_def: MetricDefinition):
        """Collection loop for a specific metric"""
        logger.info(f"Started collection loop for {metric_name}")
        
        while self.is_collecting:
            try:
                # Collect metric value
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(metric_def.collector_func):
                    value = await metric_def.collector_func()
                else:
                    # Run synchronous collector in thread pool
                    loop = asyncio.get_event_loop()
                    value = await loop.run_in_executor(
                        self.executor, 
                        metric_def.collector_func
                    )
                
                collection_time = (time.time() - start_time) * 1000  # ms
                
                # Create metric instance
                if value is not None:
                    metric = RealTimeMetric(
                        name=metric_name,
                        value=value,
                        timestamp=time.time(),
                        labels=metric_def.labels.copy(),
                        unit=metric_def.unit,
                        category=metric_def.category
                    )
                    
                    # Add to buffer
                    self.metrics_buffer[metric_name].append(metric)
                    
                    # Notify analytics service
                    analytics_service.add_real_time_metric(metric)
                    
                    # Update collection stats
                    self.collection_stats["metrics_collected"] += 1
                    self.collection_stats["last_collection"] = time.time()
                    
                    # Log slow collections
                    if collection_time > 1000:  # > 1 second
                        logger.warning(
                            f"Slow metric collection for {metric_name}: {collection_time:.2f}ms"
                        )
                
            except Exception as e:
                logger.error(f"Error collecting metric {metric_name}: {e}")
                self.collection_stats["collection_errors"] += 1
            
            # Wait for next collection
            await asyncio.sleep(metric_def.collection_interval)
    
    async def _flush_metrics_loop(self):
        """Background loop to flush metrics to storage"""
        while self.is_collecting:
            await asyncio.sleep(self.flush_interval)
            await self._flush_metrics()
    
    async def _flush_metrics(self):
        """Flush metrics from buffer to storage"""
        if not self.metrics_buffer:
            return
        
        try:
            # Prepare metrics for storage
            metrics_to_store = []
            
            for metric_name, metrics_list in self.metrics_buffer.items():
                if not metrics_list:
                    continue
                
                for metric in metrics_list:
                    # Create database record
                    db_metric = SystemMetrics(
                        metric_name=metric.name,
                        metric_type=MetricType.GAUGE,  # Default to gauge
                        value=float(metric.value) if isinstance(metric.value, (int, float)) else 0,
                        unit=metric.unit,
                        component=metric.category,
                        labels=metric.labels,
                        timestamp=datetime.fromtimestamp(metric.timestamp)
                    )
                    metrics_to_store.append(db_metric)
            
            # Batch insert to database
            if metrics_to_store:
                await self.db_service.bulk_insert(metrics_to_store)
                logger.debug(f"Flushed {len(metrics_to_store)} metrics to database")
            
            # Store recent metrics in Redis for quick access
            for metric_name, metrics_list in self.metrics_buffer.items():
                if metrics_list:
                    # Keep last 100 points for each metric
                    recent_metrics = metrics_list[-100:]
                    redis_key = f"metrics:recent:{metric_name}"
                    
                    await self.redis_client.setex(
                        redis_key,
                        300,  # 5 minutes TTL
                        json.dumps([m.to_dict() for m in recent_metrics], default=str)
                    )
            
            # Clear buffer
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
    
    async def _update_stats_loop(self):
        """Update collection statistics"""
        while self.is_collecting:
            try:
                self.collection_stats["active_collectors"] = len(
                    [task for task in self.collection_tasks.values() if not task.done()]
                )
                
                # Store stats in Redis
                await self.redis_client.setex(
                    "metrics:collection_stats",
                    60,
                    json.dumps(self.collection_stats, default=str)
                )
                
            except Exception as e:
                logger.error(f"Error updating collection stats: {e}")
            
            await asyncio.sleep(30)
    
    # Metric collector functions
    def _collect_cpu_usage(self) -> float:
        """Collect CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
    
    def _collect_memory_usage(self) -> float:
        """Collect memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def _collect_disk_usage(self) -> float:
        """Collect disk usage percentage"""
        disk = psutil.disk_usage('/')
        return (disk.used / disk.total) * 100
    
    def _collect_network_io(self) -> Dict[str, int]:
        """Collect network I/O statistics"""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
    
    async def _collect_active_connections(self) -> int:
        """Collect number of active WebSocket connections"""
        try:
            # This would connect to the WebSocket manager to get active connections
            # For now, return a simulated value based on Redis data
            active_users = await self.redis_client.get("analytics:active_users")
            return int(active_users) if active_users else 0
        except Exception:
            return 0
    
    async def _collect_request_rate(self) -> float:
        """Collect HTTP requests per second"""
        try:
            # Get request count from Prometheus metrics or similar
            # For now, calculate from recent activity
            return prometheus_metrics.get_request_rate()
        except Exception:
            return 0.0
    
    async def _collect_agent_response_time(self) -> float:
        """Collect average agent response time"""
        try:
            # Get from performance monitor or calculate from recent requests
            from ..monitoring.metrics.performance import performance_monitor
            return performance_monitor.get_average_response_time()
        except Exception:
            return 0.0
    
    async def _collect_active_sessions(self) -> int:
        """Collect number of active user sessions"""
        try:
            from ..analytics.usage_analytics import usage_analytics
            return len(usage_analytics.active_sessions)
        except Exception:
            return 0
    
    async def _collect_hourly_requests(self) -> int:
        """Collect requests processed in the last hour"""
        try:
            from ..analytics.usage_analytics import usage_analytics
            return usage_analytics.get_usage_summary(1).get("total_requests", 0)
        except Exception:
            return 0
    
    async def _collect_hourly_cost(self) -> float:
        """Collect estimated cost per hour"""
        try:
            from ..analytics.usage_analytics import usage_analytics
            return usage_analytics.get_usage_summary(1).get("estimated_cost", 0.0)
        except Exception:
            return 0.0
    
    async def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a metric"""
        try:
            # Try Redis first for recent data
            redis_key = f"metrics:recent:{metric_name}"
            recent_data = await self.redis_client.get(redis_key)
            
            if recent_data:
                metrics = json.loads(recent_data)
                # Filter by time range
                cutoff_time = time.time() - (hours * 3600)
                return [m for m in metrics if m["timestamp"] >= cutoff_time]
            
            # Fall back to database query
            query = """
                SELECT metric_name, value, unit, labels, timestamp 
                FROM system_metrics 
                WHERE metric_name = $1 
                AND timestamp >= $2 
                ORDER BY timestamp ASC
            """
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            results = await self.db_service.execute_query(query, metric_name, cutoff_time)
            
            return [
                {
                    "name": row["metric_name"],
                    "value": row["value"],
                    "unit": row["unit"],
                    "labels": row["labels"],
                    "timestamp": row["timestamp"].timestamp()
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting metric history for {metric_name}: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics"""
        return {
            **self.collection_stats,
            "registered_metrics": len(self.metric_definitions),
            "buffer_size": sum(len(metrics) for metrics in self.metrics_buffer.values()),
            "is_collecting": self.is_collecting
        }
    
    def add_custom_metric(self, name: str, value: Any, labels: Optional[Dict[str, str]] = None):
        """Add a custom metric value"""
        metric = RealTimeMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            unit="",
            category="custom"
        )
        
        # Add to buffer
        self.metrics_buffer[name].append(metric)
        
        # Notify analytics service
        analytics_service.add_real_time_metric(metric)

# Global metrics collector instance
metrics_collector = MetricsCollector()
