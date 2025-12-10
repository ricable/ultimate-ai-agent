# File: backend/monitoring/metrics/performance.py
"""
Performance monitoring system for UAP platform.
Tracks agent response times, WebSocket metrics, resource utilization, and system health.
"""

import asyncio
import time
import psutil
import statistics
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from threading import Lock
import gc
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PerformanceMetric:
    """Container for performance metric data"""
    timestamp: datetime
    metric_name: str
    value: Union[float, int]
    unit: str
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class AgentPerformanceStats:
    """Performance statistics for an agent"""
    agent_id: str
    framework: str
    total_requests: int = 0
    total_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    last_request_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.last_request_time:
            data['last_request_time'] = self.last_request_time.isoformat()
        return data

@dataclass
class SystemResourceStats:
    """System resource utilization statistics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    process_count: int
    thread_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.agent_stats: Dict[str, AgentPerformanceStats] = {}
        self.agent_response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.websocket_connections: Dict[str, Dict[str, Any]] = {}
        self.system_stats_history: deque = deque(maxlen=1000)
        self.lock = Lock()
        
        # Performance thresholds (from plan.md requirements)
        self.thresholds = {
            'agent_response_time_p95_ms': 2000,  # <2s for 95th percentile
            'ui_load_time_ms': 1000,             # <1s Time to Interactive
            'websocket_stability_percent': 99.9,  # 99.9% connection stability
            'max_concurrent_sessions': 1000,      # 1000+ concurrent sessions
            'memory_usage_percent': 85,           # Alert when memory > 85%
            'cpu_usage_percent': 80,              # Alert when CPU > 80%
            'error_rate_percent': 5,              # Alert when error rate > 5%
        }
        
        # Start background monitoring
        self.monitoring_active = True
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def record_metric(self, name: str, value: Union[float, int], unit: str, 
                     tags: Dict[str, str] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics_history.append(metric)
    
    def start_agent_request(self, agent_id: str, framework: str, request_id: str) -> Dict[str, Any]:
        """Start tracking an agent request"""
        return {
            'agent_id': agent_id,
            'framework': framework,
            'request_id': request_id,
            'start_time': time.time(),
            'timestamp': datetime.utcnow()
        }
    
    def finish_agent_request(self, request_context: Dict[str, Any], success: bool = True,
                           error_details: str = None):
        """Finish tracking an agent request and record metrics"""
        end_time = time.time()
        response_time_ms = (end_time - request_context['start_time']) * 1000
        
        agent_id = request_context['agent_id']
        framework = request_context['framework']
        
        # Record response time metric
        self.record_metric(
            f"agent.response_time",
            response_time_ms,
            "ms",
            {"agent_id": agent_id, "framework": framework, "success": str(success)}
        )
        
        # Update agent statistics
        with self.lock:
            if agent_id not in self.agent_stats:
                self.agent_stats[agent_id] = AgentPerformanceStats(
                    agent_id=agent_id,
                    framework=framework
                )
            
            stats = self.agent_stats[agent_id]
            stats.total_requests += 1
            stats.total_response_time_ms += response_time_ms
            stats.last_request_time = datetime.utcnow()
            
            if success:
                stats.success_count += 1
            else:
                stats.error_count += 1
            
            # Update response time statistics
            response_times = self.agent_response_times[agent_id]
            response_times.append(response_time_ms)
            
            # Calculate percentiles if we have enough data
            if len(response_times) >= 10:
                sorted_times = sorted(response_times)
                stats.p95_response_time_ms = self._percentile(sorted_times, 95)
                stats.p99_response_time_ms = self._percentile(sorted_times, 99)
            
            # Update other statistics
            stats.avg_response_time_ms = stats.total_response_time_ms / stats.total_requests
            stats.min_response_time_ms = min(stats.min_response_time_ms, response_time_ms)
            stats.max_response_time_ms = max(stats.max_response_time_ms, response_time_ms)
            stats.success_rate = (stats.success_count / stats.total_requests) * 100
    
    def track_websocket_connection(self, connection_id: str, agent_id: str, metadata: Dict[str, Any] = None):
        """Track a new WebSocket connection"""
        connection_info = {
            'agent_id': agent_id,
            'connected_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'metadata': metadata or {}
        }
        
        with self.lock:
            self.websocket_connections[connection_id] = connection_info
        
        # Record connection metric
        self.record_metric(
            "websocket.connection_opened",
            1,
            "count",
            {"agent_id": agent_id}
        )
        
        # Record total active connections
        self.record_metric(
            "websocket.active_connections",
            len(self.websocket_connections),
            "count"
        )
    
    def update_websocket_activity(self, connection_id: str, message_type: str, 
                                 bytes_size: int, direction: str = "sent"):
        """Update WebSocket connection activity"""
        if connection_id not in self.websocket_connections:
            return
        
        with self.lock:
            conn = self.websocket_connections[connection_id]
            conn['last_activity'] = datetime.utcnow()
            
            if direction == "sent":
                conn['messages_sent'] += 1
                conn['bytes_sent'] += bytes_size
            else:
                conn['messages_received'] += 1
                conn['bytes_received'] += bytes_size
        
        # Record activity metrics
        self.record_metric(
            f"websocket.message_{direction}",
            1,
            "count",
            {"agent_id": conn['agent_id'], "message_type": message_type}
        )
        
        self.record_metric(
            f"websocket.bytes_{direction}",
            bytes_size,
            "bytes",
            {"agent_id": conn['agent_id']}
        )
    
    def remove_websocket_connection(self, connection_id: str, reason: str = "normal"):
        """Remove a WebSocket connection and record metrics"""
        if connection_id not in self.websocket_connections:
            return
        
        with self.lock:
            conn = self.websocket_connections.pop(connection_id)
            
            # Calculate connection duration
            duration = (datetime.utcnow() - conn['connected_at']).total_seconds()
            
            # Record disconnection metrics
            self.record_metric(
                "websocket.connection_closed",
                1,
                "count",
                {"agent_id": conn['agent_id'], "reason": reason}
            )
            
            self.record_metric(
                "websocket.connection_duration",
                duration,
                "seconds",
                {"agent_id": conn['agent_id']}
            )
            
            # Update active connections count
            self.record_metric(
                "websocket.active_connections",
                len(self.websocket_connections),
                "count"
            )
    
    def collect_system_metrics(self) -> SystemResourceStats:
        """Collect current system resource metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Network metrics
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # Process metrics
        process_count = len(psutil.pids())
        current_process = psutil.Process()
        thread_count = current_process.num_threads()
        
        stats = SystemResourceStats(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            active_connections=len(self.websocket_connections),
            process_count=process_count,
            thread_count=thread_count
        )
        
        # Record individual metrics
        metrics_to_record = [
            ("system.cpu_percent", cpu_percent, "percent"),
            ("system.memory_percent", memory_percent, "percent"),
            ("system.memory_used", memory_used_mb, "mb"),
            ("system.disk_usage_percent", disk_usage_percent, "percent"),
            ("system.active_connections", len(self.websocket_connections), "count"),
            ("system.process_count", process_count, "count"),
            ("system.thread_count", thread_count, "count")
        ]
        
        for name, value, unit in metrics_to_record:
            self.record_metric(name, value, unit)
        
        # Store in history
        with self.lock:
            self.system_stats_history.append(stats)
        
        return stats
    
    def get_agent_statistics(self, agent_id: str = None) -> Union[Dict[str, AgentPerformanceStats], AgentPerformanceStats]:
        """Get performance statistics for agents"""
        with self.lock:
            if agent_id:
                return self.agent_stats.get(agent_id)
            return {aid: stats.to_dict() for aid, stats in self.agent_stats.items()}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        current_stats = self.collect_system_metrics()
        
        # Calculate health scores
        health_checks = {
            'cpu_healthy': current_stats.cpu_percent < self.thresholds['cpu_usage_percent'],
            'memory_healthy': current_stats.memory_percent < self.thresholds['memory_usage_percent'],
            'disk_healthy': current_stats.disk_usage_percent < 90,  # 90% threshold for disk
            'connections_healthy': current_stats.active_connections <= self.thresholds['max_concurrent_sessions']
        }
        
        overall_health = all(health_checks.values())
        
        # Get agent performance health
        agent_health = {}
        for agent_id, stats in self.agent_stats.items():
            agent_healthy = (
                stats.p95_response_time_ms <= self.thresholds['agent_response_time_p95_ms'] and
                stats.success_rate >= (100 - self.thresholds['error_rate_percent'])
            )
            agent_health[agent_id] = {
                'healthy': agent_healthy,
                'response_time_p95': stats.p95_response_time_ms,
                'success_rate': stats.success_rate
            }
        
        return {
            'overall_healthy': overall_health,
            'timestamp': datetime.utcnow().isoformat(),
            'system_health': health_checks,
            'agent_health': agent_health,
            'current_stats': current_stats.to_dict(),
            'thresholds': self.thresholds
        }
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        # Filter recent metrics
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.metric_name].append(metric.value)
        
        # Calculate summaries
        summary = {}
        for name, values in metric_groups.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': self._percentile(sorted(values), 95) if len(values) >= 10 else None
                }
        
        return {
            'time_window_minutes': time_window_minutes,
            'metrics_summary': summary,
            'agent_stats': {aid: stats.to_dict() for aid, stats in self.agent_stats.items()},
            'websocket_summary': {
                'active_connections': len(self.websocket_connections),
                'connections_by_agent': self._group_connections_by_agent()
            }
        }
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not sorted_values:
            return 0.0
        
        index = int((percentile / 100.0) * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        
        return sorted_values[index]
    
    def _group_connections_by_agent(self) -> Dict[str, int]:
        """Group WebSocket connections by agent"""
        agent_counts = defaultdict(int)
        for conn in self.websocket_connections.values():
            agent_counts[conn['agent_id']] += 1
        return dict(agent_counts)
    
    async def start_background_monitoring(self, interval_seconds: int = 30):
        """Start background system monitoring"""
        while self.monitoring_active:
            try:
                # Collect system metrics in thread pool to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.collect_system_metrics
                )
                
                # Cleanup old data
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._cleanup_old_data
                )
                
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                print(f"Error in background monitoring: {e}")
                await asyncio.sleep(interval_seconds)
    
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        # Clean up old WebSocket connections (disconnected more than 1 hour ago)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        with self.lock:
            # Clean up old agent response times
            for agent_id in list(self.agent_response_times.keys()):
                # Keep only recent response times
                self.agent_response_times[agent_id] = deque(
                    list(self.agent_response_times[agent_id])[-100:],  # Keep last 100
                    maxlen=1000
                )
        
        # Force garbage collection
        gc.collect()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        self.executor.shutdown(wait=True)

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Convenience functions
def start_agent_request(agent_id: str, framework: str, request_id: str) -> Dict[str, Any]:
    """Start tracking an agent request"""
    return performance_monitor.start_agent_request(agent_id, framework, request_id)

def finish_agent_request(request_context: Dict[str, Any], success: bool = True, error_details: str = None):
    """Finish tracking an agent request"""
    performance_monitor.finish_agent_request(request_context, success, error_details)

def track_websocket_connection(connection_id: str, agent_id: str, metadata: Dict[str, Any] = None):
    """Track a WebSocket connection"""
    performance_monitor.track_websocket_connection(connection_id, agent_id, metadata)

def remove_websocket_connection(connection_id: str, reason: str = "normal"):
    """Remove a WebSocket connection"""
    performance_monitor.remove_websocket_connection(connection_id, reason)

def get_system_health() -> Dict[str, Any]:
    """Get current system health"""
    return performance_monitor.get_system_health()

def get_performance_summary(time_window_minutes: int = 60) -> Dict[str, Any]:
    """Get performance summary"""
    return performance_monitor.get_performance_summary(time_window_minutes)