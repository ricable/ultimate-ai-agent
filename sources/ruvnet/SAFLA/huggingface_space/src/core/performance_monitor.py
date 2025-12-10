"""
Performance monitoring module for SAFLA HuggingFace Space.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Real-time performance tracking and monitoring."""
    
    def __init__(self, sample_interval: float = 1.0, history_size: int = 300):
        """
        Initialize performance monitor.
        
        Args:
            sample_interval: Seconds between performance samples
            history_size: Number of samples to keep in history
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        
        # Performance metrics history
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.response_time_history = deque(maxlen=history_size)
        self.throughput_history = deque(maxlen=history_size)
        self.error_count_history = deque(maxlen=history_size)
        
        # Current metrics
        self.current_metrics: Dict[str, Any] = {
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "memory_percent": 0.0,
            "avg_response_ms": 0.0,
            "throughput_ops_per_sec": 0.0,
            "error_rate": 0.0,
            "active_requests": 0,
            "total_requests": 0,
            "uptime_seconds": 0.0
        }
        
        # Request tracking
        self.active_requests = 0
        self.total_requests = 0
        self.total_errors = 0
        self.request_times: deque = deque(maxlen=100)
        
        # Monitoring control
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_time = time.time()
        self._is_running = False
        
        # Callbacks for alerts
        self._alert_callbacks: List[Callable] = []
        
        # Thresholds for alerts
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "response_ms": 1000.0,
            "error_rate": 0.05  # 5%
        }
    
    async def start(self) -> None:
        """Start performance monitoring."""
        if self._is_running:
            logger.warning("Performance monitor already running")
            return
        
        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop(self) -> None:
        """Stop performance monitoring."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_running:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                # Check thresholds and trigger alerts
                self._check_thresholds()
                
                # Wait for next sample
                await asyncio.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.sample_interval)
    
    async def _collect_metrics(self) -> None:
        """Collect current performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            # Calculate uptime
            uptime_seconds = time.time() - self._start_time
            
            # Calculate request metrics
            if self.request_times:
                avg_response_ms = statistics.mean(self.request_times) * 1000
            else:
                avg_response_ms = 0.0
            
            # Calculate throughput (requests per second over last minute)
            recent_requests = len([t for t in self.request_times 
                                 if t > time.time() - 60])
            throughput = recent_requests / 60.0 if recent_requests > 0 else 0.0
            
            # Calculate error rate
            error_rate = (self.total_errors / self.total_requests 
                         if self.total_requests > 0 else 0.0)
            
            # Update current metrics
            self.current_metrics.update({
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "memory_percent": memory_percent,
                "avg_response_ms": avg_response_ms,
                "throughput_ops_per_sec": throughput,
                "error_rate": error_rate,
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "uptime_seconds": uptime_seconds,
                "timestamp": datetime.now().isoformat()
            })
            
            # Add to history
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory_mb)
            self.response_time_history.append(avg_response_ms)
            self.throughput_history.append(throughput)
            self.error_count_history.append(self.total_errors)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def record_request_start(self) -> float:
        """
        Record the start of a request.
        
        Returns:
            Request start timestamp
        """
        self.active_requests += 1
        self.total_requests += 1
        return time.time()
    
    def record_request_end(self, start_time: float, error: bool = False) -> None:
        """
        Record the end of a request.
        
        Args:
            start_time: Request start timestamp
            error: Whether the request resulted in an error
        """
        self.active_requests = max(0, self.active_requests - 1)
        
        # Record response time
        response_time = time.time() - start_time
        self.request_times.append(response_time)
        
        # Record error if applicable
        if error:
            self.total_errors += 1
    
    def _check_thresholds(self) -> None:
        """Check metrics against thresholds and trigger alerts."""
        alerts = []
        
        if self.current_metrics["cpu_percent"] > self.thresholds["cpu_percent"]:
            alerts.append({
                "type": "cpu_high",
                "message": f"CPU usage high: {self.current_metrics['cpu_percent']:.1f}%",
                "severity": "warning"
            })
        
        if self.current_metrics["memory_percent"] > self.thresholds["memory_percent"]:
            alerts.append({
                "type": "memory_high",
                "message": f"Memory usage high: {self.current_metrics['memory_percent']:.1f}%",
                "severity": "warning"
            })
        
        if self.current_metrics["avg_response_ms"] > self.thresholds["response_ms"]:
            alerts.append({
                "type": "response_slow",
                "message": f"Response time slow: {self.current_metrics['avg_response_ms']:.0f}ms",
                "severity": "warning"
            })
        
        if self.current_metrics["error_rate"] > self.thresholds["error_rate"]:
            alerts.append({
                "type": "error_rate_high",
                "message": f"Error rate high: {self.current_metrics['error_rate']*100:.1f}%",
                "severity": "error"
            })
        
        # Trigger callbacks for alerts
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback for performance alerts."""
        self._alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.current_metrics.copy()
    
    def get_metrics_history(
        self,
        metric_name: str,
        duration_seconds: int = 300
    ) -> List[float]:
        """
        Get metric history.
        
        Args:
            metric_name: Name of metric (cpu, memory, response_time, throughput)
            duration_seconds: How many seconds of history to return
            
        Returns:
            List of metric values
        """
        samples_to_return = min(
            duration_seconds // self.sample_interval,
            self.history_size
        )
        
        if metric_name == "cpu":
            history = list(self.cpu_history)
        elif metric_name == "memory":
            history = list(self.memory_history)
        elif metric_name == "response_time":
            history = list(self.response_time_history)
        elif metric_name == "throughput":
            history = list(self.throughput_history)
        else:
            return []
        
        return history[-int(samples_to_return):]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        # Calculate statistics for each metric
        def calc_stats(history: deque) -> Dict[str, float]:
            if not history:
                return {"min": 0.0, "max": 0.0, "avg": 0.0, "current": 0.0}
            
            return {
                "min": min(history),
                "max": max(history),
                "avg": statistics.mean(history),
                "current": history[-1] if history else 0.0
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": {
                "seconds": self.current_metrics["uptime_seconds"],
                "formatted": str(timedelta(seconds=int(self.current_metrics["uptime_seconds"])))
            },
            "cpu": calc_stats(self.cpu_history),
            "memory": calc_stats(self.memory_history),
            "response_time": calc_stats(self.response_time_history),
            "throughput": calc_stats(self.throughput_history),
            "requests": {
                "total": self.total_requests,
                "active": self.active_requests,
                "errors": self.total_errors,
                "error_rate": self.current_metrics["error_rate"]
            },
            "health_score": self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> float:
        """
        Calculate overall system health score (0-100).
        
        Returns:
            Health score percentage
        """
        scores = []
        
        # CPU score (inverse of usage)
        cpu_score = max(0, 100 - self.current_metrics["cpu_percent"])
        scores.append(cpu_score * 0.25)  # 25% weight
        
        # Memory score (inverse of usage)
        mem_score = max(0, 100 - self.current_metrics["memory_percent"])
        scores.append(mem_score * 0.25)  # 25% weight
        
        # Response time score
        target_response = 100  # Target 100ms
        if self.current_metrics["avg_response_ms"] > 0:
            resp_score = min(100, (target_response / self.current_metrics["avg_response_ms"]) * 100)
        else:
            resp_score = 100
        scores.append(resp_score * 0.25)  # 25% weight
        
        # Error rate score
        error_score = max(0, (1 - self.current_metrics["error_rate"]) * 100)
        scores.append(error_score * 0.25)  # 25% weight
        
        return sum(scores)
    
    def reset_metrics(self) -> None:
        """Reset all metrics and history."""
        self.cpu_history.clear()
        self.memory_history.clear()
        self.response_time_history.clear()
        self.throughput_history.clear()
        self.error_count_history.clear()
        self.request_times.clear()
        
        self.active_requests = 0
        self.total_requests = 0
        self.total_errors = 0
        self._start_time = time.time()
        
        logger.info("Performance metrics reset")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for analysis or storage."""
        return {
            "export_time": datetime.now().isoformat(),
            "configuration": {
                "sample_interval": self.sample_interval,
                "history_size": self.history_size,
                "thresholds": self.thresholds
            },
            "current_metrics": self.current_metrics,
            "history": {
                "cpu": list(self.cpu_history),
                "memory": list(self.memory_history),
                "response_time": list(self.response_time_history),
                "throughput": list(self.throughput_history),
                "errors": list(self.error_count_history)
            },
            "summary": self.get_performance_summary()
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Decorator for monitoring function performance
def monitor_performance(func):
    """Decorator to monitor function performance."""
    async def async_wrapper(*args, **kwargs):
        start_time = performance_monitor.record_request_start()
        try:
            result = await func(*args, **kwargs)
            performance_monitor.record_request_end(start_time, error=False)
            return result
        except Exception as e:
            performance_monitor.record_request_end(start_time, error=True)
            raise
    
    def sync_wrapper(*args, **kwargs):
        start_time = performance_monitor.record_request_start()
        try:
            result = func(*args, **kwargs)
            performance_monitor.record_request_end(start_time, error=False)
            return result
        except Exception as e:
            performance_monitor.record_request_end(start_time, error=True)
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper