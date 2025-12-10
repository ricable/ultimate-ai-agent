# UAP Agent Debugger
"""
Advanced debugging infrastructure for UAP agents.
Provides request tracing, performance profiling, memory monitoring, and error analysis.
"""

import asyncio
import json
import time
import logging
import traceback
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import weakref
import psutil
import sys
from pathlib import Path

# Performance monitoring
try:
    import cProfile
    import pstats
    import io
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RequestTrace:
    """Represents a traced request/response."""
    trace_id: str
    agent_id: str
    framework: str
    timestamp: datetime
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "pending"  # pending, completed, error
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceMetric:
    """Represents a performance metric data point."""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = "ms"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class MemorySnapshot:
    """Represents a memory usage snapshot."""
    timestamp: datetime
    process_memory_mb: float
    system_memory_percent: float
    agent_memory_usage: Dict[str, Any]
    gc_stats: Dict[str, Any]
    object_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.object_counts is None:
            self.object_counts = {}


class RequestTracer:
    """Handles request tracing for debugging."""
    
    def __init__(self, max_traces: int = 10000, retention_hours: int = 24):
        self.max_traces = max_traces
        self.retention_hours = retention_hours
        self.traces: deque = deque(maxlen=max_traces)
        self.active_traces: Dict[str, RequestTrace] = {}
        self._lock = threading.Lock()
        self._trace_counter = 0
        self._cleanup_task = None
    
    def start_trace(self, agent_id: str, framework: str, request_data: Dict[str, Any]) -> str:
        """Start tracing a request."""
        with self._lock:
            self._trace_counter += 1
            trace_id = f"trace_{self._trace_counter}_{int(time.time())}"
            
            trace = RequestTrace(
                trace_id=trace_id,
                agent_id=agent_id,
                framework=framework,
                timestamp=datetime.utcnow(),
                request_data=request_data.copy(),
                metadata={
                    "thread_id": threading.get_ident(),
                    "process_id": psutil.Process().pid
                }
            )
            
            self.active_traces[trace_id] = trace
            return trace_id
    
    def complete_trace(self, trace_id: str, response_data: Dict[str, Any], duration_ms: float) -> None:
        """Complete a successful trace."""
        with self._lock:
            if trace_id in self.active_traces:
                trace = self.active_traces[trace_id]
                trace.response_data = response_data.copy()
                trace.duration_ms = duration_ms
                trace.status = "completed"
                
                self.traces.append(trace)
                del self.active_traces[trace_id]
    
    def error_trace(self, trace_id: str, error: str, duration_ms: float) -> None:
        """Mark a trace as errored."""
        with self._lock:
            if trace_id in self.active_traces:
                trace = self.active_traces[trace_id]
                trace.error = error
                trace.duration_ms = duration_ms
                trace.status = "error"
                
                self.traces.append(trace)
                del self.active_traces[trace_id]
    
    def get_traces(self, agent_id: Optional[str] = None, since: Optional[datetime] = None, 
                   status: Optional[str] = None, limit: int = 100) -> List[RequestTrace]:
        """Get traces matching the criteria."""
        with self._lock:
            filtered_traces = []
            
            for trace in reversed(self.traces):  # Most recent first
                if len(filtered_traces) >= limit:
                    break
                
                if agent_id and trace.agent_id != agent_id:
                    continue
                
                if since and trace.timestamp < since:
                    continue
                
                if status and trace.status != status:
                    continue
                
                filtered_traces.append(trace)
            
            return filtered_traces
    
    def get_trace_statistics(self, agent_id: Optional[str] = None, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get trace statistics."""
        traces = self.get_traces(agent_id=agent_id, since=since, limit=10000)
        
        if not traces:
            return {"message": "No traces found"}
        
        completed_traces = [t for t in traces if t.status == "completed"]
        error_traces = [t for t in traces if t.status == "error"]
        
        stats = {
            "total_traces": len(traces),
            "completed_traces": len(completed_traces),
            "error_traces": len(error_traces),
            "success_rate": (len(completed_traces) / len(traces)) * 100 if traces else 0,
            "time_range": {
                "start": min(t.timestamp for t in traces).isoformat() if traces else None,
                "end": max(t.timestamp for t in traces).isoformat() if traces else None
            }
        }
        
        if completed_traces:
            durations = [t.duration_ms for t in completed_traces if t.duration_ms is not None]
            if durations:
                stats["performance"] = {
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
                }
        
        if error_traces:
            error_types = defaultdict(int)
            for trace in error_traces:
                error_type = trace.error.split(":")[0] if trace.error else "Unknown"
                error_types[error_type] += 1
            
            stats["errors"] = {
                "error_types": dict(error_types),
                "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
            }
        
        return stats
    
    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_old_traces())
    
    async def _cleanup_old_traces(self) -> None:
        """Clean up old traces periodically."""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
                
                with self._lock:
                    # Remove old traces
                    while self.traces and self.traces[0].timestamp < cutoff_time:
                        self.traces.popleft()
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trace cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def stop_cleanup_task(self) -> None:
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None


class PerformanceProfiler:
    """Handles performance profiling for debugging."""
    
    def __init__(self, max_metrics: int = 50000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self._lock = threading.Lock()
        self._profilers: Dict[str, Any] = {}
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None, 
                     unit: str = "ms") -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics.append(metric)
    
    def start_profiling(self, profile_id: str) -> None:
        """Start CPU profiling."""
        if not PROFILING_AVAILABLE:
            raise RuntimeError("Profiling not available - install cProfile")
        
        profiler = cProfile.Profile()
        profiler.enable()
        self._profilers[profile_id] = profiler
    
    def stop_profiling(self, profile_id: str) -> Dict[str, Any]:
        """Stop CPU profiling and return results."""
        if profile_id not in self._profilers:
            raise ValueError(f"No active profiler with ID: {profile_id}")
        
        profiler = self._profilers.pop(profile_id)
        profiler.disable()
        
        # Capture profiling stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return {
            "profile_id": profile_id,
            "timestamp": datetime.utcnow().isoformat(),
            "stats_text": stats_stream.getvalue(),
            "total_calls": stats.total_calls,
            "total_time": stats.total_tt
        }
    
    def get_metrics(self, metric_name: Optional[str] = None, since: Optional[datetime] = None,
                   tags: Optional[Dict[str, str]] = None, limit: int = 1000) -> List[PerformanceMetric]:
        """Get performance metrics matching criteria."""
        with self._lock:
            filtered_metrics = []
            
            for metric in reversed(self.metrics):  # Most recent first
                if len(filtered_metrics) >= limit:
                    break
                
                if metric_name and metric.metric_name != metric_name:
                    continue
                
                if since and metric.timestamp < since:
                    continue
                
                if tags:
                    if not all(metric.tags.get(k) == v for k, v in tags.items()):
                        continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
    
    def get_metric_summary(self, metric_name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        metrics = self.get_metrics(metric_name=metric_name, since=since, limit=10000)
        
        if not metrics:
            return {"error": f"No metrics found for {metric_name}"}
        
        values = [m.value for m in metrics]
        
        return {
            "metric_name": metric_name,
            "data_points": len(values),
            "time_range": {
                "start": min(m.timestamp for m in metrics).isoformat(),
                "end": max(m.timestamp for m in metrics).isoformat()
            },
            "statistics": {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "median": sorted(values)[len(values) // 2],
                "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0],
                "p99": sorted(values)[int(len(values) * 0.99)] if len(values) > 1 else values[0]
            },
            "unit": metrics[0].unit if metrics else "unknown"
        }


class MemoryMonitor:
    """Monitors memory usage for debugging."""
    
    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self._lock = threading.Lock()
        self._monitoring_task = None
        self._monitoring_interval = 60  # seconds
    
    def take_snapshot(self, agent_context: Optional[Dict[str, Any]] = None) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        import gc
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.utcnow(),
            process_memory_mb=memory_info.rss / 1024 / 1024,
            system_memory_percent=psutil.virtual_memory().percent,
            agent_memory_usage=agent_context or {},
            gc_stats={
                "generation_0": len(gc.get_objects(0)) if hasattr(gc, 'get_objects') else 0,
                "generation_1": len(gc.get_objects(1)) if hasattr(gc, 'get_objects') else 0,
                "generation_2": len(gc.get_objects(2)) if hasattr(gc, 'get_objects') else 0,
                "collections": gc.get_stats() if hasattr(gc, 'get_stats') else []
            }
        )
        
        # Count object types
        if hasattr(gc, 'get_objects'):
            object_counts = defaultdict(int)
            for obj in gc.get_objects():
                object_counts[type(obj).__name__] += 1
            snapshot.object_counts = dict(object_counts)
        
        with self._lock:
            self.snapshots.append(snapshot)
        
        return snapshot
    
    def get_snapshots(self, since: Optional[datetime] = None, limit: int = 100) -> List[MemorySnapshot]:
        """Get memory snapshots."""
        with self._lock:
            filtered_snapshots = []
            
            for snapshot in reversed(self.snapshots):
                if len(filtered_snapshots) >= limit:
                    break
                
                if since and snapshot.timestamp < since:
                    continue
                
                filtered_snapshots.append(snapshot)
            
            return filtered_snapshots
    
    def detect_memory_leaks(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        since = datetime.utcnow() - timedelta(minutes=window_minutes)
        snapshots = self.get_snapshots(since=since, limit=1000)
        
        if len(snapshots) < 2:
            return {"error": "Not enough data points for leak detection"}
        
        # Analyze memory growth
        snapshots = sorted(snapshots, key=lambda s: s.timestamp)
        start_memory = snapshots[0].process_memory_mb
        end_memory = snapshots[-1].process_memory_mb
        memory_growth = end_memory - start_memory
        
        # Analyze object count growth
        object_growth = {}
        if snapshots[0].object_counts and snapshots[-1].object_counts:
            for obj_type in snapshots[-1].object_counts:
                start_count = snapshots[0].object_counts.get(obj_type, 0)
                end_count = snapshots[-1].object_counts[obj_type]
                growth = end_count - start_count
                if growth > 100:  # Significant growth
                    object_growth[obj_type] = {
                        "start_count": start_count,
                        "end_count": end_count,
                        "growth": growth
                    }
        
        leak_indicators = []
        
        # Check for significant memory growth
        if memory_growth > 100:  # More than 100MB growth
            leak_indicators.append(f"Process memory grew by {memory_growth:.1f}MB")
        
        # Check for object count growth
        if object_growth:
            for obj_type, growth_info in object_growth.items():
                leak_indicators.append(f"{obj_type} objects grew by {growth_info['growth']}")
        
        return {
            "analysis_window_minutes": window_minutes,
            "snapshots_analyzed": len(snapshots),
            "memory_growth_mb": memory_growth,
            "potential_leaks_detected": len(leak_indicators) > 0,
            "leak_indicators": leak_indicators,
            "object_growth": object_growth,
            "recommendations": self._generate_memory_recommendations(memory_growth, object_growth)
        }
    
    def _generate_memory_recommendations(self, memory_growth: float, object_growth: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if memory_growth > 200:
            recommendations.append("Significant memory growth detected - investigate potential memory leaks")
        
        if "dict" in object_growth and object_growth["dict"]["growth"] > 1000:
            recommendations.append("Large number of dict objects created - check for dictionary accumulation")
        
        if "list" in object_growth and object_growth["list"]["growth"] > 1000:
            recommendations.append("Large number of list objects created - check for list accumulation")
        
        if not recommendations:
            recommendations.append("Memory usage appears normal")
        
        return recommendations
    
    async def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous memory monitoring."""
        self._monitoring_interval = interval_seconds
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while True:
            try:
                self.take_snapshot()
                await asyncio.sleep(self._monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(60)
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None


class ErrorAnalyzer:
    """Analyzes errors for debugging."""
    
    def __init__(self, max_errors: int = 5000):
        self.max_errors = max_errors
        self.errors: deque = deque(maxlen=max_errors)
        self._lock = threading.Lock()
    
    def record_error(self, error: Exception, context: Dict[str, Any]) -> str:
        """Record an error for analysis."""
        error_id = f"error_{int(time.time())}_{id(error)}"
        
        error_record = {
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "context": context.copy(),
            "severity": self._determine_severity(error)
        }
        
        with self._lock:
            self.errors.append(error_record)
        
        return error_id
    
    def _determine_severity(self, error: Exception) -> str:
        """Determine error severity."""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return "critical"
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return "high"
        elif isinstance(error, (ValueError, TypeError)):
            return "medium"
        else:
            return "low"
    
    def get_errors(self, since: Optional[datetime] = None, severity: Optional[str] = None,
                  error_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get errors matching criteria."""
        with self._lock:
            filtered_errors = []
            
            for error in reversed(self.errors):
                if len(filtered_errors) >= limit:
                    break
                
                error_time = datetime.fromisoformat(error["timestamp"].replace('Z', '+00:00'))
                
                if since and error_time < since:
                    continue
                
                if severity and error["severity"] != severity:
                    continue
                
                if error_type and error["error_type"] != error_type:
                    continue
                
                filtered_errors.append(error)
            
            return filtered_errors
    
    def get_error_patterns(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Analyze error patterns."""
        errors = self.get_errors(since=since, limit=10000)
        
        if not errors:
            return {"message": "No errors found"}
        
        # Group by error type
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        hourly_counts = defaultdict(int)
        
        for error in errors:
            error_types[error["error_type"]] += 1
            severity_counts[error["severity"]] += 1
            
            # Group by hour for trend analysis
            error_time = datetime.fromisoformat(error["timestamp"].replace('Z', '+00:00'))
            hour_key = error_time.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1
        
        return {
            "total_errors": len(errors),
            "unique_error_types": len(error_types),
            "error_types": dict(error_types),
            "severity_distribution": dict(severity_counts),
            "hourly_distribution": dict(hourly_counts),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
            "error_rate_per_hour": len(errors) / max(len(hourly_counts), 1)
        }


class AgentDebugger:
    """Main debugger class that coordinates all debugging functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.tracer = RequestTracer(
            max_traces=self.config.get("max_traces", 10000),
            retention_hours=self.config.get("trace_retention_hours", 24)
        )
        
        self.profiler = PerformanceProfiler(
            max_metrics=self.config.get("max_metrics", 50000)
        )
        
        self.memory_monitor = MemoryMonitor(
            max_snapshots=self.config.get("max_memory_snapshots", 1000)
        )
        
        self.error_analyzer = ErrorAnalyzer(
            max_errors=self.config.get("max_errors", 5000)
        )
        
        # Debugging state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._breakpoints: Dict[str, List[Callable]] = defaultdict(list)
        self._watch_expressions: Dict[str, str] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the debugger and background tasks."""
        logger.info("Starting UAP Agent Debugger")
        
        # Start background tasks
        await self.tracer.start_cleanup_task()
        
        if self.config.get("auto_memory_monitoring", True):
            await self.memory_monitor.start_monitoring(
                interval_seconds=self.config.get("memory_monitoring_interval", 60)
            )
        
        logger.info("Agent Debugger started successfully")
    
    async def stop(self) -> None:
        """Stop the debugger and clean up."""
        logger.info("Stopping UAP Agent Debugger")
        
        # Stop background tasks
        self.tracer.stop_cleanup_task()
        self.memory_monitor.stop_monitoring()
        
        # Cancel any remaining tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        logger.info("Agent Debugger stopped")
    
    @asynccontextmanager
    async def trace_request(self, agent_id: str, framework: str, request_data: Dict[str, Any]):
        """Context manager for tracing requests."""
        trace_id = self.tracer.start_trace(agent_id, framework, request_data)
        start_time = time.perf_counter()
        
        try:
            yield trace_id
            
            # Record success
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.tracer.complete_trace(trace_id, {"status": "success"}, duration_ms)
            self.profiler.record_metric("request_duration", duration_ms, 
                                      tags={"agent_id": agent_id, "framework": framework})
            
        except Exception as e:
            # Record error
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.tracer.error_trace(trace_id, str(e), duration_ms)
            self.error_analyzer.record_error(e, {
                "agent_id": agent_id,
                "framework": framework,
                "trace_id": trace_id
            })
            raise
    
    def add_breakpoint(self, pattern: str, callback: Callable) -> None:
        """Add a debugging breakpoint."""
        self._breakpoints[pattern].append(callback)
    
    def remove_breakpoint(self, pattern: str, callback: Optional[Callable] = None) -> None:
        """Remove a debugging breakpoint."""
        if callback:
            if callback in self._breakpoints[pattern]:
                self._breakpoints[pattern].remove(callback)
        else:
            self._breakpoints[pattern].clear()
    
    def add_watch(self, name: str, expression: str) -> None:
        """Add a watch expression."""
        self._watch_expressions[name] = expression
    
    def remove_watch(self, name: str) -> None:
        """Remove a watch expression."""
        self._watch_expressions.pop(name, None)
    
    def get_debug_state(self) -> Dict[str, Any]:
        """Get current debugging state."""
        return {
            "active_traces": len(self.tracer.active_traces),
            "total_traces": len(self.tracer.traces),
            "active_breakpoints": {pattern: len(callbacks) for pattern, callbacks in self._breakpoints.items()},
            "watch_expressions": list(self._watch_expressions.keys()),
            "memory_snapshots": len(self.memory_monitor.snapshots),
            "recorded_errors": len(self.error_analyzer.errors),
            "active_sessions": len(self.active_sessions)
        }
    
    def export_debug_data(self, output_path: str, include_traces: bool = True, 
                         include_metrics: bool = True, include_memory: bool = True,
                         include_errors: bool = True) -> None:
        """Export debug data to a file."""
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "debugger_config": self.config,
            "debug_state": self.get_debug_state()
        }
        
        if include_traces:
            traces = self.tracer.get_traces(limit=10000)
            export_data["traces"] = [asdict(trace) for trace in traces]
        
        if include_metrics:
            metrics = self.profiler.get_metrics(limit=10000)
            export_data["metrics"] = [asdict(metric) for metric in metrics]
        
        if include_memory:
            snapshots = self.memory_monitor.get_snapshots(limit=1000)
            export_data["memory_snapshots"] = [asdict(snapshot) for snapshot in snapshots]
        
        if include_errors:
            errors = self.error_analyzer.get_errors(limit=5000)
            export_data["errors"] = errors
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Debug data exported to: {output_path}")


# Global debugger instance
_global_debugger: Optional[AgentDebugger] = None


def get_debugger() -> AgentDebugger:
    """Get the global debugger instance."""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = AgentDebugger()
    return _global_debugger


def initialize_debugger(config: Optional[Dict[str, Any]] = None) -> AgentDebugger:
    """Initialize the global debugger with configuration."""
    global _global_debugger
    _global_debugger = AgentDebugger(config)
    return _global_debugger


# Convenience functions
async def trace_agent_request(agent_id: str, framework: str, request_data: Dict[str, Any]):
    """Convenience function for tracing agent requests."""
    debugger = get_debugger()
    return debugger.trace_request(agent_id, framework, request_data)


def record_performance_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Convenience function for recording performance metrics."""
    debugger = get_debugger()
    debugger.profiler.record_metric(metric_name, value, tags)


def take_memory_snapshot(agent_context: Optional[Dict[str, Any]] = None):
    """Convenience function for taking memory snapshots."""
    debugger = get_debugger()
    return debugger.memory_monitor.take_snapshot(agent_context)


def record_error(error: Exception, context: Dict[str, Any]) -> str:
    """Convenience function for recording errors."""
    debugger = get_debugger()
    return debugger.error_analyzer.record_error(error, context)