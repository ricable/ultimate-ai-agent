# File: backend/optimization/memory_manager.py
"""
Memory management and optimization for UAP platform.
Provides memory monitoring, garbage collection optimization, and resource cleanup.
"""

import gc
import os
import psutil
import asyncio
import logging
import weakref
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import tracemalloc
from contextlib import contextmanager
import threading
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class MemoryThresholds:
    """Memory usage thresholds for different actions"""
    warning_percent: float = 75.0
    critical_percent: float = 90.0
    cleanup_percent: float = 85.0
    gc_trigger_percent: float = 80.0
    
    # Absolute thresholds in bytes
    max_cache_size: int = 512 * 1024 * 1024  # 512MB
    max_connection_memory: int = 100 * 1024 * 1024  # 100MB
    max_temp_objects: int = 10000

@dataclass 
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    process_memory_percent: float
    gc_collections: Dict[int, int]
    tracked_objects: int
    
    @property
    def memory_pressure(self) -> str:
        """Get memory pressure level"""
        if self.memory_percent >= 90:
            return "critical"
        elif self.memory_percent >= 75:
            return "high"
        elif self.memory_percent >= 50:
            return "medium"
        else:
            return "low"

class ObjectTracker:
    """Track object lifecycle and memory usage"""
    
    def __init__(self):
        self.tracked_objects: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        self.object_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'created': 0,
            'destroyed': 0,
            'peak_count': 0,
            'current_count': 0,
            'total_memory': 0
        })
        self.memory_hotspots: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def track_object(self, obj: Any, category: str = None):
        """Track an object"""
        if category is None:
            category = type(obj).__name__
        
        with self.lock:
            self.tracked_objects[category].add(obj)
            stats = self.object_stats[category]
            stats['created'] += 1
            stats['current_count'] = len(self.tracked_objects[category])
            stats['peak_count'] = max(stats['peak_count'], stats['current_count'])
            
            # Estimate memory usage
            try:
                obj_size = sys.getsizeof(obj)
                stats['total_memory'] += obj_size
            except (TypeError, AttributeError):
                pass
    
    def untrack_object(self, obj: Any, category: str = None):
        """Untrack an object"""
        if category is None:
            category = type(obj).__name__
        
        with self.lock:
            try:
                self.tracked_objects[category].remove(obj)
                stats = self.object_stats[category]
                stats['destroyed'] += 1
                stats['current_count'] = len(self.tracked_objects[category])
            except KeyError:
                pass
    
    def get_object_counts(self) -> Dict[str, int]:
        """Get current object counts by category"""
        with self.lock:
            return {category: len(objects) for category, objects in self.tracked_objects.items()}
    
    def get_memory_usage_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Get memory usage statistics by category"""
        with self.lock:
            result = {}
            for category, stats in self.object_stats.items():
                current_count = len(self.tracked_objects[category])
                result[category] = {
                    **stats,
                    'current_count': current_count,
                    'avg_size_bytes': stats['total_memory'] / max(1, stats['created'])
                }
            return result
    
    def find_memory_leaks(self) -> List[Dict[str, Any]]:
        """Identify potential memory leaks"""
        leaks = []
        
        with self.lock:
            for category, stats in self.object_stats.items():
                current_count = len(self.tracked_objects[category])
                
                # Check for objects that are created but never destroyed
                if stats['created'] > 0 and stats['destroyed'] == 0 and current_count > 100:
                    leaks.append({
                        'category': category,
                        'type': 'no_cleanup',
                        'current_count': current_count,
                        'created': stats['created'],
                        'destroyed': stats['destroyed']
                    })
                
                # Check for rapidly growing object counts
                if current_count > stats['peak_count'] * 0.9 and current_count > 1000:
                    leaks.append({
                        'category': category,
                        'type': 'growing_count',
                        'current_count': current_count,
                        'peak_count': stats['peak_count'],
                        'growth_rate': current_count / max(1, stats['destroyed'])
                    })
        
        return leaks

class GarbageCollectionOptimizer:
    """Optimize garbage collection behavior"""
    
    def __init__(self):
        self.gc_stats = {
            'collections': defaultdict(int),
            'objects_collected': defaultdict(int),
            'time_spent': defaultdict(float),
            'forced_collections': 0
        }
        self.gc_thresholds = gc.get_threshold()
        self.adaptive_gc = True
    
    def optimize_thresholds(self, memory_pressure: str):
        """Adjust GC thresholds based on memory pressure"""
        if not self.adaptive_gc:
            return
        
        base_threshold = self.gc_thresholds[0]
        
        if memory_pressure == "critical":
            # Aggressive collection
            new_thresholds = (base_threshold // 4, base_threshold // 2, base_threshold)
        elif memory_pressure == "high":
            # More frequent collection
            new_thresholds = (base_threshold // 2, base_threshold, base_threshold * 2)
        elif memory_pressure == "low":
            # Less frequent collection for better performance
            new_thresholds = (base_threshold * 2, base_threshold * 4, base_threshold * 8)
        else:
            # Normal thresholds
            new_thresholds = self.gc_thresholds
        
        gc.set_threshold(*new_thresholds)
        logger.debug(f"GC thresholds adjusted for {memory_pressure} pressure: {new_thresholds}")
    
    def force_collection(self, generation: int = None) -> Dict[str, int]:
        """Force garbage collection and return statistics"""
        self.gc_stats['forced_collections'] += 1
        
        if generation is not None:
            collected = gc.collect(generation)
            self.gc_stats['collections'][generation] += 1
            self.gc_stats['objects_collected'][generation] += collected
            return {f'generation_{generation}': collected}
        else:
            # Collect all generations
            results = {}
            for gen in range(3):
                collected = gc.collect(gen)
                self.gc_stats['collections'][gen] += 1
                self.gc_stats['objects_collected'][gen] += collected
                results[f'generation_{gen}'] = collected
            return results
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        gc_counts = gc.get_count()
        gc_threshold = gc.get_threshold()
        
        return {
            'current_counts': gc_counts,
            'thresholds': gc_threshold,
            'collections_by_generation': dict(self.gc_stats['collections']),
            'objects_collected_by_generation': dict(self.gc_stats['objects_collected']),
            'forced_collections': self.gc_stats['forced_collections'],
            'adaptive_gc_enabled': self.adaptive_gc
        }

class MemoryProfiler:
    """Memory profiling and analysis"""
    
    def __init__(self):
        self.profiling_enabled = False
        self.snapshots: List[Any] = []
        self.memory_traces: List[Dict[str, Any]] = []
        
    def start_profiling(self):
        """Start memory profiling"""
        if not self.profiling_enabled:
            tracemalloc.start()
            self.profiling_enabled = True
            logger.info("Memory profiling started")
    
    def stop_profiling(self):
        """Stop memory profiling"""
        if self.profiling_enabled:
            tracemalloc.stop()
            self.profiling_enabled = False
            logger.info("Memory profiling stopped")
    
    def take_snapshot(self) -> Optional[Any]:
        """Take a memory snapshot"""
        if not self.profiling_enabled:
            return None
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'timestamp': datetime.utcnow(),
            'snapshot': snapshot
        })
        
        # Keep only last 10 snapshots
        if len(self.snapshots) > 10:
            self.snapshots.pop(0)
        
        return snapshot
    
    def analyze_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze top memory allocations"""
        if not self.snapshots:
            return []
        
        latest_snapshot = self.snapshots[-1]['snapshot']
        top_stats = latest_snapshot.statistics('lineno')
        
        results = []
        for stat in top_stats[:limit]:
            results.append({
                'filename': stat.traceback.format()[0] if stat.traceback.format() else 'Unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count,
                'average_size': stat.size / max(1, stat.count)
            })
        
        return results
    
    def compare_snapshots(self) -> Optional[List[Dict[str, Any]]]:
        """Compare latest two snapshots to find memory growth"""
        if len(self.snapshots) < 2:
            return None
        
        snapshot1 = self.snapshots[-2]['snapshot']
        snapshot2 = self.snapshots[-1]['snapshot']
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        results = []
        for stat in top_stats[:10]:
            results.append({
                'filename': stat.traceback.format()[0] if stat.traceback.format() else 'Unknown',
                'size_diff_mb': stat.size_diff / 1024 / 1024,
                'count_diff': stat.count_diff,
                'size_mb': stat.size / 1024 / 1024
            })
        
        return results

class ResourceCleaner:
    """Clean up various resources to free memory"""
    
    def __init__(self):
        self.cleanup_handlers: Dict[str, Callable] = {}
        self.cleanup_stats = {
            'total_cleanups': 0,
            'memory_freed': 0,
            'objects_cleaned': 0
        }
    
    def register_cleanup_handler(self, name: str, handler: Callable):
        """Register a cleanup handler"""
        self.cleanup_handlers[name] = handler
        logger.info(f"Registered cleanup handler: {name}")
    
    async def perform_cleanup(self, memory_pressure: str) -> Dict[str, Any]:
        """Perform resource cleanup based on memory pressure"""
        self.cleanup_stats['total_cleanups'] += 1
        cleanup_results = {}
        
        logger.info(f"Starting resource cleanup (pressure: {memory_pressure})")
        
        # Run cleanup handlers
        for name, handler in self.cleanup_handlers.items():
            try:
                result = await handler(memory_pressure) if asyncio.iscoroutinefunction(handler) else handler(memory_pressure)
                cleanup_results[name] = result
                
                if isinstance(result, dict):
                    self.cleanup_stats['memory_freed'] += result.get('memory_freed', 0)
                    self.cleanup_stats['objects_cleaned'] += result.get('objects_cleaned', 0)
                    
            except Exception as e:
                logger.error(f"Cleanup handler '{name}' failed: {e}")
                cleanup_results[name] = {'error': str(e)}
        
        return cleanup_results
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics"""
        return {
            **self.cleanup_stats,
            'registered_handlers': list(self.cleanup_handlers.keys())
        }

class MemoryManager:
    """Main memory management coordinator"""
    
    def __init__(self, thresholds: MemoryThresholds = None):
        self.thresholds = thresholds or MemoryThresholds()
        self.object_tracker = ObjectTracker()
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.profiler = MemoryProfiler()
        self.resource_cleaner = ResourceCleaner()
        
        self.monitoring_enabled = False
        self.monitor_task = None
        self.snapshots: List[MemorySnapshot] = []
        
        self.manager_stats = {
            'started_at': datetime.utcnow(),
            'monitoring_cycles': 0,
            'cleanup_triggers': 0,
            'gc_optimizations': 0,
            'memory_warnings': 0
        }
        
        # Register default cleanup handlers
        self._register_default_cleanups()
    
    def _register_default_cleanups(self):
        """Register default cleanup handlers"""
        
        def cleanup_caches(pressure: str) -> Dict[str, Any]:
            """Clean up various caches"""
            cleaned = 0
            memory_freed = 0
            
            # Clear function caches
            if hasattr(gc, 'get_objects'):
                for obj in gc.get_objects():
                    if hasattr(obj, 'cache_clear') and callable(obj.cache_clear):
                        try:
                            obj.cache_clear()
                            cleaned += 1
                        except:
                            pass
            
            return {
                'objects_cleaned': cleaned,
                'memory_freed': memory_freed,
                'pressure': pressure
            }
        
        def cleanup_temp_objects(pressure: str) -> Dict[str, Any]:
            """Clean up temporary objects"""
            # This would implement cleanup of application-specific temp objects
            return {
                'objects_cleaned': 0,
                'memory_freed': 0,
                'pressure': pressure
            }
        
        self.resource_cleaner.register_cleanup_handler('caches', cleanup_caches)
        self.resource_cleaner.register_cleanup_handler('temp_objects', cleanup_temp_objects)
    
    async def start_monitoring(self, interval: int = 30):
        """Start memory monitoring"""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        
        # Start profiling if needed
        self.profiler.start_profiling()
        
        logger.info(f"Memory monitoring started (interval: {interval}s)")
    
    async def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_enabled = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.profiler.stop_profiling()
        logger.info("Memory monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                await self._perform_monitoring_cycle()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _perform_monitoring_cycle(self):
        """Perform one monitoring cycle"""
        self.manager_stats['monitoring_cycles'] += 1
        
        # Take memory snapshot
        snapshot = self.take_memory_snapshot()
        
        # Check memory pressure and take action
        pressure = snapshot.memory_pressure
        
        if pressure in ['high', 'critical']:
            self.manager_stats['memory_warnings'] += 1
            logger.warning(f"Memory pressure: {pressure} ({snapshot.memory_percent:.1f}%)")
            
            if pressure == 'critical' or snapshot.memory_percent >= self.thresholds.cleanup_percent:
                self.manager_stats['cleanup_triggers'] += 1
                await self.resource_cleaner.perform_cleanup(pressure)
            
            if snapshot.memory_percent >= self.thresholds.gc_trigger_percent:
                self.manager_stats['gc_optimizations'] += 1
                self.gc_optimizer.optimize_thresholds(pressure)
                self.gc_optimizer.force_collection()
        else:
            # Optimize GC for normal conditions
            self.gc_optimizer.optimize_thresholds(pressure)
        
        # Take profiling snapshot
        self.profiler.take_snapshot()
        
        # Check for memory leaks
        leaks = self.object_tracker.find_memory_leaks()
        if leaks:
            logger.warning(f"Potential memory leaks detected: {len(leaks)} categories")
    
    def take_memory_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot"""
        # System memory info
        memory = psutil.virtual_memory()
        
        # Process memory info
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        # Garbage collection info
        gc_stats = {i: gc.get_count()[i] for i in range(3)}
        
        snapshot = MemorySnapshot(
            timestamp=datetime.utcnow(),
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            process_memory=process_memory.rss,
            process_memory_percent=process_memory.rss / memory.total * 100,
            gc_collections=gc_stats,
            tracked_objects=sum(self.object_tracker.get_object_counts().values())
        )
        
        # Store snapshot (keep last 100)
        self.snapshots.append(snapshot)
        if len(self.snapshots) > 100:
            self.snapshots.pop(0)
        
        return snapshot
    
    @contextmanager
    def track_memory_usage(self, operation_name: str):
        """Context manager to track memory usage of an operation"""
        start_snapshot = self.take_memory_snapshot()
        start_time = datetime.utcnow()
        
        try:
            yield
        finally:
            end_snapshot = self.take_memory_snapshot()
            end_time = datetime.utcnow()
            
            memory_diff = end_snapshot.process_memory - start_snapshot.process_memory
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Memory usage for '{operation_name}': "
                       f"{memory_diff / 1024 / 1024:.2f}MB in {duration:.2f}s")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory management statistics"""
        latest_snapshot = self.snapshots[-1] if self.snapshots else None
        
        return {
            'memory_manager': {
                'monitoring_enabled': self.monitoring_enabled,
                'uptime_seconds': (datetime.utcnow() - self.manager_stats['started_at']).total_seconds(),
                **self.manager_stats,
                'current_memory_pressure': latest_snapshot.memory_pressure if latest_snapshot else 'unknown'
            },
            'current_memory': {
                'process_memory_mb': latest_snapshot.process_memory / 1024 / 1024 if latest_snapshot else 0,
                'system_memory_percent': latest_snapshot.memory_percent if latest_snapshot else 0,
                'available_memory_mb': latest_snapshot.available_memory / 1024 / 1024 if latest_snapshot else 0
            },
            'object_tracking': self.object_tracker.get_memory_usage_by_category(),
            'garbage_collection': self.gc_optimizer.get_gc_stats(),
            'resource_cleanup': self.resource_cleaner.get_cleanup_stats(),
            'memory_leaks': self.object_tracker.find_memory_leaks(),
            'thresholds': {
                'warning_percent': self.thresholds.warning_percent,
                'critical_percent': self.thresholds.critical_percent,
                'cleanup_percent': self.thresholds.cleanup_percent
            }
        }
    
    async def force_cleanup(self) -> Dict[str, Any]:
        """Force immediate cleanup regardless of memory pressure"""
        logger.info("Forcing immediate memory cleanup")
        
        # Force garbage collection
        gc_results = self.gc_optimizer.force_collection()
        
        # Force resource cleanup
        cleanup_results = await self.resource_cleaner.perform_cleanup('forced')
        
        # Take new snapshot
        snapshot = self.take_memory_snapshot()
        
        return {
            'garbage_collection': gc_results,
            'resource_cleanup': cleanup_results,
            'memory_after_cleanup': {
                'process_memory_mb': snapshot.process_memory / 1024 / 1024,
                'memory_percent': snapshot.memory_percent,
                'pressure': snapshot.memory_pressure
            }
        }

# Memory management decorators
def track_memory_usage(category: str = None):
    """Decorator to track memory usage of function calls"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            obj_category = category or f"{func.__module__}.{func.__name__}"
            
            # Track function call as an object
            memory_manager.object_tracker.track_object(func, obj_category)
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                memory_manager.object_tracker.untrack_object(func, obj_category)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            obj_category = category or f"{func.__module__}.{func.__name__}"
            
            memory_manager.object_tracker.track_object(func, obj_category)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                memory_manager.object_tracker.untrack_object(func, obj_category)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def monitor_memory_growth(threshold_mb: float = 10.0):
    """Decorator to monitor memory growth during function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with memory_manager.track_memory_usage(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Global memory manager instance
memory_manager = MemoryManager()

# Export components
__all__ = [
    'MemoryManager',
    'MemoryThresholds', 
    'MemorySnapshot',
    'ObjectTracker',
    'GarbageCollectionOptimizer',
    'MemoryProfiler',
    'ResourceCleaner',
    'memory_manager',
    'track_memory_usage',
    'monitor_memory_growth'
]
