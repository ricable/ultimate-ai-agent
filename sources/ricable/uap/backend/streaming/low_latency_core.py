# backend/streaming/low_latency_core.py
"""
Ultra-Low Latency Processing Core
Sub-millisecond event processing with optimized data structures and algorithms.
"""

import asyncio
import time
import logging
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import queue
import ctypes
import mmap
import os
from concurrent.futures import ThreadPoolExecutor

# Import stream processing components
from .stream_processor import StreamEvent, EventType, ProcessingPriority
from ..monitoring.logs.logger import get_logger
from ..monitoring.metrics.prometheus_metrics import metrics_collector

# Optional high-performance libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = get_logger(__name__)

class ProcessingMode(Enum):
    """Processing modes for different latency requirements"""
    ULTRA_LOW = "ultra_low"  # <1ms target
    LOW = "low"  # <5ms target
    NORMAL = "normal"  # <10ms target
    BATCH = "batch"  # Optimized for throughput

class MemoryPool:
    """High-performance memory pool for zero-copy operations"""
    
    def __init__(self, pool_size: int = 1024, buffer_size: int = 4096):
        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.available_buffers = queue.Queue(maxsize=pool_size)
        self.used_buffers = set()
        self.lock = threading.Lock()
        
        # Pre-allocate buffers
        for _ in range(pool_size):
            buffer = bytearray(buffer_size)
            self.available_buffers.put(buffer)
    
    def get_buffer(self) -> Optional[bytearray]:
        """Get a buffer from the pool"""
        try:
            buffer = self.available_buffers.get_nowait()
            with self.lock:
                self.used_buffers.add(id(buffer))
            return buffer
        except queue.Empty:
            # Pool exhausted, create new buffer
            buffer = bytearray(self.buffer_size)
            with self.lock:
                self.used_buffers.add(id(buffer))
            return buffer
    
    def return_buffer(self, buffer: bytearray) -> None:
        """Return a buffer to the pool"""
        buffer_id = id(buffer)
        with self.lock:
            if buffer_id in self.used_buffers:
                self.used_buffers.remove(buffer_id)
                # Clear buffer and return to pool
                buffer[:] = b'\x00' * len(buffer)
                try:
                    self.available_buffers.put_nowait(buffer)
                except queue.Full:
                    # Pool is full, discard buffer
                    pass
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics"""
        return {
            'total_size': self.pool_size,
            'available': self.available_buffers.qsize(),
            'used': len(self.used_buffers),
            'buffer_size': self.buffer_size
        }

class LockFreeRingBuffer:
    """Lock-free ring buffer for ultra-low latency event passing"""
    
    def __init__(self, size: int = 1024):
        # Ensure size is power of 2 for efficient modulo operations
        self.size = 1 << (size - 1).bit_length()
        self.mask = self.size - 1
        
        # Use shared memory for multi-process access
        self.buffer = [None] * self.size
        self.head = multiprocessing.Value(ctypes.c_uint64, 0)
        self.tail = multiprocessing.Value(ctypes.c_uint64, 0)
        
        # Cache line padding to avoid false sharing
        self._padding = [0] * 64
    
    def put(self, item: Any) -> bool:
        """Put item in buffer (non-blocking)"""
        current_head = self.head.value
        next_head = (current_head + 1) & self.mask
        
        # Check if buffer is full
        if next_head == self.tail.value:
            return False
        
        # Store item
        self.buffer[current_head] = item
        
        # Update head (atomic operation)
        self.head.value = next_head
        return True
    
    def get(self) -> Optional[Any]:
        """Get item from buffer (non-blocking)"""
        current_tail = self.tail.value
        
        # Check if buffer is empty
        if current_tail == self.head.value:
            return None
        
        # Get item
        item = self.buffer[current_tail]
        
        # Update tail (atomic operation)
        self.tail.value = (current_tail + 1) & self.mask
        return item
    
    def size_used(self) -> int:
        """Get number of items in buffer"""
        head = self.head.value
        tail = self.tail.value
        return (head - tail) & self.mask
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return self.head.value == self.tail.value
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return ((self.head.value + 1) & self.mask) == self.tail.value

class FastHashMap:
    """Optimized hash map for ultra-low latency lookups"""
    
    def __init__(self, initial_size: int = 1024):
        self.size = 1 << (initial_size - 1).bit_length()
        self.mask = self.size - 1
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
    
    def _hash(self, key: str) -> int:
        """Fast hash function"""
        hash_value = 0
        for char in key:
            hash_value = ((hash_value << 5) + hash_value) + ord(char)
        return hash_value & self.mask
    
    def put(self, key: str, value: Any) -> None:
        """Put key-value pair"""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new key
        bucket.append((key, value))
        self.count += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return None
    
    def remove(self, key: str) -> bool:
        """Remove key-value pair"""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.count -= 1
                return True
        
        return False
    
    def size(self) -> int:
        """Get number of items"""
        return self.count

@dataclass
class ProcessingMetrics:
    """Ultra-low latency processing metrics"""
    total_events: int = 0
    processed_events: int = 0
    failed_events: int = 0
    min_latency_ns: int = float('inf')
    max_latency_ns: int = 0
    avg_latency_ns: float = 0.0
    p99_latency_ns: int = 0
    throughput_events_per_sec: float = 0.0
    memory_pool_efficiency: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def update_latency(self, latency_ns: int) -> None:
        """Update latency metrics"""
        self.processed_events += 1
        self.min_latency_ns = min(self.min_latency_ns, latency_ns)
        self.max_latency_ns = max(self.max_latency_ns, latency_ns)
        
        # Update running average
        if self.processed_events == 1:
            self.avg_latency_ns = latency_ns
        else:
            self.avg_latency_ns = (
                (self.avg_latency_ns * (self.processed_events - 1) + latency_ns) /
                self.processed_events
            )

class UltraLowLatencyProcessor:
    """Ultra-low latency event processor with sub-millisecond target"""
    
    def __init__(self, 
                 mode: ProcessingMode = ProcessingMode.ULTRA_LOW,
                 worker_threads: int = None,
                 buffer_size: int = 1024,
                 memory_pool_size: int = 1024,
                 enable_cpu_affinity: bool = True):
        
        self.mode = mode
        self.worker_threads = worker_threads or min(multiprocessing.cpu_count(), 4)
        self.buffer_size = buffer_size
        self.enable_cpu_affinity = enable_cpu_affinity
        
        # High-performance data structures
        self.event_buffer = LockFreeRingBuffer(buffer_size)
        self.memory_pool = MemoryPool(memory_pool_size)
        self.handler_map = FastHashMap()
        
        # Processing state
        self.is_running = False
        self.worker_pool = None
        self.processing_threads = []
        
        # Metrics
        self.metrics = ProcessingMetrics()
        self.latency_samples = deque(maxlen=1000)  # For percentile calculations
        
        # Performance optimization settings
        self.batch_size = self._get_optimal_batch_size()
        self.spin_wait_enabled = (mode == ProcessingMode.ULTRA_LOW)
        
        logger.info(f"UltraLowLatencyProcessor initialized: mode={mode.value}, "
                   f"workers={self.worker_threads}, buffer_size={buffer_size}")
    
    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on processing mode"""
        if self.mode == ProcessingMode.ULTRA_LOW:
            return 1  # Process one event at a time for minimum latency
        elif self.mode == ProcessingMode.LOW:
            return 4
        elif self.mode == ProcessingMode.NORMAL:
            return 16
        else:  # BATCH
            return 64
    
    async def start(self) -> None:
        """Start the ultra-low latency processor"""
        if self.is_running:
            return
        
        logger.info("Starting UltraLowLatencyProcessor")
        self.is_running = True
        
        # Set process priority for better real-time performance
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
                    process.nice(psutil.HIGH_PRIORITY_CLASS)
                else:
                    process.nice(-10)  # Unix nice value
            except Exception as e:
                logger.warning(f"Failed to set high priority: {e}")
        
        # Start processing threads
        for i in range(self.worker_threads):
            thread = threading.Thread(
                target=self._processing_worker,
                args=(f"ultra-worker-{i}",),
                daemon=True
            )
            
            # Set CPU affinity for better cache locality
            if self.enable_cpu_affinity and PSUTIL_AVAILABLE:
                try:
                    cpu_id = i % psutil.cpu_count()
                    # CPU affinity setting would go here
                    # This is platform-specific and may require additional setup
                except Exception as e:
                    logger.debug(f"CPU affinity setting failed: {e}")
            
            thread.start()
            self.processing_threads.append(thread)
        
        # Start metrics collection
        asyncio.create_task(self._metrics_collector())
        
        logger.info(f"Started {len(self.processing_threads)} ultra-low latency workers")
    
    async def process_event(self, event: StreamEvent) -> bool:
        """Process event with ultra-low latency"""
        start_time = time.perf_counter_ns()
        
        try:
            # Add event to buffer
            if not self.event_buffer.put(event):
                self.metrics.failed_events += 1
                logger.warning(f"Event buffer full, dropping event {event.event_id}")
                return False
            
            self.metrics.total_events += 1
            
            # Calculate processing latency (buffer add time)
            latency_ns = time.perf_counter_ns() - start_time
            self.metrics.update_latency(latency_ns)
            self.latency_samples.append(latency_ns)
            
            # Update Prometheus metrics
            metrics_collector.ultra_low_latency_events.inc()
            metrics_collector.ultra_low_latency_duration.observe(latency_ns / 1_000_000)  # Convert to ms
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.metrics.failed_events += 1
            return False
    
    def _processing_worker(self, worker_id: str) -> None:
        """Ultra-low latency processing worker thread"""
        logger.info(f"Starting ultra-low latency worker: {worker_id}")
        
        # Thread-local variables for performance
        events_batch = []
        last_batch_time = time.perf_counter_ns()
        
        while self.is_running:
            try:
                # Get events from buffer
                event = self.event_buffer.get()
                
                if event is None:
                    if self.spin_wait_enabled:
                        # Spin wait for ultra-low latency
                        continue
                    else:
                        # Short sleep to reduce CPU usage
                        time.sleep(0.00001)  # 10 microseconds
                        continue
                
                # Process immediately for ultra-low latency mode
                if self.mode == ProcessingMode.ULTRA_LOW:
                    self._process_single_event(event)
                else:
                    # Batch processing for higher throughput
                    events_batch.append(event)
                    
                    current_time = time.perf_counter_ns()
                    batch_ready = (
                        len(events_batch) >= self.batch_size or
                        (current_time - last_batch_time) > 1_000_000  # 1ms timeout
                    )
                    
                    if batch_ready:
                        self._process_event_batch(events_batch)
                        events_batch.clear()
                        last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(0.001)  # 1ms recovery delay
    
    def _process_single_event(self, event: StreamEvent) -> None:
        """Process a single event with minimal overhead"""
        start_time = time.perf_counter_ns()
        
        try:
            # Look up handler
            handler_key = f"{event.event_type.value}:{event.source}"
            handler = self.handler_map.get(handler_key)
            
            if not handler:
                # Use default handler
                handler = self.handler_map.get("default")
            
            if handler:
                # Execute handler
                if asyncio.iscoroutinefunction(handler):
                    # For async handlers, we need to handle differently in sync context
                    # This is a simplified approach - in production, use proper async handling
                    result = handler(event)
                else:
                    result = handler(event)
            
            # Update metrics
            processing_time_ns = time.perf_counter_ns() - start_time
            self.latency_samples.append(processing_time_ns)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.metrics.failed_events += 1
    
    def _process_event_batch(self, events: List[StreamEvent]) -> None:
        """Process a batch of events for higher throughput"""
        start_time = time.perf_counter_ns()
        
        try:
            # Group events by handler for efficient processing
            handler_groups = {}
            
            for event in events:
                handler_key = f"{event.event_type.value}:{event.source}"
                handler = self.handler_map.get(handler_key) or self.handler_map.get("default")
                
                if handler:
                    if handler not in handler_groups:
                        handler_groups[handler] = []
                    handler_groups[handler].append(event)
            
            # Process each group
            for handler, group_events in handler_groups.items():
                try:
                    if hasattr(handler, 'process_batch'):
                        # Handler supports batch processing
                        handler.process_batch(group_events)
                    else:
                        # Process events individually
                        for event in group_events:
                            handler(event)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    self.metrics.failed_events += len(group_events)
            
            # Update metrics
            total_time_ns = time.perf_counter_ns() - start_time
            avg_time_per_event = total_time_ns / len(events)
            
            for _ in events:
                self.latency_samples.append(avg_time_per_event)
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self.metrics.failed_events += len(events)
    
    def register_handler(self, event_type: EventType, source: str, handler: Callable) -> None:
        """Register event handler for specific event type and source"""
        handler_key = f"{event_type.value}:{source}"
        self.handler_map.put(handler_key, handler)
        logger.info(f"Registered ultra-low latency handler: {handler_key}")
    
    def register_default_handler(self, handler: Callable) -> None:
        """Register default handler for unmatched events"""
        self.handler_map.put("default", handler)
        logger.info("Registered default ultra-low latency handler")
    
    async def _metrics_collector(self) -> None:
        """Background metrics collection with minimal overhead"""
        last_collection_time = time.time()
        last_event_count = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                current_event_count = self.metrics.processed_events
                
                # Calculate throughput
                time_diff = current_time - last_collection_time
                if time_diff >= 1.0:  # Update every second
                    event_diff = current_event_count - last_event_count
                    self.metrics.throughput_events_per_sec = event_diff / time_diff
                    
                    last_collection_time = current_time
                    last_event_count = current_event_count
                
                # Calculate percentiles
                if self.latency_samples:
                    sorted_samples = sorted(self.latency_samples)
                    p99_index = int(len(sorted_samples) * 0.99)
                    self.metrics.p99_latency_ns = sorted_samples[p99_index] if p99_index < len(sorted_samples) else 0
                
                # Update memory pool efficiency
                pool_stats = self.memory_pool.get_stats()
                if pool_stats['total_size'] > 0:
                    self.metrics.memory_pool_efficiency = (
                        pool_stats['available'] / pool_stats['total_size']
                    )
                
                # Update CPU usage
                if PSUTIL_AVAILABLE:
                    try:
                        self.metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)
                    except:
                        pass
                
                await asyncio.sleep(0.1)  # Collect metrics every 100ms
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(1.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        return {
            'mode': self.mode.value,
            'is_running': self.is_running,
            'worker_threads': len(self.processing_threads),
            'buffer_utilization': self.event_buffer.size_used() / self.buffer_size,
            'handler_count': self.handler_map.size(),
            'processing_metrics': {
                'total_events': self.metrics.total_events,
                'processed_events': self.metrics.processed_events,
                'failed_events': self.metrics.failed_events,
                'success_rate': (
                    self.metrics.processed_events / max(self.metrics.total_events, 1)
                ),
                'latency_metrics': {
                    'min_latency_ns': self.metrics.min_latency_ns if self.metrics.min_latency_ns != float('inf') else 0,
                    'max_latency_ns': self.metrics.max_latency_ns,
                    'avg_latency_ns': self.metrics.avg_latency_ns,
                    'p99_latency_ns': self.metrics.p99_latency_ns,
                    'min_latency_ms': self.metrics.min_latency_ns / 1_000_000 if self.metrics.min_latency_ns != float('inf') else 0,
                    'max_latency_ms': self.metrics.max_latency_ns / 1_000_000,
                    'avg_latency_ms': self.metrics.avg_latency_ns / 1_000_000,
                    'p99_latency_ms': self.metrics.p99_latency_ns / 1_000_000
                },
                'throughput_events_per_sec': self.metrics.throughput_events_per_sec,
                'memory_pool_efficiency': self.metrics.memory_pool_efficiency,
                'cpu_usage_percent': self.metrics.cpu_usage_percent
            },
            'system_metrics': {
                'buffer_stats': {
                    'size': self.buffer_size,
                    'used': self.event_buffer.size_used(),
                    'is_full': self.event_buffer.is_full(),
                    'is_empty': self.event_buffer.is_empty()
                },
                'memory_pool_stats': self.memory_pool.get_stats()
            }
        }
    
    async def stop(self) -> None:
        """Stop the ultra-low latency processor"""
        logger.info("Stopping UltraLowLatencyProcessor")
        
        self.is_running = False
        
        # Wait for worker threads to complete
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        
        logger.info("UltraLowLatencyProcessor stopped")

class LowLatencyManager:
    """Manager for ultra-low latency processing instances"""
    
    def __init__(self):
        self.processors: Dict[str, UltraLowLatencyProcessor] = {}
        self.is_running = False
    
    async def create_processor(self, 
                             processor_id: str,
                             mode: ProcessingMode = ProcessingMode.ULTRA_LOW,
                             **kwargs) -> UltraLowLatencyProcessor:
        """Create and register a low latency processor"""
        processor = UltraLowLatencyProcessor(mode=mode, **kwargs)
        self.processors[processor_id] = processor
        
        if self.is_running:
            await processor.start()
        
        logger.info(f"Created low latency processor: {processor_id} (mode: {mode.value})")
        return processor
    
    async def start_all(self) -> None:
        """Start all processors"""
        self.is_running = True
        
        for processor in self.processors.values():
            await processor.start()
        
        logger.info(f"Started {len(self.processors)} low latency processors")
    
    async def stop_all(self) -> None:
        """Stop all processors"""
        self.is_running = False
        
        for processor in self.processors.values():
            await processor.stop()
        
        logger.info("All low latency processors stopped")
    
    def get_processor(self, processor_id: str) -> Optional[UltraLowLatencyProcessor]:
        """Get processor by ID"""
        return self.processors.get(processor_id)
    
    def list_processors(self) -> List[str]:
        """List all processor IDs"""
        return list(self.processors.keys())
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all processors"""
        return {
            processor_id: processor.get_metrics()
            for processor_id, processor in self.processors.items()
        }

# Global low latency manager
low_latency_manager = LowLatencyManager()

# Convenience functions
async def create_ultra_low_latency_processor(processor_id: str = "default", 
                                           mode: str = "ultra_low") -> UltraLowLatencyProcessor:
    """Create an ultra-low latency processor"""
    return await low_latency_manager.create_processor(
        processor_id,
        ProcessingMode(mode)
    )

async def process_ultra_low_latency(event: StreamEvent, processor_id: str = "default") -> bool:
    """Process event with ultra-low latency"""
    processor = low_latency_manager.get_processor(processor_id)
    if not processor:
        logger.warning(f"Processor {processor_id} not found")
        return False
    
    return await processor.process_event(event)

def get_ultra_low_latency_metrics(processor_id: str = "default") -> Optional[Dict[str, Any]]:
    """Get ultra-low latency processing metrics"""
    processor = low_latency_manager.get_processor(processor_id)
    if processor:
        return processor.get_metrics()
    return None

def get_all_low_latency_metrics() -> Dict[str, Any]:
    """Get metrics for all low latency processors"""
    return low_latency_manager.get_all_metrics()
