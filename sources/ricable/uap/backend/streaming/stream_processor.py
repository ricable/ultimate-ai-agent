# backend/streaming/stream_processor.py
"""
Real-Time Stream Processing Core
Ultra-low latency event processing with sub-millisecond response times.
"""

import asyncio
import time
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, AsyncIterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import heapq

# Import performance monitoring
try:
    import psutil
    import resource
    PERFORMANCE_MONITORING = True
except ImportError:
    PERFORMANCE_MONITORING = False

# Import Redis for distributed streaming
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import existing components
from ..distributed.ray_manager import ray_cluster_manager, submit_distributed_task
from ..edge.edge_manager import EdgeManager
from ..monitoring.logs.logger import get_logger
from ..monitoring.metrics.prometheus_metrics import metrics_collector

logger = get_logger(__name__)

class EventType(Enum):
    """Types of streaming events"""
    DATA = "data"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    AGGREGATION = "aggregation"
    ANOMALY = "anomaly"
    TRIGGER = "trigger"

class ProcessingPriority(Enum):
    """Event processing priorities"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class StreamEvent:
    """Streaming event data structure"""
    event_id: str
    event_type: EventType
    timestamp: float
    data: Any
    source: str
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    metadata: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    ttl_ms: Optional[int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
    
    def is_expired(self) -> bool:
        """Check if event has expired based on TTL"""
        if self.ttl_ms is None:
            return False
        return (time.time() * 1000) - (self.timestamp * 1000) > self.ttl_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'data': self.data,
            'source': self.source,
            'priority': self.priority.value,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'ttl_ms': self.ttl_ms
        }

@dataclass
class StreamMetrics:
    """Stream processing metrics"""
    events_processed: int = 0
    events_failed: int = 0
    total_processing_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    max_processing_time_ms: float = 0.0
    throughput_events_per_sec: float = 0.0
    buffer_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def update_processing_time(self, processing_time_ms: float):
        """Update processing time metrics"""
        self.events_processed += 1
        self.total_processing_time_ms += processing_time_ms
        self.avg_processing_time_ms = self.total_processing_time_ms / self.events_processed
        self.min_processing_time_ms = min(self.min_processing_time_ms, processing_time_ms)
        self.max_processing_time_ms = max(self.max_processing_time_ms, processing_time_ms)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class StreamBuffer:
    """Ultra-low latency circular buffer for stream events"""
    
    def __init__(self, max_size: int = 10000, enable_persistence: bool = False):
        self.max_size = max_size
        self.enable_persistence = enable_persistence
        self.buffer = deque(maxlen=max_size)
        self.priority_queue = []
        self.event_index = {}
        self.lock = threading.RLock()
        
        # Performance optimization
        self._buffer_array = [None] * max_size
        self._head = 0
        self._tail = 0
        self._size = 0
    
    def put(self, event: StreamEvent) -> bool:
        """Add event to buffer with priority ordering"""
        with self.lock:
            if self._size >= self.max_size:
                # Remove oldest event
                self._remove_oldest()
            
            # Add to circular buffer
            self._buffer_array[self._tail] = event
            self._tail = (self._tail + 1) % self.max_size
            self._size += 1
            
            # Add to priority queue
            heapq.heappush(self.priority_queue, (event.priority.value, event.timestamp, event))
            
            # Index for fast lookup
            self.event_index[event.event_id] = event
            
            return True
    
    def get_next_priority(self) -> Optional[StreamEvent]:
        """Get next event by priority"""
        with self.lock:
            while self.priority_queue:
                priority, timestamp, event = heapq.heappop(self.priority_queue)
                
                # Check if event is still valid (not expired)
                if not event.is_expired() and event.event_id in self.event_index:
                    return event
            
            return None
    
    def get_fifo(self) -> Optional[StreamEvent]:
        """Get next event in FIFO order"""
        with self.lock:
            if self._size == 0:
                return None
            
            event = self._buffer_array[self._head]
            self._buffer_array[self._head] = None
            self._head = (self._head + 1) % self.max_size
            self._size -= 1
            
            # Remove from index
            if event and event.event_id in self.event_index:
                del self.event_index[event.event_id]
            
            return event
    
    def peek(self, count: int = 1) -> List[StreamEvent]:
        """Peek at next events without removing them"""
        with self.lock:
            events = []
            for i in range(min(count, self._size)):
                idx = (self._head + i) % self.max_size
                event = self._buffer_array[idx]
                if event:
                    events.append(event)
            return events
    
    def size(self) -> int:
        """Get current buffer size"""
        return self._size
    
    def utilization(self) -> float:
        """Get buffer utilization percentage"""
        return (self._size / self.max_size) * 100.0
    
    def clear_expired(self) -> int:
        """Remove expired events from buffer"""
        with self.lock:
            expired_count = 0
            # Clean up expired events from index
            expired_ids = []
            for event_id, event in self.event_index.items():
                if event.is_expired():
                    expired_ids.append(event_id)
                    expired_count += 1
            
            for event_id in expired_ids:
                del self.event_index[event_id]
            
            return expired_count
    
    def _remove_oldest(self):
        """Remove oldest event to make space"""
        if self._size > 0:
            old_event = self._buffer_array[self._head]
            if old_event and old_event.event_id in self.event_index:
                del self.event_index[old_event.event_id]
            
            self._buffer_array[self._head] = None
            self._head = (self._head + 1) % self.max_size
            self._size -= 1

class StreamProcessor:
    """Ultra-low latency stream processor"""
    
    def __init__(self, 
                 buffer_size: int = 10000,
                 worker_threads: int = 4,
                 enable_distributed: bool = True,
                 enable_edge_processing: bool = True,
                 redis_url: str = "redis://localhost:6379"):
        
        self.buffer_size = buffer_size
        self.worker_threads = worker_threads
        self.enable_distributed = enable_distributed
        self.enable_edge_processing = enable_edge_processing
        self.redis_url = redis_url
        
        # Core components
        self.buffer = StreamBuffer(buffer_size)
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.global_handlers: List[Callable] = []
        self.metrics = StreamMetrics()
        
        # Processing state
        self.is_running = False
        self.worker_pool = ThreadPoolExecutor(max_workers=worker_threads)
        self.processing_tasks = []
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Performance monitoring
        self.last_throughput_check = time.time()
        self.throughput_counter = 0
        
        # Edge processing integration
        self.edge_manager: Optional[EdgeManager] = None
        
        # Weak references for cleanup
        self._cleanup_refs = weakref.WeakSet()
    
    async def initialize(self) -> None:
        """Initialize stream processor"""
        logger.info("Initializing StreamProcessor")
        
        try:
            # Initialize Redis for distributed processing
            if REDIS_AVAILABLE and self.enable_distributed:
                self.redis_client = await aioredis.from_url(self.redis_url)
                logger.info("Redis connection established for distributed streaming")
            
            # Initialize edge processing
            if self.enable_edge_processing:
                # Edge manager would be injected in production
                logger.info("Edge processing enabled")
            
            # Start processing workers
            self.is_running = True
            await self._start_workers()
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collector())
            
            logger.info("StreamProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize StreamProcessor: {e}")
            raise
    
    async def process_event(self, event: StreamEvent) -> None:
        """Process a single event with ultra-low latency"""
        start_time = time.perf_counter()
        
        try:
            # Add to buffer
            if not self.buffer.put(event):
                logger.warning(f"Failed to buffer event {event.event_id}")
                self.metrics.events_failed += 1
                return
            
            # Update throughput counter
            self.throughput_counter += 1
            
            # Record processing time (buffer add time)
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.update_processing_time(processing_time_ms)
            
            # Update Prometheus metrics
            metrics_collector.stream_events_processed.inc()
            metrics_collector.stream_processing_latency.observe(processing_time_ms / 1000.0)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.metrics.events_failed += 1
    
    async def process_batch(self, events: List[StreamEvent]) -> None:
        """Process multiple events as a batch for efficiency"""
        start_time = time.perf_counter()
        
        try:
            # Sort events by priority
            events.sort(key=lambda e: (e.priority.value, e.timestamp))
            
            # Process each event
            for event in events:
                await self.process_event(event)
            
            # Update batch metrics
            batch_time_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Processed batch of {len(events)} events in {batch_time_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
    
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register event handler for specific event type"""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type.value}")
    
    def register_global_handler(self, handler: Callable) -> None:
        """Register global event handler (processes all events)"""
        self.global_handlers.append(handler)
        logger.info("Registered global event handler")
    
    async def create_stream(self, stream_id: str, 
                          source_config: Dict[str, Any]) -> 'StreamHandle':
        """Create a new event stream"""
        return StreamHandle(stream_id, self, source_config)
    
    async def get_metrics(self) -> StreamMetrics:
        """Get current stream processing metrics"""
        # Update system metrics if available
        if PERFORMANCE_MONITORING:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
        
        # Update buffer utilization
        self.metrics.buffer_utilization = self.buffer.utilization()
        
        return self.metrics
    
    async def _start_workers(self) -> None:
        """Start background processing workers"""
        for i in range(self.worker_threads):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.processing_tasks.append(task)
        
        logger.info(f"Started {self.worker_threads} processing workers")
    
    async def _worker(self, worker_id: str) -> None:
        """Background worker for processing events"""
        logger.info(f"Starting stream worker: {worker_id}")
        
        while self.is_running:
            try:
                # Get next event from buffer
                event = self.buffer.get_next_priority()
                
                if event is None:
                    # No events available, short sleep
                    await asyncio.sleep(0.001)  # 1ms
                    continue
                
                # Process event with handlers
                await self._process_event_with_handlers(event)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.01)  # 10ms recovery delay
    
    async def _process_event_with_handlers(self, event: StreamEvent) -> None:
        """Process event with registered handlers"""
        start_time = time.perf_counter()
        
        try:
            # Process with type-specific handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Handler error for event {event.event_id}: {e}")
            
            # Process with global handlers
            for handler in self.global_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Global handler error for event {event.event_id}: {e}")
            
            # Check for distributed processing
            if self.enable_distributed and event.priority in [ProcessingPriority.CRITICAL, ProcessingPriority.HIGH]:
                await self._distribute_event(event)
            
            # Update processing metrics
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.update_processing_time(processing_time_ms)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.metrics.events_failed += 1
    
    async def _distribute_event(self, event: StreamEvent) -> None:
        """Distribute event processing to Ray cluster"""
        try:
            if self.enable_distributed:
                # Submit to Ray cluster for distributed processing
                task_id = await submit_distributed_task(
                    task_type="stream_event_processing",
                    task_function=self._distributed_event_processor,
                    input_data={
                        'event': event.to_dict(),
                        'timestamp': time.time()
                    },
                    priority=1 if event.priority == ProcessingPriority.CRITICAL else 0
                )
                
                logger.debug(f"Distributed event {event.event_id} with task ID: {task_id}")
                
        except Exception as e:
            logger.error(f"Failed to distribute event {event.event_id}: {e}")
    
    def _distributed_event_processor(self, event_data: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Distributed event processor function for Ray"""
        try:
            # Reconstruct event
            event_dict = event_data['event']
            event = StreamEvent(
                event_id=event_dict['event_id'],
                event_type=EventType(event_dict['event_type']),
                timestamp=event_dict['timestamp'],
                data=event_dict['data'],
                source=event_dict['source'],
                priority=ProcessingPriority(event_dict['priority']),
                metadata=event_dict.get('metadata', {}),
                correlation_id=event_dict.get('correlation_id'),
                ttl_ms=event_dict.get('ttl_ms')
            )
            
            # Process event (simplified for distributed context)
            processing_start = time.time()
            
            # Simulate processing
            result = {
                'event_id': event.event_id,
                'processed_at': time.time(),
                'processing_time_ms': (time.time() - processing_start) * 1000,
                'result': f"Distributed processing completed for {event.event_type.value}"
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'event_id': event_data.get('event', {}).get('event_id', 'unknown')
            }
    
    async def _metrics_collector(self) -> None:
        """Background metrics collection task"""
        while self.is_running:
            try:
                # Calculate throughput
                current_time = time.time()
                time_diff = current_time - self.last_throughput_check
                
                if time_diff >= 1.0:  # Update every second
                    self.metrics.throughput_events_per_sec = self.throughput_counter / time_diff
                    self.throughput_counter = 0
                    self.last_throughput_check = current_time
                
                # Clean up expired events
                expired_count = self.buffer.clear_expired()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired events")
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(5.0)
    
    async def shutdown(self) -> None:
        """Shutdown stream processor"""
        logger.info("Shutting down StreamProcessor")
        
        self.is_running = False
        
        # Wait for workers to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.worker_pool.shutdown(wait=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("StreamProcessor shutdown complete")

class StreamHandle:
    """Handle for managing individual event streams"""
    
    def __init__(self, stream_id: str, processor: StreamProcessor, config: Dict[str, Any]):
        self.stream_id = stream_id
        self.processor = processor
        self.config = config
        self.is_active = True
        self.event_count = 0
        self.last_event_time = None
    
    async def emit(self, event_type: EventType, data: Any, 
                   priority: ProcessingPriority = ProcessingPriority.NORMAL,
                   correlation_id: Optional[str] = None,
                   ttl_ms: Optional[int] = None) -> str:
        """Emit an event to the stream"""
        if not self.is_active:
            raise RuntimeError(f"Stream {self.stream_id} is not active")
        
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            source=self.stream_id,
            priority=priority,
            correlation_id=correlation_id,
            ttl_ms=ttl_ms
        )
        
        await self.processor.process_event(event)
        
        self.event_count += 1
        self.last_event_time = time.time()
        
        return event.event_id
    
    async def emit_batch(self, events_data: List[Dict[str, Any]]) -> List[str]:
        """Emit multiple events as a batch"""
        events = []
        event_ids = []
        
        for event_data in events_data:
            event_id = str(uuid.uuid4())
            event = StreamEvent(
                event_id=event_id,
                event_type=EventType(event_data.get('event_type', 'data')),
                timestamp=time.time(),
                data=event_data['data'],
                source=self.stream_id,
                priority=ProcessingPriority(event_data.get('priority', ProcessingPriority.NORMAL.value)),
                correlation_id=event_data.get('correlation_id'),
                ttl_ms=event_data.get('ttl_ms')
            )
            events.append(event)
            event_ids.append(event_id)
        
        await self.processor.process_batch(events)
        
        self.event_count += len(events)
        self.last_event_time = time.time()
        
        return event_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        return {
            'stream_id': self.stream_id,
            'is_active': self.is_active,
            'event_count': self.event_count,
            'last_event_time': self.last_event_time,
            'config': self.config
        }
    
    async def close(self) -> None:
        """Close the stream"""
        self.is_active = False
        logger.info(f"Stream {self.stream_id} closed")

# Global stream processor instance
stream_processor = StreamProcessor()

# Convenience functions
async def initialize_streaming() -> None:
    """Initialize global stream processor"""
    await stream_processor.initialize()

async def process_stream_event(event_type: str, data: Any, 
                              source: str = "system",
                              priority: str = "normal") -> str:
    """Process a single stream event"""
    event = StreamEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType(event_type),
        timestamp=time.time(),
        data=data,
        source=source,
        priority=ProcessingPriority[priority.upper()]
    )
    
    await stream_processor.process_event(event)
    return event.event_id

async def get_streaming_metrics() -> Dict[str, Any]:
    """Get streaming metrics"""
    metrics = await stream_processor.get_metrics()
    return metrics.to_dict()
