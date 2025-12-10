# File: backend/optimization/request_batcher.py
"""
Request batching and optimization for UAP platform.
Provides intelligent request batching, deduplication, and parallel processing.
"""

import asyncio
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
from contextlib import asynccontextmanager
import weakref

logger = logging.getLogger(__name__)

class BatchStrategy(Enum):
    TIME_BASED = "time_based"        # Batch based on time windows
    SIZE_BASED = "size_based"        # Batch based on number of requests
    ADAPTIVE = "adaptive"            # Adaptive batching based on load
    SIMILARITY = "similarity"       # Batch similar requests together
    PRIORITY = "priority"           # Batch by priority levels

class RequestPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BatchConfig:
    """Configuration for request batching"""
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    max_batch_size: int = 50
    max_wait_time: float = 0.1  # 100ms
    min_batch_size: int = 1
    
    # Adaptive batching parameters
    load_threshold_low: float = 0.3
    load_threshold_high: float = 0.8
    adaptive_batch_size_min: int = 5
    adaptive_batch_size_max: int = 100
    
    # Similarity batching
    similarity_threshold: float = 0.8
    similarity_key_fields: List[str] = field(default_factory=lambda: ['type', 'category'])
    
    # Request deduplication
    enable_deduplication: bool = True
    dedup_window: float = 5.0  # 5 seconds
    
    # Priority handling
    priority_queues: bool = True
    priority_batch_sizes: Dict[RequestPriority, int] = field(default_factory=lambda: {
        RequestPriority.CRITICAL: 1,
        RequestPriority.HIGH: 5,
        RequestPriority.NORMAL: 20,
        RequestPriority.LOW: 50
    })

@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    data: Dict[str, Any]
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.request_id)
    
    @property
    def age(self) -> float:
        """Get request age in seconds"""
        return time.time() - self.created_at
    
    def get_similarity_key(self, fields: List[str]) -> str:
        """Generate similarity key for grouping"""
        key_data = {}
        for field in fields:
            if field in self.data:
                key_data[field] = self.data[field]
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get_dedup_key(self) -> str:
        """Generate deduplication key"""
        # Use request data for deduplication (excluding metadata)
        dedup_data = {k: v for k, v in self.data.items() if not k.startswith('_')}
        return hashlib.md5(json.dumps(dedup_data, sort_keys=True).encode()).hexdigest()

@dataclass
class RequestBatch:
    """A batch of requests to be processed together"""
    batch_id: str
    requests: List[BatchRequest] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    priority: RequestPriority = RequestPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_request(self, request: BatchRequest) -> bool:
        """Add request to batch, return True if batch is ready"""
        self.requests.append(request)
        # Update batch priority to highest priority request
        if request.priority.value > self.priority.value:
            self.priority = request.priority
        return True
    
    @property
    def size(self) -> int:
        return len(self.requests)
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at
    
    @property
    def average_request_age(self) -> float:
        if not self.requests:
            return 0.0
        return sum(req.age for req in self.requests) / len(self.requests)
    
    def get_request_data(self) -> List[Dict[str, Any]]:
        """Get data from all requests in batch"""
        return [req.data for req in self.requests]
    
    def set_results(self, results: List[Any]):
        """Set results for all requests in batch"""
        for i, request in enumerate(self.requests):
            if i < len(results):
                if not request.future.done():
                    request.future.set_result(results[i])
            else:
                if not request.future.done():
                    request.future.set_exception(IndexError(f"No result for request {i}"))
    
    def set_error(self, error: Exception):
        """Set error for all requests in batch"""
        for request in self.requests:
            if not request.future.done():
                request.future.set_exception(error)

class LoadMonitor:
    """Monitor system load for adaptive batching"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.request_times: deque = deque(maxlen=window_size)
        self.batch_times: deque = deque(maxlen=window_size)
        self.error_rates: deque = deque(maxlen=window_size)
        
    def record_request_time(self, duration: float):
        """Record request processing time"""
        self.request_times.append(duration)
    
    def record_batch_time(self, duration: float):
        """Record batch processing time"""
        self.batch_times.append(duration)
    
    def record_error_rate(self, error_rate: float):
        """Record error rate"""
        self.error_rates.append(error_rate)
    
    def get_load_metrics(self) -> Dict[str, float]:
        """Get current load metrics"""
        if not self.request_times:
            return {'load': 0.0, 'avg_request_time': 0.0, 'avg_batch_time': 0.0, 'error_rate': 0.0}
        
        avg_request_time = sum(self.request_times) / len(self.request_times)
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0.0
        avg_error_rate = sum(self.error_rates) / len(self.error_rates) if self.error_rates else 0.0
        
        # Calculate load as normalized request time (0-1 scale)
        # Assume 1 second is high load
        load = min(avg_request_time / 1.0, 1.0)
        
        return {
            'load': load,
            'avg_request_time': avg_request_time,
            'avg_batch_time': avg_batch_time,
            'error_rate': avg_error_rate
        }

class RequestDeduplicator:
    """Handle request deduplication"""
    
    def __init__(self, window_size: float = 5.0):
        self.window_size = window_size
        self.recent_requests: Dict[str, Tuple[float, List[asyncio.Future]]] = {}
        self.cleanup_interval = 1.0
        self.cleanup_task = None
        
    async def start(self):
        """Start deduplication cleanup task"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop deduplication cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self):
        """Clean up old deduplicated requests"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                current_time = time.time()
                
                expired_keys = []
                for key, (timestamp, futures) in self.recent_requests.items():
                    if current_time - timestamp > self.window_size:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.recent_requests[key]
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Deduplication cleanup error: {e}")
    
    def check_duplicate(self, request: BatchRequest) -> Optional[asyncio.Future]:
        """Check if request is a duplicate, return existing future if found"""
        dedup_key = request.get_dedup_key()
        current_time = time.time()
        
        if dedup_key in self.recent_requests:
            timestamp, futures = self.recent_requests[dedup_key]
            if current_time - timestamp <= self.window_size:
                # Duplicate found, add future to the list
                futures.append(request.future)
                return futures[0]  # Return the original future
        
        # Not a duplicate, record this request
        self.recent_requests[dedup_key] = (current_time, [request.future])
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        total_futures = sum(len(futures) for _, futures in self.recent_requests.values())
        unique_requests = len(self.recent_requests)
        
        return {
            'unique_requests': unique_requests,
            'total_futures': total_futures,
            'deduplication_ratio': (total_futures - unique_requests) / max(1, total_futures),
            'window_size': self.window_size
        }

class SimilarityGrouper:
    """Group similar requests together"""
    
    def __init__(self, threshold: float = 0.8, key_fields: List[str] = None):
        self.threshold = threshold
        self.key_fields = key_fields or ['type', 'category']
        self.similarity_groups: Dict[str, List[BatchRequest]] = defaultdict(list)
    
    def group_request(self, request: BatchRequest) -> str:
        """Group request by similarity, return group key"""
        similarity_key = request.get_similarity_key(self.key_fields)
        self.similarity_groups[similarity_key].append(request)
        return similarity_key
    
    def get_similar_requests(self, group_key: str, max_count: int = None) -> List[BatchRequest]:
        """Get similar requests from a group"""
        requests = self.similarity_groups.get(group_key, [])
        if max_count:
            return requests[:max_count]
        return requests.copy()
    
    def remove_requests(self, group_key: str, requests: List[BatchRequest]):
        """Remove requests from similarity group"""
        if group_key in self.similarity_groups:
            for request in requests:
                try:
                    self.similarity_groups[group_key].remove(request)
                except ValueError:
                    pass
            
            # Clean up empty groups
            if not self.similarity_groups[group_key]:
                del self.similarity_groups[group_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get similarity grouping statistics"""
        return {
            'total_groups': len(self.similarity_groups),
            'total_requests': sum(len(requests) for requests in self.similarity_groups.values()),
            'average_group_size': sum(len(requests) for requests in self.similarity_groups.values()) / max(1, len(self.similarity_groups)),
            'largest_group_size': max((len(requests) for requests in self.similarity_groups.values()), default=0)
        }

class AdaptiveBatcher:
    """Adaptive batching based on system load and request patterns"""
    
    def __init__(self, config: BatchConfig, load_monitor: LoadMonitor):
        self.config = config
        self.load_monitor = load_monitor
        
    def calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on current load"""
        metrics = self.load_monitor.get_load_metrics()
        load = metrics['load']
        
        if load < self.config.load_threshold_low:
            # Low load: use larger batches for efficiency
            return min(self.config.adaptive_batch_size_max, self.config.max_batch_size)
        elif load > self.config.load_threshold_high:
            # High load: use smaller batches for responsiveness
            return max(self.config.adaptive_batch_size_min, self.config.min_batch_size)
        else:
            # Medium load: scale batch size linearly
            range_size = self.config.adaptive_batch_size_max - self.config.adaptive_batch_size_min
            load_factor = (load - self.config.load_threshold_low) / (self.config.load_threshold_high - self.config.load_threshold_low)
            batch_size = self.config.adaptive_batch_size_max - int(range_size * load_factor)
            return max(self.config.min_batch_size, min(batch_size, self.config.max_batch_size))
    
    def calculate_optimal_wait_time(self) -> float:
        """Calculate optimal wait time based on current load"""
        metrics = self.load_monitor.get_load_metrics()
        load = metrics['load']
        
        if load < self.config.load_threshold_low:
            # Low load: wait longer to accumulate more requests
            return self.config.max_wait_time
        elif load > self.config.load_threshold_high:
            # High load: process quickly
            return self.config.max_wait_time * 0.1
        else:
            # Scale wait time based on load
            load_factor = (load - self.config.load_threshold_low) / (self.config.load_threshold_high - self.config.load_threshold_low)
            return self.config.max_wait_time * (1.0 - load_factor * 0.9)

class RequestBatcher:
    """Main request batching coordinator"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.load_monitor = LoadMonitor()
        self.deduplicator = RequestDeduplicator(self.config.dedup_window)
        self.similarity_grouper = SimilarityGrouper(
            self.config.similarity_threshold,
            self.config.similarity_key_fields
        )
        self.adaptive_batcher = AdaptiveBatcher(self.config, self.load_monitor)
        
        # Batch queues by priority
        self.priority_queues: Dict[RequestPriority, List[BatchRequest]] = {
            priority: [] for priority in RequestPriority
        }
        
        # Active batches being processed
        self.active_batches: Dict[str, RequestBatch] = {}
        
        # Batch processors registry
        self.batch_processors: Dict[str, Callable] = {}
        
        # Background tasks
        self.batch_processor_task = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'deduplicated_requests': 0,
            'batches_processed': 0,
            'total_processing_time': 0.0,
            'average_batch_size': 0.0,
            'errors': 0
        }
    
    async def start(self):
        """Start the request batcher"""
        if self.running:
            return
        
        self.running = True
        await self.deduplicator.start()
        
        # Start batch processing task
        self.batch_processor_task = asyncio.create_task(self._batch_processing_loop())
        
        logger.info("Request batcher started")
    
    async def stop(self):
        """Stop the request batcher"""
        self.running = False
        
        # Cancel batch processor task
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        await self.deduplicator.stop()
        
        # Complete any remaining requests with error
        for priority_queue in self.priority_queues.values():
            for request in priority_queue:
                if not request.future.done():
                    request.future.set_exception(RuntimeError("Batcher stopped"))
        
        logger.info("Request batcher stopped")
    
    def register_batch_processor(self, request_type: str, processor: Callable):
        """Register a batch processor for specific request type"""
        self.batch_processors[request_type] = processor
        logger.info(f"Registered batch processor for type: {request_type}")
    
    async def submit_request(self, request_id: str, data: Dict[str, Any], 
                           priority: RequestPriority = RequestPriority.NORMAL) -> Any:
        """Submit a request for batching"""
        if not self.running:
            raise RuntimeError("Request batcher is not running")
        
        self.stats['total_requests'] += 1
        
        # Create batch request
        request = BatchRequest(
            request_id=request_id,
            data=data,
            priority=priority
        )
        
        # Check for deduplication
        if self.config.enable_deduplication:
            existing_future = self.deduplicator.check_duplicate(request)
            if existing_future:
                self.stats['deduplicated_requests'] += 1
                return await existing_future
        
        # Add to appropriate priority queue
        self.priority_queues[priority].append(request)
        self.stats['batched_requests'] += 1
        
        # If it's a critical priority request, process immediately
        if priority == RequestPriority.CRITICAL:
            await self._process_priority_queue(priority)
        
        return await request.future
    
    async def _batch_processing_loop(self):
        """Main batch processing loop"""
        while self.running:
            try:
                # Process queues in priority order
                for priority in sorted(RequestPriority, key=lambda p: p.value, reverse=True):
                    await self._process_priority_queue(priority)
                
                # Wait before next cycle
                optimal_wait = self.adaptive_batcher.calculate_optimal_wait_time()
                await asyncio.sleep(optimal_wait)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_priority_queue(self, priority: RequestPriority):
        """Process requests in a priority queue"""
        queue = self.priority_queues[priority]
        if not queue:
            return
        
        # Determine batch size for this priority
        if self.config.strategy == BatchStrategy.ADAPTIVE:
            max_batch_size = self.adaptive_batcher.calculate_optimal_batch_size()
        else:
            max_batch_size = self.config.priority_batch_sizes.get(priority, self.config.max_batch_size)
        
        # Group requests by similarity if enabled
        if self.config.strategy == BatchStrategy.SIMILARITY:
            await self._process_similarity_groups(queue, max_batch_size)
        else:
            await self._process_sequential_batches(queue, max_batch_size)
    
    async def _process_similarity_groups(self, queue: List[BatchRequest], max_batch_size: int):
        """Process requests grouped by similarity"""
        # Group requests by similarity
        group_keys = set()
        for request in queue.copy():
            group_key = self.similarity_grouper.group_request(request)
            group_keys.add(group_key)
            queue.remove(request)
        
        # Process each similarity group
        for group_key in group_keys:
            similar_requests = self.similarity_grouper.get_similar_requests(group_key, max_batch_size)
            if similar_requests:
                await self._create_and_process_batch(similar_requests)
                self.similarity_grouper.remove_requests(group_key, similar_requests)
    
    async def _process_sequential_batches(self, queue: List[BatchRequest], max_batch_size: int):
        """Process requests in sequential batches"""
        while queue:
            # Determine how many requests to batch
            batch_size = min(len(queue), max_batch_size)
            
            # Check if we should wait for more requests (unless critical priority)
            if (batch_size < self.config.min_batch_size and 
                queue[0].priority != RequestPriority.CRITICAL and
                queue[0].age < self.adaptive_batcher.calculate_optimal_wait_time()):
                break
            
            # Create batch
            batch_requests = queue[:batch_size]
            queue[:batch_size] = []  # Remove from queue
            
            await self._create_and_process_batch(batch_requests)
    
    async def _create_and_process_batch(self, requests: List[BatchRequest]):
        """Create and process a batch of requests"""
        if not requests:
            return
        
        # Create batch
        batch_id = f"batch_{int(time.time() * 1000)}_{len(self.active_batches)}"
        batch = RequestBatch(batch_id=batch_id, strategy=self.config.strategy)
        
        for request in requests:
            batch.add_request(request)
        
        self.active_batches[batch_id] = batch
        
        try:
            # Process batch
            start_time = time.time()
            await self._process_batch(batch)
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['batches_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['average_batch_size'] = (self.stats['average_batch_size'] * (self.stats['batches_processed'] - 1) + batch.size) / self.stats['batches_processed']
            
            # Record metrics for adaptive batching
            self.load_monitor.record_batch_time(processing_time)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            batch.set_error(e)
            self.stats['errors'] += 1
        finally:
            # Remove from active batches
            self.active_batches.pop(batch_id, None)
    
    async def _process_batch(self, batch: RequestBatch):
        """Process a single batch"""
        if not batch.requests:
            return
        
        # Determine request type from first request
        request_type = batch.requests[0].data.get('type', 'default')
        
        # Get appropriate processor
        processor = self.batch_processors.get(request_type)
        if not processor:
            # Default processor: process each request individually
            results = []
            for request in batch.requests:
                try:
                    # This would be replaced with actual processing logic
                    result = await self._default_process_request(request.data)
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e)})
        else:
            # Use registered batch processor
            batch_data = batch.get_request_data()
            results = await processor(batch_data)
        
        # Set results
        batch.set_results(results)
    
    async def _default_process_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Default request processor (placeholder)"""
        # This would be replaced with actual request processing logic
        await asyncio.sleep(0.01)  # Simulate processing time
        return {'processed': True, 'data': data}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive batching statistics"""
        active_requests = sum(len(queue) for queue in self.priority_queues.values())
        
        avg_processing_time = (self.stats['total_processing_time'] / 
                             max(1, self.stats['batches_processed']))
        
        return {
            'request_batcher': {
                'running': self.running,
                'total_requests': self.stats['total_requests'],
                'batched_requests': self.stats['batched_requests'],
                'deduplicated_requests': self.stats['deduplicated_requests'],
                'batches_processed': self.stats['batches_processed'],
                'average_batch_size': round(self.stats['average_batch_size'], 2),
                'average_processing_time_ms': round(avg_processing_time * 1000, 2),
                'error_count': self.stats['errors'],
                'active_requests': active_requests,
                'active_batches': len(self.active_batches)
            },
            'load_metrics': self.load_monitor.get_load_metrics(),
            'deduplication': self.deduplicator.get_stats(),
            'similarity_grouping': self.similarity_grouper.get_stats(),
            'priority_queues': {priority.name: len(queue) for priority, queue in self.priority_queues.items()},
            'config': {
                'strategy': self.config.strategy.value,
                'max_batch_size': self.config.max_batch_size,
                'max_wait_time': self.config.max_wait_time,
                'deduplication_enabled': self.config.enable_deduplication
            }
        }

# Context manager for batch processing
@asynccontextmanager
async def batch_context(batcher: RequestBatcher, request_type: str):
    """Context manager for batch processing operations"""
    start_time = time.time()
    try:
        yield batcher
    finally:
        duration = time.time() - start_time
        batcher.load_monitor.record_request_time(duration)

# Global request batcher instance
request_batcher = RequestBatcher()

# Export components
__all__ = [
    'RequestBatcher',
    'BatchConfig',
    'BatchStrategy',
    'RequestPriority',
    'BatchRequest',
    'RequestBatch',
    'LoadMonitor',
    'RequestDeduplicator',
    'SimilarityGrouper',
    'AdaptiveBatcher',
    'request_batcher',
    'batch_context'
]
