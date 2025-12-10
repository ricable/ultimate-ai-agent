# edge/inference/inference_optimizer.py
"""
Inference Optimization for Edge Devices
Optimizes inference performance through various techniques including batching,
caching, and hardware-specific optimizations.
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import heapq
from abc import ABC, abstractmethod

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

logger = logging.getLogger(__name__)

class OptimizationTechnique(Enum):
    """Inference optimization techniques"""
    DYNAMIC_BATCHING = "dynamic_batching"
    RESULT_CACHING = "result_caching"
    MODEL_SWITCHING = "model_switching"
    HARDWARE_ACCELERATION = "hardware_acceleration"
    MEMORY_OPTIMIZATION = "memory_optimization"
    PIPELINE_PARALLELISM = "pipeline_parallelism"
    ADAPTIVE_PRECISION = "adaptive_precision"
    PREEMPTIVE_LOADING = "preemptive_loading"

class HardwareTarget(Enum):
    """Hardware optimization targets"""
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    EDGE_TPU = "edge_tpu"
    APPLE_ANE = "apple_ane"
    INTEL_OPENVINO = "intel_openvino"
    ARM_NN = "arm_nn"

@dataclass
class OptimizationConfig:
    """Configuration for inference optimization"""
    techniques: List[OptimizationTechnique]
    hardware_target: HardwareTarget = HardwareTarget.CPU
    max_batch_size: int = 8
    batch_timeout_ms: int = 10
    cache_size_mb: int = 100
    cache_ttl_minutes: int = 60
    memory_limit_mb: int = 512
    enable_model_warming: bool = True
    performance_target: str = "balanced"  # latency, throughput, balanced
    adaptive_optimization: bool = True
    
    def __post_init__(self):
        if not isinstance(self.techniques, list):
            self.techniques = [self.techniques]

@dataclass
class BatchRequest:
    """Batched inference request"""
    request_id: str
    model_id: str
    input_data: Any
    priority: int = 0
    submitted_at: float = 0.0
    timeout_ms: int = 5000
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.submitted_at == 0.0:
            self.submitted_at = time.time()

@dataclass
class BatchResult:
    """Result of batched inference"""
    request_id: str
    success: bool
    output_data: Any = None
    error_message: Optional[str] = None
    inference_time_ms: float = 0.0
    batch_size: int = 1
    cache_hit: bool = False
    optimization_applied: List[str] = None
    
    def __post_init__(self):
        if self.optimization_applied is None:
            self.optimization_applied = []

class ResultCache:
    """LRU cache for inference results"""
    
    def __init__(self, max_size_mb: int = 100, ttl_minutes: int = 60):
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_minutes * 60
        self.cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (result, timestamp, size_bytes)
        self.access_order = deque()  # For LRU eviction
        self.current_size_bytes = 0
        self.lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        with self.lock:
            self.stats['total_requests'] += 1
            
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            result, timestamp, size_bytes = self.cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                self._evict_key(key)
                self.stats['misses'] += 1
                return None
            
            # Update access order for LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats['hits'] += 1
            return result
    
    def put(self, key: str, result: Any) -> None:
        """Put result in cache"""
        with self.lock:
            # Estimate result size
            result_size = self._estimate_size(result)
            
            # Check if we need to evict
            while (self.current_size_bytes + result_size > self.max_size_mb * 1024 * 1024 and
                   self.access_order):
                oldest_key = self.access_order.popleft()
                self._evict_key(oldest_key)
            
            # Store result
            self.cache[key] = (result, time.time(), result_size)
            self.current_size_bytes += result_size
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def _evict_key(self, key: str) -> None:
        """Evict key from cache"""
        if key in self.cache:
            _, _, size_bytes = self.cache[key]
            self.current_size_bytes -= size_bytes
            del self.cache[key]
            self.stats['evictions'] += 1
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        if NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            # Rough estimate for other objects
            return len(str(obj)) * 4  # Assuming 4 bytes per character
    
    def clear(self) -> None:
        """Clear all cached results"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = self.stats['hits'] / max(self.stats['total_requests'], 1)
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'current_size_mb': self.current_size_bytes / (1024 * 1024),
                'cache_entries': len(self.cache)
            }

class DynamicBatcher:
    """Dynamic batching for inference requests"""
    
    def __init__(self, 
                 max_batch_size: int = 8,
                 timeout_ms: int = 10,
                 model_id: str = "default"):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.model_id = model_id
        
        # Batching state
        self.pending_requests: List[BatchRequest] = []
        self.batch_queue = asyncio.Queue()
        self.is_running = False
        self.batch_task = None
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_wait_time_ms': 0.0,
            'timeouts': 0
        }
        
        self.lock = threading.Lock()
    
    async def start(self) -> None:
        """Start the dynamic batcher"""
        if self.is_running:
            return
        
        self.is_running = True
        self.batch_task = asyncio.create_task(self._batch_worker())
        logger.info(f"Started dynamic batcher for model {self.model_id}")
    
    async def add_request(self, request: BatchRequest) -> None:
        """Add request to batch"""
        with self.lock:
            self.pending_requests.append(request)
            self.metrics['total_requests'] += 1
        
        # Check if we should trigger batching
        if len(self.pending_requests) >= self.max_batch_size:
            await self._trigger_batch()
    
    async def _batch_worker(self) -> None:
        """Background worker for batch processing"""
        while self.is_running:
            try:
                # Wait for timeout or until max batch size
                await asyncio.sleep(self.timeout_ms / 1000.0)
                
                if self.pending_requests:
                    await self._trigger_batch()
                    
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                await asyncio.sleep(0.1)
    
    async def _trigger_batch(self) -> None:
        """Trigger batch processing"""
        with self.lock:
            if not self.pending_requests:
                return
            
            # Create batch
            batch_size = min(len(self.pending_requests), self.max_batch_size)
            batch_requests = self.pending_requests[:batch_size]
            self.pending_requests = self.pending_requests[batch_size:]
            
            # Update metrics
            self.metrics['total_batches'] += 1
            current_avg = self.metrics['avg_batch_size']
            batch_count = self.metrics['total_batches']
            self.metrics['avg_batch_size'] = (
                (current_avg * (batch_count - 1) + batch_size) / batch_count
            )
        
        # Add batch to processing queue
        await self.batch_queue.put(batch_requests)
    
    async def get_next_batch(self) -> Optional[List[BatchRequest]]:
        """Get next batch for processing"""
        try:
            return await asyncio.wait_for(self.batch_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batching metrics"""
        with self.lock:
            return {
                **self.metrics,
                'pending_requests': len(self.pending_requests),
                'queue_size': self.batch_queue.qsize()
            }
    
    async def stop(self) -> None:
        """Stop the dynamic batcher"""
        self.is_running = False
        if self.batch_task:
            await self.batch_task

class InferenceOptimizer:
    """Main inference optimizer combining multiple optimization techniques"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.techniques = config.techniques
        
        # Optimization components
        self.result_cache = None
        self.dynamic_batchers: Dict[str, DynamicBatcher] = {}
        self.model_cache: Dict[str, Any] = {}  # Loaded models cache
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.optimization_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'batch_optimizations': 0,
            'hardware_accelerations': 0,
            'avg_latency_improvement': 0.0,
            'memory_saved_mb': 0.0
        }
        
        # Initialize components based on configuration
        self._initialize_optimizations()
        
        logger.info(f"InferenceOptimizer initialized with techniques: {[t.value for t in self.techniques]}")
    
    def _initialize_optimizations(self) -> None:
        """Initialize optimization components"""
        if OptimizationTechnique.RESULT_CACHING in self.techniques:
            self.result_cache = ResultCache(
                max_size_mb=self.config.cache_size_mb,
                ttl_minutes=self.config.cache_ttl_minutes
            )
        
        # Other initializations can be added here
    
    async def optimize_inference(self, 
                               model_id: str,
                               input_data: Any,
                               inference_func: Callable,
                               **kwargs) -> BatchResult:
        """Optimize inference request using configured techniques"""
        start_time = time.perf_counter()
        request_id = kwargs.get('request_id', f"req_{int(time.time() * 1000000)}")
        
        try:
            self.optimization_stats['total_requests'] += 1
            optimizations_applied = []
            
            # Check cache first
            if self.result_cache and OptimizationTechnique.RESULT_CACHING in self.techniques:
                cache_key = self._generate_cache_key(model_id, input_data)
                cached_result = self.result_cache.get(cache_key)
                
                if cached_result is not None:
                    self.optimization_stats['cache_hits'] += 1
                    optimizations_applied.append('result_caching')
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    return BatchResult(
                        request_id=request_id,
                        success=True,
                        output_data=cached_result,
                        inference_time_ms=latency_ms,
                        cache_hit=True,
                        optimization_applied=optimizations_applied
                    )
            
            # Apply dynamic batching if enabled
            if OptimizationTechnique.DYNAMIC_BATCHING in self.techniques:
                result = await self._apply_dynamic_batching(
                    model_id, input_data, inference_func, request_id, **kwargs
                )
                if result:
                    optimizations_applied.append('dynamic_batching')
                    result.optimization_applied = optimizations_applied
                    return result
            
            # Apply hardware acceleration if available
            if OptimizationTechnique.HARDWARE_ACCELERATION in self.techniques:
                input_data = await self._apply_hardware_acceleration(input_data)
                optimizations_applied.append('hardware_acceleration')
                self.optimization_stats['hardware_accelerations'] += 1
            
            # Apply memory optimization
            if OptimizationTechnique.MEMORY_OPTIMIZATION in self.techniques:
                input_data = await self._apply_memory_optimization(input_data)
                optimizations_applied.append('memory_optimization')
            
            # Run inference
            inference_start = time.perf_counter()
            output_data = await self._run_optimized_inference(
                inference_func, input_data, **kwargs
            )
            inference_time_ms = (time.perf_counter() - inference_start) * 1000
            
            # Cache result if caching is enabled
            if self.result_cache and OptimizationTechnique.RESULT_CACHING in self.techniques:
                cache_key = self._generate_cache_key(model_id, input_data)
                self.result_cache.put(cache_key, output_data)
            
            # Update performance history
            total_time_ms = (time.perf_counter() - start_time) * 1000
            self.performance_history.append({
                'timestamp': time.time(),
                'latency_ms': total_time_ms,
                'optimizations': optimizations_applied,
                'model_id': model_id
            })
            
            return BatchResult(
                request_id=request_id,
                success=True,
                output_data=output_data,
                inference_time_ms=inference_time_ms,
                optimization_applied=optimizations_applied
            )
            
        except Exception as e:
            logger.error(f"Optimization error for request {request_id}: {e}")
            
            error_time_ms = (time.perf_counter() - start_time) * 1000
            return BatchResult(
                request_id=request_id,
                success=False,
                error_message=str(e),
                inference_time_ms=error_time_ms
            )
    
    async def _apply_dynamic_batching(self, 
                                    model_id: str,
                                    input_data: Any,
                                    inference_func: Callable,
                                    request_id: str,
                                    **kwargs) -> Optional[BatchResult]:
        """Apply dynamic batching optimization"""
        # Get or create batcher for model
        if model_id not in self.dynamic_batchers:
            self.dynamic_batchers[model_id] = DynamicBatcher(
                max_batch_size=self.config.max_batch_size,
                timeout_ms=self.config.batch_timeout_ms,
                model_id=model_id
            )
            await self.dynamic_batchers[model_id].start()
        
        batcher = self.dynamic_batchers[model_id]
        
        # Create batch request
        batch_request = BatchRequest(
            request_id=request_id,
            model_id=model_id,
            input_data=input_data,
            **kwargs
        )
        
        # Add to batch
        await batcher.add_request(batch_request)
        
        # For now, we'll process immediately for simplicity
        # In a full implementation, this would involve a separate batch processor
        batch = await batcher.get_next_batch()
        
        if batch and len(batch) > 1:
            # Process batch
            return await self._process_batch(batch, inference_func)
        
        return None
    
    async def _process_batch(self, 
                           batch: List[BatchRequest],
                           inference_func: Callable) -> BatchResult:
        """Process a batch of requests"""
        try:
            # Combine inputs for batch processing
            if NUMPY_AVAILABLE:
                batch_input = np.stack([req.input_data for req in batch])
            else:
                batch_input = [req.input_data for req in batch]
            
            # Run batched inference
            start_time = time.perf_counter()
            batch_output = await inference_func(batch_input)
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            self.optimization_stats['batch_optimizations'] += 1
            
            # For simplicity, return result for first request
            # In practice, you'd need to split batch results
            return BatchResult(
                request_id=batch[0].request_id,
                success=True,
                output_data=batch_output[0] if isinstance(batch_output, (list, np.ndarray)) else batch_output,
                inference_time_ms=inference_time_ms,
                batch_size=len(batch),
                optimization_applied=['dynamic_batching']
            )
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return BatchResult(
                request_id=batch[0].request_id,
                success=False,
                error_message=str(e),
                batch_size=len(batch)
            )
    
    async def _apply_hardware_acceleration(self, input_data: Any) -> Any:
        """Apply hardware-specific acceleration"""
        # Hardware acceleration logic based on target
        if self.config.hardware_target == HardwareTarget.GPU:
            # GPU acceleration (e.g., move to CUDA)
            pass
        elif self.config.hardware_target == HardwareTarget.EDGE_TPU:
            # Edge TPU optimization
            pass
        elif self.config.hardware_target == HardwareTarget.APPLE_ANE:
            # Apple Neural Engine optimization
            pass
        
        return input_data
    
    async def _apply_memory_optimization(self, input_data: Any) -> Any:
        """Apply memory optimization techniques"""
        # Memory optimization logic
        if NUMPY_AVAILABLE and isinstance(input_data, np.ndarray):
            # Optimize data types if possible
            if input_data.dtype == np.float64:
                input_data = input_data.astype(np.float32)
                self.optimization_stats['memory_saved_mb'] += input_data.nbytes / (1024 * 1024)
        
        return input_data
    
    async def _run_optimized_inference(self, 
                                     inference_func: Callable,
                                     input_data: Any,
                                     **kwargs) -> Any:
        """Run inference with optimizations"""
        if asyncio.iscoroutinefunction(inference_func):
            return await inference_func(input_data, **kwargs)
        else:
            return inference_func(input_data, **kwargs)
    
    def _generate_cache_key(self, model_id: str, input_data: Any) -> str:
        """Generate cache key for input data"""
        # Simple hash-based cache key
        # In practice, you'd want a more robust hashing mechanism
        if NUMPY_AVAILABLE and isinstance(input_data, np.ndarray):
            data_hash = hash(input_data.tobytes())
        else:
            data_hash = hash(str(input_data))
        
        return f"{model_id}:{data_hash}"
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = self.optimization_stats.copy()
        
        # Add cache stats if available
        if self.result_cache:
            stats['cache_stats'] = self.result_cache.get_stats()
        
        # Add batching stats
        stats['batching_stats'] = {
            batcher_id: batcher.get_metrics()
            for batcher_id, batcher in self.dynamic_batchers.items()
        }
        
        # Calculate performance improvements
        if self.performance_history:
            recent_latencies = [p['latency_ms'] for p in list(self.performance_history)[-100:]]
            if recent_latencies:
                stats['avg_recent_latency_ms'] = sum(recent_latencies) / len(recent_latencies)
                stats['min_recent_latency_ms'] = min(recent_latencies)
                stats['max_recent_latency_ms'] = max(recent_latencies)
        
        return stats
    
    async def warm_up_models(self, model_ids: List[str]) -> None:
        """Warm up models for faster inference"""
        if not self.config.enable_model_warming:
            return
        
        logger.info(f"Warming up models: {model_ids}")
        
        for model_id in model_ids:
            try:
                # Model warming logic would go here
                # This could involve loading models into memory,
                # running dummy inferences, etc.
                pass
            except Exception as e:
                logger.error(f"Failed to warm up model {model_id}: {e}")
    
    async def adaptive_optimize(self) -> None:
        """Adaptively adjust optimization parameters based on performance"""
        if not self.config.adaptive_optimization:
            return
        
        # Analyze recent performance
        if len(self.performance_history) < 50:
            return
        
        recent_performance = list(self.performance_history)[-50:]
        avg_latency = sum(p['latency_ms'] for p in recent_performance) / len(recent_performance)
        
        # Adjust parameters based on performance target
        if self.config.performance_target == "latency":
            # Optimize for lower latency
            if avg_latency > 100:  # If average latency > 100ms
                # Reduce batch timeout for faster processing
                for batcher in self.dynamic_batchers.values():
                    batcher.timeout_ms = max(5, batcher.timeout_ms - 1)
        
        elif self.config.performance_target == "throughput":
            # Optimize for higher throughput
            if avg_latency < 50:  # If latency is acceptable
                # Increase batch size for better throughput
                for batcher in self.dynamic_batchers.values():
                    batcher.max_batch_size = min(16, batcher.max_batch_size + 1)
    
    async def cleanup(self) -> None:
        """Cleanup optimizer resources"""
        # Stop all batchers
        for batcher in self.dynamic_batchers.values():
            await batcher.stop()
        
        # Clear caches
        if self.result_cache:
            self.result_cache.clear()
        
        self.model_cache.clear()
        
        logger.info("InferenceOptimizer cleanup complete")

# Global inference optimizer
inference_optimizer = None

# Convenience functions
async def initialize_inference_optimizer(techniques: List[str] = None,
                                       hardware_target: str = "cpu",
                                       max_batch_size: int = 8,
                                       cache_size_mb: int = 100) -> InferenceOptimizer:
    """Initialize global inference optimizer"""
    global inference_optimizer
    
    if techniques is None:
        techniques = ["dynamic_batching", "result_caching"]
    
    config = OptimizationConfig(
        techniques=[OptimizationTechnique(t) for t in techniques],
        hardware_target=HardwareTarget(hardware_target),
        max_batch_size=max_batch_size,
        cache_size_mb=cache_size_mb
    )
    
    inference_optimizer = InferenceOptimizer(config)
    return inference_optimizer

async def optimize_inference_request(model_id: str,
                                   input_data: Any,
                                   inference_func: Callable,
                                   **kwargs) -> BatchResult:
    """Optimize an inference request"""
    if inference_optimizer is None:
        await initialize_inference_optimizer()
    
    return await inference_optimizer.optimize_inference(
        model_id, input_data, inference_func, **kwargs
    )

def get_inference_optimization_stats() -> Dict[str, Any]:
    """Get inference optimization statistics"""
    if inference_optimizer is None:
        return {}
    
    return inference_optimizer.get_optimization_stats()

async def warm_up_inference_models(model_ids: List[str]) -> None:
    """Warm up models for faster inference"""
    if inference_optimizer is None:
        await initialize_inference_optimizer()
    
    await inference_optimizer.warm_up_models(model_ids)
