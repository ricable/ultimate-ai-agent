"""
Memory Management Optimizer for MLX Distributed System
Implements intelligent memory pooling, quantization, activation caching, and GC optimization
"""

import gc
import mmap
import os
import threading
import time
import weakref
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import psutil
import numpy as np
from enum import Enum
import json
import pickle

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map, tree_flatten, tree_unflatten
except ImportError:
    # Fallback for testing without MLX
    class MockMLX:
        def array(self, data): return np.array(data)
        def zeros(self, shape): return np.zeros(shape)
    mx = MockMLX()

logger = logging.getLogger(__name__)

class QuantizationLevel(Enum):
    """Quantization precision levels"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    INT4 = "int4"

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory_gb: float
    used_memory_gb: float
    available_memory_gb: float
    fragmentation_ratio: float
    pool_utilization: float
    gc_pressure: float
    cache_hit_ratio: float

@dataclass
class MemoryPool:
    """Memory pool for efficient allocation"""
    pool_id: str
    size_bytes: int
    used_bytes: int = 0
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    free_blocks: List[Dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

class QuantizationEngine:
    """Advanced quantization engine for model compression"""
    
    def __init__(self):
        self.quantization_cache = {}
        self.calibration_data = {}
        self.quantization_stats = {
            "models_quantized": 0,
            "memory_saved_gb": 0.0,
            "performance_impact": 0.0
        }
    
    def quantize_model(self, model: Any, level: QuantizationLevel, 
                      calibration_data: Optional[np.ndarray] = None) -> Any:
        """Quantize a model to specified precision level"""
        try:
            model_id = self._get_model_id(model)
            
            if model_id in self.quantization_cache:
                logger.info(f"Using cached quantized model {model_id}")
                return self.quantization_cache[model_id]
            
            logger.info(f"Quantizing model to {level.value}")
            
            if level == QuantizationLevel.INT4:
                quantized_model = self._quantize_to_int4(model, calibration_data)
            elif level == QuantizationLevel.INT8:
                quantized_model = self._quantize_to_int8(model, calibration_data)
            elif level == QuantizationLevel.FLOAT16:
                quantized_model = self._quantize_to_float16(model)
            else:
                quantized_model = model  # No quantization
            
            # Cache the quantized model
            self.quantization_cache[model_id] = quantized_model
            
            # Update stats
            original_size = self._calculate_model_size(model)
            quantized_size = self._calculate_model_size(quantized_model)
            memory_saved = (original_size - quantized_size) / (1024**3)  # GB
            
            self.quantization_stats["models_quantized"] += 1
            self.quantization_stats["memory_saved_gb"] += memory_saved
            
            logger.info(f"Model quantized: {memory_saved:.2f} GB saved")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def _get_model_id(self, model: Any) -> str:
        """Generate unique ID for model"""
        return f"model_{id(model)}"
    
    def _quantize_to_int4(self, model: Any, calibration_data: Optional[np.ndarray]) -> Any:
        """Quantize model to 4-bit integers"""
        try:
            # MLX-specific quantization
            if hasattr(model, 'parameters'):
                quantized_params = {}
                for name, param in model.parameters().items():
                    if isinstance(param, np.ndarray):
                        # Scale to 4-bit range [-8, 7]
                        param_max = np.max(np.abs(param))
                        scale = param_max / 7.0
                        quantized = np.clip(np.round(param / scale), -8, 7).astype(np.int8)
                        quantized_params[name] = {'weights': quantized, 'scale': scale}
                    else:
                        quantized_params[name] = param
                
                # Create quantized model wrapper
                return QuantizedModel(quantized_params, QuantizationLevel.INT4)
            else:
                return model
                
        except Exception as e:
            logger.error(f"INT4 quantization failed: {e}")
            return model
    
    def _quantize_to_int8(self, model: Any, calibration_data: Optional[np.ndarray]) -> Any:
        """Quantize model to 8-bit integers"""
        try:
            if hasattr(model, 'parameters'):
                quantized_params = {}
                for name, param in model.parameters().items():
                    if isinstance(param, np.ndarray):
                        # Scale to 8-bit range [-128, 127]
                        param_max = np.max(np.abs(param))
                        scale = param_max / 127.0
                        quantized = np.clip(np.round(param / scale), -128, 127).astype(np.int8)
                        quantized_params[name] = {'weights': quantized, 'scale': scale}
                    else:
                        quantized_params[name] = param
                
                return QuantizedModel(quantized_params, QuantizationLevel.INT8)
            else:
                return model
                
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            return model
    
    def _quantize_to_float16(self, model: Any) -> Any:
        """Quantize model to 16-bit floats"""
        try:
            if hasattr(model, 'parameters'):
                quantized_params = {}
                for name, param in model.parameters().items():
                    if isinstance(param, np.ndarray) and param.dtype == np.float32:
                        quantized_params[name] = param.astype(np.float16)
                    else:
                        quantized_params[name] = param
                
                return QuantizedModel(quantized_params, QuantizationLevel.FLOAT16)
            else:
                return model
                
        except Exception as e:
            logger.error(f"Float16 quantization failed: {e}")
            return model
    
    def _calculate_model_size(self, model: Any) -> int:
        """Calculate model size in bytes"""
        try:
            total_size = 0
            if hasattr(model, 'parameters'):
                for param in model.parameters().values():
                    if isinstance(param, np.ndarray):
                        total_size += param.nbytes
                    elif isinstance(param, dict) and 'weights' in param:
                        total_size += param['weights'].nbytes
            return total_size
        except Exception:
            return 0

class QuantizedModel:
    """Wrapper for quantized models"""
    
    def __init__(self, quantized_params: Dict[str, Any], level: QuantizationLevel):
        self.quantized_params = quantized_params
        self.quantization_level = level
        self._original_forward = None
    
    def parameters(self):
        """Return quantized parameters"""
        return self.quantized_params
    
    def dequantize_layer(self, layer_name: str) -> np.ndarray:
        """Dequantize a specific layer for computation"""
        if layer_name in self.quantized_params:
            param = self.quantized_params[layer_name]
            if isinstance(param, dict) and 'weights' in param:
                return param['weights'].astype(np.float32) * param['scale']
            else:
                return param
        return None

class ActivationCache:
    """Intelligent activation caching system"""
    
    def __init__(self, max_size_gb: float = 8.0):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.cache = {}
        self.access_times = {}
        self.cache_size = 0
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        # Memory-mapped file for large activations
        self.mmap_file = None
        self.mmap_offset = 0
        self._setup_mmap()
    
    def _setup_mmap(self):
        """Setup memory-mapped file for large activation storage"""
        try:
            cache_dir = Path("/tmp/mlx_cache")
            cache_dir.mkdir(exist_ok=True)
            
            cache_file = cache_dir / f"activations_{os.getpid()}.cache"
            
            # Create file if it doesn't exist
            if not cache_file.exists():
                with open(cache_file, 'wb') as f:
                    f.write(b'\x00' * self.max_size_bytes)
            
            # Memory map the file
            self.mmap_file = mmap.mmap(
                open(cache_file, 'r+b').fileno(),
                self.max_size_bytes,
                access=mmap.ACCESS_WRITE
            )
            
            logger.info(f"Setup activation cache with {self.max_size_bytes/1024**3:.1f}GB capacity")
            
        except Exception as e:
            logger.error(f"Failed to setup memory-mapped cache: {e}")
            self.mmap_file = None
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get activation from cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_count += 1
                
                cached_item = self.cache[key]
                if cached_item['type'] == 'memory':
                    return cached_item['data']
                elif cached_item['type'] == 'mmap' and self.mmap_file:
                    return self._load_from_mmap(cached_item)
                
            self.miss_count += 1
            return None
    
    def put(self, key: str, activation: np.ndarray):
        """Store activation in cache"""
        with self.lock:
            activation_size = activation.nbytes
            
            # Check if we need to evict items
            while self.cache_size + activation_size > self.max_size_bytes and self.cache:
                self._evict_lru()
            
            # Store in memory or mmap based on size
            if activation_size < 100 * 1024 * 1024:  # < 100MB in memory
                self.cache[key] = {
                    'type': 'memory',
                    'data': activation.copy(),
                    'size': activation_size
                }
            else:  # Large activations in mmap
                mmap_info = self._store_in_mmap(activation)
                if mmap_info:
                    self.cache[key] = {
                        'type': 'mmap',
                        'mmap_info': mmap_info,
                        'size': activation_size
                    }
            
            self.access_times[key] = time.time()
            self.cache_size += activation_size
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        cached_item = self.cache[lru_key]
        
        self.cache_size -= cached_item['size']
        del self.cache[lru_key]
        del self.access_times[lru_key]
        
        logger.debug(f"Evicted {lru_key} from cache")
    
    def _store_in_mmap(self, activation: np.ndarray) -> Optional[Dict[str, Any]]:
        """Store activation in memory-mapped file"""
        if not self.mmap_file:
            return None
        
        try:
            activation_bytes = activation.tobytes()
            size_needed = len(activation_bytes)
            
            if self.mmap_offset + size_needed > self.max_size_bytes:
                # Reset offset if we've run out of space
                self.mmap_offset = 0
            
            # Store metadata and data
            mmap_info = {
                'offset': self.mmap_offset,
                'size': size_needed,
                'shape': activation.shape,
                'dtype': str(activation.dtype)
            }
            
            self.mmap_file[self.mmap_offset:self.mmap_offset + size_needed] = activation_bytes
            self.mmap_offset += size_needed
            
            return mmap_info
            
        except Exception as e:
            logger.error(f"Failed to store in mmap: {e}")
            return None
    
    def _load_from_mmap(self, cached_item: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load activation from memory-mapped file"""
        if not self.mmap_file:
            return None
        
        try:
            mmap_info = cached_item['mmap_info']
            
            # Read data from mmap
            data_bytes = self.mmap_file[mmap_info['offset']:mmap_info['offset'] + mmap_info['size']]
            
            # Reconstruct array
            activation = np.frombuffer(data_bytes, dtype=mmap_info['dtype'])
            activation = activation.reshape(mmap_info['shape'])
            
            return activation
            
        except Exception as e:
            logger.error(f"Failed to load from mmap: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_ratio': hit_ratio,
            'cache_size_gb': self.cache_size / (1024**3),
            'items_cached': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count
        }

class MemoryPoolManager:
    """Advanced memory pool management"""
    
    def __init__(self, total_memory_gb: float = None):
        if total_memory_gb is None:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        self.total_memory_gb = total_memory_gb
        self.pools: Dict[str, MemoryPool] = {}
        self.allocation_tracker = {}
        self.lock = threading.RLock()
        
        # Initialize memory pools
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize memory pools for different use cases"""
        pool_configs = [
            ("model_weights", 0.4),      # 40% for model weights
            ("activations", 0.3),        # 30% for activations
            ("gradients", 0.2),          # 20% for gradients
            ("temp_buffers", 0.1)        # 10% for temporary buffers
        ]
        
        for pool_name, ratio in pool_configs:
            pool_size = int(self.total_memory_gb * ratio * 1024**3)  # Convert to bytes
            self.pools[pool_name] = MemoryPool(
                pool_id=pool_name,
                size_bytes=pool_size
            )
            logger.info(f"Created pool '{pool_name}': {pool_size/1024**3:.1f}GB")
    
    def allocate(self, pool_name: str, size_bytes: int, alignment: int = 32) -> Optional[Dict[str, Any]]:
        """Allocate memory from specified pool"""
        with self.lock:
            if pool_name not in self.pools:
                logger.error(f"Pool '{pool_name}' not found")
                return None
            
            pool = self.pools[pool_name]
            
            with pool.lock:
                # Check if we have enough space
                if pool.used_bytes + size_bytes > pool.size_bytes:
                    logger.warning(f"Pool '{pool_name}' out of memory")
                    return None
                
                # Find suitable free block or create new one
                block = self._find_or_create_block(pool, size_bytes, alignment)
                if not block:
                    return None
                
                # Update pool usage
                pool.used_bytes += size_bytes
                
                # Track allocation
                allocation_id = f"{pool_name}_{len(pool.blocks)}"
                self.allocation_tracker[allocation_id] = {
                    'pool': pool_name,
                    'size': size_bytes,
                    'timestamp': time.time(),
                    'block': block
                }
                
                return {
                    'allocation_id': allocation_id,
                    'memory_ptr': block.get('memory_ptr'),
                    'size': size_bytes
                }
    
    def _find_or_create_block(self, pool: MemoryPool, size_bytes: int, alignment: int) -> Optional[Dict[str, Any]]:
        """Find existing free block or create new one"""
        # Look for suitable free block
        for i, free_block in enumerate(pool.free_blocks):
            if free_block['size'] >= size_bytes:
                # Remove from free list
                block = pool.free_blocks.pop(i)
                
                # Split block if it's much larger
                if block['size'] > size_bytes * 2:
                    remaining_block = {
                        'memory_ptr': block['memory_ptr'] + size_bytes,
                        'size': block['size'] - size_bytes,
                        'aligned': True
                    }
                    pool.free_blocks.append(remaining_block)
                    block['size'] = size_bytes
                
                pool.blocks.append(block)
                return block
        
        # Create new block
        try:
            # Simulate memory allocation (in real implementation, this would allocate actual memory)
            memory_ptr = len(pool.blocks) * 1024  # Simulated pointer
            
            block = {
                'memory_ptr': memory_ptr,
                'size': size_bytes,
                'aligned': True,
                'allocated_at': time.time()
            }
            
            pool.blocks.append(block)
            return block
            
        except Exception as e:
            logger.error(f"Failed to create memory block: {e}")
            return None
    
    def deallocate(self, allocation_id: str):
        """Deallocate memory"""
        with self.lock:
            if allocation_id not in self.allocation_tracker:
                logger.warning(f"Allocation ID '{allocation_id}' not found")
                return
            
            allocation = self.allocation_tracker[allocation_id]
            pool = self.pools[allocation['pool']]
            
            with pool.lock:
                # Find and remove block
                block_to_remove = allocation['block']
                if block_to_remove in pool.blocks:
                    pool.blocks.remove(block_to_remove)
                    pool.free_blocks.append(block_to_remove)
                    pool.used_bytes -= allocation['size']
                
                # Remove from tracker
                del self.allocation_tracker[allocation_id]
    
    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools"""
        stats = {}
        
        for pool_name, pool in self.pools.items():
            with pool.lock:
                utilization = pool.used_bytes / pool.size_bytes if pool.size_bytes > 0 else 0
                fragmentation = len(pool.free_blocks) / max(len(pool.blocks), 1)
                
                stats[pool_name] = {
                    'size_gb': pool.size_bytes / (1024**3),
                    'used_gb': pool.used_bytes / (1024**3),
                    'utilization': utilization,
                    'fragmentation': fragmentation,
                    'active_blocks': len(pool.blocks),
                    'free_blocks': len(pool.free_blocks)
                }
        
        return stats

class GCOptimizer:
    """Garbage collection optimization"""
    
    def __init__(self):
        self.gc_stats = {
            'collections': 0,
            'time_spent': 0.0,
            'objects_collected': 0
        }
        self.optimization_enabled = True
        self.gc_threshold_multiplier = 2.0
        
        # Configure garbage collection
        self._configure_gc()
    
    def _configure_gc(self):
        """Configure garbage collection for optimal performance"""
        try:
            # Get current thresholds
            thresholds = gc.get_threshold()
            
            # Increase thresholds to reduce GC frequency
            new_thresholds = [
                int(thresholds[0] * self.gc_threshold_multiplier),
                int(thresholds[1] * self.gc_threshold_multiplier),
                int(thresholds[2] * self.gc_threshold_multiplier)
            ]
            
            gc.set_threshold(*new_thresholds)
            
            logger.info(f"GC thresholds set to: {new_thresholds}")
            
        except Exception as e:
            logger.error(f"Failed to configure GC: {e}")
    
    def force_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics"""
        if not self.optimization_enabled:
            return {}
        
        start_time = time.time()
        
        # Collect garbage for all generations
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        end_time = time.time()
        collection_time = end_time - start_time
        
        # Update stats
        self.gc_stats['collections'] += 1
        self.gc_stats['time_spent'] += collection_time
        self.gc_stats['objects_collected'] += sum(collected)
        
        return {
            'collection_time_ms': collection_time * 1000,
            'objects_collected': sum(collected),
            'generation_stats': collected
        }
    
    def get_memory_pressure(self) -> float:
        """Calculate current memory pressure (0.0 to 1.0)"""
        try:
            memory_info = psutil.virtual_memory()
            return memory_info.percent / 100.0
        except Exception:
            return 0.5  # Default moderate pressure
    
    def adaptive_gc_control(self):
        """Adaptive garbage collection based on memory pressure"""
        pressure = self.get_memory_pressure()
        
        if pressure > 0.8:  # High memory pressure
            # More aggressive GC
            self.gc_threshold_multiplier = 0.5
            self.force_collection()
        elif pressure < 0.3:  # Low memory pressure
            # Less frequent GC
            self.gc_threshold_multiplier = 3.0
        else:
            # Normal GC
            self.gc_threshold_multiplier = 2.0
        
        self._configure_gc()

class MemoryOptimizer:
    """Main memory optimization coordinator"""
    
    def __init__(self, total_memory_gb: float = None):
        self.total_memory_gb = total_memory_gb or psutil.virtual_memory().total / (1024**3)
        
        # Initialize components
        self.quantization_engine = QuantizationEngine()
        self.activation_cache = ActivationCache(max_size_gb=self.total_memory_gb * 0.2)
        self.memory_pool_manager = MemoryPoolManager(self.total_memory_gb * 0.7)
        self.gc_optimizer = GCOptimizer()
        
        # Monitoring
        self.optimization_stats = {
            'memory_saved_gb': 0.0,
            'cache_hit_ratio': 0.0,
            'pool_utilization': 0.0,
            'gc_efficiency': 0.0
        }
        
        # Background optimization
        self._optimization_thread = None
        self._stop_optimization = threading.Event()
        
        logger.info(f"Memory optimizer initialized with {self.total_memory_gb:.1f}GB total memory")
    
    def start_optimization(self):
        """Start background memory optimization"""
        if self._optimization_thread is None:
            self._optimization_thread = threading.Thread(target=self._optimization_loop)
            self._optimization_thread.daemon = True
            self._optimization_thread.start()
            logger.info("Background memory optimization started")
    
    def stop_optimization(self):
        """Stop background memory optimization"""
        if self._optimization_thread:
            self._stop_optimization.set()
            self._optimization_thread.join(timeout=5)
            self._optimization_thread = None
            logger.info("Background memory optimization stopped")
    
    def _optimization_loop(self):
        """Background optimization loop"""
        while not self._stop_optimization.wait(30):  # Check every 30 seconds
            try:
                # Adaptive GC control
                self.gc_optimizer.adaptive_gc_control()
                
                # Update optimization stats
                self._update_optimization_stats()
                
                # Log current status
                self._log_memory_status()
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
    
    def _update_optimization_stats(self):
        """Update optimization statistics"""
        try:
            # Cache stats
            cache_stats = self.activation_cache.get_stats()
            self.optimization_stats['cache_hit_ratio'] = cache_stats['hit_ratio']
            
            # Pool stats
            pool_stats = self.memory_pool_manager.get_pool_stats()
            total_utilization = sum(stats['utilization'] for stats in pool_stats.values())
            self.optimization_stats['pool_utilization'] = total_utilization / len(pool_stats)
            
            # Quantization stats
            self.optimization_stats['memory_saved_gb'] = self.quantization_engine.quantization_stats['memory_saved_gb']
            
        except Exception as e:
            logger.error(f"Failed to update optimization stats: {e}")
    
    def _log_memory_status(self):
        """Log current memory status"""
        try:
            memory_info = psutil.virtual_memory()
            
            logger.info(
                f"Memory: {memory_info.percent:.1f}% used, "
                f"Cache hit ratio: {self.optimization_stats['cache_hit_ratio']:.2f}, "
                f"Pool utilization: {self.optimization_stats['pool_utilization']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to log memory status: {e}")
    
    def optimize_model(self, model: Any, quantization_level: QuantizationLevel = QuantizationLevel.INT8) -> Any:
        """Optimize a model for memory efficiency"""
        try:
            # Quantize model
            optimized_model = self.quantization_engine.quantize_model(model, quantization_level)
            
            logger.info(f"Model optimized with {quantization_level.value} quantization")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def allocate_memory(self, pool_name: str, size_bytes: int) -> Optional[Dict[str, Any]]:
        """Allocate memory from specified pool"""
        return self.memory_pool_manager.allocate(pool_name, size_bytes)
    
    def deallocate_memory(self, allocation_id: str):
        """Deallocate memory"""
        self.memory_pool_manager.deallocate(allocation_id)
    
    def cache_activation(self, key: str, activation: np.ndarray):
        """Cache activation for reuse"""
        self.activation_cache.put(key, activation)
    
    def get_cached_activation(self, key: str) -> Optional[np.ndarray]:
        """Get cached activation"""
        return self.activation_cache.get(key)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization statistics"""
        try:
            memory_info = psutil.virtual_memory()
            
            # Calculate fragmentation
            pool_stats = self.memory_pool_manager.get_pool_stats()
            avg_fragmentation = sum(stats['fragmentation'] for stats in pool_stats.values()) / len(pool_stats)
            
            return {
                'system_memory': {
                    'total_gb': memory_info.total / (1024**3),
                    'used_gb': memory_info.used / (1024**3),
                    'available_gb': memory_info.available / (1024**3),
                    'percent_used': memory_info.percent
                },
                'optimization_stats': self.optimization_stats,
                'quantization_stats': self.quantization_engine.quantization_stats,
                'cache_stats': self.activation_cache.get_stats(),
                'pool_stats': pool_stats,
                'gc_stats': self.gc_optimizer.gc_stats,
                'memory_metrics': MemoryStats(
                    total_memory_gb=memory_info.total / (1024**3),
                    used_memory_gb=memory_info.used / (1024**3),
                    available_memory_gb=memory_info.available / (1024**3),
                    fragmentation_ratio=avg_fragmentation,
                    pool_utilization=self.optimization_stats['pool_utilization'],
                    gc_pressure=self.gc_optimizer.get_memory_pressure(),
                    cache_hit_ratio=self.optimization_stats['cache_hit_ratio']
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
            return {}

# Example usage
async def main():
    """Example usage of MemoryOptimizer"""
    optimizer = MemoryOptimizer()
    
    # Start background optimization
    optimizer.start_optimization()
    
    # Simulate model optimization
    dummy_model = {"weights": np.random.randn(1000, 1000).astype(np.float32)}
    optimized_model = optimizer.optimize_model(dummy_model, QuantizationLevel.INT8)
    
    # Simulate memory allocation
    allocation = optimizer.allocate_memory("model_weights", 1024 * 1024)  # 1MB
    if allocation:
        logger.info(f"Allocated memory: {allocation['allocation_id']}")
        
        # Later deallocate
        optimizer.deallocate_memory(allocation['allocation_id'])
    
    # Cache some activations
    test_activation = np.random.randn(512, 512).astype(np.float32)
    optimizer.cache_activation("test_layer_output", test_activation)
    
    # Retrieve cached activation
    cached = optimizer.get_cached_activation("test_layer_output")
    if cached is not None:
        logger.info("Successfully retrieved cached activation")
    
    # Get comprehensive stats
    stats = optimizer.get_comprehensive_stats()
    logger.info(f"Memory optimization stats: {json.dumps(stats, indent=2, default=str)}")
    
    # Stop optimization
    optimizer.stop_optimization()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import asyncio
    asyncio.run(main())