"""
Distributed Memory Manager with Tiered Caching
Manages memory allocation and caching across the Apple Silicon cluster
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import pickle
import hashlib

logger = logging.getLogger(__name__)

class CacheTier(Enum):
    """Memory cache tiers in order of access speed"""
    L1_GPU_MEMORY = "l1_gpu_memory"          # Fastest - GPU unified memory
    L2_SYSTEM_MEMORY = "l2_system_memory"    # Fast - System RAM
    L3_NVME_CACHE = "l3_nvme_cache"         # Medium - Local NVMe storage
    L4_NETWORK_STORAGE = "l4_network_storage" # Slowest - Network shared storage

class MemoryType(Enum):
    """Types of memory objects to manage"""
    MODEL_WEIGHTS = "model_weights"
    ACTIVATIONS = "activations"
    GRADIENTS = "gradients"
    ATTENTION_CACHE = "attention_cache"
    EMBEDDINGS = "embeddings"
    TEMPORARY = "temporary"

@dataclass
class MemoryObject:
    """Represents a memory object in the cache hierarchy"""
    object_id: str
    object_type: MemoryType
    size_bytes: int
    data: Optional[Any] = None
    tier: Optional[CacheTier] = None
    node_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_pinned: bool = False
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'object_id': self.object_id,
            'object_type': self.object_type.value,
            'size_bytes': self.size_bytes,
            'tier': self.tier.value if self.tier else None,
            'node_id': self.node_id,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'is_pinned': self.is_pinned,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }

@dataclass
class MemoryStats:
    """Memory statistics for a tier or node"""
    total_capacity_bytes: int
    used_bytes: int
    available_bytes: int
    cached_objects: int
    hit_rate: float
    miss_rate: float
    eviction_count: int
    allocation_count: int
    
    @property
    def utilization_percent(self) -> float:
        return (self.used_bytes / self.total_capacity_bytes) * 100 if self.total_capacity_bytes > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_capacity_bytes': self.total_capacity_bytes,
            'used_bytes': self.used_bytes,
            'available_bytes': self.available_bytes,
            'cached_objects': self.cached_objects,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'eviction_count': self.eviction_count,
            'allocation_count': self.allocation_count,
            'utilization_percent': self.utilization_percent
        }

class TierCache:
    """Individual cache tier implementation"""
    
    def __init__(self, tier: CacheTier, capacity_bytes: int, base_path: Optional[str] = None):
        self.tier = tier
        self.capacity_bytes = capacity_bytes
        self.base_path = base_path or f"/tmp/mlx_cache_{tier.value}"
        
        # Cache storage
        self.objects: Dict[str, MemoryObject] = {}
        self.access_order: List[str] = []  # For LRU eviction
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.allocations = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize storage
        if tier in [CacheTier.L3_NVME_CACHE, CacheTier.L4_NETWORK_STORAGE]:
            os.makedirs(self.base_path, exist_ok=True)
        
        logger.debug(f"Initialized {tier.value} cache with {capacity_bytes / (1024**3):.2f}GB capacity")
    
    def get(self, object_id: str) -> Optional[MemoryObject]:
        """Get object from cache"""
        with self.lock:
            if object_id in self.objects:
                obj = self.objects[object_id]
                obj.last_accessed = time.time()
                obj.access_count += 1
                
                # Update LRU order
                if object_id in self.access_order:
                    self.access_order.remove(object_id)
                self.access_order.append(object_id)
                
                self.hits += 1
                
                # Load data if needed
                if obj.data is None and self.tier in [CacheTier.L3_NVME_CACHE, CacheTier.L4_NETWORK_STORAGE]:
                    obj.data = self._load_from_disk(object_id)
                
                return obj
            else:
                self.misses += 1
                return None
    
    def put(self, obj: MemoryObject) -> bool:
        """Put object into cache"""
        with self.lock:
            # Check if we have space
            current_usage = sum(o.size_bytes for o in self.objects.values())
            
            if current_usage + obj.size_bytes > self.capacity_bytes:
                # Need to evict objects
                if not self._evict_to_make_space(obj.size_bytes):
                    return False  # Could not make enough space
            
            # Store object
            obj.tier = self.tier
            self.objects[obj.object_id] = obj
            
            # Update access order
            if obj.object_id in self.access_order:
                self.access_order.remove(obj.object_id)
            self.access_order.append(obj.object_id)
            
            # Save to disk if needed
            if self.tier in [CacheTier.L3_NVME_CACHE, CacheTier.L4_NETWORK_STORAGE]:
                self._save_to_disk(obj)
            
            self.allocations += 1
            return True
    
    def remove(self, object_id: str) -> bool:
        """Remove object from cache"""
        with self.lock:
            if object_id in self.objects:
                obj = self.objects.pop(object_id)
                
                # Remove from access order
                if object_id in self.access_order:
                    self.access_order.remove(object_id)
                
                # Remove from disk if needed
                if self.tier in [CacheTier.L3_NVME_CACHE, CacheTier.L4_NETWORK_STORAGE]:
                    self._remove_from_disk(object_id)
                
                return True
            return False
    
    def _evict_to_make_space(self, required_bytes: int) -> bool:
        """Evict objects to make space using LRU policy"""
        freed_bytes = 0
        evicted_objects = []
        
        # Find objects to evict (LRU, but preserve pinned objects)
        for object_id in self.access_order[:]:
            obj = self.objects[object_id]
            
            if obj.is_pinned:
                continue  # Skip pinned objects
            
            evicted_objects.append(object_id)
            freed_bytes += obj.size_bytes
            
            if freed_bytes >= required_bytes:
                break
        
        # Perform evictions
        for object_id in evicted_objects:
            self.remove(object_id)
            self.evictions += 1
        
        return freed_bytes >= required_bytes
    
    def _save_to_disk(self, obj: MemoryObject) -> None:
        """Save object data to disk"""
        try:
            file_path = os.path.join(self.base_path, f"{obj.object_id}.cache")
            
            # Save metadata and data separately
            metadata = obj.to_dict()
            metadata['data_file'] = f"{obj.object_id}.data"
            
            with open(file_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            if obj.data is not None:
                data_path = os.path.join(self.base_path, f"{obj.object_id}.data")
                with open(data_path, 'wb') as f:
                    pickle.dump(obj.data, f)
                    
                # Clear data from memory for disk-based tiers
                obj.data = None
                
        except Exception as e:
            logger.error(f"Failed to save object {obj.object_id} to disk: {e}")
    
    def _load_from_disk(self, object_id: str) -> Optional[Any]:
        """Load object data from disk"""
        try:
            data_path = os.path.join(self.base_path, f"{object_id}.data")
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load object {object_id} from disk: {e}")
        return None
    
    def _remove_from_disk(self, object_id: str) -> None:
        """Remove object files from disk"""
        try:
            cache_path = os.path.join(self.base_path, f"{object_id}.cache")
            data_path = os.path.join(self.base_path, f"{object_id}.data")
            
            for path in [cache_path, data_path]:
                if os.path.exists(path):
                    os.remove(path)
                    
        except Exception as e:
            logger.error(f"Failed to remove object {object_id} from disk: {e}")
    
    def get_stats(self) -> MemoryStats:
        """Get cache statistics"""
        with self.lock:
            used_bytes = sum(obj.size_bytes for obj in self.objects.values())
            
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
            miss_rate = self.misses / total_accesses if total_accesses > 0 else 0
            
            return MemoryStats(
                total_capacity_bytes=self.capacity_bytes,
                used_bytes=used_bytes,
                available_bytes=self.capacity_bytes - used_bytes,
                cached_objects=len(self.objects),
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                eviction_count=self.evictions,
                allocation_count=self.allocations
            )
    
    def clear(self) -> None:
        """Clear all objects from cache"""
        with self.lock:
            object_ids = list(self.objects.keys())
            for object_id in object_ids:
                self.remove(object_id)

class DistributedMemoryManager:
    """
    Distributed memory manager with tiered caching
    Manages memory allocation across the Apple Silicon cluster
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.config = config
        
        # Initialize cache tiers
        self.tiers: Dict[CacheTier, TierCache] = {}
        self._initialize_tiers()
        
        # Cluster coordination
        self.peer_managers: Dict[str, 'DistributedMemoryManager'] = {}
        self.global_directory: Dict[str, Dict[str, Any]] = {}  # object_id -> location info
        
        # Configuration
        self.auto_migrate = config.get('auto_migrate', True)
        self.prefetch_enabled = config.get('prefetch_enabled', True)
        self.replication_factor = config.get('replication_factor', 1)
        
        # Statistics
        self.global_stats = {
            'total_objects': 0,
            'total_memory_used': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'migrations': 0,
            'replications': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MemoryManager")
        
        logger.info(f"Distributed memory manager initialized for node {node_id}")
    
    def _initialize_tiers(self) -> None:
        """Initialize cache tiers based on configuration"""
        tier_configs = self.config.get('tiers', {})
        
        # Default configurations
        default_tiers = {
            CacheTier.L1_GPU_MEMORY: {
                'capacity_gb': 16,  # Reserve 16GB for active inference
                'path': None
            },
            CacheTier.L2_SYSTEM_MEMORY: {
                'capacity_gb': 32,  # Use 32GB of system memory
                'path': None
            },
            CacheTier.L3_NVME_CACHE: {
                'capacity_gb': 100, # Use 100GB of NVMe storage
                'path': f"/tmp/mlx_cache_l3_{self.node_id}"
            },
            CacheTier.L4_NETWORK_STORAGE: {
                'capacity_gb': 500, # Network storage limit
                'path': f"/tmp/mlx_cache_l4_{self.node_id}"
            }
        }
        
        for tier, default_config in default_tiers.items():
            config = tier_configs.get(tier.value, default_config)
            capacity_bytes = int(config['capacity_gb'] * 1024**3)
            base_path = config.get('path')
            
            self.tiers[tier] = TierCache(tier, capacity_bytes, base_path)
    
    def store_object(self, obj: MemoryObject, preferred_tier: Optional[CacheTier] = None) -> bool:
        """Store object in the most appropriate tier"""
        with self.lock:
            # Determine best tier
            target_tier = preferred_tier or self._select_optimal_tier(obj)
            
            # Try to store in target tier
            if self.tiers[target_tier].put(obj):
                # Update global directory
                self.global_directory[obj.object_id] = {
                    'node_id': self.node_id,
                    'tier': target_tier.value,
                    'size_bytes': obj.size_bytes,
                    'object_type': obj.object_type.value,
                    'created_at': obj.created_at,
                    'is_pinned': obj.is_pinned
                }
                
                self.global_stats['total_objects'] += 1
                self.global_stats['total_memory_used'] += obj.size_bytes
                
                logger.debug(f"Stored object {obj.object_id} in {target_tier.value}")
                return True
            
            # Try fallback tiers
            for tier in CacheTier:
                if tier != target_tier and self.tiers[tier].put(obj):
                    self.global_directory[obj.object_id] = {
                        'node_id': self.node_id,
                        'tier': tier.value,
                        'size_bytes': obj.size_bytes,
                        'object_type': obj.object_type.value,
                        'created_at': obj.created_at,
                        'is_pinned': obj.is_pinned
                    }
                    
                    self.global_stats['total_objects'] += 1
                    self.global_stats['total_memory_used'] += obj.size_bytes
                    
                    logger.debug(f"Stored object {obj.object_id} in fallback tier {tier.value}")
                    return True
            
            logger.warning(f"Failed to store object {obj.object_id} in any tier")
            return False
    
    def get_object(self, object_id: str) -> Optional[MemoryObject]:
        """Get object from cache hierarchy"""
        with self.lock:
            # Check local tiers first (L1 -> L4)
            for tier in CacheTier:
                obj = self.tiers[tier].get(object_id)
                if obj:
                    self.global_stats['cache_hits'] += 1
                    
                    # Promote to faster tier if beneficial
                    if self.auto_migrate and tier != CacheTier.L1_GPU_MEMORY:
                        asyncio.create_task(self._promote_object(obj, tier))
                    
                    return obj
            
            # Check if object exists on peer nodes
            if object_id in self.global_directory:
                obj_info = self.global_directory[object_id]
                peer_node = obj_info['node_id']
                
                if peer_node != self.node_id and peer_node in self.peer_managers:
                    # Fetch from peer
                    obj = self._fetch_from_peer(object_id, peer_node)
                    if obj:
                        # Store locally for future access
                        self.store_object(obj)
                        self.global_stats['cache_hits'] += 1
                        return obj
            
            self.global_stats['cache_misses'] += 1
            return None
    
    def remove_object(self, object_id: str) -> bool:
        """Remove object from all tiers"""
        with self.lock:
            removed = False
            
            # Remove from all local tiers
            for tier in CacheTier:
                if self.tiers[tier].remove(object_id):
                    removed = True
            
            # Remove from global directory
            if object_id in self.global_directory:
                obj_info = self.global_directory.pop(object_id)
                self.global_stats['total_objects'] = max(0, self.global_stats['total_objects'] - 1)
                self.global_stats['total_memory_used'] = max(0, self.global_stats['total_memory_used'] - obj_info['size_bytes'])
                removed = True
            
            return removed
    
    def _select_optimal_tier(self, obj: MemoryObject) -> CacheTier:
        """Select the optimal tier for an object"""
        # High-priority objects go to fastest tiers
        if obj.object_type == MemoryType.ACTIVATIONS or obj.is_pinned:
            return CacheTier.L1_GPU_MEMORY
        elif obj.object_type == MemoryType.MODEL_WEIGHTS:
            return CacheTier.L2_SYSTEM_MEMORY
        elif obj.object_type == MemoryType.ATTENTION_CACHE:
            return CacheTier.L2_SYSTEM_MEMORY
        elif obj.object_type == MemoryType.EMBEDDINGS:
            return CacheTier.L3_NVME_CACHE
        else:
            return CacheTier.L3_NVME_CACHE
    
    async def _promote_object(self, obj: MemoryObject, current_tier: CacheTier) -> bool:
        """Promote object to a faster tier"""
        try:
            # Determine target tier (one level faster)
            tier_order = list(CacheTier)
            current_index = tier_order.index(current_tier)
            
            if current_index > 0:
                target_tier = tier_order[current_index - 1]
                
                # Check if target tier has space or can make space
                target_cache = self.tiers[target_tier]
                current_usage = sum(o.size_bytes for o in target_cache.objects.values())
                
                if current_usage + obj.size_bytes <= target_cache.capacity_bytes * 0.9:  # Keep 10% buffer
                    # Create copy for target tier
                    promoted_obj = MemoryObject(
                        object_id=obj.object_id,
                        object_type=obj.object_type,
                        size_bytes=obj.size_bytes,
                        data=obj.data,
                        tier=target_tier,
                        node_id=self.node_id,
                        created_at=obj.created_at,
                        last_accessed=time.time(),
                        access_count=obj.access_count,
                        is_pinned=obj.is_pinned,
                        dependencies=obj.dependencies.copy(),
                        metadata=obj.metadata.copy()
                    )
                    
                    if target_cache.put(promoted_obj):
                        # Remove from current tier
                        self.tiers[current_tier].remove(obj.object_id)
                        
                        # Update global directory
                        self.global_directory[obj.object_id]['tier'] = target_tier.value
                        
                        logger.debug(f"Promoted object {obj.object_id} from {current_tier.value} to {target_tier.value}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to promote object {obj.object_id}: {e}")
            return False
    
    def _fetch_from_peer(self, object_id: str, peer_node: str) -> Optional[MemoryObject]:
        """Fetch object from peer node"""
        # In a real implementation, this would use network communication
        # For now, we'll simulate it
        try:
            if peer_node in self.peer_managers:
                peer_manager = self.peer_managers[peer_node]
                return peer_manager.get_object(object_id)
            
            logger.debug(f"Peer node {peer_node} not available for fetching {object_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch object {object_id} from peer {peer_node}: {e}")
            return None
    
    def register_peer(self, peer_node: str, peer_manager: 'DistributedMemoryManager') -> None:
        """Register a peer memory manager"""
        with self.lock:
            self.peer_managers[peer_node] = peer_manager
            logger.debug(f"Registered peer memory manager: {peer_node}")
    
    def unregister_peer(self, peer_node: str) -> None:
        """Unregister a peer memory manager"""
        with self.lock:
            if peer_node in self.peer_managers:
                del self.peer_managers[peer_node]
                logger.debug(f"Unregistered peer memory manager: {peer_node}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        with self.lock:
            tier_stats = {}
            for tier, cache in self.tiers.items():
                tier_stats[tier.value] = cache.get_stats().to_dict()
            
            return {
                'node_id': self.node_id,
                'tier_statistics': tier_stats,
                'global_statistics': self.global_stats.copy(),
                'global_directory_size': len(self.global_directory),
                'peer_managers': list(self.peer_managers.keys())
            }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across tiers"""
        optimization_results = {
            'objects_promoted': 0,
            'objects_demoted': 0,
            'objects_evicted': 0,
            'memory_freed_bytes': 0
        }
        
        try:
            with self.lock:
                # Find candidates for promotion/demotion based on access patterns
                all_objects = []
                for tier, cache in self.tiers.items():
                    for obj in cache.objects.values():
                        all_objects.append((obj, tier))
                
                # Sort by access frequency and recency
                all_objects.sort(key=lambda x: (x[0].access_count, x[0].last_accessed), reverse=True)
                
                # Promote frequently accessed objects
                for obj, current_tier in all_objects[:10]:  # Top 10 most accessed
                    if current_tier != CacheTier.L1_GPU_MEMORY:
                        if asyncio.create_task(self._promote_object(obj, current_tier)):
                            optimization_results['objects_promoted'] += 1
                
                # Demote rarely accessed objects from fast tiers
                l1_cache = self.tiers[CacheTier.L1_GPU_MEMORY]
                if len(l1_cache.objects) > 10:  # Keep some objects in L1
                    # Find least recently used objects in L1
                    l1_objects = [(obj, CacheTier.L1_GPU_MEMORY) for obj in l1_cache.objects.values()]
                    l1_objects.sort(key=lambda x: x[0].last_accessed)
                    
                    for obj, _ in l1_objects[10:]:  # Demote excess objects
                        if not obj.is_pinned:
                            # Move to L2
                            l2_cache = self.tiers[CacheTier.L2_SYSTEM_MEMORY]
                            if l2_cache.put(obj):
                                l1_cache.remove(obj.object_id)
                                optimization_results['objects_demoted'] += 1
                
                logger.info(f"Memory optimization completed: {optimization_results}")
                
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
        
        return optimization_results
    
    def create_memory_object(self, object_id: str, object_type: MemoryType, 
                           data: Any, metadata: Optional[Dict[str, Any]] = None) -> MemoryObject:
        """Create a new memory object"""
        # Estimate size (simplified)
        if hasattr(data, 'nbytes'):
            size_bytes = data.nbytes
        elif isinstance(data, (bytes, bytearray)):
            size_bytes = len(data)
        elif isinstance(data, str):
            size_bytes = len(data.encode('utf-8'))
        else:
            # Rough estimate using pickle
            size_bytes = len(pickle.dumps(data))
        
        return MemoryObject(
            object_id=object_id,
            object_type=object_type,
            size_bytes=size_bytes,
            data=data,
            node_id=self.node_id,
            metadata=metadata or {}
        )
    
    def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up distributed memory manager...")
        
        with self.lock:
            # Clear all caches
            for tier, cache in self.tiers.items():
                cache.clear()
            
            # Clear global directory
            self.global_directory.clear()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
        
        logger.info("Memory manager cleanup complete")

# Factory function
def create_memory_manager(node_id: str, config: Optional[Dict[str, Any]] = None) -> DistributedMemoryManager:
    """Create a distributed memory manager"""
    default_config = {
        'auto_migrate': True,
        'prefetch_enabled': True,
        'replication_factor': 1,
        'tiers': {}
    }
    
    final_config = {**default_config, **(config or {})}
    return DistributedMemoryManager(node_id, final_config)

# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Test memory manager
    manager = create_memory_manager("test-node-1")
    
    # Create test objects
    test_data = np.random.rand(1000, 1000).astype(np.float32)
    obj1 = manager.create_memory_object("weights_layer_1", MemoryType.MODEL_WEIGHTS, test_data)
    
    activation_data = np.random.rand(64, 4096).astype(np.float32)
    obj2 = manager.create_memory_object("activations_batch_1", MemoryType.ACTIVATIONS, activation_data)
    
    # Store objects
    print(f"Storing object 1: {manager.store_object(obj1)}")
    print(f"Storing object 2: {manager.store_object(obj2)}")
    
    # Retrieve objects
    retrieved_obj1 = manager.get_object("weights_layer_1")
    print(f"Retrieved object 1: {retrieved_obj1 is not None}")
    
    # Show statistics
    stats = manager.get_memory_stats()
    print(f"Memory statistics: {json.dumps(stats, indent=2)}")
    
    # Optimize memory
    optimization_results = manager.optimize_memory_usage()
    print(f"Optimization results: {optimization_results}")
    
    # Cleanup
    manager.cleanup()