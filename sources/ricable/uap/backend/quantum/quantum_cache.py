"""
Quantum-Specific Caching Strategies
Optimized caching for quantum circuits, results, and algorithm parameters.
"""

import asyncio
import hashlib
import json
import logging
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import redis
from concurrent.futures import ThreadPoolExecutor

# Import quantum components
from .quantum_simulator import QuantumGate, GateType, CircuitResult
from .quantum_ml import QuantumMLResult
from .quantum_advantage import AdvantageAnalysis

# Import caching infrastructure
from ..cache.redis_cache import redis_client
from ..cache.decorators import cache_with_ttl

logger = logging.getLogger(__name__)

class CacheType(Enum):
    """Types of quantum cache entries"""
    CIRCUIT_RESULT = "circuit_result"
    OPTIMIZED_CIRCUIT = "optimized_circuit"
    QUANTUM_STATE = "quantum_state"
    ML_MODEL = "ml_model"
    ADVANTAGE_ANALYSIS = "advantage_analysis"
    ALGORITHM_PARAMETERS = "algorithm_parameters"
    NOISE_MODEL = "noise_model"
    ERROR_CORRECTION = "error_correction"

@dataclass
class CacheEntry:
    """Quantum cache entry"""
    cache_key: str
    cache_type: CacheType
    data: Any
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: int

@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    total_entries: int
    hit_rate: float
    miss_rate: float
    total_size_mb: float
    average_access_time: float
    cache_efficiency: float
    eviction_count: int
    type_distribution: Dict[str, int]

class QuantumCircuitCache:
    """Specialized cache for quantum circuits and results"""
    
    def __init__(self, max_size_mb: int = 500, default_ttl: int = 3600):
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.cache_entries: Dict[str, CacheEntry] = {}
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.total_access_time = 0.0
        self.access_count = 0
        self.eviction_count = 0
        
        # Circuit-specific optimization
        self.circuit_similarity_threshold = 0.8
        self.gate_equivalence_cache: Dict[str, str] = {}
        
        logger.info(f"Quantum circuit cache initialized with {max_size_mb}MB limit")
    
    def _generate_circuit_key(self, gates: List[QuantumGate], num_qubits: int) -> str:
        """Generate unique key for quantum circuit"""
        # Create canonical representation of circuit
        circuit_repr = {
            'num_qubits': num_qubits,
            'gates': [
                {
                    'type': gate.gate_type.value,
                    'qubits': sorted(gate.qubits),  # Sort for canonicalization
                    'parameters': [round(p, 6) for p in gate.parameters] if gate.parameters else None
                }
                for gate in gates
            ]
        }
        
        # Create hash from canonical representation
        circuit_json = json.dumps(circuit_repr, sort_keys=True)
        return hashlib.sha256(circuit_json.encode()).hexdigest()[:16]
    
    def _generate_parametric_key(self, base_key: str, parameters: Dict[str, Any]) -> str:
        """Generate key for parametric circuits/algorithms"""
        param_repr = json.dumps(parameters, sort_keys=True)
        param_hash = hashlib.md5(param_repr.encode()).hexdigest()[:8]
        return f"{base_key}_{param_hash}"
    
    async def get_circuit_result(self, gates: List[QuantumGate], 
                               num_qubits: int, shots: int) -> Optional[CircuitResult]:
        """Get cached circuit simulation result"""
        start_time = datetime.utcnow()
        
        circuit_key = self._generate_circuit_key(gates, num_qubits)
        cache_key = f"circuit_{circuit_key}_{shots}"
        
        try:
            entry = self.cache_entries.get(cache_key)
            
            if entry:
                # Update access statistics
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                self.hit_count += 1
                
                # Check TTL
                age = (datetime.utcnow() - entry.created_at).total_seconds()
                if age < entry.ttl_seconds:
                    access_time = (datetime.utcnow() - start_time).total_seconds()
                    self._update_access_stats(access_time)
                    
                    logger.debug(f"Cache hit for circuit {circuit_key}")
                    return entry.data
                else:
                    # Expired entry
                    del self.cache_entries[cache_key]
            
            # Check for similar circuits
            similar_result = await self._find_similar_circuit(gates, num_qubits, shots)
            if similar_result:
                self.hit_count += 1
                access_time = (datetime.utcnow() - start_time).total_seconds()
                self._update_access_stats(access_time)
                return similar_result
            
            # Cache miss
            self.miss_count += 1
            access_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_access_stats(access_time)
            
            return None
            
        except Exception as e:
            logger.error(f"Error accessing circuit cache: {e}")
            self.miss_count += 1
            return None
    
    async def store_circuit_result(self, gates: List[QuantumGate], 
                                 num_qubits: int, shots: int, 
                                 result: CircuitResult, ttl: Optional[int] = None) -> bool:
        """Store circuit simulation result in cache"""
        try:
            circuit_key = self._generate_circuit_key(gates, num_qubits)
            cache_key = f"circuit_{circuit_key}_{shots}"
            
            # Serialize result
            serialized_data = self._serialize_circuit_result(result)
            data_size = len(serialized_data)
            
            # Check cache size limits
            if not await self._ensure_cache_space(data_size):
                logger.warning("Could not make space for new cache entry")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                cache_key=cache_key,
                cache_type=CacheType.CIRCUIT_RESULT,
                data=result,
                metadata={
                    'circuit_key': circuit_key,
                    'num_qubits': num_qubits,
                    'shots': shots,
                    'gate_count': len(gates),
                    'circuit_depth': result.depth
                },
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=data_size,
                ttl_seconds=ttl or self.default_ttl
            )
            
            self.cache_entries[cache_key] = entry
            logger.debug(f"Stored circuit result {circuit_key} in cache")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing circuit result in cache: {e}")
            return False
    
    async def get_optimized_circuit(self, gates: List[QuantumGate]) -> Optional[List[QuantumGate]]:
        """Get cached optimized circuit"""
        circuit_key = self._generate_circuit_key(gates, 0)  # num_qubits not relevant for optimization
        cache_key = f"optimized_{circuit_key}"
        
        entry = self.cache_entries.get(cache_key)
        if entry and self._is_entry_valid(entry):
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            self.hit_count += 1
            return entry.data
        
        self.miss_count += 1
        return None
    
    async def store_optimized_circuit(self, original_gates: List[QuantumGate], 
                                    optimized_gates: List[QuantumGate]) -> bool:
        """Store optimized circuit in cache"""
        try:
            circuit_key = self._generate_circuit_key(original_gates, 0)
            cache_key = f"optimized_{circuit_key}"
            
            # Calculate optimization metrics
            optimization_ratio = len(optimized_gates) / len(original_gates)
            
            entry = CacheEntry(
                cache_key=cache_key,
                cache_type=CacheType.OPTIMIZED_CIRCUIT,
                data=optimized_gates,
                metadata={
                    'original_gates': len(original_gates),
                    'optimized_gates': len(optimized_gates),
                    'optimization_ratio': optimization_ratio
                },
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=len(pickle.dumps(optimized_gates)),
                ttl_seconds=self.default_ttl * 2  # Longer TTL for optimizations
            )
            
            self.cache_entries[cache_key] = entry
            return True
            
        except Exception as e:
            logger.error(f"Error storing optimized circuit: {e}")
            return False
    
    async def _find_similar_circuit(self, gates: List[QuantumGate], 
                                  num_qubits: int, shots: int) -> Optional[CircuitResult]:
        """Find cached result for similar circuit"""
        try:
            # Look for circuits with similar structure
            for cache_key, entry in self.cache_entries.items():
                if (entry.cache_type == CacheType.CIRCUIT_RESULT and
                    entry.metadata.get('num_qubits') == num_qubits and
                    abs(entry.metadata.get('shots', 0) - shots) <= shots * 0.1):  # 10% tolerance
                    
                    # Calculate circuit similarity
                    similarity = self._calculate_circuit_similarity(gates, entry.metadata)
                    
                    if similarity >= self.circuit_similarity_threshold:
                        logger.debug(f"Found similar circuit with {similarity:.2f} similarity")
                        entry.last_accessed = datetime.utcnow()
                        entry.access_count += 1
                        return entry.data
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar circuit: {e}")
            return None
    
    def _calculate_circuit_similarity(self, gates: List[QuantumGate], 
                                    cached_metadata: Dict[str, Any]) -> float:
        """Calculate similarity between circuits"""
        try:
            # Simple similarity based on gate count and depth
            gate_count = len(gates)
            cached_gate_count = cached_metadata.get('gate_count', 0)
            
            if gate_count == 0 or cached_gate_count == 0:
                return 0.0
            
            gate_similarity = 1.0 - abs(gate_count - cached_gate_count) / max(gate_count, cached_gate_count)
            
            # In a more sophisticated implementation, we would compare
            # gate types, qubit connectivity, and circuit structure
            
            return gate_similarity
            
        except Exception:
            return 0.0
    
    def _serialize_circuit_result(self, result: CircuitResult) -> bytes:
        """Serialize circuit result for size calculation"""
        # Convert to dictionary and serialize
        result_dict = {
            'final_state': result.final_state.tolist() if hasattr(result.final_state, 'tolist') else result.final_state,
            'measurements': [asdict(m) for m in result.measurements],
            'fidelity': result.fidelity,
            'execution_time': result.execution_time,
            'gate_count': result.gate_count,
            'depth': result.depth,
            'metadata': result.metadata
        }
        
        return pickle.dumps(result_dict)
    
    async def _ensure_cache_space(self, required_bytes: int) -> bool:
        """Ensure there's enough cache space by evicting entries if necessary"""
        current_size = self._calculate_total_size()
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if current_size + required_bytes <= max_size_bytes:
            return True
        
        # Need to evict entries
        space_needed = (current_size + required_bytes) - max_size_bytes
        return await self._evict_entries(space_needed)
    
    async def _evict_entries(self, space_needed: int) -> bool:
        """Evict cache entries to free up space"""
        try:
            # Sort entries by eviction score (LRU + size + access frequency)
            entries_with_scores = []
            
            for cache_key, entry in self.cache_entries.items():
                score = self._calculate_eviction_score(entry)
                entries_with_scores.append((score, cache_key, entry))
            
            # Sort by eviction score (lower scores evicted first)
            entries_with_scores.sort(key=lambda x: x[0])
            
            freed_space = 0
            evicted_count = 0
            
            for score, cache_key, entry in entries_with_scores:
                if freed_space >= space_needed:
                    break
                
                freed_space += entry.size_bytes
                del self.cache_entries[cache_key]
                evicted_count += 1
                self.eviction_count += 1
            
            logger.info(f"Evicted {evicted_count} entries, freed {freed_space} bytes")
            return freed_space >= space_needed
            
        except Exception as e:
            logger.error(f"Error during cache eviction: {e}")
            return False
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score for cache entry (lower = more likely to evict)"""
        now = datetime.utcnow()
        
        # Time since last access (higher = more likely to evict)
        time_score = (now - entry.last_accessed).total_seconds() / 3600.0  # Hours
        
        # Access frequency (lower = more likely to evict)
        age_hours = max(1, (now - entry.created_at).total_seconds() / 3600.0)
        frequency_score = entry.access_count / age_hours
        
        # Size penalty (larger entries more likely to evict)
        size_score = entry.size_bytes / (1024 * 1024)  # MB
        
        # TTL remaining (expired entries evicted first)
        ttl_remaining = entry.ttl_seconds - (now - entry.created_at).total_seconds()
        ttl_score = max(0, ttl_remaining / 3600.0)  # Hours
        
        # Combined score (lower = more likely to evict)
        eviction_score = time_score + size_score - (frequency_score * 2) - (ttl_score * 0.5)
        
        return eviction_score
    
    def _calculate_total_size(self) -> int:
        """Calculate total cache size in bytes"""
        return sum(entry.size_bytes for entry in self.cache_entries.values())
    
    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        age = (datetime.utcnow() - entry.created_at).total_seconds()
        return age < entry.ttl_seconds
    
    def _update_access_stats(self, access_time: float):
        """Update access statistics"""
        self.total_access_time += access_time
        self.access_count += 1
    
    async def cleanup_expired_entries(self) -> int:
        """Remove expired cache entries"""
        now = datetime.utcnow()
        expired_keys = []
        
        for cache_key, entry in self.cache_entries.items():
            age = (now - entry.created_at).total_seconds()
            if age >= entry.ttl_seconds:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.cache_entries[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_cache_statistics(self) -> CacheStatistics:
        """Get comprehensive cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        miss_rate = 1.0 - hit_rate
        
        total_size_mb = self._calculate_total_size() / (1024 * 1024)
        
        avg_access_time = (self.total_access_time / self.access_count 
                          if self.access_count > 0 else 0.0)
        
        # Cache efficiency: hit rate weighted by time savings
        cache_efficiency = hit_rate * 0.9  # Assume 90% time savings on cache hits
        
        # Type distribution
        type_distribution = {}
        for entry in self.cache_entries.values():
            cache_type = entry.cache_type.value
            type_distribution[cache_type] = type_distribution.get(cache_type, 0) + 1
        
        return CacheStatistics(
            total_entries=len(self.cache_entries),
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            total_size_mb=total_size_mb,
            average_access_time=avg_access_time,
            cache_efficiency=cache_efficiency,
            eviction_count=self.eviction_count,
            type_distribution=type_distribution
        )

class QuantumMLCache:
    """Specialized cache for quantum machine learning models and results"""
    
    def __init__(self, redis_prefix: str = "quantum_ml"):
        self.redis_prefix = redis_prefix
        self.model_cache: Dict[str, Any] = {}
        
    async def get_trained_model(self, algorithm: str, 
                              dataset_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached trained quantum ML model"""
        cache_key = f"{self.redis_prefix}:model:{algorithm}:{dataset_hash}"
        
        try:
            # Try Redis first
            if redis_client:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            
            # Fall back to local cache
            return self.model_cache.get(cache_key)
            
        except Exception as e:
            logger.error(f"Error retrieving ML model from cache: {e}")
            return None
    
    async def store_trained_model(self, algorithm: str, dataset_hash: str,
                                model_data: Dict[str, Any], ttl: int = 7200) -> bool:
        """Store trained quantum ML model in cache"""
        cache_key = f"{self.redis_prefix}:model:{algorithm}:{dataset_hash}"
        
        try:
            serialized_data = json.dumps(model_data)
            
            # Store in Redis if available
            if redis_client:
                await redis_client.setex(cache_key, ttl, serialized_data)
            
            # Also store locally
            self.model_cache[cache_key] = model_data
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing ML model in cache: {e}")
            return False
    
    async def get_advantage_analysis(self, problem_signature: str) -> Optional[AdvantageAnalysis]:
        """Get cached quantum advantage analysis"""
        cache_key = f"{self.redis_prefix}:advantage:{problem_signature}"
        
        try:
            if redis_client:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    # Reconstruct AdvantageAnalysis object
                    # This would need proper deserialization logic
                    return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving advantage analysis from cache: {e}")
            return None
    
    async def store_advantage_analysis(self, problem_signature: str,
                                     analysis: AdvantageAnalysis, ttl: int = 3600) -> bool:
        """Store quantum advantage analysis in cache"""
        cache_key = f"{self.redis_prefix}:advantage:{problem_signature}"
        
        try:
            # Serialize AdvantageAnalysis
            analysis_data = asdict(analysis)
            # Convert datetime objects to ISO strings
            analysis_data['timestamp'] = analysis.timestamp.isoformat()
            
            serialized_data = json.dumps(analysis_data)
            
            if redis_client:
                await redis_client.setex(cache_key, ttl, serialized_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing advantage analysis in cache: {e}")
            return False

class QuantumCacheManager:
    """Unified manager for all quantum caching strategies"""
    
    def __init__(self, circuit_cache_size_mb: int = 500):
        self.circuit_cache = QuantumCircuitCache(max_size_mb=circuit_cache_size_mb)
        self.ml_cache = QuantumMLCache()
        
        # Background cleanup task
        self.cleanup_task = None
        self.cleanup_interval = 300  # 5 minutes
        
        logger.info("Quantum cache manager initialized")
    
    async def start_background_cleanup(self):
        """Start background cache cleanup task"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._background_cleanup())
    
    async def stop_background_cleanup(self):
        """Stop background cache cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
    
    async def _background_cleanup(self):
        """Background task for cache maintenance"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Clean up expired entries
                expired_count = await self.circuit_cache.cleanup_expired_entries()
                
                # Log cache statistics periodically
                stats = self.circuit_cache.get_cache_statistics()
                logger.info(f"Cache stats: {stats.total_entries} entries, "
                          f"{stats.hit_rate:.2%} hit rate, {stats.total_size_mb:.1f}MB")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all cache types"""
        circuit_stats = self.circuit_cache.get_cache_statistics()
        
        return {
            'circuit_cache': asdict(circuit_stats),
            'ml_cache': {
                'local_entries': len(self.ml_cache.model_cache),
                'redis_available': redis_client is not None
            },
            'total_memory_mb': circuit_stats.total_size_mb,
            'overall_efficiency': circuit_stats.cache_efficiency,
            'timestamp': datetime.utcnow().isoformat()
        }

# Global cache manager instance
quantum_cache_manager: Optional[QuantumCacheManager] = None

def initialize_quantum_cache(circuit_cache_size_mb: int = 500) -> QuantumCacheManager:
    """Initialize the global quantum cache manager"""
    global quantum_cache_manager
    quantum_cache_manager = QuantumCacheManager(circuit_cache_size_mb)
    return quantum_cache_manager

def get_quantum_cache() -> Optional[QuantumCacheManager]:
    """Get the global quantum cache manager instance"""
    return quantum_cache_manager

# Convenience functions for quantum-specific caching
async def cache_circuit_result(gates: List[QuantumGate], num_qubits: int, 
                             shots: int, result: CircuitResult) -> bool:
    """Cache a quantum circuit simulation result"""
    if quantum_cache_manager:
        return await quantum_cache_manager.circuit_cache.store_circuit_result(
            gates, num_qubits, shots, result
        )
    return False

async def get_cached_circuit_result(gates: List[QuantumGate], num_qubits: int, 
                                  shots: int) -> Optional[CircuitResult]:
    """Retrieve cached quantum circuit simulation result"""
    if quantum_cache_manager:
        return await quantum_cache_manager.circuit_cache.get_circuit_result(
            gates, num_qubits, shots
        )
    return None

async def cache_optimized_circuit(original: List[QuantumGate], 
                                optimized: List[QuantumGate]) -> bool:
    """Cache an optimized quantum circuit"""
    if quantum_cache_manager:
        return await quantum_cache_manager.circuit_cache.store_optimized_circuit(
            original, optimized
        )
    return False

async def get_cached_optimized_circuit(gates: List[QuantumGate]) -> Optional[List[QuantumGate]]:
    """Retrieve cached optimized quantum circuit"""
    if quantum_cache_manager:
        return await quantum_cache_manager.circuit_cache.get_optimized_circuit(gates)
    return None
