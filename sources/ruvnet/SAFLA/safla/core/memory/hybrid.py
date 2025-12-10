"""
Hybrid memory system implementation for SAFLA.

This module integrates vector, episodic, semantic, and working memory systems
into a unified, coordinated memory architecture with cross-memory interactions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from .base import MemoryInterface, MemoryItem, SimilarityMetric
from .vector import VectorMemoryManager, VectorMemory
from .episodic import EpisodicMemoryManager, EpisodeMemory
from .semantic import SemanticMemoryManager, SemanticNode
from .working import WorkingMemoryManager, WorkingMemoryItem

logger = logging.getLogger(__name__)


@dataclass
class HybridMemoryResult:
    """
    Result from hybrid memory query with cross-memory connections.
    """
    primary_results: List[MemoryItem] = field(default_factory=list)
    vector_results: List[VectorMemory] = field(default_factory=list)
    episodic_results: List[EpisodeMemory] = field(default_factory=list)
    semantic_results: List[SemanticNode] = field(default_factory=list)
    working_results: List[WorkingMemoryItem] = field(default_factory=list)
    cross_connections: Dict[str, List[str]] = field(default_factory=dict)
    confidence_score: float = 0.0
    query_time: float = 0.0


@dataclass
class MemoryConfiguration:
    """
    Configuration for hybrid memory system.
    """
    # Memory system limits
    max_vector_items: int = 50000
    max_episodic_items: int = 10000
    max_semantic_nodes: int = 50000
    max_working_items: int = 200
    
    # Cross-memory settings
    enable_cross_connections: bool = True
    connection_threshold: float = 0.7
    max_connections_per_item: int = 10
    
    # Performance settings
    parallel_queries: bool = True
    max_query_threads: int = 4
    query_timeout: float = 30.0
    
    # Auto-consolidation settings
    consolidation_interval: timedelta = timedelta(hours=6)
    consolidation_threshold: int = 100  # Items before consolidation
    working_to_longterm_threshold: float = 0.8


class HybridMemorySystem(MemoryInterface):
    """
    Unified memory system integrating all memory types with cross-connections.
    """
    
    def __init__(self, config: Optional[MemoryConfiguration] = None):
        """
        Initialize hybrid memory system.
        
        Args:
            config: Memory configuration, uses defaults if None
        """
        self.config = config or MemoryConfiguration()
        self._lock = threading.RLock()
        
        # Initialize memory managers
        self.vector_memory = VectorMemoryManager(dimension=768)
        self.episodic_memory = EpisodicMemoryManager(
            max_episodes=self.config.max_episodic_items
        )
        self.semantic_memory = SemanticMemoryManager(
            max_nodes=self.config.max_semantic_nodes
        )
        self.working_memory = WorkingMemoryManager(
            max_items=self.config.max_working_items
        )
        
        # Cross-memory connections
        self.cross_connections: Dict[str, Dict[str, Set[str]]] = {
            'vector': defaultdict(set),
            'episodic': defaultdict(set),
            'semantic': defaultdict(set),
            'working': defaultdict(set)
        }
        
        # Consolidation tracking
        self.last_consolidation = datetime.now()
        self.consolidation_candidates: Set[str] = set()
        
        # Thread pool for parallel operations
        if self.config.parallel_queries:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_query_threads)
        else:
            self.thread_pool = None
        
        logger.info("Initialized hybrid memory system")
    
    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None,
              memory_type: str = "auto", **kwargs) -> str:
        """
        Store item in appropriate memory system(s).
        
        Args:
            content: Content to store
            metadata: Optional metadata
            memory_type: Target memory type ("auto", "vector", "episodic", "semantic", "working")
            **kwargs: Additional parameters for specific memory types
        """
        with self._lock:
            metadata = metadata or {}
            memory_ids = []
            
            # Determine storage strategy
            if memory_type == "auto":
                storage_types = self._determine_storage_types(content, metadata, **kwargs)
            else:
                storage_types = [memory_type]
            
            # Store in each determined memory type
            for storage_type in storage_types:
                try:
                    if storage_type == "vector":
                        memory_id = self.vector_memory.store(content, metadata, kwargs.get('vector'))
                        memory_ids.append(('vector', memory_id))
                    
                    elif storage_type == "episodic":
                        memory_id = self.episodic_memory.store(
                            content, metadata,
                            context=kwargs.get('context'),
                            triggers=kwargs.get('triggers'),
                            outcomes=kwargs.get('outcomes')
                        )
                        memory_ids.append(('episodic', memory_id))
                    
                    elif storage_type == "semantic":
                        memory_id = self.semantic_memory.store(
                            content, metadata,
                            concept=kwargs.get('concept'),
                            category=kwargs.get('category', 'general'),
                            attributes=kwargs.get('attributes')
                        )
                        memory_ids.append(('semantic', memory_id))
                    
                    elif storage_type == "working":
                        memory_id = self.working_memory.store(
                            content, metadata,
                            priority=kwargs.get('priority', 5),
                            context_tags=kwargs.get('context_tags'),
                            dependencies=kwargs.get('dependencies')
                        )
                        memory_ids.append(('working', memory_id))
                        
                except Exception as e:
                    logger.error(f"Failed to store in {storage_type} memory: {e}")
            
            # Create cross-connections if enabled
            if self.config.enable_cross_connections and len(memory_ids) > 1:
                self._create_cross_connections(memory_ids)
            
            # Add to consolidation candidates if working memory
            if any(storage_type == "working" for storage_type, _ in memory_ids):
                for _, memory_id in memory_ids:
                    self.consolidation_candidates.add(memory_id)
            
            # Check for auto-consolidation
            self._check_auto_consolidation()
            
            # Return primary memory ID (first one stored)
            primary_id = memory_ids[0][1] if memory_ids else str(uuid.uuid4())
            
            logger.debug(f"Stored in hybrid memory: {memory_ids}")
            return primary_id
    
    def retrieve(self, query: Union[str, Dict[str, Any]], k: int = 5,
                memory_types: Optional[List[str]] = None,
                include_cross_connections: bool = True,
                **kwargs) -> HybridMemoryResult:
        """
        Retrieve from multiple memory systems with cross-connections.
        """
        start_time = datetime.now()
        memory_types = memory_types or ['vector', 'episodic', 'semantic', 'working']
        
        result = HybridMemoryResult()
        
        try:
            if self.config.parallel_queries and self.thread_pool:
                # Parallel retrieval
                result = self._parallel_retrieve(query, k, memory_types, **kwargs)
            else:
                # Sequential retrieval
                result = self._sequential_retrieve(query, k, memory_types, **kwargs)
            
            # Add cross-connections if enabled
            if include_cross_connections and self.config.enable_cross_connections:
                result.cross_connections = self._get_cross_connections(result)
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence(result)
            
        except Exception as e:
            logger.error(f"Hybrid memory retrieval failed: {e}")
        
        result.query_time = (datetime.now() - start_time).total_seconds()
        return result
    
    def _determine_storage_types(self, content: Any, metadata: Dict[str, Any], 
                               **kwargs) -> List[str]:
        """Automatically determine which memory systems to use."""
        storage_types = []
        
        # Always store in vector memory for similarity search
        storage_types.append("vector")
        
        # Check for episodic indicators
        if (kwargs.get('context') or kwargs.get('triggers') or 
            'timestamp' in metadata or 'sequence' in metadata):
            storage_types.append("episodic")
        
        # Check for semantic indicators
        if (kwargs.get('concept') or kwargs.get('category') or 
            kwargs.get('attributes') or 'relationships' in metadata):
            storage_types.append("semantic")
        
        # Check for working memory indicators
        if (kwargs.get('priority', 0) > 7 or kwargs.get('context_tags') or
            metadata.get('temporary', False) or kwargs.get('dependencies')):
            storage_types.append("working")
        
        # Ensure at least vector storage
        if not storage_types:
            storage_types = ["vector"]
        
        return storage_types
    
    def _parallel_retrieve(self, query: Union[str, Dict[str, Any]], k: int,
                          memory_types: List[str], **kwargs) -> HybridMemoryResult:
        """Retrieve from multiple memory systems in parallel."""
        result = HybridMemoryResult()
        futures = {}
        
        # Submit retrieval tasks
        if 'vector' in memory_types:
            futures['vector'] = self.thread_pool.submit(
                self.vector_memory.retrieve, query, k,
                kwargs.get('threshold', 0.5)
            )
        
        if 'episodic' in memory_types:
            futures['episodic'] = self.thread_pool.submit(
                self.episodic_memory.retrieve, query, k,
                kwargs.get('time_window')
            )
        
        if 'semantic' in memory_types:
            futures['semantic'] = self.thread_pool.submit(
                self.semantic_memory.retrieve, query, k,
                kwargs.get('include_related', True),
                kwargs.get('max_depth', 2)
            )
        
        if 'working' in memory_types:
            futures['working'] = self.thread_pool.submit(
                self.working_memory.retrieve, query, k,
                kwargs.get('context_tags'),
                kwargs.get('min_activation', 0.1)
            )
        
        # Collect results with timeout
        for memory_type, future in futures.items():
            try:
                results = future.result(timeout=self.config.query_timeout)
                
                if memory_type == 'vector':
                    result.vector_results = results
                elif memory_type == 'episodic':
                    result.episodic_results = results
                elif memory_type == 'semantic':
                    result.semantic_results = results
                elif memory_type == 'working':
                    result.working_results = results
                
            except Exception as e:
                logger.error(f"Failed to retrieve from {memory_type} memory: {e}")
        
        # Combine all results
        result.primary_results = (
            result.vector_results + result.episodic_results + 
            result.semantic_results + result.working_results
        )
        
        return result
    
    def _sequential_retrieve(self, query: Union[str, Dict[str, Any]], k: int,
                           memory_types: List[str], **kwargs) -> HybridMemoryResult:
        """Retrieve from memory systems sequentially."""
        result = HybridMemoryResult()
        
        if 'vector' in memory_types:
            try:
                result.vector_results = self.vector_memory.retrieve(
                    query, k, kwargs.get('threshold', 0.5)
                )
            except Exception as e:
                logger.error(f"Vector memory retrieval failed: {e}")
        
        if 'episodic' in memory_types:
            try:
                result.episodic_results = self.episodic_memory.retrieve(
                    query, k, kwargs.get('time_window')
                )
            except Exception as e:
                logger.error(f"Episodic memory retrieval failed: {e}")
        
        if 'semantic' in memory_types:
            try:
                result.semantic_results = self.semantic_memory.retrieve(
                    query, k, kwargs.get('include_related', True),
                    kwargs.get('max_depth', 2)
                )
            except Exception as e:
                logger.error(f"Semantic memory retrieval failed: {e}")
        
        if 'working' in memory_types:
            try:
                result.working_results = self.working_memory.retrieve(
                    query, k, kwargs.get('context_tags'),
                    kwargs.get('min_activation', 0.1)
                )
            except Exception as e:
                logger.error(f"Working memory retrieval failed: {e}")
        
        # Combine all results
        result.primary_results = (
            result.vector_results + result.episodic_results + 
            result.semantic_results + result.working_results
        )
        
        return result
    
    def _create_cross_connections(self, memory_ids: List[Tuple[str, str]]) -> None:
        """Create cross-connections between memory items."""
        for i, (type1, id1) in enumerate(memory_ids):
            for j, (type2, id2) in enumerate(memory_ids):
                if i != j:  # Don't connect to self
                    # Add bidirectional connections
                    self.cross_connections[type1][id1].add(f"{type2}:{id2}")
                    self.cross_connections[type2][id2].add(f"{type1}:{id1}")
    
    def _get_cross_connections(self, result: HybridMemoryResult) -> Dict[str, List[str]]:
        """Get cross-connections for retrieved items."""
        connections = {}
        
        # Check connections for all result items
        all_items = [
            ('vector', item) for item in result.vector_results
        ] + [
            ('episodic', item) for item in result.episodic_results
        ] + [
            ('semantic', item) for item in result.semantic_results
        ] + [
            ('working', item) for item in result.working_results
        ]
        
        for memory_type, item in all_items:
            item_key = f"{memory_type}:{item.memory_id}"
            item_connections = []
            
            if hasattr(item, 'memory_id'):
                memory_id = item.memory_id
            elif hasattr(item, 'episode_id'):
                memory_id = item.episode_id
            elif hasattr(item, 'node_id'):
                memory_id = item.node_id
            else:
                continue
            
            # Get connections from cross_connections
            if memory_type in self.cross_connections:
                connected_items = self.cross_connections[memory_type].get(memory_id, set())
                item_connections.extend(list(connected_items))
            
            if item_connections:
                connections[item_key] = item_connections
        
        return connections
    
    def _calculate_confidence(self, result: HybridMemoryResult) -> float:
        """Calculate confidence score for hybrid result."""
        if not result.primary_results:
            return 0.0
        
        # Base confidence from number of results
        result_count_factor = min(1.0, len(result.primary_results) / 5.0)
        
        # Memory type diversity factor
        memory_types_used = 0
        if result.vector_results:
            memory_types_used += 1
        if result.episodic_results:
            memory_types_used += 1
        if result.semantic_results:
            memory_types_used += 1
        if result.working_results:
            memory_types_used += 1
        
        diversity_factor = memory_types_used / 4.0
        
        # Cross-connection factor
        connection_factor = min(1.0, len(result.cross_connections) / 10.0)
        
        # Working memory activation factor
        working_activation = 0.0
        if result.working_results:
            activations = [item.activation_level for item in result.working_results]
            working_activation = sum(activations) / len(activations)
        
        # Combine factors
        confidence = (
            result_count_factor * 0.3 +
            diversity_factor * 0.3 +
            connection_factor * 0.2 +
            working_activation * 0.2
        )
        
        return min(1.0, confidence)
    
    def consolidate_memories(self, working_memory_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Consolidate working memories to long-term storage.
        
        Args:
            working_memory_ids: Specific IDs to consolidate, or None for auto-selection
        
        Returns:
            Dictionary with consolidation statistics
        """
        with self._lock:
            stats = {'vector': 0, 'episodic': 0, 'semantic': 0, 'removed': 0}
            
            # Select items for consolidation
            if working_memory_ids is None:
                candidates = []
                for item in self.working_memory.memories.values():
                    if (item.calculate_working_importance() >= self.config.working_to_longterm_threshold or
                        item.access_count > 5):
                        candidates.append(item)
            else:
                candidates = [
                    self.working_memory.memories[mid] 
                    for mid in working_memory_ids 
                    if mid in self.working_memory.memories
                ]
            
            # Consolidate each candidate
            for item in candidates:
                try:
                    # Store in long-term memory systems
                    if hasattr(item, 'context_tags') and item.context_tags:
                        # Store as semantic memory
                        self.semantic_memory.store(
                            item.content, item.metadata,
                            concept=str(item.content)[:50],
                            category=list(item.context_tags)[0] if item.context_tags else "general",
                            attributes={'consolidated_from': 'working', 'original_priority': item.priority}
                        )
                        stats['semantic'] += 1
                    
                    # Store as episodic if it has temporal context
                    if 'timestamp' in item.metadata or item.access_count > 3:
                        self.episodic_memory.store(
                            item.content, item.metadata,
                            context={'consolidated_from': 'working', 'priority': item.priority},
                            triggers=[f"working_memory_{item.priority}"]
                        )
                        stats['episodic'] += 1
                    
                    # Always store as vector for similarity search
                    self.vector_memory.store(item.content, {
                        **item.metadata,
                        'consolidated_from': 'working',
                        'original_priority': item.priority
                    })
                    stats['vector'] += 1
                    
                    # Remove from working memory
                    self.working_memory.delete(item.memory_id)
                    self.consolidation_candidates.discard(item.memory_id)
                    stats['removed'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to consolidate memory {item.memory_id}: {e}")
            
            self.last_consolidation = datetime.now()
            logger.info(f"Consolidated {len(candidates)} working memories: {stats}")
            return stats
    
    def _check_auto_consolidation(self) -> None:
        """Check if auto-consolidation should be triggered."""
        now = datetime.now()
        
        # Check time interval
        if now - self.last_consolidation >= self.config.consolidation_interval:
            self.consolidate_memories()
            return
        
        # Check candidate threshold
        if len(self.consolidation_candidates) >= self.config.consolidation_threshold:
            candidate_ids = list(self.consolidation_candidates)[:self.config.consolidation_threshold]
            self.consolidate_memories(candidate_ids)
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory item across all systems."""
        updated = False
        
        # Try updating in each memory system
        systems = [
            ('vector', self.vector_memory),
            ('episodic', self.episodic_memory),
            ('semantic', self.semantic_memory),
            ('working', self.working_memory)
        ]
        
        for system_name, system in systems:
            try:
                if system.update(memory_id, updates):
                    updated = True
                    logger.debug(f"Updated {memory_id} in {system_name} memory")
            except Exception as e:
                logger.debug(f"Memory {memory_id} not found in {system_name}: {e}")
        
        return updated
    
    def delete(self, memory_id: str) -> bool:
        """Delete memory item from all systems."""
        deleted = False
        
        # Remove from all memory systems
        systems = [
            ('vector', self.vector_memory),
            ('episodic', self.episodic_memory),
            ('semantic', self.semantic_memory),
            ('working', self.working_memory)
        ]
        
        for system_name, system in systems:
            try:
                if system.delete(memory_id):
                    deleted = True
                    logger.debug(f"Deleted {memory_id} from {system_name} memory")
            except Exception as e:
                logger.debug(f"Memory {memory_id} not found in {system_name}: {e}")
        
        # Remove cross-connections
        for memory_type in self.cross_connections:
            if memory_id in self.cross_connections[memory_type]:
                del self.cross_connections[memory_type][memory_id]
            
            # Remove references to this memory
            for connections in self.cross_connections[memory_type].values():
                connections.discard(f"{memory_type}:{memory_id}")
        
        # Remove from consolidation candidates
        self.consolidation_candidates.discard(memory_id)
        
        return deleted
    
    def clear(self) -> None:
        """Clear all memory systems."""
        with self._lock:
            self.vector_memory.clear()
            self.episodic_memory.clear()
            self.semantic_memory.clear()
            self.working_memory.clear()
            
            # Clear cross-connections
            for memory_type in self.cross_connections:
                self.cross_connections[memory_type].clear()
            
            self.consolidation_candidates.clear()
            self.last_consolidation = datetime.now()
            
            logger.info("Cleared all hybrid memory systems")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        with self._lock:
            stats = {
                'vector_memory': self.vector_memory.get_stats(),
                'episodic_memory': self.episodic_memory.get_stats(),
                'semantic_memory': self.semantic_memory.get_stats(),
                'working_memory': self.working_memory.get_stats(),
                'cross_connections': {
                    memory_type: len(connections)
                    for memory_type, connections in self.cross_connections.items()
                },
                'consolidation_candidates': len(self.consolidation_candidates),
                'last_consolidation': self.last_consolidation.isoformat(),
                'configuration': {
                    'max_vector_items': self.config.max_vector_items,
                    'max_episodic_items': self.config.max_episodic_items,
                    'max_semantic_nodes': self.config.max_semantic_nodes,
                    'max_working_items': self.config.max_working_items,
                    'enable_cross_connections': self.config.enable_cross_connections,
                    'parallel_queries': self.config.parallel_queries
                }
            }
            
            # Calculate total items across all systems
            stats['total_items'] = (
                stats['vector_memory']['total_memories'] +
                stats['episodic_memory']['total_episodes'] +
                stats['semantic_memory']['total_nodes'] +
                stats['working_memory']['total_items']
            )
            
            return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup thread pool."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


# Import collections.defaultdict which was missing
from collections import defaultdict