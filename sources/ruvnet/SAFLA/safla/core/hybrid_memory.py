"""
Hybrid Memory Architecture for SAFLA.

This module implements a comprehensive memory system with multiple specialized
memory types that work together to provide efficient storage, retrieval, and
consolidation of information.

Components:
- Vector Memory Management: High-dimensional vector storage with similarity search
- Episodic Memory: Sequential experience storage with temporal indexing
- Semantic Memory: Knowledge graph representation with relationship mapping
- Working Memory: Active context management with attention mechanisms
- Memory Consolidation: Transfer between memory types with importance weighting

Technical Features:
- Support for multiple embedding dimensions (512, 768, 1024, 1536)
- Similarity search with configurable distance metrics (cosine, euclidean)
- Temporal indexing with timestamp-based retrieval
- Graph traversal algorithms for semantic relationships
- LRU and importance-based eviction policies
- Asynchronous operations for scalability
"""

import asyncio
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import heapq
import logging
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Supported similarity metrics for vector operations."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class EvictionPolicy(Enum):
    """Supported eviction policies for memory management."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    IMPORTANCE = "importance"  # Based on importance scores
    TEMPORAL = "temporal"  # Based on age


@dataclass
class MemoryItem:
    """Base class for memory items with common attributes."""
    item_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance_score: float = 0.0


@dataclass
class SimilarityResult:
    """Result from similarity search operations."""
    item_id: str
    similarity_score: float
    item: Optional[MemoryItem] = None


@dataclass
class EpisodicEvent:
    """Represents an episodic memory event with temporal context."""
    event_id: str
    timestamp: datetime
    event_type: str
    context: Dict[str, Any]
    embedding: np.ndarray
    importance_score: float = 0.0
    access_count: int = 0


@dataclass
class SemanticNode:
    """Represents a node in the semantic knowledge graph."""
    node_id: str
    concept: str
    attributes: Dict[str, Any]
    embedding: np.ndarray
    creation_time: datetime = field(default_factory=datetime.now)
    access_count: int = 0


@dataclass
class SemanticEdge:
    """Represents a relationship between semantic nodes."""
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)


@dataclass
class SemanticSubgraph:
    """Represents a subgraph extracted from semantic memory."""
    nodes: List[SemanticNode]
    edges: List[SemanticEdge]
    center_node_id: str
    extraction_depth: int


@dataclass
class WorkingMemoryContext:
    """Represents active context in working memory."""
    context_id: str
    content: str
    attention_weight: float
    timestamp: datetime
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0


class VectorMemoryManager:
    """
    High-performance vector memory management with similarity search.
    
    Supports multiple embedding dimensions and similarity metrics with
    efficient storage and retrieval operations.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        max_capacity: int = 10000,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ):
        """Initialize vector memory manager."""
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        
        # Validate supported dimensions
        supported_dims = [512, 768, 1024, 1536]
        if embedding_dim not in supported_dims:
            logger.warning(f"Embedding dimension {embedding_dim} not in standard sizes {supported_dims}")
        
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.max_capacity = max_capacity
        self.eviction_policy = eviction_policy
        
        # Storage structures
        self._items: Dict[str, MemoryItem] = {}
        self._embeddings: List[np.ndarray] = []
        self._item_ids: List[str] = []
        self._access_order: OrderedDict = OrderedDict()
        
        logger.info(f"Initialized VectorMemoryManager: dim={embedding_dim}, metric={similarity_metric.value}")
    
    @property
    def current_size(self) -> int:
        """Get current number of stored items."""
        return len(self._items)
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return self.current_size == 0
    
    def contains(self, item_id: str) -> bool:
        """Check if item exists in memory."""
        return item_id in self._items
    
    def store(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Store a vector with metadata and return item ID."""
        # Validate embedding dimension
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
        
        # Generate unique ID
        item_id = str(uuid.uuid4())
        
        # Create memory item
        item = MemoryItem(
            item_id=item_id,
            embedding=embedding.copy(),
            metadata=metadata.copy(),
            timestamp=datetime.now()
        )
        
        # Check capacity and evict if necessary
        if self.current_size >= self.max_capacity:
            self._evict_items(1)
        
        # Store item
        self._items[item_id] = item
        self._embeddings.append(embedding.copy())
        self._item_ids.append(item_id)
        self._access_order[item_id] = datetime.now()
        
        logger.debug(f"Stored item {item_id} with embedding shape {embedding.shape}")
        return item_id
    
    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve item by ID."""
        if item_id not in self._items:
            return None
        
        item = self._items[item_id]
        item.access_count += 1
        self._access_order[item_id] = datetime.now()
        
        return item
    
    def delete(self, item_id: str) -> bool:
        """Delete item by ID."""
        if item_id not in self._items:
            return False
        
        # Find index in embeddings list
        try:
            index = self._item_ids.index(item_id)
            del self._embeddings[index]
            del self._item_ids[index]
        except ValueError:
            logger.error(f"Inconsistent state: item {item_id} not found in embeddings list")
        
        # Remove from other structures
        del self._items[item_id]
        self._access_order.pop(item_id, None)
        
        logger.debug(f"Deleted item {item_id}")
        return True
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[SimilarityResult]:
        """Perform similarity search and return top-k results."""
        if self.is_empty():
            return []
        
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.embedding_dim}, got {query_embedding.shape[0]}")
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(self._embeddings):
            similarity = self._calculate_similarity(query_embedding, embedding)
            if similarity >= similarity_threshold:
                similarities.append((similarity, self._item_ids[i]))
        
        # Sort by similarity (descending for cosine/dot product, ascending for distance metrics)
        reverse_sort = self.similarity_metric in [SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT]
        similarities.sort(key=lambda x: x[0], reverse=reverse_sort)
        
        # Return top-k results
        results = []
        for similarity, item_id in similarities[:k]:
            item = self._items[item_id]
            item.access_count += 1
            self._access_order[item_id] = datetime.now()
            
            results.append(SimilarityResult(
                item_id=item_id,
                similarity_score=similarity,
                item=item
            ))
        
        return results
    
    def batch_store(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[Dict[str, Any]]
    ) -> List[str]:
        """Store multiple embeddings in batch."""
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata lists must have same length")
        
        item_ids = []
        for embedding, metadata in zip(embeddings, metadata_list):
            item_id = self.store(embedding, metadata)
            item_ids.append(item_id)
        
        return item_ids
    
    def batch_retrieve(self, item_ids: List[str]) -> List[Optional[MemoryItem]]:
        """Retrieve multiple items in batch."""
        return [self.retrieve(item_id) for item_id in item_ids]
    
    def update_metadata(self, item_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for an existing item."""
        if item_id not in self._items:
            return False
        
        self._items[item_id].metadata.update(metadata)
        self._access_order[item_id] = datetime.now()
        return True
    
    def clear(self):
        """Clear all stored items."""
        self._items.clear()
        self._embeddings.clear()
        self._item_ids.clear()
        self._access_order.clear()
        logger.info("Cleared all items from vector memory")
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two embeddings."""
        if self.similarity_metric == SimilarityMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance (lower is more similar)
            return np.linalg.norm(embedding1 - embedding2)
        
        elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            # Dot product similarity
            return np.dot(embedding1, embedding2)
        
        elif self.similarity_metric == SimilarityMetric.MANHATTAN:
            # Manhattan distance (lower is more similar)
            return np.sum(np.abs(embedding1 - embedding2))
        
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
    
    def _evict_items(self, count: int):
        """Evict items based on eviction policy."""
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used items
            items_to_remove = list(self._access_order.keys())[:count]
            for item_id in items_to_remove:
                self.delete(item_id)
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used items
            items_by_access = sorted(
                self._items.items(),
                key=lambda x: x[1].access_count
            )
            for i in range(min(count, len(items_by_access))):
                self.delete(items_by_access[i][0])
        
        elif self.eviction_policy == EvictionPolicy.IMPORTANCE:
            # Remove items with lowest importance scores
            items_by_importance = sorted(
                self._items.items(),
                key=lambda x: x[1].importance_score
            )
            for i in range(min(count, len(items_by_importance))):
                self.delete(items_by_importance[i][0])
        
        elif self.eviction_policy == EvictionPolicy.TEMPORAL:
            # Remove oldest items
            items_by_age = sorted(
                self._items.items(),
                key=lambda x: x[1].timestamp
            )
            for i in range(min(count, len(items_by_age))):
                self.delete(items_by_age[i][0])


class EpisodicMemory:
    """
    Episodic memory for storing sequential experiences with temporal indexing.
    
    Provides efficient storage and retrieval of events based on time,
    type, and content similarity.
    """
    
    def __init__(self, max_capacity: int = 1000):
        """Initialize episodic memory."""
        self.max_capacity = max_capacity
        
        # Storage structures
        self._events: Dict[str, EpisodicEvent] = {}
        self._temporal_index: List[Tuple[datetime, str]] = []  # (timestamp, event_id)
        self._type_index: Dict[str, List[str]] = defaultdict(list)  # event_type -> [event_ids]
        
        logger.info(f"Initialized EpisodicMemory with capacity {max_capacity}")
    
    @property
    def current_size(self) -> int:
        """Get current number of stored events."""
        return len(self._events)
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return self.current_size == 0
    
    def store_event(self, event: EpisodicEvent) -> bool:
        """Store an episodic event."""
        # Check capacity and evict if necessary
        if self.current_size >= self.max_capacity:
            self._evict_oldest_events(1)
        
        # Store event
        self._events[event.event_id] = event
        
        # Update indices
        self._temporal_index.append((event.timestamp, event.event_id))
        self._temporal_index.sort(key=lambda x: x[0])  # Keep sorted by timestamp
        
        self._type_index[event.event_type].append(event.event_id)
        
        logger.debug(f"Stored episodic event {event.event_id} of type {event.event_type}")
        return True
    
    def retrieve_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[EpisodicEvent]:
        """Retrieve events within a time range."""
        events = []
        for timestamp, event_id in self._temporal_index:
            if start_time <= timestamp <= end_time:
                event = self._events[event_id]
                event.access_count += 1
                events.append(event)
            elif timestamp > end_time:
                break  # Since list is sorted, we can stop here
        
        return events
    
    def retrieve_recent(self, count: int = 10) -> List[EpisodicEvent]:
        """Retrieve most recent events."""
        recent_items = self._temporal_index[-count:] if count <= len(self._temporal_index) else self._temporal_index
        events = []
        
        # Return in reverse chronological order (most recent first)
        for timestamp, event_id in reversed(recent_items):
            event = self._events[event_id]
            event.access_count += 1
            events.append(event)
        
        return events
    
    def retrieve_by_type(self, event_type: str) -> List[EpisodicEvent]:
        """Retrieve events by type."""
        event_ids = self._type_index.get(event_type, [])
        events = []
        
        for event_id in event_ids:
            if event_id in self._events:  # Check in case of inconsistency
                event = self._events[event_id]
                event.access_count += 1
                events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        return events
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[EpisodicEvent]:
        """Find events with similar embeddings."""
        if self.is_empty():
            return []
        
        # Calculate similarities
        similarities = []
        for event in self._events.values():
            # Use cosine similarity
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(event.embedding)
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(query_embedding, event.embedding) / (norm1 * norm2)
                similarities.append((similarity, event))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k events
        results = []
        for similarity, event in similarities[:k]:
            event.access_count += 1
            results.append(event)
        
        return results
    
    def get_temporal_clusters(self, window_minutes: int = 60) -> List[List[EpisodicEvent]]:
        """Group events into temporal clusters."""
        if self.is_empty():
            return []
        
        clusters = []
        current_cluster = []
        window_delta = timedelta(minutes=window_minutes)
        
        for timestamp, event_id in self._temporal_index:
            event = self._events[event_id]
            
            if not current_cluster:
                current_cluster = [event]
            else:
                # Check if event fits in current cluster
                cluster_start = current_cluster[0].timestamp
                if timestamp - cluster_start <= window_delta:
                    current_cluster.append(event)
                else:
                    # Start new cluster
                    clusters.append(current_cluster)
                    current_cluster = [event]
        
        # Add final cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _evict_oldest_events(self, count: int):
        """Evict oldest events to make space."""
        for _ in range(min(count, len(self._temporal_index))):
            if self._temporal_index:
                timestamp, event_id = self._temporal_index.pop(0)
                
                if event_id in self._events:
                    event = self._events[event_id]
                    
                    # Remove from type index
                    if event.event_type in self._type_index:
                        try:
                            self._type_index[event.event_type].remove(event_id)
                        except ValueError:
                            pass
                    
                    # Remove from main storage
                    del self._events[event_id]
                    
                    logger.debug(f"Evicted episodic event {event_id}")


class SemanticMemory:
    """
    Semantic memory implementing a knowledge graph with nodes and relationships.
    
    Provides graph-based storage and traversal for conceptual knowledge
    with support for similarity search and subgraph extraction.
    """
    
    def __init__(self):
        """Initialize semantic memory."""
        # Storage structures
        self._nodes: Dict[str, SemanticNode] = {}
        self._edges: Dict[str, List[SemanticEdge]] = defaultdict(list)  # source_id -> [edges]
        self._reverse_edges: Dict[str, List[SemanticEdge]] = defaultdict(list)  # target_id -> [edges]
        
        logger.info("Initialized SemanticMemory")
    
    @property
    def node_count(self) -> int:
        """Get number of nodes in the graph."""
        return len(self._nodes)
    
    @property
    def edge_count(self) -> int:
        """Get number of edges in the graph."""
        return sum(len(edges) for edges in self._edges.values())
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return self.node_count == 0
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._nodes
    
    def has_relationship(self, source_id: str, target_id: str) -> bool:
        """Check if relationship exists between nodes."""
        if source_id not in self._edges:
            return False
        
        return any(edge.target_id == target_id for edge in self._edges[source_id])
    
    def add_node(self, node: SemanticNode) -> bool:
        """Add a semantic node."""
        if node.node_id in self._nodes:
            return False  # Node already exists
        
        self._nodes[node.node_id] = node
        logger.debug(f"Added semantic node {node.node_id} with concept '{node.concept}'")
        return True
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a relationship between nodes."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return False  # One or both nodes don't exist
        
        # Create edge
        edge = SemanticEdge(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            attributes=attributes or {}
        )
        
        # Add to indices
        self._edges[source_id].append(edge)
        self._reverse_edges[target_id].append(edge)
        
        logger.debug(f"Added relationship {source_id} --{relationship_type}--> {target_id}")
        return True
    
    def get_neighbors(self, node_id: str) -> List[SemanticNode]:
        """Get neighboring nodes."""
        if node_id not in self._nodes:
            return []
        
        neighbors = []
        
        # Outgoing edges
        for edge in self._edges.get(node_id, []):
            if edge.target_id in self._nodes:
                neighbor = self._nodes[edge.target_id]
                neighbor.access_count += 1
                neighbors.append(neighbor)
        
        # Incoming edges
        for edge in self._reverse_edges.get(node_id, []):
            if edge.source_id in self._nodes:
                neighbor = self._nodes[edge.source_id]
                neighbor.access_count += 1
                neighbors.append(neighbor)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_neighbors = []
        for neighbor in neighbors:
            if neighbor.node_id not in seen:
                seen.add(neighbor.node_id)
                unique_neighbors.append(neighbor)
        
        return unique_neighbors
    
    def find_shortest_path(self, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between two nodes using BFS."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return []
        
        if source_id == target_id:
            return [source_id]
        
        # BFS
        queue = [(source_id, [source_id])]
        visited = {source_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            # Check outgoing edges
            for edge in self._edges.get(current_id, []):
                next_id = edge.target_id
                if next_id == target_id:
                    return path + [next_id]
                
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))
            
            # Check incoming edges (for undirected traversal)
            for edge in self._reverse_edges.get(current_id, []):
                next_id = edge.source_id
                if next_id == target_id:
                    return path + [next_id]
                
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))
        
        return []  # No path found
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[SemanticNode]:
        """Find nodes with similar concept embeddings."""
        if self.is_empty():
            return []
        
        # Calculate similarities
        similarities = []
        for node in self._nodes.values():
            # Use cosine similarity
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(node.embedding)
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(query_embedding, node.embedding) / (norm1 * norm2)
                similarities.append((similarity, node))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k nodes
        results = []
        for similarity, node in similarities[:k]:
            node.access_count += 1
            results.append(node)
        
        return results
    
    def extract_subgraph(self, center_node_id: str, depth: int = 1) -> SemanticSubgraph:
        """Extract subgraph around a center node."""
        if center_node_id not in self._nodes:
            return SemanticSubgraph([], [], center_node_id, depth)
        
        visited_nodes = set()
        subgraph_edges = []
        
        # BFS to collect nodes within depth
        queue = [(center_node_id, 0)]
        visited_nodes.add(center_node_id)
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_depth < depth:
                # Add outgoing edges
                for edge in self._edges.get(current_id, []):
                    target_id = edge.target_id
                    subgraph_edges.append(edge)
                    
                    if target_id not in visited_nodes:
                        visited_nodes.add(target_id)
                        queue.append((target_id, current_depth + 1))
                
                # Add incoming edges
                for edge in self._reverse_edges.get(current_id, []):
                    source_id = edge.source_id
                    subgraph_edges.append(edge)
                    
                    if source_id not in visited_nodes:
                        visited_nodes.add(source_id)
                        queue.append((source_id, current_depth + 1))
        
        # Collect nodes
        subgraph_nodes = [self._nodes[node_id] for node_id in visited_nodes if node_id in self._nodes]
        
        return SemanticSubgraph(
            nodes=subgraph_nodes,
            edges=subgraph_edges,
            center_node_id=center_node_id,
            extraction_depth=depth
        )
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its relationships."""
        if node_id not in self._nodes:
            return False
        
        # Remove all edges involving this node
        # Outgoing edges
        if node_id in self._edges:
            for edge in self._edges[node_id]:
                # Remove from reverse index
                if edge.target_id in self._reverse_edges:
                    self._reverse_edges[edge.target_id] = [
                        e for e in self._reverse_edges[edge.target_id]
                        if e.source_id != node_id
                    ]
            del self._edges[node_id]
        
        # Incoming edges
        if node_id in self._reverse_edges:
            for edge in self._reverse_edges[node_id]:
                # Remove from forward index
                if edge.source_id in self._edges:
                    self._edges[edge.source_id] = [
                        e for e in self._edges[edge.source_id]
                        if e.target_id != node_id
                    ]
            del self._reverse_edges[node_id]
        
        # Remove node
        del self._nodes[node_id]
        
        logger.debug(f"Removed semantic node {node_id}")
        return True


class WorkingMemory:
    """
    Working memory for active context management with attention mechanisms.
    
    Manages currently active information with attention-based prioritization
    and temporal decay mechanisms.
    """
    
    def __init__(self, capacity: int = 10, attention_window: int = 5):
        """Initialize working memory."""
        self.capacity = capacity
        self.attention_window = attention_window
        
        # Storage structures
        self._contexts: Dict[str, WorkingMemoryContext] = {}
        self._attention_heap: List[Tuple[float, str]] = []  # (negative_weight, context_id) for max-heap
        
        logger.info(f"Initialized WorkingMemory: capacity={capacity}, attention_window={attention_window}")
    
    @property
    def current_size(self) -> int:
        """Get current number of contexts."""
        return len(self._contexts)
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return self.current_size == 0
    
    def has_context(self, context_id: str) -> bool:
        """Check if context exists."""
        return context_id in self._contexts
    
    def add_context(self, context: WorkingMemoryContext) -> bool:
        """Add context to working memory."""
        # Check capacity and evict if necessary
        if self.current_size >= self.capacity:
            self._evict_lowest_attention()
        
        # Store context
        self._contexts[context.context_id] = context
        
        # Add to attention heap (use negative weight for max-heap behavior)
        heapq.heappush(self._attention_heap, (-context.attention_weight, context.context_id))
        
        logger.debug(f"Added working memory context {context.context_id} with attention {context.attention_weight}")
        return True
    
    def get_context(self, context_id: str) -> Optional[WorkingMemoryContext]:
        """Get context by ID."""
        if context_id not in self._contexts:
            return None
        
        context = self._contexts[context_id]
        context.access_count += 1
        return context
    
    def get_active_contexts(self) -> List[WorkingMemoryContext]:
        """Get all active contexts sorted by attention weight."""
        contexts = list(self._contexts.values())
        contexts.sort(key=lambda x: x.attention_weight, reverse=True)
        return contexts
    
    def get_attention_window(self) -> List[WorkingMemoryContext]:
        """Get contexts in the attention window."""
        active_contexts = self.get_active_contexts()
        return active_contexts[:self.attention_window]
    
    def update_attention_weight(self, context_id: str, new_weight: float) -> bool:
        """Update attention weight of a context."""
        if context_id not in self._contexts:
            return False
        
        self._contexts[context_id].attention_weight = new_weight
        
        # Rebuild heap (simple approach - could be optimized)
        self._rebuild_attention_heap()
        
        logger.debug(f"Updated attention weight for {context_id} to {new_weight}")
        return True
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[WorkingMemoryContext]:
        """Find contexts with similar embeddings."""
        if self.is_empty():
            return []
        
        # Calculate similarities
        similarities = []
        for context in self._contexts.values():
            # Use cosine similarity
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(context.embedding)
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(query_embedding, context.embedding) / (norm1 * norm2)
                similarities.append((similarity, context))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k contexts
        results = []
        for similarity, context in similarities[:k]:
            context.access_count += 1
            results.append(context)
        
        return results
    
    def apply_temporal_decay(self, decay_rate: float = 0.1):
        """Apply temporal decay to attention weights."""
        current_time = datetime.now()
        
        for context in self._contexts.values():
            # Calculate time since last access
            time_diff = current_time - context.timestamp
            hours_passed = time_diff.total_seconds() / 3600
            
            # Apply exponential decay
            decay_factor = np.exp(-decay_rate * hours_passed)
            context.attention_weight *= decay_factor
        
        # Rebuild attention heap
        self._rebuild_attention_heap()
        
        logger.debug(f"Applied temporal decay with rate {decay_rate}")
    
    def clear(self):
        """Clear all contexts."""
        self._contexts.clear()
        self._attention_heap.clear()
        logger.info("Cleared working memory")
    
    def _evict_lowest_attention(self):
        """Evict context with lowest attention weight."""
        if not self._contexts:
            return
        
        # Find context with lowest attention weight
        min_context = min(self._contexts.values(), key=lambda x: x.attention_weight)
        context_id = min_context.context_id
        
        # Remove from storage
        del self._contexts[context_id]
        
        # Rebuild heap
        self._rebuild_attention_heap()
        
        logger.debug(f"Evicted working memory context {context_id}")
    
    def _rebuild_attention_heap(self):
        """Rebuild the attention heap."""
        self._attention_heap.clear()
        for context in self._contexts.values():
            heapq.heappush(self._attention_heap, (-context.attention_weight, context.context_id))


class MemoryConsolidator:
    """
    Memory consolidation system for transferring information between memory types.
    
    Implements importance-based consolidation algorithms to move information
    from working memory to episodic memory and from episodic to semantic memory.
    """
    
    def __init__(
        self,
        vector_memory: VectorMemoryManager,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        working_memory: WorkingMemory
    ):
        """Initialize memory consolidator."""
        self.vector_memory = vector_memory
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory
        
        # Consolidation metrics
        self._consolidation_stats = {
            'total_consolidations': 0,
            'working_to_episodic_count': 0,
            'episodic_to_semantic_count': 0,
            'average_importance_score': 0.0
        }
        
        # Consolidation task
        self._consolidation_task: Optional[asyncio.Task] = None
        self._consolidation_running = False
        
        logger.info("Initialized MemoryConsolidator")
    
    async def consolidate_working_to_episodic(
        self,
        importance_threshold: float = 0.7
    ) -> int:
        """Consolidate important contexts from working memory to episodic memory."""
        consolidated_count = 0
        contexts_to_remove = []
        
        for context in self.working_memory.get_active_contexts():
            # Calculate importance score
            importance = await self._calculate_context_importance(context)
            
            if importance >= importance_threshold:
                # Create episodic event
                event = EpisodicEvent(
                    event_id=f"consolidated_{context.context_id}",
                    timestamp=context.timestamp,
                    event_type="working_memory_consolidation",
                    context={
                        'content': context.content,
                        'original_context_id': context.context_id,
                        'attention_weight': context.attention_weight,
                        'metadata': context.metadata
                    },
                    embedding=context.embedding.copy(),
                    importance_score=importance
                )
                
                # Store in episodic memory
                if self.episodic_memory.store_event(event):
                    contexts_to_remove.append(context.context_id)
                    consolidated_count += 1
                    
                    logger.debug(f"Consolidated context {context.context_id} to episodic memory")
        
        # Remove consolidated contexts from working memory
        for context_id in contexts_to_remove:
            if context_id in self.working_memory._contexts:
                del self.working_memory._contexts[context_id]
        
        # Rebuild working memory heap
        self.working_memory._rebuild_attention_heap()
        
        # Update stats
        self._consolidation_stats['working_to_episodic_count'] += consolidated_count
        self._consolidation_stats['total_consolidations'] += consolidated_count
        
        logger.info(f"Consolidated {consolidated_count} contexts from working to episodic memory")
        return consolidated_count
    
    async def consolidate_episodic_to_semantic(
        self,
        similarity_threshold: float = 0.8,
        min_event_count: int = 3
    ) -> int:
        """Consolidate related episodic events into semantic knowledge."""
        if self.episodic_memory.is_empty():
            return 0
        
        consolidated_count = 0
        
        # Get all events
        all_events = []
        for event_id, event in self.episodic_memory._events.items():
            all_events.append(event)
        
        # Group similar events
        event_clusters = await self._cluster_similar_events(all_events, similarity_threshold)
        
        for cluster in event_clusters:
            if len(cluster) >= min_event_count:
                # Create semantic node from cluster
                node = await self._create_semantic_node_from_cluster(cluster)
                
                if self.semantic_memory.add_node(node):
                    consolidated_count += 1
                    logger.debug(f"Created semantic node {node.node_id} from {len(cluster)} events")
        
        # Update stats
        self._consolidation_stats['episodic_to_semantic_count'] += consolidated_count
        self._consolidation_stats['total_consolidations'] += consolidated_count
        
        logger.info(f"Consolidated {consolidated_count} semantic nodes from episodic events")
        return consolidated_count
    
    async def calculate_importance_scores(self) -> Dict[str, float]:
        """Calculate importance scores for all working memory contexts."""
        scores = {}
        
        for context_id, context in self.working_memory._contexts.items():
            importance = await self._calculate_context_importance(context)
            scores[context_id] = importance
        
        return scores
    
    async def start_scheduled_consolidation(self, interval_seconds: float = 300):
        """Start scheduled consolidation process."""
        if self._consolidation_running:
            return
        
        self._consolidation_running = True
        self._consolidation_task = asyncio.create_task(
            self._consolidation_loop(interval_seconds)
        )
        
        logger.info(f"Started scheduled consolidation with {interval_seconds}s interval")
    
    async def stop_scheduled_consolidation(self):
        """Stop scheduled consolidation process."""
        self._consolidation_running = False
        
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
            self._consolidation_task = None
        
        logger.info("Stopped scheduled consolidation")
    
    def get_consolidation_metrics(self) -> Dict[str, Any]:
        """Get consolidation metrics."""
        return self._consolidation_stats.copy()
    
    async def _calculate_context_importance(self, context: WorkingMemoryContext) -> float:
        """Calculate importance score for a context."""
        # Base importance from attention weight
        importance = context.attention_weight
        
        # Boost for high access count
        access_boost = min(context.access_count * 0.1, 0.3)
        importance += access_boost
        
        # Boost for recent activity
        time_since_creation = datetime.now() - context.timestamp
        if time_since_creation.total_seconds() < 3600:  # Within last hour
            importance += 0.1
        
        # Normalize to [0, 1]
        return min(importance, 1.0)
    
    async def _cluster_similar_events(
        self,
        events: List[EpisodicEvent],
        similarity_threshold: float
    ) -> List[List[EpisodicEvent]]:
        """Cluster similar events based on embedding similarity."""
        if not events:
            return []
        
        clusters = []
        used_events = set()
        
        for i, event in enumerate(events):
            if i in used_events:
                continue
            
            cluster = [event]
            used_events.add(i)
            
            # Find similar events
            for j, other_event in enumerate(events):
                if j in used_events or i == j:
                    continue
                
                # Calculate similarity
                norm1 = np.linalg.norm(event.embedding)
                norm2 = np.linalg.norm(other_event.embedding)
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(event.embedding, other_event.embedding) / (norm1 * norm2)
                    
                    if similarity >= similarity_threshold:
                        cluster.append(other_event)
                        used_events.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    async def _create_semantic_node_from_cluster(
        self,
        event_cluster: List[EpisodicEvent]
    ) -> SemanticNode:
        """Create a semantic node from a cluster of related events."""
        # Generate node ID
        node_id = f"semantic_{uuid.uuid4()}"
        
        # Extract common concept
        event_types = [event.event_type for event in event_cluster]
        most_common_type = max(set(event_types), key=event_types.count)
        
        # Average embeddings
        embeddings = np.array([event.embedding for event in event_cluster])
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Collect attributes
        attributes = {
            'source_events': [event.event_id for event in event_cluster],
            'event_count': len(event_cluster),
            'common_type': most_common_type,
            'time_span': {
                'start': min(event.timestamp for event in event_cluster).isoformat(),
                'end': max(event.timestamp for event in event_cluster).isoformat()
            }
        }
        
        return SemanticNode(
            node_id=node_id,
            concept=f"consolidated_{most_common_type}",
            attributes=attributes,
            embedding=avg_embedding
        )
    
    async def _consolidation_loop(self, interval_seconds: float):
        """Main consolidation loop."""
        while self._consolidation_running:
            try:
                # Perform consolidation
                await self.consolidate_working_to_episodic()
                await self.consolidate_episodic_to_semantic()
                
                # Wait for next cycle
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                await asyncio.sleep(interval_seconds)


class HybridMemoryArchitecture:
    """
    Integrated hybrid memory architecture combining all memory types.
    
    Provides a unified interface for memory operations across vector,
    episodic, semantic, and working memory systems with automatic
    consolidation and cross-memory search capabilities.
    """
    
    def __init__(
        self,
        vector_config: Optional[Dict[str, Any]] = None,
        episodic_config: Optional[Dict[str, Any]] = None,
        working_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize hybrid memory architecture."""
        # Initialize memory components
        vector_config = vector_config or {}
        self.vector_memory = VectorMemoryManager(**vector_config)
        
        episodic_config = episodic_config or {}
        self.episodic_memory = EpisodicMemory(**episodic_config)
        
        self.semantic_memory = SemanticMemory()
        
        working_config = working_config or {}
        self.working_memory = WorkingMemory(**working_config)
        
        # Initialize consolidator
        self.consolidator = MemoryConsolidator(
            vector_memory=self.vector_memory,
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            working_memory=self.working_memory
        )
        
        logger.info("Initialized HybridMemoryArchitecture")
    
    def get_memory_statistics(self) -> Dict[str, int]:
        """Get statistics for all memory components."""
        return {
            'vector_memory_size': self.vector_memory.current_size,
            'episodic_memory_size': self.episodic_memory.current_size,
            'semantic_memory_nodes': self.semantic_memory.node_count,
            'semantic_memory_edges': self.semantic_memory.edge_count,
            'working_memory_size': self.working_memory.current_size,
            'total_memory_items': (
                self.vector_memory.current_size +
                self.episodic_memory.current_size +
                self.semantic_memory.node_count +
                self.working_memory.current_size
            )
        }
    
    async def integrated_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        search_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform integrated search across all memory types."""
        if search_types is None:
            search_types = ['vector', 'episodic', 'semantic', 'working']
        
        results = []
        
        # Vector memory search
        if 'vector' in search_types:
            vector_results = self.vector_memory.similarity_search(query_embedding, k)
            for result in vector_results:
                results.append({
                    'type': 'vector',
                    'item_id': result.item_id,
                    'similarity_score': result.similarity_score,
                    'item': result.item
                })
        
        # Episodic memory search
        if 'episodic' in search_types:
            episodic_results = self.episodic_memory.similarity_search(query_embedding, k)
            for event in episodic_results:
                results.append({
                    'type': 'episodic',
                    'item_id': event.event_id,
                    'item': event
                })
        
        # Semantic memory search
        if 'semantic' in search_types:
            semantic_results = self.semantic_memory.similarity_search(query_embedding, k)
            for node in semantic_results:
                results.append({
                    'type': 'semantic',
                    'item_id': node.node_id,
                    'item': node
                })
        
        # Working memory search
        if 'working' in search_types:
            working_results = self.working_memory.similarity_search(query_embedding, k)
            for context in working_results:
                results.append({
                    'type': 'working',
                    'item_id': context.context_id,
                    'item': context
                })
        
        return results
    
    def cleanup_memory(
        self,
        max_age_hours: float = 24,
        min_importance: float = 0.5
    ) -> Dict[str, Any]:
        """Perform memory cleanup and maintenance."""
        cleanup_stats = {
            'items_removed': 0,
            'memory_freed': 0
        }
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        # Clean working memory
        contexts_to_remove = []
        for context_id, context in self.working_memory._contexts.items():
            if (context.timestamp < cutoff_time or 
                context.attention_weight < min_importance):
                contexts_to_remove.append(context_id)
        
        for context_id in contexts_to_remove:
            del self.working_memory._contexts[context_id]
            cleanup_stats['items_removed'] += 1
        
        # Rebuild working memory heap
        self.working_memory._rebuild_attention_heap()
        
        logger.info(f"Cleaned up {cleanup_stats['items_removed']} items from memory")
        return cleanup_stats
    
    async def save_to_disk(self, file_path: Union[str, Path], encrypt: bool = True) -> None:
        """
        Save memory state to disk with optional encryption.
        
        Args:
            file_path: Path to save memory state
            encrypt: Whether to encrypt the saved data
        """
        from safla.security import DataEncryptor, EncryptionError
        
        file_path = Path(file_path)
        
        # Prepare memory state
        memory_state = {
            'vector_memory': {
                'memories': list(self.vector_memory.memories.values()),
                'metadata': self.vector_memory.index_metadata
            },
            'episodic_memory': {
                'memories': list(self.episodic_memory.memories.values()),
                'sequences': list(self.episodic_memory.sequences.values())
            },
            'semantic_memory': {
                'nodes': list(self.semantic_memory.nodes.values()),
                'edges': self.semantic_memory.edges
            },
            'working_memory': {
                'contexts': list(self.working_memory.contexts.values()),
                'attention_scores': dict(self.working_memory.attention_scores)
            },
            'metadata': {
                'save_time': datetime.now().isoformat(),
                'version': '1.0',
                'encrypted': encrypt
            }
        }
        
        try:
            # Serialize to JSON
            json_data = json.dumps(memory_state, default=str)
            
            if encrypt:
                # Encrypt the data
                encryptor = DataEncryptor()
                encrypted_data = encryptor.encrypt_string(json_data)
                
                # Save encrypted data
                with open(file_path, 'w') as f:
                    f.write(encrypted_data)
                
                logger.info(f"Memory state saved and encrypted to {file_path}")
            else:
                # Save unencrypted
                with open(file_path, 'w') as f:
                    f.write(json_data)
                
                logger.info(f"Memory state saved to {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to save memory state: {e}")
            raise
    
    async def load_from_disk(self, file_path: Union[str, Path]) -> None:
        """
        Load memory state from disk with automatic decryption if needed.
        
        Args:
            file_path: Path to load memory state from
        """
        from safla.security import DataEncryptor, EncryptionError
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Memory state file not found: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try to parse as JSON first (unencrypted)
            try:
                memory_state = json.loads(content)
            except json.JSONDecodeError:
                # Might be encrypted, try to decrypt
                try:
                    encryptor = DataEncryptor()
                    decrypted = encryptor.decrypt_string(content)
                    memory_state = json.loads(decrypted)
                except (EncryptionError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to decrypt or parse memory state: {e}")
                    raise ValueError("Invalid memory state file format")
            
            # Clear existing memories
            await self.clear_all()
            
            # Restore vector memories
            for memory_data in memory_state.get('vector_memory', {}).get('memories', []):
                memory = VectorMemory(**memory_data)
                self.vector_memory.memories[memory.memory_id] = memory
                # Note: FAISS index would need to be rebuilt
            
            # Restore episodic memories
            for episode_data in memory_state.get('episodic_memory', {}).get('memories', []):
                episode = EpisodeMemory(**episode_data)
                self.episodic_memory.memories[episode.episode_id] = episode
            
            for sequence_data in memory_state.get('episodic_memory', {}).get('sequences', []):
                sequence = Sequence(**sequence_data)
                self.episodic_memory.sequences[sequence.sequence_id] = sequence
            
            # Restore semantic memories
            for node_data in memory_state.get('semantic_memory', {}).get('nodes', []):
                node = SemanticNode(**node_data)
                self.semantic_memory.nodes[node.node_id] = node
            
            self.semantic_memory.edges = memory_state.get('semantic_memory', {}).get('edges', defaultdict(list))
            
            # Restore working memory
            for context_data in memory_state.get('working_memory', {}).get('contexts', []):
                context = WorkingContext(**context_data)
                self.working_memory.contexts[context.context_id] = context
            
            self.working_memory.attention_scores = memory_state.get('working_memory', {}).get('attention_scores', {})
            
            # Rebuild indices
            self.vector_memory._rebuild_index()
            self.episodic_memory._rebuild_temporal_index()
            self.working_memory._rebuild_attention_heap()
            
            logger.info(f"Memory state loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load memory state: {e}")
            raise