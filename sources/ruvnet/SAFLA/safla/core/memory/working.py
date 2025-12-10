"""
Working memory implementation for the SAFLA memory system.

This module provides short-term, active memory capabilities for immediate
context management and temporary information storage.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import logging

from .base import MemoryItem, MemoryInterface

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem(MemoryItem):
    """
    Working memory item with priority and activation tracking.
    """
    priority: int = 5  # 1-10, higher is more important
    activation_level: float = 1.0  # Current activation strength
    decay_rate: float = 0.1  # How fast activation decays
    context_tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)  # Other memory IDs this depends on
    
    def update_activation(self, boost: float = 0.2) -> None:
        """Update activation level with boost."""
        self.activation_level = min(1.0, self.activation_level + boost)
        self.update_access()
    
    def decay_activation(self, time_delta: float) -> None:
        """Decay activation over time."""
        decay_amount = self.decay_rate * time_delta
        self.activation_level = max(0.0, self.activation_level - decay_amount)
    
    def calculate_working_importance(self) -> float:
        """Calculate importance for working memory retention."""
        base_importance = self.calculate_current_importance()
        
        # Factor in activation level
        activation_factor = self.activation_level * 0.4
        
        # Factor in priority
        priority_factor = (self.priority / 10.0) * 0.3
        
        # Recent access factor
        time_since_access = (datetime.now() - self.last_accessed).total_seconds()
        recency_factor = max(0, 1 - (time_since_access / 3600)) * 0.3  # Decay over 1 hour
        
        return min(1.0, base_importance + activation_factor + priority_factor + recency_factor)


@dataclass
class WorkingMemoryCluster:
    """
    A cluster of related working memory items.
    """
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    items: Set[str] = field(default_factory=set)  # Memory IDs
    cluster_type: str = "general"
    coherence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_item(self, memory_id: str) -> None:
        """Add item to cluster."""
        self.items.add(memory_id)
        self.last_updated = datetime.now()
    
    def remove_item(self, memory_id: str) -> None:
        """Remove item from cluster."""
        self.items.discard(memory_id)
        self.last_updated = datetime.now()
    
    def calculate_relevance(self, context_tags: Set[str]) -> float:
        """Calculate cluster relevance to context."""
        if not context_tags:
            return 0.0
        
        # Simple relevance based on cluster type matching context
        relevance = 0.0
        if self.cluster_type in context_tags:
            relevance += 0.5
        
        relevance += self.coherence_score * 0.3
        
        # Recency factor
        time_diff = (datetime.now() - self.last_updated).total_seconds()
        recency = max(0, 1 - (time_diff / 3600))  # Decay over 1 hour
        relevance += recency * 0.2
        
        return min(1.0, relevance)


class WorkingMemoryManager(MemoryInterface):
    """
    Manager for working memory operations with activation-based retention.
    """
    
    def __init__(self, max_items: int = 200, max_clusters: int = 50):
        """
        Initialize working memory manager.
        
        Args:
            max_items: Maximum number of working memory items
            max_clusters: Maximum number of memory clusters
        """
        self.max_items = max_items
        self.max_clusters = max_clusters
        self.memories: Dict[str, WorkingMemoryItem] = {}
        self.clusters: Dict[str, WorkingMemoryCluster] = {}
        self._lock = threading.RLock()
        
        # Indices for fast access
        self.priority_index: Dict[int, Set[str]] = defaultdict(set)  # priority -> memory_ids
        self.context_index: Dict[str, Set[str]] = defaultdict(set)  # context_tag -> memory_ids
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # memory_id -> dependents
        
        # Decay management
        self.last_decay_time = datetime.now()
        self.decay_interval = timedelta(minutes=5)  # Run decay every 5 minutes
    
    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None,
              priority: int = 5, context_tags: Optional[Set[str]] = None,
              dependencies: Optional[Set[str]] = None) -> str:
        """Store working memory item."""
        with self._lock:
            # Create working memory item
            item = WorkingMemoryItem(
                content=content,
                metadata=metadata or {},
                priority=max(1, min(10, priority)),
                context_tags=context_tags or set(),
                dependencies=dependencies or set()
            )
            
            # Store item
            self.memories[item.memory_id] = item
            
            # Update indices
            self.priority_index[item.priority].add(item.memory_id)
            for tag in item.context_tags:
                self.context_index[tag].add(item.memory_id)
            
            # Update dependency graph
            for dep_id in item.dependencies:
                self.dependency_graph[dep_id].add(item.memory_id)
            
            # Auto-cluster similar items
            self._auto_cluster_item(item)
            
            # Perform periodic decay
            self._periodic_decay()
            
            # Cleanup if needed
            if len(self.memories) > self.max_items:
                self._cleanup_low_activation()
            
            logger.debug(f"Stored working memory: {item.memory_id} with priority {item.priority}")
            return item.memory_id
    
    def retrieve(self, query: Union[str, Dict[str, Any]], k: int = 5,
                context_tags: Optional[Set[str]] = None,
                min_activation: float = 0.1) -> List[WorkingMemoryItem]:
        """Retrieve active working memory items."""
        with self._lock:
            self._periodic_decay()
            
            if isinstance(query, str):
                return self._search_by_content(query, k, context_tags, min_activation)
            elif isinstance(query, dict):
                return self._search_by_metadata(query, k, context_tags, min_activation)
            else:
                return []
    
    def _search_by_content(self, query: str, k: int, context_tags: Optional[Set[str]],
                          min_activation: float) -> List[WorkingMemoryItem]:
        """Search by content similarity."""
        matches = []
        
        for item in self.memories.values():
            # Skip low activation items
            if item.activation_level < min_activation:
                continue
            
            # Context filtering
            if context_tags and not (item.context_tags & context_tags):
                continue
            
            # Simple content matching
            if isinstance(item.content, str) and query.lower() in item.content.lower():
                matches.append(item)
        
        # Sort by working memory importance
        matches.sort(key=lambda x: x.calculate_working_importance(), reverse=True)
        
        # Update activation for retrieved items
        for item in matches[:k]:
            item.update_activation()
        
        return matches[:k]
    
    def _search_by_metadata(self, metadata_query: Dict[str, Any], k: int,
                           context_tags: Optional[Set[str]], min_activation: float) -> List[WorkingMemoryItem]:
        """Search by metadata matching."""
        matches = []
        
        for item in self.memories.values():
            # Skip low activation items
            if item.activation_level < min_activation:
                continue
            
            # Context filtering
            if context_tags and not (item.context_tags & context_tags):
                continue
            
            # Metadata matching
            match_score = 0
            for key, value in metadata_query.items():
                if key in item.metadata and item.metadata[key] == value:
                    match_score += 1
            
            if match_score > 0:
                # Add relevance score based on matches
                relevance = match_score / len(metadata_query)
                matches.append((relevance, item))
        
        # Sort by relevance and working importance
        matches.sort(key=lambda x: (x[0], x[1].calculate_working_importance()), reverse=True)
        
        # Update activation for retrieved items
        results = []
        for relevance, item in matches[:k]:
            item.update_activation()
            results.append(item)
        
        return results
    
    def activate_item(self, memory_id: str, boost: float = 0.3) -> bool:
        """Manually activate a working memory item."""
        with self._lock:
            if memory_id not in self.memories:
                return False
            
            item = self.memories[memory_id]
            item.update_activation(boost)
            
            # Also activate dependent items slightly
            for dependent_id in self.dependency_graph.get(memory_id, set()):
                if dependent_id in self.memories:
                    self.memories[dependent_id].update_activation(boost * 0.3)
            
            logger.debug(f"Activated working memory: {memory_id}")
            return True
    
    def get_active_context(self, context_tags: Optional[Set[str]] = None,
                          min_activation: float = 0.3) -> List[WorkingMemoryItem]:
        """Get currently active items in working memory."""
        with self._lock:
            self._periodic_decay()
            
            active_items = []
            for item in self.memories.values():
                if item.activation_level >= min_activation:
                    if context_tags is None or (item.context_tags & context_tags):
                        active_items.append(item)
            
            # Sort by activation level and importance
            active_items.sort(
                key=lambda x: (x.activation_level, x.calculate_working_importance()),
                reverse=True
            )
            
            return active_items
    
    def get_clusters(self, context_tags: Optional[Set[str]] = None) -> List[WorkingMemoryCluster]:
        """Get memory clusters, optionally filtered by context."""
        with self._lock:
            if context_tags is None:
                clusters = list(self.clusters.values())
            else:
                clusters = []
                for cluster in self.clusters.values():
                    if cluster.calculate_relevance(context_tags) > 0.3:
                        clusters.append(cluster)
            
            # Sort by relevance
            if context_tags:
                clusters.sort(key=lambda c: c.calculate_relevance(context_tags), reverse=True)
            else:
                clusters.sort(key=lambda c: c.coherence_score, reverse=True)
            
            return clusters
    
    def _auto_cluster_item(self, item: WorkingMemoryItem) -> None:
        """Automatically cluster similar working memory items."""
        best_cluster = None
        best_similarity = 0.0
        
        # Find best matching cluster
        for cluster in self.clusters.values():
            similarity = self._calculate_cluster_similarity(item, cluster)
            if similarity > best_similarity and similarity > 0.6:
                best_similarity = similarity
                best_cluster = cluster
        
        if best_cluster:
            # Add to existing cluster
            best_cluster.add_item(item.memory_id)
            best_cluster.coherence_score = (best_cluster.coherence_score + best_similarity) / 2
        else:
            # Create new cluster if we have similar items
            similar_items = self._find_similar_items(item)
            if len(similar_items) >= 2:  # Need at least 2 similar items
                cluster = WorkingMemoryCluster(
                    items={item.memory_id} | {i.memory_id for i in similar_items},
                    cluster_type=list(item.context_tags)[0] if item.context_tags else "general",
                    coherence_score=0.7
                )
                self.clusters[cluster.cluster_id] = cluster
                
                # Cleanup old clusters if needed
                if len(self.clusters) > self.max_clusters:
                    self._cleanup_weak_clusters()
    
    def _calculate_cluster_similarity(self, item: WorkingMemoryItem, 
                                    cluster: WorkingMemoryCluster) -> float:
        """Calculate similarity between item and cluster."""
        similarity = 0.0
        
        # Context tag overlap
        cluster_items = [self.memories[mid] for mid in cluster.items if mid in self.memories]
        if cluster_items:
            all_tags = set()
            for ci in cluster_items:
                all_tags.update(ci.context_tags)
            
            if all_tags and item.context_tags:
                tag_overlap = len(item.context_tags & all_tags) / len(item.context_tags | all_tags)
                similarity += tag_overlap * 0.5
            
            # Priority similarity
            avg_priority = sum(ci.priority for ci in cluster_items) / len(cluster_items)
            priority_sim = 1.0 - abs(item.priority - avg_priority) / 10.0
            similarity += priority_sim * 0.3
            
            # Coherence factor
            similarity += cluster.coherence_score * 0.2
        
        return similarity
    
    def _find_similar_items(self, item: WorkingMemoryItem) -> List[WorkingMemoryItem]:
        """Find items similar to the given item."""
        similar = []
        
        for other in self.memories.values():
            if other.memory_id != item.memory_id:
                similarity = 0.0
                
                # Context tag similarity
                if item.context_tags and other.context_tags:
                    tag_overlap = len(item.context_tags & other.context_tags)
                    tag_union = len(item.context_tags | other.context_tags)
                    similarity += (tag_overlap / tag_union) * 0.6 if tag_union > 0 else 0
                
                # Priority similarity
                priority_sim = 1.0 - abs(item.priority - other.priority) / 10.0
                similarity += priority_sim * 0.4
                
                if similarity > 0.5:
                    similar.append(other)
        
        return similar
    
    def _periodic_decay(self) -> None:
        """Perform periodic activation decay."""
        now = datetime.now()
        if now - self.last_decay_time >= self.decay_interval:
            time_delta = (now - self.last_decay_time).total_seconds() / 3600  # Hours
            
            for item in self.memories.values():
                item.decay_activation(time_delta)
            
            self.last_decay_time = now
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update working memory item."""
        with self._lock:
            if memory_id not in self.memories:
                return False
            
            item = self.memories[memory_id]
            old_priority = item.priority
            old_context_tags = item.context_tags.copy()
            
            # Update fields
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            
            # Update indices if needed
            if item.priority != old_priority:
                self.priority_index[old_priority].discard(memory_id)
                self.priority_index[item.priority].add(memory_id)
            
            if item.context_tags != old_context_tags:
                for tag in old_context_tags:
                    self.context_index[tag].discard(memory_id)
                for tag in item.context_tags:
                    self.context_index[tag].add(memory_id)
            
            # Boost activation since item was updated
            item.update_activation()
            
            logger.debug(f"Updated working memory: {memory_id}")
            return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete working memory item."""
        with self._lock:
            if memory_id not in self.memories:
                return False
            
            item = self.memories[memory_id]
            
            # Remove from indices
            self.priority_index[item.priority].discard(memory_id)
            for tag in item.context_tags:
                self.context_index[tag].discard(memory_id)
            
            # Remove from dependency graph
            for dep_id in item.dependencies:
                self.dependency_graph[dep_id].discard(memory_id)
            
            # Remove from clusters
            for cluster in self.clusters.values():
                cluster.remove_item(memory_id)
            
            del self.memories[memory_id]
            
            logger.debug(f"Deleted working memory: {memory_id}")
            return True
    
    def clear(self) -> None:
        """Clear all working memory."""
        with self._lock:
            self.memories.clear()
            self.clusters.clear()
            self.priority_index.clear()
            self.context_index.clear()
            self.dependency_graph.clear()
    
    def _cleanup_low_activation(self) -> None:
        """Remove items with low activation levels."""
        items = list(self.memories.values())
        items.sort(key=lambda x: x.activation_level)
        
        # Remove lowest activation items
        to_remove = items[:len(items) - self.max_items + 10]  # Remove extra to avoid frequent cleanup
        for item in to_remove:
            if item.activation_level < 0.1:  # Only remove very low activation
                self.delete(item.memory_id)
    
    def _cleanup_weak_clusters(self) -> None:
        """Remove weak or small clusters."""
        clusters = list(self.clusters.values())
        clusters.sort(key=lambda c: (len(c.items), c.coherence_score))
        
        # Remove weakest clusters
        to_remove = clusters[:len(clusters) - self.max_clusters + 5]
        for cluster in to_remove:
            if len(cluster.items) < 2 or cluster.coherence_score < 0.3:
                del self.clusters[cluster.cluster_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        with self._lock:
            total_items = len(self.memories)
            total_clusters = len(self.clusters)
            
            # Activation statistics
            activations = [m.activation_level for m in self.memories.values()]
            avg_activation = sum(activations) / len(activations) if activations else 0
            active_items = sum(1 for a in activations if a > 0.3)
            
            # Priority distribution
            priority_dist = {}
            for priority, items in self.priority_index.items():
                priority_dist[f"priority_{priority}"] = len(items)
            
            # Context diversity
            unique_contexts = len(self.context_index)
            
            return {
                "total_items": total_items,
                "total_clusters": total_clusters,
                "active_items": active_items,
                "average_activation": avg_activation,
                "priority_distribution": priority_dist,
                "unique_contexts": unique_contexts,
                "dependency_connections": len(self.dependency_graph),
                "max_items": self.max_items,
                "max_clusters": self.max_clusters
            }