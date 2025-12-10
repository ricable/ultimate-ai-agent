"""
Semantic memory implementation for the SAFLA memory system.

This module provides concept-based knowledge storage and retrieval
with relationship modeling and semantic reasoning capabilities.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import logging
import networkx as nx

from .base import MemoryItem, MemoryInterface

logger = logging.getLogger(__name__)


@dataclass
class SemanticNode(MemoryItem):
    """
    Semantic memory node representing a concept or knowledge item.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    concept: str = ""
    category: str = "general"
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "user"
    
    def add_attribute(self, key: str, value: Any) -> None:
        """Add or update an attribute."""
        self.attributes[key] = value
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get attribute value."""
        return self.attributes.get(key, default)
    
    def calculate_semantic_similarity(self, other: 'SemanticNode') -> float:
        """Calculate semantic similarity with another node."""
        similarity = 0.0
        
        # Category similarity
        if self.category == other.category:
            similarity += 0.3
        
        # Concept name similarity (simplified)
        if self.concept and other.concept:
            concept_sim = self._string_similarity(self.concept, other.concept)
            similarity += concept_sim * 0.4
        
        # Attribute overlap
        common_attrs = set(self.attributes.keys()) & set(other.attributes.keys())
        if common_attrs:
            attr_similarity = 0.0
            for attr in common_attrs:
                if self.attributes[attr] == other.attributes[attr]:
                    attr_similarity += 1.0
            attr_similarity /= len(common_attrs)
            similarity += attr_similarity * 0.3
        
        return min(1.0, similarity)
    
    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Simple string similarity using common words."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


@dataclass
class SemanticEdge:
    """
    Relationship between semantic nodes.
    """
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_node: str = ""
    to_node: str = ""
    relationship_type: str = "related_to"
    strength: float = 1.0
    bidirectional: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate edge parameters."""
        self.strength = max(0.0, min(1.0, self.strength))


class SemanticMemoryManager(MemoryInterface):
    """
    Manager for semantic memory operations with graph-based knowledge representation.
    """
    
    def __init__(self, max_nodes: int = 50000, max_edges: int = 100000):
        """
        Initialize semantic memory manager.
        
        Args:
            max_nodes: Maximum number of semantic nodes
            max_edges: Maximum number of edges (relationships)
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nodes: Dict[str, SemanticNode] = {}
        self.edges: Dict[str, SemanticEdge] = {}
        self._lock = threading.RLock()
        
        # Graph structure for efficient traversal
        self.graph = nx.DiGraph()
        
        # Indices for fast lookup
        self.concept_index: Dict[str, Set[str]] = defaultdict(set)  # concept -> node_ids
        self.category_index: Dict[str, Set[str]] = defaultdict(set)  # category -> node_ids
        self.attribute_index: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Relationship type index
        self.relationship_index: Dict[str, Set[str]] = defaultdict(set)  # rel_type -> edge_ids
    
    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None,
              concept: Optional[str] = None, category: str = "general",
              attributes: Optional[Dict[str, Any]] = None) -> str:
        """Store semantic node."""
        with self._lock:
            # Create semantic node
            node = SemanticNode(
                content=content,
                metadata=metadata or {},
                concept=concept or str(content)[:50],
                category=category,
                attributes=attributes or {}
            )
            
            # Store node
            self.nodes[node.node_id] = node
            
            # Add to graph
            self.graph.add_node(node.node_id, **{
                'concept': node.concept,
                'category': node.category,
                'attributes': node.attributes
            })
            
            # Update indices
            self.concept_index[node.concept].add(node.node_id)
            self.category_index[node.category].add(node.node_id)
            
            for attr_key, attr_value in node.attributes.items():
                self.attribute_index[attr_key][attr_value].add(node.node_id)
            
            # Auto-create relationships with similar nodes
            self._auto_link_node(node)
            
            # Cleanup if needed
            if len(self.nodes) > self.max_nodes:
                self._cleanup_old_nodes()
            
            logger.debug(f"Stored semantic node: {node.node_id} for concept '{node.concept}'")
            return node.node_id
    
    def add_relationship(self, from_node_id: str, to_node_id: str,
                        relationship_type: str = "related_to",
                        strength: float = 1.0, bidirectional: bool = True) -> str:
        """Add relationship between nodes."""
        with self._lock:
            if from_node_id not in self.nodes or to_node_id not in self.nodes:
                raise ValueError("Both nodes must exist before creating relationship")
            
            # Create edge
            edge = SemanticEdge(
                from_node=from_node_id,
                to_node=to_node_id,
                relationship_type=relationship_type,
                strength=strength,
                bidirectional=bidirectional
            )
            
            # Store edge
            self.edges[edge.edge_id] = edge
            
            # Add to graph
            self.graph.add_edge(from_node_id, to_node_id, **{
                'edge_id': edge.edge_id,
                'type': relationship_type,
                'strength': strength
            })
            
            if bidirectional:
                self.graph.add_edge(to_node_id, from_node_id, **{
                    'edge_id': edge.edge_id,
                    'type': relationship_type,
                    'strength': strength
                })
            
            # Update relationship index
            self.relationship_index[relationship_type].add(edge.edge_id)
            
            # Cleanup edges if needed
            if len(self.edges) > self.max_edges:
                self._cleanup_old_edges()
            
            logger.debug(f"Added relationship: {from_node_id} -> {to_node_id} ({relationship_type})")
            return edge.edge_id
    
    def retrieve(self, query: Union[str, Dict[str, Any]], k: int = 5,
                include_related: bool = True, max_depth: int = 2) -> List[SemanticNode]:
        """Retrieve relevant semantic nodes."""
        with self._lock:
            if isinstance(query, str):
                return self._search_by_concept(query, k, include_related, max_depth)
            elif isinstance(query, dict):
                return self._search_by_attributes(query, k, include_related, max_depth)
            else:
                return []
    
    def _search_by_concept(self, concept: str, k: int, include_related: bool,
                          max_depth: int) -> List[SemanticNode]:
        """Search nodes by concept."""
        matches = []
        
        # Exact concept match
        if concept in self.concept_index:
            for node_id in self.concept_index[concept]:
                if node_id in self.nodes:
                    matches.append(self.nodes[node_id])
        
        # Partial concept match
        for stored_concept, node_ids in self.concept_index.items():
            if concept.lower() in stored_concept.lower() or stored_concept.lower() in concept.lower():
                for node_id in node_ids:
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        if node not in matches:
                            matches.append(node)
        
        # Include related nodes if requested
        if include_related and matches:
            related_nodes = set()
            for node in matches[:min(3, len(matches))]:  # Start from top 3 matches
                related = self._get_related_nodes(node.node_id, max_depth)
                related_nodes.update(related)
            
            for node_id in related_nodes:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    if node not in matches:
                        matches.append(node)
        
        # Sort by relevance and access
        matches.sort(key=lambda n: (n.calculate_current_importance(), n.confidence), reverse=True)
        
        # Update access for retrieved nodes
        for node in matches[:k]:
            node.update_access()
        
        return matches[:k]
    
    def _search_by_attributes(self, attributes: Dict[str, Any], k: int,
                             include_related: bool, max_depth: int) -> List[SemanticNode]:
        """Search nodes by attributes."""
        matching_nodes = None
        
        # Find nodes that match all specified attributes
        for attr_key, attr_value in attributes.items():
            if attr_key in self.attribute_index:
                if attr_value in self.attribute_index[attr_key]:
                    candidate_nodes = self.attribute_index[attr_key][attr_value]
                    if matching_nodes is None:
                        matching_nodes = candidate_nodes.copy()
                    else:
                        matching_nodes &= candidate_nodes
                else:
                    matching_nodes = set()
                    break
            else:
                matching_nodes = set()
                break
        
        if not matching_nodes:
            return []
        
        matches = [self.nodes[node_id] for node_id in matching_nodes if node_id in self.nodes]
        
        # Include related nodes if requested
        if include_related and matches:
            related_nodes = set()
            for node in matches[:min(3, len(matches))]:
                related = self._get_related_nodes(node.node_id, max_depth)
                related_nodes.update(related)
            
            for node_id in related_nodes:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    if node not in matches:
                        matches.append(node)
        
        # Sort by relevance
        matches.sort(key=lambda n: (n.calculate_current_importance(), n.confidence), reverse=True)
        
        # Update access for retrieved nodes
        for node in matches[:k]:
            node.update_access()
        
        return matches[:k]
    
    def _get_related_nodes(self, node_id: str, max_depth: int) -> Set[str]:
        """Get nodes related to the given node within max_depth."""
        if node_id not in self.graph:
            return set()
        
        related = set()
        
        try:
            # Get nodes within max_depth using BFS
            for depth in range(1, max_depth + 1):
                if depth == 1:
                    # Direct neighbors
                    neighbors = set(self.graph.neighbors(node_id))
                    related.update(neighbors)
                else:
                    # Nodes at specific depth
                    paths = nx.single_source_shortest_path_length(
                        self.graph, node_id, cutoff=depth
                    )
                    depth_nodes = {n for n, d in paths.items() if d == depth}
                    related.update(depth_nodes)
        
        except nx.NetworkXError:
            # Handle disconnected components
            pass
        
        return related
    
    def get_node_relationships(self, node_id: str) -> List[Tuple[str, SemanticEdge]]:
        """Get all relationships for a node."""
        with self._lock:
            relationships = []
            
            for edge in self.edges.values():
                if edge.from_node == node_id:
                    relationships.append((edge.to_node, edge))
                elif edge.bidirectional and edge.to_node == node_id:
                    relationships.append((edge.from_node, edge))
            
            return relationships
    
    def find_path(self, from_node_id: str, to_node_id: str,
                  max_length: int = 5) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        with self._lock:
            try:
                path = nx.shortest_path(
                    self.graph, from_node_id, to_node_id, weight=None
                )
                if len(path) <= max_length + 1:  # +1 because path includes both endpoints
                    return path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            
            return None
    
    def get_concept_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """Get clusters of related concepts."""
        with self._lock:
            try:
                # Find connected components
                undirected = self.graph.to_undirected()
                components = list(nx.connected_components(undirected))
                
                # Filter by minimum size
                clusters = [list(component) for component in components 
                           if len(component) >= min_cluster_size]
                
                return clusters
            except:
                return []
    
    def _auto_link_node(self, node: SemanticNode) -> None:
        """Automatically create relationships with similar nodes."""
        # Find similar nodes in same category
        if node.category in self.category_index:
            category_nodes = self.category_index[node.category]
            
            for other_node_id in category_nodes:
                if other_node_id != node.node_id and other_node_id in self.nodes:
                    other_node = self.nodes[other_node_id]
                    similarity = node.calculate_semantic_similarity(other_node)
                    
                    # Create relationship if similarity is high enough
                    if similarity > 0.7:
                        try:
                            self.add_relationship(
                                node.node_id, other_node_id,
                                "similar_to", similarity, True
                            )
                        except ValueError:
                            # Relationship might already exist
                            pass
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update semantic node."""
        with self._lock:
            if memory_id not in self.nodes:
                return False
            
            node = self.nodes[memory_id]
            old_concept = node.concept
            old_category = node.category
            old_attributes = node.attributes.copy()
            
            # Update fields
            for key, value in updates.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            
            # Update indices if needed
            if node.concept != old_concept:
                self.concept_index[old_concept].discard(memory_id)
                self.concept_index[node.concept].add(memory_id)
            
            if node.category != old_category:
                self.category_index[old_category].discard(memory_id)
                self.category_index[node.category].add(memory_id)
            
            # Update attribute indices
            for attr_key, attr_value in old_attributes.items():
                self.attribute_index[attr_key][attr_value].discard(memory_id)
            
            for attr_key, attr_value in node.attributes.items():
                self.attribute_index[attr_key][attr_value].add(memory_id)
            
            # Update graph node attributes
            if memory_id in self.graph:
                self.graph.nodes[memory_id].update({
                    'concept': node.concept,
                    'category': node.category,
                    'attributes': node.attributes
                })
            
            logger.debug(f"Updated semantic node: {memory_id}")
            return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete semantic node and its relationships."""
        with self._lock:
            if memory_id not in self.nodes:
                return False
            
            node = self.nodes[memory_id]
            
            # Remove from indices
            self.concept_index[node.concept].discard(memory_id)
            self.category_index[node.category].discard(memory_id)
            
            for attr_key, attr_value in node.attributes.items():
                self.attribute_index[attr_key][attr_value].discard(memory_id)
            
            # Remove all edges involving this node
            edges_to_remove = []
            for edge_id, edge in self.edges.items():
                if edge.from_node == memory_id or edge.to_node == memory_id:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                edge = self.edges[edge_id]
                self.relationship_index[edge.relationship_type].discard(edge_id)
                del self.edges[edge_id]
            
            # Remove from graph
            if memory_id in self.graph:
                self.graph.remove_node(memory_id)
            
            del self.nodes[memory_id]
            
            logger.debug(f"Deleted semantic node: {memory_id}")
            return True
    
    def clear(self) -> None:
        """Clear all semantic memories."""
        with self._lock:
            self.nodes.clear()
            self.edges.clear()
            self.graph.clear()
            self.concept_index.clear()
            self.category_index.clear()
            self.attribute_index.clear()
            self.relationship_index.clear()
    
    def _cleanup_old_nodes(self) -> None:
        """Remove old, less important nodes."""
        nodes = list(self.nodes.values())
        nodes.sort(key=lambda n: (n.calculate_current_importance(), n.timestamp))
        
        # Remove least important nodes
        to_remove = nodes[:len(nodes) - self.max_nodes + 1]
        for node in to_remove:
            self.delete(node.node_id)
    
    def _cleanup_old_edges(self) -> None:
        """Remove old, weak edges."""
        edges = list(self.edges.values())
        edges.sort(key=lambda e: (e.strength, e.created_at))
        
        # Remove weakest edges
        to_remove = edges[:len(edges) - self.max_edges + 1]
        for edge in to_remove:
            # Remove from graph
            try:
                self.graph.remove_edge(edge.from_node, edge.to_node)
                if edge.bidirectional:
                    self.graph.remove_edge(edge.to_node, edge.from_node)
            except:
                pass
            
            # Remove from relationship index
            self.relationship_index[edge.relationship_type].discard(edge.edge_id)
            del self.edges[edge.edge_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        with self._lock:
            total_nodes = len(self.nodes)
            total_edges = len(self.edges)
            
            # Category distribution
            category_counts = {cat: len(nodes) for cat, nodes in self.category_index.items()}
            
            # Relationship type distribution
            rel_type_counts = {rel_type: len(edges) for rel_type, edges in self.relationship_index.items()}
            
            # Graph metrics
            try:
                if total_nodes > 0:
                    density = nx.density(self.graph)
                    num_components = nx.number_connected_components(self.graph.to_undirected())
                    avg_clustering = nx.average_clustering(self.graph.to_undirected())
                else:
                    density = 0
                    num_components = 0
                    avg_clustering = 0
            except:
                density = 0
                num_components = 0
                avg_clustering = 0
            
            # Access statistics
            access_counts = [n.access_count for n in self.nodes.values()]
            avg_access = sum(access_counts) / len(access_counts) if access_counts else 0
            
            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "unique_concepts": len(self.concept_index),
                "unique_categories": len(self.category_index),
                "category_distribution": category_counts,
                "relationship_types": rel_type_counts,
                "graph_density": density,
                "connected_components": num_components,
                "average_clustering": avg_clustering,
                "average_access_count": avg_access,
                "max_nodes": self.max_nodes,
                "max_edges": self.max_edges
            }