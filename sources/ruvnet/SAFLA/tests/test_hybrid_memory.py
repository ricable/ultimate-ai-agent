"""
Test suite for Hybrid Memory Architecture implementation.

This module tests the core memory management components:
- Vector Memory Management with similarity search
- Episodic Memory with temporal indexing
- Semantic Memory with knowledge graphs
- Working Memory with attention mechanisms
- Memory Consolidation between memory types

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock

# Import the classes we'll implement
from safla.core.hybrid_memory import (
    VectorMemoryManager,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    MemoryConsolidator,
    HybridMemoryArchitecture,
    MemoryItem,
    EpisodicEvent,
    SemanticNode,
    WorkingMemoryContext,
    SimilarityMetric,
    EvictionPolicy
)


class TestVectorMemoryManager:
    """Test suite for Vector Memory Management functionality."""
    
    @pytest.fixture
    def vector_manager(self):
        """Create a VectorMemoryManager instance for testing."""
        return VectorMemoryManager(
            embedding_dim=512,
            similarity_metric=SimilarityMetric.COSINE,
            max_capacity=1000
        )
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)  # For reproducible tests
        return [
            np.random.rand(512).astype(np.float32) for _ in range(10)
        ]
    
    def test_vector_manager_initialization(self, vector_manager):
        """Test VectorMemoryManager initialization with correct parameters."""
        assert vector_manager.embedding_dim == 512
        assert vector_manager.similarity_metric == SimilarityMetric.COSINE
        assert vector_manager.max_capacity == 1000
        assert vector_manager.current_size == 0
        assert vector_manager.is_empty()
    
    def test_vector_manager_invalid_dimensions(self):
        """Test VectorMemoryManager rejects invalid embedding dimensions."""
        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            VectorMemoryManager(embedding_dim=0)
        
        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            VectorMemoryManager(embedding_dim=-1)
    
    def test_vector_manager_supported_dimensions(self):
        """Test VectorMemoryManager supports standard embedding dimensions."""
        supported_dims = [512, 768, 1024, 1536]
        for dim in supported_dims:
            manager = VectorMemoryManager(embedding_dim=dim)
            assert manager.embedding_dim == dim
    
    def test_store_single_vector(self, vector_manager, sample_embeddings):
        """Test storing a single vector with metadata."""
        embedding = sample_embeddings[0]
        metadata = {"type": "test", "timestamp": datetime.now()}
        
        item_id = vector_manager.store(embedding, metadata)
        
        assert item_id is not None
        assert vector_manager.current_size == 1
        assert not vector_manager.is_empty()
        assert vector_manager.contains(item_id)
    
    def test_store_multiple_vectors(self, vector_manager, sample_embeddings):
        """Test storing multiple vectors."""
        item_ids = []
        for i, embedding in enumerate(sample_embeddings):
            metadata = {"index": i, "type": "batch_test"}
            item_id = vector_manager.store(embedding, metadata)
            item_ids.append(item_id)
        
        assert len(item_ids) == len(sample_embeddings)
        assert vector_manager.current_size == len(sample_embeddings)
        assert all(vector_manager.contains(item_id) for item_id in item_ids)
    
    def test_store_invalid_embedding_dimension(self, vector_manager):
        """Test storing embedding with wrong dimensions raises error."""
        wrong_dim_embedding = np.random.rand(256).astype(np.float32)
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            vector_manager.store(wrong_dim_embedding, {})
    
    def test_similarity_search_cosine(self, vector_manager, sample_embeddings):
        """Test cosine similarity search functionality."""
        # Store embeddings
        stored_ids = []
        for i, embedding in enumerate(sample_embeddings):
            item_id = vector_manager.store(embedding, {"index": i})
            stored_ids.append(item_id)
        
        # Search with the first embedding (should return itself as most similar)
        query_embedding = sample_embeddings[0]
        results = vector_manager.similarity_search(query_embedding, k=3)
        
        assert len(results) == 3
        assert results[0].item_id == stored_ids[0]  # Most similar should be itself
        assert results[0].similarity_score == pytest.approx(1.0, abs=1e-6)
        
        # Check results are sorted by similarity (descending)
        similarities = [result.similarity_score for result in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_similarity_search_euclidean(self, sample_embeddings):
        """Test Euclidean distance similarity search."""
        manager = VectorMemoryManager(
            embedding_dim=512,
            similarity_metric=SimilarityMetric.EUCLIDEAN
        )
        
        # Store embeddings
        for i, embedding in enumerate(sample_embeddings):
            manager.store(embedding, {"index": i})
        
        # Search with the first embedding
        query_embedding = sample_embeddings[0]
        results = manager.similarity_search(query_embedding, k=3)
        
        assert len(results) == 3
        assert results[0].similarity_score == pytest.approx(0.0, abs=1e-6)  # Distance to itself
    
    def test_similarity_search_empty_store(self, vector_manager):
        """Test similarity search on empty vector store."""
        query_embedding = np.random.rand(512).astype(np.float32)
        results = vector_manager.similarity_search(query_embedding, k=5)
        
        assert len(results) == 0
    
    def test_similarity_search_k_larger_than_store(self, vector_manager, sample_embeddings):
        """Test similarity search when k is larger than stored items."""
        # Store only 3 embeddings
        for i in range(3):
            vector_manager.store(sample_embeddings[i], {"index": i})
        
        # Search for 10 items
        query_embedding = sample_embeddings[0]
        results = vector_manager.similarity_search(query_embedding, k=10)
        
        assert len(results) == 3  # Should return only available items
    
    def test_retrieve_by_id(self, vector_manager, sample_embeddings):
        """Test retrieving stored items by ID."""
        embedding = sample_embeddings[0]
        metadata = {"type": "test_retrieve"}
        
        item_id = vector_manager.store(embedding, metadata)
        retrieved_item = vector_manager.retrieve(item_id)
        
        assert retrieved_item is not None
        assert retrieved_item.item_id == item_id
        assert np.array_equal(retrieved_item.embedding, embedding)
        assert retrieved_item.metadata == metadata
    
    def test_retrieve_nonexistent_id(self, vector_manager):
        """Test retrieving non-existent item returns None."""
        fake_id = "non_existent_id"
        retrieved_item = vector_manager.retrieve(fake_id)
        
        assert retrieved_item is None
    
    def test_delete_item(self, vector_manager, sample_embeddings):
        """Test deleting items from vector store."""
        embedding = sample_embeddings[0]
        item_id = vector_manager.store(embedding, {})
        
        assert vector_manager.contains(item_id)
        assert vector_manager.current_size == 1
        
        success = vector_manager.delete(item_id)
        
        assert success
        assert not vector_manager.contains(item_id)
        assert vector_manager.current_size == 0
        assert vector_manager.is_empty()
    
    def test_delete_nonexistent_item(self, vector_manager):
        """Test deleting non-existent item returns False."""
        fake_id = "non_existent_id"
        success = vector_manager.delete(fake_id)
        
        assert not success
    
    def test_capacity_management(self):
        """Test vector manager respects capacity limits."""
        small_manager = VectorMemoryManager(embedding_dim=512, max_capacity=3)
        
        # Store items up to capacity
        embeddings = [np.random.rand(512).astype(np.float32) for _ in range(5)]
        stored_ids = []
        
        for i, embedding in enumerate(embeddings):
            item_id = small_manager.store(embedding, {"index": i})
            stored_ids.append(item_id)
        
        # Should not exceed capacity
        assert small_manager.current_size <= 3
        
        # Oldest items should be evicted (LRU policy)
        assert not small_manager.contains(stored_ids[0])
        assert not small_manager.contains(stored_ids[1])
        assert small_manager.contains(stored_ids[2])
        assert small_manager.contains(stored_ids[3])
        assert small_manager.contains(stored_ids[4])
    
    def test_update_item_metadata(self, vector_manager, sample_embeddings):
        """Test updating metadata of stored items."""
        embedding = sample_embeddings[0]
        original_metadata = {"type": "original", "value": 1}
        
        item_id = vector_manager.store(embedding, original_metadata)
        
        # Update metadata
        new_metadata = {"type": "updated", "value": 2}
        success = vector_manager.update_metadata(item_id, new_metadata)
        
        assert success
        
        # Verify update
        retrieved_item = vector_manager.retrieve(item_id)
        assert retrieved_item.metadata == new_metadata
    
    def test_batch_operations(self, vector_manager, sample_embeddings):
        """Test batch storage and retrieval operations."""
        metadata_list = [{"index": i} for i in range(len(sample_embeddings))]
        
        # Batch store
        item_ids = vector_manager.batch_store(sample_embeddings, metadata_list)
        
        assert len(item_ids) == len(sample_embeddings)
        assert vector_manager.current_size == len(sample_embeddings)
        
        # Batch retrieve
        retrieved_items = vector_manager.batch_retrieve(item_ids)
        
        assert len(retrieved_items) == len(item_ids)
        assert all(item is not None for item in retrieved_items)
    
    def test_clear_all(self, vector_manager, sample_embeddings):
        """Test clearing all stored vectors."""
        # Store some items
        for embedding in sample_embeddings[:3]:
            vector_manager.store(embedding, {})
        
        assert vector_manager.current_size == 3
        
        # Clear all
        vector_manager.clear()
        
        assert vector_manager.current_size == 0
        assert vector_manager.is_empty()


class TestEpisodicMemory:
    """Test suite for Episodic Memory functionality."""
    
    @pytest.fixture
    def episodic_memory(self):
        """Create an EpisodicMemory instance for testing."""
        return EpisodicMemory(max_capacity=100)
    
    @pytest.fixture
    def sample_events(self):
        """Generate sample episodic events for testing."""
        base_time = datetime.now()
        return [
            EpisodicEvent(
                event_id=f"event_{i}",
                timestamp=base_time + timedelta(minutes=i),
                event_type="test_event",
                context={"action": f"action_{i}", "result": f"result_{i}"},
                embedding=np.random.rand(512).astype(np.float32)
            )
            for i in range(10)
        ]
    
    def test_episodic_memory_initialization(self, episodic_memory):
        """Test EpisodicMemory initialization."""
        assert episodic_memory.max_capacity == 100
        assert episodic_memory.current_size == 0
        assert episodic_memory.is_empty()
    
    def test_store_episodic_event(self, episodic_memory, sample_events):
        """Test storing a single episodic event."""
        event = sample_events[0]
        
        success = episodic_memory.store_event(event)
        
        assert success
        assert episodic_memory.current_size == 1
        assert not episodic_memory.is_empty()
    
    def test_store_multiple_events(self, episodic_memory, sample_events):
        """Test storing multiple episodic events."""
        for event in sample_events:
            success = episodic_memory.store_event(event)
            assert success
        
        assert episodic_memory.current_size == len(sample_events)
    
    def test_retrieve_by_time_range(self, episodic_memory, sample_events):
        """Test retrieving events by time range."""
        # Store events
        for event in sample_events:
            episodic_memory.store_event(event)
        
        # Query for events in middle time range
        start_time = sample_events[2].timestamp
        end_time = sample_events[7].timestamp
        
        retrieved_events = episodic_memory.retrieve_by_time_range(start_time, end_time)
        
        assert len(retrieved_events) == 6  # Events 2-7 inclusive
        assert all(start_time <= event.timestamp <= end_time for event in retrieved_events)
        
        # Events should be sorted by timestamp
        timestamps = [event.timestamp for event in retrieved_events]
        assert timestamps == sorted(timestamps)
    
    def test_retrieve_recent_events(self, episodic_memory, sample_events):
        """Test retrieving most recent events."""
        # Store events
        for event in sample_events:
            episodic_memory.store_event(event)
        
        # Get 3 most recent events
        recent_events = episodic_memory.retrieve_recent(count=3)
        
        assert len(recent_events) == 3
        
        # Should be the last 3 events in reverse chronological order
        expected_events = sample_events[-3:][::-1]
        assert [e.event_id for e in recent_events] == [e.event_id for e in expected_events]
    
    def test_retrieve_by_event_type(self, episodic_memory):
        """Test retrieving events by type."""
        # Create events with different types
        events = [
            EpisodicEvent(
                event_id=f"event_{i}",
                timestamp=datetime.now() + timedelta(minutes=i),
                event_type="type_a" if i % 2 == 0 else "type_b",
                context={},
                embedding=np.random.rand(512).astype(np.float32)
            )
            for i in range(6)
        ]
        
        for event in events:
            episodic_memory.store_event(event)
        
        # Retrieve type_a events
        type_a_events = episodic_memory.retrieve_by_type("type_a")
        assert len(type_a_events) == 3
        assert all(event.event_type == "type_a" for event in type_a_events)
        
        # Retrieve type_b events
        type_b_events = episodic_memory.retrieve_by_type("type_b")
        assert len(type_b_events) == 3
        assert all(event.event_type == "type_b" for event in type_b_events)
    
    def test_similarity_search_events(self, episodic_memory, sample_events):
        """Test similarity search within episodic events."""
        # Store events
        for event in sample_events:
            episodic_memory.store_event(event)
        
        # Search using first event's embedding
        query_embedding = sample_events[0].embedding
        similar_events = episodic_memory.similarity_search(query_embedding, k=3)
        
        assert len(similar_events) == 3
        assert similar_events[0].event_id == sample_events[0].event_id  # Most similar to itself
    
    def test_temporal_clustering(self, episodic_memory, sample_events):
        """Test temporal clustering of events."""
        # Store events
        for event in sample_events:
            episodic_memory.store_event(event)
        
        # Get temporal clusters with 5-minute windows
        clusters = episodic_memory.get_temporal_clusters(window_minutes=5)
        
        assert len(clusters) > 0
        # Each cluster should contain events within the time window
        for cluster in clusters:
            if len(cluster) > 1:
                timestamps = [event.timestamp for event in cluster]
                time_span = max(timestamps) - min(timestamps)
                assert time_span <= timedelta(minutes=5)
    
    def test_episodic_memory_capacity_management(self):
        """Test episodic memory respects capacity limits."""
        small_memory = EpisodicMemory(max_capacity=3)
        
        # Create more events than capacity
        events = [
            EpisodicEvent(
                event_id=f"event_{i}",
                timestamp=datetime.now() + timedelta(minutes=i),
                event_type="test",
                context={},
                embedding=np.random.rand(512).astype(np.float32)
            )
            for i in range(5)
        ]
        
        for event in events:
            small_memory.store_event(event)
        
        # Should not exceed capacity
        assert small_memory.current_size <= 3
        
        # Should keep most recent events
        recent_events = small_memory.retrieve_recent(count=3)
        assert len(recent_events) == 3
        assert recent_events[0].event_id == "event_4"  # Most recent


class TestSemanticMemory:
    """Test suite for Semantic Memory functionality."""
    
    @pytest.fixture
    def semantic_memory(self):
        """Create a SemanticMemory instance for testing."""
        return SemanticMemory()
    
    @pytest.fixture
    def sample_nodes(self):
        """Generate sample semantic nodes for testing."""
        return [
            SemanticNode(
                node_id=f"node_{i}",
                concept=f"concept_{i}",
                attributes={"type": "test", "value": i},
                embedding=np.random.rand(512).astype(np.float32)
            )
            for i in range(5)
        ]
    
    def test_semantic_memory_initialization(self, semantic_memory):
        """Test SemanticMemory initialization."""
        assert semantic_memory.node_count == 0
        assert semantic_memory.edge_count == 0
        assert semantic_memory.is_empty()
    
    def test_add_semantic_node(self, semantic_memory, sample_nodes):
        """Test adding semantic nodes."""
        node = sample_nodes[0]
        
        success = semantic_memory.add_node(node)
        
        assert success
        assert semantic_memory.node_count == 1
        assert semantic_memory.has_node(node.node_id)
    
    def test_add_duplicate_node(self, semantic_memory, sample_nodes):
        """Test adding duplicate node fails."""
        node = sample_nodes[0]
        
        # Add node first time
        success1 = semantic_memory.add_node(node)
        assert success1
        
        # Try to add same node again
        success2 = semantic_memory.add_node(node)
        assert not success2
        assert semantic_memory.node_count == 1
    
    def test_add_relationship(self, semantic_memory, sample_nodes):
        """Test adding relationships between nodes."""
        node1, node2 = sample_nodes[0], sample_nodes[1]
        
        # Add nodes first
        semantic_memory.add_node(node1)
        semantic_memory.add_node(node2)
        
        # Add relationship
        success = semantic_memory.add_relationship(
            node1.node_id, 
            node2.node_id, 
            "related_to", 
            weight=0.8
        )
        
        assert success
        assert semantic_memory.edge_count == 1
        assert semantic_memory.has_relationship(node1.node_id, node2.node_id)
    
    def test_add_relationship_nonexistent_nodes(self, semantic_memory):
        """Test adding relationship with non-existent nodes fails."""
        success = semantic_memory.add_relationship(
            "nonexistent1", 
            "nonexistent2", 
            "related_to"
        )
        
        assert not success
        assert semantic_memory.edge_count == 0
    
    def test_get_neighbors(self, semantic_memory, sample_nodes):
        """Test retrieving neighboring nodes."""
        # Add nodes
        for node in sample_nodes[:3]:
            semantic_memory.add_node(node)
        
        # Add relationships: node_0 -> node_1, node_0 -> node_2
        semantic_memory.add_relationship(sample_nodes[0].node_id, sample_nodes[1].node_id, "connects")
        semantic_memory.add_relationship(sample_nodes[0].node_id, sample_nodes[2].node_id, "links")
        
        # Get neighbors of node_0
        neighbors = semantic_memory.get_neighbors(sample_nodes[0].node_id)
        
        assert len(neighbors) == 2
        neighbor_ids = [n.node_id for n in neighbors]
        assert sample_nodes[1].node_id in neighbor_ids
        assert sample_nodes[2].node_id in neighbor_ids
    
    def test_shortest_path(self, semantic_memory, sample_nodes):
        """Test finding shortest path between nodes."""
        # Add nodes
        for node in sample_nodes[:4]:
            semantic_memory.add_node(node)
        
        # Create path: node_0 -> node_1 -> node_2 -> node_3
        semantic_memory.add_relationship(sample_nodes[0].node_id, sample_nodes[1].node_id, "next")
        semantic_memory.add_relationship(sample_nodes[1].node_id, sample_nodes[2].node_id, "next")
        semantic_memory.add_relationship(sample_nodes[2].node_id, sample_nodes[3].node_id, "next")
        
        # Find shortest path
        path = semantic_memory.find_shortest_path(sample_nodes[0].node_id, sample_nodes[3].node_id)
        
        assert len(path) == 4
        assert path[0] == sample_nodes[0].node_id
        assert path[-1] == sample_nodes[3].node_id
    
    def test_concept_similarity_search(self, semantic_memory, sample_nodes):
        """Test similarity search by concept embeddings."""
        # Add nodes
        for node in sample_nodes:
            semantic_memory.add_node(node)
        
        # Search using first node's embedding
        query_embedding = sample_nodes[0].embedding
        similar_concepts = semantic_memory.similarity_search(query_embedding, k=3)
        
        assert len(similar_concepts) == 3
        assert similar_concepts[0].node_id == sample_nodes[0].node_id  # Most similar to itself
    
    def test_subgraph_extraction(self, semantic_memory, sample_nodes):
        """Test extracting subgraphs around nodes."""
        # Add nodes and create connected component
        for node in sample_nodes:
            semantic_memory.add_node(node)
        
        # Create connections: 0-1-2, 3-4 (two components)
        semantic_memory.add_relationship(sample_nodes[0].node_id, sample_nodes[1].node_id, "connects")
        semantic_memory.add_relationship(sample_nodes[1].node_id, sample_nodes[2].node_id, "connects")
        semantic_memory.add_relationship(sample_nodes[3].node_id, sample_nodes[4].node_id, "connects")
        
        # Extract subgraph around node_1 with depth 1
        subgraph = semantic_memory.extract_subgraph(sample_nodes[1].node_id, depth=1)
        
        assert len(subgraph.nodes) == 3  # nodes 0, 1, 2
        assert len(subgraph.edges) == 2  # two connections
    
    def test_remove_node(self, semantic_memory, sample_nodes):
        """Test removing nodes and their relationships."""
        # Add nodes and relationships
        for node in sample_nodes[:3]:
            semantic_memory.add_node(node)
        
        semantic_memory.add_relationship(sample_nodes[0].node_id, sample_nodes[1].node_id, "connects")
        semantic_memory.add_relationship(sample_nodes[1].node_id, sample_nodes[2].node_id, "connects")
        
        initial_edges = semantic_memory.edge_count
        
        # Remove middle node
        success = semantic_memory.remove_node(sample_nodes[1].node_id)
        
        assert success
        assert semantic_memory.node_count == 2
        assert semantic_memory.edge_count < initial_edges  # Relationships should be removed
        assert not semantic_memory.has_node(sample_nodes[1].node_id)


class TestWorkingMemory:
    """Test suite for Working Memory functionality."""
    
    @pytest.fixture
    def working_memory(self):
        """Create a WorkingMemory instance for testing."""
        return WorkingMemory(capacity=5, attention_window=3)
    
    @pytest.fixture
    def sample_contexts(self):
        """Generate sample working memory contexts."""
        return [
            WorkingMemoryContext(
                context_id=f"ctx_{i}",
                content=f"context_content_{i}",
                attention_weight=0.5 + (i * 0.1),
                timestamp=datetime.now() + timedelta(seconds=i),
                embedding=np.random.rand(512).astype(np.float32)
            )
            for i in range(7)
        ]
    
    def test_working_memory_initialization(self, working_memory):
        """Test WorkingMemory initialization."""
        assert working_memory.capacity == 5
        assert working_memory.attention_window == 3
        assert working_memory.current_size == 0
        assert working_memory.is_empty()
    
    def test_add_context(self, working_memory, sample_contexts):
        """Test adding context to working memory."""
        context = sample_contexts[0]
        
        success = working_memory.add_context(context)
        
        assert success
        assert working_memory.current_size == 1
        assert working_memory.has_context(context.context_id)
    
    def test_capacity_management_with_attention(self, working_memory, sample_contexts):
        """Test working memory capacity management using attention weights."""
        # Add more contexts than capacity
        for context in sample_contexts:
            working_memory.add_context(context)
        
        # Should not exceed capacity
        assert working_memory.current_size <= working_memory.capacity
        
        # Should keep contexts with highest attention weights
        active_contexts = working_memory.get_active_contexts()
        attention_weights = [ctx.attention_weight for ctx in active_contexts]
        
        # Verify contexts are sorted by attention weight (descending)
        assert attention_weights == sorted(attention_weights, reverse=True)
    
    def test_attention_window(self, working_memory, sample_contexts):
        """Test attention window functionality."""
        # Add contexts
        for context in sample_contexts[:5]:
            working_memory.add_context(context)
        
        # Get attention window
        attention_contexts = working_memory.get_attention_window()
        
        assert len(attention_contexts) <= working_memory.attention_window
        
        # Should be highest attention weight contexts
        all_contexts = working_memory.get_active_contexts()
        expected_attention = sorted(all_contexts, key=lambda x: x.attention_weight, reverse=True)[:3]
        
        assert len(attention_contexts) == len(expected_attention)
    
    def test_update_attention_weights(self, working_memory, sample_contexts):
        """Test updating attention weights of contexts."""
        context = sample_contexts[0]
        working_memory.add_context(context)
        
        original_weight = context.attention_weight
        new_weight = 0.9
        
        success = working_memory.update_attention_weight(context.context_id, new_weight)
        
        assert success
        
        # Verify weight was updated
        retrieved_context = working_memory.get_context(context.context_id)
        assert retrieved_context.attention_weight == new_weight
        assert retrieved_context.attention_weight != original_weight
    
    def test_context_similarity_search(self, working_memory, sample_contexts):
        """Test similarity search within working memory contexts."""
        # Add contexts
        for context in sample_contexts[:4]:
            working_memory.add_context(context)
        
        # Search using first context's embedding
        query_embedding = sample_contexts[0].embedding
        similar_contexts = working_memory.similarity_search(query_embedding, k=2)
        
        assert len(similar_contexts) == 2
        assert similar_contexts[0].context_id == sample_contexts[0].context_id
    
    def test_temporal_decay(self, working_memory):
        """Test temporal decay of attention weights."""
        # Create context with timestamp in the past
        old_context = WorkingMemoryContext(
            context_id="old_ctx",
            content="old content",
            attention_weight=0.8,
            timestamp=datetime.now() - timedelta(minutes=10),
            embedding=np.random.rand(512).astype(np.float32)
        )
        
        working_memory.add_context(old_context)
        
        # Apply temporal decay
        working_memory.apply_temporal_decay(decay_rate=0.1)
        
        # Attention weight should have decreased
        retrieved_context = working_memory.get_context("old_ctx")
        assert retrieved_context.attention_weight < 0.8
    
    def test_clear_working_memory(self, working_memory, sample_contexts):
        """Test clearing working memory."""
        # Add contexts
        for context in sample_contexts[:3]:
            working_memory.add_context(context)
        
        assert working_memory.current_size == 3
        
        # Clear memory
        working_memory.clear()
        
        assert working_memory.current_size == 0
        assert working_memory.is_empty()


class TestMemoryConsolidator:
    """Test suite for Memory Consolidation functionality."""
    
    @pytest.fixture
    def memory_components(self):
        """Create memory components for consolidation testing."""
        return {
            'vector_memory': VectorMemoryManager(embedding_dim=512),
            'episodic_memory': EpisodicMemory(max_capacity=100),
            'semantic_memory': SemanticMemory(),
            'working_memory': WorkingMemory(capacity=10)
        }
    
    @pytest.fixture
    def consolidator(self, memory_components):
        """Create a MemoryConsolidator instance."""
        return MemoryConsolidator(
            vector_memory=memory_components['vector_memory'],
            episodic_memory=memory_components['episodic_memory'],
            semantic_memory=memory_components['semantic_memory'],
            working_memory=memory_components['working_memory']
        )
    
    def test_consolidator_initialization(self, consolidator):
        """Test MemoryConsolidator initialization."""
        assert consolidator.vector_memory is not None
        assert consolidator.episodic_memory is not None
        assert consolidator.semantic_memory is not None
        assert consolidator.working_memory is not None
    
    @pytest.mark.asyncio
    async def test_working_to_episodic_consolidation(self, consolidator):
        """Test consolidation from working memory to episodic memory."""
        # Add context to working memory
        context = WorkingMemoryContext(
            context_id="test_ctx",
            content="important decision made",
            attention_weight=0.9,
            timestamp=datetime.now(),
            embedding=np.random.rand(512).astype(np.float32)
        )
        
        consolidator.working_memory.add_context(context)
        
        # Trigger consolidation
        consolidated_count = await consolidator.consolidate_working_to_episodic(
            importance_threshold=0.8
        )
        
        assert consolidated_count > 0
        
        # Verify event was created in episodic memory
        recent_events = consolidator.episodic_memory.retrieve_recent(count=1)
        assert len(recent_events) == 1
        assert "important decision made" in recent_events[0].context.get("content", "")
    
    @pytest.mark.asyncio
    async def test_episodic_to_semantic_consolidation(self, consolidator):
        """Test consolidation from episodic memory to semantic memory."""
        # Add related events to episodic memory
        events = [
            EpisodicEvent(
                event_id=f"event_{i}",
                timestamp=datetime.now() + timedelta(minutes=i),
                event_type="learning_event",
                context={"concept": "machine_learning", "detail": f"detail_{i}"},
                embedding=np.random.rand(512).astype(np.float32)
            )
            for i in range(3)
        ]
        
        for event in events:
            consolidator.episodic_memory.store_event(event)
        
        # Trigger consolidation
        consolidated_count = await consolidator.consolidate_episodic_to_semantic(
            similarity_threshold=0.7,
            min_event_count=2
        )
        
        assert consolidated_count > 0
        
        # Verify semantic nodes were created
        assert consolidator.semantic_memory.node_count > 0
    
    @pytest.mark.asyncio
    async def test_importance_scoring(self, consolidator):
        """Test importance scoring for consolidation decisions."""
        # Create contexts with different characteristics
        contexts = [
            WorkingMemoryContext(
                context_id="low_importance",
                content="routine task",
                attention_weight=0.3,
                timestamp=datetime.now(),
                embedding=np.random.rand(512).astype(np.float32)
            ),
            WorkingMemoryContext(
                context_id="high_importance",
                content="critical error resolved",
                attention_weight=0.9,
                timestamp=datetime.now(),
                embedding=np.random.rand(512).astype(np.float32)
            )
        ]
        
        for context in contexts:
            consolidator.working_memory.add_context(context)
        
        # Calculate importance scores
        scores = await consolidator.calculate_importance_scores()
        
        assert len(scores) == 2
        assert scores["high_importance"] > scores["low_importance"]
    
    @pytest.mark.asyncio
    async def test_consolidation_scheduling(self, consolidator):
        """Test automatic consolidation scheduling."""
        # Mock the consolidation methods to track calls
        consolidator.consolidate_working_to_episodic = AsyncMock(return_value=1)
        consolidator.consolidate_episodic_to_semantic = AsyncMock(return_value=1)
        
        # Start scheduled consolidation
        await consolidator.start_scheduled_consolidation(interval_seconds=0.1)
        
        # Wait briefly for consolidation to run
        await asyncio.sleep(0.2)
        
        # Stop consolidation
        await consolidator.stop_scheduled_consolidation()
        
        # Verify consolidation methods were called
        assert consolidator.consolidate_working_to_episodic.called
        assert consolidator.consolidate_episodic_to_semantic.called
    
    def test_consolidation_metrics(self, consolidator):
        """Test consolidation metrics tracking."""
        # Get initial metrics
        metrics = consolidator.get_consolidation_metrics()
        
        assert "total_consolidations" in metrics
        assert "working_to_episodic_count" in metrics
        assert "episodic_to_semantic_count" in metrics
        assert "average_importance_score" in metrics
        
        # All should start at 0
        assert metrics["total_consolidations"] == 0


class TestHybridMemoryArchitecture:
    """Test suite for integrated Hybrid Memory Architecture."""
    
    @pytest.fixture
    def hybrid_memory(self):
        """Create a complete HybridMemoryArchitecture instance."""
        return HybridMemoryArchitecture(
            vector_config={
                'embedding_dim': 512,
                'similarity_metric': SimilarityMetric.COSINE,
                'max_capacity': 1000
            },
            episodic_config={
                'max_capacity': 100
            },
            working_config={
                'capacity': 10,
                'attention_window': 5
            }
        )
    
    def test_hybrid_memory_initialization(self, hybrid_memory):
        """Test HybridMemoryArchitecture initialization."""
        assert hybrid_memory.vector_memory is not None
        assert hybrid_memory.episodic_memory is not None
        assert hybrid_memory.semantic_memory is not None
        assert hybrid_memory.working_memory is not None
        assert hybrid_memory.consolidator is not None
    
    @pytest.mark.asyncio
    async def test_integrated_memory_workflow(self, hybrid_memory):
        """Test complete memory workflow integration."""
        # 1. Add information to working memory
        context = WorkingMemoryContext(
            context_id="workflow_test",
            content="learned new programming pattern",
            attention_weight=0.8,
            timestamp=datetime.now(),
            embedding=np.random.rand(512).astype(np.float32)
        )
        
        success = hybrid_memory.working_memory.add_context(context)
        assert success
        
        # 2. Trigger consolidation to episodic memory
        consolidated = await hybrid_memory.consolidator.consolidate_working_to_episodic(
            importance_threshold=0.7
        )
        assert consolidated > 0
        
        # 3. Verify episodic memory has the event
        recent_events = hybrid_memory.episodic_memory.retrieve_recent(count=1)
        assert len(recent_events) == 1
        
        # 4. Store in vector memory for similarity search
        item_id = hybrid_memory.vector_memory.store(
            context.embedding,
            {"source": "working_memory", "content": context.content}
        )
        assert item_id is not None
        
        # 5. Test cross-memory similarity search
        similar_items = hybrid_memory.vector_memory.similarity_search(
            context.embedding, k=1
        )
        assert len(similar_items) == 1
        assert similar_items[0].item_id == item_id
    
    def test_memory_statistics(self, hybrid_memory):
        """Test memory architecture statistics."""
        stats = hybrid_memory.get_memory_statistics()
        
        expected_keys = [
            'vector_memory_size',
            'episodic_memory_size', 
            'semantic_memory_nodes',
            'semantic_memory_edges',
            'working_memory_size',
            'total_memory_items'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], int)
            assert stats[key] >= 0
    
    @pytest.mark.asyncio
    async def test_memory_search_integration(self, hybrid_memory):
        """Test integrated search across all memory types."""
        # Add data to different memory types
        embedding = np.random.rand(512).astype(np.float32)
        
        # Vector memory
        vector_id = hybrid_memory.vector_memory.store(embedding, {"type": "vector_test"})
        
        # Episodic memory
        event = EpisodicEvent(
            event_id="search_test_event",
            timestamp=datetime.now(),
            event_type="test",
            context={"action": "search_test"},
            embedding=embedding
        )
        hybrid_memory.episodic_memory.store_event(event)
        
        # Working memory
        context = WorkingMemoryContext(
            context_id="search_test_context",
            content="search test content",
            attention_weight=0.7,
            timestamp=datetime.now(),
            embedding=embedding
        )
        hybrid_memory.working_memory.add_context(context)
        
        # Perform integrated search
        search_results = await hybrid_memory.integrated_search(
            query_embedding=embedding,
            k=2,
            search_types=['vector', 'episodic', 'working']
        )
        
        assert len(search_results) > 0
        assert any(result['type'] == 'vector' for result in search_results)
        assert any(result['type'] == 'episodic' for result in search_results)
        assert any(result['type'] == 'working' for result in search_results)
    
    def test_memory_cleanup(self, hybrid_memory):
        """Test memory cleanup and maintenance operations."""
        # Add some test data
        embedding = np.random.rand(512).astype(np.float32)
        
        hybrid_memory.vector_memory.store(embedding, {"test": True})
        
        context = WorkingMemoryContext(
            context_id="cleanup_test",
            content="test content",
            attention_weight=0.5,
            timestamp=datetime.now() - timedelta(hours=1),
            embedding=embedding
        )
        hybrid_memory.working_memory.add_context(context)
        
        # Perform cleanup
        cleanup_stats = hybrid_memory.cleanup_memory(
            max_age_hours=0.5,  # Remove items older than 30 minutes
            min_importance=0.6   # Remove items with low importance
        )
        
        assert "items_removed" in cleanup_stats
        assert "memory_freed" in cleanup_stats
        assert isinstance(cleanup_stats["items_removed"], int)