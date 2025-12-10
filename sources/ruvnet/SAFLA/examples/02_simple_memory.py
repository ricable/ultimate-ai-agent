#!/usr/bin/env python3
"""
SAFLA Simple Memory Operations - Memory System Basics
====================================================

This example demonstrates SAFLA's powerful hybrid memory system with practical examples.
Learn how to store, retrieve, and manage different types of memories.

Learning Objectives:
- Understand the four memory types (Vector, Episodic, Semantic, Working)
- Learn memory storage and retrieval patterns
- Explore similarity search and memory consolidation
- Practice memory metadata and organization

Time to Complete: 10-15 minutes
Complexity: Beginner to Intermediate
"""

import asyncio
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from safla import HybridMemoryArchitecture, get_logger

logger = get_logger(__name__)


class MemoryDemo:
    """Comprehensive memory system demonstration."""
    
    def __init__(self):
        self.memory = None
        self.stored_memories = []
    
    async def initialize(self):
        """Initialize the memory system."""
        print("🧠 Initializing SAFLA Hybrid Memory System...")
        self.memory = HybridMemoryArchitecture()
        # Memory system is initialized automatically
        print("✅ Memory system ready!")
    
    async def demonstrate_vector_memory(self):
        """Demonstrate vector memory operations."""
        print("\n📊 Vector Memory Demonstration")
        print("-" * 40)
        
        # Example: Storing document embeddings
        documents = [
            "SAFLA is a sophisticated AI system with hybrid memory architecture",
            "The vector memory component stores high-dimensional embeddings for similarity search",
            "Machine learning models benefit from efficient vector storage and retrieval",
            "Natural language processing requires semantic understanding through embeddings",
            "SAFLA provides multiple similarity metrics for flexible vector search"
        ]
        
        print("Storing document embeddings...")
        for i, doc in enumerate(documents):
            # Simulate embeddings (in real use, you'd use actual embedding models)
            embedding = self._generate_realistic_embedding(doc, dimension=512)
            
            memory_id = await self.memory.store_vector_memory(
                content=doc,
                embedding=embedding,
                metadata={
                    "type": "document",
                    "index": i,
                    "timestamp": datetime.now().isoformat(),
                    "category": "technical_docs"
                }
            )
            self.stored_memories.append(memory_id)
            print(f"  📝 Stored: '{doc[:50]}...' (ID: {memory_id})")
        
        # Demonstrate similarity search
        print("\n🔍 Searching for similar documents...")
        query = "artificial intelligence memory systems"
        query_embedding = self._generate_realistic_embedding(query, dimension=512)
        
        similar_memories = await self.memory.search_similar_memories(
            query_embedding=query_embedding,
            top_k=3,
            similarity_threshold=0.6
        )
        
        print(f"Query: '{query}'")
        print("Similar documents found:")
        for mem in similar_memories:
            print(f"  📄 {mem.content[:60]}... (similarity: {mem.similarity:.3f})")
    
    async def demonstrate_episodic_memory(self):
        """Demonstrate episodic memory operations."""
        print("\n📅 Episodic Memory Demonstration")
        print("-" * 40)
        
        # Simulate user interaction episodes
        episodes = [
            {
                "content": "User logged in and started a new project",
                "context": {"user_id": "user_123", "action": "login", "project": "new_ai_model"},
                "outcome": "success",
                "duration": 2.5
            },
            {
                "content": "User trained a machine learning model with 1000 samples",
                "context": {"user_id": "user_123", "action": "train_model", "samples": 1000},
                "outcome": "completed",
                "duration": 120.0
            },
            {
                "content": "User saved model and exported results to CSV",
                "context": {"user_id": "user_123", "action": "export", "format": "csv"},
                "outcome": "success",
                "duration": 5.2
            }
        ]
        
        print("Storing user interaction episodes...")
        for i, episode in enumerate(episodes):
            episode_id = await self.memory.store_episodic_memory(
                content=episode["content"],
                context=episode["context"],
                outcome=episode["outcome"],
                metadata={
                    "session_id": "session_456",
                    "duration": episode["duration"],
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat()
                }
            )
            print(f"  🎬 Episode {i+1}: {episode['content']}")
        
        # Retrieve episodes by timeframe
        print("\n📍 Retrieving recent episodes...")
        start_time = datetime.now() - timedelta(hours=3)
        end_time = datetime.now()
        
        recent_episodes = await self.memory.get_episodic_memories(
            start_time=start_time,
            end_time=end_time,
            limit=5
        )
        
        print("Recent user activities:")
        for episode in recent_episodes:
            print(f"  ⏰ {episode.content} (outcome: {episode.outcome})")
    
    async def demonstrate_semantic_memory(self):
        """Demonstrate semantic memory operations."""
        print("\n🕸️ Semantic Memory Demonstration")
        print("-" * 40)
        
        # Build a knowledge graph about AI concepts
        concepts = [
            ("Machine Learning", "concept", {"domain": "AI", "complexity": "intermediate"}),
            ("Neural Networks", "concept", {"domain": "AI", "complexity": "advanced"}),
            ("Deep Learning", "concept", {"domain": "AI", "complexity": "advanced"}),
            ("SAFLA", "system", {"domain": "AI", "type": "architecture"}),
            ("Memory Architecture", "concept", {"domain": "CS", "complexity": "advanced"})
        ]
        
        print("Building knowledge graph...")
        concept_ids = {}
        
        for concept, node_type, properties in concepts:
            node_id = await self.memory.add_semantic_node(
                content=concept,
                node_type=node_type,
                properties=properties
            )
            concept_ids[concept] = node_id
            print(f"  🔗 Added concept: {concept} ({node_type})")
        
        # Add relationships between concepts
        relationships = [
            ("Deep Learning", "Neural Networks", "is_subset_of", 0.9),
            ("Neural Networks", "Machine Learning", "is_subset_of", 0.8),
            ("SAFLA", "Memory Architecture", "implements", 0.95),
            ("Machine Learning", "Memory Architecture", "uses", 0.7)
        ]
        
        print("\nAdding relationships...")
        for source, target, relation, weight in relationships:
            if source in concept_ids and target in concept_ids:
                await self.memory.add_semantic_edge(
                    source_id=concept_ids[source],
                    target_id=concept_ids[target],
                    relationship=relation,
                    weight=weight
                )
                print(f"  🔄 {source} --{relation}--> {target} (weight: {weight})")
        
        # Query related concepts
        print("\n🔍 Finding related concepts...")
        if "SAFLA" in concept_ids:
            related = await self.memory.get_related_concepts(
                node_id=concept_ids["SAFLA"],
                max_depth=2,
                min_weight=0.6
            )
            print("Concepts related to SAFLA:")
            for concept in related:
                print(f"  🌐 {concept.content} (relationship strength: {concept.weight:.2f})")
    
    async def demonstrate_working_memory(self):
        """Demonstrate working memory operations."""
        print("\n⚡ Working Memory Demonstration")
        print("-" * 40)
        
        # Simulate active context management
        print("Managing active context in working memory...")
        
        # Add current task context
        task_context = {
            "current_task": "memory_system_demo",
            "step": "working_memory_example",
            "user_focus": "learning_safla",
            "active_documents": ["demo_script", "safla_docs"],
            "recent_actions": ["vector_search", "episodic_storage"]
        }
        
        context_id = await self.memory.store_working_memory(
            content="Active demonstration context",
            context=task_context,
            attention_weight=0.9,
            ttl=300  # 5 minutes
        )
        print(f"  🎯 Stored active context (ID: {context_id})")
        
        # Update attention based on user actions
        await self.memory.update_attention(
            memory_id=context_id,
            attention_weight=0.95,
            reason="user_engaged_with_demo"
        )
        print("  📈 Updated attention weight based on user engagement")
        
        # Retrieve current context
        current_context = await self.memory.get_current_context(limit=3)
        print("\nCurrent working memory context:")
        for item in current_context:
            print(f"  🧩 {item.content} (attention: {item.attention_weight:.2f})")
    
    async def demonstrate_memory_consolidation(self):
        """Demonstrate memory consolidation process."""
        print("\n🔄 Memory Consolidation Demonstration")
        print("-" * 40)
        
        print("Triggering memory consolidation...")
        
        # In a real system, this would happen automatically
        # Here we demonstrate the concept
        consolidation_stats = await self.memory.consolidate_memories(
            max_items=10,
            similarity_threshold=0.8
        )
        
        print("Consolidation results:")
        print(f"  📊 Memories processed: {consolidation_stats.get('processed', 0)}")
        print(f"  🔗 Similar memories merged: {consolidation_stats.get('merged', 0)}")
        print(f"  🗂️ Memories archived: {consolidation_stats.get('archived', 0)}")
        print(f"  ⚡ Processing time: {consolidation_stats.get('duration', 0):.2f}s")
    
    async def demonstrate_memory_statistics(self):
        """Show memory system statistics."""
        print("\n📈 Memory System Statistics")
        print("-" * 40)
        
        stats = await self.memory.get_memory_statistics()
        
        print("Current memory usage:")
        print(f"  📊 Vector memories: {stats.get('vector_count', 0)}")
        print(f"  📅 Episodic memories: {stats.get('episodic_count', 0)}")
        print(f"  🕸️ Semantic nodes: {stats.get('semantic_nodes', 0)}")
        print(f"  ⚡ Working memory items: {stats.get('working_count', 0)}")
        print(f"  💾 Total memory usage: {stats.get('memory_usage_mb', 0):.1f} MB")
        print(f"  🔍 Total searches performed: {stats.get('search_count', 0)}")
        print(f"  ⏱️ Average search time: {stats.get('avg_search_time', 0):.2f}ms")
    
    def _generate_realistic_embedding(self, text: str, dimension: int = 512) -> List[float]:
        """Generate a realistic-looking embedding based on text content."""
        # This is a simple demonstration - in real use, you'd use actual embedding models
        random.seed(hash(text) % 2**32)  # Deterministic based on text
        
        # Generate base embedding
        embedding = [random.gauss(0, 0.1) for _ in range(dimension)]
        
        # Add some content-based patterns
        words = text.lower().split()
        for i, word in enumerate(words[:10]):  # Use first 10 words
            word_hash = hash(word) % dimension
            embedding[word_hash] += 0.3
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up memory system...")
        if self.memory:
            # Memory system cleanup is automatic
            pass
        print("✅ Cleanup complete!")


async def main():
    """Run the memory demonstration."""
    print("🧠 SAFLA Memory System Demonstration")
    print("=" * 60)
    
    demo = MemoryDemo()
    
    try:
        # Initialize
        await demo.initialize()
        
        # Run demonstrations
        await demo.demonstrate_vector_memory()
        await demo.demonstrate_episodic_memory()
        await demo.demonstrate_semantic_memory()
        await demo.demonstrate_working_memory()
        await demo.demonstrate_memory_consolidation()
        await demo.demonstrate_memory_statistics()
        
        print("\n🎉 Memory demonstration completed successfully!")
        print("\nKey Takeaways:")
        print("  • Vector memory: High-dimensional similarity search")
        print("  • Episodic memory: Time-based experience storage")
        print("  • Semantic memory: Knowledge graph relationships")
        print("  • Working memory: Active context management")
        print("  • Consolidation: Automatic memory optimization")
        
        print("\nNext Steps:")
        print("  1. Run 03_basic_safety.py to explore safety features")
        print("  2. Try 04_memory_operations.py for advanced memory patterns")
        
    except Exception as e:
        logger.exception("Demo failed")
        print(f"❌ Demo failed: {e}")
    
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())