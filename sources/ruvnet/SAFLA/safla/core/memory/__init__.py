"""
SAFLA Hybrid Memory Architecture - Modular Implementation.

This package provides a modular implementation of the hybrid memory system,
splitting the original monolithic module into focused components.
"""

from .base import MemoryItem, SimilarityMetric, MemoryInterface
from .vector import VectorMemory, VectorMemoryManager
from .episodic import EpisodeMemory, EpisodicMemoryManager, Sequence
from .semantic import SemanticNode, SemanticMemoryManager, SemanticEdge
from .working import WorkingMemoryItem, WorkingMemoryManager, WorkingMemoryCluster
from .hybrid import HybridMemorySystem, HybridMemoryResult, MemoryConfiguration

__all__ = [
    # Base types
    "MemoryItem",
    "SimilarityMetric",
    "MemoryInterface",
    
    # Vector memory
    "VectorMemory",
    "VectorMemoryManager",
    
    # Episodic memory
    "EpisodeMemory",
    "EpisodicMemoryManager",
    "Sequence",
    
    # Semantic memory
    "SemanticNode",
    "SemanticMemoryManager",
    "SemanticEdge",
    
    # Working memory
    "WorkingMemoryItem",
    "WorkingMemoryManager",
    "WorkingMemoryCluster",
    
    # Hybrid system
    "HybridMemorySystem",
    "HybridMemoryResult",
    "MemoryConfiguration"
]