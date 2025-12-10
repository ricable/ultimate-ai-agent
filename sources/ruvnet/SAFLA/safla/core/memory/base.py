"""
Base classes and types for the SAFLA memory system.

This module defines the fundamental types and interfaces used across
all memory subsystems.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Supported similarity metrics for vector operations."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"


class MemoryType(Enum):
    """Types of memory supported by the hybrid architecture."""
    VECTOR = "vector"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


@dataclass
class MemoryItem:
    """
    Base class for all memory items in the hybrid architecture.
    
    This provides common fields and functionality shared across different
    memory types.
    """
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance: float = 0.5
    decay_rate: float = 0.01
    
    def update_access(self) -> None:
        """Update access statistics for the memory item."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_current_importance(self) -> float:
        """
        Calculate current importance considering temporal decay.
        
        Returns:
            Current importance value between 0 and 1
        """
        if self.last_accessed is None:
            time_since_access = (datetime.now() - self.timestamp).total_seconds()
        else:
            time_since_access = (datetime.now() - self.last_accessed).total_seconds()
        
        # Apply exponential decay
        decay_factor = np.exp(-self.decay_rate * time_since_access / 3600)  # hourly decay
        current_importance = self.importance * decay_factor
        
        # Boost based on access count
        access_boost = min(0.2, self.access_count * 0.01)
        current_importance = min(1.0, current_importance + access_boost)
        
        return current_importance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory item to dictionary representation."""
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'importance': self.importance,
            'decay_rate': self.decay_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create memory item from dictionary representation."""
        item = cls(
            memory_id=data.get('memory_id', str(uuid.uuid4())),
            content=data.get('content'),
            metadata=data.get('metadata', {}),
            access_count=data.get('access_count', 0),
            importance=data.get('importance', 0.5),
            decay_rate=data.get('decay_rate', 0.01)
        )
        
        # Handle datetime fields
        if 'timestamp' in data:
            item.timestamp = datetime.fromisoformat(data['timestamp'])
        if data.get('last_accessed'):
            item.last_accessed = datetime.fromisoformat(data['last_accessed'])
        
        return item


class MemoryInterface:
    """Abstract interface for memory subsystems."""
    
    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in memory."""
        raise NotImplementedError
    
    def retrieve(self, query: Any, k: int = 5) -> List[Any]:
        """Retrieve relevant memories."""
        raise NotImplementedError
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing memory."""
        raise NotImplementedError
    
    def delete(self, memory_id: str) -> bool:
        """Delete memory."""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Clear all memories."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        raise NotImplementedError