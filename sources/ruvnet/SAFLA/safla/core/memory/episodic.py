"""
Episodic memory implementation for the SAFLA memory system.

This module provides temporal sequence storage and retrieval capabilities,
enabling the system to remember and learn from past experiences.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import bisect
import logging

from .base import MemoryItem, MemoryInterface

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMemory(MemoryItem):
    """
    Episodic memory item representing a specific experience or event.
    """
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    duration: Optional[timedelta] = None
    emotional_valence: float = 0.0  # -1 to 1, negative to positive
    
    def add_trigger(self, trigger: str) -> None:
        """Add a trigger that led to this episode."""
        if trigger not in self.triggers:
            self.triggers.append(trigger)
    
    def add_outcome(self, outcome: str) -> None:
        """Add an outcome from this episode."""
        if outcome not in self.outcomes:
            self.outcomes.append(outcome)
    
    def calculate_relevance(self, current_context: Dict[str, Any]) -> float:
        """Calculate relevance to current context."""
        relevance = 0.0
        
        # Context similarity
        context_overlap = 0
        for key, value in current_context.items():
            if key in self.context and self.context[key] == value:
                context_overlap += 1
        
        if self.context:
            relevance += (context_overlap / len(self.context)) * 0.5
        
        # Temporal relevance (recent episodes more relevant)
        time_diff = (datetime.now() - self.timestamp).total_seconds()
        temporal_factor = max(0, 1 - (time_diff / 86400))  # Decay over 24 hours
        relevance += temporal_factor * 0.3
        
        # Importance and access count
        relevance += self.calculate_current_importance() * 0.2
        
        return min(1.0, relevance)


@dataclass
class Sequence:
    """
    A temporal sequence of related episodes.
    """
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episodes: List[str] = field(default_factory=list)  # Episode IDs
    sequence_type: str = "general"
    pattern_strength: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    
    def add_episode(self, episode_id: str) -> None:
        """Add episode to sequence."""
        if episode_id not in self.episodes:
            self.episodes.append(episode_id)
    
    def update_access(self) -> None:
        """Update access time."""
        self.last_accessed = datetime.now()


class EpisodicMemoryManager(MemoryInterface):
    """
    Manager for episodic memory operations with temporal indexing.
    """
    
    def __init__(self, max_episodes: int = 10000, max_sequences: int = 1000):
        """
        Initialize episodic memory manager.
        
        Args:
            max_episodes: Maximum number of episodes to store
            max_sequences: Maximum number of sequences to maintain
        """
        self.max_episodes = max_episodes
        self.max_sequences = max_sequences
        self.memories: Dict[str, EpisodeMemory] = {}
        self.sequences: Dict[str, Sequence] = {}
        self._lock = threading.RLock()
        
        # Temporal indices
        self.temporal_index: List[Tuple[datetime, str]] = []  # (timestamp, episode_id)
        self.context_index: Dict[str, List[str]] = defaultdict(list)  # context_key -> episode_ids
        self.trigger_index: Dict[str, List[str]] = defaultdict(list)  # trigger -> episode_ids
        
        # Sequence detection
        self.sequence_buffer: deque = deque(maxlen=100)  # Recent episodes for pattern detection
        self.sequence_patterns: Dict[str, int] = defaultdict(int)  # Pattern -> frequency
    
    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None,
              context: Optional[Dict[str, Any]] = None,
              triggers: Optional[List[str]] = None,
              outcomes: Optional[List[str]] = None) -> str:
        """Store episodic memory item."""
        with self._lock:
            # Create episode
            episode = EpisodeMemory(
                content=content,
                metadata=metadata or {},
                context=context or {},
                triggers=triggers or [],
                outcomes=outcomes or []
            )
            
            # Store episode
            self.memories[episode.episode_id] = episode
            
            # Update temporal index
            bisect.insort(self.temporal_index, (episode.timestamp, episode.episode_id))
            
            # Update context index
            for key, value in episode.context.items():
                context_key = f"{key}:{value}"
                self.context_index[context_key].append(episode.episode_id)
            
            # Update trigger index
            for trigger in episode.triggers:
                self.trigger_index[trigger].append(episode.episode_id)
            
            # Add to sequence buffer for pattern detection
            self.sequence_buffer.append(episode.episode_id)
            
            # Detect sequences
            self._detect_sequences()
            
            # Cleanup if needed
            if len(self.memories) > self.max_episodes:
                self._cleanup_old_episodes()
            
            logger.debug(f"Stored episodic memory: {episode.episode_id}")
            return episode.episode_id
    
    def retrieve(self, query: Union[str, Dict[str, Any]], k: int = 5,
                time_window: Optional[timedelta] = None) -> List[EpisodeMemory]:
        """Retrieve relevant episodes."""
        with self._lock:
            if isinstance(query, str):
                # Search by content or trigger
                return self._search_by_text(query, k, time_window)
            elif isinstance(query, dict):
                # Search by context
                return self._search_by_context(query, k, time_window)
            else:
                return []
    
    def _search_by_text(self, query: str, k: int, 
                       time_window: Optional[timedelta]) -> List[EpisodeMemory]:
        """Search episodes by text query."""
        matches = []
        
        # Search in trigger index
        if query in self.trigger_index:
            for episode_id in self.trigger_index[query]:
                if episode_id in self.memories:
                    episode = self.memories[episode_id]
                    if self._within_time_window(episode, time_window):
                        matches.append(episode)
        
        # Search in content (simplified text matching)
        for episode in self.memories.values():
            if self._within_time_window(episode, time_window):
                if isinstance(episode.content, str) and query.lower() in episode.content.lower():
                    if episode not in matches:
                        matches.append(episode)
        
        # Sort by relevance and timestamp
        matches.sort(key=lambda e: (e.calculate_current_importance(), e.timestamp), reverse=True)
        
        # Update access for retrieved episodes
        for episode in matches[:k]:
            episode.update_access()
        
        return matches[:k]
    
    def _search_by_context(self, context: Dict[str, Any], k: int,
                          time_window: Optional[timedelta]) -> List[EpisodeMemory]:
        """Search episodes by context similarity."""
        relevance_scores = []
        
        for episode in self.memories.values():
            if self._within_time_window(episode, time_window):
                relevance = episode.calculate_relevance(context)
                if relevance > 0:
                    relevance_scores.append((relevance, episode))
        
        # Sort by relevance
        relevance_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Update access for retrieved episodes
        results = []
        for relevance, episode in relevance_scores[:k]:
            episode.update_access()
            results.append(episode)
        
        return results
    
    def _within_time_window(self, episode: EpisodeMemory, 
                           time_window: Optional[timedelta]) -> bool:
        """Check if episode is within time window."""
        if time_window is None:
            return True
        
        time_diff = datetime.now() - episode.timestamp
        return time_diff <= time_window
    
    def _detect_sequences(self) -> None:
        """Detect sequential patterns in recent episodes."""
        if len(self.sequence_buffer) < 3:
            return
        
        # Simple pattern detection: look for repeating context patterns
        recent_episodes = list(self.sequence_buffer)[-10:]  # Last 10 episodes
        
        for i in range(len(recent_episodes) - 2):
            pattern_episodes = recent_episodes[i:i+3]
            
            # Extract pattern signature
            pattern_sig = []
            for episode_id in pattern_episodes:
                if episode_id in self.memories:
                    episode = self.memories[episode_id]
                    # Use triggers as pattern signature
                    pattern_sig.extend(episode.triggers)
            
            if pattern_sig:
                pattern_key = "_".join(sorted(pattern_sig))
                self.sequence_patterns[pattern_key] += 1
                
                # Create sequence if pattern is strong enough
                if self.sequence_patterns[pattern_key] >= 3:
                    self._create_sequence(pattern_episodes, pattern_key)
    
    def _create_sequence(self, episode_ids: List[str], pattern_type: str) -> str:
        """Create a new sequence from episodes."""
        sequence = Sequence(
            episodes=episode_ids.copy(),
            sequence_type=pattern_type,
            pattern_strength=self.sequence_patterns[pattern_type] / 10.0
        )
        
        self.sequences[sequence.sequence_id] = sequence
        
        # Cleanup old sequences if needed
        if len(self.sequences) > self.max_sequences:
            self._cleanup_old_sequences()
        
        logger.debug(f"Created sequence: {sequence.sequence_id} with pattern {pattern_type}")
        return sequence.sequence_id
    
    def get_sequences(self, episode_id: str) -> List[Sequence]:
        """Get sequences containing a specific episode."""
        with self._lock:
            sequences = []
            for sequence in self.sequences.values():
                if episode_id in sequence.episodes:
                    sequence.update_access()
                    sequences.append(sequence)
            
            # Sort by pattern strength
            sequences.sort(key=lambda s: s.pattern_strength, reverse=True)
            return sequences
    
    def get_temporal_neighbors(self, episode_id: str, 
                              window: timedelta = timedelta(hours=1)) -> List[EpisodeMemory]:
        """Get episodes that occurred near the given episode in time."""
        with self._lock:
            if episode_id not in self.memories:
                return []
            
            target_episode = self.memories[episode_id]
            target_time = target_episode.timestamp
            
            neighbors = []
            for episode in self.memories.values():
                if episode.episode_id != episode_id:
                    time_diff = abs((episode.timestamp - target_time).total_seconds())
                    if time_diff <= window.total_seconds():
                        neighbors.append(episode)
            
            # Sort by temporal proximity
            neighbors.sort(key=lambda e: abs((e.timestamp - target_time).total_seconds()))
            return neighbors
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update episodic memory item."""
        with self._lock:
            if memory_id not in self.memories:
                return False
            
            episode = self.memories[memory_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(episode, key):
                    setattr(episode, key, value)
            
            # Update indices if context or triggers changed
            if 'context' in updates or 'triggers' in updates:
                self._rebuild_temporal_index()
            
            logger.debug(f"Updated episodic memory: {memory_id}")
            return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete episodic memory item."""
        with self._lock:
            if memory_id not in self.memories:
                return False
            
            # Remove from all indices
            episode = self.memories[memory_id]
            
            # Remove from temporal index
            self.temporal_index = [
                (ts, eid) for ts, eid in self.temporal_index 
                if eid != memory_id
            ]
            
            # Remove from context index
            for context_key, episode_ids in self.context_index.items():
                if memory_id in episode_ids:
                    episode_ids.remove(memory_id)
            
            # Remove from trigger index
            for trigger, episode_ids in self.trigger_index.items():
                if memory_id in episode_ids:
                    episode_ids.remove(memory_id)
            
            # Remove from sequences
            for sequence in self.sequences.values():
                if memory_id in sequence.episodes:
                    sequence.episodes.remove(memory_id)
            
            del self.memories[memory_id]
            
            logger.debug(f"Deleted episodic memory: {memory_id}")
            return True
    
    def clear(self) -> None:
        """Clear all episodic memories."""
        with self._lock:
            self.memories.clear()
            self.sequences.clear()
            self.temporal_index.clear()
            self.context_index.clear()
            self.trigger_index.clear()
            self.sequence_buffer.clear()
            self.sequence_patterns.clear()
    
    def _cleanup_old_episodes(self) -> None:
        """Remove old, less important episodes."""
        # Sort by importance and age
        episodes = list(self.memories.values())
        episodes.sort(key=lambda e: (e.calculate_current_importance(), e.timestamp))
        
        # Remove least important episodes
        to_remove = episodes[:len(episodes) - self.max_episodes + 1]
        for episode in to_remove:
            self.delete(episode.episode_id)
    
    def _cleanup_old_sequences(self) -> None:
        """Remove old, weak sequences."""
        sequences = list(self.sequences.values())
        sequences.sort(key=lambda s: (s.pattern_strength, s.created_at))
        
        # Remove weakest sequences
        to_remove = sequences[:len(sequences) - self.max_sequences + 1]
        for sequence in to_remove:
            del self.sequences[sequence.sequence_id]
    
    def _rebuild_temporal_index(self) -> None:
        """Rebuild temporal index from current memories."""
        self.temporal_index.clear()
        self.context_index.clear()
        self.trigger_index.clear()
        
        for episode in self.memories.values():
            # Temporal index
            bisect.insort(self.temporal_index, (episode.timestamp, episode.episode_id))
            
            # Context index
            for key, value in episode.context.items():
                context_key = f"{key}:{value}"
                self.context_index[context_key].append(episode.episode_id)
            
            # Trigger index
            for trigger in episode.triggers:
                self.trigger_index[trigger].append(episode.episode_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        with self._lock:
            total_episodes = len(self.memories)
            total_sequences = len(self.sequences)
            
            # Calculate time span
            if self.temporal_index:
                oldest = self.temporal_index[0][0]
                newest = self.temporal_index[-1][0]
                time_span = (newest - oldest).total_seconds() / 3600  # hours
            else:
                time_span = 0
            
            # Context diversity
            unique_contexts = len(self.context_index)
            
            # Access statistics
            access_counts = [e.access_count for e in self.memories.values()]
            avg_access = sum(access_counts) / len(access_counts) if access_counts else 0
            
            return {
                "total_episodes": total_episodes,
                "total_sequences": total_sequences,
                "time_span_hours": time_span,
                "unique_contexts": unique_contexts,
                "unique_triggers": len(self.trigger_index),
                "average_access_count": avg_access,
                "sequence_patterns": len(self.sequence_patterns),
                "max_episodes": self.max_episodes,
                "max_sequences": self.max_sequences
            }