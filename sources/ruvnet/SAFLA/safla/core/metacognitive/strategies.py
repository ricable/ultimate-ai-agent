"""
Strategy Selection system for context-aware strategy selection and optimization.

This module manages a repository of strategies and selects the most appropriate
strategy based on current context and historical performance.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Represents a strategy in the strategy selection system."""
    id: str
    name: str = ""
    description: str = ""
    applicable_contexts: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    implementation_complexity: float = 0.5
    success_rate: float = 0.0
    last_used: Optional[float] = None
    learned_context_preferences: Dict[str, float] = field(default_factory=dict)


class StrategySelector:
    """
    Strategy Selection system for context-aware strategy selection and optimization.
    
    This module manages a repository of strategies and selects the most appropriate
    strategy based on current context and historical performance.
    """
    
    def __init__(self):
        self.strategy_repository: Dict[str, Strategy] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.context_analyzer = self._create_context_analyzer()
        self._lock = threading.Lock()
    
    def _create_context_analyzer(self):
        """Create a context analyzer for strategy selection."""
        return type('ContextAnalyzer', (), {
            'analyze': lambda context: self._analyze_context(context)
        })()
    
    def add_strategy(self, strategy: Strategy):
        """Add a strategy to the repository."""
        with self._lock:
            self.strategy_repository[strategy.id] = strategy
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Get a strategy from the repository."""
        with self._lock:
            return self.strategy_repository.get(strategy_id)
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the repository."""
        with self._lock:
            if strategy_id in self.strategy_repository:
                del self.strategy_repository[strategy_id]
                return True
            return False
    
    def list_strategies(self) -> List[Strategy]:
        """List all strategies in the repository."""
        with self._lock:
            return list(self.strategy_repository.values())
    
    def select_strategy(self, context: Dict[str, Any]) -> Optional[Strategy]:
        """Select the most appropriate strategy for the given context."""
        with self._lock:
            best_strategy = None
            best_score = -1.0
            
            for strategy in self.strategy_repository.values():
                score = self._calculate_strategy_score(strategy, context)
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
            
            if best_strategy:
                best_strategy.last_used = time.time()
            
            return best_strategy
    
    def _calculate_strategy_score(self, strategy: Strategy, context: Dict[str, Any]) -> float:
        """Calculate the suitability score for a strategy given the context."""
        score = 0.0
        
        # Context matching score
        current_situation = context.get('current_situation', '')
        if current_situation in strategy.applicable_contexts:
            score += 0.5
        
        # Resource availability score
        available_resources = context.get('available_resources', {})
        cpu_available = available_resources.get('cpu', 1.0)
        memory_available = available_resources.get('memory', 1.0)
        
        cpu_required = strategy.resource_requirements.get('cpu', 0.0)
        memory_required = strategy.resource_requirements.get('memory', 0.0)
        
        if cpu_required <= cpu_available and memory_required <= memory_available:
            score += 0.3
        
        # Performance score
        if context.get('time_pressure', 0.0) > 0.7:
            # Prefer fast strategies under time pressure
            speed = strategy.performance_metrics.get('speed', 0.5)
            score += speed * 0.2
        else:
            # Prefer accurate strategies when time allows
            accuracy = strategy.performance_metrics.get('accuracy', 0.5)
            score += accuracy * 0.2
        
        return score
    
    def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the context for strategy selection."""
        analysis = {
            'time_critical': context.get('time_pressure', 0.0) > 0.7,
            'resource_constrained': (
                context.get('available_resources', {}).get('cpu', 1.0) < 0.5 or
                context.get('available_resources', {}).get('memory', 1.0) < 0.5
            ),
            'accuracy_critical': context.get('accuracy_requirements', 0.5) > 0.8
        }
        return analysis
    
    def record_performance(self, performance_record: Dict[str, Any]):
        """Record the performance of a strategy execution."""
        with self._lock:
            self.performance_history.append({
                **performance_record,
                'recorded_at': time.time()
            })
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
    
    def optimize_strategy(self, strategy_id: str):
        """Optimize a strategy based on performance history."""
        optimization_result = type('StrategyOptimizationResult', (), {})()
        
        with self._lock:
            if strategy_id not in self.strategy_repository:
                optimization_result.performance_improved = False
                return optimization_result
            
            strategy = self.strategy_repository[strategy_id]
            
            # Find performance records for this strategy
            strategy_records = [
                record for record in self.performance_history
                if record.get('strategy_id') == strategy_id
            ]
            
            if len(strategy_records) >= 3:
                # Calculate average performance improvement
                scores = [record.get('score', 0.0) for record in strategy_records]
                avg_score = np.mean(scores)
                
                # Update strategy performance metrics
                original_score = strategy.performance_metrics.get('baseline_score', 0.5)
                if avg_score > original_score:
                    strategy.performance_metrics['baseline_score'] = avg_score
                    optimization_result.performance_improved = True
                    optimization_result.new_performance_metrics = strategy.performance_metrics.copy()
                    optimization_result.optimization_details = {
                        'original_score': original_score,
                        'new_score': avg_score,
                        'improvement': avg_score - original_score,
                        'records_analyzed': len(strategy_records)
                    }
                else:
                    optimization_result.performance_improved = False
            else:
                optimization_result.performance_improved = False
            
            return optimization_result
    
    def learn_from_experience(self, strategy_id: str, learning_data: List[Dict[str, Any]]):
        """Learn from experience to improve strategy selection."""
        learning_result = type('StrategyLearningResult', (), {})()
        
        with self._lock:
            if strategy_id not in self.strategy_repository:
                learning_result.strategy_updated = False
                return learning_result
            
            strategy = self.strategy_repository[strategy_id]
            
            # Analyze patterns in learning data
            context_performance = defaultdict(list)
            for data_point in learning_data:
                context_key = str(sorted(data_point['context'].items()))
                context_performance[context_key].append(data_point['outcome'])
            
            # Identify learned patterns
            learned_patterns = []
            context_preferences = {}
            
            for context_key, outcomes in context_performance.items():
                avg_outcome = np.mean(outcomes)
                context_preferences[context_key] = avg_outcome
                
                if avg_outcome > 0.7:
                    learned_patterns.append(f"high_performance_in_{context_key}")
                elif avg_outcome < 0.3:
                    learned_patterns.append(f"low_performance_in_{context_key}")
            
            # Store learned preferences on the strategy object
            strategy.learned_context_preferences = context_preferences
            
            learning_result.strategy_updated = True
            learning_result.learned_patterns = learned_patterns
            learning_result.context_preferences = context_preferences
            
            return learning_result
    
    def get_selection_confidence(self, strategy_id: str, context: Dict[str, Any]) -> float:
        """Get confidence score for selecting a strategy in a given context."""
        with self._lock:
            if strategy_id not in self.strategy_repository:
                return 0.0
            
            strategy = self.strategy_repository[strategy_id]
            
            # Base confidence on context matching
            confidence = 0.0
            current_situation = context.get('current_situation', '')
            if current_situation in strategy.applicable_contexts:
                confidence += 0.5
            
            # Add confidence based on learned context preferences
            if hasattr(strategy, 'learned_context_preferences'):
                # Base boost for having learned patterns
                confidence += 0.2
                
                # Try exact match first
                context_key = str(sorted(context.items()))
                if context_key in strategy.learned_context_preferences:
                    learned_confidence = strategy.learned_context_preferences[context_key]
                    confidence += learned_confidence * 0.5
                else:
                    # Find similar contexts for partial matching
                    best_similarity = 0.0
                    best_confidence = 0.0
                    
                    for learned_key, learned_conf in strategy.learned_context_preferences.items():
                        # Parse the learned context
                        try:
                            learned_context = eval(learned_key)  # Convert string back to list of tuples
                            learned_dict = dict(learned_context)
                            
                            # Calculate similarity based on matching keys
                            similarity = 0.0
                            matching_keys = 0
                            for key, value in context.items():
                                if key in learned_dict:
                                    matching_keys += 1
                                    if learned_dict[key] == value:
                                        similarity += 1.0
                                    elif isinstance(value, (int, float)) and isinstance(learned_dict[key], (int, float)):
                                        # For numeric values, use proximity
                                        diff = abs(value - learned_dict[key])
                                        if diff < 0.5:  # Close enough
                                            similarity += max(0.0, 1.0 - diff * 2)
                            
                            if matching_keys > 0:
                                similarity /= matching_keys
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_confidence = learned_conf
                        except:
                            continue
                    
                    if best_similarity > 0.3:  # Lower threshold for similarity
                        confidence += best_confidence * 0.7 * best_similarity  # Higher weight for learned patterns
            
            # Add confidence based on historical performance
            relevant_records = [
                record for record in self.performance_history
                if record.get('strategy_id') == strategy_id
            ]
            
            if relevant_records:
                avg_score = np.mean([record.get('score', 0.0) for record in relevant_records])
                confidence += avg_score * 0.3  # Reduced weight to make room for learned patterns
            
            return min(1.0, confidence)
    
    def get_strategy_statistics(self, strategy_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific strategy."""
        with self._lock:
            if strategy_id not in self.strategy_repository:
                return {}
            
            strategy = self.strategy_repository[strategy_id]
            
            # Find all performance records for this strategy
            strategy_records = [
                record for record in self.performance_history
                if record.get('strategy_id') == strategy_id
            ]
            
            if not strategy_records:
                return {
                    'strategy_id': strategy_id,
                    'name': strategy.name,
                    'times_used': 0,
                    'avg_score': 0.0,
                    'success_rate': strategy.success_rate,
                    'last_used': strategy.last_used
                }
            
            scores = [record.get('score', 0.0) for record in strategy_records]
            success_count = sum(1 for record in strategy_records if record.get('success', False))
            
            return {
                'strategy_id': strategy_id,
                'name': strategy.name,
                'times_used': len(strategy_records),
                'avg_score': np.mean(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'std_score': np.std(scores),
                'success_rate': success_count / len(strategy_records),
                'last_used': strategy.last_used,
                'performance_trend': self._calculate_performance_trend(scores)
            }
    
    def _calculate_performance_trend(self, scores: List[float]) -> str:
        """Calculate the performance trend from a list of scores."""
        if len(scores) < 3:
            return 'insufficient_data'
        
        # Use linear regression to find trend
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all strategies."""
        with self._lock:
            stats = {}
            for strategy_id in self.strategy_repository:
                stats[strategy_id] = self.get_strategy_statistics(strategy_id)
            return stats
    
    def recommend_strategies(self, context: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
        """Recommend top N strategies for a given context."""
        with self._lock:
            recommendations = []
            
            for strategy in self.strategy_repository.values():
                score = self._calculate_strategy_score(strategy, context)
                confidence = self.get_selection_confidence(strategy.id, context)
                
                recommendations.append({
                    'strategy_id': strategy.id,
                    'name': strategy.name,
                    'score': score,
                    'confidence': confidence,
                    'combined_score': score * 0.7 + confidence * 0.3
                })
            
            # Sort by combined score
            recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return recommendations[:top_n]
    
    def update_strategy_success_rate(self, strategy_id: str, success: bool):
        """Update the success rate of a strategy based on recent execution."""
        with self._lock:
            if strategy_id not in self.strategy_repository:
                return
            
            strategy = self.strategy_repository[strategy_id]
            
            # Simple exponential moving average
            alpha = 0.1  # Learning rate
            strategy.success_rate = (
                alpha * (1.0 if success else 0.0) + 
                (1 - alpha) * strategy.success_rate
            )