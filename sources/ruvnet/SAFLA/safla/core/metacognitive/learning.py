"""
Adaptation Engine for continuous learning and self-modification capabilities.

This module implements machine learning integration, experience-based learning,
and controlled self-modification capabilities.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class AdaptationResult:
    """Represents the result of an adaptation operation."""
    adaptation_applied: bool
    modifications_made: List[Dict[str, Any]] = field(default_factory=list)
    expected_improvement: float = 0.0
    safety_validation_passed: bool = True
    confidence_score: float = 0.0


class AdaptationEngine:
    """
    Adaptation Engine for continuous learning and self-modification capabilities.
    
    This module implements machine learning integration, experience-based learning,
    and controlled self-modification capabilities.
    """
    
    def __init__(self):
        self.learning_algorithms: Dict[str, Any] = {}
        self.experience_database: List[Dict[str, Any]] = []
        self.pattern_recognizer = self._create_pattern_recognizer()
        self.self_modification_engine = self._create_self_modification_engine()
        self.learning_rate = 0.1
        self.modification_config: Dict[str, Any] = {
            'adaptation_threshold': 0.2,
            'modification_scope': ['strategy_parameters', 'goal_priorities'],
            'safety_constraints': ['no_core_system_changes']
        }
        self.ml_models: Dict[str, Any] = {}
        self.ml_config: Dict[str, Any] = {}
        self.learning_episodes: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def _create_pattern_recognizer(self):
        """Create a pattern recognizer for learning from experiences."""
        return type('PatternRecognizer', (), {
            'recognize': lambda experiences: self._recognize_patterns(experiences)
        })()
    
    def _create_self_modification_engine(self):
        """Create a self-modification engine for system adaptation."""
        return type('SelfModificationEngine', (), {
            'apply_modifications': lambda modifications: self._apply_modifications(modifications)
        })()
    
    def add_experience(self, experience: Dict[str, Any]):
        """Add an experience to the experience database."""
        with self._lock:
            experience_with_id = {
                'id': len(self.experience_database),
                'timestamp': time.time(),
                **experience
            }
            self.experience_database.append(experience_with_id)
            
            # Keep only recent experiences
            if len(self.experience_database) > 10000:
                self.experience_database = self.experience_database[-10000:]
    
    def learn_from_experiences(self):
        """Learn patterns from accumulated experiences."""
        learning_result = type('LearningResult', (), {})()
        
        with self._lock:
            if len(self.experience_database) < 3:
                learning_result.patterns_discovered = 0
                learning_result.learned_rules = []
                learning_result.confidence_scores = {}
                return learning_result
            
            # Analyze patterns in experiences
            context_action_outcomes = defaultdict(list)
            
            for experience in self.experience_database:
                context_key = self._serialize_context(experience.get('context', {}))
                action = experience.get('action', 'unknown')
                outcome = experience.get('outcome', {})
                success = outcome.get('success', False)
                
                context_action_outcomes[(context_key, action)].append({
                    'success': success,
                    'performance_gain': outcome.get('performance_gain', 0.0),
                    'time_taken': outcome.get('time_taken', 0.0)
                })
            
            # Generate learned rules
            learned_rules = []
            confidence_scores = {}
            
            for (context_key, action), outcomes in context_action_outcomes.items():
                if len(outcomes) >= 1:  # Allow single examples for initial learning
                    success_rate = sum(1 for o in outcomes if o['success']) / len(outcomes)
                    avg_performance_gain = np.mean([o['performance_gain'] for o in outcomes])
                    
                    # Lower threshold for pattern discovery
                    if success_rate > 0.5 or (len(outcomes) == 1 and outcomes[0]['success']):
                        confidence = success_rate if len(outcomes) > 1 else 0.6  # Lower confidence for single examples
                        rule = f"In context {context_key}, action {action} shows promise"
                        learned_rules.append(rule)
                        confidence_scores[rule] = confidence
                    elif success_rate <= 0.5:
                        # Also learn from failures
                        rule = f"In context {context_key}, action {action} may be ineffective"
                        learned_rules.append(rule)
                        confidence_scores[rule] = 1.0 - success_rate
            
            learning_result.patterns_discovered = len(learned_rules)
            learning_result.learned_rules = learned_rules
            learning_result.confidence_scores = confidence_scores
            
            return learning_result
    
    def _serialize_context(self, context: Dict[str, Any]) -> str:
        """Serialize context for pattern matching."""
        return str(sorted(context.items()))
    
    def _recognize_patterns(self, experiences):
        """Internal pattern recognition method."""
        return self.learn_from_experiences()
    
    def recommend_action(self, context: Dict[str, Any]):
        """Recommend an action based on learned patterns."""
        recommendation = type('ActionRecommendation', (), {})()
        
        with self._lock:
            context_key = self._serialize_context(context)
            
            # Find similar contexts in experience database
            similar_experiences = []
            for experience in self.experience_database:
                exp_context_key = self._serialize_context(experience.get('context', {}))
                if self._contexts_similar(context_key, exp_context_key):
                    similar_experiences.append(experience)
            
            if similar_experiences:
                # Find the most successful action
                action_success = defaultdict(list)
                for exp in similar_experiences:
                    action = exp.get('action', 'unknown')
                    success = exp.get('outcome', {}).get('success', False)
                    action_success[action].append(success)
                
                best_action = None
                best_success_rate = 0.0
                
                for action, successes in action_success.items():
                    success_rate = sum(successes) / len(successes)
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_action = action
                
                recommendation.action = best_action or 'default_action'
                recommendation.confidence = best_success_rate
                recommendation.based_on_experiences = len(similar_experiences)
            else:
                recommendation.action = 'default_action'
                recommendation.confidence = 0.5
                recommendation.based_on_experiences = 0
            
            return recommendation
    
    def _contexts_similar(self, context1: str, context2: str) -> bool:
        """Check if two contexts are similar."""
        # Simple similarity check - in practice, this could be more sophisticated
        return context1 == context2
    
    def configure_self_modification(self, modification_config: Dict[str, Any]):
        """Configure self-modification parameters."""
        with self._lock:
            self.modification_config = modification_config.copy()
    
    def adapt_system(self, performance_feedback: Dict[str, Any]) -> AdaptationResult:
        """
        Adapt the system based on performance feedback.
        
        Args:
            performance_feedback: Dictionary containing performance metrics and context
                - performance_gap: Float indicating the performance gap (required)
                - current_performance: Current system performance level
                - target_performance: Target performance level
                - context: Additional context for adaptation decisions
        
        Returns:
            AdaptationResult: Object containing adaptation status and details
        
        Raises:
            ValueError: If performance_feedback is invalid
        """
        if not isinstance(performance_feedback, dict):
            raise ValueError("performance_feedback must be a dictionary")
            
        adaptation_result = AdaptationResult(adaptation_applied=False)
        
        with self._lock:
            performance_gap = performance_feedback.get('performance_gap', 0.0)
            adaptation_threshold = self.modification_config.get('adaptation_threshold', 0.2)
            
            if performance_gap >= adaptation_threshold:
                try:
                    # Generate modifications
                    modifications = self._generate_modifications(performance_feedback)
                    
                    # Validate safety constraints
                    safety_validation = self._validate_safety_constraints(modifications)
                    
                    if safety_validation:
                        # Apply modifications
                        self._apply_modifications(modifications)
                        
                        adaptation_result.adaptation_applied = True
                        adaptation_result.modifications_made = modifications
                        adaptation_result.expected_improvement = min(performance_gap * 0.5, 0.2)
                        adaptation_result.safety_validation_passed = True
                        adaptation_result.confidence_score = 0.8
                    else:
                        adaptation_result.safety_validation_passed = False
                        logger.warning("Safety validation failed for system adaptations")
                except Exception as e:
                    logger.error(f"Error during system adaptation: {e}")
                    adaptation_result.adaptation_applied = False
                    adaptation_result.safety_validation_passed = False
            
            return adaptation_result
    
    def _generate_modifications(self, performance_feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system modifications based on performance feedback."""
        modifications = []
        
        current_performance = performance_feedback.get('current_performance', 0.5)
        target_performance = performance_feedback.get('target_performance', 0.8)
        
        if current_performance < target_performance:
            # Suggest strategy parameter adjustments
            modifications.append({
                'type': 'strategy_parameter_adjustment',
                'scope': 'strategy_parameters',
                'change': 'increase_learning_rate',
                'value': min(self.learning_rate * 1.1, 0.5)
            })
            
            # Suggest goal priority adjustments
            modifications.append({
                'type': 'goal_priority_adjustment',
                'scope': 'goal_priorities',
                'change': 'increase_performance_goal_priority',
                'value': 0.1
            })
        
        return modifications
    
    def _validate_safety_constraints(self, modifications: List[Dict[str, Any]]) -> bool:
        """Validate that modifications respect safety constraints."""
        safety_constraints = self.modification_config.get('safety_constraints', [])
        modification_scope = self.modification_config.get('modification_scope', [])
        
        for modification in modifications:
            # Check if modification scope is allowed
            if modification.get('scope') not in modification_scope:
                return False
            
            # Check safety constraints
            if 'no_core_system_changes' in safety_constraints:
                if modification.get('type') == 'core_system_change':
                    return False
        
        return True
    
    def _apply_modifications(self, modifications: List[Dict[str, Any]]):
        """Apply system modifications."""
        for modification in modifications:
            if modification.get('type') == 'strategy_parameter_adjustment':
                if modification.get('change') == 'increase_learning_rate':
                    self.learning_rate = modification.get('value', self.learning_rate)
    
    def configure_ml_models(self, ml_config: Dict[str, Any]):
        """Configure machine learning models for adaptation."""
        with self._lock:
            self.ml_config = ml_config.copy()
            
            # Initialize mock ML models
            self.ml_models = {
                'pattern_recognition_model': type('MockModel', (), {'trained': False})(),
                'performance_prediction_model': type('MockModel', (), {'trained': False})(),
                'anomaly_detection_model': type('MockModel', (), {'trained': False})()
            }
    
    def train_ml_models(self, training_data: List[Dict[str, Any]]):
        """Train machine learning models with provided data."""
        training_result = type('TrainingResult', (), {})()
        
        with self._lock:
            if len(training_data) < 10:
                training_result.training_successful = False
                return training_result
            
            # Mock training process
            for model_name, model in self.ml_models.items():
                model.trained = True
            
            # Calculate mock accuracies
            training_result.training_successful = True
            training_result.model_accuracies = {
                'pattern_recognition_model': 0.85,
                'performance_prediction_model': 0.78,
                'anomaly_detection_model': 0.82
            }
            training_result.feature_importances = {
                'feature_1': 0.4,
                'feature_2': 0.35,
                'feature_3': 0.25
            }
            
            return training_result
    
    def predict_performance(self, context_features: Dict[str, float]):
        """Predict performance using ML models."""
        prediction = type('PerformancePrediction', (), {})()
        
        with self._lock:
            if hasattr(self, 'ml_models') and self.ml_models.get('performance_prediction_model', {}).trained:
                # Mock prediction based on features
                feature_sum = sum(context_features.values())
                predicted_performance = min(1.0, max(0.0, feature_sum / len(context_features)))
                
                prediction.predicted_performance = predicted_performance
                prediction.confidence_interval = (
                    max(0.0, predicted_performance - 0.1),
                    min(1.0, predicted_performance + 0.1)
                )
            else:
                prediction.predicted_performance = 0.5
                prediction.confidence_interval = (0.4, 0.6)
            
            return prediction
    
    def set_learning_rate(self, learning_rate: float):
        """Set the learning rate for adaptation."""
        with self._lock:
            self.learning_rate = max(0.01, min(1.0, learning_rate))
    
    def get_learning_rate(self) -> float:
        """Get the current learning rate."""
        with self._lock:
            return self.learning_rate
    
    def process_learning_episode(self, episode: Dict[str, Any]):
        """Process a learning episode and adapt learning parameters."""
        with self._lock:
            # Store the episode
            self.learning_episodes.append(episode)
            if len(self.learning_episodes) > 1000:
                self.learning_episodes = self.learning_episodes[-1000:]
            
            success_rate = episode.get('success_rate', 0.5)
            context_complexity = episode.get('context_complexity', 0.5)
            
            # Adapt learning rate based on success and complexity
            if success_rate > 0.8 and context_complexity < 0.5:
                # Easy context with high success - can increase learning rate
                self.learning_rate = min(0.5, self.learning_rate * 1.05)
            elif success_rate < 0.4 or context_complexity > 0.8:
                # Difficult context or low success - decrease learning rate
                self.learning_rate = max(0.01, self.learning_rate * 0.95)
    
    def perform_meta_learning(self):
        """Perform meta-learning to optimize learning strategies."""
        meta_learning_result = type('MetaLearningResult', (), {})()
        
        with self._lock:
            # Analyze learning performance over time
            if self.learning_episodes:
                recent_episodes = self.learning_episodes[-10:]  # Last 10 episodes
                avg_success_rate = np.mean([ep.get('success_rate', 0.5) for ep in recent_episodes])
                
                if avg_success_rate > 0.7:
                    meta_learning_result.learning_strategy_updated = True
                    meta_learning_result.optimal_learning_rate = self.learning_rate * 1.1
                else:
                    meta_learning_result.learning_strategy_updated = True
                    meta_learning_result.optimal_learning_rate = self.learning_rate * 0.9
            else:
                meta_learning_result.learning_strategy_updated = False
                meta_learning_result.optimal_learning_rate = self.learning_rate
            
            meta_learning_result.context_specific_adaptations = {
                'high_complexity': {'learning_rate': 0.05, 'exploration_rate': 0.3},
                'low_complexity': {'learning_rate': 0.2, 'exploration_rate': 0.1}
            }
            
            return meta_learning_result
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning process."""
        with self._lock:
            if not self.experience_database:
                return {
                    'total_experiences': 0,
                    'learning_episodes': 0,
                    'current_learning_rate': self.learning_rate,
                    'patterns_discovered': 0
                }
            
            # Analyze experiences
            success_count = sum(
                1 for exp in self.experience_database
                if exp.get('outcome', {}).get('success', False)
            )
            
            learning_result = self.learn_from_experiences()
            
            return {
                'total_experiences': len(self.experience_database),
                'learning_episodes': len(self.learning_episodes),
                'current_learning_rate': self.learning_rate,
                'success_rate': success_count / len(self.experience_database),
                'patterns_discovered': learning_result.patterns_discovered,
                'adaptation_threshold': self.modification_config.get('adaptation_threshold', 0.2),
                'safety_constraints': self.modification_config.get('safety_constraints', [])
            }