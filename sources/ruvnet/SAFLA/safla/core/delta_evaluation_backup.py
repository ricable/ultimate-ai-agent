"""
Delta Evaluation System for SAFLA

Implements the formal quantification of improvement using:
Δ_total = α₁ × Δ_performance + α₂ × Δ_efficiency + α₃ × Δ_stability + α₄ × Δ_capability

This module provides comprehensive evaluation of system improvements across multiple
dimensions with adaptive weighting based on context.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np


@dataclass
class DeltaMetrics:
    """
    Data structure for storing delta evaluation metrics.
    
    Attributes:
        performance: Change in performance metrics
        efficiency: Change in efficiency metrics  
        stability: Change in stability metrics
        capability: Change in capability metrics
        confidence: Confidence level of the evaluation (0.0 to 1.0)
        metadata: Additional context and information
    """
    performance: float
    efficiency: float
    stability: float
    capability: float
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if not (0.0 <= self.capability <= 1.0):
            raise ValueError("Capability must be between 0 and 1")
            
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DeltaResult:
    """
    Result structure for delta evaluation containing the computed total delta
    and component analysis.
    
    Attributes:
        total_delta: Weighted sum of all delta components
        component_deltas: Individual delta values
        weights: Weights used in calculation
        timestamp: Timestamp of the evaluation
        confidence: Overall confidence in the result
        improvement_detected: Whether improvement was detected
        metadata: Additional result information
    """
    total_delta: float
    component_deltas: DeltaMetrics
    weights: 'AdaptiveWeights'
    timestamp: datetime
    confidence: float
    improvement_detected: bool = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate result after initialization and set defaults."""
        if self.improvement_detected is None:
            self.improvement_detected = self.total_delta > 0.0
        if self.metadata is None:
            self.metadata = {}
    
    def is_improvement(self) -> bool:
        """Check if the result indicates improvement."""
        return self.improvement_detected


class PerformanceDelta:
    """
    Calculates performance delta based on reward improvements per token.
    
    Performance is measured as (current_reward - previous_reward) / tokens_used.
    """
    
    def __init__(self):
        """Initialize performance delta calculator."""
        self.history = []
    
    def calculate(self, current_reward: float, previous_reward: float, tokens_used: float) -> float:
        """
        Calculate performance delta: (current_reward - previous_reward) / tokens_used.
        
        Args:
            current_reward: Current reward value
            previous_reward: Previous reward value
            tokens_used: Number of tokens used
            
        Returns:
            Performance delta value
        """
        if tokens_used <= 0:
            raise ValueError("Tokens used must be greater than zero")
            
        return (current_reward - previous_reward) / tokens_used
    
    def add_data_point(self, reward: float, tokens: float, timestamp: int):
        """Add a historical data point."""
        self.history.append({
            'reward': reward,
            'tokens': tokens,
            'timestamp': timestamp
        })
    
    def calculate_with_history(self, reward: float, tokens: float, timestamp: int) -> float:
        """Calculate performance delta using historical data."""
        if not self.history:
            return 0.0
        
        # Use most recent historical point
        previous_reward = self.history[-1]['reward']
        delta = self.calculate(reward, previous_reward, tokens)
        
        # Add current data point to history
        self.add_data_point(reward, tokens, timestamp)
        
        return delta


class EfficiencyDelta:
    """
    Calculates efficiency delta based on throughput improvements per resource.
    
    Efficiency is measured as (current_throughput - previous_throughput) / resource_used.
    """
    
    def calculate(self, current_throughput: float, previous_throughput: float, resource_used: float) -> float:
        """
        Calculate efficiency delta: (current_throughput - previous_throughput) / resource_used.
        
        Args:
            current_throughput: Current throughput value
            previous_throughput: Previous throughput value
            resource_used: Amount of resource used
            
        Returns:
            Efficiency delta value
        """
        if resource_used <= 0:
            raise ValueError("Resource used must be greater than zero")
            
        return (current_throughput - previous_throughput) / resource_used
    
    def calculate_multi_resource(self, current_throughput: float, previous_throughput: float, 
                               resources: Dict[str, float]) -> float:
        """
        Calculate efficiency delta with multiple resource types.
        
        Args:
            current_throughput: Current throughput value
            previous_throughput: Previous throughput value
            resources: Dictionary of resource types and amounts
            
        Returns:
            Efficiency delta value
        """
        # Calculate total resource cost (weighted sum)
        total_resource = sum(resources.values())
        if total_resource <= 0:
            raise ValueError("Total resource usage must be greater than zero")
            
        return (current_throughput - previous_throughput) / total_resource


class StabilityDelta:
    """
    Calculates stability delta based on divergence score.
    
    Stability is measured as 1 - divergence_score, where lower divergence
    indicates higher stability.
    """
    def calculate(self, divergence_score: float) -> float:
        """
        Calculate stability delta: 1 - divergence_score.
        For degradation detection, high divergence (>0.5) results in negative delta.
        
        Args:
            divergence_score: Divergence score (0.0 to 1.0)
            
        Returns:
            Stability delta value (can be negative for high divergence)
        """
        if not (0.0 <= divergence_score <= 1.0):
            raise ValueError("Divergence score must be between 0 and 1")
        
        # Calculate stability as deviation from ideal (0.5 baseline)
        # Low divergence (< 0.5) = positive delta
        # High divergence (> 0.5) = negative delta
        return 0.5 - divergence_score
        return 1.0 - divergence_score
    
    def calculate_with_trend(self, current_score: float, historical_scores: List[float]) -> float:
        """
        Calculate stability delta with trend analysis.
        
        Args:
            current_score: Current divergence score
            historical_scores: List of historical divergence scores
            
        Returns:
            Stability delta with trend consideration
        """
        if not (0.0 <= current_score <= 1.0):
            raise ValueError("Divergence score must be between 0 and 1")
            
        base_stability = 1.0 - current_score
        
        if not historical_scores:
            return base_stability
            
        # Calculate trend (improvement over time)
        avg_historical = sum(historical_scores) / len(historical_scores)
        trend_factor = (avg_historical - current_score) * 0.1  # Weight trend at 10%
        
        return base_stability + trend_factor


class CapabilityDelta:
    """
    Calculates capability delta based on new capabilities acquired.
    
    Capability measures the expansion of system abilities relative to
    the total capability space.
    """
    
    def calculate(self, new_capabilities: int, total_capabilities: int) -> float:
        """
        Calculate capability delta: new_capabilities / total_capabilities.
        
        Args:
            new_capabilities: Number of new capabilities acquired
            total_capabilities: Total number of capabilities
            
        Returns:
            Capability delta value
        """
        if total_capabilities <= 0:
            raise ValueError("Total capabilities must be positive")
            
        if new_capabilities > total_capabilities:
            raise ValueError("New capabilities cannot exceed total capabilities")
            
        return new_capabilities / total_capabilities


class AdaptiveWeights:
    """
    Manages adaptive weighting system that adjusts based on context and priorities.
    
    Weights are dynamically adjusted based on:
    - Current system state
    - Performance history
    - Strategic priorities
    - Environmental context
    """
    
    def __init__(self, performance: float, efficiency: float, stability: float, capability: float):
        """Initialize with specific weights."""
        if abs(performance + efficiency + stability + capability - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
            
        self.performance = performance
        self.efficiency = efficiency
        self.stability = stability
        self.capability = capability
        
    def sum(self) -> float:
        """Return the sum of all weights."""
        return self.performance + self.efficiency + self.stability + self.capability
    
    def adjust_for_context(self, context: str) -> 'AdaptiveWeights':
        """
        Adjust weights based on context.
        
        Args:
            context: Context string indicating priority
            
        Returns:
            New AdaptiveWeights instance with adjusted weights
        """
        if context == "performance_critical":
            return AdaptiveWeights(0.5, 0.2, 0.2, 0.1)
        elif context == "stability_critical":
            return AdaptiveWeights(0.2, 0.2, 0.5, 0.1)
        elif context == "capability_critical":
            return AdaptiveWeights(0.2, 0.2, 0.2, 0.4)
        else:
            # Return copy of current weights
            return AdaptiveWeights(self.performance, self.efficiency, self.stability, self.capability)


class DeltaEvaluator:
    """
    Main delta evaluation engine that orchestrates the complete evaluation process.
    
    Integrates all delta calculation components and produces comprehensive
    evaluation results with confidence scoring.
    """
    
    def __init__(self, performance_calculator=None, efficiency_calculator=None,
                 stability_calculator=None, capability_calculator=None, weights=None):
        """Initialize the delta evaluator with component calculators."""
        self.performance_calculator = performance_calculator or PerformanceDelta()
        self.efficiency_calculator = efficiency_calculator or EfficiencyDelta()
        self.stability_calculator = stability_calculator or StabilityDelta()
        self.capability_calculator = capability_calculator or CapabilityDelta()
        self.weights = weights or AdaptiveWeights(0.25, 0.25, 0.25, 0.25)
        self.historical_data = []
        
    def add_historical_data(self, data: Dict[str, Any]):
        """Add historical data point."""
        self.historical_data.append(data)
        
    def evaluate_delta(self, performance_data=None, efficiency_data=None,
                      stability_data=None, capability_data=None, context=None, use_trend_analysis=False) -> DeltaResult:
        """
        Perform comprehensive delta evaluation.
        
        Args:
            performance_data: Performance metrics
            efficiency_data: Efficiency metrics
            stability_data: Stability metrics
            capability_data: Capability metrics
            context: Evaluation context
            
        Returns:
            Complete delta evaluation result
        """
        # Calculate individual deltas
        perf_delta = 0.0
        if performance_data:
            perf_delta = self.performance_calculator.calculate(
                performance_data['current_reward'],
                performance_data['previous_reward'],
                performance_data['tokens_used']
            )
        
        eff_delta = 0.0
        if efficiency_data:
            eff_delta = self.efficiency_calculator.calculate(
                efficiency_data['current_throughput'],
                efficiency_data['previous_throughput'],
                efficiency_data['resource_used']
            )
        
        stab_delta = 0.0
        if stability_data:
            stab_delta = self.stability_calculator.calculate(
                stability_data['divergence_score']
            )
        
        cap_delta = 0.0
        if capability_data:
            cap_delta = self.capability_calculator.calculate(
                capability_data['new_capabilities'],
                capability_data['total_capabilities']
            )
        
        # Adjust weights based on context
        if context:
            weights = self.weights.adjust_for_context(context)
        else:
            weights = self.weights
        
        # Calculate total weighted delta
        total_delta = (
            weights.performance * perf_delta +
            weights.efficiency * eff_delta +
            weights.stability * stab_delta +
            weights.capability * cap_delta
        )
        
        # Calculate confidence based on data availability
        data_count = sum(1 for data in [performance_data, efficiency_data, stability_data, capability_data] if data)
        confidence = data_count / 4.0  # Simple confidence based on data completeness
        
        # Create component deltas structure
        component_deltas = DeltaMetrics(
            performance=perf_delta,
            efficiency=eff_delta,
            stability=stab_delta,
            capability=cap_delta,
            confidence=confidence
        )
        
        # Determine if improvement was detected
        improvement_detected = total_delta > 0.01  # Threshold for significant improvement
        return DeltaResult(
            total_delta=total_delta,
            component_deltas=component_deltas,
            weights=weights,
            timestamp=datetime.now(),
            improvement_detected=improvement_detected,
            confidence=confidence
        )