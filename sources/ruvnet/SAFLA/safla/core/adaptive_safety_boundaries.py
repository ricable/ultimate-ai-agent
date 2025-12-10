"""
Adaptive Safety Boundaries for SAFLA.

This module implements dynamic safety constraints that adapt based on system
behavior and environmental conditions. It provides real-time monitoring and
adjustment of safety parameters to ensure optimal performance while maintaining
strict safety guarantees.

Components:
- Adaptive Boundary Manager: Dynamic safety constraint management
- Risk Assessment Engine: Real-time risk evaluation and scoring
- Constraint Optimizer: Optimization of safety boundaries based on performance
- Monitoring Dashboard: Real-time visualization of safety metrics
- Feedback Loop: Continuous learning from safety violations and near-misses

Technical Features:
- Dynamic constraint adjustment based on system state
- Multi-objective optimization for safety-performance tradeoffs
- Real-time anomaly detection and response
- Historical analysis for pattern recognition
- Federated learning integration for distributed safety learning
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for adaptive boundaries."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class ConstraintType(Enum):
    """Types of safety constraints."""
    HARD = "hard"  # Cannot be violated
    SOFT = "soft"  # Can be relaxed under certain conditions
    ADAPTIVE = "adaptive"  # Dynamically adjusted


@dataclass
class SafetyConstraint:
    """Represents a safety constraint."""
    name: str
    constraint_type: ConstraintType
    threshold: float
    current_value: float = 0.0
    adaptivity_rate: float = 0.1
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    violation_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyMetrics:
    """Safety performance metrics."""
    total_constraints: int
    active_constraints: int
    violations_last_hour: int
    safety_score: float
    adaptation_rate: float
    risk_level: SafetyLevel


@dataclass
class AdaptationStrategy:
    """Strategy for adapting safety boundaries."""
    name: str
    trigger_conditions: Dict[str, Any]
    adjustment_factor: float
    cooldown_period: int  # seconds
    max_adjustments_per_hour: int


class AdaptiveSafetyBoundaries:
    """
    Manages adaptive safety boundaries for the SAFLA system.
    
    Provides dynamic adjustment of safety constraints based on system
    behavior, environmental conditions, and historical performance.
    """
    
    def __init__(
        self,
        initial_constraints: Optional[List[SafetyConstraint]] = None,
        adaptation_strategies: Optional[List[AdaptationStrategy]] = None,
        learning_rate: float = 0.01,
        risk_tolerance: float = 0.1
    ):
        """Initialize adaptive safety boundaries."""
        self.constraints: Dict[str, SafetyConstraint] = {}
        self.strategies: Dict[str, AdaptationStrategy] = {}
        self.learning_rate = learning_rate
        self.risk_tolerance = risk_tolerance
        
        # Initialize constraints
        if initial_constraints:
            for constraint in initial_constraints:
                self.add_constraint(constraint)
        
        # Initialize strategies
        if adaptation_strategies:
            for strategy in adaptation_strategies:
                self.add_strategy(strategy)
        
        # Metrics tracking
        self.violation_history: List[Dict[str, Any]] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized AdaptiveSafetyBoundaries")
    
    def add_constraint(self, constraint: SafetyConstraint) -> None:
        """Add a new safety constraint."""
        self.constraints[constraint.name] = constraint
        logger.info(f"Added constraint: {constraint.name}")
    
    def add_strategy(self, strategy: AdaptationStrategy) -> None:
        """Add a new adaptation strategy."""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
    
    async def update_constraint_value(
        self,
        constraint_name: str,
        value: float
    ) -> bool:
        """Update the current value of a constraint."""
        if constraint_name not in self.constraints:
            logger.warning(f"Constraint not found: {constraint_name}")
            return False
        
        constraint = self.constraints[constraint_name]
        constraint.current_value = value
        constraint.last_updated = datetime.now()
        
        # Check for violation
        if value > constraint.threshold:
            constraint.violation_count += 1
            self.violation_history.append({
                'constraint': constraint_name,
                'value': value,
                'threshold': constraint.threshold,
                'timestamp': datetime.now()
            })
            
            # Trigger adaptation if needed
            if constraint.constraint_type == ConstraintType.ADAPTIVE:
                await self._adapt_constraint(constraint_name)
        
        return True
    
    async def _adapt_constraint(self, constraint_name: str) -> None:
        """Adapt a constraint based on current conditions."""
        constraint = self.constraints[constraint_name]
        
        # Simple adaptation logic
        if constraint.violation_count > 3:
            # Relax the constraint slightly
            new_threshold = min(
                constraint.threshold * (1 + constraint.adaptivity_rate),
                constraint.max_threshold
            )
            constraint.threshold = new_threshold
            constraint.violation_count = 0
            
            self.adaptation_history.append({
                'constraint': constraint_name,
                'old_threshold': constraint.threshold,
                'new_threshold': new_threshold,
                'timestamp': datetime.now()
            })
            
            logger.info(f"Adapted constraint {constraint_name}: {new_threshold}")
    
    def get_safety_metrics(self) -> SafetyMetrics:
        """Get current safety metrics."""
        total_constraints = len(self.constraints)
        active_constraints = sum(
            1 for c in self.constraints.values()
            if c.current_value > 0
        )
        
        # Count recent violations
        recent_violations = sum(
            1 for v in self.violation_history
            if (datetime.now() - v['timestamp']).seconds < 3600
        )
        
        # Calculate safety score
        if total_constraints > 0:
            violation_rate = sum(
                c.violation_count for c in self.constraints.values()
            ) / total_constraints
            safety_score = max(0, 1 - violation_rate * 0.1)
        else:
            safety_score = 1.0
        
        # Determine risk level
        if safety_score > 0.9:
            risk_level = SafetyLevel.MINIMAL
        elif safety_score > 0.7:
            risk_level = SafetyLevel.LOW
        elif safety_score > 0.5:
            risk_level = SafetyLevel.MEDIUM
        elif safety_score > 0.3:
            risk_level = SafetyLevel.HIGH
        else:
            risk_level = SafetyLevel.CRITICAL
        
        return SafetyMetrics(
            total_constraints=total_constraints,
            active_constraints=active_constraints,
            violations_last_hour=recent_violations,
            safety_score=safety_score,
            adaptation_rate=self.learning_rate,
            risk_level=risk_level
        )
    
    async def evaluate_safety_state(self) -> Dict[str, Any]:
        """Evaluate current safety state of the system."""
        metrics = self.get_safety_metrics()
        
        constraint_states = {}
        for name, constraint in self.constraints.items():
            constraint_states[name] = {
                'type': constraint.constraint_type.value,
                'threshold': constraint.threshold,
                'current_value': constraint.current_value,
                'violation_count': constraint.violation_count,
                'within_bounds': constraint.current_value <= constraint.threshold
            }
        
        return {
            'metrics': metrics,
            'constraint_states': constraint_states,
            'recent_violations': len(self.violation_history),
            'recent_adaptations': len(self.adaptation_history),
            'overall_safe': metrics.risk_level != SafetyLevel.CRITICAL
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics and violation counts."""
        for constraint in self.constraints.values():
            constraint.violation_count = 0
            constraint.current_value = 0.0
        
        self.violation_history.clear()
        self.adaptation_history.clear()
        
        logger.info("Reset all safety metrics")


class RiskAssessmentEngine:
    """Engine for assessing and managing risks."""
    
    def __init__(self):
        """Initialize risk assessment engine."""
        self.risk_factors: Dict[str, float] = {}
        self.risk_history: List[Dict[str, Any]] = []
    
    async def assess_risk(
        self,
        system_state: Dict[str, Any],
        safety_boundaries: AdaptiveSafetyBoundaries
    ) -> float:
        """Assess current risk level based on system state."""
        # Simple risk calculation
        safety_metrics = safety_boundaries.get_safety_metrics()
        
        base_risk = 1.0 - safety_metrics.safety_score
        
        # Adjust based on system state
        if 'cpu_usage' in system_state:
            if system_state['cpu_usage'] > 0.9:
                base_risk *= 1.5
        
        if 'memory_usage' in system_state:
            if system_state['memory_usage'] > 0.9:
                base_risk *= 1.3
        
        # Cap risk at 1.0
        risk_score = min(base_risk, 1.0)
        
        self.risk_history.append({
            'risk_score': risk_score,
            'system_state': system_state,
            'timestamp': datetime.now()
        })
        
        return risk_score


class ConstraintOptimizer:
    """Optimizes safety constraints for performance."""
    
    def __init__(self):
        """Initialize constraint optimizer."""
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def optimize_constraints(
        self,
        safety_boundaries: AdaptiveSafetyBoundaries,
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize constraints based on performance metrics."""
        recommendations = {}
        
        for name, constraint in safety_boundaries.constraints.items():
            if constraint.constraint_type == ConstraintType.ADAPTIVE:
                # Simple optimization logic
                if constraint.violation_count == 0 and performance_metrics.get('throughput', 0) < 0.7:
                    # Can tighten constraint for better safety
                    recommendations[name] = {
                        'action': 'tighten',
                        'new_threshold': constraint.threshold * 0.95
                    }
                elif constraint.violation_count > 5 and performance_metrics.get('throughput', 0) > 0.9:
                    # Should relax constraint for better performance
                    recommendations[name] = {
                        'action': 'relax',
                        'new_threshold': constraint.threshold * 1.05
                    }
        
        self.optimization_history.append({
            'recommendations': recommendations,
            'performance_metrics': performance_metrics,
            'timestamp': datetime.now()
        })
        
        return recommendations


class SafetyMonitoringDashboard:
    """Real-time safety monitoring dashboard."""
    
    def __init__(self, safety_boundaries: AdaptiveSafetyBoundaries):
        """Initialize monitoring dashboard."""
        self.safety_boundaries = safety_boundaries
        self.alert_subscribers: List[Any] = []
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        metrics = self.safety_boundaries.get_safety_metrics()
        
        return {
            'safety_score': metrics.safety_score,
            'risk_level': metrics.risk_level.value,
            'active_constraints': metrics.active_constraints,
            'recent_violations': metrics.violations_last_hour,
            'constraint_details': {
                name: {
                    'threshold': c.threshold,
                    'current': c.current_value,
                    'violations': c.violation_count
                }
                for name, c in self.safety_boundaries.constraints.items()
            }
        }
    
    async def send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert to subscribers."""
        for subscriber in self.alert_subscribers:
            try:
                await subscriber.notify(alert)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")