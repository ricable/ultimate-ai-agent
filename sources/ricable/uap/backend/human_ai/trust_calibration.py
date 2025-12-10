"""
Trust Calibration System for Human-AI Collaboration
Implements dynamic trust calibration, trust monitoring, and trust-based adaptations.
"""

import asyncio
import json
import logging
import numpy as np
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import uuid4
import statistics

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust level categories"""
    VERY_LOW = "very_low"     # 0.0 - 0.2
    LOW = "low"               # 0.2 - 0.4
    MODERATE = "moderate"     # 0.4 - 0.6
    HIGH = "high"             # 0.6 - 0.8
    VERY_HIGH = "very_high"   # 0.8 - 1.0


class TrustDimension(Enum):
    """Different dimensions of trust"""
    RELIABILITY = "reliability"         # Consistency of AI performance
    COMPETENCE = "competence"           # AI's ability to perform tasks
    TRANSPARENCY = "transparency"       # Explainability of AI decisions
    BENEVOLENCE = "benevolence"         # AI's intention to help
    PREDICTABILITY = "predictability"   # Consistency of AI behavior
    INTEGRITY = "integrity"             # Honesty and ethical behavior


class TrustEvent(Enum):
    """Events that can affect trust"""
    CORRECT_PREDICTION = "correct_prediction"
    INCORRECT_PREDICTION = "incorrect_prediction"
    HELPFUL_SUGGESTION = "helpful_suggestion"
    UNHELPFUL_SUGGESTION = "unhelpful_suggestion"
    CLEAR_EXPLANATION = "clear_explanation"
    UNCLEAR_EXPLANATION = "unclear_explanation"
    SYSTEM_ERROR = "system_error"
    SUCCESSFUL_TASK = "successful_task"
    FAILED_TASK = "failed_task"
    USER_FEEDBACK_POSITIVE = "user_feedback_positive"
    USER_FEEDBACK_NEGATIVE = "user_feedback_negative"
    TRANSPARENCY_INCREASED = "transparency_increased"
    TRANSPARENCY_DECREASED = "transparency_decreased"


@dataclass
class TrustMeasurement:
    """Single trust measurement"""
    measurement_id: str
    user_id: str
    session_id: str
    dimension: TrustDimension
    value: float  # 0-1 scale
    confidence: float  # Confidence in this measurement
    measurement_method: str  # How this was measured
    context: Dict[str, Any]
    timestamp: datetime
    contributing_events: List[str]  # Event IDs that led to this measurement


@dataclass
class TrustEvent:
    """Event that affects trust"""
    event_id: str
    user_id: str
    session_id: str
    event_type: str
    event_data: Dict[str, Any]
    trust_impact: Dict[TrustDimension, float]  # Impact on each trust dimension
    severity: float  # 0-1, how much this event should affect trust
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class TrustCalibrationRule:
    """Rule for calibrating trust based on events"""
    rule_id: str
    rule_name: str
    trigger_conditions: Dict[str, Any]
    trust_adjustments: Dict[TrustDimension, float]
    confidence_threshold: float
    decay_factor: float  # How quickly this rule's effect decays
    created_at: datetime
    effectiveness_score: float
    usage_count: int


@dataclass
class TrustProfile:
    """Complete trust profile for a user"""
    user_id: str
    overall_trust: float
    dimension_trust: Dict[TrustDimension, float]
    trust_history: List[TrustMeasurement]
    trust_events: List[TrustEvent]
    trust_volatility: float  # How quickly trust changes
    trust_baseline: float   # User's baseline trust level
    last_calibration: datetime
    calibration_count: int
    trust_trends: Dict[TrustDimension, float]  # Recent trends


class TrustCalculator:
    """Calculates trust scores based on various factors"""
    
    def __init__(self):
        self.dimension_weights = {
            TrustDimension.RELIABILITY: 0.25,
            TrustDimension.COMPETENCE: 0.20,
            TrustDimension.TRANSPARENCY: 0.15,
            TrustDimension.BENEVOLENCE: 0.15,
            TrustDimension.PREDICTABILITY: 0.15,
            TrustDimension.INTEGRITY: 0.10
        }
        
        self.event_impact_matrix = self._initialize_event_impact_matrix()
        self.trust_decay_rate = 0.99  # Daily decay rate for trust memories
        
    def _initialize_event_impact_matrix(self) -> Dict[str, Dict[TrustDimension, float]]:
        """Initialize impact matrix for different events on trust dimensions"""
        return {
            "correct_prediction": {
                TrustDimension.RELIABILITY: 0.05,
                TrustDimension.COMPETENCE: 0.08,
                TrustDimension.PREDICTABILITY: 0.03,
                TrustDimension.TRANSPARENCY: 0.0,
                TrustDimension.BENEVOLENCE: 0.0,
                TrustDimension.INTEGRITY: 0.02
            },
            "incorrect_prediction": {
                TrustDimension.RELIABILITY: -0.08,
                TrustDimension.COMPETENCE: -0.10,
                TrustDimension.PREDICTABILITY: -0.05,
                TrustDimension.TRANSPARENCY: 0.0,
                TrustDimension.BENEVOLENCE: 0.0,
                TrustDimension.INTEGRITY: -0.02
            },
            "helpful_suggestion": {
                TrustDimension.COMPETENCE: 0.06,
                TrustDimension.BENEVOLENCE: 0.08,
                TrustDimension.RELIABILITY: 0.03,
                TrustDimension.TRANSPARENCY: 0.02,
                TrustDimension.PREDICTABILITY: 0.0,
                TrustDimension.INTEGRITY: 0.02
            },
            "unhelpful_suggestion": {
                TrustDimension.COMPETENCE: -0.06,
                TrustDimension.BENEVOLENCE: -0.04,
                TrustDimension.RELIABILITY: -0.03,
                TrustDimension.TRANSPARENCY: 0.0,
                TrustDimension.PREDICTABILITY: 0.0,
                TrustDimension.INTEGRITY: 0.0
            },
            "clear_explanation": {
                TrustDimension.TRANSPARENCY: 0.10,
                TrustDimension.COMPETENCE: 0.04,
                TrustDimension.BENEVOLENCE: 0.03,
                TrustDimension.RELIABILITY: 0.02,
                TrustDimension.PREDICTABILITY: 0.02,
                TrustDimension.INTEGRITY: 0.02
            },
            "unclear_explanation": {
                TrustDimension.TRANSPARENCY: -0.12,
                TrustDimension.COMPETENCE: -0.03,
                TrustDimension.BENEVOLENCE: -0.02,
                TrustDimension.RELIABILITY: 0.0,
                TrustDimension.PREDICTABILITY: 0.0,
                TrustDimension.INTEGRITY: 0.0
            },
            "system_error": {
                TrustDimension.RELIABILITY: -0.15,
                TrustDimension.COMPETENCE: -0.08,
                TrustDimension.PREDICTABILITY: -0.10,
                TrustDimension.TRANSPARENCY: 0.0,
                TrustDimension.BENEVOLENCE: 0.0,
                TrustDimension.INTEGRITY: -0.05
            },
            "successful_task": {
                TrustDimension.COMPETENCE: 0.07,
                TrustDimension.RELIABILITY: 0.05,
                TrustDimension.BENEVOLENCE: 0.03,
                TrustDimension.PREDICTABILITY: 0.03,
                TrustDimension.TRANSPARENCY: 0.0,
                TrustDimension.INTEGRITY: 0.02
            },
            "failed_task": {
                TrustDimension.COMPETENCE: -0.10,
                TrustDimension.RELIABILITY: -0.08,
                TrustDimension.BENEVOLENCE: -0.02,
                TrustDimension.PREDICTABILITY: -0.05,
                TrustDimension.TRANSPARENCY: 0.0,
                TrustDimension.INTEGRITY: 0.0
            },
            "user_feedback_positive": {
                TrustDimension.BENEVOLENCE: 0.06,
                TrustDimension.COMPETENCE: 0.05,
                TrustDimension.RELIABILITY: 0.04,
                TrustDimension.TRANSPARENCY: 0.03,
                TrustDimension.PREDICTABILITY: 0.02,
                TrustDimension.INTEGRITY: 0.03
            },
            "user_feedback_negative": {
                TrustDimension.BENEVOLENCE: -0.06,
                TrustDimension.COMPETENCE: -0.05,
                TrustDimension.RELIABILITY: -0.04,
                TrustDimension.TRANSPARENCY: -0.02,
                TrustDimension.PREDICTABILITY: -0.03,
                TrustDimension.INTEGRITY: -0.02
            }
        }
    
    async def calculate_overall_trust(self, dimension_trust: Dict[TrustDimension, float]) -> float:
        """Calculate overall trust from dimensional trust scores"""
        weighted_sum = sum(
            dimension_trust.get(dimension, 0.5) * weight
            for dimension, weight in self.dimension_weights.items()
        )
        return max(0.0, min(1.0, weighted_sum))
    
    async def update_trust_from_event(self, current_trust: Dict[TrustDimension, float],
                                    event_type: str, event_context: Dict[str, Any]) -> Dict[TrustDimension, float]:
        """Update trust scores based on an event"""
        if event_type not in self.event_impact_matrix:
            logger.warning(f"Unknown event type for trust calculation: {event_type}")
            return current_trust
        
        impact = self.event_impact_matrix[event_type]
        updated_trust = current_trust.copy()
        
        # Apply context-based modifications
        severity_multiplier = event_context.get("severity", 1.0)
        confidence_multiplier = event_context.get("confidence", 1.0)
        
        for dimension, base_impact in impact.items():
            # Apply multipliers
            adjusted_impact = base_impact * severity_multiplier * confidence_multiplier
            
            # Update trust with bounds checking
            current_value = updated_trust.get(dimension, 0.5)
            new_value = current_value + adjusted_impact
            updated_trust[dimension] = max(0.0, min(1.0, new_value))
        
        return updated_trust
    
    async def apply_temporal_decay(self, trust_measurements: List[TrustMeasurement]) -> Dict[TrustDimension, float]:
        """Apply temporal decay to trust measurements"""
        current_time = datetime.utcnow()
        dimension_values = {dimension: [] for dimension in TrustDimension}
        
        for measurement in trust_measurements:
            # Calculate decay based on age
            age_days = (current_time - measurement.timestamp).total_seconds() / 86400
            decay_factor = self.trust_decay_rate ** age_days
            
            # Apply decay to measurement
            decayed_value = measurement.value * decay_factor * measurement.confidence
            dimension_values[measurement.dimension].append(decayed_value)
        
        # Calculate weighted averages
        dimension_trust = {}
        for dimension, values in dimension_values.items():
            if values:
                dimension_trust[dimension] = statistics.mean(values)
            else:
                dimension_trust[dimension] = 0.5  # Default neutral trust
        
        return dimension_trust
    
    def categorize_trust_level(self, trust_score: float) -> TrustLevel:
        """Categorize numerical trust score into trust level"""
        if trust_score < 0.2:
            return TrustLevel.VERY_LOW
        elif trust_score < 0.4:
            return TrustLevel.LOW
        elif trust_score < 0.6:
            return TrustLevel.MODERATE
        elif trust_score < 0.8:
            return TrustLevel.HIGH
        else:
            return TrustLevel.VERY_HIGH
    
    async def calculate_trust_volatility(self, trust_history: List[TrustMeasurement]) -> float:
        """Calculate trust volatility (how much trust changes over time)"""
        if len(trust_history) < 3:
            return 0.0
        
        # Get recent overall trust scores
        recent_measurements = sorted(trust_history, key=lambda x: x.timestamp)[-10:]
        trust_scores = []
        
        # Group measurements by time and calculate overall trust for each time point
        time_groups = {}
        for measurement in recent_measurements:
            time_key = measurement.timestamp.strftime("%Y-%m-%d %H")
            if time_key not in time_groups:
                time_groups[time_key] = {}
            time_groups[time_key][measurement.dimension] = measurement.value
        
        # Calculate overall trust for each time group
        for time_key, dimension_values in time_groups.items():
            overall_trust = await self.calculate_overall_trust(dimension_values)
            trust_scores.append(overall_trust)
        
        if len(trust_scores) < 2:
            return 0.0
        
        # Calculate standard deviation as volatility measure
        return float(np.std(trust_scores))


class TrustMonitor:
    """Monitors trust-related events and patterns"""
    
    def __init__(self):
        self.trust_events_buffer: List[TrustEvent] = []
        self.trust_patterns: Dict[str, Any] = {}
        self.anomaly_threshold = 0.3  # Threshold for detecting trust anomalies
        
    async def monitor_trust_event(self, event: TrustEvent) -> Dict[str, Any]:
        """Monitor and analyze a trust-related event"""
        self.trust_events_buffer.append(event)
        
        # Keep buffer size manageable
        if len(self.trust_events_buffer) > 1000:
            self.trust_events_buffer = self.trust_events_buffer[-500:]
        
        analysis = {
            "event_analysis": await self._analyze_event(event),
            "pattern_detection": await self._detect_patterns(event),
            "anomaly_detection": await self._detect_anomalies(event),
            "trust_trend": await self._calculate_trust_trend(event.user_id),
            "recommendations": await self._generate_recommendations(event)
        }
        
        return analysis
    
    async def _analyze_event(self, event: TrustEvent) -> Dict[str, Any]:
        """Analyze individual trust event"""
        return {
            "event_type": event.event_type,
            "severity": event.severity,
            "dimensions_affected": list(event.trust_impact.keys()),
            "positive_impact": sum(1 for impact in event.trust_impact.values() if impact > 0),
            "negative_impact": sum(1 for impact in event.trust_impact.values() if impact < 0),
            "strongest_impact_dimension": max(event.trust_impact.items(), key=lambda x: abs(x[1]))[0].value,
            "context_richness": len(event.context)
        }
    
    async def _detect_patterns(self, event: TrustEvent) -> Dict[str, Any]:
        """Detect patterns in trust events"""
        user_events = [e for e in self.trust_events_buffer if e.user_id == event.user_id]
        
        if len(user_events) < 3:
            return {"patterns_detected": []}
        
        patterns = []
        
        # Pattern: Repeated negative events
        recent_events = user_events[-5:]
        negative_events = [e for e in recent_events if any(impact < 0 for impact in e.trust_impact.values())]
        
        if len(negative_events) >= 3:
            patterns.append({
                "pattern_type": "repeated_negative_events",
                "frequency": len(negative_events),
                "severity": "high",
                "recommendation": "intervention_needed"
            })
        
        # Pattern: Trust recovery
        if len(user_events) >= 5:
            recent_trend = [sum(e.trust_impact.values()) for e in user_events[-5:]]
            if recent_trend[0] < 0 and recent_trend[-1] > 0:
                patterns.append({
                    "pattern_type": "trust_recovery",
                    "trend": "positive",
                    "severity": "low",
                    "recommendation": "maintain_current_approach"
                })
        
        # Pattern: High volatility
        impact_values = [sum(e.trust_impact.values()) for e in recent_events]
        if len(impact_values) > 2 and np.std(impact_values) > 0.1:
            patterns.append({
                "pattern_type": "high_volatility",
                "volatility_score": float(np.std(impact_values)),
                "severity": "medium",
                "recommendation": "stabilize_interactions"
            })
        
        return {"patterns_detected": patterns}
    
    async def _detect_anomalies(self, event: TrustEvent) -> Dict[str, Any]:
        """Detect anomalous trust events"""
        user_events = [e for e in self.trust_events_buffer if e.user_id == event.user_id]
        
        if len(user_events) < 5:
            return {"anomalies_detected": []}
        
        anomalies = []
        
        # Calculate baseline trust impact for this user
        recent_impacts = [sum(e.trust_impact.values()) for e in user_events[-10:]]
        baseline_impact = statistics.mean(recent_impacts)
        impact_std = statistics.stdev(recent_impacts) if len(recent_impacts) > 1 else 0.1
        
        current_impact = sum(event.trust_impact.values())
        
        # Check if current event is anomalous
        if abs(current_impact - baseline_impact) > (2 * impact_std + self.anomaly_threshold):
            anomalies.append({
                "anomaly_type": "unusual_impact_magnitude",
                "current_impact": current_impact,
                "baseline_impact": baseline_impact,
                "deviation": abs(current_impact - baseline_impact),
                "severity": "high" if abs(current_impact - baseline_impact) > 0.5 else "medium"
            })
        
        # Check for unusual event frequency
        recent_time_window = datetime.utcnow() - timedelta(minutes=10)
        recent_events_count = len([e for e in user_events if e.timestamp > recent_time_window])
        
        if recent_events_count > 10:  # Too many events in short time
            anomalies.append({
                "anomaly_type": "high_event_frequency",
                "event_count": recent_events_count,
                "time_window_minutes": 10,
                "severity": "medium"
            })
        
        return {"anomalies_detected": anomalies}
    
    async def _calculate_trust_trend(self, user_id: str) -> Dict[str, Any]:
        """Calculate trust trend for a user"""
        user_events = [e for e in self.trust_events_buffer if e.user_id == user_id]
        
        if len(user_events) < 3:
            return {"trend": "insufficient_data"}
        
        # Calculate trend over different time windows
        now = datetime.utcnow()
        time_windows = {
            "last_hour": timedelta(hours=1),
            "last_day": timedelta(days=1),
            "last_week": timedelta(weeks=1)
        }
        
        trends = {}
        for window_name, window_duration in time_windows.items():
            window_start = now - window_duration
            window_events = [e for e in user_events if e.timestamp > window_start]
            
            if len(window_events) >= 2:
                impacts = [sum(e.trust_impact.values()) for e in window_events]
                trend_slope = np.polyfit(range(len(impacts)), impacts, 1)[0] if len(impacts) > 1 else 0
                
                trends[window_name] = {
                    "slope": float(trend_slope),
                    "direction": "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable",
                    "event_count": len(window_events),
                    "average_impact": float(np.mean(impacts))
                }
            else:
                trends[window_name] = {"trend": "insufficient_data"}
        
        return trends
    
    async def _generate_recommendations(self, event: TrustEvent) -> List[str]:
        """Generate recommendations based on trust event"""
        recommendations = []
        
        # Recommendations based on event type
        if event.event_type in ["incorrect_prediction", "failed_task", "system_error"]:
            recommendations.extend([
                "provide_explanation_for_failure",
                "offer_alternative_approach",
                "increase_transparency",
                "acknowledge_limitation"
            ])
        
        if event.event_type in ["unclear_explanation", "transparency_decreased"]:
            recommendations.extend([
                "improve_explanation_clarity",
                "provide_step_by_step_reasoning",
                "offer_additional_context",
                "use_simpler_language"
            ])
        
        # Recommendations based on severity
        if event.severity > 0.7:
            recommendations.extend([
                "immediate_attention_required",
                "consider_human_intervention",
                "escalate_to_supervisor"
            ])
        
        # Recommendations based on trust impact
        negative_dimensions = [dim for dim, impact in event.trust_impact.items() if impact < -0.05]
        
        if TrustDimension.TRANSPARENCY in negative_dimensions:
            recommendations.append("increase_explanation_detail")
        
        if TrustDimension.COMPETENCE in negative_dimensions:
            recommendations.append("demonstrate_expertise")
        
        if TrustDimension.RELIABILITY in negative_dimensions:
            recommendations.append("improve_consistency")
        
        return list(set(recommendations))  # Remove duplicates


class TrustCalibrationEngine:
    """Main engine for trust calibration and management"""
    
    def __init__(self):
        self.trust_calculator = TrustCalculator()
        self.trust_monitor = TrustMonitor()
        self.user_trust_profiles: Dict[str, TrustProfile] = {}
        self.calibration_rules: Dict[str, TrustCalibrationRule] = {}
        self.trust_thresholds = {
            "intervention_threshold": 0.3,  # Below this, intervention is needed
            "optimal_threshold": 0.7,      # Above this, trust is optimal
            "volatility_threshold": 0.2    # Above this, trust is too volatile
        }
        
        # Initialize default calibration rules
        self._initialize_calibration_rules()
    
    async def initialize_user_trust(self, user_id: str, initial_data: Dict[str, Any] = None) -> TrustProfile:
        """Initialize trust profile for a new user"""
        if initial_data is None:
            initial_data = {}
        
        # Set initial trust levels
        initial_trust = initial_data.get("initial_trust", 0.6)  # Slightly optimistic default
        
        trust_profile = TrustProfile(
            user_id=user_id,
            overall_trust=initial_trust,
            dimension_trust={
                dimension: initial_trust for dimension in TrustDimension
            },
            trust_history=[],
            trust_events=[],
            trust_volatility=0.0,
            trust_baseline=initial_trust,
            last_calibration=datetime.utcnow(),
            calibration_count=0,
            trust_trends={dimension: 0.0 for dimension in TrustDimension}
        )
        
        self.user_trust_profiles[user_id] = trust_profile
        logger.info(f"Initialized trust profile for user {user_id} with initial trust {initial_trust}")
        
        return trust_profile
    
    async def record_trust_event(self, user_id: str, session_id: str, event_type: str,
                               event_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Record a trust-affecting event and update trust"""
        if user_id not in self.user_trust_profiles:
            await self.initialize_user_trust(user_id)
        
        trust_profile = self.user_trust_profiles[user_id]
        
        # Calculate trust impact
        severity = event_data.get("severity", 1.0)
        trust_impact = await self._calculate_event_impact(event_type, event_data, context or {})
        
        # Create trust event
        trust_event = TrustEvent(
            event_id=str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            event_data=event_data,
            trust_impact=trust_impact,
            severity=severity,
            timestamp=datetime.utcnow(),
            context=context or {}
        )
        
        # Store event
        trust_profile.trust_events.append(trust_event)
        
        # Monitor event
        monitoring_analysis = await self.trust_monitor.monitor_trust_event(trust_event)
        
        # Update trust scores
        updated_trust = await self.trust_calculator.update_trust_from_event(
            trust_profile.dimension_trust, event_type, {
                "severity": severity,
                "confidence": event_data.get("confidence", 1.0)
            }
        )
        
        trust_profile.dimension_trust = updated_trust
        trust_profile.overall_trust = await self.trust_calculator.calculate_overall_trust(updated_trust)
        
        # Create trust measurement
        measurement = TrustMeasurement(
            measurement_id=str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            dimension=TrustDimension.RELIABILITY,  # This would be determined by event type
            value=trust_profile.overall_trust,
            confidence=event_data.get("confidence", 0.8),
            measurement_method="event_based",
            context=context or {},
            timestamp=datetime.utcnow(),
            contributing_events=[trust_event.event_id]
        )
        
        trust_profile.trust_history.append(measurement)
        
        # Update volatility
        trust_profile.trust_volatility = await self.trust_calculator.calculate_trust_volatility(
            trust_profile.trust_history
        )
        
        # Apply calibration rules
        calibration_result = await self._apply_calibration_rules(trust_profile, trust_event)
        
        # Update trust trends
        trust_profile.trust_trends = await self._calculate_trust_trends(trust_profile)
        
        logger.info(f"Recorded trust event {event_type} for user {user_id}, new overall trust: {trust_profile.overall_trust:.3f}")
        
        return {
            "trust_event_id": trust_event.event_id,
            "updated_overall_trust": trust_profile.overall_trust,
            "updated_dimension_trust": {dim.value: score for dim, score in trust_profile.dimension_trust.items()},
            "trust_level": self.trust_calculator.categorize_trust_level(trust_profile.overall_trust).value,
            "trust_volatility": trust_profile.trust_volatility,
            "monitoring_analysis": monitoring_analysis,
            "calibration_result": calibration_result,
            "recommendations": await self._generate_trust_recommendations(trust_profile)
        }
    
    async def _calculate_event_impact(self, event_type: str, event_data: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[TrustDimension, float]:
        """Calculate the impact of an event on trust dimensions"""
        base_impact = self.trust_calculator.event_impact_matrix.get(event_type, {})
        
        # Apply context-based modifications
        modified_impact = {}
        for dimension, impact in base_impact.items():
            # Modify impact based on context
            context_multiplier = 1.0
            
            # User expertise affects impact
            if "user_expertise" in context:
                expertise = context["user_expertise"]
                if expertise < 0.3:  # Novice users more affected by negative events
                    context_multiplier = 1.2 if impact < 0 else 1.0
                elif expertise > 0.7:  # Expert users less affected by positive events
                    context_multiplier = 1.0 if impact < 0 else 0.8
            
            # Task complexity affects impact
            if "task_complexity" in context:
                complexity = context["task_complexity"]
                if complexity == "high":
                    context_multiplier *= 1.1  # Higher impact for complex tasks
            
            modified_impact[dimension] = impact * context_multiplier
        
        return modified_impact
    
    async def _apply_calibration_rules(self, trust_profile: TrustProfile,
                                     trust_event: TrustEvent) -> Dict[str, Any]:
        """Apply calibration rules to adjust trust"""
        applied_rules = []
        total_adjustment = {dimension: 0.0 for dimension in TrustDimension}
        
        for rule in self.calibration_rules.values():
            if await self._rule_applies(rule, trust_profile, trust_event):
                # Apply rule adjustments
                for dimension, adjustment in rule.trust_adjustments.items():
                    total_adjustment[dimension] += adjustment
                
                applied_rules.append(rule.rule_id)
                rule.usage_count += 1
        
        # Apply total adjustments
        for dimension, adjustment in total_adjustment.items():
            current_value = trust_profile.dimension_trust.get(dimension, 0.5)
            new_value = max(0.0, min(1.0, current_value + adjustment))
            trust_profile.dimension_trust[dimension] = new_value
        
        # Recalculate overall trust
        trust_profile.overall_trust = await self.trust_calculator.calculate_overall_trust(
            trust_profile.dimension_trust
        )
        
        trust_profile.calibration_count += 1
        trust_profile.last_calibration = datetime.utcnow()
        
        return {
            "rules_applied": applied_rules,
            "total_adjustments": {dim.value: adj for dim, adj in total_adjustment.items()},
            "new_overall_trust": trust_profile.overall_trust
        }
    
    async def _rule_applies(self, rule: TrustCalibrationRule, trust_profile: TrustProfile,
                          trust_event: TrustEvent) -> bool:
        """Check if a calibration rule applies to the current situation"""
        conditions = rule.trigger_conditions
        
        # Check event type condition
        if "event_type" in conditions:
            if trust_event.event_type not in conditions["event_type"]:
                return False
        
        # Check trust level condition
        if "trust_level_below" in conditions:
            if trust_profile.overall_trust >= conditions["trust_level_below"]:
                return False
        
        if "trust_level_above" in conditions:
            if trust_profile.overall_trust <= conditions["trust_level_above"]:
                return False
        
        # Check volatility condition
        if "volatility_above" in conditions:
            if trust_profile.trust_volatility <= conditions["volatility_above"]:
                return False
        
        # Check event severity condition
        if "severity_above" in conditions:
            if trust_event.severity <= conditions["severity_above"]:
                return False
        
        return True
    
    async def _calculate_trust_trends(self, trust_profile: TrustProfile) -> Dict[TrustDimension, float]:
        """Calculate trust trends for each dimension"""
        trends = {}
        
        for dimension in TrustDimension:
            # Get recent measurements for this dimension
            dimension_measurements = [
                m for m in trust_profile.trust_history[-10:]
                if m.dimension == dimension
            ]
            
            if len(dimension_measurements) >= 3:
                values = [m.value for m in dimension_measurements]
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                trends[dimension] = float(trend_slope)
            else:
                trends[dimension] = 0.0
        
        return trends
    
    async def _generate_trust_recommendations(self, trust_profile: TrustProfile) -> List[str]:
        """Generate recommendations based on trust profile"""
        recommendations = []
        
        # Overall trust recommendations
        if trust_profile.overall_trust < self.trust_thresholds["intervention_threshold"]:
            recommendations.extend([
                "immediate_trust_intervention_needed",
                "increase_transparency",
                "provide_more_explanations",
                "reduce_ai_autonomy",
                "involve_human_oversight"
            ])
        elif trust_profile.overall_trust > self.trust_thresholds["optimal_threshold"]:
            recommendations.extend([
                "trust_level_optimal",
                "maintain_current_approach",
                "consider_increased_autonomy"
            ])
        else:
            recommendations.extend([
                "moderate_trust_level",
                "gradual_trust_building",
                "consistent_performance_important"
            ])
        
        # Volatility recommendations
        if trust_profile.trust_volatility > self.trust_thresholds["volatility_threshold"]:
            recommendations.extend([
                "trust_too_volatile",
                "stabilize_ai_behavior",
                "improve_consistency",
                "reduce_unexpected_changes"
            ])
        
        # Dimension-specific recommendations
        for dimension, score in trust_profile.dimension_trust.items():
            if score < 0.4:
                if dimension == TrustDimension.TRANSPARENCY:
                    recommendations.append("improve_explanation_quality")
                elif dimension == TrustDimension.COMPETENCE:
                    recommendations.append("demonstrate_expertise_better")
                elif dimension == TrustDimension.RELIABILITY:
                    recommendations.append("improve_consistency")
                elif dimension == TrustDimension.BENEVOLENCE:
                    recommendations.append("show_more_user_focus")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _initialize_calibration_rules(self):
        """Initialize default calibration rules"""
        rules = [
            TrustCalibrationRule(
                rule_id="low_trust_recovery",
                rule_name="Low Trust Recovery Rule",
                trigger_conditions={
                    "trust_level_below": 0.3,
                    "event_type": ["correct_prediction", "successful_task", "helpful_suggestion"]
                },
                trust_adjustments={
                    TrustDimension.RELIABILITY: 0.02,
                    TrustDimension.COMPETENCE: 0.03
                },
                confidence_threshold=0.7,
                decay_factor=0.95,
                created_at=datetime.utcnow(),
                effectiveness_score=0.8,
                usage_count=0
            ),
            TrustCalibrationRule(
                rule_id="high_volatility_stabilization",
                rule_name="High Volatility Stabilization Rule",
                trigger_conditions={
                    "volatility_above": 0.25
                },
                trust_adjustments={
                    TrustDimension.PREDICTABILITY: 0.05
                },
                confidence_threshold=0.6,
                decay_factor=0.9,
                created_at=datetime.utcnow(),
                effectiveness_score=0.7,
                usage_count=0
            ),
            TrustCalibrationRule(
                rule_id="severe_error_mitigation",
                rule_name="Severe Error Mitigation Rule",
                trigger_conditions={
                    "event_type": ["system_error", "failed_task"],
                    "severity_above": 0.7
                },
                trust_adjustments={
                    TrustDimension.TRANSPARENCY: 0.08,
                    TrustDimension.BENEVOLENCE: 0.05
                },
                confidence_threshold=0.8,
                decay_factor=0.85,
                created_at=datetime.utcnow(),
                effectiveness_score=0.75,
                usage_count=0
            )
        ]
        
        for rule in rules:
            self.calibration_rules[rule.rule_id] = rule
    
    async def get_user_trust_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive trust status for a user"""
        if user_id not in self.user_trust_profiles:
            return {"error": "User trust profile not found"}
        
        trust_profile = self.user_trust_profiles[user_id]
        
        return {
            "user_id": user_id,
            "overall_trust": trust_profile.overall_trust,
            "trust_level": self.trust_calculator.categorize_trust_level(trust_profile.overall_trust).value,
            "dimension_trust": {dim.value: score for dim, score in trust_profile.dimension_trust.items()},
            "trust_volatility": trust_profile.trust_volatility,
            "trust_trends": {dim.value: trend for dim, trend in trust_profile.trust_trends.items()},
            "recent_events_count": len([e for e in trust_profile.trust_events 
                                       if (datetime.utcnow() - e.timestamp).total_seconds() < 3600]),
            "calibration_count": trust_profile.calibration_count,
            "last_calibration": trust_profile.last_calibration.isoformat(),
            "trust_baseline": trust_profile.trust_baseline,
            "recommendations": await self._generate_trust_recommendations(trust_profile)
        }
    
    async def calibrate_trust_for_context(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrate trust based on current context"""
        if user_id not in self.user_trust_profiles:
            await self.initialize_user_trust(user_id)
        
        trust_profile = self.user_trust_profiles[user_id]
        
        # Context-based trust adjustments
        context_adjustments = {}
        
        # Adjust based on task complexity
        if "task_complexity" in context:
            complexity = context["task_complexity"]
            if complexity == "high" and trust_profile.overall_trust < 0.6:
                context_adjustments["reduce_ai_autonomy"] = True
                context_adjustments["increase_human_oversight"] = True
        
        # Adjust based on stakes
        if "task_stakes" in context:
            stakes = context["task_stakes"]
            if stakes == "high" and trust_profile.overall_trust < 0.7:
                context_adjustments["require_human_confirmation"] = True
                context_adjustments["provide_uncertainty_estimates"] = True
        
        # Adjust based on user expertise
        if "user_expertise" in context:
            expertise = context["user_expertise"]
            if expertise < 0.3 and trust_profile.overall_trust > 0.8:
                context_adjustments["provide_more_explanations"] = True
                context_adjustments["use_simpler_language"] = True
        
        return {
            "user_id": user_id,
            "context_adjustments": context_adjustments,
            "current_trust_level": trust_profile.overall_trust,
            "recommended_ai_autonomy": self._calculate_recommended_autonomy(trust_profile, context),
            "trust_status": await self.get_user_trust_status(user_id)
        }
    
    def _calculate_recommended_autonomy(self, trust_profile: TrustProfile, context: Dict[str, Any]) -> float:
        """Calculate recommended level of AI autonomy based on trust and context"""
        base_autonomy = trust_profile.overall_trust
        
        # Adjust based on context
        if context.get("task_stakes") == "high":
            base_autonomy *= 0.8
        elif context.get("task_stakes") == "low":
            base_autonomy *= 1.1
        
        if context.get("task_complexity") == "high":
            base_autonomy *= 0.9
        
        if trust_profile.trust_volatility > 0.2:
            base_autonomy *= 0.85
        
        return max(0.0, min(1.0, base_autonomy))
    
    async def get_system_trust_status(self) -> Dict[str, Any]:
        """Get overall system trust status"""
        if not self.user_trust_profiles:
            return {"message": "No user trust profiles available"}
        
        # Calculate aggregate statistics
        all_trust_scores = [profile.overall_trust for profile in self.user_trust_profiles.values()]
        all_volatilities = [profile.trust_volatility for profile in self.user_trust_profiles.values()]
        
        trust_distribution = {
            "very_low": sum(1 for score in all_trust_scores if score < 0.2),
            "low": sum(1 for score in all_trust_scores if 0.2 <= score < 0.4),
            "moderate": sum(1 for score in all_trust_scores if 0.4 <= score < 0.6),
            "high": sum(1 for score in all_trust_scores if 0.6 <= score < 0.8),
            "very_high": sum(1 for score in all_trust_scores if score >= 0.8)
        }
        
        return {
            "total_users": len(self.user_trust_profiles),
            "average_trust": statistics.mean(all_trust_scores),
            "median_trust": statistics.median(all_trust_scores),
            "trust_std_deviation": statistics.stdev(all_trust_scores) if len(all_trust_scores) > 1 else 0.0,
            "trust_distribution": trust_distribution,
            "average_volatility": statistics.mean(all_volatilities),
            "high_volatility_users": sum(1 for vol in all_volatilities if vol > 0.2),
            "low_trust_users": sum(1 for score in all_trust_scores if score < 0.4),
            "total_calibration_rules": len(self.calibration_rules),
            "total_trust_events": sum(len(profile.trust_events) for profile in self.user_trust_profiles.values()),
            "recent_events": sum(len([e for e in profile.trust_events 
                                     if (datetime.utcnow() - e.timestamp).total_seconds() < 3600])
                                for profile in self.user_trust_profiles.values())
        }
