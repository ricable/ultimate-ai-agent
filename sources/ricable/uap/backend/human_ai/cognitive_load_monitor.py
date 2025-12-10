"""
Cognitive Load Monitor for Human-AI Collaboration
Monitors user cognitive load and provides adaptive interface recommendations.
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
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class CognitiveLoadLevel(Enum):
    """Cognitive load level categories"""
    VERY_LOW = "very_low"       # < 0.2
    LOW = "low"                 # 0.2 - 0.4
    MODERATE = "moderate"       # 0.4 - 0.6
    HIGH = "high"               # 0.6 - 0.8
    OVERLOAD = "overload"       # > 0.8


class LoadIndicator(Enum):
    """Types of cognitive load indicators"""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    TASK_SWITCHING = "task_switching"
    MULTITASKING = "multitasking"
    MESSAGE_COMPLEXITY = "message_complexity"
    INTERACTION_FREQUENCY = "interaction_frequency"
    ATTENTION_FOCUS = "attention_focus"
    DECISION_QUALITY = "decision_quality"
    COMPREHENSION_SPEED = "comprehension_speed"
    FRUSTRATION_LEVEL = "frustration_level"


class AdaptationTrigger(Enum):
    """Triggers for cognitive load adaptations"""
    HIGH_LOAD_DETECTED = "high_load_detected"
    OVERLOAD_DETECTED = "overload_detected"
    LOW_LOAD_DETECTED = "low_load_detected"
    LOAD_SPIKE = "load_spike"
    SUSTAINED_HIGH_LOAD = "sustained_high_load"
    RAPID_LOAD_INCREASE = "rapid_load_increase"
    ATTENTION_FRAGMENTATION = "attention_fragmentation"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class CognitiveLoadMeasurement:
    """Single cognitive load measurement"""
    measurement_id: str
    user_id: str
    session_id: str
    overall_load: float  # 0-1 scale
    load_level: CognitiveLoadLevel
    indicator_scores: Dict[LoadIndicator, float]
    contributing_factors: List[str]
    context: Dict[str, Any]
    timestamp: datetime
    measurement_confidence: float


@dataclass
class LoadAdaptation:
    """Cognitive load adaptation recommendation"""
    adaptation_id: str
    trigger: AdaptationTrigger
    adaptation_type: str
    target_load_reduction: float
    recommended_actions: List[str]
    ui_modifications: Dict[str, Any]
    interaction_changes: Dict[str, Any]
    priority: str  # "immediate", "high", "medium", "low"
    estimated_effectiveness: float
    created_at: datetime


@dataclass
class CognitiveProfile:
    """User's cognitive profile and patterns"""
    user_id: str
    baseline_load: float
    load_capacity: float  # Maximum sustainable load
    load_patterns: Dict[str, Any]
    attention_span: float  # Average attention span in minutes
    task_switching_tolerance: float
    preferred_information_density: str  # "low", "medium", "high"
    cognitive_style: str  # "analytical", "intuitive", "visual"
    load_recovery_rate: float  # How quickly load decreases
    stress_indicators: List[str]
    optimal_performance_conditions: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class LoadIndicatorCalculator:
    """Calculates individual cognitive load indicators"""
    
    def __init__(self):
        self.indicator_weights = {
            LoadIndicator.RESPONSE_TIME: 0.15,
            LoadIndicator.ERROR_RATE: 0.20,
            LoadIndicator.TASK_SWITCHING: 0.10,
            LoadIndicator.MULTITASKING: 0.10,
            LoadIndicator.MESSAGE_COMPLEXITY: 0.15,
            LoadIndicator.INTERACTION_FREQUENCY: 0.10,
            LoadIndicator.ATTENTION_FOCUS: 0.10,
            LoadIndicator.DECISION_QUALITY: 0.10
        }
        
        # Baseline thresholds for different indicators
        self.indicator_thresholds = {
            LoadIndicator.RESPONSE_TIME: {"low": 2, "medium": 10, "high": 30},  # seconds
            LoadIndicator.ERROR_RATE: {"low": 0.05, "medium": 0.15, "high": 0.30},  # rate
            LoadIndicator.MESSAGE_COMPLEXITY: {"low": 0.3, "medium": 0.6, "high": 0.8},  # 0-1 scale
            LoadIndicator.INTERACTION_FREQUENCY: {"low": 0.5, "medium": 2.0, "high": 5.0}  # per minute
        }
    
    async def calculate_response_time_load(self, response_time: float, 
                                         user_profile: CognitiveProfile) -> float:
        """Calculate cognitive load from response time"""
        if response_time <= 0:
            return 0.0
        
        # Normalize based on user's baseline
        baseline_response_time = user_profile.load_patterns.get("avg_response_time", 5.0)
        
        # Calculate load based on deviation from baseline
        if response_time <= baseline_response_time:
            return 0.1  # Fast response, low load
        else:
            # Load increases with longer response times
            excess_time = response_time - baseline_response_time
            load = min(1.0, 0.1 + (excess_time / 60.0))  # Saturate at 1 minute excess
            return load
    
    async def calculate_error_rate_load(self, recent_errors: List[Dict[str, Any]],
                                      total_interactions: int) -> float:
        """Calculate cognitive load from error rate"""
        if total_interactions == 0:
            return 0.0
        
        error_rate = len(recent_errors) / total_interactions
        
        # Map error rate to load
        if error_rate < 0.05:
            return 0.1
        elif error_rate < 0.15:
            return 0.4
        elif error_rate < 0.30:
            return 0.7
        else:
            return 1.0
    
    async def calculate_task_switching_load(self, task_switches: int,
                                          time_window_minutes: float) -> float:
        """Calculate cognitive load from task switching frequency"""
        if time_window_minutes <= 0:
            return 0.0
        
        switches_per_minute = task_switches / time_window_minutes
        
        # Task switching creates cognitive load
        if switches_per_minute < 0.2:
            return 0.1
        elif switches_per_minute < 0.5:
            return 0.3
        elif switches_per_minute < 1.0:
            return 0.6
        else:
            return 0.9
    
    async def calculate_multitasking_load(self, concurrent_tasks: int,
                                        user_profile: CognitiveProfile) -> float:
        """Calculate cognitive load from multitasking"""
        if concurrent_tasks <= 1:
            return 0.1
        
        # Load increases exponentially with concurrent tasks
        base_load = 0.2
        additional_load = (concurrent_tasks - 1) * 0.3
        
        # Adjust for user's multitasking tolerance
        tolerance = user_profile.task_switching_tolerance
        adjusted_load = base_load + (additional_load / tolerance)
        
        return min(1.0, adjusted_load)
    
    async def calculate_message_complexity_load(self, message: str,
                                              user_profile: CognitiveProfile) -> float:
        """Calculate cognitive load from message complexity"""
        if not message:
            return 0.0
        
        # Analyze message complexity
        words = message.split()
        sentences = len([s for s in message.split('.') if s.strip()])
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Calculate complexity score
        word_count_score = min(1.0, len(words) / 100.0)  # Normalize to 100 words
        sentence_complexity = min(1.0, (len(words) / max(1, sentences)) / 20.0)  # Avg words per sentence
        word_complexity = min(1.0, avg_word_length / 10.0)  # Normalize to 10 char words
        
        complexity = (word_count_score + sentence_complexity + word_complexity) / 3.0
        
        # Adjust for user's cognitive style
        if user_profile.cognitive_style == "analytical":
            complexity *= 0.8  # Better at handling complex information
        elif user_profile.cognitive_style == "intuitive":
            complexity *= 1.2  # May struggle with very detailed information
        
        return complexity
    
    async def calculate_interaction_frequency_load(self, interactions_per_minute: float,
                                                 user_profile: CognitiveProfile) -> float:
        """Calculate cognitive load from interaction frequency"""
        # Optimal frequency is around 1-2 interactions per minute
        optimal_frequency = user_profile.load_patterns.get("optimal_interaction_frequency", 1.5)
        
        if interactions_per_minute < optimal_frequency * 0.5:
            # Too slow, might indicate confusion or overload
            return 0.6
        elif interactions_per_minute > optimal_frequency * 3:
            # Too fast, indicates stress or rushing
            return 0.8
        else:
            # Within reasonable range
            deviation = abs(interactions_per_minute - optimal_frequency) / optimal_frequency
            return 0.2 + (deviation * 0.3)
    
    async def calculate_attention_focus_load(self, attention_metrics: Dict[str, Any]) -> float:
        """Calculate cognitive load from attention focus metrics"""
        focus_score = attention_metrics.get("focus_score", 0.5)
        attention_switches = attention_metrics.get("attention_switches", 0)
        sustained_attention = attention_metrics.get("sustained_attention", 0.5)
        
        # Higher attention switches indicate higher load
        switch_load = min(1.0, attention_switches / 10.0)
        
        # Lower focus and sustained attention indicate higher load
        focus_load = 1.0 - focus_score
        sustained_load = 1.0 - sustained_attention
        
        return (switch_load + focus_load + sustained_load) / 3.0
    
    async def calculate_decision_quality_load(self, decision_metrics: Dict[str, Any]) -> float:
        """Calculate cognitive load from decision quality"""
        decision_time = decision_metrics.get("decision_time", 10.0)
        decision_confidence = decision_metrics.get("confidence", 0.5)
        decision_reversals = decision_metrics.get("reversals", 0)
        
        # Very quick or very slow decisions may indicate load
        if decision_time < 2:
            time_load = 0.7  # Rushed decision
        elif decision_time > 60:
            time_load = 0.8  # Overthinking
        else:
            time_load = 0.3
        
        # Low confidence indicates higher load
        confidence_load = 1.0 - decision_confidence
        
        # Decision reversals indicate uncertainty/load
        reversal_load = min(1.0, decision_reversals / 3.0)
        
        return (time_load + confidence_load + reversal_load) / 3.0


class LoadPatternAnalyzer:
    """Analyzes patterns in cognitive load over time"""
    
    def __init__(self, max_history: int = 500):
        self.load_history: deque = deque(maxlen=max_history)
        self.pattern_cache = {}
        
    async def analyze_load_patterns(self, user_id: str, 
                                  load_measurements: List[CognitiveLoadMeasurement]) -> Dict[str, Any]:
        """Analyze cognitive load patterns for a user"""
        if len(load_measurements) < 3:
            return {"patterns": [], "message": "Insufficient data for pattern analysis"}
        
        patterns = {
            "temporal_patterns": await self._analyze_temporal_patterns(load_measurements),
            "load_trends": await self._analyze_load_trends(load_measurements),
            "threshold_patterns": await self._analyze_threshold_patterns(load_measurements),
            "recovery_patterns": await self._analyze_recovery_patterns(load_measurements),
            "trigger_patterns": await self._analyze_trigger_patterns(load_measurements),
            "performance_correlations": await self._analyze_performance_correlations(load_measurements)
        }
        
        return patterns
    
    async def _analyze_temporal_patterns(self, measurements: List[CognitiveLoadMeasurement]) -> Dict[str, Any]:
        """Analyze temporal patterns in cognitive load"""
        # Group by hour of day
        hourly_loads = {}
        for measurement in measurements:
            hour = measurement.timestamp.hour
            if hour not in hourly_loads:
                hourly_loads[hour] = []
            hourly_loads[hour].append(measurement.overall_load)
        
        # Calculate average load by hour
        hourly_averages = {}
        for hour, loads in hourly_loads.items():
            hourly_averages[hour] = np.mean(loads)
        
        # Find peak and low load hours
        if hourly_averages:
            peak_hour = max(hourly_averages, key=hourly_averages.get)
            low_hour = min(hourly_averages, key=hourly_averages.get)
        else:
            peak_hour = low_hour = None
        
        # Analyze day-of-week patterns
        daily_loads = {}
        for measurement in measurements:
            day = measurement.timestamp.weekday()
            if day not in daily_loads:
                daily_loads[day] = []
            daily_loads[day].append(measurement.overall_load)
        
        daily_averages = {day: np.mean(loads) for day, loads in daily_loads.items()}
        
        return {
            "hourly_patterns": hourly_averages,
            "peak_load_hour": peak_hour,
            "low_load_hour": low_hour,
            "daily_patterns": daily_averages,
            "temporal_stability": self._calculate_temporal_stability(measurements)
        }
    
    async def _analyze_load_trends(self, measurements: List[CognitiveLoadMeasurement]) -> Dict[str, Any]:
        """Analyze trends in cognitive load over time"""
        if len(measurements) < 5:
            return {"trend": "insufficient_data"}
        
        # Recent measurements (last 10)
        recent = measurements[-10:]
        loads = [m.overall_load for m in recent]
        
        # Calculate trend
        x = np.arange(len(loads))
        trend_slope = np.polyfit(x, loads, 1)[0] if len(loads) > 1 else 0
        
        # Categorize trend
        if trend_slope > 0.05:
            trend_direction = "increasing"
        elif trend_slope < -0.05:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Calculate volatility
        volatility = np.std(loads) if len(loads) > 1 else 0
        
        return {
            "trend_direction": trend_direction,
            "trend_slope": float(trend_slope),
            "load_volatility": float(volatility),
            "recent_average": float(np.mean(loads)),
            "load_range": {"min": float(min(loads)), "max": float(max(loads))}
        }
    
    async def _analyze_threshold_patterns(self, measurements: List[CognitiveLoadMeasurement]) -> Dict[str, Any]:
        """Analyze patterns around load thresholds"""
        loads = [m.overall_load for m in measurements]
        
        # Define thresholds
        thresholds = {
            "low_threshold": 0.4,
            "high_threshold": 0.7,
            "overload_threshold": 0.85
        }
        
        # Count threshold crossings
        threshold_stats = {}
        for name, threshold in thresholds.items():
            above_threshold = [load > threshold for load in loads]
            crossings = sum(1 for i in range(1, len(above_threshold)) 
                          if above_threshold[i] != above_threshold[i-1])
            time_above = sum(above_threshold) / len(above_threshold) if above_threshold else 0
            
            threshold_stats[name] = {
                "crossings": crossings,
                "time_above_threshold": time_above,
                "max_consecutive_above": self._max_consecutive(above_threshold, True),
                "max_consecutive_below": self._max_consecutive(above_threshold, False)
            }
        
        return threshold_stats
    
    async def _analyze_recovery_patterns(self, measurements: List[CognitiveLoadMeasurement]) -> Dict[str, Any]:
        """Analyze cognitive load recovery patterns"""
        loads = [m.overall_load for m in measurements]
        
        # Find load spikes and recovery periods
        spikes = []
        for i in range(1, len(loads) - 1):
            if loads[i] > 0.7 and loads[i-1] < 0.6:  # Load spike
                # Find recovery period
                recovery_time = 0
                for j in range(i+1, len(loads)):
                    if loads[j] < 0.5:  # Recovered
                        recovery_time = j - i
                        break
                    if j - i > 10:  # Max recovery window
                        break
                
                spikes.append({
                    "spike_index": i,
                    "peak_load": loads[i],
                    "recovery_time": recovery_time
                })
        
        # Calculate recovery statistics
        if spikes:
            recovery_times = [s["recovery_time"] for s in spikes if s["recovery_time"] > 0]
            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
            recovery_success_rate = len(recovery_times) / len(spikes)
        else:
            avg_recovery_time = 0
            recovery_success_rate = 1.0
        
        return {
            "load_spikes_detected": len(spikes),
            "average_recovery_time": avg_recovery_time,
            "recovery_success_rate": recovery_success_rate,
            "spike_details": spikes[:5]  # Last 5 spikes
        }
    
    async def _analyze_trigger_patterns(self, measurements: List[CognitiveLoadMeasurement]) -> Dict[str, Any]:
        """Analyze what triggers high cognitive load"""
        high_load_measurements = [m for m in measurements if m.overall_load > 0.7]
        
        if not high_load_measurements:
            return {"triggers": [], "message": "No high load episodes detected"}
        
        # Analyze contributing factors for high load episodes
        factor_frequency = {}
        for measurement in high_load_measurements:
            for factor in measurement.contributing_factors:
                factor_frequency[factor] = factor_frequency.get(factor, 0) + 1
        
        # Sort by frequency
        common_triggers = sorted(factor_frequency.items(), key=lambda x: x[1], reverse=True)
        
        # Analyze context patterns
        context_patterns = {}
        for measurement in high_load_measurements:
            for key, value in measurement.context.items():
                if key not in context_patterns:
                    context_patterns[key] = {}
                str_value = str(value)
                context_patterns[key][str_value] = context_patterns[key].get(str_value, 0) + 1
        
        return {
            "common_triggers": common_triggers[:5],
            "trigger_contexts": context_patterns,
            "high_load_frequency": len(high_load_measurements) / len(measurements)
        }
    
    async def _analyze_performance_correlations(self, measurements: List[CognitiveLoadMeasurement]) -> Dict[str, Any]:
        """Analyze correlations between load and performance"""
        # This would require performance data - placeholder implementation
        return {
            "load_performance_correlation": -0.6,  # Negative correlation (higher load, lower performance)
            "optimal_load_range": {"min": 0.3, "max": 0.6},
            "performance_drop_threshold": 0.75
        }
    
    def _calculate_temporal_stability(self, measurements: List[CognitiveLoadMeasurement]) -> float:
        """Calculate temporal stability of cognitive load"""
        loads = [m.overall_load for m in measurements]
        
        if len(loads) < 2:
            return 1.0
        
        # Calculate variance as inverse of stability
        variance = np.var(loads)
        stability = max(0.0, 1.0 - variance)
        
        return float(stability)
    
    def _max_consecutive(self, boolean_list: List[bool], target_value: bool) -> int:
        """Find maximum consecutive occurrences of target value"""
        max_count = 0
        current_count = 0
        
        for value in boolean_list:
            if value == target_value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count


class AdaptationEngine:
    """Generates and manages cognitive load adaptations"""
    
    def __init__(self):
        self.adaptation_rules = self._initialize_adaptation_rules()
        self.adaptation_history: List[LoadAdaptation] = []
        self.effectiveness_tracking = {}
        
    def _initialize_adaptation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize adaptation rules for different load conditions"""
        return {
            "high_load": {
                "ui_simplification": {
                    "reduce_information_density": True,
                    "increase_white_space": True,
                    "simplify_navigation": True,
                    "reduce_color_complexity": True
                },
                "interaction_changes": {
                    "reduce_options": True,
                    "provide_defaults": True,
                    "break_tasks_into_steps": True,
                    "add_progress_indicators": True
                },
                "content_adaptations": {
                    "use_simpler_language": True,
                    "provide_summaries": True,
                    "reduce_detail_level": True,
                    "add_visual_aids": True
                }
            },
            "overload": {
                "immediate_actions": {
                    "pause_non_critical_notifications": True,
                    "simplify_interface_dramatically": True,
                    "provide_single_focus_mode": True,
                    "offer_break_suggestion": True
                },
                "emergency_simplification": {
                    "show_only_essential_elements": True,
                    "disable_animations": True,
                    "use_minimal_color_palette": True,
                    "provide_one_action_at_a_time": True
                }
            },
            "low_load": {
                "engagement_enhancement": {
                    "increase_information_richness": True,
                    "provide_advanced_options": True,
                    "add_supplementary_details": True,
                    "enable_parallel_tasks": True
                },
                "efficiency_improvements": {
                    "reduce_confirmation_steps": True,
                    "enable_keyboard_shortcuts": True,
                    "provide_batch_operations": True,
                    "show_expert_features": True
                }
            },
            "load_spike": {
                "immediate_relief": {
                    "provide_context_reminder": True,
                    "simplify_current_task": True,
                    "offer_help_or_guidance": True,
                    "reduce_time_pressure": True
                }
            },
            "sustained_high_load": {
                "long_term_support": {
                    "suggest_training": True,
                    "provide_learning_resources": True,
                    "adjust_task_complexity": True,
                    "implement_gradual_exposure": True
                }
            }
        }
    
    async def generate_adaptations(self, load_measurement: CognitiveLoadMeasurement,
                                 load_patterns: Dict[str, Any],
                                 user_profile: CognitiveProfile) -> List[LoadAdaptation]:
        """Generate adaptations based on cognitive load assessment"""
        adaptations = []
        
        # Determine adaptation triggers
        triggers = await self._identify_adaptation_triggers(
            load_measurement, load_patterns, user_profile
        )
        
        # Generate adaptations for each trigger
        for trigger in triggers:
            adaptation = await self._create_adaptation(
                trigger, load_measurement, user_profile
            )
            if adaptation:
                adaptations.append(adaptation)
        
        # Store adaptations
        self.adaptation_history.extend(adaptations)
        
        return adaptations
    
    async def _identify_adaptation_triggers(self, load_measurement: CognitiveLoadMeasurement,
                                          load_patterns: Dict[str, Any],
                                          user_profile: CognitiveProfile) -> List[AdaptationTrigger]:
        """Identify what triggers adaptations are needed"""
        triggers = []
        current_load = load_measurement.overall_load
        
        # High load trigger
        if current_load > 0.7:
            triggers.append(AdaptationTrigger.HIGH_LOAD_DETECTED)
        
        # Overload trigger
        if current_load > 0.85:
            triggers.append(AdaptationTrigger.OVERLOAD_DETECTED)
        
        # Low load trigger
        if current_load < 0.3:
            triggers.append(AdaptationTrigger.LOW_LOAD_DETECTED)
        
        # Load spike trigger
        trend = load_patterns.get("load_trends", {})
        if trend.get("trend_slope", 0) > 0.15:  # Rapid increase
            triggers.append(AdaptationTrigger.RAPID_LOAD_INCREASE)
            if current_load > 0.6:
                triggers.append(AdaptationTrigger.LOAD_SPIKE)
        
        # Sustained high load trigger
        threshold_patterns = load_patterns.get("threshold_patterns", {})
        high_threshold_time = threshold_patterns.get("high_threshold", {}).get("time_above_threshold", 0)
        if high_threshold_time > 0.7:  # Above high threshold 70% of time
            triggers.append(AdaptationTrigger.SUSTAINED_HIGH_LOAD)
        
        # Attention fragmentation trigger
        attention_load = load_measurement.indicator_scores.get(LoadIndicator.ATTENTION_FOCUS, 0.5)
        task_switching_load = load_measurement.indicator_scores.get(LoadIndicator.TASK_SWITCHING, 0.5)
        if attention_load > 0.7 or task_switching_load > 0.7:
            triggers.append(AdaptationTrigger.ATTENTION_FRAGMENTATION)
        
        # Performance degradation trigger
        decision_quality_load = load_measurement.indicator_scores.get(LoadIndicator.DECISION_QUALITY, 0.5)
        error_rate_load = load_measurement.indicator_scores.get(LoadIndicator.ERROR_RATE, 0.5)
        if decision_quality_load > 0.7 or error_rate_load > 0.7:
            triggers.append(AdaptationTrigger.PERFORMANCE_DEGRADATION)
        
        return triggers
    
    async def _create_adaptation(self, trigger: AdaptationTrigger,
                               load_measurement: CognitiveLoadMeasurement,
                               user_profile: CognitiveProfile) -> Optional[LoadAdaptation]:
        """Create specific adaptation for a trigger"""
        adaptation_id = str(uuid4())
        
        if trigger == AdaptationTrigger.HIGH_LOAD_DETECTED:
            return await self._create_high_load_adaptation(
                adaptation_id, load_measurement, user_profile
            )
        elif trigger == AdaptationTrigger.OVERLOAD_DETECTED:
            return await self._create_overload_adaptation(
                adaptation_id, load_measurement, user_profile
            )
        elif trigger == AdaptationTrigger.LOW_LOAD_DETECTED:
            return await self._create_low_load_adaptation(
                adaptation_id, load_measurement, user_profile
            )
        elif trigger == AdaptationTrigger.LOAD_SPIKE:
            return await self._create_load_spike_adaptation(
                adaptation_id, load_measurement, user_profile
            )
        elif trigger == AdaptationTrigger.SUSTAINED_HIGH_LOAD:
            return await self._create_sustained_load_adaptation(
                adaptation_id, load_measurement, user_profile
            )
        elif trigger == AdaptationTrigger.ATTENTION_FRAGMENTATION:
            return await self._create_attention_adaptation(
                adaptation_id, load_measurement, user_profile
            )
        elif trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
            return await self._create_performance_adaptation(
                adaptation_id, load_measurement, user_profile
            )
        
        return None
    
    async def _create_high_load_adaptation(self, adaptation_id: str,
                                         load_measurement: CognitiveLoadMeasurement,
                                         user_profile: CognitiveProfile) -> LoadAdaptation:
        """Create adaptation for high cognitive load"""
        rules = self.adaptation_rules["high_load"]
        
        return LoadAdaptation(
            adaptation_id=adaptation_id,
            trigger=AdaptationTrigger.HIGH_LOAD_DETECTED,
            adaptation_type="load_reduction",
            target_load_reduction=0.2,
            recommended_actions=[
                "Simplify user interface",
                "Reduce information density",
                "Provide step-by-step guidance",
                "Break complex tasks into smaller parts"
            ],
            ui_modifications=rules["ui_simplification"],
            interaction_changes=rules["interaction_changes"],
            priority="high",
            estimated_effectiveness=0.7,
            created_at=datetime.utcnow()
        )
    
    async def _create_overload_adaptation(self, adaptation_id: str,
                                        load_measurement: CognitiveLoadMeasurement,
                                        user_profile: CognitiveProfile) -> LoadAdaptation:
        """Create adaptation for cognitive overload"""
        rules = self.adaptation_rules["overload"]
        
        return LoadAdaptation(
            adaptation_id=adaptation_id,
            trigger=AdaptationTrigger.OVERLOAD_DETECTED,
            adaptation_type="emergency_simplification",
            target_load_reduction=0.4,
            recommended_actions=[
                "Activate emergency simplification mode",
                "Pause all non-essential notifications",
                "Suggest taking a break",
                "Provide single-focus interface",
                "Offer human assistance"
            ],
            ui_modifications={**rules["immediate_actions"], **rules["emergency_simplification"]},
            interaction_changes={
                "disable_multitasking": True,
                "provide_single_action_flow": True,
                "add_stress_relief_options": True
            },
            priority="immediate",
            estimated_effectiveness=0.8,
            created_at=datetime.utcnow()
        )
    
    async def _create_low_load_adaptation(self, adaptation_id: str,
                                        load_measurement: CognitiveLoadMeasurement,
                                        user_profile: CognitiveProfile) -> LoadAdaptation:
        """Create adaptation for low cognitive load"""
        rules = self.adaptation_rules["low_load"]
        
        return LoadAdaptation(
            adaptation_id=adaptation_id,
            trigger=AdaptationTrigger.LOW_LOAD_DETECTED,
            adaptation_type="engagement_enhancement",
            target_load_reduction=-0.2,  # Negative because we want to increase engagement
            recommended_actions=[
                "Increase information richness",
                "Provide advanced features",
                "Enable parallel task processing",
                "Reduce confirmation steps"
            ],
            ui_modifications=rules["engagement_enhancement"],
            interaction_changes=rules["efficiency_improvements"],
            priority="medium",
            estimated_effectiveness=0.6,
            created_at=datetime.utcnow()
        )
    
    async def _create_load_spike_adaptation(self, adaptation_id: str,
                                          load_measurement: CognitiveLoadMeasurement,
                                          user_profile: CognitiveProfile) -> LoadAdaptation:
        """Create adaptation for cognitive load spike"""
        rules = self.adaptation_rules["load_spike"]
        
        return LoadAdaptation(
            adaptation_id=adaptation_id,
            trigger=AdaptationTrigger.LOAD_SPIKE,
            adaptation_type="immediate_relief",
            target_load_reduction=0.3,
            recommended_actions=[
                "Provide context reminder",
                "Simplify current task",
                "Offer guided assistance",
                "Reduce time pressure"
            ],
            ui_modifications=rules["immediate_relief"],
            interaction_changes={
                "extend_timeouts": True,
                "provide_undo_options": True,
                "add_help_prompts": True
            },
            priority="high",
            estimated_effectiveness=0.75,
            created_at=datetime.utcnow()
        )
    
    async def _create_sustained_load_adaptation(self, adaptation_id: str,
                                              load_measurement: CognitiveLoadMeasurement,
                                              user_profile: CognitiveProfile) -> LoadAdaptation:
        """Create adaptation for sustained high load"""
        rules = self.adaptation_rules["sustained_high_load"]
        
        return LoadAdaptation(
            adaptation_id=adaptation_id,
            trigger=AdaptationTrigger.SUSTAINED_HIGH_LOAD,
            adaptation_type="long_term_support",
            target_load_reduction=0.25,
            recommended_actions=[
                "Suggest training or learning resources",
                "Adjust task complexity gradually",
                "Implement gradual skill building",
                "Provide performance feedback"
            ],
            ui_modifications=rules["long_term_support"],
            interaction_changes={
                "adaptive_difficulty": True,
                "progressive_disclosure": True,
                "learning_mode": True
            },
            priority="medium",
            estimated_effectiveness=0.6,
            created_at=datetime.utcnow()
        )
    
    async def _create_attention_adaptation(self, adaptation_id: str,
                                         load_measurement: CognitiveLoadMeasurement,
                                         user_profile: CognitiveProfile) -> LoadAdaptation:
        """Create adaptation for attention fragmentation"""
        return LoadAdaptation(
            adaptation_id=adaptation_id,
            trigger=AdaptationTrigger.ATTENTION_FRAGMENTATION,
            adaptation_type="attention_focus",
            target_load_reduction=0.25,
            recommended_actions=[
                "Enable focus mode",
                "Reduce distractions",
                "Limit concurrent tasks",
                "Provide attention anchors"
            ],
            ui_modifications={
                "hide_non_essential_elements": True,
                "reduce_animations": True,
                "minimize_notifications": True,
                "provide_focus_indicators": True
            },
            interaction_changes={
                "limit_task_switching": True,
                "provide_attention_breaks": True,
                "sequential_task_flow": True
            },
            priority="high",
            estimated_effectiveness=0.7,
            created_at=datetime.utcnow()
        )
    
    async def _create_performance_adaptation(self, adaptation_id: str,
                                           load_measurement: CognitiveLoadMeasurement,
                                           user_profile: CognitiveProfile) -> LoadAdaptation:
        """Create adaptation for performance degradation"""
        return LoadAdaptation(
            adaptation_id=adaptation_id,
            trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
            adaptation_type="performance_support",
            target_load_reduction=0.3,
            recommended_actions=[
                "Provide decision support",
                "Add error prevention",
                "Increase confirmation steps",
                "Offer alternative approaches"
            ],
            ui_modifications={
                "add_safety_checks": True,
                "provide_decision_aids": True,
                "highlight_important_info": True,
                "add_confirmation_dialogs": True
            },
            interaction_changes={
                "slow_down_interactions": True,
                "provide_undo_options": True,
                "add_review_steps": True,
                "enable_assistance_mode": True
            },
            priority="high",
            estimated_effectiveness=0.65,
            created_at=datetime.utcnow()
        )


class CognitiveLoadMonitor:
    """Main cognitive load monitoring system"""
    
    def __init__(self):
        self.indicator_calculator = LoadIndicatorCalculator()
        self.pattern_analyzer = LoadPatternAnalyzer()
        self.adaptation_engine = AdaptationEngine()
        
        self.user_profiles: Dict[str, CognitiveProfile] = {}
        self.load_measurements: Dict[str, List[CognitiveLoadMeasurement]] = {}
        self.active_adaptations: Dict[str, List[LoadAdaptation]] = {}
        
        # Monitoring thresholds
        self.monitoring_thresholds = {
            "high_load_threshold": 0.7,
            "overload_threshold": 0.85,
            "low_load_threshold": 0.3,
            "spike_threshold": 0.15,  # Increase rate
            "sustained_threshold": 0.7   # Time above high load
        }
    
    async def initialize_user_profile(self, user_id: str, 
                                     initial_data: Dict[str, Any] = None) -> CognitiveProfile:
        """Initialize cognitive profile for a user"""
        if initial_data is None:
            initial_data = {}
        
        profile = CognitiveProfile(
            user_id=user_id,
            baseline_load=initial_data.get("baseline_load", 0.4),
            load_capacity=initial_data.get("load_capacity", 0.8),
            load_patterns=initial_data.get("load_patterns", {}),
            attention_span=initial_data.get("attention_span", 15.0),  # minutes
            task_switching_tolerance=initial_data.get("task_switching_tolerance", 0.7),
            preferred_information_density=initial_data.get("preferred_density", "medium"),
            cognitive_style=initial_data.get("cognitive_style", "mixed"),
            load_recovery_rate=initial_data.get("recovery_rate", 0.8),
            stress_indicators=initial_data.get("stress_indicators", []),
            optimal_performance_conditions=initial_data.get("optimal_conditions", {}),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.user_profiles[user_id] = profile
        self.load_measurements[user_id] = []
        self.active_adaptations[user_id] = []
        
        logger.info(f"Initialized cognitive profile for user {user_id}")
        return profile
    
    async def measure_cognitive_load(self, user_id: str, session_id: str,
                                   interaction_data: Dict[str, Any],
                                   context: Dict[str, Any] = None) -> CognitiveLoadMeasurement:
        """Measure current cognitive load for a user"""
        if user_id not in self.user_profiles:
            await self.initialize_user_profile(user_id)
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate individual indicator scores
        indicator_scores = {}
        
        # Response time load
        if "response_time" in interaction_data:
            indicator_scores[LoadIndicator.RESPONSE_TIME] = await self.indicator_calculator.calculate_response_time_load(
                interaction_data["response_time"], user_profile
            )
        
        # Error rate load
        if "recent_errors" in interaction_data:
            indicator_scores[LoadIndicator.ERROR_RATE] = await self.indicator_calculator.calculate_error_rate_load(
                interaction_data["recent_errors"], interaction_data.get("total_interactions", 1)
            )
        
        # Task switching load
        if "task_switches" in interaction_data:
            indicator_scores[LoadIndicator.TASK_SWITCHING] = await self.indicator_calculator.calculate_task_switching_load(
                interaction_data["task_switches"], interaction_data.get("time_window", 10.0)
            )
        
        # Multitasking load
        if "concurrent_tasks" in interaction_data:
            indicator_scores[LoadIndicator.MULTITASKING] = await self.indicator_calculator.calculate_multitasking_load(
                interaction_data["concurrent_tasks"], user_profile
            )
        
        # Message complexity load
        if "message" in interaction_data:
            indicator_scores[LoadIndicator.MESSAGE_COMPLEXITY] = await self.indicator_calculator.calculate_message_complexity_load(
                interaction_data["message"], user_profile
            )
        
        # Interaction frequency load
        if "interactions_per_minute" in interaction_data:
            indicator_scores[LoadIndicator.INTERACTION_FREQUENCY] = await self.indicator_calculator.calculate_interaction_frequency_load(
                interaction_data["interactions_per_minute"], user_profile
            )
        
        # Attention focus load
        if "attention_metrics" in interaction_data:
            indicator_scores[LoadIndicator.ATTENTION_FOCUS] = await self.indicator_calculator.calculate_attention_focus_load(
                interaction_data["attention_metrics"]
            )
        
        # Decision quality load
        if "decision_metrics" in interaction_data:
            indicator_scores[LoadIndicator.DECISION_QUALITY] = await self.indicator_calculator.calculate_decision_quality_load(
                interaction_data["decision_metrics"]
            )
        
        # Calculate overall cognitive load
        overall_load = await self._calculate_overall_load(indicator_scores)
        
        # Determine load level
        load_level = self._categorize_load_level(overall_load)
        
        # Identify contributing factors
        contributing_factors = await self._identify_contributing_factors(
            indicator_scores, interaction_data, context or {}
        )
        
        # Create measurement
        measurement = CognitiveLoadMeasurement(
            measurement_id=str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            overall_load=overall_load,
            load_level=load_level,
            indicator_scores=indicator_scores,
            contributing_factors=contributing_factors,
            context=context or {},
            timestamp=datetime.utcnow(),
            measurement_confidence=await self._calculate_measurement_confidence(indicator_scores)
        )
        
        # Store measurement
        self.load_measurements[user_id].append(measurement)
        
        # Keep history manageable
        if len(self.load_measurements[user_id]) > 1000:
            self.load_measurements[user_id] = self.load_measurements[user_id][-500:]
        
        logger.info(f"Measured cognitive load for user {user_id}: {overall_load:.2f} ({load_level.value})")
        
        return measurement
    
    async def _calculate_overall_load(self, indicator_scores: Dict[LoadIndicator, float]) -> float:
        """Calculate overall cognitive load from individual indicators"""
        if not indicator_scores:
            return 0.5  # Neutral default
        
        # Weighted average of available indicators
        total_weight = 0
        weighted_sum = 0
        
        for indicator, score in indicator_scores.items():
            weight = self.indicator_calculator.indicator_weights.get(indicator, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        overall_load = weighted_sum / total_weight
        return max(0.0, min(1.0, overall_load))
    
    def _categorize_load_level(self, overall_load: float) -> CognitiveLoadLevel:
        """Categorize overall load into level"""
        if overall_load < 0.2:
            return CognitiveLoadLevel.VERY_LOW
        elif overall_load < 0.4:
            return CognitiveLoadLevel.LOW
        elif overall_load < 0.6:
            return CognitiveLoadLevel.MODERATE
        elif overall_load < 0.8:
            return CognitiveLoadLevel.HIGH
        else:
            return CognitiveLoadLevel.OVERLOAD
    
    async def _identify_contributing_factors(self, indicator_scores: Dict[LoadIndicator, float],
                                           interaction_data: Dict[str, Any],
                                           context: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to cognitive load"""
        factors = []
        
        # Check which indicators are high
        for indicator, score in indicator_scores.items():
            if score > 0.6:
                if indicator == LoadIndicator.RESPONSE_TIME:
                    factors.append("slow_response_time")
                elif indicator == LoadIndicator.ERROR_RATE:
                    factors.append("high_error_rate")
                elif indicator == LoadIndicator.TASK_SWITCHING:
                    factors.append("frequent_task_switching")
                elif indicator == LoadIndicator.MULTITASKING:
                    factors.append("multitasking_overload")
                elif indicator == LoadIndicator.MESSAGE_COMPLEXITY:
                    factors.append("complex_information")
                elif indicator == LoadIndicator.INTERACTION_FREQUENCY:
                    factors.append("interaction_pacing_issues")
                elif indicator == LoadIndicator.ATTENTION_FOCUS:
                    factors.append("attention_fragmentation")
                elif indicator == LoadIndicator.DECISION_QUALITY:
                    factors.append("decision_difficulty")
        
        # Context-based factors
        if context.get("task_complexity") == "high":
            factors.append("high_task_complexity")
        
        if context.get("time_pressure") == "high":
            factors.append("time_pressure")
        
        if context.get("interruptions", 0) > 2:
            factors.append("frequent_interruptions")
        
        if interaction_data.get("concurrent_tasks", 1) > 2:
            factors.append("task_overload")
        
        return factors
    
    async def _calculate_measurement_confidence(self, indicator_scores: Dict[LoadIndicator, float]) -> float:
        """Calculate confidence in the load measurement"""
        # Confidence is higher with more indicators
        num_indicators = len(indicator_scores)
        indicator_confidence = min(1.0, num_indicators / 8.0)  # 8 total indicators
        
        # Confidence is higher when indicators agree
        if num_indicators > 1:
            score_variance = np.var(list(indicator_scores.values()))
            agreement_confidence = max(0.0, 1.0 - score_variance)
        else:
            agreement_confidence = 0.5
        
        return (indicator_confidence + agreement_confidence) / 2.0
    
    async def analyze_and_adapt(self, user_id: str) -> Dict[str, Any]:
        """Analyze cognitive load patterns and generate adaptations"""
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}
        
        user_profile = self.user_profiles[user_id]
        user_measurements = self.load_measurements.get(user_id, [])
        
        if len(user_measurements) < 3:
            return {"message": "Insufficient data for analysis"}
        
        # Analyze patterns
        load_patterns = await self.pattern_analyzer.analyze_load_patterns(
            user_id, user_measurements
        )
        
        # Get latest measurement
        latest_measurement = user_measurements[-1]
        
        # Generate adaptations
        adaptations = await self.adaptation_engine.generate_adaptations(
            latest_measurement, load_patterns, user_profile
        )
        
        # Update active adaptations
        self.active_adaptations[user_id] = adaptations
        
        # Update user profile based on patterns
        await self._update_user_profile(user_id, load_patterns)
        
        return {
            "user_id": user_id,
            "current_load": latest_measurement.overall_load,
            "load_level": latest_measurement.load_level.value,
            "load_patterns": load_patterns,
            "adaptations": [asdict(adaptation) for adaptation in adaptations],
            "profile_updated": True
        }
    
    async def _update_user_profile(self, user_id: str, load_patterns: Dict[str, Any]):
        """Update user profile based on observed patterns"""
        user_profile = self.user_profiles[user_id]
        
        # Update load patterns
        user_profile.load_patterns.update(load_patterns.get("temporal_patterns", {}))
        
        # Update baseline load
        recent_loads = [m.overall_load for m in self.load_measurements[user_id][-20:]]
        if recent_loads:
            user_profile.baseline_load = np.mean(recent_loads)
        
        # Update load capacity (highest sustainable load)
        high_loads = [load for load in recent_loads if load > 0.6]
        if high_loads:
            user_profile.load_capacity = max(high_loads)
        
        # Update recovery rate
        recovery_patterns = load_patterns.get("recovery_patterns", {})
        if "average_recovery_time" in recovery_patterns:
            recovery_time = recovery_patterns["average_recovery_time"]
            user_profile.load_recovery_rate = max(0.1, 1.0 - (recovery_time / 10.0))
        
        user_profile.updated_at = datetime.utcnow()
    
    async def get_user_load_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive load status for a user"""
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}
        
        user_profile = self.user_profiles[user_id]
        user_measurements = self.load_measurements.get(user_id, [])
        active_adaptations = self.active_adaptations.get(user_id, [])
        
        # Current status
        current_status = {}
        if user_measurements:
            latest = user_measurements[-1]
            current_status = {
                "current_load": latest.overall_load,
                "load_level": latest.load_level.value,
                "indicator_breakdown": {indicator.value: score 
                                       for indicator, score in latest.indicator_scores.items()},
                "contributing_factors": latest.contributing_factors,
                "measurement_time": latest.timestamp.isoformat()
            }
        
        # Recent trends
        recent_trends = {}
        if len(user_measurements) >= 5:
            recent_loads = [m.overall_load for m in user_measurements[-10:]]
            recent_trends = {
                "average_load": np.mean(recent_loads),
                "load_trend": "increasing" if recent_loads[-1] > recent_loads[0] else "decreasing",
                "load_volatility": np.std(recent_loads),
                "peak_load": max(recent_loads),
                "minimum_load": min(recent_loads)
            }
        
        return {
            "user_id": user_id,
            "current_status": current_status,
            "recent_trends": recent_trends,
            "user_profile": {
                "baseline_load": user_profile.baseline_load,
                "load_capacity": user_profile.load_capacity,
                "attention_span": user_profile.attention_span,
                "cognitive_style": user_profile.cognitive_style,
                "preferred_density": user_profile.preferred_information_density
            },
            "active_adaptations": len(active_adaptations),
            "total_measurements": len(user_measurements),
            "profile_last_updated": user_profile.updated_at.isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system monitoring status"""
        total_users = len(self.user_profiles)
        total_measurements = sum(len(measurements) for measurements in self.load_measurements.values())
        total_adaptations = sum(len(adaptations) for adaptations in self.active_adaptations.values())
        
        # Calculate aggregate statistics
        all_recent_loads = []
        for user_measurements in self.load_measurements.values():
            if user_measurements:
                all_recent_loads.extend([m.overall_load for m in user_measurements[-5:]])
        
        aggregate_stats = {}
        if all_recent_loads:
            aggregate_stats = {
                "average_load": np.mean(all_recent_loads),
                "median_load": np.median(all_recent_loads),
                "load_std_deviation": np.std(all_recent_loads),
                "high_load_users": sum(1 for load in all_recent_loads if load > 0.7),
                "overload_users": sum(1 for load in all_recent_loads if load > 0.85)
            }
        
        # Load level distribution
        load_distribution = {level.value: 0 for level in CognitiveLoadLevel}
        for user_measurements in self.load_measurements.values():
            if user_measurements:
                latest_level = user_measurements[-1].load_level.value
                load_distribution[latest_level] += 1
        
        return {
            "total_users": total_users,
            "total_measurements": total_measurements,
            "total_active_adaptations": total_adaptations,
            "aggregate_statistics": aggregate_stats,
            "load_level_distribution": load_distribution,
            "monitoring_thresholds": self.monitoring_thresholds,
            "system_health": "operational"
        }
