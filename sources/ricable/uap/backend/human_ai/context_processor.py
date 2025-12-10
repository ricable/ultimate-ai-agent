"""
Advanced Context Processing for Human-AI Collaboration
Handles multi-dimensional context awareness and adaptive processing.
"""

import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import uuid4
import re
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context information"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SOCIAL = "social"
    TASK = "task"
    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    ENVIRONMENTAL = "environmental"
    BEHAVIORAL = "behavioral"


class ContextRelevance(Enum):
    """Context relevance levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"


@dataclass
class ContextFeature:
    """Individual context feature"""
    feature_id: str
    feature_type: ContextType
    value: Any
    confidence: float  # 0-1
    relevance: ContextRelevance
    temporal_weight: float  # Decay factor for temporal relevance
    source: str
    extracted_at: datetime
    expires_at: Optional[datetime]


@dataclass
class ContextPattern:
    """Detected context pattern"""
    pattern_id: str
    pattern_name: str
    features: List[str]  # Feature IDs that compose this pattern
    confidence: float
    frequency: int  # How often this pattern occurs
    last_seen: datetime
    associated_outcomes: List[str]
    adaptation_triggers: List[str]


@dataclass
class ContextPrediction:
    """Prediction about future context state"""
    prediction_id: str
    predicted_context: Dict[str, Any]
    probability: float
    time_horizon: timedelta
    trigger_conditions: List[str]
    confidence_interval: Tuple[float, float]
    created_at: datetime


class TemporalContextAnalyzer:
    """Analyzes temporal patterns in context"""
    
    def __init__(self, max_history: int = 1000):
        self.context_timeline: deque = deque(maxlen=max_history)
        self.temporal_patterns: Dict[str, ContextPattern] = {}
        self.time_decay_factor = 0.95  # How quickly context relevance decays
        
    async def analyze_temporal_context(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal aspects of context"""
        now = datetime.utcnow()
        
        # Add to timeline
        self.context_timeline.append({
            "timestamp": now,
            "context": current_context
        })
        
        temporal_analysis = {
            "current_time": now.isoformat(),
            "time_since_start": self._calculate_session_duration(),
            "interaction_frequency": self._calculate_interaction_frequency(),
            "temporal_patterns": await self._detect_temporal_patterns(),
            "time_of_day_context": self._analyze_time_of_day(now),
            "session_phase": self._determine_session_phase(),
            "temporal_stability": self._calculate_temporal_stability(),
            "predicted_next_interaction": await self._predict_next_interaction()
        }
        
        return temporal_analysis
    
    def _calculate_session_duration(self) -> float:
        """Calculate current session duration in minutes"""
        if len(self.context_timeline) < 2:
            return 0.0
        
        start_time = self.context_timeline[0]["timestamp"]
        current_time = self.context_timeline[-1]["timestamp"]
        
        return (current_time - start_time).total_seconds() / 60.0
    
    def _calculate_interaction_frequency(self) -> Dict[str, float]:
        """Calculate interaction frequency metrics"""
        if len(self.context_timeline) < 2:
            return {"interactions_per_minute": 0.0, "average_gap": 0.0}
        
        # Calculate gaps between interactions
        gaps = []
        for i in range(1, len(self.context_timeline)):
            gap = (self.context_timeline[i]["timestamp"] - 
                  self.context_timeline[i-1]["timestamp"]).total_seconds()
            gaps.append(gap)
        
        avg_gap = np.mean(gaps) if gaps else 0.0
        interactions_per_minute = 60.0 / avg_gap if avg_gap > 0 else 0.0
        
        return {
            "interactions_per_minute": interactions_per_minute,
            "average_gap_seconds": avg_gap,
            "gap_variance": np.var(gaps) if gaps else 0.0
        }
    
    async def _detect_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Detect recurring temporal patterns"""
        patterns = []
        
        # Analyze hourly patterns
        hourly_activity = defaultdict(int)
        for entry in self.context_timeline:
            hour = entry["timestamp"].hour
            hourly_activity[hour] += 1
        
        # Find peak activity hours
        if hourly_activity:
            peak_hour = max(hourly_activity, key=hourly_activity.get)
            patterns.append({
                "type": "peak_activity_hour",
                "hour": peak_hour,
                "activity_count": hourly_activity[peak_hour],
                "confidence": min(1.0, hourly_activity[peak_hour] / 10.0)
            })
        
        # Analyze interaction rhythm
        if len(self.context_timeline) > 5:
            recent_gaps = [
                (self.context_timeline[i]["timestamp"] - 
                 self.context_timeline[i-1]["timestamp"]).total_seconds()
                for i in range(-5, 0)
            ]
            
            avg_rhythm = np.mean(recent_gaps)
            rhythm_stability = 1.0 - (np.std(recent_gaps) / avg_rhythm) if avg_rhythm > 0 else 0.0
            
            patterns.append({
                "type": "interaction_rhythm",
                "average_gap": avg_rhythm,
                "stability": max(0.0, min(1.0, rhythm_stability)),
                "confidence": 0.7
            })
        
        return patterns
    
    def _analyze_time_of_day(self, timestamp: datetime) -> Dict[str, Any]:
        """Analyze time-of-day context"""
        hour = timestamp.hour
        
        # Categorize time periods
        if 6 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"
        
        # Determine likely energy/attention level
        attention_map = {
            "morning": 0.8,
            "afternoon": 0.6,
            "evening": 0.5,
            "night": 0.3
        }
        
        return {
            "hour": hour,
            "period": period,
            "expected_attention_level": attention_map[period],
            "is_business_hours": 9 <= hour <= 17,
            "is_peak_productivity": 9 <= hour <= 11 or 14 <= hour <= 16
        }
    
    def _determine_session_phase(self) -> str:
        """Determine current phase of interaction session"""
        duration = self._calculate_session_duration()
        interaction_count = len(self.context_timeline)
        
        if duration < 2 and interaction_count < 3:
            return "initiation"
        elif duration < 10 and interaction_count < 10:
            return "exploration"
        elif duration < 30:
            return "collaboration"
        else:
            return "extended_session"
    
    def _calculate_temporal_stability(self) -> float:
        """Calculate stability of temporal context"""
        if len(self.context_timeline) < 3:
            return 1.0
        
        # Analyze consistency of interaction patterns
        recent_entries = list(self.context_timeline)[-10:]  # Last 10 interactions
        
        # Calculate variance in interaction timing
        gaps = [
            (recent_entries[i]["timestamp"] - recent_entries[i-1]["timestamp"]).total_seconds()
            for i in range(1, len(recent_entries))
        ]
        
        if not gaps:
            return 1.0
        
        mean_gap = np.mean(gaps)
        variance = np.var(gaps)
        
        # Stability is inverse of coefficient of variation
        stability = 1.0 - min(1.0, np.sqrt(variance) / mean_gap) if mean_gap > 0 else 1.0
        
        return max(0.0, stability)
    
    async def _predict_next_interaction(self) -> Dict[str, Any]:
        """Predict when next interaction might occur"""
        if len(self.context_timeline) < 3:
            return {"prediction": "insufficient_data"}
        
        # Calculate average gap from recent interactions
        recent_gaps = [
            (self.context_timeline[i]["timestamp"] - 
             self.context_timeline[i-1]["timestamp"]).total_seconds()
            for i in range(-min(5, len(self.context_timeline)-1), 0)
        ]
        
        avg_gap = np.mean(recent_gaps)
        std_gap = np.std(recent_gaps)
        
        # Predict next interaction time
        last_interaction = self.context_timeline[-1]["timestamp"]
        predicted_next = last_interaction + timedelta(seconds=avg_gap)
        
        return {
            "predicted_time": predicted_next.isoformat(),
            "expected_gap_seconds": avg_gap,
            "confidence_interval": (avg_gap - std_gap, avg_gap + std_gap),
            "confidence": min(1.0, 5.0 / (std_gap + 1.0))  # Higher confidence with lower variance
        }


class CognitiveContextAnalyzer:
    """Analyzes cognitive context and mental model"""
    
    def __init__(self):
        self.cognitive_patterns: Dict[str, ContextPattern] = {}
        self.mental_model: Dict[str, Any] = {}
        self.cognitive_load_history: List[Dict[str, Any]] = []
        
    async def analyze_cognitive_context(self, interaction_data: Dict[str, Any],
                                      user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cognitive aspects of current context"""
        cognitive_analysis = {
            "cognitive_load": await self._assess_cognitive_load(interaction_data),
            "mental_model_state": await self._analyze_mental_model(interaction_data, user_profile),
            "cognitive_style_indicators": self._detect_cognitive_style(interaction_data),
            "attention_indicators": self._analyze_attention_patterns(interaction_data),
            "comprehension_level": await self._assess_comprehension(interaction_data),
            "decision_making_style": self._analyze_decision_style(interaction_data),
            "learning_indicators": self._detect_learning_signals(interaction_data)
        }
        
        return cognitive_analysis
    
    async def _assess_cognitive_load(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current cognitive load level"""
        load_indicators = {
            "message_complexity": self._calculate_message_complexity(interaction_data),
            "response_time": interaction_data.get("response_time", 0),
            "error_rate": self._calculate_recent_error_rate(),
            "task_switching": self._detect_task_switching(interaction_data),
            "multitasking_signals": self._detect_multitasking(interaction_data)
        }
        
        # Calculate overall cognitive load
        load_score = self._calculate_cognitive_load_score(load_indicators)
        
        # Store in history
        self.cognitive_load_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "load_score": load_score,
            "indicators": load_indicators
        })
        
        return {
            "current_load": load_score,
            "load_level": self._categorize_load_level(load_score),
            "load_trend": self._calculate_load_trend(),
            "indicators": load_indicators,
            "recommendations": self._generate_load_recommendations(load_score)
        }
    
    def _calculate_message_complexity(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate complexity of user's message"""
        message = interaction_data.get("message", "")
        
        if not message:
            return 0.0
        
        # Various complexity indicators
        word_count = len(message.split())
        sentence_count = len(re.split(r'[.!?]+', message))
        avg_word_length = np.mean([len(word) for word in message.split()])
        unique_words = len(set(message.lower().split()))
        
        # Normalize complexity score
        complexity = (
            min(1.0, word_count / 50.0) * 0.3 +
            min(1.0, sentence_count / 10.0) * 0.2 +
            min(1.0, avg_word_length / 10.0) * 0.2 +
            min(1.0, unique_words / word_count) * 0.3 if word_count > 0 else 0.0
        )
        
        return complexity
    
    def _calculate_recent_error_rate(self) -> float:
        """Calculate recent error rate (placeholder)"""
        # In a real implementation, this would track actual errors
        return 0.1  # Placeholder error rate
    
    def _detect_task_switching(self, interaction_data: Dict[str, Any]) -> bool:
        """Detect if user is switching between tasks"""
        current_topic = interaction_data.get("topic", "")
        previous_topic = getattr(self, "_last_topic", "")
        
        self._last_topic = current_topic
        
        if not current_topic or not previous_topic:
            return False
        
        # Simple topic similarity check
        common_words = set(current_topic.lower().split()) & set(previous_topic.lower().split())
        similarity = len(common_words) / max(len(current_topic.split()), len(previous_topic.split()))
        
        return similarity < 0.3  # Low similarity indicates task switching
    
    def _detect_multitasking(self, interaction_data: Dict[str, Any]) -> bool:
        """Detect signals of multitasking"""
        # Look for indicators like long response times, fragmented messages, etc.
        response_time = interaction_data.get("response_time", 0)
        message = interaction_data.get("message", "")
        
        # Indicators of multitasking
        long_response = response_time > 60  # More than 1 minute
        fragmented_message = len(message.split()) < 3 and response_time > 10
        
        return long_response or fragmented_message
    
    def _calculate_cognitive_load_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall cognitive load score"""
        # Weighted combination of indicators
        weights = {
            "message_complexity": 0.2,
            "response_time": 0.2,
            "error_rate": 0.3,
            "task_switching": 0.15,
            "multitasking_signals": 0.15
        }
        
        score = 0.0
        for indicator, weight in weights.items():
            value = indicators.get(indicator, 0)
            
            # Normalize different types of indicators
            if indicator == "response_time":
                normalized_value = min(1.0, value / 120.0)  # 2 minutes max
            elif indicator in ["task_switching", "multitasking_signals"]:
                normalized_value = 1.0 if value else 0.0
            else:
                normalized_value = float(value)
            
            score += normalized_value * weight
        
        return min(1.0, score)
    
    def _categorize_load_level(self, load_score: float) -> CognitiveLoadLevel:
        """Categorize cognitive load level"""
        if load_score < 0.3:
            return CognitiveLoadLevel.LOW
        elif load_score < 0.6:
            return CognitiveLoadLevel.MODERATE
        elif load_score < 0.8:
            return CognitiveLoadLevel.HIGH
        else:
            return CognitiveLoadLevel.OVERLOAD
    
    def _calculate_load_trend(self) -> str:
        """Calculate trend in cognitive load"""
        if len(self.cognitive_load_history) < 3:
            return "stable"
        
        recent_scores = [entry["load_score"] for entry in self.cognitive_load_history[-3:]]
        
        if recent_scores[-1] > recent_scores[0] + 0.1:
            return "increasing"
        elif recent_scores[-1] < recent_scores[0] - 0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_load_recommendations(self, load_score: float) -> List[str]:
        """Generate recommendations based on cognitive load"""
        recommendations = []
        
        if load_score > 0.7:
            recommendations.extend([
                "Consider taking a short break",
                "Simplify current task",
                "Reduce information density",
                "Focus on one task at a time"
            ])
        elif load_score > 0.5:
            recommendations.extend([
                "Moderate complexity is fine",
                "Consider chunking information",
                "Take breaks between complex tasks"
            ])
        else:
            recommendations.extend([
                "Current load is manageable",
                "Can handle more complex tasks",
                "Good time for learning new concepts"
            ])
        
        return recommendations
    
    async def _analyze_mental_model(self, interaction_data: Dict[str, Any],
                                  user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's mental model and understanding"""
        return {
            "domain_knowledge": user_profile.get("expertise_level", 0.5),
            "conceptual_understanding": self._assess_conceptual_understanding(interaction_data),
            "mental_model_accuracy": 0.7,  # Placeholder
            "knowledge_gaps": self._identify_knowledge_gaps(interaction_data),
            "misconceptions": self._detect_misconceptions(interaction_data)
        }
    
    def _assess_conceptual_understanding(self, interaction_data: Dict[str, Any]) -> float:
        """Assess level of conceptual understanding"""
        message = interaction_data.get("message", "")
        
        # Look for indicators of understanding
        understanding_indicators = [
            "understand", "clear", "makes sense", "i see", "got it",
            "because", "therefore", "this means", "in other words"
        ]
        
        confusion_indicators = [
            "confused", "don't understand", "unclear", "what does", "how does",
            "i'm lost", "not sure", "?"
        ]
        
        understanding_count = sum(1 for indicator in understanding_indicators 
                                if indicator in message.lower())
        confusion_count = sum(1 for indicator in confusion_indicators 
                            if indicator in message.lower())
        
        # Calculate understanding score
        if understanding_count + confusion_count == 0:
            return 0.5  # Neutral
        
        understanding_ratio = understanding_count / (understanding_count + confusion_count)
        return understanding_ratio
    
    def _identify_knowledge_gaps(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Identify potential knowledge gaps"""
        message = interaction_data.get("message", "")
        gaps = []
        
        # Look for question patterns indicating gaps
        question_patterns = [
            r"what is (\w+)",
            r"how does (\w+) work",
            r"why does (\w+)",
            r"what's the difference between (\w+) and (\w+)"
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, message.lower())
            for match in matches:
                if isinstance(match, tuple):
                    gaps.extend(match)
                else:
                    gaps.append(match)
        
        return gaps[:5]  # Return top 5 gaps
    
    def _detect_misconceptions(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Detect potential misconceptions"""
        # Placeholder implementation
        # In practice, this would use more sophisticated NLP and domain knowledge
        return []
    
    def _detect_cognitive_style(self, interaction_data: Dict[str, Any]) -> Dict[str, float]:
        """Detect indicators of cognitive style"""
        message = interaction_data.get("message", "")
        
        # Analytical style indicators
        analytical_indicators = ["analyze", "data", "evidence", "research", "compare", "evaluate"]
        analytical_score = sum(1 for indicator in analytical_indicators 
                             if indicator in message.lower()) / len(analytical_indicators)
        
        # Intuitive style indicators
        intuitive_indicators = ["feel", "sense", "intuition", "gut", "impression", "seems"]
        intuitive_score = sum(1 for indicator in intuitive_indicators 
                            if indicator in message.lower()) / len(intuitive_indicators)
        
        # Visual style indicators
        visual_indicators = ["see", "picture", "image", "visualize", "show", "diagram"]
        visual_score = sum(1 for indicator in visual_indicators 
                         if indicator in message.lower()) / len(visual_indicators)
        
        return {
            "analytical": min(1.0, analytical_score),
            "intuitive": min(1.0, intuitive_score),
            "visual": min(1.0, visual_score)
        }
    
    def _analyze_attention_patterns(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention and focus patterns"""
        return {
            "attention_span": self._estimate_attention_span(interaction_data),
            "focus_level": self._assess_focus_level(interaction_data),
            "attention_switches": self._count_attention_switches(interaction_data),
            "sustained_attention": self._assess_sustained_attention()
        }
    
    def _estimate_attention_span(self, interaction_data: Dict[str, Any]) -> float:
        """Estimate current attention span in minutes"""
        # Based on message length and complexity
        message = interaction_data.get("message", "")
        word_count = len(message.split())
        
        # Rough estimation: longer, more complex messages suggest longer attention span
        estimated_span = min(20.0, max(1.0, word_count / 10.0))
        return estimated_span
    
    def _assess_focus_level(self, interaction_data: Dict[str, Any]) -> float:
        """Assess current focus level"""
        # Look at message coherence, length, and response time
        message = interaction_data.get("message", "")
        response_time = interaction_data.get("response_time", 0)
        
        # Focused messages are typically coherent and reasonably lengthy
        word_count = len(message.split())
        
        if word_count == 0:
            return 0.0
        
        # Good focus: moderate length, reasonable response time
        length_score = min(1.0, word_count / 20.0) if word_count <= 50 else max(0.5, 50.0 / word_count)
        time_score = 1.0 if 2 <= response_time <= 30 else 0.5
        
        return (length_score + time_score) / 2.0
    
    def _count_attention_switches(self, interaction_data: Dict[str, Any]) -> int:
        """Count attention switches in current interaction"""
        message = interaction_data.get("message", "")
        
        # Look for topic switches within the message
        sentences = re.split(r'[.!?]+', message)
        
        # Simple heuristic: count sentences that seem to change topic
        topic_switches = 0
        for i in range(1, len(sentences)):
            # Very simple topic change detection
            curr_words = set(sentences[i].lower().split())
            prev_words = set(sentences[i-1].lower().split())
            
            overlap = len(curr_words & prev_words)
            if overlap < 2 and len(curr_words) > 2 and len(prev_words) > 2:
                topic_switches += 1
        
        return topic_switches
    
    def _assess_sustained_attention(self) -> float:
        """Assess sustained attention over recent interactions"""
        if len(self.cognitive_load_history) < 3:
            return 1.0
        
        # Look at consistency of focus over recent interactions
        recent_focus_scores = [
            entry.get("focus_level", 0.5) 
            for entry in self.cognitive_load_history[-5:]
        ]
        
        if not recent_focus_scores:
            return 1.0
        
        # Sustained attention is high when focus is consistently high
        avg_focus = np.mean(recent_focus_scores)
        focus_stability = 1.0 - np.std(recent_focus_scores)
        
        return (avg_focus + focus_stability) / 2.0
    
    async def _assess_comprehension(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess level of comprehension"""
        return {
            "understanding_level": self._assess_conceptual_understanding(interaction_data),
            "comprehension_confidence": 0.7,  # Placeholder
            "clarification_needed": self._needs_clarification(interaction_data),
            "comprehension_indicators": self._extract_comprehension_indicators(interaction_data)
        }
    
    def _needs_clarification(self, interaction_data: Dict[str, Any]) -> bool:
        """Determine if user needs clarification"""
        message = interaction_data.get("message", "")
        
        clarification_indicators = [
            "?", "what", "how", "why", "unclear", "confused", "explain", "clarify"
        ]
        
        return any(indicator in message.lower() for indicator in clarification_indicators)
    
    def _extract_comprehension_indicators(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract indicators of comprehension level"""
        message = interaction_data.get("message", "")
        indicators = []
        
        if "understand" in message.lower():
            indicators.append("understanding_claimed")
        if "?" in message:
            indicators.append("questioning")
        if any(word in message.lower() for word in ["because", "therefore", "since"]):
            indicators.append("causal_reasoning")
        if any(word in message.lower() for word in ["example", "instance", "like"]):
            indicators.append("seeking_examples")
        
        return indicators
    
    def _analyze_decision_style(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze decision-making style"""
        message = interaction_data.get("message", "")
        
        # Decision style indicators
        quick_decision_indicators = ["quickly", "immediately", "right away", "fast"]
        deliberate_decision_indicators = ["carefully", "consider", "think about", "analyze"]
        collaborative_indicators = ["together", "with you", "help me", "what do you think"]
        
        quick_score = sum(1 for indicator in quick_decision_indicators 
                         if indicator in message.lower())
        deliberate_score = sum(1 for indicator in deliberate_decision_indicators 
                              if indicator in message.lower())
        collaborative_score = sum(1 for indicator in collaborative_indicators 
                                 if indicator in message.lower())
        
        return {
            "decision_speed_preference": "quick" if quick_score > deliberate_score else "deliberate",
            "collaboration_preference": collaborative_score > 0,
            "confidence_in_decisions": 0.7,  # Placeholder
            "risk_tolerance": 0.5  # Placeholder
        }
    
    def _detect_learning_signals(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect signals indicating learning activity"""
        message = interaction_data.get("message", "")
        
        learning_indicators = [
            "learn", "understand", "new", "discover", "realize", "aha", "now i see"
        ]
        
        learning_score = sum(1 for indicator in learning_indicators 
                           if indicator in message.lower())
        
        return {
            "learning_activity": learning_score > 0,
            "learning_intensity": min(1.0, learning_score / 3.0),
            "knowledge_building": self._detect_knowledge_building(message),
            "skill_development": self._detect_skill_development(message)
        }
    
    def _detect_knowledge_building(self, message: str) -> bool:
        """Detect knowledge building activity"""
        knowledge_indicators = ["connect", "relate", "similar", "different", "build on"]
        return any(indicator in message.lower() for indicator in knowledge_indicators)
    
    def _detect_skill_development(self, message: str) -> bool:
        """Detect skill development activity"""
        skill_indicators = ["practice", "try", "attempt", "apply", "use"]
        return any(indicator in message.lower() for indicator in skill_indicators)


class AdvancedContextProcessor:
    """Advanced context processor combining multiple analysis types"""
    
    def __init__(self):
        self.temporal_analyzer = TemporalContextAnalyzer()
        self.cognitive_analyzer = CognitiveContextAnalyzer()
        self.context_features: Dict[str, ContextFeature] = {}
        self.context_patterns: Dict[str, ContextPattern] = {}
        self.context_predictions: List[ContextPrediction] = []
        
    async def process_comprehensive_context(self, interaction_data: Dict[str, Any],
                                          user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Process comprehensive context using all analyzers"""
        # Temporal analysis
        temporal_context = await self.temporal_analyzer.analyze_temporal_context(
            interaction_data
        )
        
        # Cognitive analysis
        cognitive_context = await self.cognitive_analyzer.analyze_cognitive_context(
            interaction_data, user_profile
        )
        
        # Extract and store context features
        features = await self._extract_context_features(
            interaction_data, temporal_context, cognitive_context, user_profile
        )
        
        # Detect patterns
        patterns = await self._detect_context_patterns(features)
        
        # Generate predictions
        predictions = await self._generate_context_predictions(features, patterns)
        
        # Comprehensive context summary
        comprehensive_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "temporal_context": temporal_context,
            "cognitive_context": cognitive_context,
            "context_features": {fid: asdict(feature) for fid, feature in features.items()},
            "detected_patterns": [asdict(pattern) for pattern in patterns],
            "context_predictions": [asdict(pred) for pred in predictions],
            "adaptation_recommendations": await self._generate_adaptation_recommendations(
                features, patterns, cognitive_context
            ),
            "context_summary": await self._generate_context_summary(
                temporal_context, cognitive_context, patterns
            )
        }
        
        return comprehensive_context
    
    async def _extract_context_features(self, interaction_data: Dict[str, Any],
                                       temporal_context: Dict[str, Any],
                                       cognitive_context: Dict[str, Any],
                                       user_profile: Dict[str, Any]) -> Dict[str, ContextFeature]:
        """Extract individual context features"""
        features = {}
        now = datetime.utcnow()
        
        # Temporal features
        features["session_duration"] = ContextFeature(
            feature_id="session_duration",
            feature_type=ContextType.TEMPORAL,
            value=temporal_context["time_since_start"],
            confidence=1.0,
            relevance=ContextRelevance.HIGH,
            temporal_weight=1.0,
            source="temporal_analyzer",
            extracted_at=now,
            expires_at=None
        )
        
        features["interaction_frequency"] = ContextFeature(
            feature_id="interaction_frequency",
            feature_type=ContextType.BEHAVIORAL,
            value=temporal_context["interaction_frequency"]["interactions_per_minute"],
            confidence=0.8,
            relevance=ContextRelevance.MEDIUM,
            temporal_weight=0.9,
            source="temporal_analyzer",
            extracted_at=now,
            expires_at=now + timedelta(minutes=30)
        )
        
        # Cognitive features
        cognitive_load = cognitive_context["cognitive_load"]
        features["cognitive_load"] = ContextFeature(
            feature_id="cognitive_load",
            feature_type=ContextType.COGNITIVE,
            value=cognitive_load["current_load"],
            confidence=0.7,
            relevance=ContextRelevance.CRITICAL,
            temporal_weight=0.95,
            source="cognitive_analyzer",
            extracted_at=now,
            expires_at=now + timedelta(minutes=10)
        )
        
        # Task features
        if "task_type" in interaction_data:
            features["task_type"] = ContextFeature(
                feature_id="task_type",
                feature_type=ContextType.TASK,
                value=interaction_data["task_type"],
                confidence=1.0,
                relevance=ContextRelevance.HIGH,
                temporal_weight=1.0,
                source="interaction_data",
                extracted_at=now,
                expires_at=None
            )
        
        # User features
        features["expertise_level"] = ContextFeature(
            feature_id="expertise_level",
            feature_type=ContextType.COGNITIVE,
            value=user_profile.get("expertise_level", 0.5),
            confidence=0.9,
            relevance=ContextRelevance.HIGH,
            temporal_weight=1.0,
            source="user_profile",
            extracted_at=now,
            expires_at=None
        )
        
        # Store features
        self.context_features.update(features)
        
        return features
    
    async def _detect_context_patterns(self, features: Dict[str, ContextFeature]) -> List[ContextPattern]:
        """Detect patterns in context features"""
        patterns = []
        
        # High cognitive load pattern
        if "cognitive_load" in features and features["cognitive_load"].value > 0.7:
            pattern_id = "high_cognitive_load"
            if pattern_id not in self.context_patterns:
                self.context_patterns[pattern_id] = ContextPattern(
                    pattern_id=pattern_id,
                    pattern_name="High Cognitive Load",
                    features=["cognitive_load"],
                    confidence=0.8,
                    frequency=1,
                    last_seen=datetime.utcnow(),
                    associated_outcomes=["reduced_performance", "need_simplification"],
                    adaptation_triggers=["reduce_complexity", "provide_support"]
                )
            else:
                self.context_patterns[pattern_id].frequency += 1
                self.context_patterns[pattern_id].last_seen = datetime.utcnow()
            
            patterns.append(self.context_patterns[pattern_id])
        
        # Expertise mismatch pattern
        if ("expertise_level" in features and "task_type" in features and
            features["expertise_level"].value < 0.3):
            pattern_id = "low_expertise_complex_task"
            if pattern_id not in self.context_patterns:
                self.context_patterns[pattern_id] = ContextPattern(
                    pattern_id=pattern_id,
                    pattern_name="Low Expertise Complex Task",
                    features=["expertise_level", "task_type"],
                    confidence=0.7,
                    frequency=1,
                    last_seen=datetime.utcnow(),
                    associated_outcomes=["learning_opportunity", "need_guidance"],
                    adaptation_triggers=["increase_support", "provide_examples"]
                )
            else:
                self.context_patterns[pattern_id].frequency += 1
                self.context_patterns[pattern_id].last_seen = datetime.utcnow()
            
            patterns.append(self.context_patterns[pattern_id])
        
        return patterns
    
    async def _generate_context_predictions(self, features: Dict[str, ContextFeature],
                                          patterns: List[ContextPattern]) -> List[ContextPrediction]:
        """Generate predictions about future context states"""
        predictions = []
        
        # Predict cognitive load trend
        if "cognitive_load" in features:
            current_load = features["cognitive_load"].value
            
            # Simple prediction: if load is high, predict it might continue or increase
            if current_load > 0.6:
                prediction = ContextPrediction(
                    prediction_id=str(uuid4()),
                    predicted_context={"cognitive_load": min(1.0, current_load + 0.1)},
                    probability=0.7,
                    time_horizon=timedelta(minutes=10),
                    trigger_conditions=["continued_complex_tasks"],
                    confidence_interval=(current_load, min(1.0, current_load + 0.2)),
                    created_at=datetime.utcnow()
                )
                predictions.append(prediction)
        
        # Store predictions
        self.context_predictions.extend(predictions)
        
        return predictions
    
    async def _generate_adaptation_recommendations(self, features: Dict[str, ContextFeature],
                                                 patterns: List[ContextPattern],
                                                 cognitive_context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for context-based adaptations"""
        recommendations = []
        
        # Cognitive load adaptations
        cognitive_load = cognitive_context.get("cognitive_load", {})
        load_level = cognitive_load.get("load_level", "moderate")
        
        if load_level == "high" or load_level == "overload":
            recommendations.extend([
                "reduce_information_density",
                "simplify_language",
                "break_tasks_into_steps",
                "provide_clear_structure"
            ])
        elif load_level == "low":
            recommendations.extend([
                "increase_information_richness",
                "provide_advanced_options",
                "offer_additional_details"
            ])
        
        # Pattern-based adaptations
        for pattern in patterns:
            recommendations.extend(pattern.adaptation_triggers)
        
        # Remove duplicates
        return list(set(recommendations))
    
    async def _generate_context_summary(self, temporal_context: Dict[str, Any],
                                       cognitive_context: Dict[str, Any],
                                       patterns: List[ContextPattern]) -> str:
        """Generate human-readable context summary"""
        summary_parts = []
        
        # Temporal summary
        session_duration = temporal_context.get("time_since_start", 0)
        session_phase = temporal_context.get("session_phase", "unknown")
        summary_parts.append(f"Session: {session_duration:.1f}min ({session_phase} phase)")
        
        # Cognitive summary
        cognitive_load = cognitive_context.get("cognitive_load", {})
        load_level = cognitive_load.get("load_level", "moderate")
        summary_parts.append(f"Cognitive load: {load_level}")
        
        # Pattern summary
        if patterns:
            pattern_names = [pattern.pattern_name for pattern in patterns]
            summary_parts.append(f"Patterns: {', '.join(pattern_names)}")
        
        return "; ".join(summary_parts)
    
    async def get_context_health(self) -> Dict[str, Any]:
        """Get overall context processing health metrics"""
        return {
            "total_features": len(self.context_features),
            "active_patterns": len(self.context_patterns),
            "recent_predictions": len([p for p in self.context_predictions 
                                     if (datetime.utcnow() - p.created_at).total_seconds() < 3600]),
            "temporal_analyzer_health": {
                "timeline_size": len(self.temporal_analyzer.context_timeline),
                "patterns_detected": len(self.temporal_analyzer.temporal_patterns)
            },
            "cognitive_analyzer_health": {
                "load_history_size": len(self.cognitive_analyzer.cognitive_load_history),
                "patterns_detected": len(self.cognitive_analyzer.cognitive_patterns)
            }
        }
