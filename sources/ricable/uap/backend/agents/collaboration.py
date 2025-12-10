"""
Agent 38: Advanced Human-AI Collaboration Agent
Orchestrates collaborative AI assistants with context awareness, trust calibration,
and adaptive interfaces based on cognitive load.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import uuid4

# Import collaboration components
from ..human_ai.collaboration_engine import (
    CollaborationEngine, UserProfile, CollaborationSession, CollaborationMode,
    InteractionType, CollaborationDecision
)
from ..human_ai.context_processor import (
    AdvancedContextProcessor, ContextType, ContextRelevance
)
from ..human_ai.trust_calibration import (
    TrustCalibrationEngine, TrustLevel, TrustDimension, TrustEvent
)
from ..human_ai.explanation_generator import (
    ExplanationGenerator, ExplanationRequest, ExplanationType, 
    ExplanationStyle, ExplanationLevel, ExplanationFeedback
)
from ..human_ai.cognitive_load_monitor import (
    CognitiveLoadMonitor, CognitiveLoadLevel, LoadIndicator, 
    AdaptationTrigger, LoadAdaptation
)

# Import integration dependencies
from ..neurosymbolic.neural_symbolic_engine import NeuroSymbolicEngine
from ..customer.personalization import PersonalizationEngine

logger = logging.getLogger(__name__)


class CollaborationAgentMode(Enum):
    """Operating modes for the collaboration agent"""
    ADAPTIVE = "adaptive"           # Fully adaptive based on user state
    ASSISTIVE = "assistive"         # Focus on assisting user decisions
    COLLABORATIVE = "collaborative" # Equal partnership mode
    SUPERVISORY = "supervisory"     # AI under human supervision
    AUTONOMOUS = "autonomous"       # AI takes initiative with oversight
    LEARNING = "learning"           # Focus on user learning and growth


class InteractionComplexity(Enum):
    """Complexity levels for interactions"""
    SIMPLE = "simple"     # Basic Q&A
    MODERATE = "moderate"  # Multi-step tasks
    COMPLEX = "complex"    # Complex problem solving
    EXPERT = "expert"      # Domain expert level


@dataclass
class CollaborationMetrics:
    """Metrics for collaboration quality"""
    session_id: str
    collaboration_effectiveness: float  # 0-1 scale
    user_satisfaction: float
    trust_level: float
    cognitive_load: float
    adaptation_count: int
    decision_quality: float
    learning_progress: float
    interaction_efficiency: float
    timestamp: datetime


@dataclass
class AdaptationSuggestion:
    """Suggestion for interface or behavior adaptation"""
    suggestion_id: str
    adaptation_type: str
    rationale: str
    implementation: Dict[str, Any]
    priority: str  # "immediate", "high", "medium", "low"
    expected_benefit: str
    estimated_effort: str
    created_at: datetime


class CollaborationAgent:
    """Main agent for advanced human-AI collaboration"""
    
    def __init__(self):
        # Core collaboration components
        self.collaboration_engine = CollaborationEngine()
        self.context_processor = AdvancedContextProcessor()
        self.trust_calibration = TrustCalibrationEngine()
        self.explanation_generator = ExplanationGenerator()
        self.cognitive_load_monitor = CognitiveLoadMonitor()
        
        # Integration components (will be injected)
        self.neurosymbolic_engine: Optional[NeuroSymbolicEngine] = None
        self.personalization_engine: Optional[PersonalizationEngine] = None
        
        # Agent state
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.collaboration_metrics: List[CollaborationMetrics] = []
        self.adaptation_suggestions: Dict[str, List[AdaptationSuggestion]] = {}
        
        # Configuration
        self.agent_config = {
            "max_concurrent_sessions": 100,
            "session_timeout_minutes": 60,
            "adaptation_frequency_seconds": 30,
            "trust_update_threshold": 0.1,
            "load_monitoring_interval": 10
        }
        
        # Performance tracking
        self.performance_metrics = {
            "total_sessions": 0,
            "successful_collaborations": 0,
            "average_satisfaction": 0.0,
            "average_trust_level": 0.0,
            "adaptations_applied": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the collaboration agent"""
        try:
            logger.info("Initializing Advanced Human-AI Collaboration Agent")
            
            # Initialize all components
            await self.collaboration_engine.get_system_status()
            await self.context_processor.get_context_health()
            await self.trust_calibration.get_system_trust_status()
            await self.explanation_generator.get_explanation_stats()
            await self.cognitive_load_monitor.get_system_status()
            
            logger.info("Human-AI Collaboration Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Collaboration Agent: {e}")
            return False
    
    async def start_collaboration_session(self, user_id: str, task_description: str,
                                        initial_context: Dict[str, Any],
                                        user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start a new collaboration session"""
        try:
            # Check session limits
            if len(self.active_sessions) >= self.agent_config["max_concurrent_sessions"]:
                return {
                    "error": "Maximum concurrent sessions reached",
                    "max_sessions": self.agent_config["max_concurrent_sessions"]
                }
            
            # Initialize user profiles if needed
            await self._ensure_user_profiles_exist(user_id, user_preferences or {})
            
            # Start collaboration session
            session_id = await self.collaboration_engine.start_collaboration_session(
                user_id, "collaboration_agent", task_description, initial_context
            )
            
            # Process initial context
            user_profile = self.collaboration_engine.user_profiles[user_id]
            processed_context = await self.context_processor.process_comprehensive_context(
                initial_context, asdict(user_profile)
            )
            
            # Initial cognitive load measurement
            await self.cognitive_load_monitor.measure_cognitive_load(
                user_id, session_id, {
                    "message": task_description,
                    "response_time": 0,
                    "total_interactions": 1
                }, processed_context
            )
            
            # Store session
            session = self.collaboration_engine.active_sessions[session_id]
            self.active_sessions[session_id] = session
            
            # Generate initial adaptation suggestions
            adaptations = await self._generate_initial_adaptations(
                user_id, session_id, processed_context
            )
            
            # Update metrics
            self.performance_metrics["total_sessions"] += 1
            
            logger.info(f"Started collaboration session {session_id} for user {user_id}")
            
            return {
                "session_id": session_id,
                "status": "active",
                "collaboration_mode": session.collaboration_mode.value,
                "initial_context": processed_context,
                "initial_adaptations": [asdict(adapt) for adapt in adaptations],
                "session_config": {
                    "timeout_minutes": self.agent_config["session_timeout_minutes"],
                    "adaptation_frequency": self.agent_config["adaptation_frequency_seconds"]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start collaboration session: {e}")
            return {"error": str(e)}
    
    async def process_interaction(self, session_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a user interaction within a collaboration session"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            # Determine interaction type
            interaction_type = self._determine_interaction_type(interaction_data)
            
            # Process with collaboration engine
            collaboration_response = await self.collaboration_engine.process_interaction(
                session_id, interaction_type, interaction_data
            )
            
            # Process context
            user_profile = self.collaboration_engine.user_profiles[user_id]
            current_context = await self.context_processor.process_comprehensive_context(
                interaction_data.get("context", {}), asdict(user_profile)
            )
            
            # Monitor cognitive load
            cognitive_measurement = await self.cognitive_load_monitor.measure_cognitive_load(
                user_id, session_id, interaction_data, current_context
            )
            
            # Update trust based on interaction
            trust_result = await self._update_trust_from_interaction(
                user_id, session_id, interaction_data, collaboration_response
            )
            
            # Generate explanation if needed
            explanation = None
            if interaction_data.get("request_explanation", False):
                explanation = await self._generate_contextual_explanation(
                    user_id, interaction_data, collaboration_response, current_context
                )
            
            # Generate adaptations
            adaptations = await self._generate_interaction_adaptations(
                user_id, session_id, cognitive_measurement, current_context
            )
            
            # Create comprehensive response
            response = {
                "interaction_id": str(uuid4()),
                "session_id": session_id,
                "collaboration_response": collaboration_response,
                "cognitive_load": {
                    "current_load": cognitive_measurement.overall_load,
                    "load_level": cognitive_measurement.load_level.value,
                    "contributing_factors": cognitive_measurement.contributing_factors
                },
                "trust_update": trust_result,
                "context_summary": current_context.get("context_summary", ""),
                "adaptations": [asdict(adapt) for adapt in adaptations],
                "explanation": asdict(explanation) if explanation else None,
                "recommendations": await self._generate_interaction_recommendations(
                    user_id, cognitive_measurement, current_context
                ),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Record metrics
            await self._record_interaction_metrics(
                session_id, interaction_data, response, cognitive_measurement
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process interaction: {e}")
            return {"error": str(e)}
    
    async def make_collaborative_decision(self, session_id: str, 
                                        decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Facilitate a collaborative decision"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            # Get user profiles
            user_profile = self.collaboration_engine.user_profiles[user_id]
            cognitive_profile = self.cognitive_load_monitor.user_profiles.get(user_id)
            
            # Enhance decision context with AI insights
            if self.neurosymbolic_engine:
                ai_insights = await self.neurosymbolic_engine.hybrid_reasoning({
                    "decision_context": decision_context,
                    "user_profile": asdict(user_profile),
                    "goal": decision_context.get("goal", "make_best_decision")
                })
                decision_context["ai_insights"] = ai_insights
            
            # Make collaborative decision
            decision = await self.collaboration_engine.make_collaborative_decision(
                session_id, decision_context
            )
            
            # Generate explanation for the decision
            decision_explanation = await self._generate_decision_explanation(
                user_id, decision, decision_context
            )
            
            # Update trust based on decision quality
            await self.trust_calibration.record_trust_event(
                user_id, session_id, "collaborative_decision", {
                    "decision_type": decision.decision_type,
                    "confidence": decision.confidence_scores.get("ai_confidence", 0.5),
                    "user_involvement": decision.decision_method
                }
            )
            
            # Record decision metrics
            await self._record_decision_metrics(session_id, decision, decision_context)
            
            return {
                "decision": asdict(decision),
                "explanation": asdict(decision_explanation) if decision_explanation else None,
                "trust_impact": "positive" if decision.confidence_scores.get("ai_confidence", 0.5) > 0.7 else "neutral",
                "collaboration_quality": await self._assess_collaboration_quality(session_id),
                "recommendations": await self._generate_decision_recommendations(decision, user_profile)
            }
            
        except Exception as e:
            logger.error(f"Failed to make collaborative decision: {e}")
            return {"error": str(e)}
    
    async def provide_learning_support(self, session_id: str, 
                                     learning_request: Dict[str, Any]) -> Dict[str, Any]:
        """Provide adaptive learning support"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            # Get current cognitive load
            user_measurements = self.cognitive_load_monitor.load_measurements.get(user_id, [])
            current_load = user_measurements[-1].overall_load if user_measurements else 0.5
            
            # Adapt learning approach based on cognitive load
            learning_approach = await self._determine_learning_approach(
                user_id, current_load, learning_request
            )
            
            # Generate learning content
            learning_content = await self._generate_adaptive_learning_content(
                user_id, learning_request, learning_approach
            )
            
            # Create learning interaction
            learning_interaction = {
                "type": "learning_session",
                "learning_objective": learning_request.get("objective", "understanding"),
                "content": learning_content,
                "approach": learning_approach,
                "cognitive_load_consideration": current_load
            }
            
            # Process as interaction
            response = await self.collaboration_engine.process_interaction(
                session_id, InteractionType.LEARNING_SESSION, learning_interaction
            )
            
            return {
                "learning_response": response,
                "learning_content": learning_content,
                "approach_used": learning_approach,
                "cognitive_load_adapted": current_load > 0.6,
                "next_steps": await self._suggest_learning_next_steps(user_id, learning_request)
            }
            
        except Exception as e:
            logger.error(f"Failed to provide learning support: {e}")
            return {"error": str(e)}
    
    async def calibrate_trust(self, session_id: str, trust_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrate trust based on user feedback"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            # Record trust event
            trust_result = await self.trust_calibration.record_trust_event(
                user_id, session_id, "user_feedback", trust_feedback
            )
            
            # Get updated trust status
            trust_status = await self.trust_calibration.get_user_trust_status(user_id)
            
            # Calibrate collaboration mode based on trust
            calibration_result = await self.trust_calibration.calibrate_trust_for_context(
                user_id, {
                    "current_session": session_id,
                    "task_complexity": session.current_context.get("task_context", {}).get("complexity", "medium"),
                    "user_expertise": session.current_context.get("user_context", {}).get("expertise_level", 0.5)
                }
            )
            
            # Update collaboration mode if needed
            await self._update_collaboration_mode_from_trust(session_id, trust_status)
            
            return {
                "trust_update": trust_result,
                "current_trust_status": trust_status,
                "calibration_result": calibration_result,
                "collaboration_adjustments": await self._get_trust_based_adjustments(trust_status)
            }
            
        except Exception as e:
            logger.error(f"Failed to calibrate trust: {e}")
            return {"error": str(e)}
    
    async def handle_explanation_request(self, session_id: str, 
                                       explanation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user request for explanation"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            # Create explanation request
            request = ExplanationRequest(
                request_id=str(uuid4()),
                user_id=user_id,
                session_id=session_id,
                explanation_target=explanation_request.get("target", "last_response"),
                context=explanation_request.get("context", {}),
                user_profile=asdict(self.collaboration_engine.user_profiles[user_id]),
                preferred_type=ExplanationType(explanation_request.get("type")) if explanation_request.get("type") else None,
                preferred_style=ExplanationStyle(explanation_request.get("style")) if explanation_request.get("style") else None,
                preferred_level=ExplanationLevel(explanation_request.get("level")) if explanation_request.get("level") else None,
                max_length=explanation_request.get("max_length"),
                include_uncertainty=explanation_request.get("include_uncertainty", True),
                timestamp=datetime.utcnow()
            )
            
            # Generate explanation
            explanation = await self.explanation_generator.generate_explanation(request)
            
            # Record trust event for explanation provision
            await self.trust_calibration.record_trust_event(
                user_id, session_id, "clear_explanation", {
                    "explanation_type": explanation.explanation_type.value,
                    "confidence": explanation.confidence,
                    "clarity_score": explanation.clarity_score
                }
            )
            
            return {
                "explanation": asdict(explanation),
                "understanding_check": await self._generate_understanding_check(explanation),
                "follow_up_options": explanation.follow_up_questions
            }
            
        except Exception as e:
            logger.error(f"Failed to handle explanation request: {e}")
            return {"error": str(e)}
    
    async def end_collaboration_session(self, session_id: str, 
                                      feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        """End a collaboration session and collect metrics"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            # Collect final metrics
            final_metrics = await self._collect_session_metrics(session_id)
            
            # Process user feedback if provided
            if feedback:
                await self._process_session_feedback(session_id, feedback)
            
            # End collaboration session
            session_result = await self.collaboration_engine.end_session(session_id)
            
            # Generate session summary
            session_summary = await self._generate_session_summary(session_id, final_metrics)
            
            # Update performance metrics
            if final_metrics.collaboration_effectiveness > 0.7:
                self.performance_metrics["successful_collaborations"] += 1
            
            # Update averages
            total_sessions = self.performance_metrics["total_sessions"]
            if total_sessions > 0:
                self.performance_metrics["average_satisfaction"] = (
                    (self.performance_metrics["average_satisfaction"] * (total_sessions - 1) + 
                     final_metrics.user_satisfaction) / total_sessions
                )
                self.performance_metrics["average_trust_level"] = (
                    (self.performance_metrics["average_trust_level"] * (total_sessions - 1) + 
                     final_metrics.trust_level) / total_sessions
                )
            
            # Clean up
            del self.active_sessions[session_id]
            if user_id in self.adaptation_suggestions:
                del self.adaptation_suggestions[user_id]
            
            logger.info(f"Ended collaboration session {session_id}")
            
            return {
                "session_result": session_result,
                "final_metrics": asdict(final_metrics),
                "session_summary": session_summary,
                "recommendations": await self._generate_future_recommendations(user_id, final_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to end collaboration session: {e}")
            return {"error": str(e)}
    
    # Helper methods
    
    async def _ensure_user_profiles_exist(self, user_id: str, preferences: Dict[str, Any]):
        """Ensure all user profiles exist across components"""
        # Collaboration engine profile
        if user_id not in self.collaboration_engine.user_profiles:
            await self.collaboration_engine.create_user_profile(user_id, preferences)
        
        # Trust calibration profile
        if user_id not in self.trust_calibration.user_trust_profiles:
            await self.trust_calibration.initialize_user_trust(user_id, preferences)
        
        # Cognitive load monitor profile
        if user_id not in self.cognitive_load_monitor.user_profiles:
            await self.cognitive_load_monitor.initialize_user_profile(user_id, preferences)
    
    def _determine_interaction_type(self, interaction_data: Dict[str, Any]) -> InteractionType:
        """Determine the type of interaction"""
        if "decision" in interaction_data:
            return InteractionType.DECISION_SUPPORT
        elif "feedback" in interaction_data:
            return InteractionType.FEEDBACK_LOOP
        elif "task" in interaction_data:
            return InteractionType.COLLABORATIVE_TASK
        elif "learning" in interaction_data:
            return InteractionType.LEARNING_SESSION
        elif "trust" in interaction_data:
            return InteractionType.TRUST_CALIBRATION
        else:
            return InteractionType.QUERY_RESPONSE
    
    async def _update_trust_from_interaction(self, user_id: str, session_id: str,
                                           interaction_data: Dict[str, Any],
                                           response: Dict[str, Any]) -> Dict[str, Any]:
        """Update trust based on interaction outcome"""
        # Determine trust event type
        if response.get("type") == "error":
            event_type = "system_error"
        elif "helpful" in str(interaction_data).lower():
            event_type = "helpful_suggestion"
        elif response.get("confidence", 0.5) > 0.8:
            event_type = "correct_prediction"
        else:
            event_type = "successful_task"
        
        return await self.trust_calibration.record_trust_event(
            user_id, session_id, event_type, {
                "interaction_type": response.get("type", "unknown"),
                "confidence": response.get("confidence", 0.5),
                "user_satisfaction": interaction_data.get("satisfaction", 0.7)
            }
        )
    
    async def _generate_contextual_explanation(self, user_id: str, interaction_data: Dict[str, Any],
                                             response: Dict[str, Any], context: Dict[str, Any]):
        """Generate contextual explanation"""
        request = ExplanationRequest(
            request_id=str(uuid4()),
            user_id=user_id,
            session_id=context.get("session_id", ""),
            explanation_target="last_response",
            context={
                "response": response,
                "interaction": interaction_data,
                "full_context": context
            },
            user_profile=asdict(self.collaboration_engine.user_profiles[user_id]),
            preferred_type=None,
            preferred_style=None,
            preferred_level=None,
            max_length=None,
            include_uncertainty=True,
            timestamp=datetime.utcnow()
        )
        
        return await self.explanation_generator.generate_explanation(request)
    
    async def _generate_initial_adaptations(self, user_id: str, session_id: str,
                                          context: Dict[str, Any]) -> List[AdaptationSuggestion]:
        """Generate initial adaptation suggestions"""
        adaptations = []
        
        # Get user profiles
        user_profile = self.collaboration_engine.user_profiles[user_id]
        
        # Trust-based adaptations
        if user_profile.trust_level < 0.5:
            adaptations.append(AdaptationSuggestion(
                suggestion_id=str(uuid4()),
                adaptation_type="trust_building",
                rationale="Low initial trust detected",
                implementation={
                    "increase_transparency": True,
                    "provide_more_explanations": True,
                    "show_confidence_scores": True
                },
                priority="high",
                expected_benefit="Improved user trust and engagement",
                estimated_effort="low",
                created_at=datetime.utcnow()
            ))
        
        # Expertise-based adaptations
        if user_profile.expertise_level < 0.3:
            adaptations.append(AdaptationSuggestion(
                suggestion_id=str(uuid4()),
                adaptation_type="novice_support",
                rationale="Novice user detected",
                implementation={
                    "simplify_language": True,
                    "provide_more_guidance": True,
                    "break_down_tasks": True
                },
                priority="medium",
                expected_benefit="Better user understanding and success",
                estimated_effort="medium",
                created_at=datetime.utcnow()
            ))
        
        self.adaptation_suggestions[user_id] = adaptations
        return adaptations
    
    async def _generate_interaction_adaptations(self, user_id: str, session_id: str,
                                              cognitive_measurement, context: Dict[str, Any]) -> List[AdaptationSuggestion]:
        """Generate adaptations based on current interaction"""
        adaptations = []
        
        # Cognitive load adaptations
        if cognitive_measurement.load_level in [CognitiveLoadLevel.HIGH, CognitiveLoadLevel.OVERLOAD]:
            load_adaptations = await self.cognitive_load_monitor.analyze_and_adapt(user_id)
            
            for adaptation_dict in load_adaptations.get("adaptations", []):
                adaptations.append(AdaptationSuggestion(
                    suggestion_id=str(uuid4()),
                    adaptation_type="cognitive_load_reduction",
                    rationale=f"High cognitive load detected: {cognitive_measurement.load_level.value}",
                    implementation=adaptation_dict.get("ui_modifications", {}),
                    priority=adaptation_dict.get("priority", "medium"),
                    expected_benefit="Reduced cognitive load and improved performance",
                    estimated_effort="low",
                    created_at=datetime.utcnow()
                ))
        
        return adaptations
    
    async def _generate_interaction_recommendations(self, user_id: str, cognitive_measurement,
                                                  context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for the user"""
        recommendations = []
        
        # Cognitive load recommendations
        if cognitive_measurement.load_level == CognitiveLoadLevel.HIGH:
            recommendations.extend([
                "Consider taking a short break",
                "Focus on one task at a time",
                "Ask for simpler explanations if needed"
            ])
        elif cognitive_measurement.load_level == CognitiveLoadLevel.OVERLOAD:
            recommendations.extend([
                "Take a break - cognitive overload detected",
                "Consider postponing complex tasks",
                "Ask for human assistance if available"
            ])
        elif cognitive_measurement.load_level == CognitiveLoadLevel.LOW:
            recommendations.extend([
                "You can handle more complex tasks",
                "Consider exploring advanced features",
                "Good time for learning new concepts"
            ])
        
        # Context-based recommendations
        adaptation_triggers = context.get("adaptation_triggers", [])
        if "high_complexity" in adaptation_triggers:
            recommendations.append("Complex task detected - I'll provide extra support")
        
        if "time_pressure" in adaptation_triggers:
            recommendations.append("Time pressure detected - let's focus on essentials")
        
        return recommendations
    
    async def _record_interaction_metrics(self, session_id: str, interaction_data: Dict[str, Any],
                                        response: Dict[str, Any], cognitive_measurement):
        """Record metrics for an interaction"""
        session = self.active_sessions[session_id]
        user_id = session.user_id
        
        # Get current trust level
        trust_status = await self.trust_calibration.get_user_trust_status(user_id)
        current_trust = trust_status.get("overall_trust", 0.5)
        
        # Calculate interaction efficiency
        response_time = interaction_data.get("response_time", 0)
        interaction_efficiency = max(0.0, 1.0 - (response_time / 60.0))  # Normalize by 1 minute
        
        # Create metrics
        metrics = CollaborationMetrics(
            session_id=session_id,
            collaboration_effectiveness=response.get("collaboration_effectiveness", 0.7),
            user_satisfaction=interaction_data.get("satisfaction", 0.7),
            trust_level=current_trust,
            cognitive_load=cognitive_measurement.overall_load,
            adaptation_count=len(response.get("adaptations", [])),
            decision_quality=response.get("decision_quality", 0.7),
            learning_progress=response.get("learning_progress", 0.0),
            interaction_efficiency=interaction_efficiency,
            timestamp=datetime.utcnow()
        )
        
        self.collaboration_metrics.append(metrics)
    
    async def _determine_learning_approach(self, user_id: str, current_load: float,
                                         learning_request: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal learning approach based on user state"""
        user_profile = self.collaboration_engine.user_profiles[user_id]
        
        approach = {
            "method": "adaptive",
            "pace": "moderate",
            "detail_level": "medium",
            "interaction_style": "collaborative"
        }
        
        # Adapt based on cognitive load
        if current_load > 0.7:
            approach.update({
                "method": "simplified",
                "pace": "slow",
                "detail_level": "low",
                "break_frequency": "high"
            })
        elif current_load < 0.3:
            approach.update({
                "method": "accelerated",
                "pace": "fast",
                "detail_level": "high",
                "challenge_level": "increased"
            })
        
        # Adapt based on expertise
        if user_profile.expertise_level < 0.3:
            approach.update({
                "foundation_building": True,
                "examples_heavy": True,
                "scaffolding": "high"
            })
        elif user_profile.expertise_level > 0.7:
            approach.update({
                "advanced_concepts": True,
                "self_directed": True,
                "scaffolding": "minimal"
            })
        
        return approach
    
    async def _generate_adaptive_learning_content(self, user_id: str, learning_request: Dict[str, Any],
                                                approach: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptive learning content"""
        content = {
            "objective": learning_request.get("objective", "understanding"),
            "content_type": approach.get("method", "adaptive"),
            "materials": [],
            "exercises": [],
            "assessments": []
        }
        
        # Generate content based on approach
        if approach.get("examples_heavy"):
            content["materials"].extend([
                "concrete_examples",
                "step_by_step_demonstrations",
                "practice_scenarios"
            ])
        
        if approach.get("foundation_building"):
            content["materials"].extend([
                "prerequisite_concepts",
                "basic_principles",
                "glossary_terms"
            ])
        
        if approach.get("advanced_concepts"):
            content["materials"].extend([
                "theoretical_frameworks",
                "complex_applications",
                "research_insights"
            ])
        
        return content
    
    async def _assess_collaboration_quality(self, session_id: str) -> float:
        """Assess the quality of collaboration in a session"""
        # Get recent metrics for this session
        session_metrics = [
            m for m in self.collaboration_metrics 
            if m.session_id == session_id and 
            (datetime.utcnow() - m.timestamp).total_seconds() < 600  # Last 10 minutes
        ]
        
        if not session_metrics:
            return 0.7  # Default moderate quality
        
        # Calculate average quality metrics
        avg_effectiveness = sum(m.collaboration_effectiveness for m in session_metrics) / len(session_metrics)
        avg_satisfaction = sum(m.user_satisfaction for m in session_metrics) / len(session_metrics)
        avg_trust = sum(m.trust_level for m in session_metrics) / len(session_metrics)
        
        # Weighted combination
        quality = (avg_effectiveness * 0.4 + avg_satisfaction * 0.3 + avg_trust * 0.3)
        
        return quality
    
    async def _collect_session_metrics(self, session_id: str) -> CollaborationMetrics:
        """Collect final metrics for a session"""
        session = self.active_sessions[session_id]
        user_id = session.user_id
        
        # Get session metrics
        session_metrics = [m for m in self.collaboration_metrics if m.session_id == session_id]
        
        if session_metrics:
            # Average session metrics
            final_metrics = CollaborationMetrics(
                session_id=session_id,
                collaboration_effectiveness=sum(m.collaboration_effectiveness for m in session_metrics) / len(session_metrics),
                user_satisfaction=sum(m.user_satisfaction for m in session_metrics) / len(session_metrics),
                trust_level=sum(m.trust_level for m in session_metrics) / len(session_metrics),
                cognitive_load=sum(m.cognitive_load for m in session_metrics) / len(session_metrics),
                adaptation_count=sum(m.adaptation_count for m in session_metrics),
                decision_quality=sum(m.decision_quality for m in session_metrics) / len(session_metrics),
                learning_progress=sum(m.learning_progress for m in session_metrics) / len(session_metrics),
                interaction_efficiency=sum(m.interaction_efficiency for m in session_metrics) / len(session_metrics),
                timestamp=datetime.utcnow()
            )
        else:
            # Default metrics if no data
            final_metrics = CollaborationMetrics(
                session_id=session_id,
                collaboration_effectiveness=0.7,
                user_satisfaction=0.7,
                trust_level=0.7,
                cognitive_load=0.5,
                adaptation_count=0,
                decision_quality=0.7,
                learning_progress=0.0,
                interaction_efficiency=0.7,
                timestamp=datetime.utcnow()
            )
        
        return final_metrics
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_name": "Advanced Human-AI Collaboration Agent",
            "status": "operational",
            "active_sessions": len(self.active_sessions),
            "performance_metrics": self.performance_metrics,
            "component_status": {
                "collaboration_engine": await self.collaboration_engine.get_system_status(),
                "context_processor": await self.context_processor.get_context_health(),
                "trust_calibration": await self.trust_calibration.get_system_trust_status(),
                "explanation_generator": await self.explanation_generator.get_explanation_stats(),
                "cognitive_load_monitor": await self.cognitive_load_monitor.get_system_status()
            },
            "configuration": self.agent_config,
            "recent_metrics": [
                asdict(m) for m in self.collaboration_metrics[-10:]
            ]
        }
    
    async def cleanup(self):
        """Cleanup agent resources"""
        # End any active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_collaboration_session(session_id)
        
        logger.info("Collaboration Agent cleanup complete")
