# File: backend/api_routes/collaboration.py
"""
API Routes for Human-AI Collaboration
Provides REST endpoints for the collaboration dashboard and components.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import logging

from ..human_ai.collaboration_engine import CollaborationEngine, InteractionType, CollaborationMode
from ..human_ai.trust_calibration import TrustCalibrationEngine
from ..human_ai.cognitive_load_monitor import CognitiveLoadMonitor
from ..human_ai.context_processor import AdvancedContextProcessor
from ..human_ai.explanation_generator import ExplanationGenerator, ExplanationRequest, ExplanationFeedback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collaboration", tags=["collaboration"])

# Global instances (in production, these would be dependency injected)
collaboration_engine = CollaborationEngine()
trust_engine = TrustCalibrationEngine()
cognitive_monitor = CognitiveLoadMonitor()
context_processor = AdvancedContextProcessor()
explanation_generator = ExplanationGenerator()


# Request/Response Models
class StartSessionRequest(BaseModel):
    user_id: str
    ai_agent_id: str
    task_description: str
    initial_context: Dict[str, Any]


class InteractionRequest(BaseModel):
    session_id: str
    interaction_type: str
    interaction_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class ExplanationRequestModel(BaseModel):
    user_id: str
    session_id: str
    explanation_target: str
    preferred_type: Optional[str] = None
    preferred_style: Optional[str] = None
    preferred_level: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    user_profile: Optional[Dict[str, Any]] = None


class TrustEventRequest(BaseModel):
    user_id: str
    session_id: str
    event_type: str
    event_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class CognitiveLoadRequest(BaseModel):
    user_id: str
    session_id: str
    interaction_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class ProfileUpdateRequest(BaseModel):
    user_id: str
    expertise_level: Optional[float] = None
    preferred_collaboration_mode: Optional[str] = None
    cognitive_style: Optional[str] = None
    trust_level: Optional[float] = None
    preferences: Optional[Dict[str, Any]] = None


# Collaboration Session Management
@router.post("/start-session")
async def start_collaboration_session(request: StartSessionRequest):
    """Start a new human-AI collaboration session"""
    try:
        session_id = await collaboration_engine.start_collaboration_session(
            user_id=request.user_id,
            ai_agent_id=request.ai_agent_id,
            task_description=request.task_description,
            initial_context=request.initial_context
        )
        
        # Initialize trust profile if needed
        await trust_engine.initialize_user_trust(request.user_id)
        
        # Initialize cognitive profile if needed
        await cognitive_monitor.initialize_user_profile(request.user_id)
        
        return {
            "session_id": session_id,
            "status": "active",
            "started_at": datetime.utcnow().isoformat(),
            "message": "Collaboration session started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start collaboration session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    """Get current status of a collaboration session"""
    try:
        status = await collaboration_engine.get_session_status(session_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/interaction")
async def process_interaction(request: InteractionRequest):
    """Process a human-AI interaction"""
    try:
        # Map string to InteractionType enum
        interaction_type_map = {
            "query_response": InteractionType.QUERY_RESPONSE,
            "collaborative_task": InteractionType.COLLABORATIVE_TASK,
            "decision_support": InteractionType.DECISION_SUPPORT,
            "learning_session": InteractionType.LEARNING_SESSION,
            "feedback_loop": InteractionType.FEEDBACK_LOOP,
            "trust_calibration": InteractionType.TRUST_CALIBRATION
        }
        
        interaction_type = interaction_type_map.get(
            request.interaction_type, 
            InteractionType.QUERY_RESPONSE
        )
        
        response = await collaboration_engine.process_interaction(
            session_id=request.session_id,
            interaction_type=interaction_type,
            interaction_data=request.interaction_data
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to process interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/end-session/{session_id}")
async def end_collaboration_session(session_id: str):
    """End a collaboration session"""
    try:
        result = await collaboration_engine.end_session(session_id)
        return result
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# User Profile Management
@router.get("/user-profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get user collaboration profile"""
    try:
        # Get profile from collaboration engine
        if user_id in collaboration_engine.user_profiles:
            profile = collaboration_engine.user_profiles[user_id]
            return {
                "user_id": profile.user_id,
                "expertise_level": profile.expertise_level,
                "preferred_collaboration_mode": profile.preferred_collaboration_mode.value,
                "trust_level": profile.trust_level,
                "cognitive_style": profile.cognitive_style,
                "interaction_history": profile.interaction_history[-10:],  # Last 10
                "preferences": profile.preferences,
                "performance_metrics": profile.performance_metrics,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            }
        else:
            # Create default profile
            profile = await collaboration_engine.create_user_profile(user_id, {})
            return {
                "user_id": profile.user_id,
                "expertise_level": profile.expertise_level,
                "preferred_collaboration_mode": profile.preferred_collaboration_mode.value,
                "trust_level": profile.trust_level,
                "cognitive_style": profile.cognitive_style,
                "interaction_history": [],
                "preferences": profile.preferences,
                "performance_metrics": profile.performance_metrics,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/update-profile")
async def update_user_profile(request: ProfileUpdateRequest):
    """Update user collaboration profile"""
    try:
        # Get existing profile or create new one
        if request.user_id not in collaboration_engine.user_profiles:
            await collaboration_engine.create_user_profile(request.user_id, {})
        
        profile = collaboration_engine.user_profiles[request.user_id]
        
        # Update fields if provided
        if request.expertise_level is not None:
            profile.expertise_level = request.expertise_level
        if request.preferred_collaboration_mode is not None:
            profile.preferred_collaboration_mode = CollaborationMode(request.preferred_collaboration_mode)
        if request.cognitive_style is not None:
            profile.cognitive_style = request.cognitive_style
        if request.trust_level is not None:
            profile.trust_level = request.trust_level
        if request.preferences is not None:
            profile.preferences.update(request.preferences)
        
        profile.updated_at = datetime.utcnow()
        
        return {"message": "Profile updated successfully"}
        
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Trust Calibration
@router.get("/trust/user-status/{user_id}")
async def get_trust_status(user_id: str):
    """Get trust status for a user"""
    try:
        status = await trust_engine.get_user_trust_status(user_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get trust status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trust/record-event")
async def record_trust_event(request: TrustEventRequest):
    """Record a trust-affecting event"""
    try:
        result = await trust_engine.record_trust_event(
            user_id=request.user_id,
            session_id=request.session_id,
            event_type=request.event_type,
            event_data=request.event_data,
            context=request.context or {}
        )
        return result
    except Exception as e:
        logger.error(f"Failed to record trust event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trust/system-status")
async def get_trust_system_status():
    """Get overall trust system status"""
    try:
        status = await trust_engine.get_system_trust_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get trust system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cognitive Load Monitoring
@router.get("/cognitive-load/user-status/{user_id}")
async def get_cognitive_load_status(user_id: str):
    """Get cognitive load status for a user"""
    try:
        status = await cognitive_monitor.get_user_load_status(user_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get cognitive load status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cognitive-load/measure")
async def measure_cognitive_load(request: CognitiveLoadRequest):
    """Measure current cognitive load"""
    try:
        measurement = await cognitive_monitor.measure_cognitive_load(
            user_id=request.user_id,
            session_id=request.session_id,
            interaction_data=request.interaction_data,
            context=request.context or {}
        )
        
        # Analyze and generate adaptations
        analysis = await cognitive_monitor.analyze_and_adapt(request.user_id)
        
        return {
            "measurement": {
                "measurement_id": measurement.measurement_id,
                "overall_load": measurement.overall_load,
                "load_level": measurement.load_level.value,
                "indicator_scores": {indicator.value: score for indicator, score in measurement.indicator_scores.items()},
                "contributing_factors": measurement.contributing_factors,
                "timestamp": measurement.timestamp.isoformat(),
                "confidence": measurement.measurement_confidence
            },
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Failed to measure cognitive load: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cognitive-load/system-status")
async def get_cognitive_load_system_status():
    """Get cognitive load monitoring system status"""
    try:
        status = await cognitive_monitor.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get cognitive load system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Context Processing
@router.get("/context/current/{user_id}")
async def get_current_context(user_id: str):
    """Get current processed context for a user"""
    try:
        # For this demo, we'll create a mock context based on current state
        interaction_data = {
            "message": "",
            "response_time": 5.0,
            "task_type": "general",
            "complexity": "medium"
        }
        
        user_profile = {}
        if user_id in collaboration_engine.user_profiles:
            profile = collaboration_engine.user_profiles[user_id]
            user_profile = {
                "expertise_level": profile.expertise_level,
                "cognitive_style": profile.cognitive_style,
                "trust_level": profile.trust_level
            }
        else:
            user_profile = {"expertise_level": 0.5, "cognitive_style": "mixed", "trust_level": 0.7}
        
        context = await context_processor.process_comprehensive_context(
            interaction_data=interaction_data,
            user_profile=user_profile
        )
        
        return context
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/health")
async def get_context_health():
    """Get context processing health metrics"""
    try:
        health = await context_processor.get_context_health()
        return health
    except Exception as e:
        logger.error(f"Failed to get context health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Explanation Generation
@router.post("/explanation/generate")
async def generate_explanation(request: ExplanationRequestModel):
    """Generate an AI explanation"""
    try:
        # Create explanation request
        exp_request = ExplanationRequest(
            request_id=f"req_{datetime.utcnow().timestamp()}",
            user_id=request.user_id,
            session_id=request.session_id,
            explanation_target=request.explanation_target,
            context=request.context or {},
            user_profile=request.user_profile or {},
            preferred_type=None,  # Will be determined by the generator
            preferred_style=None,
            preferred_level=None,
            max_length=None,
            include_uncertainty=True,
            timestamp=datetime.utcnow()
        )
        
        explanation = await explanation_generator.generate_explanation(exp_request)
        
        return {
            "explanation_id": explanation.explanation_id,
            "request_id": explanation.request_id,
            "content": explanation.content,
            "explanation_type": explanation.explanation_type.value,
            "explanation_style": explanation.explanation_style.value,
            "explanation_level": explanation.explanation_level.value,
            "confidence": explanation.confidence,
            "completeness": explanation.completeness,
            "clarity_score": explanation.clarity_score,
            "supporting_evidence": explanation.supporting_evidence,
            "key_concepts": explanation.key_concepts,
            "assumptions": explanation.assumptions,
            "limitations": explanation.limitations,
            "follow_up_questions": explanation.follow_up_questions,
            "generated_at": explanation.generated_at.isoformat(),
            "generation_time_ms": explanation.generation_time_ms
        }
        
    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explanation/feedback")
async def record_explanation_feedback(feedback_data: Dict[str, Any]):
    """Record feedback on an explanation"""
    try:
        feedback = ExplanationFeedback(
            feedback_id=f"fb_{datetime.utcnow().timestamp()}",
            explanation_id=feedback_data["explanation_id"],
            user_id=feedback_data["user_id"],
            clarity_rating=feedback_data.get("clarity_rating", 3),
            helpfulness_rating=feedback_data.get("helpfulness_rating", 3),
            completeness_rating=feedback_data.get("completeness_rating", 3),
            satisfaction_rating=feedback_data.get("satisfaction_rating", 3),
            feedback_text=feedback_data.get("feedback_text"),
            improvement_suggestions=feedback_data.get("improvement_suggestions", []),
            timestamp=datetime.utcnow()
        )
        
        result = await explanation_generator.record_feedback(feedback)
        return result
        
    except Exception as e:
        logger.error(f"Failed to record explanation feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explanation/stats")
async def get_explanation_stats():
    """Get explanation generation statistics"""
    try:
        stats = await explanation_generator.get_explanation_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get explanation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System Status
@router.get("/system-status")
async def get_collaboration_system_status():
    """Get overall collaboration system status"""
    try:
        collab_status = await collaboration_engine.get_system_status()
        trust_status = await trust_engine.get_system_trust_status()
        cognitive_status = await cognitive_monitor.get_system_status()
        context_health = await context_processor.get_context_health()
        explanation_stats = await explanation_generator.get_explanation_stats()
        
        return {
            "collaboration_engine": collab_status,
            "trust_calibration": trust_status,
            "cognitive_monitoring": cognitive_status,
            "context_processing": context_health,
            "explanation_generation": explanation_stats,
            "system_health": "operational",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))