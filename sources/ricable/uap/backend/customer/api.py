# File: backend/customer/api.py
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import logging
from uuid import UUID
from pydantic import BaseModel, Field

from ..database.session import get_db
from ..services.auth import get_current_user
from ..models.user import User
from ..models.customer import (
    CustomerProfile, CustomerInteraction, CustomerJourneyEvent, 
    CustomerFeedback, CustomerRecommendation, JourneyStage, 
    SentimentType, InteractionType, SatisfactionLevel
)

from .sentiment_routing import SentimentRouter, RoutingDecision
from .journey_tracking import JourneyTracker, JourneyAnalyzer, JourneyMetrics
from .personalization import (
    PersonalizationEngine, RecommendationEngine, 
    RecommendationRequest, PersonalizationStrategy, RecommendationType
)
from .feedback_system import (
    FeedbackCollector, SatisfactionTracker, 
    FeedbackRequest, FeedbackResponse, FeedbackType, SurveyTemplate, FeedbackTrigger
)

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class InteractionRequest(BaseModel):
    message: str
    interaction_type: InteractionType
    context: Optional[Dict[str, Any]] = {}

class InteractionResponse(BaseModel):
    interaction_id: str
    routing_decision: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    estimated_wait_time: Optional[int]

class JourneyEventRequest(BaseModel):
    event_name: str
    event_type: str
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = {}
    event_value: Optional[float] = None

class PersonalizationRequest(BaseModel):
    recommendation_types: List[RecommendationType]
    max_recommendations: int = 5
    strategy: PersonalizationStrategy = PersonalizationStrategy.HYBRID
    context: Optional[Dict[str, Any]] = {}

class FeedbackSubmissionRequest(BaseModel):
    feedback_type: str
    ratings: Dict[str, int]
    text_feedback: Optional[str] = None
    responses: Optional[Dict[str, Any]] = {}
    interaction_id: Optional[str] = None

class CustomerProfileResponse(BaseModel):
    id: str
    customer_segment: Optional[str]
    customer_tier: Optional[str]
    current_journey_stage: str
    current_satisfaction: Optional[str]
    lifetime_value: float
    activity_score: float
    churn_risk_score: float
    total_interactions: int
    satisfaction_updated_at: Optional[datetime]
    created_at: datetime

class CustomerExperienceAPI:
    """Customer Experience API Router"""
    
    def __init__(self):
        self.router = APIRouter(prefix="/api/customer-experience", tags=["Customer Experience"])
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""
        
        # Customer Profile endpoints
        self.router.add_api_route(
            "/profile/{customer_id}",
            self.get_customer_profile,
            methods=["GET"],
            response_model=CustomerProfileResponse
        )
        
        self.router.add_api_route(
            "/profile/{customer_id}",
            self.update_customer_profile,
            methods=["PUT"]
        )
        
        # Sentiment and Routing endpoints
        self.router.add_api_route(
            "/interactions/{customer_id}/route",
            self.route_customer_interaction,
            methods=["POST"],
            response_model=InteractionResponse
        )
        
        self.router.add_api_route(
            "/routing/analytics",
            self.get_routing_analytics,
            methods=["GET"]
        )
        
        # Journey Tracking endpoints
        self.router.add_api_route(
            "/journey/{customer_id}/events",
            self.track_journey_event,
            methods=["POST"]
        )
        
        self.router.add_api_route(
            "/journey/{customer_id}/metrics",
            self.get_journey_metrics,
            methods=["GET"]
        )
        
        self.router.add_api_route(
            "/journey/{customer_id}/insights",
            self.get_journey_insights,
            methods=["GET"]
        )
        
        self.router.add_api_route(
            "/journey/analytics/stages",
            self.get_stage_analytics,
            methods=["GET"]
        )
        
        # Personalization endpoints
        self.router.add_api_route(
            "/personalization/{customer_id}/recommendations",
            self.get_personalized_recommendations,
            methods=["POST"]
        )
        
        self.router.add_api_route(
            "/personalization/{customer_id}/profile",
            self.get_personalization_profile,
            methods=["GET"]
        )
        
        self.router.add_api_route(
            "/recommendations/{recommendation_id}/track",
            self.track_recommendation_interaction,
            methods=["POST"]
        )
        
        # Feedback endpoints
        self.router.add_api_route(
            "/feedback/{customer_id}/request",
            self.request_customer_feedback,
            methods=["POST"]
        )
        
        self.router.add_api_route(
            "/feedback/{customer_id}/submit",
            self.submit_feedback,
            methods=["POST"]
        )
        
        self.router.add_api_route(
            "/feedback/{customer_id}/satisfaction",
            self.get_satisfaction_metrics,
            methods=["GET"]
        )
        
        self.router.add_api_route(
            "/feedback/{customer_id}/analysis",
            self.get_feedback_analysis,
            methods=["GET"]
        )
        
        self.router.add_api_route(
            "/feedback/benchmarks",
            self.get_satisfaction_benchmarks,
            methods=["GET"]
        )
        
        # Dashboard and Analytics endpoints
        self.router.add_api_route(
            "/dashboard/{customer_id}",
            self.get_customer_dashboard,
            methods=["GET"]
        )
        
        self.router.add_api_route(
            "/analytics/overview",
            self.get_analytics_overview,
            methods=["GET"]
        )

    async def get_customer_profile(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ) -> CustomerProfileResponse:
        """Get customer profile information"""
        try:
            customer = db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if not customer:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer profile not found"
                )
            
            return CustomerProfileResponse(
                id=str(customer.id),
                customer_segment=customer.customer_segment,
                customer_tier=customer.customer_tier,
                current_journey_stage=customer.current_journey_stage.value,
                current_satisfaction=customer.current_satisfaction.value if customer.current_satisfaction else None,
                lifetime_value=customer.lifetime_value,
                activity_score=customer.activity_score,
                churn_risk_score=customer.churn_risk_score,
                total_interactions=customer.total_interactions,
                satisfaction_updated_at=customer.satisfaction_updated_at,
                created_at=customer.created_at
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting customer profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve customer profile"
            )

    async def update_customer_profile(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        profile_data: Dict[str, Any] = Body(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Update customer profile"""
        try:
            customer = db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if not customer:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer profile not found"
                )
            
            # Update allowed fields
            updatable_fields = [
                'customer_segment', 'customer_tier', 'communication_preferences',
                'feature_preferences', 'personalization_tags', 'custom_attributes'
            ]
            
            for field, value in profile_data.items():
                if field in updatable_fields and hasattr(customer, field):
                    setattr(customer, field, value)
            
            db.commit()
            
            return {"message": "Customer profile updated successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating customer profile: {e}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update customer profile"
            )

    async def route_customer_interaction(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        request: InteractionRequest = Body(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ) -> InteractionResponse:
        """Route customer interaction based on sentiment and context"""
        try:
            sentiment_router = SentimentRouter(db)
            
            routing_decision = await sentiment_router.route_customer_interaction(
                customer_id=customer_id,
                message=request.message,
                interaction_type=request.interaction_type,
                context=request.context
            )
            
            return InteractionResponse(
                interaction_id=f"interaction_{customer_id}_{datetime.now().timestamp()}",
                routing_decision={
                    "channel": routing_decision.channel.value,
                    "priority": routing_decision.priority.value,
                    "agent_type": routing_decision.agent_type,
                    "routing_reason": routing_decision.routing_reason
                },
                sentiment_analysis=routing_decision.context,
                estimated_wait_time=routing_decision.estimated_wait_time
            )
            
        except Exception as e:
            logger.error(f"Error routing customer interaction: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to route customer interaction"
            )

    async def get_routing_analytics(
        self,
        days: int = Query(30, description="Number of days for analytics"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get routing analytics"""
        try:
            sentiment_router = SentimentRouter(db)
            analytics = await sentiment_router.get_routing_analytics(days)
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting routing analytics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve routing analytics"
            )

    async def track_journey_event(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        request: JourneyEventRequest = Body(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Track a customer journey event"""
        try:
            journey_tracker = JourneyTracker(db)
            
            from .journey_tracking import EventType, TriggerSource
            
            event = await journey_tracker.track_event(
                customer_id=customer_id,
                event_type=EventType(request.event_type),
                event_name=request.event_name,
                description=request.description,
                properties=request.properties,
                event_value=request.event_value
            )
            
            if event:
                return {
                    "event_id": str(event.id),
                    "message": "Journey event tracked successfully"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to track journey event"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error tracking journey event: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to track journey event"
            )

    async def get_journey_metrics(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get customer journey metrics"""
        try:
            journey_tracker = JourneyTracker(db)
            metrics = await journey_tracker.get_customer_journey_metrics(customer_id)
            
            return {
                "current_stage": metrics.current_stage.value,
                "days_in_current_stage": metrics.days_in_current_stage,
                "total_journey_days": metrics.total_journey_days,
                "stages_completed": [stage.value for stage in metrics.stages_completed],
                "conversion_rate": metrics.conversion_rate,
                "engagement_score": metrics.engagement_score,
                "satisfaction_trend": metrics.satisfaction_trend,
                "key_events": metrics.key_events
            }
            
        except Exception as e:
            logger.error(f"Error getting journey metrics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve journey metrics"
            )

    async def get_journey_insights(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get customer journey insights"""
        try:
            journey_analyzer = JourneyAnalyzer(db)
            insights = await journey_analyzer.analyze_customer_journey(customer_id)
            
            return {
                "insights": [
                    {
                        "type": insight.insight_type,
                        "description": insight.description,
                        "impact_score": insight.impact_score,
                        "recommended_actions": insight.recommended_actions,
                        "urgency": insight.urgency,
                        "context": insight.context
                    }
                    for insight in insights
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting journey insights: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve journey insights"
            )

    async def get_stage_analytics(
        self,
        days: int = Query(90, description="Number of days for analytics"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get analytics for all journey stages"""
        try:
            journey_analyzer = JourneyAnalyzer(db)
            analytics = await journey_analyzer.get_stage_analytics(days)
            
            return {
                "stage_analytics": [
                    {
                        "stage": stage_analytics.stage.value,
                        "total_customers": stage_analytics.total_customers,
                        "avg_duration_days": stage_analytics.avg_duration_days,
                        "conversion_rate": stage_analytics.conversion_rate,
                        "common_exit_points": stage_analytics.common_exit_points,
                        "success_factors": stage_analytics.success_factors,
                        "bottlenecks": stage_analytics.bottlenecks
                    }
                    for stage_analytics in analytics
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting stage analytics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve stage analytics"
            )

    async def get_personalized_recommendations(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        request: PersonalizationRequest = Body(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get personalized recommendations for customer"""
        try:
            recommendation_engine = RecommendationEngine(db)
            
            rec_request = RecommendationRequest(
                customer_id=customer_id,
                context=request.context,
                recommendation_types=request.recommendation_types,
                max_recommendations=request.max_recommendations,
                strategy=request.strategy
            )
            
            recommendations = await recommendation_engine.generate_recommendations(rec_request)
            
            return {
                "recommendations": [
                    {
                        "id": rec.recommendation_id,
                        "type": rec.type.value,
                        "title": rec.title,
                        "description": rec.description,
                        "confidence_score": rec.confidence_score,
                        "relevance_score": rec.relevance_score,
                        "priority": rec.priority,
                        "content_data": rec.content_data,
                        "action_url": rec.action_url,
                        "image_url": rec.image_url,
                        "personalization_factors": rec.personalization_factors,
                        "reasoning": rec.reasoning
                    }
                    for rec in recommendations
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate personalized recommendations"
            )

    async def get_personalization_profile(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get customer personalization profile"""
        try:
            personalization_engine = PersonalizationEngine(db)
            profile = await personalization_engine.build_personalization_profile(customer_id)
            
            return {
                "customer_id": profile.customer_id,
                "preferences": profile.preferences,
                "behavior_patterns": profile.behavior_patterns,
                "journey_context": profile.journey_context,
                "segmentation_data": profile.segmentation_data,
                "engagement_metrics": profile.engagement_metrics,
                "recommendation_history_count": len(profile.recommendation_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting personalization profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve personalization profile"
            )

    async def track_recommendation_interaction(
        self,
        recommendation_id: str = Path(..., description="Recommendation ID"),
        interaction_data: Dict[str, Any] = Body(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Track interaction with recommendation"""
        try:
            recommendation_engine = RecommendationEngine(db)
            
            await recommendation_engine.track_recommendation_interaction(
                recommendation_id=recommendation_id,
                interaction_type=interaction_data.get("interaction_type", "view"),
                customer_id=interaction_data.get("customer_id")
            )
            
            return {"message": "Recommendation interaction tracked successfully"}
            
        except Exception as e:
            logger.error(f"Error tracking recommendation interaction: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to track recommendation interaction"
            )

    async def request_customer_feedback(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        feedback_request_data: Dict[str, Any] = Body(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Request feedback from customer"""
        try:
            feedback_collector = FeedbackCollector(db)
            
            request = FeedbackRequest(
                customer_id=customer_id,
                feedback_type=FeedbackType(feedback_request_data.get("feedback_type", "post_interaction")),
                survey_template=SurveyTemplate(feedback_request_data.get("survey_template", "csat")),
                trigger=FeedbackTrigger(feedback_request_data.get("trigger", "manual_request")),
                context=feedback_request_data.get("context", {}),
                priority=feedback_request_data.get("priority", "medium")
            )
            
            request_id = await feedback_collector.request_feedback(request)
            
            if request_id:
                return {
                    "request_id": request_id,
                    "message": "Feedback request sent successfully"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot request feedback at this time"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error requesting feedback: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to request feedback"
            )

    async def submit_feedback(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        request: FeedbackSubmissionRequest = Body(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Submit customer feedback"""
        try:
            feedback_collector = FeedbackCollector(db)
            
            # Determine sentiment and satisfaction from ratings
            sentiment = SentimentType.NEUTRAL
            satisfaction = SatisfactionLevel.NEUTRAL
            
            if request.text_feedback:
                # Use sentiment analyzer
                from .sentiment_routing import SentimentAnalyzer
                analyzer = SentimentAnalyzer()
                sentiment_result = await analyzer.analyze_sentiment(request.text_feedback)
                sentiment = sentiment_result.sentiment
            
            # Map ratings to satisfaction
            overall_rating = request.ratings.get('overall_satisfaction', 3)
            if overall_rating >= 4:
                satisfaction = SatisfactionLevel.SATISFIED
            elif overall_rating <= 2:
                satisfaction = SatisfactionLevel.DISSATISFIED
            
            response = FeedbackResponse(
                request_id=f"feedback_{customer_id}_{datetime.now().timestamp()}",
                customer_id=customer_id,
                responses=request.responses or {},
                ratings=request.ratings,
                text_feedback=request.text_feedback or "",
                sentiment=sentiment,
                satisfaction_level=satisfaction,
                completion_rate=1.0,  # Assume complete for now
                submitted_at=datetime.now(timezone.utc)
            )
            
            feedback = await feedback_collector.process_feedback_response(response)
            
            if feedback:
                return {
                    "feedback_id": str(feedback.id),
                    "message": "Feedback submitted successfully"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to process feedback"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to submit feedback"
            )

    async def get_satisfaction_metrics(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        days: int = Query(90, description="Number of days for metrics"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get customer satisfaction metrics"""
        try:
            satisfaction_tracker = SatisfactionTracker(db)
            metrics = await satisfaction_tracker.get_satisfaction_metrics(customer_id, days)
            
            return {
                "csat_score": metrics.csat_score,
                "nps_score": metrics.nps_score,
                "ces_score": metrics.ces_score,
                "satisfaction_trend": metrics.satisfaction_trend,
                "response_rate": metrics.response_rate,
                "completion_rate": metrics.completion_rate,
                "time_to_respond": metrics.time_to_respond,
                "feedback_volume": metrics.feedback_volume
            }
            
        except Exception as e:
            logger.error(f"Error getting satisfaction metrics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve satisfaction metrics"
            )

    async def get_feedback_analysis(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        days: int = Query(180, description="Number of days for analysis"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get feedback analysis for customer"""
        try:
            satisfaction_tracker = SatisfactionTracker(db)
            analysis = await satisfaction_tracker.analyze_feedback_trends(customer_id, days)
            
            return {
                "overall_sentiment": analysis.overall_sentiment.value,
                "satisfaction_trend": analysis.satisfaction_trend,
                "key_themes": analysis.key_themes,
                "improvement_areas": analysis.improvement_areas,
                "satisfaction_drivers": analysis.satisfaction_drivers,
                "risk_indicators": analysis.risk_indicators,
                "recommendations": analysis.recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback analysis: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve feedback analysis"
            )

    async def get_satisfaction_benchmarks(
        self,
        segment: Optional[str] = Query(None, description="Customer segment for benchmarking"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get satisfaction benchmarks"""
        try:
            satisfaction_tracker = SatisfactionTracker(db)
            benchmarks = await satisfaction_tracker.get_satisfaction_benchmarks(segment)
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error getting satisfaction benchmarks: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve satisfaction benchmarks"
            )

    async def get_customer_dashboard(
        self,
        customer_id: str = Path(..., description="Customer ID"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get comprehensive customer dashboard data"""
        try:
            # Get customer profile
            customer = db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if not customer:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer not found"
                )
            
            # Get journey metrics
            journey_tracker = JourneyTracker(db)
            journey_metrics = await journey_tracker.get_customer_journey_metrics(customer_id)
            
            # Get satisfaction metrics
            satisfaction_tracker = SatisfactionTracker(db)
            satisfaction_metrics = await satisfaction_tracker.get_satisfaction_metrics(customer_id)
            
            # Get recent interactions
            recent_interactions = db.query(CustomerInteraction).filter(
                CustomerInteraction.customer_id == customer_id
            ).order_by(CustomerInteraction.started_at.desc()).limit(10).all()
            
            # Get active recommendations
            active_recommendations = db.query(CustomerRecommendation).filter(
                CustomerRecommendation.customer_id == customer_id,
                CustomerRecommendation.is_active == True
            ).order_by(CustomerRecommendation.created_at.desc()).limit(5).all()
            
            return {
                "customer_profile": {
                    "id": str(customer.id),
                    "segment": customer.customer_segment,
                    "tier": customer.customer_tier,
                    "lifetime_value": customer.lifetime_value,
                    "current_stage": customer.current_journey_stage.value,
                    "activity_score": customer.activity_score,
                    "churn_risk_score": customer.churn_risk_score
                },
                "journey_metrics": {
                    "current_stage": journey_metrics.current_stage.value,
                    "days_in_stage": journey_metrics.days_in_current_stage,
                    "conversion_rate": journey_metrics.conversion_rate,
                    "engagement_score": journey_metrics.engagement_score
                },
                "satisfaction_metrics": {
                    "csat_score": satisfaction_metrics.csat_score,
                    "nps_score": satisfaction_metrics.nps_score,
                    "satisfaction_trend": satisfaction_metrics.satisfaction_trend[-5:] if satisfaction_metrics.satisfaction_trend else []
                },
                "recent_interactions": [
                    {
                        "id": str(interaction.id),
                        "type": interaction.interaction_type.value,
                        "sentiment": interaction.sentiment.value if interaction.sentiment else None,
                        "satisfaction": interaction.satisfaction_rating,
                        "timestamp": interaction.started_at.isoformat()
                    }
                    for interaction in recent_interactions
                ],
                "active_recommendations": [
                    {
                        "id": str(rec.id),
                        "type": rec.recommendation_type,
                        "title": rec.title,
                        "confidence": rec.confidence_score,
                        "created_at": rec.created_at.isoformat()
                    }
                    for rec in active_recommendations
                ]
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting customer dashboard: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve customer dashboard"
            )

    async def get_analytics_overview(
        self,
        days: int = Query(30, description="Number of days for analytics"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get customer experience analytics overview"""
        try:
            from_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get total customers
            total_customers = db.query(CustomerProfile).count()
            
            # Get satisfaction metrics
            satisfaction_tracker = SatisfactionTracker(db)
            benchmarks = await satisfaction_tracker.get_satisfaction_benchmarks()
            
            # Get routing analytics
            sentiment_router = SentimentRouter(db)
            routing_analytics = await sentiment_router.get_routing_analytics(days)
            
            # Get journey stage distribution
            stage_distribution = {}
            for stage in JourneyStage:
                count = db.query(CustomerProfile).filter(
                    CustomerProfile.current_journey_stage == stage
                ).count()
                stage_distribution[stage.value] = count
            
            # Get churn risk distribution
            high_risk_customers = db.query(CustomerProfile).filter(
                CustomerProfile.churn_risk_score > 0.7
            ).count()
            
            medium_risk_customers = db.query(CustomerProfile).filter(
                CustomerProfile.churn_risk_score.between(0.4, 0.7)
            ).count()
            
            return {
                "overview": {
                    "total_customers": total_customers,
                    "high_risk_customers": high_risk_customers,
                    "medium_risk_customers": medium_risk_customers,
                    "avg_satisfaction": benchmarks.get("avg_csat", 0),
                    "avg_nps": benchmarks.get("avg_nps", 0)
                },
                "satisfaction_benchmarks": benchmarks,
                "routing_analytics": routing_analytics,
                "stage_distribution": stage_distribution,
                "analysis_period_days": days
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics overview: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve analytics overview"
            )

# Create router instance
customer_experience_api = CustomerExperienceAPI()
router = customer_experience_api.router