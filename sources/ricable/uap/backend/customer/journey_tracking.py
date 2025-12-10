# File: backend/customer/journey_tracking.py
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import json

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func

from ..models.customer import (
    CustomerProfile, CustomerJourneyEvent, CustomerInteraction, 
    JourneyStage, SentimentType, InteractionType
)
from ..models.user import User
from ..database.session import get_db

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Journey event types"""
    STAGE_CHANGE = "stage_change"
    MILESTONE_REACHED = "milestone_reached"
    ACTION_COMPLETED = "action_completed"
    BEHAVIOR_DETECTED = "behavior_detected"
    ENGAGEMENT_EVENT = "engagement_event"
    SUPPORT_INTERACTION = "support_interaction"

class TriggerSource(Enum):
    """Event trigger sources"""
    USER_ACTION = "user_action"
    SYSTEM_DETECTION = "system_detection"
    MANUAL_UPDATE = "manual_update"
    AUTOMATED_RULE = "automated_rule"
    INTEGRATION_EVENT = "integration_event"

@dataclass
class JourneyMetrics:
    """Customer journey metrics"""
    current_stage: JourneyStage
    days_in_current_stage: int
    total_journey_days: int
    stages_completed: List[JourneyStage]
    conversion_rate: float
    engagement_score: float
    satisfaction_trend: List[float]
    key_events: List[Dict[str, Any]]

@dataclass
class JourneyInsight:
    """Journey analysis insight"""
    insight_type: str
    description: str
    impact_score: float
    recommended_actions: List[str]
    urgency: str  # low, medium, high
    context: Dict[str, Any]

@dataclass
class StageAnalytics:
    """Analytics for a specific journey stage"""
    stage: JourneyStage
    total_customers: int
    avg_duration_days: float
    conversion_rate: float
    common_exit_points: List[str]
    success_factors: List[str]
    bottlenecks: List[str]

class JourneyTracker:
    """Tracks customer journey progression and events"""
    
    def __init__(self, db: Session):
        self.db = db
        self.stage_progression_rules = self._initialize_stage_rules()
        self.milestone_definitions = self._initialize_milestones()

    def _initialize_stage_rules(self) -> Dict[JourneyStage, Dict]:
        """Initialize stage progression rules"""
        return {
            JourneyStage.AWARENESS: {
                'next_stages': [JourneyStage.CONSIDERATION],
                'auto_progression_events': ['visited_pricing', 'downloaded_content', 'signed_up'],
                'duration_threshold_days': 30,
                'required_actions': ['initial_engagement']
            },
            JourneyStage.CONSIDERATION: {
                'next_stages': [JourneyStage.TRIAL, JourneyStage.PURCHASE],
                'auto_progression_events': ['started_trial', 'requested_demo', 'contacted_sales'],
                'duration_threshold_days': 14,
                'required_actions': ['product_exploration']
            },
            JourneyStage.TRIAL: {
                'next_stages': [JourneyStage.PURCHASE, JourneyStage.CHURNED],
                'auto_progression_events': ['made_purchase', 'trial_expired'],
                'duration_threshold_days': 14,
                'required_actions': ['feature_usage']
            },
            JourneyStage.PURCHASE: {
                'next_stages': [JourneyStage.ONBOARDING],
                'auto_progression_events': ['completed_setup', 'first_login'],
                'duration_threshold_days': 3,
                'required_actions': ['payment_completed']
            },
            JourneyStage.ONBOARDING: {
                'next_stages': [JourneyStage.ACTIVE_USE],
                'auto_progression_events': ['completed_onboarding', 'first_successful_use'],
                'duration_threshold_days': 7,
                'required_actions': ['setup_completion']
            },
            JourneyStage.ACTIVE_USE: {
                'next_stages': [JourneyStage.RENEWAL, JourneyStage.EXPANSION, JourneyStage.CHURN_RISK],
                'auto_progression_events': ['renewal_due', 'usage_increase', 'usage_decline'],
                'duration_threshold_days': 365,
                'required_actions': ['regular_usage']
            },
            JourneyStage.RENEWAL: {
                'next_stages': [JourneyStage.ACTIVE_USE, JourneyStage.EXPANSION, JourneyStage.CHURNED],
                'auto_progression_events': ['renewed_subscription', 'renewal_failed'],
                'duration_threshold_days': 30,
                'required_actions': ['renewal_decision']
            },
            JourneyStage.EXPANSION: {
                'next_stages': [JourneyStage.ACTIVE_USE],
                'auto_progression_events': ['upgrade_completed', 'additional_purchase'],
                'duration_threshold_days': 14,
                'required_actions': ['expansion_decision']
            },
            JourneyStage.CHURN_RISK: {
                'next_stages': [JourneyStage.ACTIVE_USE, JourneyStage.CHURNED],
                'auto_progression_events': ['reactivated', 'churned'],
                'duration_threshold_days': 90,
                'required_actions': ['retention_intervention']
            }
        }

    def _initialize_milestones(self) -> Dict[str, Dict]:
        """Initialize milestone definitions"""
        return {
            'first_login': {
                'stage': JourneyStage.ONBOARDING,
                'description': 'Customer logged in for the first time',
                'value': 10,
                'required_for_progression': True
            },
            'completed_setup': {
                'stage': JourneyStage.ONBOARDING,
                'description': 'Customer completed initial setup',
                'value': 25,
                'required_for_progression': True
            },
            'first_successful_use': {
                'stage': JourneyStage.ONBOARDING,
                'description': 'Customer successfully used core feature',
                'value': 50,
                'required_for_progression': True
            },
            'invited_team_member': {
                'stage': JourneyStage.ACTIVE_USE,
                'description': 'Customer invited team members',
                'value': 30,
                'required_for_progression': False
            },
            'created_first_project': {
                'stage': JourneyStage.ACTIVE_USE,
                'description': 'Customer created their first project',
                'value': 35,
                'required_for_progression': False
            },
            'reached_usage_limit': {
                'stage': JourneyStage.ACTIVE_USE,
                'description': 'Customer reached usage limits (expansion opportunity)',
                'value': 40,
                'required_for_progression': False
            },
            'provided_feedback': {
                'stage': JourneyStage.ACTIVE_USE,
                'description': 'Customer provided product feedback',
                'value': 20,
                'required_for_progression': False
            },
            'referred_customer': {
                'stage': JourneyStage.ACTIVE_USE,
                'description': 'Customer referred another customer',
                'value': 60,
                'required_for_progression': False
            }
        }

    async def track_event(
        self,
        customer_id: str,
        event_type: EventType,
        event_name: str,
        description: str = None,
        properties: Dict[str, Any] = None,
        trigger_source: TriggerSource = TriggerSource.SYSTEM_DETECTION,
        event_value: float = None
    ) -> Optional[CustomerJourneyEvent]:
        """Track a customer journey event"""
        try:
            customer = self.db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if not customer:
                logger.warning(f"Customer {customer_id} not found")
                return None

            # Create journey event
            journey_event = CustomerJourneyEvent(
                customer_id=customer_id,
                event_type=event_type.value,
                event_name=event_name,
                event_description=description,
                from_stage=customer.current_journey_stage,
                to_stage=customer.current_journey_stage,  # Will be updated if stage changes
                event_value=event_value,
                trigger_source=trigger_source.value,
                properties=properties or {},
                occurred_at=datetime.now(timezone.utc)
            )

            # Check if this event should trigger a stage change
            new_stage = await self._check_stage_progression(customer, event_name, event_type)
            if new_stage and new_stage != customer.current_journey_stage:
                journey_event.to_stage = new_stage
                await self._update_customer_stage(customer, new_stage)

            self.db.add(journey_event)
            self.db.commit()
            
            # Update customer metrics
            await self._update_customer_metrics(customer_id)
            
            return journey_event

        except Exception as e:
            logger.error(f"Error tracking journey event: {e}")
            self.db.rollback()
            return None

    async def _check_stage_progression(
        self,
        customer: CustomerProfile,
        event_name: str,
        event_type: EventType
    ) -> Optional[JourneyStage]:
        """Check if event should trigger stage progression"""
        current_stage = customer.current_journey_stage
        stage_rules = self.stage_progression_rules.get(current_stage, {})
        
        # Check auto-progression events
        auto_events = stage_rules.get('auto_progression_events', [])
        if event_name in auto_events:
            next_stages = stage_rules.get('next_stages', [])
            if next_stages:
                return self._determine_next_stage(customer, event_name, next_stages)
        
        # Check duration-based progression
        if customer.journey_stage_updated_at:
            days_in_stage = (datetime.now(timezone.utc) - customer.journey_stage_updated_at).days
            threshold = stage_rules.get('duration_threshold_days', 30)
            
            if days_in_stage >= threshold:
                return self._determine_next_stage_by_duration(customer, current_stage)
        
        return None

    def _determine_next_stage(
        self,
        customer: CustomerProfile,
        event_name: str,
        possible_stages: List[JourneyStage]
    ) -> JourneyStage:
        """Determine the next stage based on event context"""
        
        # Event-specific stage mapping
        event_stage_mapping = {
            'started_trial': JourneyStage.TRIAL,
            'made_purchase': JourneyStage.PURCHASE,
            'completed_setup': JourneyStage.ONBOARDING,
            'first_successful_use': JourneyStage.ACTIVE_USE,
            'renewal_due': JourneyStage.RENEWAL,
            'usage_increase': JourneyStage.EXPANSION,
            'usage_decline': JourneyStage.CHURN_RISK,
            'churned': JourneyStage.CHURNED,
            'trial_expired': JourneyStage.CHURNED if customer.current_journey_stage == JourneyStage.TRIAL else JourneyStage.CHURN_RISK
        }
        
        if event_name in event_stage_mapping:
            target_stage = event_stage_mapping[event_name]
            if target_stage in possible_stages:
                return target_stage
        
        # Default to first possible stage
        return possible_stages[0] if possible_stages else customer.current_journey_stage

    def _determine_next_stage_by_duration(
        self,
        customer: CustomerProfile,
        current_stage: JourneyStage
    ) -> Optional[JourneyStage]:
        """Determine next stage based on duration thresholds"""
        
        # Analyze customer behavior to determine progression
        recent_interactions = self.db.query(CustomerInteraction).filter(
            and_(
                CustomerInteraction.customer_id == customer.id,
                CustomerInteraction.started_at >= datetime.now(timezone.utc) - timedelta(days=30)
            )
        ).count()
        
        if recent_interactions == 0:
            # No recent activity might indicate churn risk
            if current_stage in [JourneyStage.ACTIVE_USE, JourneyStage.TRIAL]:
                return JourneyStage.CHURN_RISK
        
        # Default progression based on stage
        default_progressions = {
            JourneyStage.AWARENESS: JourneyStage.CONSIDERATION,
            JourneyStage.CONSIDERATION: JourneyStage.TRIAL,
            JourneyStage.TRIAL: JourneyStage.CHURNED,  # If trial expired without purchase
            JourneyStage.PURCHASE: JourneyStage.ONBOARDING,
            JourneyStage.ONBOARDING: JourneyStage.ACTIVE_USE,
            JourneyStage.CHURN_RISK: JourneyStage.CHURNED
        }
        
        return default_progressions.get(current_stage)

    async def _update_customer_stage(self, customer: CustomerProfile, new_stage: JourneyStage):
        """Update customer's journey stage"""
        try:
            customer.current_journey_stage = new_stage
            customer.journey_stage_updated_at = datetime.now(timezone.utc)
            self.db.commit()
            
            logger.info(f"Customer {customer.id} progressed to stage {new_stage.value}")
            
        except Exception as e:
            logger.error(f"Error updating customer stage: {e}")
            self.db.rollback()

    async def _update_customer_metrics(self, customer_id: str):
        """Update customer activity and engagement metrics"""
        try:
            customer = self.db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if not customer:
                return
            
            # Calculate activity score based on recent events
            recent_events = self.db.query(CustomerJourneyEvent).filter(
                and_(
                    CustomerJourneyEvent.customer_id == customer_id,
                    CustomerJourneyEvent.occurred_at >= datetime.now(timezone.utc) - timedelta(days=30)
                )
            ).count()
            
            # Activity score: 0-100 based on events in last 30 days
            customer.activity_score = min(100, recent_events * 10)
            
            # Update last interaction time
            customer.last_interaction_at = datetime.now(timezone.utc)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating customer metrics: {e}")
            self.db.rollback()

    async def get_customer_journey_metrics(self, customer_id: str) -> JourneyMetrics:
        """Get comprehensive journey metrics for a customer"""
        try:
            customer = self.db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if not customer:
                raise ValueError(f"Customer {customer_id} not found")
            
            # Get journey events
            events = self.db.query(CustomerJourneyEvent).filter(
                CustomerJourneyEvent.customer_id == customer_id
            ).order_by(CustomerJourneyEvent.occurred_at.desc()).all()
            
            # Calculate metrics
            current_stage = customer.current_journey_stage
            
            # Days in current stage
            days_in_current_stage = 0
            if customer.journey_stage_updated_at:
                days_in_current_stage = (datetime.now(timezone.utc) - customer.journey_stage_updated_at).days
            
            # Total journey days
            total_journey_days = (datetime.now(timezone.utc) - customer.created_at).days
            
            # Stages completed
            stages_completed = []
            for event in events:
                if event.event_type == EventType.STAGE_CHANGE.value and event.to_stage:
                    if event.to_stage not in stages_completed:
                        stages_completed.append(event.to_stage)
            
            # Conversion rate (simplified)
            stage_values = {
                JourneyStage.AWARENESS: 0.1,
                JourneyStage.CONSIDERATION: 0.2,
                JourneyStage.TRIAL: 0.4,
                JourneyStage.PURCHASE: 0.6,
                JourneyStage.ONBOARDING: 0.7,
                JourneyStage.ACTIVE_USE: 0.8,
                JourneyStage.RENEWAL: 0.9,
                JourneyStage.EXPANSION: 1.0
            }
            conversion_rate = stage_values.get(current_stage, 0.0)
            
            # Engagement score
            engagement_score = customer.activity_score
            
            # Satisfaction trend (last 6 months)
            satisfaction_trend = await self._calculate_satisfaction_trend(customer_id)
            
            # Key events
            key_events = [
                {
                    'event_name': event.event_name,
                    'event_type': event.event_type,
                    'occurred_at': event.occurred_at.isoformat(),
                    'description': event.event_description,
                    'value': event.event_value
                }
                for event in events[:10]  # Last 10 events
            ]
            
            return JourneyMetrics(
                current_stage=current_stage,
                days_in_current_stage=days_in_current_stage,
                total_journey_days=total_journey_days,
                stages_completed=stages_completed,
                conversion_rate=conversion_rate,
                engagement_score=engagement_score,
                satisfaction_trend=satisfaction_trend,
                key_events=key_events
            )
            
        except Exception as e:
            logger.error(f"Error getting journey metrics: {e}")
            return JourneyMetrics(
                current_stage=JourneyStage.AWARENESS,
                days_in_current_stage=0,
                total_journey_days=0,
                stages_completed=[],
                conversion_rate=0.0,
                engagement_score=0.0,
                satisfaction_trend=[],
                key_events=[]
            )

    async def _calculate_satisfaction_trend(self, customer_id: str) -> List[float]:
        """Calculate satisfaction trend over time"""
        try:
            # Get interactions with satisfaction ratings from last 6 months
            six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
            
            interactions = self.db.query(CustomerInteraction).filter(
                and_(
                    CustomerInteraction.customer_id == customer_id,
                    CustomerInteraction.started_at >= six_months_ago,
                    CustomerInteraction.satisfaction_rating.isnot(None)
                )
            ).order_by(CustomerInteraction.started_at).all()
            
            if not interactions:
                return []
            
            # Group by month and calculate average satisfaction
            monthly_satisfaction = {}
            for interaction in interactions:
                month_key = interaction.started_at.strftime('%Y-%m')
                if month_key not in monthly_satisfaction:
                    monthly_satisfaction[month_key] = []
                monthly_satisfaction[month_key].append(interaction.satisfaction_rating)
            
            # Calculate monthly averages
            trend = []
            for month in sorted(monthly_satisfaction.keys()):
                avg_satisfaction = sum(monthly_satisfaction[month]) / len(monthly_satisfaction[month])
                trend.append(avg_satisfaction)
            
            return trend
            
        except Exception as e:
            logger.error(f"Error calculating satisfaction trend: {e}")
            return []

class JourneyAnalyzer:
    """Analyzes customer journey patterns and provides insights"""
    
    def __init__(self, db: Session):
        self.db = db

    async def analyze_customer_journey(self, customer_id: str) -> List[JourneyInsight]:
        """Analyze customer journey and provide insights"""
        try:
            customer = self.db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if not customer:
                return []
            
            insights = []
            
            # Churn risk analysis
            churn_insight = await self._analyze_churn_risk(customer)
            if churn_insight:
                insights.append(churn_insight)
            
            # Engagement analysis
            engagement_insight = await self._analyze_engagement(customer)
            if engagement_insight:
                insights.append(engagement_insight)
            
            # Expansion opportunity analysis
            expansion_insight = await self._analyze_expansion_opportunity(customer)
            if expansion_insight:
                insights.append(expansion_insight)
            
            # Satisfaction analysis
            satisfaction_insight = await self._analyze_satisfaction(customer)
            if satisfaction_insight:
                insights.append(satisfaction_insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing customer journey: {e}")
            return []

    async def _analyze_churn_risk(self, customer: CustomerProfile) -> Optional[JourneyInsight]:
        """Analyze churn risk indicators"""
        risk_factors = []
        risk_score = customer.churn_risk_score
        
        # Check inactivity
        if customer.last_interaction_at:
            days_inactive = (datetime.now(timezone.utc) - customer.last_interaction_at).days
            if days_inactive > 30:
                risk_factors.append(f"Inactive for {days_inactive} days")
                risk_score += 0.2
        
        # Check support issues
        if customer.support_ticket_count > 5:
            risk_factors.append("High support ticket volume")
            risk_score += 0.1
        
        # Check satisfaction
        if customer.current_satisfaction in [customer.current_satisfaction.DISSATISFIED, customer.current_satisfaction.VERY_DISSATISFIED]:
            risk_factors.append("Low satisfaction score")
            risk_score += 0.3
        
        if risk_score > 0.6:
            urgency = "high" if risk_score > 0.8 else "medium"
            return JourneyInsight(
                insight_type="churn_risk",
                description=f"Customer shows churn risk indicators: {', '.join(risk_factors)}",
                impact_score=risk_score,
                recommended_actions=[
                    "Reach out with personalized support",
                    "Offer product training or consultation",
                    "Provide incentives for continued engagement",
                    "Escalate to retention specialist"
                ],
                urgency=urgency,
                context={
                    "risk_factors": risk_factors,
                    "risk_score": risk_score,
                    "current_stage": customer.current_journey_stage.value
                }
            )
        
        return None

    async def _analyze_engagement(self, customer: CustomerProfile) -> Optional[JourneyInsight]:
        """Analyze customer engagement patterns"""
        if customer.activity_score < 30:
            return JourneyInsight(
                insight_type="low_engagement",
                description="Customer shows low engagement with the platform",
                impact_score=0.7,
                recommended_actions=[
                    "Send onboarding reminders",
                    "Provide feature highlights and tips",
                    "Offer guided product tour",
                    "Schedule check-in call"
                ],
                urgency="medium",
                context={
                    "activity_score": customer.activity_score,
                    "current_stage": customer.current_journey_stage.value
                }
            )
        
        return None

    async def _analyze_expansion_opportunity(self, customer: CustomerProfile) -> Optional[JourneyInsight]:
        """Analyze expansion opportunities"""
        if (customer.current_journey_stage == JourneyStage.ACTIVE_USE and 
            customer.activity_score > 70 and 
            customer.lifetime_value > 1000):
            
            return JourneyInsight(
                insight_type="expansion_opportunity",
                description="Customer is a good candidate for expansion/upselling",
                impact_score=0.8,
                recommended_actions=[
                    "Present premium features",
                    "Offer usage analytics and insights",
                    "Propose team expansion",
                    "Schedule expansion consultation"
                ],
                urgency="low",
                context={
                    "activity_score": customer.activity_score,
                    "lifetime_value": customer.lifetime_value,
                    "current_stage": customer.current_journey_stage.value
                }
            )
        
        return None

    async def _analyze_satisfaction(self, customer: CustomerProfile) -> Optional[JourneyInsight]:
        """Analyze customer satisfaction patterns"""
        if customer.current_satisfaction == customer.current_satisfaction.VERY_SATISFIED:
            return JourneyInsight(
                insight_type="satisfaction_opportunity",
                description="Highly satisfied customer - opportunity for advocacy",
                impact_score=0.9,
                recommended_actions=[
                    "Request testimonial or case study",
                    "Invite to referral program",
                    "Ask for product review",
                    "Engage for community participation"
                ],
                urgency="low",
                context={
                    "satisfaction_level": customer.current_satisfaction.value if customer.current_satisfaction else "unknown",
                    "current_stage": customer.current_journey_stage.value
                }
            )
        
        return None

    async def get_stage_analytics(self, days: int = 90) -> List[StageAnalytics]:
        """Get analytics for all journey stages"""
        try:
            analytics = []
            
            for stage in JourneyStage:
                stage_analytics = await self._calculate_stage_analytics(stage, days)
                analytics.append(stage_analytics)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting stage analytics: {e}")
            return []

    async def _calculate_stage_analytics(self, stage: JourneyStage, days: int) -> StageAnalytics:
        """Calculate analytics for a specific stage"""
        try:
            from_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get customers currently in this stage
            current_customers = self.db.query(CustomerProfile).filter(
                CustomerProfile.current_journey_stage == stage
            ).count()
            
            # Get customers who were in this stage during the period
            stage_events = self.db.query(CustomerJourneyEvent).filter(
                and_(
                    or_(
                        CustomerJourneyEvent.from_stage == stage,
                        CustomerJourneyEvent.to_stage == stage
                    ),
                    CustomerJourneyEvent.occurred_at >= from_date
                )
            ).all()
            
            # Calculate average duration
            durations = []
            for event in stage_events:
                if event.to_stage != stage:  # Exiting the stage
                    # Find when they entered the stage
                    enter_event = self.db.query(CustomerJourneyEvent).filter(
                        and_(
                            CustomerJourneyEvent.customer_id == event.customer_id,
                            CustomerJourneyEvent.to_stage == stage,
                            CustomerJourneyEvent.occurred_at < event.occurred_at
                        )
                    ).order_by(desc(CustomerJourneyEvent.occurred_at)).first()
                    
                    if enter_event:
                        duration = (event.occurred_at - enter_event.occurred_at).days
                        durations.append(duration)
            
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Calculate conversion rate (simplified)
            total_entries = len([e for e in stage_events if e.to_stage == stage])
            successful_exits = len([e for e in stage_events if e.from_stage == stage and e.to_stage != JourneyStage.CHURNED])
            conversion_rate = (successful_exits / total_entries) if total_entries > 0 else 0
            
            return StageAnalytics(
                stage=stage,
                total_customers=current_customers,
                avg_duration_days=avg_duration,
                conversion_rate=conversion_rate,
                common_exit_points=["churned", "next_stage"],  # Simplified
                success_factors=["engagement", "satisfaction"],  # Simplified
                bottlenecks=["low_engagement", "support_issues"]  # Simplified
            )
            
        except Exception as e:
            logger.error(f"Error calculating stage analytics for {stage.value}: {e}")
            return StageAnalytics(
                stage=stage,
                total_customers=0,
                avg_duration_days=0,
                conversion_rate=0,
                common_exit_points=[],
                success_factors=[],
                bottlenecks=[]
            )