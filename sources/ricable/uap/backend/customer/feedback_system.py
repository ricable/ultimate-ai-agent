# File: backend/customer/feedback_system.py
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import json
import statistics

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func

from ..models.customer import (
    CustomerProfile, CustomerFeedback, CustomerInteraction, 
    SentimentType, SatisfactionLevel, InteractionType
)
from ..models.user import User
from ..database.session import get_db
from .sentiment_routing import SentimentAnalyzer

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback collection"""
    POST_INTERACTION = "post_interaction"
    PERIODIC_SURVEY = "periodic_survey"
    MILESTONE_BASED = "milestone_based"
    PROACTIVE_OUTREACH = "proactive_outreach"
    UNSOLICITED = "unsolicited"
    EXIT_SURVEY = "exit_survey"

class SurveyTemplate(Enum):
    """Pre-defined survey templates"""
    CSAT = "customer_satisfaction"
    NPS = "net_promoter_score"
    CES = "customer_effort_score"
    PMF = "product_market_fit"
    FEATURE_FEEDBACK = "feature_feedback"
    SUPPORT_FEEDBACK = "support_feedback"
    ONBOARDING_FEEDBACK = "onboarding_feedback"

class FeedbackTrigger(Enum):
    """Feedback collection triggers"""
    INTERACTION_COMPLETE = "interaction_complete"
    TIME_BASED = "time_based"
    MILESTONE_REACHED = "milestone_reached"
    STAGE_TRANSITION = "stage_transition"
    MANUAL_REQUEST = "manual_request"
    CHURN_RISK = "churn_risk"

@dataclass
class FeedbackRequest:
    """Feedback collection request"""
    customer_id: str
    feedback_type: FeedbackType
    survey_template: SurveyTemplate
    trigger: FeedbackTrigger
    context: Dict[str, Any]
    priority: str = "medium"
    expires_at: Optional[datetime] = None

@dataclass
class FeedbackResponse:
    """Customer feedback response"""
    request_id: str
    customer_id: str
    responses: Dict[str, Any]
    ratings: Dict[str, int]
    text_feedback: str
    sentiment: SentimentType
    satisfaction_level: SatisfactionLevel
    completion_rate: float
    submitted_at: datetime

@dataclass
class FeedbackAnalysis:
    """Analysis of feedback data"""
    overall_sentiment: SentimentType
    satisfaction_trend: List[float]
    key_themes: List[str]
    improvement_areas: List[str]
    satisfaction_drivers: List[str]
    risk_indicators: List[str]
    recommendations: List[str]

@dataclass
class SatisfactionMetrics:
    """Customer satisfaction metrics"""
    csat_score: float
    nps_score: int
    ces_score: float
    satisfaction_trend: List[float]
    response_rate: float
    completion_rate: float
    time_to_respond: float
    feedback_volume: int

class FeedbackCollector:
    """Manages feedback collection and processing"""
    
    def __init__(self, db: Session):
        self.db = db
        self.sentiment_analyzer = SentimentAnalyzer()
        self.survey_templates = self._initialize_survey_templates()
        self.feedback_rules = self._initialize_feedback_rules()

    def _initialize_survey_templates(self) -> Dict[SurveyTemplate, Dict]:
        """Initialize survey templates"""
        return {
            SurveyTemplate.CSAT: {
                'name': 'Customer Satisfaction Survey',
                'questions': [
                    {
                        'id': 'overall_satisfaction',
                        'type': 'rating',
                        'question': 'How satisfied are you with our service?',
                        'scale': '1-5',
                        'required': True
                    },
                    {
                        'id': 'recommendation_likelihood',
                        'type': 'rating',
                        'question': 'How likely are you to recommend us?',
                        'scale': '1-10',
                        'required': True
                    },
                    {
                        'id': 'improvement_suggestions',
                        'type': 'text',
                        'question': 'What could we improve?',
                        'required': False
                    }
                ],
                'estimated_time': 2
            },
            SurveyTemplate.NPS: {
                'name': 'Net Promoter Score Survey',
                'questions': [
                    {
                        'id': 'nps_rating',
                        'type': 'rating',
                        'question': 'How likely are you to recommend our product to a friend or colleague?',
                        'scale': '0-10',
                        'required': True
                    },
                    {
                        'id': 'nps_reason',
                        'type': 'text',
                        'question': 'What is the main reason for your score?',
                        'required': False
                    }
                ],
                'estimated_time': 1
            },
            SurveyTemplate.CES: {
                'name': 'Customer Effort Score Survey',
                'questions': [
                    {
                        'id': 'effort_rating',
                        'type': 'rating',
                        'question': 'How easy was it to get your issue resolved?',
                        'scale': '1-7',
                        'required': True
                    },
                    {
                        'id': 'effort_details',
                        'type': 'text',
                        'question': 'What made it easy or difficult?',
                        'required': False
                    }
                ],
                'estimated_time': 1
            },
            SurveyTemplate.FEATURE_FEEDBACK: {
                'name': 'Feature Feedback Survey',
                'questions': [
                    {
                        'id': 'feature_satisfaction',
                        'type': 'rating',
                        'question': 'How satisfied are you with this feature?',
                        'scale': '1-5',
                        'required': True
                    },
                    {
                        'id': 'feature_usefulness',
                        'type': 'rating',
                        'question': 'How useful is this feature for your needs?',
                        'scale': '1-5',
                        'required': True
                    },
                    {
                        'id': 'feature_improvements',
                        'type': 'text',
                        'question': 'How could we improve this feature?',
                        'required': False
                    }
                ],
                'estimated_time': 3
            }
        }

    def _initialize_feedback_rules(self) -> Dict[str, Dict]:
        """Initialize feedback collection rules"""
        return {
            'post_interaction_delay': timedelta(minutes=5),
            'survey_frequency_limit': timedelta(days=7),  # Don't survey same customer too frequently
            'max_surveys_per_month': 3,
            'response_timeout': timedelta(days=7),
            'reminder_schedule': [
                timedelta(days=1),
                timedelta(days=3),
                timedelta(days=5)
            ],
            'priority_rules': {
                'churn_risk': 'high',
                'negative_sentiment': 'high',
                'vip_customer': 'high',
                'milestone_reached': 'medium',
                'periodic_survey': 'low'
            }
        }

    async def request_feedback(self, request: FeedbackRequest) -> Optional[str]:
        """Request feedback from customer"""
        try:
            # Check if customer can be surveyed
            if not await self._can_survey_customer(request.customer_id):
                logger.info(f"Customer {request.customer_id} cannot be surveyed at this time")
                return None

            # Create feedback request record
            feedback_request = await self._create_feedback_request(request)
            
            # Send feedback request to customer
            await self._send_feedback_request(feedback_request)
            
            # Schedule follow-up reminders
            await self._schedule_reminders(feedback_request['id'])
            
            return feedback_request['id']
            
        except Exception as e:
            logger.error(f"Error requesting feedback: {e}")
            return None

    async def _can_survey_customer(self, customer_id: str) -> bool:
        """Check if customer can be surveyed based on rules"""
        try:
            # Check recent survey frequency
            recent_surveys = self.db.query(CustomerFeedback).filter(
                and_(
                    CustomerFeedback.customer_id == customer_id,
                    CustomerFeedback.submitted_at >= datetime.now(timezone.utc) - self.feedback_rules['survey_frequency_limit']
                )
            ).count()
            
            if recent_surveys > 0:
                return False
            
            # Check monthly survey limit
            this_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0)
            monthly_surveys = self.db.query(CustomerFeedback).filter(
                and_(
                    CustomerFeedback.customer_id == customer_id,
                    CustomerFeedback.submitted_at >= this_month
                )
            ).count()
            
            if monthly_surveys >= self.feedback_rules['max_surveys_per_month']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking survey eligibility: {e}")
            return False

    async def _create_feedback_request(self, request: FeedbackRequest) -> Dict[str, Any]:
        """Create feedback request record"""
        # This would typically create a record in a feedback_requests table
        # For now, we'll return a mock structure
        return {
            'id': f"feedback_{request.customer_id}_{datetime.now().timestamp()}",
            'customer_id': request.customer_id,
            'feedback_type': request.feedback_type.value,
            'survey_template': request.survey_template.value,
            'trigger': request.trigger.value,
            'context': request.context,
            'priority': request.priority,
            'created_at': datetime.now(timezone.utc),
            'expires_at': request.expires_at or datetime.now(timezone.utc) + self.feedback_rules['response_timeout']
        }

    async def _send_feedback_request(self, feedback_request: Dict[str, Any]):
        """Send feedback request to customer"""
        # This would integrate with email/SMS/in-app notification systems
        # For now, we'll log the request
        logger.info(f"Sending feedback request {feedback_request['id']} to customer {feedback_request['customer_id']}")

    async def _schedule_reminders(self, request_id: str):
        """Schedule follow-up reminders for feedback request"""
        # This would typically integrate with a task queue system
        # For now, we'll log the scheduling
        logger.info(f"Scheduling reminders for feedback request {request_id}")

    async def process_feedback_response(self, response: FeedbackResponse) -> Optional[CustomerFeedback]:
        """Process customer feedback response"""
        try:
            # Analyze sentiment of text feedback
            sentiment_result = None
            if response.text_feedback:
                sentiment_result = await self.sentiment_analyzer.analyze_sentiment(response.text_feedback)

            # Create feedback record
            feedback = CustomerFeedback(
                customer_id=response.customer_id,
                feedback_type=FeedbackType.POST_INTERACTION.value,  # Default type
                rating=response.ratings.get('overall_satisfaction'),
                nps_score=response.ratings.get('nps_rating'),
                ces_score=response.ratings.get('effort_rating'),
                csat_score=response.ratings.get('overall_satisfaction'),
                title=f"Feedback from {response.submitted_at.strftime('%Y-%m-%d')}",
                content=response.text_feedback,
                categories=self._extract_feedback_categories(response.responses),
                sentiment=sentiment_result.sentiment if sentiment_result else SentimentType.NEUTRAL,
                sentiment_score=sentiment_result.scores.get('positive', 0) - sentiment_result.scores.get('negative', 0) if sentiment_result else 0,
                status='received',
                response_required=self._requires_response(response),
                submitted_at=response.submitted_at
            )

            self.db.add(feedback)
            self.db.commit()

            # Update customer satisfaction metrics
            await self._update_customer_satisfaction(response.customer_id, feedback)

            # Trigger follow-up actions if needed
            await self._trigger_feedback_actions(feedback)

            return feedback

        except Exception as e:
            logger.error(f"Error processing feedback response: {e}")
            self.db.rollback()
            return None

    def _extract_feedback_categories(self, responses: Dict[str, Any]) -> List[str]:
        """Extract categories from feedback responses"""
        categories = []
        
        # Analyze responses to determine categories
        for question_id, answer in responses.items():
            if isinstance(answer, str):
                answer_lower = answer.lower()
                
                # Category mapping based on keywords
                if any(word in answer_lower for word in ['feature', 'functionality', 'tool']):
                    categories.append('feature')
                elif any(word in answer_lower for word in ['support', 'help', 'assistance']):
                    categories.append('support')
                elif any(word in answer_lower for word in ['ui', 'interface', 'design', 'usability']):
                    categories.append('usability')
                elif any(word in answer_lower for word in ['performance', 'speed', 'slow']):
                    categories.append('performance')
                elif any(word in answer_lower for word in ['billing', 'price', 'cost']):
                    categories.append('billing')
        
        return list(set(categories)) if categories else ['general']

    def _requires_response(self, response: FeedbackResponse) -> bool:
        """Determine if feedback requires a response"""
        # Require response for negative sentiment or low ratings
        if response.sentiment in [SentimentType.NEGATIVE, SentimentType.FRUSTRATED]:
            return True
        
        # Require response for low CSAT scores
        if response.ratings.get('overall_satisfaction', 5) <= 2:
            return True
        
        # Require response for low NPS scores (detractors)
        if response.ratings.get('nps_rating', 10) <= 6:
            return True
        
        return False

    async def _update_customer_satisfaction(self, customer_id: str, feedback: CustomerFeedback):
        """Update customer satisfaction metrics"""
        try:
            customer = self.db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if customer:
                # Update satisfaction level based on feedback
                if feedback.csat_score:
                    if feedback.csat_score >= 4:
                        customer.current_satisfaction = SatisfactionLevel.SATISFIED
                    elif feedback.csat_score >= 3:
                        customer.current_satisfaction = SatisfactionLevel.NEUTRAL
                    else:
                        customer.current_satisfaction = SatisfactionLevel.DISSATISFIED
                
                # Update NPS score
                if feedback.nps_score is not None:
                    customer.net_promoter_score = feedback.nps_score
                
                # Update CES score
                if feedback.ces_score is not None:
                    customer.customer_effort_score = feedback.ces_score
                
                customer.satisfaction_updated_at = datetime.now(timezone.utc)
                self.db.commit()
                
        except Exception as e:
            logger.error(f"Error updating customer satisfaction: {e}")
            self.db.rollback()

    async def _trigger_feedback_actions(self, feedback: CustomerFeedback):
        """Trigger actions based on feedback"""
        try:
            # Negative feedback - escalate to support
            if feedback.sentiment in [SentimentType.NEGATIVE, SentimentType.FRUSTRATED]:
                await self._escalate_to_support(feedback)
            
            # Low ratings - trigger retention workflow
            if feedback.csat_score and feedback.csat_score <= 2:
                await self._trigger_retention_workflow(feedback)
            
            # High ratings - trigger advocacy workflow
            if feedback.csat_score and feedback.csat_score >= 4:
                await self._trigger_advocacy_workflow(feedback)
            
        except Exception as e:
            logger.error(f"Error triggering feedback actions: {e}")

    async def _escalate_to_support(self, feedback: CustomerFeedback):
        """Escalate negative feedback to support team"""
        logger.info(f"Escalating feedback {feedback.id} to support team")
        # This would typically integrate with support ticket system

    async def _trigger_retention_workflow(self, feedback: CustomerFeedback):
        """Trigger retention workflow for dissatisfied customers"""
        logger.info(f"Triggering retention workflow for feedback {feedback.id}")
        # This would typically trigger automated retention campaigns

    async def _trigger_advocacy_workflow(self, feedback: CustomerFeedback):
        """Trigger advocacy workflow for satisfied customers"""
        logger.info(f"Triggering advocacy workflow for feedback {feedback.id}")
        # This would typically invite customers to provide testimonials, reviews, etc.

class SatisfactionTracker:
    """Tracks and analyzes customer satisfaction metrics"""
    
    def __init__(self, db: Session):
        self.db = db

    async def get_satisfaction_metrics(self, customer_id: str, days: int = 90) -> SatisfactionMetrics:
        """Get satisfaction metrics for a customer"""
        try:
            from_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get feedback from the period
            feedback_entries = self.db.query(CustomerFeedback).filter(
                and_(
                    CustomerFeedback.customer_id == customer_id,
                    CustomerFeedback.submitted_at >= from_date
                )
            ).order_by(CustomerFeedback.submitted_at).all()
            
            if not feedback_entries:
                return SatisfactionMetrics(
                    csat_score=0.0,
                    nps_score=0,
                    ces_score=0.0,
                    satisfaction_trend=[],
                    response_rate=0.0,
                    completion_rate=0.0,
                    time_to_respond=0.0,
                    feedback_volume=0
                )
            
            # Calculate CSAT score
            csat_scores = [f.csat_score for f in feedback_entries if f.csat_score]
            csat_score = statistics.mean(csat_scores) if csat_scores else 0.0
            
            # Calculate NPS score (most recent)
            nps_scores = [f.nps_score for f in feedback_entries if f.nps_score is not None]
            nps_score = nps_scores[-1] if nps_scores else 0
            
            # Calculate CES score
            ces_scores = [f.ces_score for f in feedback_entries if f.ces_score]
            ces_score = statistics.mean(ces_scores) if ces_scores else 0.0
            
            # Calculate satisfaction trend
            satisfaction_trend = []
            for entry in feedback_entries:
                if entry.csat_score:
                    satisfaction_trend.append(entry.csat_score)
            
            return SatisfactionMetrics(
                csat_score=csat_score,
                nps_score=nps_score,
                ces_score=ces_score,
                satisfaction_trend=satisfaction_trend,
                response_rate=1.0,  # Simplified - assume all responded
                completion_rate=1.0,  # Simplified - assume all completed
                time_to_respond=1.0,  # Simplified - assume 1 day average
                feedback_volume=len(feedback_entries)
            )
            
        except Exception as e:
            logger.error(f"Error getting satisfaction metrics: {e}")
            return SatisfactionMetrics(
                csat_score=0.0,
                nps_score=0,
                ces_score=0.0,
                satisfaction_trend=[],
                response_rate=0.0,
                completion_rate=0.0,
                time_to_respond=0.0,
                feedback_volume=0
            )

    async def analyze_feedback_trends(self, customer_id: str, days: int = 180) -> FeedbackAnalysis:
        """Analyze feedback trends and provide insights"""
        try:
            from_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            feedback_entries = self.db.query(CustomerFeedback).filter(
                and_(
                    CustomerFeedback.customer_id == customer_id,
                    CustomerFeedback.submitted_at >= from_date
                )
            ).order_by(CustomerFeedback.submitted_at).all()
            
            if not feedback_entries:
                return FeedbackAnalysis(
                    overall_sentiment=SentimentType.NEUTRAL,
                    satisfaction_trend=[],
                    key_themes=[],
                    improvement_areas=[],
                    satisfaction_drivers=[],
                    risk_indicators=[],
                    recommendations=[]
                )
            
            # Analyze overall sentiment
            sentiments = [f.sentiment for f in feedback_entries if f.sentiment]
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            overall_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else SentimentType.NEUTRAL
            
            # Calculate satisfaction trend
            satisfaction_trend = []
            for entry in feedback_entries:
                if entry.csat_score:
                    satisfaction_trend.append(entry.csat_score)
            
            # Extract key themes from feedback content
            key_themes = await self._extract_themes(feedback_entries)
            
            # Identify improvement areas
            improvement_areas = await self._identify_improvement_areas(feedback_entries)
            
            # Identify satisfaction drivers
            satisfaction_drivers = await self._identify_satisfaction_drivers(feedback_entries)
            
            # Identify risk indicators
            risk_indicators = await self._identify_risk_indicators(feedback_entries)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(feedback_entries, overall_sentiment)
            
            return FeedbackAnalysis(
                overall_sentiment=overall_sentiment,
                satisfaction_trend=satisfaction_trend,
                key_themes=key_themes,
                improvement_areas=improvement_areas,
                satisfaction_drivers=satisfaction_drivers,
                risk_indicators=risk_indicators,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing feedback trends: {e}")
            return FeedbackAnalysis(
                overall_sentiment=SentimentType.NEUTRAL,
                satisfaction_trend=[],
                key_themes=[],
                improvement_areas=[],
                satisfaction_drivers=[],
                risk_indicators=[],
                recommendations=[]
            )

    async def _extract_themes(self, feedback_entries: List[CustomerFeedback]) -> List[str]:
        """Extract key themes from feedback content"""
        themes = {}
        
        for feedback in feedback_entries:
            if feedback.content:
                # Simple keyword extraction
                content_lower = feedback.content.lower()
                
                # Theme keywords
                theme_keywords = {
                    'usability': ['easy', 'difficult', 'confusing', 'intuitive', 'user-friendly'],
                    'performance': ['slow', 'fast', 'loading', 'responsive', 'speed'],
                    'features': ['feature', 'functionality', 'tool', 'option', 'capability'],
                    'support': ['support', 'help', 'assistance', 'service', 'response'],
                    'pricing': ['price', 'cost', 'expensive', 'cheap', 'value'],
                    'reliability': ['reliable', 'stable', 'crash', 'bug', 'error']
                }
                
                for theme, keywords in theme_keywords.items():
                    if any(keyword in content_lower for keyword in keywords):
                        themes[theme] = themes.get(theme, 0) + 1
        
        # Return top themes
        return sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]

    async def _identify_improvement_areas(self, feedback_entries: List[CustomerFeedback]) -> List[str]:
        """Identify areas for improvement based on feedback"""
        improvement_areas = []
        
        # Analyze negative feedback
        negative_feedback = [f for f in feedback_entries if f.sentiment in [SentimentType.NEGATIVE, SentimentType.FRUSTRATED]]
        
        for feedback in negative_feedback:
            if feedback.categories:
                improvement_areas.extend(feedback.categories)
        
        # Count occurrences
        area_counts = {}
        for area in improvement_areas:
            area_counts[area] = area_counts.get(area, 0) + 1
        
        # Return top improvement areas
        return [area for area, count in sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:5]]

    async def _identify_satisfaction_drivers(self, feedback_entries: List[CustomerFeedback]) -> List[str]:
        """Identify what drives customer satisfaction"""
        drivers = []
        
        # Analyze positive feedback
        positive_feedback = [f for f in feedback_entries if f.sentiment == SentimentType.POSITIVE]
        
        for feedback in positive_feedback:
            if feedback.categories:
                drivers.extend(feedback.categories)
        
        # Count occurrences
        driver_counts = {}
        for driver in drivers:
            driver_counts[driver] = driver_counts.get(driver, 0) + 1
        
        # Return top satisfaction drivers
        return [driver for driver, count in sorted(driver_counts.items(), key=lambda x: x[1], reverse=True)[:5]]

    async def _identify_risk_indicators(self, feedback_entries: List[CustomerFeedback]) -> List[str]:
        """Identify risk indicators from feedback"""
        indicators = []
        
        # Check for declining satisfaction
        recent_scores = [f.csat_score for f in feedback_entries[-3:] if f.csat_score]
        if len(recent_scores) >= 2:
            if recent_scores[-1] < recent_scores[0]:
                indicators.append("Declining satisfaction scores")
        
        # Check for repeated negative feedback
        negative_count = len([f for f in feedback_entries if f.sentiment in [SentimentType.NEGATIVE, SentimentType.FRUSTRATED]])
        if negative_count >= 2:
            indicators.append("Multiple negative feedback instances")
        
        # Check for low NPS scores
        recent_nps = [f.nps_score for f in feedback_entries if f.nps_score is not None]
        if recent_nps and recent_nps[-1] <= 6:
            indicators.append("Low Net Promoter Score (detractor)")
        
        return indicators

    async def _generate_recommendations(self, feedback_entries: List[CustomerFeedback], overall_sentiment: SentimentType) -> List[str]:
        """Generate recommendations based on feedback analysis"""
        recommendations = []
        
        if overall_sentiment in [SentimentType.NEGATIVE, SentimentType.FRUSTRATED]:
            recommendations.extend([
                "Schedule immediate customer success check-in",
                "Provide personalized support and training",
                "Investigate specific pain points mentioned in feedback"
            ])
        elif overall_sentiment == SentimentType.POSITIVE:
            recommendations.extend([
                "Invite customer to provide testimonial or case study",
                "Explore expansion opportunities",
                "Request referrals or reviews"
            ])
        else:
            recommendations.extend([
                "Conduct regular satisfaction check-ins",
                "Provide additional product education",
                "Monitor for changes in satisfaction levels"
            ])
        
        return recommendations

    async def get_satisfaction_benchmarks(self, segment: str = None) -> Dict[str, float]:
        """Get satisfaction benchmarks for comparison"""
        try:
            # Get recent feedback for benchmarking
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            
            query = self.db.query(CustomerFeedback).filter(
                CustomerFeedback.submitted_at >= thirty_days_ago
            )
            
            if segment:
                # Join with customer profile to filter by segment
                query = query.join(CustomerProfile).filter(
                    CustomerProfile.customer_segment == segment
                )
            
            feedback_entries = query.all()
            
            if not feedback_entries:
                return {
                    'avg_csat': 0.0,
                    'avg_nps': 0.0,
                    'avg_ces': 0.0,
                    'satisfaction_rate': 0.0
                }
            
            # Calculate benchmarks
            csat_scores = [f.csat_score for f in feedback_entries if f.csat_score]
            nps_scores = [f.nps_score for f in feedback_entries if f.nps_score is not None]
            ces_scores = [f.ces_score for f in feedback_entries if f.ces_score]
            
            satisfied_count = len([f for f in feedback_entries if f.csat_score and f.csat_score >= 4])
            satisfaction_rate = satisfied_count / len(feedback_entries) if feedback_entries else 0
            
            return {
                'avg_csat': statistics.mean(csat_scores) if csat_scores else 0.0,
                'avg_nps': statistics.mean(nps_scores) if nps_scores else 0.0,
                'avg_ces': statistics.mean(ces_scores) if ces_scores else 0.0,
                'satisfaction_rate': satisfaction_rate
            }
            
        except Exception as e:
            logger.error(f"Error getting satisfaction benchmarks: {e}")
            return {
                'avg_csat': 0.0,
                'avg_nps': 0.0,
                'avg_ces': 0.0,
                'satisfaction_rate': 0.0
            }