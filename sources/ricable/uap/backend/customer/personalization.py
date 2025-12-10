# File: backend/customer/personalization.py
import asyncio
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
import logging
from enum import Enum
from collections import defaultdict
import json

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func

from ..models.customer import (
    CustomerProfile, CustomerRecommendation, CustomerInteraction, CustomerJourneyEvent,
    JourneyStage, SentimentType, InteractionType
)
from ..models.user import User
from ..database.session import get_db

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """Types of recommendations"""
    FEATURE = "feature"
    CONTENT = "content"
    ACTION = "action"
    PRODUCT = "product"
    TRAINING = "training"
    SUPPORT = "support"
    INTEGRATION = "integration"
    WORKFLOW = "workflow"

class PersonalizationStrategy(Enum):
    """Personalization strategies"""
    BEHAVIORAL = "behavioral"
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"

@dataclass
class PersonalizationProfile:
    """Customer personalization profile"""
    customer_id: str
    preferences: Dict[str, Any]
    behavior_patterns: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    journey_context: Dict[str, Any]
    segmentation_data: Dict[str, Any]
    recommendation_history: List[Dict[str, Any]]
    engagement_metrics: Dict[str, float]

@dataclass
class RecommendationRequest:
    """Request for personalized recommendations"""
    customer_id: str
    context: Dict[str, Any]
    recommendation_types: List[RecommendationType]
    max_recommendations: int = 5
    exclude_previous: bool = True
    strategy: PersonalizationStrategy = PersonalizationStrategy.HYBRID

@dataclass
class PersonalizedRecommendation:
    """Personalized recommendation with scoring"""
    recommendation_id: str
    type: RecommendationType
    title: str
    description: str
    confidence_score: float
    relevance_score: float
    priority: str
    content_data: Dict[str, Any]
    action_url: Optional[str]
    image_url: Optional[str]
    personalization_factors: List[str]
    reasoning: str

class PersonalizationEngine:
    """Advanced personalization engine for customer experiences"""
    
    def __init__(self, db: Session):
        self.db = db
        self.feature_catalog = self._initialize_feature_catalog()
        self.content_catalog = self._initialize_content_catalog()
        self.behavioral_signals = self._initialize_behavioral_signals()

    def _initialize_feature_catalog(self) -> Dict[str, Dict]:
        """Initialize feature catalog with metadata"""
        return {
            'advanced_analytics': {
                'category': 'analytics',
                'complexity': 'advanced',
                'prerequisites': ['basic_analytics'],
                'value_props': ['data_insights', 'performance_optimization'],
                'target_stages': [JourneyStage.ACTIVE_USE, JourneyStage.EXPANSION],
                'user_segments': ['enterprise', 'power_user']
            },
            'team_collaboration': {
                'category': 'collaboration',
                'complexity': 'intermediate',
                'prerequisites': [],
                'value_props': ['team_productivity', 'communication'],
                'target_stages': [JourneyStage.ACTIVE_USE, JourneyStage.EXPANSION],
                'user_segments': ['team', 'enterprise']
            },
            'api_integration': {
                'category': 'integration',
                'complexity': 'advanced',
                'prerequisites': ['basic_setup'],
                'value_props': ['automation', 'workflow_optimization'],
                'target_stages': [JourneyStage.ACTIVE_USE, JourneyStage.EXPANSION],
                'user_segments': ['developer', 'enterprise']
            },
            'custom_dashboards': {
                'category': 'visualization',
                'complexity': 'intermediate',
                'prerequisites': ['basic_analytics'],
                'value_props': ['customization', 'insights'],
                'target_stages': [JourneyStage.ACTIVE_USE],
                'user_segments': ['analyst', 'manager']
            },
            'automated_workflows': {
                'category': 'automation',
                'complexity': 'advanced',
                'prerequisites': ['basic_setup', 'api_integration'],
                'value_props': ['efficiency', 'time_saving'],
                'target_stages': [JourneyStage.ACTIVE_USE, JourneyStage.EXPANSION],
                'user_segments': ['power_user', 'enterprise']
            }
        }

    def _initialize_content_catalog(self) -> Dict[str, Dict]:
        """Initialize content catalog for recommendations"""
        return {
            'getting_started_guide': {
                'category': 'onboarding',
                'content_type': 'guide',
                'difficulty': 'beginner',
                'estimated_time': '15 minutes',
                'target_stages': [JourneyStage.ONBOARDING],
                'topics': ['setup', 'first_steps', 'basic_features']
            },
            'advanced_features_webinar': {
                'category': 'education',
                'content_type': 'webinar',
                'difficulty': 'advanced',
                'estimated_time': '60 minutes',
                'target_stages': [JourneyStage.ACTIVE_USE, JourneyStage.EXPANSION],
                'topics': ['advanced_features', 'best_practices', 'optimization']
            },
            'integration_cookbook': {
                'category': 'technical',
                'content_type': 'documentation',
                'difficulty': 'intermediate',
                'estimated_time': '30 minutes',
                'target_stages': [JourneyStage.ACTIVE_USE],
                'topics': ['api', 'integration', 'automation']
            },
            'success_stories': {
                'category': 'inspiration',
                'content_type': 'case_study',
                'difficulty': 'beginner',
                'estimated_time': '10 minutes',
                'target_stages': [JourneyStage.CONSIDERATION, JourneyStage.TRIAL],
                'topics': ['use_cases', 'results', 'best_practices']
            }
        }

    def _initialize_behavioral_signals(self) -> Dict[str, Dict]:
        """Initialize behavioral signal definitions"""
        return {
            'feature_exploration': {
                'weight': 0.8,
                'decay_days': 30,
                'indicators': ['feature_usage', 'time_spent', 'return_visits']
            },
            'content_engagement': {
                'weight': 0.6,
                'decay_days': 60,
                'indicators': ['content_views', 'completion_rate', 'sharing']
            },
            'support_interaction': {
                'weight': 0.9,
                'decay_days': 90,
                'indicators': ['ticket_creation', 'resolution_satisfaction', 'self_service_usage']
            },
            'social_signals': {
                'weight': 0.7,
                'decay_days': 45,
                'indicators': ['sharing', 'referrals', 'community_participation']
            }
        }

    async def build_personalization_profile(self, customer_id: str) -> PersonalizationProfile:
        """Build comprehensive personalization profile for customer"""
        try:
            customer = self.db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if not customer:
                raise ValueError(f"Customer {customer_id} not found")

            # Get customer preferences
            preferences = await self._extract_preferences(customer)
            
            # Analyze behavior patterns
            behavior_patterns = await self._analyze_behavior_patterns(customer_id)
            
            # Get interaction history
            interaction_history = await self._get_interaction_history(customer_id)
            
            # Build journey context
            journey_context = await self._build_journey_context(customer)
            
            # Get segmentation data
            segmentation_data = await self._get_segmentation_data(customer)
            
            # Get recommendation history
            recommendation_history = await self._get_recommendation_history(customer_id)
            
            # Calculate engagement metrics
            engagement_metrics = await self._calculate_engagement_metrics(customer_id)
            
            return PersonalizationProfile(
                customer_id=customer_id,
                preferences=preferences,
                behavior_patterns=behavior_patterns,
                interaction_history=interaction_history,
                journey_context=journey_context,
                segmentation_data=segmentation_data,
                recommendation_history=recommendation_history,
                engagement_metrics=engagement_metrics
            )
            
        except Exception as e:
            logger.error(f"Error building personalization profile: {e}")
            return PersonalizationProfile(
                customer_id=customer_id,
                preferences={},
                behavior_patterns={},
                interaction_history=[],
                journey_context={},
                segmentation_data={},
                recommendation_history=[],
                engagement_metrics={}
            )

    async def _extract_preferences(self, customer: CustomerProfile) -> Dict[str, Any]:
        """Extract customer preferences from profile and behavior"""
        preferences = customer.communication_preferences.copy()
        feature_prefs = customer.feature_preferences.copy()
        
        # Infer preferences from interaction patterns
        recent_interactions = self.db.query(CustomerInteraction).filter(
            and_(
                CustomerInteraction.customer_id == customer.id,
                CustomerInteraction.started_at >= datetime.now(timezone.utc) - timedelta(days=30)
            )
        ).all()
        
        # Analyze interaction types
        interaction_types = defaultdict(int)
        for interaction in recent_interactions:
            interaction_types[interaction.interaction_type.value] += 1
        
        # Infer communication preferences
        if interaction_types.get('chat', 0) > interaction_types.get('email', 0):
            preferences['preferred_communication'] = 'chat'
        else:
            preferences['preferred_communication'] = 'email'
        
        # Infer feature interests based on support queries
        feature_interests = set()
        for interaction in recent_interactions:
            if interaction.content:
                content_lower = interaction.content.lower()
                for feature in self.feature_catalog:
                    if feature.replace('_', ' ') in content_lower:
                        feature_interests.add(feature)
        
        preferences['feature_interests'] = list(feature_interests)
        
        return {
            'communication': preferences,
            'features': feature_prefs,
            'inferred_interests': list(feature_interests)
        }

    async def _analyze_behavior_patterns(self, customer_id: str) -> Dict[str, Any]:
        """Analyze customer behavior patterns"""
        # Get interactions from last 90 days
        ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
        
        interactions = self.db.query(CustomerInteraction).filter(
            and_(
                CustomerInteraction.customer_id == customer_id,
                CustomerInteraction.started_at >= ninety_days_ago
            )
        ).order_by(CustomerInteraction.started_at).all()
        
        patterns = {
            'activity_times': defaultdict(int),
            'interaction_frequency': {},
            'support_patterns': {},
            'content_preferences': {},
            'engagement_trends': []
        }
        
        # Analyze activity times
        for interaction in interactions:
            hour = interaction.started_at.hour
            patterns['activity_times'][hour] += 1
        
        # Calculate interaction frequency
        if interactions:
            total_days = (interactions[-1].started_at - interactions[0].started_at).days or 1
            patterns['interaction_frequency'] = {
                'avg_per_day': len(interactions) / total_days,
                'total_interactions': len(interactions),
                'active_days': len(set(i.started_at.date() for i in interactions))
            }
        
        # Analyze support patterns
        support_interactions = [i for i in interactions if i.interaction_type == InteractionType.TICKET]
        if support_interactions:
            patterns['support_patterns'] = {
                'avg_resolution_time': sum(
                    i.resolution_time_minutes for i in support_interactions 
                    if i.resolution_time_minutes
                ) / len(support_interactions),
                'satisfaction_avg': sum(
                    i.satisfaction_rating for i in support_interactions 
                    if i.satisfaction_rating
                ) / len([i for i in support_interactions if i.satisfaction_rating])
            }
        
        return patterns

    async def _get_interaction_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get formatted interaction history"""
        interactions = self.db.query(CustomerInteraction).filter(
            CustomerInteraction.customer_id == customer_id
        ).order_by(desc(CustomerInteraction.started_at)).limit(50).all()
        
        return [
            {
                'id': str(interaction.id),
                'type': interaction.interaction_type.value,
                'channel': interaction.channel,
                'sentiment': interaction.sentiment.value if interaction.sentiment else None,
                'satisfaction': interaction.satisfaction_rating,
                'timestamp': interaction.started_at.isoformat(),
                'tags': interaction.tags
            }
            for interaction in interactions
        ]

    async def _build_journey_context(self, customer: CustomerProfile) -> Dict[str, Any]:
        """Build journey context for personalization"""
        return {
            'current_stage': customer.current_journey_stage.value,
            'days_in_stage': (datetime.now(timezone.utc) - customer.journey_stage_updated_at).days if customer.journey_stage_updated_at else 0,
            'total_journey_days': (datetime.now(timezone.utc) - customer.created_at).days,
            'churn_risk_score': customer.churn_risk_score,
            'activity_score': customer.activity_score,
            'satisfaction_level': customer.current_satisfaction.value if customer.current_satisfaction else None
        }

    async def _get_segmentation_data(self, customer: CustomerProfile) -> Dict[str, Any]:
        """Get customer segmentation data"""
        return {
            'segment': customer.customer_segment,
            'tier': customer.customer_tier,
            'lifetime_value': customer.lifetime_value,
            'acquisition_channel': customer.acquisition_channel,
            'personalization_tags': customer.personalization_tags
        }

    async def _get_recommendation_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get customer's recommendation history"""
        recommendations = self.db.query(CustomerRecommendation).filter(
            CustomerRecommendation.customer_id == customer_id
        ).order_by(desc(CustomerRecommendation.created_at)).limit(20).all()
        
        return [
            {
                'id': str(rec.id),
                'type': rec.recommendation_type,
                'title': rec.title,
                'confidence': rec.confidence_score,
                'views': rec.views_count,
                'clicks': rec.clicks_count,
                'conversions': rec.conversions_count,
                'created_at': rec.created_at.isoformat()
            }
            for rec in recommendations
        ]

    async def _calculate_engagement_metrics(self, customer_id: str) -> Dict[str, float]:
        """Calculate engagement metrics for personalization"""
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Get recent interactions
        interactions = self.db.query(CustomerInteraction).filter(
            and_(
                CustomerInteraction.customer_id == customer_id,
                CustomerInteraction.started_at >= thirty_days_ago
            )
        ).all()
        
        # Get recent recommendations
        recommendations = self.db.query(CustomerRecommendation).filter(
            and_(
                CustomerRecommendation.customer_id == customer_id,
                CustomerRecommendation.created_at >= thirty_days_ago
            )
        ).all()
        
        metrics = {
            'interaction_frequency': len(interactions) / 30,
            'avg_satisfaction': 0.0,
            'response_rate': 0.0,
            'recommendation_ctr': 0.0,
            'engagement_score': 0.0
        }
        
        # Calculate average satisfaction
        satisfaction_ratings = [i.satisfaction_rating for i in interactions if i.satisfaction_rating]
        if satisfaction_ratings:
            metrics['avg_satisfaction'] = sum(satisfaction_ratings) / len(satisfaction_ratings)
        
        # Calculate recommendation CTR
        if recommendations:
            total_views = sum(r.views_count for r in recommendations)
            total_clicks = sum(r.clicks_count for r in recommendations)
            metrics['recommendation_ctr'] = (total_clicks / total_views) if total_views > 0 else 0.0
        
        # Calculate overall engagement score
        metrics['engagement_score'] = (
            metrics['interaction_frequency'] * 0.3 +
            metrics['avg_satisfaction'] / 5 * 0.3 +
            metrics['recommendation_ctr'] * 0.4
        )
        
        return metrics

class RecommendationEngine:
    """Generates personalized recommendations using multiple strategies"""
    
    def __init__(self, db: Session):
        self.db = db
        self.personalization_engine = PersonalizationEngine(db)

    async def generate_recommendations(
        self,
        request: RecommendationRequest
    ) -> List[PersonalizedRecommendation]:
        """Generate personalized recommendations"""
        try:
            # Build personalization profile
            profile = await self.personalization_engine.build_personalization_profile(
                request.customer_id
            )
            
            # Generate recommendations using different strategies
            recommendations = []
            
            if request.strategy in [PersonalizationStrategy.BEHAVIORAL, PersonalizationStrategy.HYBRID]:
                behavioral_recs = await self._generate_behavioral_recommendations(profile, request)
                recommendations.extend(behavioral_recs)
            
            if request.strategy in [PersonalizationStrategy.CONTENT_BASED, PersonalizationStrategy.HYBRID]:
                content_recs = await self._generate_content_based_recommendations(profile, request)
                recommendations.extend(content_recs)
            
            if request.strategy in [PersonalizationStrategy.CONTEXTUAL, PersonalizationStrategy.HYBRID]:
                contextual_recs = await self._generate_contextual_recommendations(profile, request)
                recommendations.extend(contextual_recs)
            
            # Score and rank recommendations
            scored_recommendations = await self._score_and_rank_recommendations(
                recommendations, profile, request
            )
            
            # Filter and limit results
            final_recommendations = await self._filter_recommendations(
                scored_recommendations, profile, request
            )
            
            # Store recommendations
            await self._store_recommendations(final_recommendations, request.customer_id)
            
            return final_recommendations[:request.max_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    async def _generate_behavioral_recommendations(
        self,
        profile: PersonalizationProfile,
        request: RecommendationRequest
    ) -> List[PersonalizedRecommendation]:
        """Generate recommendations based on behavioral patterns"""
        recommendations = []
        
        # Analyze recent behavior
        behavior = profile.behavior_patterns
        
        # Feature recommendations based on usage patterns
        if RecommendationType.FEATURE in request.recommendation_types:
            feature_recs = await self._recommend_features_by_behavior(profile)
            recommendations.extend(feature_recs)
        
        # Content recommendations based on engagement
        if RecommendationType.CONTENT in request.recommendation_types:
            content_recs = await self._recommend_content_by_behavior(profile)
            recommendations.extend(content_recs)
        
        # Action recommendations based on patterns
        if RecommendationType.ACTION in request.recommendation_types:
            action_recs = await self._recommend_actions_by_behavior(profile)
            recommendations.extend(action_recs)
        
        return recommendations

    async def _recommend_features_by_behavior(
        self,
        profile: PersonalizationProfile
    ) -> List[PersonalizedRecommendation]:
        """Recommend features based on behavioral patterns"""
        recommendations = []
        
        # Get current journey stage
        current_stage = JourneyStage(profile.journey_context['current_stage'])
        
        # Recommend features appropriate for current stage
        for feature_name, feature_data in self.personalization_engine.feature_catalog.items():
            if current_stage in feature_data['target_stages']:
                # Check if customer segment matches
                customer_segment = profile.segmentation_data.get('segment', 'individual')
                if customer_segment in feature_data.get('user_segments', [customer_segment]):
                    
                    # Calculate relevance based on interests
                    interests = profile.preferences.get('inferred_interests', [])
                    relevance = 0.5  # Base relevance
                    
                    if feature_name in interests:
                        relevance += 0.3
                    
                    # Boost based on engagement
                    if profile.engagement_metrics.get('engagement_score', 0) > 0.7:
                        relevance += 0.2
                    
                    recommendations.append(PersonalizedRecommendation(
                        recommendation_id=f"feature_{feature_name}",
                        type=RecommendationType.FEATURE,
                        title=f"Try {feature_name.replace('_', ' ').title()}",
                        description=f"Unlock {', '.join(feature_data['value_props'])} with {feature_name.replace('_', ' ')}",
                        confidence_score=0.8,
                        relevance_score=relevance,
                        priority="medium",
                        content_data={
                            'feature_name': feature_name,
                            'category': feature_data['category'],
                            'complexity': feature_data['complexity'],
                            'value_props': feature_data['value_props']
                        },
                        action_url=f"/features/{feature_name}",
                        image_url=f"/images/features/{feature_name}.jpg",
                        personalization_factors=['journey_stage', 'customer_segment', 'interests'],
                        reasoning=f"Recommended based on {current_stage.value} stage and {customer_segment} segment"
                    ))
        
        return recommendations

    async def _recommend_content_by_behavior(
        self,
        profile: PersonalizationProfile
    ) -> List[PersonalizedRecommendation]:
        """Recommend content based on behavior patterns"""
        recommendations = []
        
        current_stage = JourneyStage(profile.journey_context['current_stage'])
        
        for content_name, content_data in self.personalization_engine.content_catalog.items():
            if current_stage in content_data['target_stages']:
                # Calculate relevance based on engagement history
                relevance = 0.6  # Base relevance
                
                # Boost based on engagement metrics
                engagement_score = profile.engagement_metrics.get('engagement_score', 0)
                if engagement_score > 0.5:
                    relevance += 0.2
                
                # Adjust based on content type preferences
                if content_data['content_type'] in ['guide', 'documentation']:
                    # Prefer guides for new users
                    if current_stage in [JourneyStage.ONBOARDING, JourneyStage.CONSIDERATION]:
                        relevance += 0.1
                
                recommendations.append(PersonalizedRecommendation(
                    recommendation_id=f"content_{content_name}",
                    type=RecommendationType.CONTENT,
                    title=content_data.get('title', content_name.replace('_', ' ').title()),
                    description=f"Learn about {', '.join(content_data['topics'])} in this {content_data['content_type']}",
                    confidence_score=0.7,
                    relevance_score=relevance,
                    priority="low",
                    content_data=content_data,
                    action_url=f"/content/{content_name}",
                    image_url=f"/images/content/{content_name}.jpg",
                    personalization_factors=['journey_stage', 'engagement_score'],
                    reasoning=f"Recommended based on {current_stage.value} stage and content preferences"
                ))
        
        return recommendations

    async def _recommend_actions_by_behavior(
        self,
        profile: PersonalizationProfile
    ) -> List[PersonalizedRecommendation]:
        """Recommend actions based on behavior patterns"""
        recommendations = []
        
        # Low engagement - recommend engagement actions
        if profile.engagement_metrics.get('engagement_score', 0) < 0.3:
            recommendations.append(PersonalizedRecommendation(
                recommendation_id="action_boost_engagement",
                type=RecommendationType.ACTION,
                title="Boost Your Engagement",
                description="Complete your profile and explore key features to get the most out of the platform",
                confidence_score=0.9,
                relevance_score=0.8,
                priority="high",
                content_data={
                    'action_type': 'engagement_boost',
                    'steps': ['complete_profile', 'explore_features', 'set_preferences']
                },
                action_url="/onboarding/engagement",
                image_url="/images/actions/engagement.jpg",
                personalization_factors=['low_engagement'],
                reasoning="Low engagement detected - encouraging platform exploration"
            ))
        
        # Churn risk - recommend retention actions
        churn_risk = profile.journey_context.get('churn_risk_score', 0)
        if churn_risk > 0.6:
            recommendations.append(PersonalizedRecommendation(
                recommendation_id="action_retention_support",
                type=RecommendationType.ACTION,
                title="Get Personalized Help",
                description="Schedule a one-on-one session with our success team to maximize your results",
                confidence_score=0.95,
                relevance_score=0.9,
                priority="high",
                content_data={
                    'action_type': 'retention_support',
                    'contact_method': 'schedule_call'
                },
                action_url="/support/schedule",
                image_url="/images/actions/support.jpg",
                personalization_factors=['churn_risk'],
                reasoning=f"High churn risk detected ({churn_risk:.2f}) - offering personalized support"
            ))
        
        return recommendations

    async def _generate_content_based_recommendations(
        self,
        profile: PersonalizationProfile,
        request: RecommendationRequest
    ) -> List[PersonalizedRecommendation]:
        """Generate content-based recommendations"""
        recommendations = []
        
        # Find similar customers and their successful content
        similar_customers = await self._find_similar_customers(profile)
        
        # Recommend content that worked for similar customers
        for similar_customer in similar_customers[:3]:  # Top 3 similar customers
            successful_content = await self._get_successful_content(similar_customer)
            
            for content in successful_content:
                recommendations.append(PersonalizedRecommendation(
                    recommendation_id=f"similar_{content['id']}",
                    type=RecommendationType.CONTENT,
                    title=content['title'],
                    description=f"Customers like you found this helpful: {content['description']}",
                    confidence_score=0.6,
                    relevance_score=content['success_rate'],
                    priority="medium",
                    content_data=content,
                    action_url=content['url'],
                    image_url=content.get('image_url'),
                    personalization_factors=['similar_customers'],
                    reasoning="Recommended based on similar customer success patterns"
                ))
        
        return recommendations

    async def _generate_contextual_recommendations(
        self,
        profile: PersonalizationProfile,
        request: RecommendationRequest
    ) -> List[PersonalizedRecommendation]:
        """Generate contextual recommendations based on current context"""
        recommendations = []
        
        context = request.context
        current_stage = JourneyStage(profile.journey_context['current_stage'])
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            if current_stage == JourneyStage.TRIAL:
                recommendations.append(PersonalizedRecommendation(
                    recommendation_id="contextual_trial_support",
                    type=RecommendationType.SUPPORT,
                    title="Trial Support Available Now",
                    description="Get live help during business hours to make the most of your trial",
                    confidence_score=0.8,
                    relevance_score=0.7,
                    priority="medium",
                    content_data={
                        'support_type': 'live_chat',
                        'availability': 'business_hours'
                    },
                    action_url="/support/chat",
                    image_url="/images/support/live_chat.jpg",
                    personalization_factors=['time_context', 'trial_stage'],
                    reasoning="Business hours + trial stage = offer live support"
                ))
        
        return recommendations

    async def _score_and_rank_recommendations(
        self,
        recommendations: List[PersonalizedRecommendation],
        profile: PersonalizationProfile,
        request: RecommendationRequest
    ) -> List[PersonalizedRecommendation]:
        """Score and rank recommendations"""
        
        for rec in recommendations:
            # Calculate composite score
            composite_score = (
                rec.confidence_score * 0.4 +
                rec.relevance_score * 0.6
            )
            
            # Apply profile-specific adjustments
            if profile.engagement_metrics.get('engagement_score', 0) > 0.8:
                # High engagement users get complex recommendations
                if rec.content_data.get('complexity') == 'advanced':
                    composite_score += 0.1
            
            # Priority adjustments
            priority_multipliers = {
                'high': 1.2,
                'medium': 1.0,
                'low': 0.8
            }
            composite_score *= priority_multipliers.get(rec.priority, 1.0)
            
            # Update relevance score with composite
            rec.relevance_score = min(1.0, composite_score)
        
        # Sort by relevance score
        return sorted(recommendations, key=lambda x: x.relevance_score, reverse=True)

    async def _filter_recommendations(
        self,
        recommendations: List[PersonalizedRecommendation],
        profile: PersonalizationProfile,
        request: RecommendationRequest
    ) -> List[PersonalizedRecommendation]:
        """Filter recommendations based on request criteria"""
        filtered = []
        
        # Get previous recommendations if excluding
        previous_ids = set()
        if request.exclude_previous:
            previous_ids = set(
                rec['id'] for rec in profile.recommendation_history
                if rec['created_at'] > (datetime.now() - timedelta(days=7)).isoformat()
            )
        
        for rec in recommendations:
            # Skip if previously recommended
            if request.exclude_previous and rec.recommendation_id in previous_ids:
                continue
            
            # Filter by requested types
            if rec.type not in request.recommendation_types:
                continue
            
            # Minimum relevance threshold
            if rec.relevance_score < 0.3:
                continue
            
            filtered.append(rec)
        
        return filtered

    async def _store_recommendations(
        self,
        recommendations: List[PersonalizedRecommendation],
        customer_id: str
    ):
        """Store recommendations in database for tracking"""
        try:
            for rec in recommendations:
                db_rec = CustomerRecommendation(
                    customer_id=customer_id,
                    recommendation_type=rec.type.value,
                    title=rec.title,
                    description=rec.description,
                    confidence_score=rec.confidence_score,
                    relevance_score=rec.relevance_score,
                    priority=rec.priority,
                    content_data=rec.content_data,
                    action_url=rec.action_url,
                    image_url=rec.image_url,
                    personalization_factors=rec.personalization_factors,
                    target_attributes={'reasoning': rec.reasoning},
                    valid_from=datetime.now(timezone.utc),
                    valid_until=datetime.now(timezone.utc) + timedelta(days=30)
                )
                
                self.db.add(db_rec)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing recommendations: {e}")
            self.db.rollback()

    async def _find_similar_customers(self, profile: PersonalizationProfile) -> List[str]:
        """Find customers similar to the current customer"""
        # Simplified similarity based on segment and stage
        current_segment = profile.segmentation_data.get('segment')
        current_stage = profile.journey_context['current_stage']
        
        similar_customers = self.db.query(CustomerProfile).filter(
            and_(
                CustomerProfile.customer_segment == current_segment,
                CustomerProfile.current_journey_stage == JourneyStage(current_stage),
                CustomerProfile.id != profile.customer_id
            )
        ).limit(10).all()
        
        return [str(customer.id) for customer in similar_customers]

    async def _get_successful_content(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get content that was successful for a customer"""
        # Simplified - return mock successful content
        return [
            {
                'id': 'success_content_1',
                'title': 'Advanced Feature Guide',
                'description': 'Learn advanced features that boost productivity',
                'success_rate': 0.8,
                'url': '/content/advanced_guide',
                'image_url': '/images/content/advanced.jpg'
            }
        ]

    async def track_recommendation_interaction(
        self,
        recommendation_id: str,
        interaction_type: str,  # view, click, conversion
        customer_id: str
    ):
        """Track customer interactions with recommendations"""
        try:
            recommendation = self.db.query(CustomerRecommendation).filter(
                and_(
                    CustomerRecommendation.id == recommendation_id,
                    CustomerRecommendation.customer_id == customer_id
                )
            ).first()
            
            if recommendation:
                if interaction_type == 'view':
                    recommendation.views_count += 1
                elif interaction_type == 'click':
                    recommendation.clicks_count += 1
                elif interaction_type == 'conversion':
                    recommendation.conversions_count += 1
                
                self.db.commit()
                logger.info(f"Tracked {interaction_type} for recommendation {recommendation_id}")
            
        except Exception as e:
            logger.error(f"Error tracking recommendation interaction: {e}")
            self.db.rollback()