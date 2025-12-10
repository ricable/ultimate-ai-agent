# File: backend/customer/sentiment_routing.py
import asyncio
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from ..models.customer import (
    CustomerProfile, CustomerInteraction, SentimentType, InteractionType,
    SatisfactionLevel, JourneyStage
)
from ..database.session import get_db
from ..services.agent_orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

class RoutingPriority(Enum):
    """Routing priority levels"""
    IMMEDIATE = "immediate"      # Urgent negative sentiment
    HIGH = "high"               # Negative sentiment
    NORMAL = "normal"           # Neutral sentiment  
    LOW = "low"                # Positive sentiment

class RoutingChannel(Enum):
    """Available routing channels"""
    HUMAN_AGENT = "human_agent"
    AI_AGENT = "ai_agent"
    SPECIALIZED_TEAM = "specialized_team"
    ESCALATION = "escalation"
    SELF_SERVICE = "self_service"

@dataclass
class SentimentAnalysisResult:
    """Result of sentiment analysis"""
    sentiment: SentimentType
    confidence: float
    scores: Dict[str, float]
    keywords: List[str]
    urgency_indicators: List[str]
    emotional_indicators: List[str]

@dataclass  
class RoutingDecision:
    """Routing decision with context"""
    channel: RoutingChannel
    priority: RoutingPriority
    agent_type: Optional[str]
    routing_reason: str
    estimated_wait_time: Optional[int]
    context: Dict[str, Any]

class SentimentAnalyzer:
    """Advanced sentiment analysis with context awareness"""
    
    def __init__(self):
        self.positive_keywords = {
            'love', 'great', 'excellent', 'amazing', 'fantastic', 'perfect', 
            'wonderful', 'awesome', 'brilliant', 'outstanding', 'satisfied',
            'happy', 'pleased', 'delighted', 'impressed', 'recommend'
        }
        
        self.negative_keywords = {
            'hate', 'terrible', 'awful', 'horrible', 'disgusting', 'worst',
            'frustrated', 'angry', 'disappointed', 'dissatisfied', 'broken',
            'useless', 'confusing', 'difficult', 'slow', 'error', 'problem',
            'issue', 'bug', 'crash', 'fail', 'wrong', 'bad'
        }
        
        self.urgency_indicators = {
            'urgent', 'emergency', 'critical', 'immediately', 'asap', 'now',
            'help', 'stuck', 'lost', 'cant', 'wont', 'broken', 'down',
            'billing', 'charged', 'refund', 'cancel', 'delete'
        }
        
        self.emotional_intensifiers = {
            'very', 'extremely', 'completely', 'totally', 'absolutely',
            'really', 'quite', 'so', 'too', 'incredibly', 'unbelievably'
        }
        
        self.context_modifiers = {
            'sarcasm': ['yeah right', 'sure', 'obviously', 'of course'],
            'negation': ['not', 'dont', 'wont', 'cant', 'never', 'no'],
            'conditional': ['if', 'when', 'unless', 'provided', 'assuming']
        }

    async def analyze_sentiment(self, text: str, context: Optional[Dict] = None) -> SentimentAnalysisResult:
        """Perform comprehensive sentiment analysis"""
        try:
            # Normalize text
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            # Calculate basic sentiment scores
            positive_score = self._calculate_keyword_score(words, self.positive_keywords)
            negative_score = self._calculate_keyword_score(words, self.negative_keywords)
            
            # Apply context modifiers
            positive_score, negative_score = self._apply_context_modifiers(
                text_lower, positive_score, negative_score
            )
            
            # Calculate intensity multiplier
            intensity_multiplier = self._calculate_intensity(words)
            positive_score *= intensity_multiplier
            negative_score *= intensity_multiplier
            
            # Determine sentiment and confidence
            sentiment, confidence = self._determine_sentiment(positive_score, negative_score)
            
            # Extract keywords and indicators
            keywords = self._extract_keywords(words)
            urgency_indicators = self._extract_urgency_indicators(words)
            emotional_indicators = self._extract_emotional_indicators(text_lower)
            
            return SentimentAnalysisResult(
                sentiment=sentiment,
                confidence=confidence,
                scores={
                    'positive': positive_score,
                    'negative': negative_score,
                    'neutral': max(0, 1 - positive_score - negative_score)
                },
                keywords=keywords,
                urgency_indicators=urgency_indicators,
                emotional_indicators=emotional_indicators
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentAnalysisResult(
                sentiment=SentimentType.NEUTRAL,
                confidence=0.0,
                scores={'positive': 0, 'negative': 0, 'neutral': 1},
                keywords=[],
                urgency_indicators=[],
                emotional_indicators=[]
            )

    def _calculate_keyword_score(self, words: List[str], keywords: set) -> float:
        """Calculate sentiment score based on keyword matches"""
        matches = sum(1 for word in words if word in keywords)
        return min(matches / len(words) * 3, 1.0) if words else 0

    def _apply_context_modifiers(self, text: str, positive: float, negative: float) -> Tuple[float, float]:
        """Apply context modifiers like negation"""
        # Check for negation
        for negation in self.context_modifiers['negation']:
            if negation in text:
                positive, negative = negative * 0.8, positive * 0.8
                break
                
        # Check for sarcasm indicators
        for sarcasm in self.context_modifiers['sarcasm']:
            if sarcasm in text:
                positive *= 0.3  # Reduce positive sentiment for sarcasm
                break
                
        return positive, negative

    def _calculate_intensity(self, words: List[str]) -> float:
        """Calculate emotional intensity multiplier"""
        intensifier_count = sum(1 for word in words if word in self.emotional_intensifiers)
        return min(1.0 + (intensifier_count * 0.2), 2.0)

    def _determine_sentiment(self, positive: float, negative: float) -> Tuple[SentimentType, float]:
        """Determine final sentiment and confidence"""
        diff = abs(positive - negative)
        
        if diff < 0.1:  # Very close scores
            return SentimentType.NEUTRAL, 0.6
        elif positive > negative:
            if positive > 0.6:
                return SentimentType.POSITIVE, min(0.9, 0.5 + diff)
            else:
                return SentimentType.NEUTRAL, 0.7
        else:
            if negative > 0.6:
                return SentimentType.NEGATIVE, min(0.9, 0.5 + diff)
            elif negative > 0.4:
                return SentimentType.FRUSTRATED, min(0.8, 0.5 + diff)
            else:
                return SentimentType.NEUTRAL, 0.7

    def _extract_keywords(self, words: List[str]) -> List[str]:
        """Extract relevant keywords"""
        relevant_words = []
        for word in words:
            if (word in self.positive_keywords or 
                word in self.negative_keywords or 
                word in self.urgency_indicators):
                relevant_words.append(word)
        return list(set(relevant_words))

    def _extract_urgency_indicators(self, words: List[str]) -> List[str]:
        """Extract urgency indicators from text"""
        return [word for word in words if word in self.urgency_indicators]

    def _extract_emotional_indicators(self, text: str) -> List[str]:
        """Extract emotional indicators from text"""
        indicators = []
        
        # Check for caps (shouting)
        if re.search(r'[A-Z]{3,}', text):
            indicators.append('caps_usage')
            
        # Check for excessive punctuation
        if re.search(r'[!]{2,}|[?]{2,}', text):
            indicators.append('excessive_punctuation')
            
        # Check for repetition
        if re.search(r'(.)\1{2,}', text):
            indicators.append('letter_repetition')
            
        return indicators

class SentimentRouter:
    """Intelligent routing based on sentiment and context"""
    
    def __init__(self, db: Session):
        self.db = db
        self.sentiment_analyzer = SentimentAnalyzer()
        self.agent_orchestrator = AgentOrchestrator()

    async def route_customer_interaction(
        self, 
        customer_id: str,
        message: str,
        interaction_type: InteractionType,
        context: Optional[Dict] = None
    ) -> RoutingDecision:
        """Route customer interaction based on sentiment and context"""
        try:
            # Get customer profile for context
            customer = self.db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            # Analyze sentiment
            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(
                message, context
            )
            
            # Get routing decision
            routing_decision = await self._determine_routing(
                customer, sentiment_result, interaction_type, context
            )
            
            # Record interaction
            await self._record_interaction(
                customer_id, message, interaction_type, sentiment_result, routing_decision
            )
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error routing customer interaction: {e}")
            return RoutingDecision(
                channel=RoutingChannel.AI_AGENT,
                priority=RoutingPriority.NORMAL,
                agent_type="general",
                routing_reason="Fallback routing due to error",
                estimated_wait_time=30,
                context={}
            )

    async def _determine_routing(
        self,
        customer: Optional[CustomerProfile],
        sentiment: SentimentAnalysisResult,
        interaction_type: InteractionType,
        context: Optional[Dict]
    ) -> RoutingDecision:
        """Determine optimal routing based on multiple factors"""
        
        # Initialize routing factors
        priority = RoutingPriority.NORMAL
        channel = RoutingChannel.AI_AGENT
        agent_type = "general"
        routing_reason = "Standard AI routing"
        wait_time = 30
        
        # Customer history factors
        if customer:
            # High-value customers get priority
            if customer.lifetime_value > 10000:
                priority = RoutingPriority.HIGH
                routing_reason = "High-value customer"
                
            # Customers with escalation history
            if customer.escalation_count > 2:
                channel = RoutingChannel.HUMAN_AGENT
                priority = RoutingPriority.HIGH
                routing_reason = "History of escalations"
                wait_time = 60
                
            # Customers at risk of churn
            if customer.churn_risk_score > 0.7:
                channel = RoutingChannel.SPECIALIZED_TEAM
                priority = RoutingPriority.HIGH
                agent_type = "retention"
                routing_reason = "Churn risk mitigation"
                wait_time = 45
        
        # Sentiment-based routing
        if sentiment.sentiment in [SentimentType.NEGATIVE, SentimentType.FRUSTRATED]:
            if sentiment.confidence > 0.8:
                priority = RoutingPriority.HIGH
                channel = RoutingChannel.HUMAN_AGENT
                routing_reason = f"High confidence {sentiment.sentiment.value} sentiment"
                wait_time = 90
            elif sentiment.urgency_indicators:
                priority = RoutingPriority.IMMEDIATE
                channel = RoutingChannel.ESCALATION
                routing_reason = "Urgent negative sentiment detected"
                wait_time = 15
                
        elif sentiment.sentiment == SentimentType.POSITIVE:
            if sentiment.confidence > 0.8:
                # Positive customers can handle AI well
                channel = RoutingChannel.AI_AGENT
                priority = RoutingPriority.LOW
                agent_type = "conversational"
                routing_reason = "Positive sentiment - AI capable"
                wait_time = 10

        # Content-based routing
        keywords = ' '.join(sentiment.keywords).lower()
        if any(billing_word in keywords for billing_word in ['billing', 'payment', 'refund', 'charge']):
            channel = RoutingChannel.SPECIALIZED_TEAM
            agent_type = "billing"
            priority = RoutingPriority.HIGH
            routing_reason = "Billing-related inquiry"
            wait_time = 45
            
        elif any(tech_word in keywords for tech_word in ['bug', 'error', 'crash', 'broken']):
            channel = RoutingChannel.SPECIALIZED_TEAM
            agent_type = "technical"
            routing_reason = "Technical issue detected"
            wait_time = 60

        # Interaction type considerations
        if interaction_type == InteractionType.PHONE:
            # Phone calls indicate higher urgency
            if priority == RoutingPriority.NORMAL:
                priority = RoutingPriority.HIGH
            wait_time = min(wait_time, 30)
            
        return RoutingDecision(
            channel=channel,
            priority=priority,
            agent_type=agent_type,
            routing_reason=routing_reason,
            estimated_wait_time=wait_time,
            context={
                'sentiment_confidence': sentiment.confidence,
                'urgency_indicators': sentiment.urgency_indicators,
                'emotional_indicators': sentiment.emotional_indicators,
                'customer_ltv': customer.lifetime_value if customer else 0,
                'customer_risk_score': customer.churn_risk_score if customer else 0
            }
        )

    async def _record_interaction(
        self,
        customer_id: str,
        message: str,
        interaction_type: InteractionType,
        sentiment: SentimentAnalysisResult,
        routing: RoutingDecision
    ):
        """Record the interaction in the database"""
        try:
            interaction = CustomerInteraction(
                customer_id=customer_id,
                interaction_type=interaction_type,
                channel=routing.channel.value,
                content=message,
                sentiment=sentiment.sentiment,
                sentiment_score=sentiment.scores.get('positive', 0) - sentiment.scores.get('negative', 0),
                sentiment_confidence=sentiment.confidence,
                context_data={
                    'routing_decision': {
                        'channel': routing.channel.value,
                        'priority': routing.priority.value,
                        'agent_type': routing.agent_type,
                        'reason': routing.routing_reason
                    },
                    'sentiment_analysis': {
                        'keywords': sentiment.keywords,
                        'urgency_indicators': sentiment.urgency_indicators,
                        'emotional_indicators': sentiment.emotional_indicators
                    }
                },
                tags=sentiment.keywords + sentiment.urgency_indicators,
                started_at=datetime.now(timezone.utc)
            )
            
            self.db.add(interaction)
            self.db.commit()
            
            # Update customer profile
            await self._update_customer_profile(customer_id, sentiment, routing)
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
            self.db.rollback()

    async def _update_customer_profile(
        self,
        customer_id: str,
        sentiment: SentimentAnalysisResult,
        routing: RoutingDecision
    ):
        """Update customer profile based on interaction"""
        try:
            customer = self.db.query(CustomerProfile).filter(
                CustomerProfile.id == customer_id
            ).first()
            
            if customer:
                # Update interaction count
                customer.total_interactions += 1
                
                # Update satisfaction based on sentiment
                if sentiment.sentiment in [SentimentType.POSITIVE, SentimentType.SATISFIED]:
                    if sentiment.confidence > 0.7:
                        customer.current_satisfaction = SatisfactionLevel.SATISFIED
                elif sentiment.sentiment in [SentimentType.NEGATIVE, SentimentType.FRUSTRATED]:
                    if sentiment.confidence > 0.7:
                        customer.current_satisfaction = SatisfactionLevel.DISSATISFIED
                        
                # Update churn risk based on negative interactions
                if sentiment.sentiment in [SentimentType.NEGATIVE, SentimentType.FRUSTRATED]:
                    customer.churn_risk_score = min(1.0, customer.churn_risk_score + 0.1)
                elif sentiment.sentiment == SentimentType.POSITIVE:
                    customer.churn_risk_score = max(0.0, customer.churn_risk_score - 0.05)
                
                # Update last interaction time
                customer.last_interaction_at = datetime.now(timezone.utc)
                
                # Update escalation count if routed to escalation
                if routing.channel == RoutingChannel.ESCALATION:
                    customer.escalation_count += 1
                
                self.db.commit()
                
        except Exception as e:
            logger.error(f"Error updating customer profile: {e}")
            self.db.rollback()

    async def get_routing_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get routing analytics for the specified period"""
        try:
            # Get interactions from the last N days
            from_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
            from_date = from_date.replace(day=from_date.day - days)
            
            interactions = self.db.query(CustomerInteraction).filter(
                CustomerInteraction.started_at >= from_date
            ).all()
            
            # Calculate analytics
            total_interactions = len(interactions)
            sentiment_breakdown = {}
            routing_breakdown = {}
            
            for interaction in interactions:
                # Sentiment analytics
                sentiment = interaction.sentiment.value if interaction.sentiment else 'unknown'
                sentiment_breakdown[sentiment] = sentiment_breakdown.get(sentiment, 0) + 1
                
                # Routing analytics
                routing_data = interaction.context_data.get('routing_decision', {})
                channel = routing_data.get('channel', 'unknown')
                routing_breakdown[channel] = routing_breakdown.get(channel, 0) + 1
            
            return {
                'total_interactions': total_interactions,
                'sentiment_breakdown': sentiment_breakdown,
                'routing_breakdown': routing_breakdown,
                'analysis_period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting routing analytics: {e}")
            return {
                'total_interactions': 0,
                'sentiment_breakdown': {},
                'routing_breakdown': {},
                'analysis_period_days': days,
                'error': str(e)
            }