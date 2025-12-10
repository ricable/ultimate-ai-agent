# File: backend/customer/__init__.py
"""
Customer Experience Module

This module provides comprehensive customer experience management including:
- Sentiment analysis and routing
- Customer journey tracking
- Personalization and recommendations
- Feedback collection and analysis
- Proactive engagement automation
"""

from .sentiment_routing import SentimentAnalyzer, SentimentRouter
from .journey_tracking import JourneyTracker, JourneyAnalyzer
from .personalization import PersonalizationEngine, RecommendationEngine
from .feedback_system import FeedbackCollector, SatisfactionTracker
from .api import CustomerExperienceAPI

__all__ = [
    'SentimentAnalyzer',
    'SentimentRouter', 
    'JourneyTracker',
    'JourneyAnalyzer',
    'PersonalizationEngine',
    'RecommendationEngine',
    'FeedbackCollector',
    'SatisfactionTracker',
    'CustomerExperienceAPI'
]

__version__ = "1.0.0"