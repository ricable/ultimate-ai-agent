# File: backend/nlp/__init__.py
"""
Advanced Natural Language Processing Module for UAP Platform

This module provides comprehensive NLP capabilities including:
- Entity extraction and recognition
- Sentiment analysis and emotion detection
- Intent classification and understanding
- Multi-language support and translation
- Conversational AI with context management
- Voice-to-text and text-to-speech integration
"""

from .nlp_pipeline import NLPPipeline
from .entity_extraction import EntityExtractor
from .sentiment_analysis import SentimentAnalyzer
from .intent_classification import IntentClassifier
from .translation import TranslationService
from .speech_processing import SpeechProcessor
from .context_manager import ConversationContextManager

__all__ = [
    'NLPPipeline',
    'EntityExtractor',
    'SentimentAnalyzer', 
    'IntentClassifier',
    'TranslationService',
    'SpeechProcessor',
    'ConversationContextManager'
]

__version__ = "1.0.0"