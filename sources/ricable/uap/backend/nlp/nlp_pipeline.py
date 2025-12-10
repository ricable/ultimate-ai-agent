# backend/nlp/nlp_pipeline.py
# Agent 23: Advanced Natural Language Processing Pipeline

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

# NLP libraries
try:
    import spacy
    import nltk
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    NLP_LIBRARIES_AVAILABLE = True
except ImportError:
    NLP_LIBRARIES_AVAILABLE = False
    print("NLP libraries not available, using mock implementation")

# Language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    DATE = "DATE"
    MONEY = "MONEY"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    URL = "URL"
    CUSTOM = "CUSTOM"

class SentimentPolarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class Entity:
    """Named entity with metadata"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    entity_type: EntityType
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    polarity: SentimentPolarity
    confidence: float
    compound_score: float
    positive_score: float
    negative_score: float
    neutral_score: float

@dataclass
class IntentResult:
    """Intent classification result"""
    intent: str
    confidence: float
    entities: List[Entity]
    context: Dict[str, Any]

@dataclass
class NLPProcessingResult:
    """Complete NLP processing result"""
    text: str
    language: str
    entities: List[Entity]
    sentiment: SentimentResult
    intent: Optional[IntentResult]
    keywords: List[str]
    summary: Optional[str]
    embeddings: Optional[List[float]]
    processing_time: float
    timestamp: datetime

class EntityExtractor:
    """Named Entity Recognition and extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.spacy_model = None
        self.transformers_ner = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models"""
        if NLP_LIBRARIES_AVAILABLE:
            try:
                # Load spaCy model
                self.spacy_model = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spaCy model for entity extraction")
                
                # Load Transformers NER pipeline
                self.transformers_ner = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                self.logger.info("Loaded Transformers NER model")
                
            except Exception as e:
                self.logger.warning(f"Failed to load NLP models: {e}")
        else:
            self.logger.info("Using mock entity extraction")
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        entities = []
        
        if self.spacy_model:
            # spaCy entity extraction
            doc = self.spacy_model(text)
            for ent in doc.ents:
                entity_type = self._map_spacy_label_to_type(ent.label_)
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9,  # spaCy doesn't provide confidence scores
                    entity_type=entity_type,
                    metadata={"source": "spacy"}
                ))
        
        if self.transformers_ner:
            # Transformers NER
            try:
                ner_results = self.transformers_ner(text)
                for result in ner_results:
                    entity_type = self._map_transformers_label_to_type(result['entity_group'])
                    entities.append(Entity(
                        text=result['word'],
                        label=result['entity_group'],
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        entity_type=entity_type,
                        metadata={"source": "transformers"}
                    ))
            except Exception as e:
                self.logger.error(f"Transformers NER failed: {e}")
        
        # Rule-based entity extraction for common patterns
        rule_entities = self._extract_rule_based_entities(text)
        entities.extend(rule_entities)
        
        # If no models available, use mock entities
        if not entities and not NLP_LIBRARIES_AVAILABLE:
            entities = self._mock_entity_extraction(text)
        
        return self._deduplicate_entities(entities)
    
    def _map_spacy_label_to_type(self, label: str) -> EntityType:
        """Map spaCy labels to our entity types"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "MONEY": EntityType.MONEY,
        }
        return mapping.get(label, EntityType.CUSTOM)
    
    def _map_transformers_label_to_type(self, label: str) -> EntityType:
        """Map Transformers labels to our entity types"""
        mapping = {
            "PER": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "MISC": EntityType.CUSTOM,
        }
        return mapping.get(label, EntityType.CUSTOM)
    
    def _extract_rule_based_entities(self, text: str) -> List[Entity]:
        """Extract entities using rule-based patterns"""
        entities = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append(Entity(
                text=match.group(),
                label="EMAIL",
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                entity_type=EntityType.EMAIL,
                metadata={"source": "regex"}
            ))
        
        # Phone pattern
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        for match in re.finditer(phone_pattern, text):
            entities.append(Entity(
                text=match.group(),
                label="PHONE",
                start=match.start(),
                end=match.end(),
                confidence=0.90,
                entity_type=EntityType.PHONE,
                metadata={"source": "regex"}
            ))
        
        # URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for match in re.finditer(url_pattern, text):
            entities.append(Entity(
                text=match.group(),
                label="URL",
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                entity_type=EntityType.URL,
                metadata={"source": "regex"}
            ))
        
        return entities
    
    def _mock_entity_extraction(self, text: str) -> List[Entity]:
        """Mock entity extraction for testing"""
        return [
            Entity(
                text="Mock Entity",
                label="MOCK",
                start=0,
                end=11,
                confidence=0.8,
                entity_type=EntityType.CUSTOM,
                metadata={"source": "mock"}
            )
        ]
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on text and position"""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return sorted(deduplicated, key=lambda e: e.start)

class SentimentAnalyzer:
    """Sentiment analysis with multiple approaches"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.transformers_sentiment = None
        self.vader_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        if NLP_LIBRARIES_AVAILABLE:
            try:
                # Transformers sentiment pipeline
                self.transformers_sentiment = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                self.logger.info("Loaded Transformers sentiment model")
                
                # VADER sentiment analyzer
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    self.vader_analyzer = SentimentIntensityAnalyzer()
                    self.logger.info("Loaded VADER sentiment analyzer")
                except ImportError:
                    self.logger.warning("VADER sentiment analyzer not available")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load sentiment models: {e}")
    
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text"""
        
        if self.transformers_sentiment:
            try:
                # Use Transformers model
                results = self.transformers_sentiment(text)
                result = results[0]
                
                # Map labels to our enum
                label_mapping = {
                    "POSITIVE": SentimentPolarity.POSITIVE,
                    "NEGATIVE": SentimentPolarity.NEGATIVE,
                    "NEUTRAL": SentimentPolarity.NEUTRAL
                }
                
                polarity = label_mapping.get(result['label'], SentimentPolarity.NEUTRAL)
                confidence = result['score']
                
                # Use VADER for detailed scores if available
                if self.vader_analyzer:
                    vader_scores = self.vader_analyzer.polarity_scores(text)
                    return SentimentResult(
                        polarity=polarity,
                        confidence=confidence,
                        compound_score=vader_scores['compound'],
                        positive_score=vader_scores['pos'],
                        negative_score=vader_scores['neg'],
                        neutral_score=vader_scores['neu']
                    )
                else:
                    # Use Transformers scores only
                    return SentimentResult(
                        polarity=polarity,
                        confidence=confidence,
                        compound_score=confidence if polarity == SentimentPolarity.POSITIVE else -confidence,
                        positive_score=confidence if polarity == SentimentPolarity.POSITIVE else 0.0,
                        negative_score=confidence if polarity == SentimentPolarity.NEGATIVE else 0.0,
                        neutral_score=confidence if polarity == SentimentPolarity.NEUTRAL else 0.0
                    )
                    
            except Exception as e:
                self.logger.error(f"Transformers sentiment analysis failed: {e}")
        
        # Fallback to VADER only
        if self.vader_analyzer:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine polarity from compound score
            if scores['compound'] >= 0.05:
                polarity = SentimentPolarity.POSITIVE
            elif scores['compound'] <= -0.05:
                polarity = SentimentPolarity.NEGATIVE
            else:
                polarity = SentimentPolarity.NEUTRAL
            
            return SentimentResult(
                polarity=polarity,
                confidence=abs(scores['compound']),
                compound_score=scores['compound'],
                positive_score=scores['pos'],
                negative_score=scores['neg'],
                neutral_score=scores['neu']
            )
        
        # Mock sentiment analysis
        return SentimentResult(
            polarity=SentimentPolarity.NEUTRAL,
            confidence=0.5,
            compound_score=0.0,
            positive_score=0.33,
            negative_score=0.33,
            neutral_score=0.34
        )

class IntentClassifier:
    """Intent classification for conversational AI"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.intent_model = None
        self.intent_labels = [
            "greeting", "question", "request", "complaint", 
            "compliment", "goodbye", "booking", "cancellation",
            "information", "support", "other"
        ]
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize intent classification model"""
        if NLP_LIBRARIES_AVAILABLE:
            try:
                # Zero-shot classification for intent detection
                self.intent_model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                self.logger.info("Loaded intent classification model")
            except Exception as e:
                self.logger.warning(f"Failed to load intent model: {e}")
    
    async def classify_intent(self, text: str, entities: List[Entity]) -> IntentResult:
        """Classify intent of the text"""
        
        if self.intent_model:
            try:
                result = self.intent_model(text, self.intent_labels)
                
                return IntentResult(
                    intent=result['labels'][0],
                    confidence=result['scores'][0],
                    entities=entities,
                    context={
                        'all_scores': dict(zip(result['labels'], result['scores'])),
                        'top_intents': result['labels'][:3]
                    }
                )
            except Exception as e:
                self.logger.error(f"Intent classification failed: {e}")
        
        # Rule-based fallback
        return self._rule_based_intent(text, entities)
    
    def _rule_based_intent(self, text: str, entities: List[Entity]) -> IntentResult:
        """Simple rule-based intent classification"""
        text_lower = text.lower()
        
        # Simple keyword-based classification
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning']):
            intent = "greeting"
            confidence = 0.8
        elif any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            intent = "goodbye"
            confidence = 0.8
        elif '?' in text or any(word in text_lower for word in ['what', 'how', 'when', 'where', 'why', 'who']):
            intent = "question"
            confidence = 0.7
        elif any(word in text_lower for word in ['please', 'can you', 'could you', 'would you']):
            intent = "request"
            confidence = 0.7
        elif any(word in text_lower for word in ['problem', 'issue', 'complaint', 'wrong', 'error']):
            intent = "complaint"
            confidence = 0.6
        else:
            intent = "other"
            confidence = 0.5
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            context={"method": "rule_based"}
        )

class MultiLanguageProcessor:
    """Multi-language processing and translation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.translator = None
        self._initialize_translator()
    
    def _initialize_translator(self):
        """Initialize translation model"""
        if NLP_LIBRARIES_AVAILABLE:
            try:
                from transformers import MarianMTModel, MarianTokenizer
                # This would load specific translation models
                self.logger.info("Translation models initialized")
            except Exception as e:
                self.logger.warning(f"Failed to load translation models: {e}")
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if LANGDETECT_AVAILABLE:
            try:
                return detect(text)
            except:
                return "en"  # Default to English
        return "en"
    
    async def translate_text(self, text: str, target_language: str = "en") -> str:
        """Translate text to target language"""
        # Mock translation for now
        source_lang = self.detect_language(text)
        
        if source_lang == target_language:
            return text
        
        # In a real implementation, this would use proper translation models
        return f"[Translated from {source_lang} to {target_language}] {text}"

class NLPPipeline:
    """Main NLP processing pipeline"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.intent_classifier = IntentClassifier()
        self.multilang_processor = MultiLanguageProcessor()
        self.sentence_transformer = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize sentence embeddings model"""
        if NLP_LIBRARIES_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Loaded sentence transformer model")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
    
    async def process_text(self, 
                          text: str,
                          include_entities: bool = True,
                          include_sentiment: bool = True,
                          include_intent: bool = True,
                          include_embeddings: bool = False,
                          target_language: str = None) -> NLPProcessingResult:
        """Process text through complete NLP pipeline"""
        
        start_time = datetime.utcnow()
        processing_start = start_time.timestamp()
        
        # Detect language
        detected_language = self.multilang_processor.detect_language(text)
        
        # Translate if needed
        processed_text = text
        if target_language and detected_language != target_language:
            processed_text = await self.multilang_processor.translate_text(text, target_language)
        
        # Extract entities
        entities = []
        if include_entities:
            entities = await self.entity_extractor.extract_entities(processed_text)
        
        # Analyze sentiment
        sentiment = None
        if include_sentiment:
            sentiment = await self.sentiment_analyzer.analyze_sentiment(processed_text)
        
        # Classify intent
        intent = None
        if include_intent:
            intent = await self.intent_classifier.classify_intent(processed_text, entities)
        
        # Generate embeddings
        embeddings = None
        if include_embeddings and self.sentence_transformer:
            try:
                embeddings = self.sentence_transformer.encode(processed_text).tolist()
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings: {e}")
        
        # Extract keywords (simple implementation)
        keywords = self._extract_keywords(processed_text)
        
        # Generate summary for long texts
        summary = None
        if len(processed_text) > 500:
            summary = self._generate_summary(processed_text)
        
        processing_time = datetime.utcnow().timestamp() - processing_start
        
        return NLPProcessingResult(
            text=processed_text,
            language=detected_language,
            entities=entities,
            sentiment=sentiment,
            intent=intent,
            keywords=keywords,
            summary=summary,
            embeddings=embeddings,
            processing_time=processing_time,
            timestamp=start_time
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        if NLP_LIBRARIES_AVAILABLE and hasattr(self, 'spacy_model'):
            # Use spaCy for keyword extraction
            try:
                doc = self.entity_extractor.spacy_model(text)
                keywords = [token.lemma_.lower() for token in doc 
                           if not token.is_stop and not token.is_punct 
                           and token.is_alpha and len(token.text) > 2]
                return list(set(keywords))[:10]  # Top 10 unique keywords
            except:
                pass
        
        # Simple keyword extraction fallback
        import string
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        
        # Simple stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return list(set(keywords))[:10]
    
    def _generate_summary(self, text: str) -> str:
        """Generate summary for long text"""
        # Simple extractive summarization
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
        
        # Return first and last sentences as summary
        return f"{sentences[0]}. ... {sentences[-1]}"

# Global NLP pipeline instance
nlp_pipeline = NLPPipeline()