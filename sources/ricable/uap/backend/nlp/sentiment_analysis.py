# File: backend/nlp/sentiment_analysis.py
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from ..monitoring.logs.logger import uap_logger, LogLevel, EventType
from ..cache.decorators import cache_sentiment_analysis

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    sentiment: str  # positive, negative, neutral
    score: float    # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    emotions: Dict[str, float]  # emotion scores
    subjectivity: float  # 0.0 to 1.0 (objective to subjective)
    intensity: float  # 0.0 to 1.0 (low to high intensity)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment": self.sentiment,
            "score": self.score,
            "confidence": self.confidence,
            "emotions": self.emotions,
            "subjectivity": self.subjectivity,
            "intensity": self.intensity
        }

class SentimentAnalyzer:
    """
    Advanced Sentiment Analysis and Emotion Detection
    
    Provides comprehensive sentiment analysis including:
    - Basic sentiment classification (positive/negative/neutral)
    - Emotion detection (joy, anger, fear, sadness, etc.)
    - Subjectivity analysis (objective vs subjective)
    - Intensity measurement
    - Context-aware sentiment analysis
    """
    
    def __init__(self):
        self._initialized = False
        
        # Sentiment lexicons
        self.positive_words = {
            'excellent', 'amazing', 'fantastic', 'wonderful', 'great', 'good', 'awesome',
            'outstanding', 'superb', 'brilliant', 'perfect', 'love', 'like', 'enjoy',
            'happy', 'pleased', 'delighted', 'satisfied', 'content', 'glad', 'cheerful',
            'joyful', 'thrilled', 'excited', 'enthusiastic', 'optimistic', 'positive',
            'beautiful', 'nice', 'pleasant', 'lovely', 'charming', 'attractive',
            'impressive', 'remarkable', 'incredible', 'extraordinary', 'magnificent',
            'spectacular', 'stunning', 'breathtaking', 'marvelous', 'phenomenal',
            'exceptional', 'superior', 'first-rate', 'top-notch', 'premium', 'quality',
            'success', 'successful', 'achieve', 'accomplished', 'triumph', 'victory',
            'win', 'winner', 'best', 'better', 'improved', 'progress', 'advance',
            'benefit', 'advantage', 'profit', 'gain', 'valuable', 'worthy', 'useful',
            'helpful', 'effective', 'efficient', 'productive', 'reliable', 'trustworthy',
            'honest', 'sincere', 'genuine', 'authentic', 'loyal', 'faithful',
            'kind', 'generous', 'caring', 'compassionate', 'understanding', 'supportive',
            'friendly', 'warm', 'welcoming', 'inviting', 'comfortable', 'cozy',
            'peaceful', 'calm', 'relaxed', 'serene', 'tranquil', 'harmonious',
            'balanced', 'stable', 'secure', 'safe', 'protected', 'confident',
            'strong', 'powerful', 'capable', 'skilled', 'talented', 'gifted',
            'intelligent', 'smart', 'clever', 'wise', 'brilliant', 'creative',
            'innovative', 'original', 'unique', 'special', 'distinctive', 'memorable',
            'inspiring', 'motivating', 'encouraging', 'uplifting', 'empowering',
            'refreshing', 'energizing', 'invigorating', 'revitalizing', 'rejuvenating'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'dislike',
            'sad', 'angry', 'frustrated', 'annoyed', 'irritated', 'upset', 'disappointed',
            'depressed', 'miserable', 'unhappy', 'gloomy', 'pessimistic', 'negative',
            'ugly', 'disgusting', 'repulsive', 'offensive', 'unpleasant', 'disagreeable',
            'boring', 'dull', 'tedious', 'monotonous', 'tiresome', 'exhausting',
            'difficult', 'hard', 'challenging', 'problematic', 'troublesome', 'complex',
            'confusing', 'unclear', 'ambiguous', 'vague', 'uncertain', 'doubtful',
            'suspicious', 'questionable', 'unreliable', 'untrustworthy', 'dishonest',
            'fake', 'false', 'misleading', 'deceptive', 'fraudulent', 'corrupt',
            'unfair', 'unjust', 'biased', 'prejudiced', 'discriminatory', 'hostile',
            'aggressive', 'violent', 'cruel', 'harsh', 'severe', 'brutal',
            'dangerous', 'risky', 'hazardous', 'unsafe', 'threatening', 'scary',
            'fearful', 'anxious', 'worried', 'concerned', 'nervous', 'stressed',
            'overwhelmed', 'exhausted', 'tired', 'weak', 'powerless', 'helpless',
            'useless', 'worthless', 'meaningless', 'pointless', 'unnecessary', 'wasteful',
            'expensive', 'costly', 'overpriced', 'cheap', 'inferior', 'poor',
            'low-quality', 'defective', 'broken', 'damaged', 'faulty', 'flawed',
            'wrong', 'incorrect', 'mistaken', 'error', 'failure', 'disaster',
            'catastrophe', 'crisis', 'emergency', 'urgent', 'critical', 'serious',
            'severe', 'extreme', 'intense', 'overwhelming', 'unbearable', 'intolerable'
        }
        
        # Emotion lexicons
        self.emotion_words = {
            'joy': {'happy', 'joyful', 'cheerful', 'delighted', 'pleased', 'glad', 'excited', 'thrilled', 'elated', 'ecstatic', 'blissful', 'content', 'satisfied', 'fulfilled'},
            'anger': {'angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed', 'frustrated', 'outraged', 'livid', 'enraged', 'hostile', 'aggressive', 'violent'},
            'fear': {'afraid', 'scared', 'fearful', 'terrified', 'frightened', 'anxious', 'worried', 'nervous', 'panicked', 'alarmed', 'concerned', 'apprehensive'},
            'sadness': {'sad', 'depressed', 'miserable', 'unhappy', 'gloomy', 'melancholy', 'sorrowful', 'grief', 'despair', 'dejected', 'downhearted', 'mournful'},
            'surprise': {'surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'startled', 'bewildered', 'confused', 'perplexed', 'puzzled', 'baffled'},
            'disgust': {'disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'appalled', 'horrified', 'offended', 'outraged', 'repelled'},
            'trust': {'trust', 'confident', 'assured', 'certain', 'secure', 'reliable', 'dependable', 'faithful', 'loyal', 'honest', 'sincere'},
            'anticipation': {'excited', 'eager', 'enthusiastic', 'hopeful', 'optimistic', 'expectant', 'anticipating', 'looking forward', 'awaiting'}
        }
        
        # Intensity modifiers
        self.intensity_amplifiers = {
            'extremely', 'very', 'really', 'absolutely', 'completely', 'totally',
            'incredibly', 'amazingly', 'exceptionally', 'remarkably', 'significantly',
            'substantially', 'considerably', 'tremendously', 'enormously', 'immensely',
            'highly', 'deeply', 'profoundly', 'intensely', 'strongly', 'powerfully'
        }
        
        self.intensity_dampeners = {
            'slightly', 'somewhat', 'rather', 'fairly', 'quite', 'pretty',
            'relatively', 'moderately', 'reasonably', 'partially', 'barely',
            'hardly', 'scarcely', 'almost', 'nearly', 'just', 'only'
        }
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere',
            'neither', 'nor', 'cannot', 'cant', 'couldnt', 'shouldnt',
            'wouldnt', 'wont', 'dont', 'doesnt', 'didnt', 'isnt', 'arent',
            'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'without'
        }
        
        # Subjectivity indicators
        self.subjective_indicators = {
            'i think', 'i believe', 'i feel', 'in my opinion', 'personally',
            'i suppose', 'i guess', 'i assume', 'it seems', 'apparently',
            'presumably', 'allegedly', 'supposedly', 'probably', 'maybe',
            'perhaps', 'possibly', 'likely', 'unlikely', 'definitely',
            'certainly', 'obviously', 'clearly', 'surely', 'undoubtedly'
        }
        
        # Statistics
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_processing_time": 0,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
        }
    
    async def initialize(self):
        """Initialize sentiment analyzer"""
        if self._initialized:
            return
            
        try:
            # Convert word sets to lowercase for case-insensitive matching
            self.positive_words = {word.lower() for word in self.positive_words}
            self.negative_words = {word.lower() for word in self.negative_words}
            
            for emotion in self.emotion_words:
                self.emotion_words[emotion] = {word.lower() for word in self.emotion_words[emotion]}
            
            self.intensity_amplifiers = {word.lower() for word in self.intensity_amplifiers}
            self.intensity_dampeners = {word.lower() for word in self.intensity_dampeners}
            self.negation_words = {word.lower() for word in self.negation_words}
            
            self._initialized = True
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Sentiment Analyzer initialized successfully",
                EventType.SYSTEM,
                {
                    "positive_words": len(self.positive_words),
                    "negative_words": len(self.negative_words),
                    "emotion_categories": len(self.emotion_words)
                },
                "sentiment_analyzer"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Sentiment Analyzer initialization failed: {str(e)}",
                EventType.SYSTEM,
                {"error": str(e)},
                "sentiment_analyzer"
            )
            raise
    
    @cache_sentiment_analysis
    async def analyze_sentiment(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            context: Optional context for analysis
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Analyze sentiment
            sentiment_result = await self._analyze_text_sentiment(processed_text, context)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update statistics
            self.analysis_stats["total_analyses"] += 1
            self.analysis_stats["successful_analyses"] += 1
            self.analysis_stats["sentiment_distribution"][sentiment_result.sentiment] += 1
            self.analysis_stats["avg_processing_time"] = (
                (self.analysis_stats["avg_processing_time"] * (self.analysis_stats["total_analyses"] - 1) + processing_time) /
                self.analysis_stats["total_analyses"]
            )
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Sentiment analysis completed: {sentiment_result.sentiment} ({sentiment_result.score:.3f})",
                EventType.NLP,
                {
                    "text_length": len(text),
                    "sentiment": sentiment_result.sentiment,
                    "score": sentiment_result.score,
                    "confidence": sentiment_result.confidence,
                    "processing_time_ms": processing_time
                },
                "sentiment_analyzer"
            )
            
            return sentiment_result.to_dict()
            
        except Exception as e:
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.analysis_stats["failed_analyses"] += 1
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Sentiment analysis failed: {str(e)}",
                EventType.NLP,
                {
                    "text_length": len(text),
                    "error": str(e),
                    "processing_time_ms": processing_time
                },
                "sentiment_analyzer"
            )
            
            raise
    
    async def _analyze_text_sentiment(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]]
    ) -> SentimentResult:
        """Perform detailed sentiment analysis"""
        
        # Tokenize text
        words = self._tokenize(text)
        
        # Calculate basic sentiment score
        sentiment_score = self._calculate_sentiment_score(words, text)
        
        # Analyze emotions
        emotions = self._analyze_emotions(words)
        
        # Calculate subjectivity
        subjectivity = self._calculate_subjectivity(text, words)
        
        # Calculate intensity
        intensity = self._calculate_intensity(words, text)
        
        # Determine sentiment label
        sentiment_label = self._determine_sentiment_label(sentiment_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(sentiment_score, emotions, subjectivity, intensity)
        
        # Apply context adjustments if available
        if context:
            sentiment_score, confidence = self._apply_context_adjustments(
                sentiment_score, confidence, context
            )
            sentiment_label = self._determine_sentiment_label(sentiment_score)
        
        return SentimentResult(
            sentiment=sentiment_label,
            score=sentiment_score,
            confidence=confidence,
            emotions=emotions,
            subjectivity=subjectivity,
            intensity=intensity
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenization - can be enhanced with more sophisticated methods
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _calculate_sentiment_score(self, words: List[str], text: str) -> float:
        """Calculate sentiment score from -1.0 to 1.0"""
        positive_score = 0
        negative_score = 0
        
        # Track negation context
        negation_context = False
        negation_window = 0
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.negation_words:
                negation_context = True
                negation_window = 3  # Negation affects next 3 words
                continue
            
            # Reduce negation window
            if negation_window > 0:
                negation_window -= 1
                if negation_window == 0:
                    negation_context = False
            
            # Check intensity modifiers
            intensity_multiplier = 1.0
            if i > 0:
                prev_word = words[i-1]
                if prev_word in self.intensity_amplifiers:
                    intensity_multiplier = 1.5
                elif prev_word in self.intensity_dampeners:
                    intensity_multiplier = 0.5
            
            # Calculate sentiment contribution
            if word in self.positive_words:
                score = 1.0 * intensity_multiplier
                if negation_context:
                    negative_score += score
                else:
                    positive_score += score
            
            elif word in self.negative_words:
                score = 1.0 * intensity_multiplier
                if negation_context:
                    positive_score += score
                else:
                    negative_score += score
        
        # Calculate final score
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Normalize scores
        positive_score = positive_score / total_words
        negative_score = negative_score / total_words
        
        # Final sentiment score
        sentiment_score = positive_score - negative_score
        
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, sentiment_score))
    
    def _analyze_emotions(self, words: List[str]) -> Dict[str, float]:
        """Analyze emotions in text"""
        emotions = {}
        
        for emotion, emotion_words in self.emotion_words.items():
            emotion_score = 0
            
            for word in words:
                if word in emotion_words:
                    emotion_score += 1
            
            # Normalize by text length
            if len(words) > 0:
                emotions[emotion] = emotion_score / len(words)
            else:
                emotions[emotion] = 0.0
        
        return emotions
    
    def _calculate_subjectivity(self, text: str, words: List[str]) -> float:
        """Calculate subjectivity score from 0.0 (objective) to 1.0 (subjective)"""
        subjective_score = 0
        
        # Check for subjective indicators
        text_lower = text.lower()
        for indicator in self.subjective_indicators:
            if indicator in text_lower:
                subjective_score += 1
        
        # Check for first-person pronouns
        first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself'}
        for word in words:
            if word in first_person_pronouns:
                subjective_score += 0.5
        
        # Check for opinion words
        opinion_words = {'think', 'believe', 'feel', 'opinion', 'view', 'perspective'}
        for word in words:
            if word in opinion_words:
                subjective_score += 0.3
        
        # Normalize
        max_possible_score = len(self.subjective_indicators) + len(words) * 0.5 + len(words) * 0.3
        if max_possible_score > 0:
            subjectivity = min(1.0, subjective_score / max_possible_score)
        else:
            subjectivity = 0.0
        
        return subjectivity
    
    def _calculate_intensity(self, words: List[str], text: str) -> float:
        """Calculate intensity score from 0.0 to 1.0"""
        intensity_score = 0
        
        # Check for intensity modifiers
        for word in words:
            if word in self.intensity_amplifiers:
                intensity_score += 1.0
            elif word in self.intensity_dampeners:
                intensity_score += 0.3
        
        # Check for capitalization (SHOUTING)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            intensity_score += 0.5
        
        # Check for exclamation marks
        exclamations = text.count('!')
        if exclamations > 0:
            intensity_score += min(1.0, exclamations * 0.2)
        
        # Check for question marks (curiosity/confusion)
        questions = text.count('?')
        if questions > 0:
            intensity_score += min(0.5, questions * 0.1)
        
        # Normalize
        max_possible_score = len(words) + 0.5 + 1.0 + 0.5
        if max_possible_score > 0:
            intensity = min(1.0, intensity_score / max_possible_score)
        else:
            intensity = 0.0
        
        return intensity
    
    def _determine_sentiment_label(self, score: float) -> str:
        """Determine sentiment label from score"""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(
        self, 
        sentiment_score: float, 
        emotions: Dict[str, float],
        subjectivity: float,
        intensity: float
    ) -> float:
        """Calculate confidence in sentiment analysis"""
        
        # Base confidence from sentiment score magnitude
        base_confidence = abs(sentiment_score)
        
        # Adjust based on emotion presence
        emotion_confidence = sum(emotions.values())
        
        # Adjust based on subjectivity
        subjectivity_confidence = subjectivity * 0.5
        
        # Adjust based on intensity
        intensity_confidence = intensity * 0.3
        
        # Combined confidence
        confidence = (
            base_confidence * 0.5 +
            emotion_confidence * 0.3 +
            subjectivity_confidence * 0.1 +
            intensity_confidence * 0.1
        )
        
        return min(1.0, confidence)
    
    def _apply_context_adjustments(
        self, 
        sentiment_score: float, 
        confidence: float,
        context: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Apply context-based adjustments to sentiment analysis"""
        
        # Adjust based on conversation history
        if "conversation_history" in context:
            history = context["conversation_history"]
            if len(history) > 0:
                # Consider sentiment trend in conversation
                prev_sentiments = [msg.get("sentiment_score", 0) for msg in history[-3:]]
                if prev_sentiments:
                    avg_prev_sentiment = sum(prev_sentiments) / len(prev_sentiments)
                    # Slight adjustment based on conversation trend
                    sentiment_score = sentiment_score * 0.8 + avg_prev_sentiment * 0.2
        
        # Adjust based on user profile
        if "user_profile" in context:
            profile = context["user_profile"]
            # Consider user's typical sentiment patterns
            user_sentiment_bias = profile.get("avg_sentiment", 0)
            if abs(user_sentiment_bias) > 0.1:
                # Slight adjustment for user's typical sentiment
                sentiment_score = sentiment_score * 0.9 + user_sentiment_bias * 0.1
        
        # Adjust based on domain/topic
        if "domain" in context:
            domain = context["domain"]
            # Different domains might have different sentiment baselines
            domain_adjustments = {
                "support": -0.1,  # Support conversations tend to be more negative
                "sales": 0.1,     # Sales conversations tend to be more positive
                "feedback": 0.0,  # Feedback is neutral
                "casual": 0.05    # Casual conversations slightly positive
            }
            
            adjustment = domain_adjustments.get(domain, 0)
            sentiment_score += adjustment
        
        # Ensure score stays in bounds
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        return sentiment_score, confidence
    
    async def analyze_conversation_sentiment_trend(
        self, 
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment trend across conversation messages"""
        
        if not messages:
            return {"error": "No messages provided"}
        
        try:
            sentiment_history = []
            
            for msg in messages:
                content = msg.get("content", "")
                if content:
                    sentiment_result = await self.analyze_sentiment(content)
                    sentiment_history.append({
                        "timestamp": msg.get("timestamp"),
                        "sentiment": sentiment_result["sentiment"],
                        "score": sentiment_result["score"],
                        "confidence": sentiment_result["confidence"]
                    })
            
            if not sentiment_history:
                return {"error": "No valid messages found"}
            
            # Calculate trend metrics
            scores = [s["score"] for s in sentiment_history]
            
            trend_analysis = {
                "message_count": len(sentiment_history),
                "sentiment_history": sentiment_history,
                "average_sentiment": sum(scores) / len(scores),
                "sentiment_variance": self._calculate_variance(scores),
                "trend_direction": self._calculate_trend_direction(scores),
                "dominant_sentiment": self._get_dominant_sentiment(sentiment_history),
                "sentiment_changes": self._count_sentiment_changes(sentiment_history)
            }
            
            return trend_analysis
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Conversation sentiment trend analysis failed: {str(e)}",
                EventType.NLP,
                {"error": str(e)},
                "sentiment_analyzer"
            )
            raise
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance in sentiment scores"""
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        return variance
    
    def _calculate_trend_direction(self, scores: List[float]) -> str:
        """Calculate overall trend direction"""
        if len(scores) < 2:
            return "stable"
        
        # Simple linear trend calculation
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        
        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _get_dominant_sentiment(self, sentiment_history: List[Dict]) -> str:
        """Get the dominant sentiment in the conversation"""
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for item in sentiment_history:
            sentiment = item["sentiment"]
            sentiment_counts[sentiment] += 1
        
        return max(sentiment_counts, key=sentiment_counts.get)
    
    def _count_sentiment_changes(self, sentiment_history: List[Dict]) -> int:
        """Count how many times sentiment changes in the conversation"""
        if len(sentiment_history) < 2:
            return 0
        
        changes = 0
        for i in range(1, len(sentiment_history)):
            if sentiment_history[i]["sentiment"] != sentiment_history[i-1]["sentiment"]:
                changes += 1
        
        return changes
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics"""
        return {
            **self.analysis_stats,
            "lexicon_sizes": {
                "positive_words": len(self.positive_words),
                "negative_words": len(self.negative_words),
                "emotion_categories": len(self.emotion_words),
                "intensity_amplifiers": len(self.intensity_amplifiers),
                "intensity_dampeners": len(self.intensity_dampeners)
            },
            "initialized": self._initialized
        }
    
    async def cleanup(self):
        """Clean up sentiment analyzer resources"""
        try:
            # Clear lexicons
            self.positive_words.clear()
            self.negative_words.clear()
            self.emotion_words.clear()
            self.intensity_amplifiers.clear()
            self.intensity_dampeners.clear()
            self.negation_words.clear()
            self.subjective_indicators.clear()
            
            self._initialized = False
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Sentiment Analyzer cleanup completed",
                EventType.SYSTEM,
                self.analysis_stats,
                "sentiment_analyzer"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Sentiment Analyzer cleanup failed: {str(e)}",
                EventType.SYSTEM,
                {"error": str(e)},
                "sentiment_analyzer"
            )