# File: backend/nlp/intent_classification.py
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from ..monitoring.logs.logger import uap_logger, LogLevel, EventType
from ..cache.decorators import cache_intent_classification

@dataclass
class IntentResult:
    """Intent classification result"""
    intent: str
    confidence: float
    sub_intents: List[Dict[str, Any]]
    entities_required: List[str]
    action_type: str  # query, command, request, response
    urgency: str  # low, medium, high, critical
    complexity: str  # simple, moderate, complex
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "sub_intents": self.sub_intents,
            "entities_required": self.entities_required,
            "action_type": self.action_type,
            "urgency": self.urgency,
            "complexity": self.complexity
        }

class IntentClassifier:
    """
    Advanced Intent Classification and Understanding
    
    Classifies user intents including:
    - Primary intent identification
    - Sub-intent detection
    - Action type classification
    - Urgency and complexity assessment
    - Context-aware intent understanding
    - Multi-intent detection
    """
    
    def __init__(self):
        self._initialized = False
        
        # Intent patterns and keywords
        self.intent_patterns = {
            # Information seeking intents
            "information_request": {
                "keywords": {
                    "what", "how", "when", "where", "why", "who", "which", "explain",
                    "tell me", "show me", "describe", "define", "meaning", "information",
                    "details", "facts", "data", "statistics", "report", "summary"
                },
                "patterns": [
                    r"what is|what are|what's",
                    r"how do|how to|how can|how does",
                    r"when is|when does|when will",
                    r"where is|where can|where do",
                    r"why is|why does|why do",
                    r"who is|who are|who can"
                ],
                "action_type": "query",
                "entities_required": [],
                "urgency": "low",
                "complexity": "simple"
            },
            
            # Help and support intents
            "help_request": {
                "keywords": {
                    "help", "assist", "support", "guide", "tutorial", "instructions",
                    "stuck", "confused", "problem", "issue", "error", "trouble",
                    "can you", "please", "need", "want", "assistance"
                },
                "patterns": [
                    r"help me|help with|need help",
                    r"can you help|could you help",
                    r"i need|i want|i would like",
                    r"how do i|show me how",
                    r"i'm stuck|i'm confused"
                ],
                "action_type": "request",
                "entities_required": ["problem_description"],
                "urgency": "medium",
                "complexity": "moderate"
            },
            
            # Task execution intents
            "task_execution": {
                "keywords": {
                    "create", "make", "build", "generate", "produce", "develop",
                    "write", "draft", "compose", "design", "implement", "setup",
                    "configure", "install", "execute", "run", "perform", "do"
                },
                "patterns": [
                    r"create|make|build|generate",
                    r"write|draft|compose",
                    r"setup|configure|install",
                    r"execute|run|perform",
                    r"please create|please make"
                ],
                "action_type": "command",
                "entities_required": ["task_description", "parameters"],
                "urgency": "medium",
                "complexity": "moderate"
            },
            
            # Data analysis intents
            "data_analysis": {
                "keywords": {
                    "analyze", "examine", "review", "study", "investigate", "explore",
                    "compare", "contrast", "evaluate", "assess", "calculate", "compute",
                    "statistics", "trends", "patterns", "insights", "metrics", "data"
                },
                "patterns": [
                    r"analyze|examine|review",
                    r"compare|contrast|evaluate",
                    r"calculate|compute|measure",
                    r"find patterns|identify trends",
                    r"what are the trends"
                ],
                "action_type": "query",
                "entities_required": ["data_source", "analysis_type"],
                "urgency": "low",
                "complexity": "complex"
            },
            
            # Document processing intents
            "document_processing": {
                "keywords": {
                    "document", "file", "pdf", "text", "paper", "report", "article",
                    "summarize", "extract", "parse", "read", "process", "convert",
                    "upload", "download", "save", "export", "import", "format"
                },
                "patterns": [
                    r"process|parse|extract from",
                    r"summarize|summary of",
                    r"read|review|analyze document",
                    r"convert|transform|format",
                    r"upload|download|save"
                ],
                "action_type": "command",
                "entities_required": ["document_path", "processing_type"],
                "urgency": "medium",
                "complexity": "moderate"
            },
            
            # Problem solving intents
            "problem_solving": {
                "keywords": {
                    "solve", "fix", "resolve", "debug", "troubleshoot", "diagnose",
                    "error", "bug", "issue", "problem", "fault", "failure",
                    "broken", "not working", "failed", "crashed", "exception"
                },
                "patterns": [
                    r"solve|fix|resolve",
                    r"debug|troubleshoot|diagnose",
                    r"not working|doesn't work",
                    r"error|bug|issue|problem",
                    r"how to fix|how to solve"
                ],
                "action_type": "request",
                "entities_required": ["problem_description", "error_details"],
                "urgency": "high",
                "complexity": "complex"
            },
            
            # Configuration intents
            "configuration": {
                "keywords": {
                    "configure", "setup", "install", "settings", "preferences",
                    "options", "parameters", "config", "customize", "adjust",
                    "modify", "change", "update", "upgrade", "enable", "disable"
                },
                "patterns": [
                    r"configure|setup|install",
                    r"change|modify|adjust",
                    r"enable|disable|activate",
                    r"settings|preferences|options",
                    r"how to configure|how to setup"
                ],
                "action_type": "command",
                "entities_required": ["component", "settings"],
                "urgency": "medium",
                "complexity": "moderate"
            },
            
            # Search intents
            "search": {
                "keywords": {
                    "search", "find", "look for", "locate", "discover", "identify",
                    "browse", "explore", "filter", "query", "retrieve", "get"
                },
                "patterns": [
                    r"search for|find|look for",
                    r"where can i find|how to find",
                    r"browse|explore|discover",
                    r"filter|query|retrieve"
                ],
                "action_type": "query",
                "entities_required": ["search_terms"],
                "urgency": "low",
                "complexity": "simple"
            },
            
            # Communication intents
            "communication": {
                "keywords": {
                    "send", "email", "message", "notify", "alert", "contact",
                    "share", "publish", "post", "announce", "broadcast", "communicate"
                },
                "patterns": [
                    r"send|email|message",
                    r"notify|alert|contact",
                    r"share|publish|post",
                    r"announce|broadcast"
                ],
                "action_type": "command",
                "entities_required": ["recipient", "content"],
                "urgency": "medium",
                "complexity": "simple"
            },
            
            # Learning intents
            "learning": {
                "keywords": {
                    "learn", "teach", "explain", "understand", "study", "practice",
                    "tutorial", "lesson", "course", "training", "education",
                    "knowledge", "skill", "expertise", "master", "improve"
                },
                "patterns": [
                    r"learn|teach|explain",
                    r"tutorial|lesson|course",
                    r"how to learn|want to understand",
                    r"practice|study|improve"
                ],
                "action_type": "request",
                "entities_required": ["topic", "learning_level"],
                "urgency": "low",
                "complexity": "moderate"
            },
            
            # Comparison intents
            "comparison": {
                "keywords": {
                    "compare", "versus", "vs", "difference", "similar", "better",
                    "worse", "best", "worst", "pros", "cons", "advantages",
                    "disadvantages", "alternative", "option", "choice"
                },
                "patterns": [
                    r"compare|versus|vs",
                    r"difference between|what's different",
                    r"better|worse|best|worst",
                    r"pros and cons|advantages",
                    r"which is better|which should i"
                ],
                "action_type": "query",
                "entities_required": ["items_to_compare"],
                "urgency": "low",
                "complexity": "moderate"
            },
            
            # Planning intents
            "planning": {
                "keywords": {
                    "plan", "schedule", "organize", "arrange", "prepare", "strategy",
                    "roadmap", "timeline", "agenda", "calendar", "appointment",
                    "meeting", "event", "project", "task", "goal", "objective"
                },
                "patterns": [
                    r"plan|schedule|organize",
                    r"prepare|arrange|setup",
                    r"create plan|make schedule",
                    r"timeline|roadmap|strategy"
                ],
                "action_type": "command",
                "entities_required": ["planning_scope", "timeline"],
                "urgency": "medium",
                "complexity": "complex"
            },
            
            # Emergency intents
            "emergency": {
                "keywords": {
                    "urgent", "emergency", "critical", "asap", "immediately", "now",
                    "crisis", "disaster", "failure", "down", "crashed", "broken",
                    "help!", "emergency!", "urgent!", "critical!", "fix now"
                },
                "patterns": [
                    r"urgent|emergency|critical",
                    r"asap|immediately|right now",
                    r"help!|emergency!|urgent!",
                    r"system down|crashed|broken"
                ],
                "action_type": "request",
                "entities_required": ["emergency_type", "impact"],
                "urgency": "critical",
                "complexity": "complex"
            },
            
            # Greeting intents
            "greeting": {
                "keywords": {
                    "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
                    "good evening", "welcome", "thanks", "thank you", "goodbye", "bye"
                },
                "patterns": [
                    r"hello|hi|hey|greetings",
                    r"good morning|good afternoon|good evening",
                    r"thanks|thank you|thanks!",
                    r"goodbye|bye|see you"
                ],
                "action_type": "response",
                "entities_required": [],
                "urgency": "low",
                "complexity": "simple"
            }
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        
        # Context modifiers
        self.context_keywords = {
            "temporal": {"now", "today", "tomorrow", "yesterday", "soon", "later", "asap", "immediately"},
            "conditional": {"if", "when", "unless", "provided", "assuming", "suppose"},
            "quantitative": {"all", "some", "many", "few", "several", "multiple", "single"},
            "qualitative": {"best", "worst", "better", "good", "bad", "excellent", "poor"}
        }
        
        # Intent confidence thresholds
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        # Statistics
        self.classification_stats = {
            "total_classifications": 0,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "intent_distribution": {},
            "avg_processing_time": 0,
            "avg_confidence": 0
        }
    
    async def initialize(self):
        """Initialize intent classifier"""
        if self._initialized:
            return
        
        try:
            # Compile regex patterns for better performance
            for intent, config in self.intent_patterns.items():
                patterns = config.get("patterns", [])
                if patterns:
                    combined_pattern = "|".join(patterns)
                    self.compiled_patterns[intent] = re.compile(combined_pattern, re.IGNORECASE)
            
            self._initialized = True
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Intent Classifier initialized successfully",
                EventType.SYSTEM,
                {
                    "intent_types": len(self.intent_patterns),
                    "compiled_patterns": len(self.compiled_patterns)
                },
                "intent_classifier"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Intent Classifier initialization failed: {str(e)}",
                EventType.SYSTEM,
                {"error": str(e)},
                "intent_classifier"
            )
            raise
    
    @cache_intent_classification
    async def classify_intent(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify intent of text
        
        Args:
            text: Text to classify
            context: Optional context for classification
            
        Returns:
            Dictionary with intent classification results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Classify intent
            intent_result = await self._classify_text_intent(processed_text, text, context)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update statistics
            self.classification_stats["total_classifications"] += 1
            self.classification_stats["successful_classifications"] += 1
            
            intent = intent_result.intent
            if intent not in self.classification_stats["intent_distribution"]:
                self.classification_stats["intent_distribution"][intent] = 0
            self.classification_stats["intent_distribution"][intent] += 1
            
            self.classification_stats["avg_processing_time"] = (
                (self.classification_stats["avg_processing_time"] * (self.classification_stats["total_classifications"] - 1) + processing_time) /
                self.classification_stats["total_classifications"]
            )
            
            self.classification_stats["avg_confidence"] = (
                (self.classification_stats["avg_confidence"] * (self.classification_stats["total_classifications"] - 1) + intent_result.confidence) /
                self.classification_stats["total_classifications"]
            )
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Intent classification completed: {intent_result.intent} ({intent_result.confidence:.3f})",
                EventType.NLP,
                {
                    "text_length": len(text),
                    "intent": intent_result.intent,
                    "confidence": intent_result.confidence,
                    "action_type": intent_result.action_type,
                    "urgency": intent_result.urgency,
                    "processing_time_ms": processing_time
                },
                "intent_classifier"
            )
            
            return intent_result.to_dict()
            
        except Exception as e:
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.classification_stats["failed_classifications"] += 1
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Intent classification failed: {str(e)}",
                EventType.NLP,
                {
                    "text_length": len(text),
                    "error": str(e),
                    "processing_time_ms": processing_time
                },
                "intent_classifier"
            )
            
            raise
    
    async def _classify_text_intent(
        self, 
        processed_text: str, 
        original_text: str,
        context: Optional[Dict[str, Any]]
    ) -> IntentResult:
        """Perform detailed intent classification"""
        
        # Calculate scores for each intent
        intent_scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = self._calculate_intent_score(processed_text, original_text, intent, config)
            if score > 0:
                intent_scores[intent] = score
        
        # Find best matching intent
        if not intent_scores:
            # Fallback to information request for unknown intents
            best_intent = "information_request"
            best_score = 0.3
        else:
            best_intent = max(intent_scores, key=intent_scores.get)
            best_score = intent_scores[best_intent]
        
        # Get intent configuration
        intent_config = self.intent_patterns[best_intent]
        
        # Detect sub-intents
        sub_intents = self._detect_sub_intents(processed_text, original_text, intent_scores)
        
        # Apply context adjustments
        if context:
            best_score, urgency, complexity = self._apply_context_adjustments(
                best_intent, best_score, context, intent_config
            )
        else:
            urgency = intent_config["urgency"]
            complexity = intent_config["complexity"]
        
        # Calculate final confidence
        confidence = min(1.0, best_score)
        
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            sub_intents=sub_intents,
            entities_required=intent_config["entities_required"],
            action_type=intent_config["action_type"],
            urgency=urgency,
            complexity=complexity
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for intent classification"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
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
    
    def _calculate_intent_score(
        self, 
        processed_text: str, 
        original_text: str,
        intent: str, 
        config: Dict[str, Any]
    ) -> float:
        """Calculate score for a specific intent"""
        score = 0.0
        
        # Check keyword matches
        keywords = config.get("keywords", set())
        text_words = set(processed_text.split())
        
        keyword_matches = len(keywords.intersection(text_words))
        if keyword_matches > 0:
            # Weighted by keyword density
            keyword_score = keyword_matches / max(len(text_words), 1)
            score += keyword_score * 0.6
        
        # Check pattern matches
        if intent in self.compiled_patterns:
            pattern = self.compiled_patterns[intent]
            matches = pattern.findall(processed_text)
            if matches:
                # Weight by number and strength of pattern matches
                pattern_score = min(1.0, len(matches) * 0.3)
                score += pattern_score * 0.4
        
        # Check for urgency indicators
        urgency_words = {"urgent", "asap", "immediately", "now", "emergency", "critical"}
        if any(word in processed_text for word in urgency_words):
            if intent in ["emergency", "problem_solving", "help_request"]:
                score += 0.2
        
        # Check for question indicators
        question_words = {"what", "how", "when", "where", "why", "who", "which"}
        has_question_mark = "?" in original_text
        has_question_words = any(word in processed_text for word in question_words)
        
        if has_question_mark or has_question_words:
            if intent in ["information_request", "help_request", "comparison"]:
                score += 0.15
        
        # Check for command indicators
        command_words = {"create", "make", "do", "execute", "run", "perform", "please"}
        if any(word in processed_text for word in command_words):
            if intent in ["task_execution", "configuration", "document_processing"]:
                score += 0.15
        
        return min(1.0, score)
    
    def _detect_sub_intents(
        self, 
        processed_text: str, 
        original_text: str,
        intent_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect sub-intents or secondary intents"""
        sub_intents = []
        
        # Find intents with significant but not highest scores
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        for intent, score in sorted_intents[1:]:  # Skip the primary intent
            if score >= 0.3:  # Significant secondary intent
                sub_intents.append({
                    "intent": intent,
                    "confidence": score,
                    "relationship": self._determine_intent_relationship(sorted_intents[0][0], intent)
                })
        
        return sub_intents
    
    def _determine_intent_relationship(self, primary_intent: str, secondary_intent: str) -> str:
        """Determine relationship between primary and secondary intents"""
        
        # Define intent relationships
        relationships = {
            ("information_request", "comparison"): "specification",
            ("help_request", "problem_solving"): "escalation",
            ("task_execution", "configuration"): "prerequisite",
            ("document_processing", "data_analysis"): "continuation",
            ("search", "information_request"): "refinement",
            ("learning", "help_request"): "support",
            ("planning", "task_execution"): "implementation"
        }
        
        # Check for known relationships
        key = (primary_intent, secondary_intent)
        if key in relationships:
            return relationships[key]
        
        # Default relationships based on action types
        primary_config = self.intent_patterns.get(primary_intent, {})
        secondary_config = self.intent_patterns.get(secondary_intent, {})
        
        primary_action = primary_config.get("action_type", "")
        secondary_action = secondary_config.get("action_type", "")
        
        if primary_action == "query" and secondary_action == "command":
            return "follow_up"
        elif primary_action == "request" and secondary_action == "query":
            return "clarification"
        else:
            return "related"
    
    def _apply_context_adjustments(
        self, 
        intent: str, 
        score: float, 
        context: Dict[str, Any],
        intent_config: Dict[str, Any]
    ) -> Tuple[float, str, str]:
        """Apply context-based adjustments to intent classification"""
        
        adjusted_score = score
        urgency = intent_config["urgency"]
        complexity = intent_config["complexity"]
        
        # Adjust based on conversation history
        if "conversation_history" in context:
            history = context["conversation_history"]
            if len(history) > 0:
                # Check for intent patterns in conversation
                recent_intents = [msg.get("intent", "") for msg in history[-3:]]
                if intent in recent_intents:
                    adjusted_score += 0.1  # Boost for consistent intent
        
        # Adjust based on user profile
        if "user_profile" in context:
            profile = context["user_profile"]
            user_expertise = profile.get("expertise_level", "beginner")
            
            if user_expertise == "expert" and intent in ["help_request", "learning"]:
                adjusted_score -= 0.1  # Experts less likely to need basic help
                complexity = "simple" if complexity == "moderate" else complexity
            elif user_expertise == "beginner" and intent in ["configuration", "problem_solving"]:
                adjusted_score += 0.1  # Beginners more likely to need configuration help
                complexity = "complex" if complexity == "moderate" else complexity
        
        # Adjust based on domain context
        if "domain" in context:
            domain = context["domain"]
            domain_adjustments = {
                ("support", "help_request"): 0.2,
                ("support", "problem_solving"): 0.2,
                ("development", "task_execution"): 0.15,
                ("development", "configuration"): 0.15,
                ("analysis", "data_analysis"): 0.2,
                ("analysis", "information_request"): 0.1
            }
            
            key = (domain, intent)
            if key in domain_adjustments:
                adjusted_score += domain_adjustments[key]
        
        # Adjust urgency based on context
        if "urgency_indicators" in context:
            context_urgency = context["urgency_indicators"]
            if context_urgency > urgency:
                urgency = context_urgency
        
        # Time-based urgency adjustment
        if "timestamp" in context:
            # Could add business hours logic, deadline proximity, etc.
            pass
        
        # Ensure score stays in bounds
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        return adjusted_score, urgency, complexity
    
    async def detect_multi_intents(self, text: str) -> List[Dict[str, Any]]:
        """Detect multiple intents in a single text"""
        
        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            multi_intents = []
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 3:  # Skip very short sentences
                    intent_result = await self.classify_intent(sentence)
                    
                    if intent_result["confidence"] > 0.4:  # Only include confident classifications
                        multi_intents.append({
                            "sentence_index": i,
                            "sentence": sentence,
                            "intent": intent_result["intent"],
                            "confidence": intent_result["confidence"],
                            "action_type": intent_result["action_type"]
                        })
            
            return multi_intents
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Multi-intent detection failed: {str(e)}",
                EventType.NLP,
                {"error": str(e)},
                "intent_classifier"
            )
            return []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def generate_summary(self, text: str, language: str = "en") -> Optional[str]:
        """Generate a summary of the text (basic implementation)"""
        try:
            # Basic extractive summarization
            sentences = self._split_into_sentences(text)
            
            if len(sentences) <= 2:
                return text  # Already short enough
            
            # Score sentences based on important words
            sentence_scores = {}
            important_words = {"important", "key", "main", "primary", "essential", "critical", "significant"}
            
            for i, sentence in enumerate(sentences):
                score = 0
                words = sentence.lower().split()
                
                # Score based on position (first and last sentences often important)
                if i == 0 or i == len(sentences) - 1:
                    score += 0.5
                
                # Score based on important words
                for word in words:
                    if word in important_words:
                        score += 1
                
                # Score based on length (not too short, not too long)
                word_count = len(words)
                if 10 <= word_count <= 30:
                    score += 0.3
                
                sentence_scores[i] = score
            
            # Select top sentences (up to 1/3 of original)
            num_sentences = max(1, len(sentences) // 3)
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            
            # Sort by original order
            selected_indices = sorted([idx for idx, _ in top_sentences])
            summary_sentences = [sentences[i] for i in selected_indices]
            
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Summary generation failed: {str(e)}",
                EventType.NLP,
                {"error": str(e)},
                "intent_classifier"
            )
            return None
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get intent classification statistics"""
        return {
            **self.classification_stats,
            "supported_intents": list(self.intent_patterns.keys()),
            "confidence_thresholds": self.confidence_thresholds,
            "initialized": self._initialized
        }
    
    def get_intent_info(self, intent: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific intent"""
        if intent in self.intent_patterns:
            config = self.intent_patterns[intent].copy()
            # Convert set to list for JSON serialization
            if "keywords" in config:
                config["keywords"] = list(config["keywords"])
            return config
        return None
    
    async def cleanup(self):
        """Clean up intent classifier resources"""
        try:
            self.compiled_patterns.clear()
            self.intent_patterns.clear()
            self.context_keywords.clear()
            
            self._initialized = False
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Intent Classifier cleanup completed",
                EventType.SYSTEM,
                self.classification_stats,
                "intent_classifier"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Intent Classifier cleanup failed: {str(e)}",
                EventType.SYSTEM,
                {"error": str(e)},
                "intent_classifier"
            )