"""
AI Explanation Generator for Human-AI Collaboration
Generates context-aware, adaptive explanations to improve transparency and trust.
"""

import asyncio
import json
import logging
import numpy as np
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations that can be generated"""
    CAUSAL = "causal"           # Why something happened
    PROCEDURAL = "procedural"   # How something was done
    CONTEXTUAL = "contextual"   # Situational factors
    COMPARATIVE = "comparative" # Comparison with alternatives
    PREDICTIVE = "predictive"   # What might happen next
    COUNTERFACTUAL = "counterfactual"  # What if scenarios
    EXAMPLE_BASED = "example_based"    # Using examples
    ANALOGICAL = "analogical"   # Using analogies


class ExplanationStyle(Enum):
    """Different explanation styles for different users"""
    CONCISE = "concise"         # Brief, high-level
    DETAILED = "detailed"       # Comprehensive, technical
    STEP_BY_STEP = "step_by_step"  # Sequential breakdown
    VISUAL = "visual"           # Diagram-oriented descriptions
    NARRATIVE = "narrative"     # Story-like explanations
    ANALYTICAL = "analytical"   # Data-driven explanations
    INTUITIVE = "intuitive"     # Feeling-based explanations


class ExplanationLevel(Enum):
    """Explanation detail levels"""
    NOVICE = "novice"           # Beginner-friendly
    INTERMEDIATE = "intermediate"  # Some domain knowledge
    EXPERT = "expert"           # Advanced technical details
    ADAPTIVE = "adaptive"       # Adjusts based on feedback


@dataclass
class ExplanationRequest:
    """Request for an explanation"""
    request_id: str
    user_id: str
    session_id: str
    explanation_target: str  # What needs to be explained
    context: Dict[str, Any]
    user_profile: Dict[str, Any]
    preferred_type: Optional[ExplanationType]
    preferred_style: Optional[ExplanationStyle]
    preferred_level: Optional[ExplanationLevel]
    max_length: Optional[int]
    include_uncertainty: bool
    timestamp: datetime


@dataclass
class GeneratedExplanation:
    """Generated explanation with metadata"""
    explanation_id: str
    request_id: str
    content: str
    explanation_type: ExplanationType
    explanation_style: ExplanationStyle
    explanation_level: ExplanationLevel
    confidence: float
    completeness: float  # How complete the explanation is
    clarity_score: float
    supporting_evidence: List[str]
    key_concepts: List[str]
    assumptions: List[str]
    limitations: List[str]
    follow_up_questions: List[str]
    generated_at: datetime
    generation_time_ms: float


@dataclass
class ExplanationFeedback:
    """User feedback on explanations"""
    feedback_id: str
    explanation_id: str
    user_id: str
    clarity_rating: float  # 1-5 scale
    helpfulness_rating: float  # 1-5 scale
    completeness_rating: float  # 1-5 scale
    satisfaction_rating: float  # 1-5 scale
    feedback_text: Optional[str]
    improvement_suggestions: List[str]
    timestamp: datetime


class ExplanationTemplateEngine:
    """Manages explanation templates for different contexts"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.template_usage_stats = {}
        
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize explanation templates"""
        return {
            "prediction_explanation": {
                "causal": "I predicted {prediction} because {main_factors}. The key factors that led to this prediction were: {detailed_factors}.",
                "procedural": "Here's how I made this prediction: 1) I analyzed {input_data}, 2) I identified patterns like {patterns}, 3) I compared with similar cases where {similar_cases}, 4) I concluded {conclusion}.",
                "contextual": "Given the current context of {context_summary}, this prediction makes sense because {contextual_reasoning}.",
                "comparative": "Compared to other possible outcomes like {alternatives}, I chose {prediction} because {comparison_reasoning}."
            },
            "decision_explanation": {
                "causal": "I recommended {decision} because {main_reasons}. The primary drivers were {key_drivers}.",
                "procedural": "My decision process involved: {decision_steps}. Each step was important because {step_reasoning}.",
                "comparative": "I considered these options: {options}. I chose {decision} over {alternatives} because {comparison}.",
                "counterfactual": "If we had chosen {alternative}, the likely outcome would be {alternative_outcome}, but with {decision}, we expect {expected_outcome}."
            },
            "error_explanation": {
                "causal": "The error occurred because {error_cause}. The root cause was {root_cause}.",
                "procedural": "Here's what went wrong: {error_sequence}. To prevent this in the future: {prevention_steps}.",
                "contextual": "This error happened in the context of {error_context}, which affected the outcome because {context_impact}."
            },
            "recommendation_explanation": {
                "causal": "I recommend {recommendation} because {main_benefits}. This will help achieve {goals}.",
                "procedural": "To implement this recommendation: {implementation_steps}. The expected timeline is {timeline}.",
                "comparative": "This recommendation is better than {alternatives} because {advantages}.",
                "predictive": "If you follow this recommendation, you can expect {predicted_outcomes} within {timeframe}."
            }
        }
    
    async def get_template(self, explanation_target: str, explanation_type: ExplanationType,
                          context: Dict[str, Any]) -> str:
        """Get appropriate template for explanation"""
        target_category = self._categorize_explanation_target(explanation_target)
        
        if target_category in self.templates:
            category_templates = self.templates[target_category]
            if explanation_type.value in category_templates:
                template = category_templates[explanation_type.value]
                
                # Track usage
                template_key = f"{target_category}_{explanation_type.value}"
                self.template_usage_stats[template_key] = self.template_usage_stats.get(template_key, 0) + 1
                
                return template
        
        # Fallback to generic template
        return "Let me explain {target}: {explanation_content}. {additional_context}"
    
    def _categorize_explanation_target(self, target: str) -> str:
        """Categorize what needs to be explained"""
        target_lower = target.lower()
        
        if any(keyword in target_lower for keyword in ["predict", "forecast", "estimate"]):
            return "prediction_explanation"
        elif any(keyword in target_lower for keyword in ["decision", "choice", "recommend"]):
            return "decision_explanation"
        elif any(keyword in target_lower for keyword in ["error", "mistake", "wrong", "failed"]):
            return "error_explanation"
        elif any(keyword in target_lower for keyword in ["suggest", "advice", "recommend"]):
            return "recommendation_explanation"
        else:
            return "general_explanation"


class ClarityAnalyzer:
    """Analyzes and improves explanation clarity"""
    
    def __init__(self):
        self.clarity_metrics = {
            "readability": self._calculate_readability,
            "coherence": self._calculate_coherence,
            "conciseness": self._calculate_conciseness,
            "completeness": self._calculate_completeness,
            "jargon_level": self._calculate_jargon_level
        }
    
    async def analyze_clarity(self, explanation: str, user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Analyze clarity of an explanation"""
        clarity_scores = {}
        
        for metric_name, metric_func in self.clarity_metrics.items():
            try:
                score = await metric_func(explanation, user_profile)
                clarity_scores[metric_name] = score
            except Exception as e:
                logger.warning(f"Error calculating {metric_name}: {e}")
                clarity_scores[metric_name] = 0.5  # Default neutral score
        
        # Calculate overall clarity score
        clarity_scores["overall_clarity"] = np.mean(list(clarity_scores.values()))
        
        return clarity_scores
    
    async def _calculate_readability(self, explanation: str, user_profile: Dict[str, Any]) -> float:
        """Calculate readability score"""
        # Simple readability metrics
        sentences = re.split(r'[.!?]+', explanation)
        words = explanation.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Adjust for user expertise
        expertise_level = user_profile.get("expertise_level", 0.5)
        
        # Lower complexity for novice users
        if expertise_level < 0.3:
            target_sentence_length = 15
            target_word_length = 5
        elif expertise_level > 0.7:
            target_sentence_length = 25
            target_word_length = 7
        else:
            target_sentence_length = 20
            target_word_length = 6
        
        # Calculate readability based on targets
        sentence_score = max(0.0, 1.0 - abs(avg_sentence_length - target_sentence_length) / target_sentence_length)
        word_score = max(0.0, 1.0 - abs(avg_word_length - target_word_length) / target_word_length)
        
        return (sentence_score + word_score) / 2.0
    
    async def _calculate_coherence(self, explanation: str, user_profile: Dict[str, Any]) -> float:
        """Calculate coherence score"""
        sentences = re.split(r'[.!?]+', explanation)
        
        if len(sentences) < 2:
            return 1.0
        
        # Look for coherence indicators
        coherence_indicators = [
            "because", "therefore", "thus", "consequently", "as a result",
            "first", "second", "then", "next", "finally",
            "however", "although", "despite", "on the other hand",
            "for example", "specifically", "in particular"
        ]
        
        indicator_count = sum(1 for indicator in coherence_indicators 
                             if indicator in explanation.lower())
        
        # Normalize by number of sentences
        coherence_score = min(1.0, indicator_count / max(1, len(sentences) - 1))
        
        return coherence_score
    
    async def _calculate_conciseness(self, explanation: str, user_profile: Dict[str, Any]) -> float:
        """Calculate conciseness score"""
        words = explanation.split()
        word_count = len(words)
        
        # Target word count based on user preference
        user_preference = user_profile.get("explanation_preference", "moderate")
        
        if user_preference == "brief":
            target_words = 50
        elif user_preference == "detailed":
            target_words = 200
        else:
            target_words = 100
        
        # Calculate conciseness based on deviation from target
        if word_count <= target_words:
            return 1.0
        else:
            excess = word_count - target_words
            return max(0.0, 1.0 - (excess / target_words))
    
    async def _calculate_completeness(self, explanation: str, user_profile: Dict[str, Any]) -> float:
        """Calculate completeness score"""
        # Check for key explanation components
        components = {
            "what": ["is", "are", "means", "refers to"],
            "why": ["because", "due to", "reason", "causes"],
            "how": ["by", "through", "process", "method", "steps"],
            "when": ["when", "timing", "time", "schedule"],
            "where": ["where", "location", "place", "context"]
        }
        
        present_components = 0
        explanation_lower = explanation.lower()
        
        for component, keywords in components.items():
            if any(keyword in explanation_lower for keyword in keywords):
                present_components += 1
        
        # Calculate completeness based on present components
        completeness = present_components / len(components)
        
        # Adjust for user expertise (experts expect more completeness)
        expertise_level = user_profile.get("expertise_level", 0.5)
        if expertise_level > 0.7:
            completeness *= 1.2  # Boost requirement for experts
        
        return min(1.0, completeness)
    
    async def _calculate_jargon_level(self, explanation: str, user_profile: Dict[str, Any]) -> float:
        """Calculate appropriateness of jargon level"""
        # Technical terms that might be jargon
        technical_terms = [
            "algorithm", "optimize", "parameter", "coefficient", "correlation",
            "regression", "classification", "neural", "vector", "matrix",
            "probability", "distribution", "variance", "standard deviation",
            "hypothesis", "inference", "validation", "training", "model"
        ]
        
        words = explanation.lower().split()
        jargon_count = sum(1 for word in words if any(term in word for term in technical_terms))
        jargon_ratio = jargon_count / len(words) if words else 0
        
        # Determine appropriate jargon level based on user expertise
        expertise_level = user_profile.get("expertise_level", 0.5)
        
        if expertise_level < 0.3:
            # Novice: prefer low jargon
            target_jargon_ratio = 0.02
        elif expertise_level > 0.7:
            # Expert: can handle more jargon
            target_jargon_ratio = 0.15
        else:
            # Intermediate: moderate jargon
            target_jargon_ratio = 0.08
        
        # Score based on deviation from target
        deviation = abs(jargon_ratio - target_jargon_ratio)
        jargon_appropriateness = max(0.0, 1.0 - (deviation / target_jargon_ratio))
        
        return jargon_appropriateness


class ExplanationPersonalizer:
    """Personalizes explanations based on user characteristics"""
    
    def __init__(self):
        self.personalization_rules = self._initialize_personalization_rules()
    
    def _initialize_personalization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize personalization rules"""
        return {
            "cognitive_style": {
                "analytical": {
                    "preferred_type": ExplanationType.CAUSAL,
                    "include_data": True,
                    "include_reasoning_steps": True,
                    "use_logical_structure": True
                },
                "intuitive": {
                    "preferred_type": ExplanationType.ANALOGICAL,
                    "include_examples": True,
                    "use_metaphors": True,
                    "focus_on_big_picture": True
                },
                "visual": {
                    "preferred_style": ExplanationStyle.VISUAL,
                    "include_diagrams": True,
                    "use_spatial_metaphors": True
                }
            },
            "expertise_level": {
                "novice": {
                    "explanation_level": ExplanationLevel.NOVICE,
                    "include_basic_concepts": True,
                    "avoid_jargon": True,
                    "provide_background": True
                },
                "expert": {
                    "explanation_level": ExplanationLevel.EXPERT,
                    "include_technical_details": True,
                    "assume_domain_knowledge": True,
                    "focus_on_novel_aspects": True
                }
            },
            "trust_level": {
                "low": {
                    "include_uncertainty": True,
                    "provide_evidence": True,
                    "explain_limitations": True,
                    "offer_alternatives": True
                },
                "high": {
                    "be_concise": True,
                    "focus_on_key_points": True,
                    "assume_competence": True
                }
            }
        }
    
    async def personalize_explanation(self, base_explanation: str, user_profile: Dict[str, Any],
                                    context: Dict[str, Any]) -> str:
        """Personalize explanation based on user profile"""
        personalized = base_explanation
        
        # Apply cognitive style personalization
        cognitive_style = user_profile.get("cognitive_style", "mixed")
        if cognitive_style in self.personalization_rules["cognitive_style"]:
            rules = self.personalization_rules["cognitive_style"][cognitive_style]
            personalized = await self._apply_cognitive_style_rules(personalized, rules, context)
        
        # Apply expertise level personalization
        expertise_level = user_profile.get("expertise_level", 0.5)
        if expertise_level < 0.3:
            level_rules = self.personalization_rules["expertise_level"]["novice"]
        elif expertise_level > 0.7:
            level_rules = self.personalization_rules["expertise_level"]["expert"]
        else:
            level_rules = {}  # Intermediate - use defaults
        
        if level_rules:
            personalized = await self._apply_expertise_rules(personalized, level_rules, context)
        
        # Apply trust level personalization
        trust_level = user_profile.get("trust_level", 0.5)
        if trust_level < 0.4:
            trust_rules = self.personalization_rules["trust_level"]["low"]
        elif trust_level > 0.7:
            trust_rules = self.personalization_rules["trust_level"]["high"]
        else:
            trust_rules = {}  # Moderate trust - use defaults
        
        if trust_rules:
            personalized = await self._apply_trust_rules(personalized, trust_rules, context)
        
        return personalized
    
    async def _apply_cognitive_style_rules(self, explanation: str, rules: Dict[str, Any],
                                         context: Dict[str, Any]) -> str:
        """Apply cognitive style personalization rules"""
        modified = explanation
        
        if rules.get("include_data") and "data" in context:
            data_snippet = str(context["data"])[:100]  # Brief data sample
            modified += f" Based on data: {data_snippet}..."
        
        if rules.get("include_reasoning_steps"):
            modified = "Let me walk through my reasoning: " + modified
        
        if rules.get("include_examples"):
            modified += " For example, this is similar to when..."
        
        if rules.get("use_metaphors"):
            modified = self._add_metaphor(modified, context)
        
        return modified
    
    async def _apply_expertise_rules(self, explanation: str, rules: Dict[str, Any],
                                   context: Dict[str, Any]) -> str:
        """Apply expertise level personalization rules"""
        modified = explanation
        
        if rules.get("include_basic_concepts"):
            modified = "To understand this, it's helpful to know that... " + modified
        
        if rules.get("avoid_jargon"):
            modified = self._simplify_jargon(modified)
        
        if rules.get("provide_background"):
            modified = "In this context, " + modified
        
        if rules.get("include_technical_details") and "technical_details" in context:
            technical_info = context["technical_details"]
            modified += f" Technical details: {technical_info}"
        
        if rules.get("focus_on_novel_aspects"):
            modified = "What's particularly interesting here is that " + modified
        
        return modified
    
    async def _apply_trust_rules(self, explanation: str, rules: Dict[str, Any],
                               context: Dict[str, Any]) -> str:
        """Apply trust level personalization rules"""
        modified = explanation
        
        if rules.get("include_uncertainty") and "confidence" in context:
            confidence = context["confidence"]
            modified += f" I'm {confidence*100:.0f}% confident in this explanation."
        
        if rules.get("provide_evidence"):
            modified += " This is supported by evidence such as..."
        
        if rules.get("explain_limitations"):
            modified += " However, please note that this explanation has limitations..."
        
        if rules.get("offer_alternatives"):
            modified += " Alternative perspectives might include..."
        
        if rules.get("be_concise"):
            # Shorten explanation by removing redundant phrases
            modified = self._make_concise(modified)
        
        return modified
    
    def _add_metaphor(self, explanation: str, context: Dict[str, Any]) -> str:
        """Add appropriate metaphor to explanation"""
        # Simple metaphor addition - in practice would be more sophisticated
        if "decision" in explanation.lower():
            return explanation + " Think of this like choosing the best route on a map."
        elif "prediction" in explanation.lower():
            return explanation + " This is like forecasting the weather based on current conditions."
        else:
            return explanation
    
    def _simplify_jargon(self, explanation: str) -> str:
        """Replace jargon with simpler terms"""
        replacements = {
            "algorithm": "method",
            "optimize": "improve",
            "parameter": "setting",
            "correlation": "relationship",
            "probability": "chance",
            "variance": "variation",
            "inference": "conclusion"
        }
        
        for jargon, simple in replacements.items():
            explanation = explanation.replace(jargon, simple)
        
        return explanation
    
    def _make_concise(self, explanation: str) -> str:
        """Make explanation more concise"""
        # Remove redundant phrases
        redundant_phrases = [
            "it is important to note that",
            "as you can see",
            "it should be mentioned that",
            "please be aware that"
        ]
        
        for phrase in redundant_phrases:
            explanation = explanation.replace(phrase, "")
        
        return explanation.strip()


class ExplanationGenerator:
    """Main explanation generator engine"""
    
    def __init__(self):
        self.template_engine = ExplanationTemplateEngine()
        self.clarity_analyzer = ClarityAnalyzer()
        self.personalizer = ExplanationPersonalizer()
        self.explanation_history: List[GeneratedExplanation] = []
        self.feedback_history: List[ExplanationFeedback] = []
        
        # Explanation quality thresholds
        self.quality_thresholds = {
            "clarity_minimum": 0.6,
            "completeness_minimum": 0.5,
            "confidence_minimum": 0.4
        }
    
    async def generate_explanation(self, request: ExplanationRequest) -> GeneratedExplanation:
        """Generate a comprehensive explanation"""
        start_time = time.time()
        
        # Determine explanation characteristics
        explanation_type = await self._determine_explanation_type(request)
        explanation_style = await self._determine_explanation_style(request)
        explanation_level = await self._determine_explanation_level(request)
        
        # Generate base explanation
        base_content = await self._generate_base_explanation(
            request, explanation_type, explanation_style, explanation_level
        )
        
        # Personalize explanation
        personalized_content = await self.personalizer.personalize_explanation(
            base_content, request.user_profile, request.context
        )
        
        # Analyze clarity
        clarity_scores = await self.clarity_analyzer.analyze_clarity(
            personalized_content, request.user_profile
        )
        
        # Improve if necessary
        if clarity_scores["overall_clarity"] < self.quality_thresholds["clarity_minimum"]:
            personalized_content = await self._improve_clarity(
                personalized_content, clarity_scores, request.user_profile
            )
            # Re-analyze clarity
            clarity_scores = await self.clarity_analyzer.analyze_clarity(
                personalized_content, request.user_profile
            )
        
        # Generate supporting information
        supporting_evidence = await self._generate_supporting_evidence(request)
        key_concepts = await self._extract_key_concepts(personalized_content, request)
        assumptions = await self._identify_assumptions(request)
        limitations = await self._identify_limitations(request)
        follow_up_questions = await self._generate_follow_up_questions(request)
        
        # Calculate confidence and completeness
        confidence = await self._calculate_explanation_confidence(request, clarity_scores)
        completeness = clarity_scores.get("completeness", 0.5)
        
        generation_time = (time.time() - start_time) * 1000
        
        # Create explanation object
        explanation = GeneratedExplanation(
            explanation_id=str(uuid4()),
            request_id=request.request_id,
            content=personalized_content,
            explanation_type=explanation_type,
            explanation_style=explanation_style,
            explanation_level=explanation_level,
            confidence=confidence,
            completeness=completeness,
            clarity_score=clarity_scores["overall_clarity"],
            supporting_evidence=supporting_evidence,
            key_concepts=key_concepts,
            assumptions=assumptions,
            limitations=limitations,
            follow_up_questions=follow_up_questions,
            generated_at=datetime.utcnow(),
            generation_time_ms=generation_time
        )
        
        # Store in history
        self.explanation_history.append(explanation)
        
        logger.info(f"Generated explanation {explanation.explanation_id} in {generation_time:.1f}ms")
        
        return explanation
    
    async def _determine_explanation_type(self, request: ExplanationRequest) -> ExplanationType:
        """Determine the most appropriate explanation type"""
        if request.preferred_type:
            return request.preferred_type
        
        # Analyze target to determine type
        target = request.explanation_target.lower()
        
        if any(keyword in target for keyword in ["why", "reason", "cause"]):
            return ExplanationType.CAUSAL
        elif any(keyword in target for keyword in ["how", "process", "steps"]):
            return ExplanationType.PROCEDURAL
        elif any(keyword in target for keyword in ["compare", "versus", "alternative"]):
            return ExplanationType.COMPARATIVE
        elif any(keyword in target for keyword in ["predict", "future", "expect"]):
            return ExplanationType.PREDICTIVE
        elif any(keyword in target for keyword in ["example", "instance", "case"]):
            return ExplanationType.EXAMPLE_BASED
        else:
            # Default based on user cognitive style
            cognitive_style = request.user_profile.get("cognitive_style", "mixed")
            if cognitive_style == "analytical":
                return ExplanationType.CAUSAL
            elif cognitive_style == "intuitive":
                return ExplanationType.ANALOGICAL
            else:
                return ExplanationType.CONTEXTUAL
    
    async def _determine_explanation_style(self, request: ExplanationRequest) -> ExplanationStyle:
        """Determine the most appropriate explanation style"""
        if request.preferred_style:
            return request.preferred_style
        
        # Determine based on user profile
        cognitive_style = request.user_profile.get("cognitive_style", "mixed")
        expertise_level = request.user_profile.get("expertise_level", 0.5)
        
        if cognitive_style == "analytical" and expertise_level > 0.6:
            return ExplanationStyle.ANALYTICAL
        elif cognitive_style == "visual":
            return ExplanationStyle.VISUAL
        elif expertise_level < 0.3:
            return ExplanationStyle.STEP_BY_STEP
        elif expertise_level > 0.7:
            return ExplanationStyle.DETAILED
        else:
            return ExplanationStyle.CONCISE
    
    async def _determine_explanation_level(self, request: ExplanationRequest) -> ExplanationLevel:
        """Determine the most appropriate explanation level"""
        if request.preferred_level:
            return request.preferred_level
        
        expertise_level = request.user_profile.get("expertise_level", 0.5)
        
        if expertise_level < 0.3:
            return ExplanationLevel.NOVICE
        elif expertise_level > 0.7:
            return ExplanationLevel.EXPERT
        else:
            return ExplanationLevel.INTERMEDIATE
    
    async def _generate_base_explanation(self, request: ExplanationRequest,
                                       explanation_type: ExplanationType,
                                       explanation_style: ExplanationStyle,
                                       explanation_level: ExplanationLevel) -> str:
        """Generate base explanation content"""
        # Get template
        template = await self.template_engine.get_template(
            request.explanation_target, explanation_type, request.context
        )
        
        # Fill template with context data
        explanation_data = await self._prepare_explanation_data(request)
        
        try:
            base_explanation = template.format(**explanation_data)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            # Fallback to simple explanation
            base_explanation = f"Regarding {request.explanation_target}: {explanation_data.get('explanation_content', 'I can provide more details if needed.')}"
        
        return base_explanation
    
    async def _prepare_explanation_data(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Prepare data for filling explanation templates"""
        context = request.context
        
        # Extract key information from context
        explanation_data = {
            "target": request.explanation_target,
            "prediction": context.get("prediction", "the result"),
            "decision": context.get("decision", "this choice"),
            "main_factors": context.get("main_factors", "several key factors"),
            "detailed_factors": context.get("detailed_factors", "various considerations"),
            "context_summary": context.get("context_summary", "the current situation"),
            "contextual_reasoning": context.get("contextual_reasoning", "the circumstances support this"),
            "alternatives": context.get("alternatives", "other options"),
            "comparison_reasoning": context.get("comparison_reasoning", "comparative analysis"),
            "main_reasons": context.get("main_reasons", "important considerations"),
            "key_drivers": context.get("key_drivers", "primary factors"),
            "decision_steps": context.get("decision_steps", "systematic analysis"),
            "step_reasoning": context.get("step_reasoning", "logical progression"),
            "error_cause": context.get("error_cause", "underlying issues"),
            "root_cause": context.get("root_cause", "fundamental problem"),
            "main_benefits": context.get("main_benefits", "significant advantages"),
            "goals": context.get("goals", "desired outcomes"),
            "explanation_content": context.get("explanation_content", "detailed information")
        }
        
        return explanation_data
    
    async def _improve_clarity(self, explanation: str, clarity_scores: Dict[str, float],
                             user_profile: Dict[str, Any]) -> str:
        """Improve explanation clarity based on analysis"""
        improved = explanation
        
        # Improve readability if low
        if clarity_scores.get("readability", 0.5) < 0.5:
            improved = await self._improve_readability(improved, user_profile)
        
        # Improve coherence if low
        if clarity_scores.get("coherence", 0.5) < 0.5:
            improved = await self._improve_coherence(improved)
        
        # Improve completeness if low
        if clarity_scores.get("completeness", 0.5) < 0.5:
            improved = await self._improve_completeness(improved)
        
        return improved
    
    async def _improve_readability(self, explanation: str, user_profile: Dict[str, Any]) -> str:
        """Improve readability of explanation"""
        # Break long sentences
        sentences = re.split(r'[.!?]+', explanation)
        improved_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 20:  # Long sentence
                # Try to break at logical points
                if " because " in sentence:
                    parts = sentence.split(" because ")
                    improved_sentences.append(parts[0] + ".")
                    improved_sentences.append("This is because " + " because ".join(parts[1:]))
                elif " and " in sentence:
                    parts = sentence.split(" and ", 1)
                    improved_sentences.append(parts[0] + ".")
                    improved_sentences.append("Additionally, " + parts[1])
                else:
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)
        
        return ". ".join(improved_sentences).replace("..", ".")
    
    async def _improve_coherence(self, explanation: str) -> str:
        """Improve coherence by adding connecting words"""
        sentences = re.split(r'[.!?]+', explanation)
        
        if len(sentences) < 2:
            return explanation
        
        improved_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i].strip()
            if sentence:
                # Add appropriate connector
                if i == 1:
                    connector = "Furthermore, "
                elif "result" in sentence.lower() or "outcome" in sentence.lower():
                    connector = "As a result, "
                elif "example" in sentence.lower():
                    connector = "For example, "
                else:
                    connector = "Additionally, "
                
                improved_sentences.append(connector + sentence)
        
        return ". ".join(improved_sentences)
    
    async def _improve_completeness(self, explanation: str) -> str:
        """Improve completeness by adding missing elements"""
        # Add what/why/how if missing
        if "why" not in explanation.lower() and "because" not in explanation.lower():
            explanation += " This is important because it helps achieve the desired outcome."
        
        if "how" not in explanation.lower() and "process" not in explanation.lower():
            explanation += " The process involves systematic analysis and decision-making."
        
        return explanation
    
    async def _generate_supporting_evidence(self, request: ExplanationRequest) -> List[str]:
        """Generate supporting evidence for the explanation"""
        evidence = []
        context = request.context
        
        # Add data-based evidence
        if "data" in context:
            evidence.append(f"Based on analysis of {context['data']}")
        
        # Add performance evidence
        if "accuracy" in context:
            evidence.append(f"Historical accuracy: {context['accuracy']*100:.1f}%")
        
        # Add similarity evidence
        if "similar_cases" in context:
            evidence.append(f"Similar patterns observed in {context['similar_cases']} cases")
        
        # Add expert knowledge
        if "domain_knowledge" in context:
            evidence.append(f"Consistent with established {context['domain_knowledge']} principles")
        
        return evidence[:3]  # Limit to top 3 pieces of evidence
    
    async def _extract_key_concepts(self, explanation: str, request: ExplanationRequest) -> List[str]:
        """Extract key concepts from the explanation"""
        # Simple keyword extraction - in practice would use more sophisticated NLP
        words = explanation.lower().split()
        
        # Important concept indicators
        concept_indicators = [
            "algorithm", "model", "pattern", "prediction", "decision", "analysis",
            "factor", "variable", "relationship", "correlation", "trend", "outcome"
        ]
        
        key_concepts = [word for word in concept_indicators if word in words]
        
        # Add domain-specific concepts from context
        if "domain" in request.context:
            domain = request.context["domain"]
            if domain == "finance":
                key_concepts.extend(["risk", "return", "volatility", "portfolio"])
            elif domain == "healthcare":
                key_concepts.extend(["diagnosis", "treatment", "symptoms", "prognosis"])
            elif domain == "marketing":
                key_concepts.extend(["conversion", "engagement", "segmentation", "attribution"])
        
        return list(set(key_concepts))[:5]  # Return top 5 unique concepts
    
    async def _identify_assumptions(self, request: ExplanationRequest) -> List[str]:
        """Identify key assumptions in the explanation"""
        assumptions = []
        context = request.context
        
        # Common AI/ML assumptions
        assumptions.append("Past patterns are indicative of future behavior")
        
        if "data" in context:
            assumptions.append("The data is representative and unbiased")
        
        if "model" in context:
            assumptions.append("The model is appropriate for this use case")
        
        # Context-specific assumptions
        if "user_behavior" in context:
            assumptions.append("User behavior remains consistent over time")
        
        return assumptions[:3]  # Limit to top 3 assumptions
    
    async def _identify_limitations(self, request: ExplanationRequest) -> List[str]:
        """Identify limitations of the explanation"""
        limitations = []
        context = request.context
        
        # General limitations
        if "confidence" in context and context["confidence"] < 0.8:
            limitations.append("Prediction confidence is moderate")
        
        if "data_size" in context and context["data_size"] < 1000:
            limitations.append("Limited data available for analysis")
        
        # Domain-specific limitations
        if "temporal" in context and context["temporal"]:
            limitations.append("Results may change over time")
        
        if "complexity" in context and context["complexity"] == "high":
            limitations.append("High complexity may introduce uncertainty")
        
        limitations.append("Explanation is based on current understanding and may evolve")
        
        return limitations[:3]  # Limit to top 3 limitations
    
    async def _generate_follow_up_questions(self, request: ExplanationRequest) -> List[str]:
        """Generate relevant follow-up questions"""
        questions = []
        
        # General follow-up questions
        questions.append("Would you like me to explain any part in more detail?")
        questions.append("Are there specific aspects you'd like to explore further?")
        
        # Context-specific questions
        if "alternatives" in request.context:
            questions.append("Would you like to know about alternative approaches?")
        
        if "prediction" in request.context:
            questions.append("What factors would you like to see change in future predictions?")
        
        if "decision" in request.context:
            questions.append("Would you like to understand the trade-offs in this decision?")
        
        return questions[:3]  # Limit to top 3 questions
    
    async def _calculate_explanation_confidence(self, request: ExplanationRequest,
                                              clarity_scores: Dict[str, float]) -> float:
        """Calculate confidence in the explanation quality"""
        # Base confidence from context
        base_confidence = request.context.get("confidence", 0.7)
        
        # Adjust based on clarity
        clarity_factor = clarity_scores.get("overall_clarity", 0.5)
        
        # Adjust based on completeness of context
        context_completeness = len(request.context) / 10.0  # Normalize by expected fields
        context_factor = min(1.0, context_completeness)
        
        # Combine factors
        combined_confidence = (base_confidence * 0.5 + 
                             clarity_factor * 0.3 + 
                             context_factor * 0.2)
        
        return max(0.0, min(1.0, combined_confidence))
    
    async def record_feedback(self, feedback: ExplanationFeedback) -> Dict[str, Any]:
        """Record user feedback on explanation"""
        self.feedback_history.append(feedback)
        
        # Update explanation quality models based on feedback
        explanation = next((e for e in self.explanation_history 
                          if e.explanation_id == feedback.explanation_id), None)
        
        if explanation:
            # Calculate feedback impact
            feedback_impact = await self._analyze_feedback_impact(feedback, explanation)
            
            # Update quality thresholds if needed
            await self._update_quality_thresholds(feedback, explanation)
            
            logger.info(f"Recorded feedback for explanation {feedback.explanation_id}")
            
            return {
                "feedback_recorded": True,
                "feedback_impact": feedback_impact,
                "explanation_quality_updated": True
            }
        else:
            return {"error": "Explanation not found"}
    
    async def _analyze_feedback_impact(self, feedback: ExplanationFeedback,
                                     explanation: GeneratedExplanation) -> Dict[str, Any]:
        """Analyze the impact of user feedback"""
        return {
            "clarity_gap": feedback.clarity_rating / 5.0 - explanation.clarity_score,
            "helpfulness_score": feedback.helpfulness_rating / 5.0,
            "user_satisfaction": feedback.satisfaction_rating / 5.0,
            "improvement_areas": feedback.improvement_suggestions
        }
    
    async def _update_quality_thresholds(self, feedback: ExplanationFeedback,
                                       explanation: GeneratedExplanation):
        """Update quality thresholds based on feedback"""
        # If feedback is consistently low, lower thresholds
        avg_rating = (feedback.clarity_rating + feedback.helpfulness_rating + 
                     feedback.satisfaction_rating) / 3.0
        
        if avg_rating < 2.5:  # Low satisfaction
            # Slightly increase quality thresholds to improve future explanations
            self.quality_thresholds["clarity_minimum"] = min(0.8, 
                self.quality_thresholds["clarity_minimum"] + 0.05)
    
    async def get_explanation_stats(self) -> Dict[str, Any]:
        """Get statistics about generated explanations"""
        if not self.explanation_history:
            return {"message": "No explanations generated yet"}
        
        # Calculate statistics
        total_explanations = len(self.explanation_history)
        avg_clarity = np.mean([e.clarity_score for e in self.explanation_history])
        avg_confidence = np.mean([e.confidence for e in self.explanation_history])
        avg_completeness = np.mean([e.completeness for e in self.explanation_history])
        avg_generation_time = np.mean([e.generation_time_ms for e in self.explanation_history])
        
        # Type distribution
        type_distribution = {}
        for explanation in self.explanation_history:
            exp_type = explanation.explanation_type.value
            type_distribution[exp_type] = type_distribution.get(exp_type, 0) + 1
        
        # Feedback statistics
        feedback_stats = {}
        if self.feedback_history:
            avg_clarity_rating = np.mean([f.clarity_rating for f in self.feedback_history])
            avg_helpfulness_rating = np.mean([f.helpfulness_rating for f in self.feedback_history])
            avg_satisfaction_rating = np.mean([f.satisfaction_rating for f in self.feedback_history])
            
            feedback_stats = {
                "total_feedback": len(self.feedback_history),
                "average_clarity_rating": avg_clarity_rating,
                "average_helpfulness_rating": avg_helpfulness_rating,
                "average_satisfaction_rating": avg_satisfaction_rating
            }
        
        return {
            "total_explanations": total_explanations,
            "average_clarity_score": avg_clarity,
            "average_confidence": avg_confidence,
            "average_completeness": avg_completeness,
            "average_generation_time_ms": avg_generation_time,
            "explanation_type_distribution": type_distribution,
            "feedback_statistics": feedback_stats,
            "quality_thresholds": self.quality_thresholds
        }
