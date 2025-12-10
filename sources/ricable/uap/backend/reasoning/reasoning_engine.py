"""
Reasoning Engine for Neuro-Symbolic AI System
Provides advanced reasoning capabilities including logical inference,
pattern matching, and explanation generation.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import uuid4
from enum import Enum
import networkx as nx
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning supported"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    COUNTERFACTUAL = "counterfactual"


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning chain"""
    step_id: str
    reasoning_type: ReasoningType
    premise: List[str]
    conclusion: str
    rule_applied: Optional[str]
    confidence: float
    explanation: str
    evidence: List[str]
    timestamp: datetime


@dataclass
class ReasoningChain:
    """Represents a complete reasoning chain"""
    chain_id: str
    query: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    reasoning_path: List[str]
    alternative_paths: List[List[str]]
    created_at: datetime


@dataclass
class Pattern:
    """Represents a reasoning pattern"""
    pattern_id: str
    pattern_type: str
    template: str
    variables: List[str]
    constraints: Dict[str, Any]
    confidence: float
    usage_count: int
    examples: List[str]


@dataclass
class Analogy:
    """Represents an analogical reasoning structure"""
    analogy_id: str
    source_domain: str
    target_domain: str
    mapping: Dict[str, str]
    similarity_score: float
    confidence: float
    explanation: str


class LogicalInferenceEngine:
    """Handles various types of logical inference"""
    
    def __init__(self):
        self.inference_rules = {}
        self.fact_database = set()
        self.inference_history = []
        
    def add_inference_rule(self, rule_id: str, premises: List[str], 
                          conclusion: str, confidence: float = 1.0):
        """Add an inference rule"""
        self.inference_rules[rule_id] = {
            "premises": premises,
            "conclusion": conclusion,
            "confidence": confidence,
            "type": "general"
        }
    
    async def deductive_reasoning(self, premises: List[str], 
                                rules: List[str] = None) -> List[ReasoningStep]:
        """Perform deductive reasoning"""
        steps = []
        
        # Use provided rules or all available rules
        applicable_rules = rules or list(self.inference_rules.keys())
        
        for rule_id in applicable_rules:
            rule = self.inference_rules[rule_id]
            
            # Check if all premises of the rule are satisfied
            if all(premise in premises for premise in rule["premises"]):
                step = ReasoningStep(
                    step_id=str(uuid4()),
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    premise=rule["premises"],
                    conclusion=rule["conclusion"],
                    rule_applied=rule_id,
                    confidence=rule["confidence"],
                    explanation=f"Applied rule {rule_id}: {' ∧ '.join(rule['premises'])} → {rule['conclusion']}",
                    evidence=rule["premises"],
                    timestamp=datetime.utcnow()
                )
                steps.append(step)
                
                # Add the conclusion as a new fact for further reasoning
                premises.append(rule["conclusion"])
        
        return steps
    
    async def inductive_reasoning(self, observations: List[Dict[str, Any]]) -> List[ReasoningStep]:
        """Perform inductive reasoning to generate general rules"""
        steps = []
        
        # Group observations by pattern
        patterns = self._identify_patterns(observations)
        
        for pattern in patterns:
            if pattern["support"] >= 3:  # Minimum support for induction
                step = ReasoningStep(
                    step_id=str(uuid4()),
                    reasoning_type=ReasoningType.INDUCTIVE,
                    premise=[f"observation_{i}" for i in range(pattern["support"])],
                    conclusion=pattern["rule"],
                    rule_applied="inductive_generalization",
                    confidence=min(0.9, pattern["support"] / 10.0),
                    explanation=f"Induced general rule from {pattern['support']} observations",
                    evidence=[str(obs) for obs in pattern["observations"]],
                    timestamp=datetime.utcnow()
                )
                steps.append(step)
        
        return steps
    
    async def abductive_reasoning(self, observation: str, 
                                possible_explanations: List[str]) -> List[ReasoningStep]:
        """Perform abductive reasoning to find best explanation"""
        steps = []
        
        # Evaluate each possible explanation
        explanations_with_scores = []
        
        for explanation in possible_explanations:
            # Score based on simplicity, prior probability, and explanatory power
            score = self._score_explanation(explanation, observation)
            explanations_with_scores.append((explanation, score))
        
        # Sort by score and create reasoning steps
        explanations_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (explanation, score) in enumerate(explanations_with_scores[:3]):
            step = ReasoningStep(
                step_id=str(uuid4()),
                reasoning_type=ReasoningType.ABDUCTIVE,
                premise=[observation],
                conclusion=f"Best explanation: {explanation}",
                rule_applied="abductive_inference",
                confidence=score,
                explanation=f"Explanation ranked {i+1} with score {score:.3f}",
                evidence=[observation],
                timestamp=datetime.utcnow()
            )
            steps.append(step)
        
        return steps
    
    def _identify_patterns(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify patterns in observations for inductive reasoning"""
        patterns = []
        
        # Simple pattern detection based on common attributes
        attribute_patterns = defaultdict(list)
        
        for obs in observations:
            for key, value in obs.items():
                attribute_patterns[key].append(value)
        
        # Look for frequent patterns
        for attribute, values in attribute_patterns.items():
            value_counts = defaultdict(int)
            for value in values:
                value_counts[value] += 1
            
            # Find majority patterns
            for value, count in value_counts.items():
                if count >= len(observations) * 0.6:  # 60% threshold
                    patterns.append({
                        "rule": f"Most instances have {attribute} = {value}",
                        "support": count,
                        "observations": [obs for obs in observations if obs.get(attribute) == value]
                    })
        
        return patterns
    
    def _score_explanation(self, explanation: str, observation: str) -> float:
        """Score an explanation for abductive reasoning"""
        # Simple scoring based on explanation characteristics
        score = 0.5  # Base score
        
        # Prefer simpler explanations (Occam's razor)
        simplicity_score = 1.0 / (len(explanation.split()) / 10.0 + 1)
        score += 0.3 * simplicity_score
        
        # Prefer explanations that directly relate to observation
        if observation.lower() in explanation.lower():
            score += 0.2
        
        return min(score, 1.0)


class PatternMatcher:
    """Matches and applies reasoning patterns"""
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_usage = defaultdict(int)
        
    def add_pattern(self, pattern: Pattern):
        """Add a reasoning pattern"""
        self.patterns[pattern.pattern_id] = pattern
    
    async def match_patterns(self, input_text: str) -> List[Tuple[Pattern, Dict[str, str]]]:
        """Match input against known patterns"""
        matches = []
        
        for pattern in self.patterns.values():
            variables = self._extract_variables(input_text, pattern.template)
            if variables and self._check_constraints(variables, pattern.constraints):
                matches.append((pattern, variables))
                self.pattern_usage[pattern.pattern_id] += 1
        
        # Sort by confidence and usage
        matches.sort(key=lambda x: (x[0].confidence, self.pattern_usage[x[0].pattern_id]), 
                    reverse=True)
        
        return matches
    
    def _extract_variables(self, text: str, template: str) -> Optional[Dict[str, str]]:
        """Extract variables from text using pattern template"""
        # Convert template to regex pattern
        regex_pattern = template
        variables = re.findall(r'\{(\w+)\}', template)
        
        for var in variables:
            regex_pattern = regex_pattern.replace(f'{{{var}}}', r'([^,\s]+)')
        
        match = re.search(regex_pattern, text, re.IGNORECASE)
        if match:
            return dict(zip(variables, match.groups()))
        
        return None
    
    def _check_constraints(self, variables: Dict[str, str], 
                          constraints: Dict[str, Any]) -> bool:
        """Check if extracted variables satisfy constraints"""
        for var, constraint in constraints.items():
            if var in variables:
                value = variables[var]
                
                if constraint["type"] == "numeric":
                    try:
                        num_value = float(value)
                        if "min" in constraint and num_value < constraint["min"]:
                            return False
                        if "max" in constraint and num_value > constraint["max"]:
                            return False
                    except ValueError:
                        return False
                
                elif constraint["type"] == "categorical":
                    if value.lower() not in [c.lower() for c in constraint["values"]]:
                        return False
        
        return True


class AnalogicalReasoner:
    """Handles analogical reasoning"""
    
    def __init__(self):
        self.analogies: Dict[str, Analogy] = {}
        self.domain_knowledge = defaultdict(dict)
        
    def add_analogy(self, analogy: Analogy):
        """Add an analogy to the knowledge base"""
        self.analogies[analogy.analogy_id] = analogy
    
    async def find_analogies(self, source_domain: str, 
                           target_domain: str) -> List[Analogy]:
        """Find analogies between domains"""
        relevant_analogies = []
        
        for analogy in self.analogies.values():
            if (analogy.source_domain == source_domain and 
                analogy.target_domain == target_domain):
                relevant_analogies.append(analogy)
            elif (analogy.source_domain == target_domain and 
                  analogy.target_domain == source_domain):
                # Reverse analogy
                reversed_analogy = self._reverse_analogy(analogy)
                relevant_analogies.append(reversed_analogy)
        
        return sorted(relevant_analogies, key=lambda x: x.similarity_score, reverse=True)
    
    async def create_analogy(self, source_description: str, 
                           target_description: str) -> Optional[Analogy]:
        """Create a new analogy between two descriptions"""
        # Extract entities and relationships from descriptions
        source_entities = self._extract_entities(source_description)
        target_entities = self._extract_entities(target_description)
        
        # Find potential mappings
        mapping = self._find_structural_mapping(source_entities, target_entities)
        
        if mapping:
            similarity = self._calculate_similarity(source_entities, target_entities, mapping)
            
            analogy = Analogy(
                analogy_id=str(uuid4()),
                source_domain=self._infer_domain(source_description),
                target_domain=self._infer_domain(target_description),
                mapping=mapping,
                similarity_score=similarity,
                confidence=min(similarity + 0.1, 1.0),
                explanation=f"Structural mapping between {len(mapping)} entities"
            )
            
            return analogy
        
        return None
    
    def _reverse_analogy(self, analogy: Analogy) -> Analogy:
        """Reverse an analogy"""
        reversed_mapping = {v: k for k, v in analogy.mapping.items()}
        
        return Analogy(
            analogy_id=f"rev_{analogy.analogy_id}",
            source_domain=analogy.target_domain,
            target_domain=analogy.source_domain,
            mapping=reversed_mapping,
            similarity_score=analogy.similarity_score,
            confidence=analogy.confidence * 0.9,  # Slightly lower confidence
            explanation=f"Reversed: {analogy.explanation}"
        )
    
    def _extract_entities(self, description: str) -> List[str]:
        """Extract entities from description"""
        # Simple entity extraction (would use NER in practice)
        words = description.split()
        entities = [word for word in words if word[0].isupper()]
        return entities
    
    def _find_structural_mapping(self, source_entities: List[str], 
                               target_entities: List[str]) -> Dict[str, str]:
        """Find structural mapping between entities"""
        mapping = {}
        
        # Simple mapping based on position and similarity
        for i, source_entity in enumerate(source_entities):
            if i < len(target_entities):
                mapping[source_entity] = target_entities[i]
        
        return mapping
    
    def _calculate_similarity(self, source_entities: List[str], 
                            target_entities: List[str], 
                            mapping: Dict[str, str]) -> float:
        """Calculate similarity score for analogy"""
        if not mapping:
            return 0.0
        
        # Simple similarity based on mapping completeness
        mapped_ratio = len(mapping) / max(len(source_entities), len(target_entities))
        return mapped_ratio
    
    def _infer_domain(self, description: str) -> str:
        """Infer domain from description"""
        # Simple domain inference based on keywords
        biology_keywords = ["cell", "organism", "gene", "evolution"]
        physics_keywords = ["force", "energy", "particle", "wave"]
        computer_keywords = ["algorithm", "data", "network", "software"]
        
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in biology_keywords):
            return "biology"
        elif any(keyword in description_lower for keyword in physics_keywords):
            return "physics"
        elif any(keyword in description_lower for keyword in computer_keywords):
            return "computer_science"
        else:
            return "general"


class ExplanationGenerator:
    """Generates human-readable explanations for reasoning"""
    
    def __init__(self):
        self.explanation_templates = {
            ReasoningType.DEDUCTIVE: "Based on the rule '{rule}', since {premises}, we can conclude {conclusion}.",
            ReasoningType.INDUCTIVE: "From {num_observations} observations, we can generalize that {conclusion}.",
            ReasoningType.ABDUCTIVE: "The best explanation for {observation} is {conclusion}.",
            ReasoningType.ANALOGICAL: "By analogy with {source_domain}, in {target_domain} we expect {conclusion}.",
            ReasoningType.CAUSAL: "Because {cause} leads to {effect}, we can infer {conclusion}.",
        }
        
        self.confidence_phrases = {
            ConfidenceLevel.VERY_LOW: "it's possible that",
            ConfidenceLevel.LOW: "it seems likely that",
            ConfidenceLevel.MEDIUM: "we can reasonably conclude that",
            ConfidenceLevel.HIGH: "we can confidently say that",
            ConfidenceLevel.VERY_HIGH: "we are quite certain that",
            ConfidenceLevel.CERTAIN: "we know with certainty that"
        }
    
    async def generate_step_explanation(self, step: ReasoningStep) -> str:
        """Generate explanation for a single reasoning step"""
        template = self.explanation_templates.get(
            step.reasoning_type, 
            "Through {reasoning_type} reasoning, {conclusion}."
        )
        
        confidence_phrase = self._get_confidence_phrase(step.confidence)
        
        explanation = template.format(
            rule=step.rule_applied or "logical inference",
            premises=" and ".join(step.premise),
            conclusion=step.conclusion,
            observation=" and ".join(step.evidence),
            num_observations=len(step.evidence),
            source_domain="the known domain",
            target_domain="the target domain",
            cause="the given conditions",
            effect="the observed outcome",
            reasoning_type=step.reasoning_type.value
        )
        
        return f"{confidence_phrase.capitalize()} {explanation}"
    
    async def generate_chain_explanation(self, chain: ReasoningChain) -> str:
        """Generate explanation for an entire reasoning chain"""
        explanations = []
        
        explanations.append(f"To answer '{chain.query}', I followed this reasoning:")
        
        for i, step in enumerate(chain.steps, 1):
            step_explanation = await self.generate_step_explanation(step)
            explanations.append(f"{i}. {step_explanation}")
        
        explanations.append(f"\nTherefore, {chain.final_conclusion} (confidence: {chain.overall_confidence:.2f})")
        
        return "\n".join(explanations)
    
    def _get_confidence_phrase(self, confidence: float) -> str:
        """Get appropriate confidence phrase"""
        if confidence >= 0.9:
            return self.confidence_phrases[ConfidenceLevel.VERY_HIGH]
        elif confidence >= 0.7:
            return self.confidence_phrases[ConfidenceLevel.HIGH]
        elif confidence >= 0.5:
            return self.confidence_phrases[ConfidenceLevel.MEDIUM]
        elif confidence >= 0.3:
            return self.confidence_phrases[ConfidenceLevel.LOW]
        else:
            return self.confidence_phrases[ConfidenceLevel.VERY_LOW]


class TemporalReasoner:
    """Handles temporal reasoning"""
    
    def __init__(self):
        self.temporal_facts = []
        self.temporal_rules = {}
        
    def add_temporal_fact(self, fact: str, timestamp: datetime, duration: Optional[int] = None):
        """Add a temporal fact"""
        self.temporal_facts.append({
            "fact": fact,
            "timestamp": timestamp,
            "duration": duration,
            "end_time": timestamp + timedelta(seconds=duration) if duration else None
        })
    
    async def temporal_query(self, query: str, time_point: datetime) -> List[str]:
        """Query facts that were true at a specific time point"""
        valid_facts = []
        
        for fact_entry in self.temporal_facts:
            if fact_entry["timestamp"] <= time_point:
                if fact_entry["end_time"] is None or fact_entry["end_time"] >= time_point:
                    if query.lower() in fact_entry["fact"].lower():
                        valid_facts.append(fact_entry["fact"])
        
        return valid_facts
    
    async def temporal_sequence_reasoning(self, events: List[Tuple[str, datetime]]) -> List[str]:
        """Reason about temporal sequences of events"""
        conclusions = []
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x[1])
        
        # Look for temporal patterns
        for i in range(len(sorted_events) - 1):
            current_event, current_time = sorted_events[i]
            next_event, next_time = sorted_events[i + 1]
            
            time_diff = (next_time - current_time).total_seconds()
            
            if time_diff < 3600:  # Within an hour
                conclusions.append(f"{current_event} was quickly followed by {next_event}")
            elif time_diff < 86400:  # Within a day
                conclusions.append(f"{current_event} preceded {next_event} on the same day")
            else:
                conclusions.append(f"{current_event} occurred before {next_event}")
        
        return conclusions


class ReasoningEngine:
    """Main reasoning engine coordinating all reasoning components"""
    
    def __init__(self):
        self.logical_engine = LogicalInferenceEngine()
        self.pattern_matcher = PatternMatcher()
        self.analogical_reasoner = AnalogicalReasoner()
        self.explanation_generator = ExplanationGenerator()
        self.temporal_reasoner = TemporalReasoner()
        
        # Reasoning history and performance tracking
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.performance_metrics = {
            "total_queries": 0,
            "successful_reasonings": 0,
            "average_confidence": 0.0,
            "reasoning_type_usage": defaultdict(int)
        }
        
        # Initialize with some default patterns
        self._initialize_default_patterns()
    
    async def reason(self, query: str, context: Dict[str, Any] = None, 
                    reasoning_types: List[ReasoningType] = None) -> ReasoningChain:
        """Main reasoning method"""
        self.performance_metrics["total_queries"] += 1
        
        chain = ReasoningChain(
            chain_id=str(uuid4()),
            query=query,
            steps=[],
            final_conclusion="",
            overall_confidence=0.0,
            reasoning_path=[],
            alternative_paths=[],
            created_at=datetime.utcnow()
        )
        
        context = context or {}
        reasoning_types = reasoning_types or [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE]
        
        # Apply different reasoning strategies
        for reasoning_type in reasoning_types:
            steps = await self._apply_reasoning_strategy(reasoning_type, query, context)
            chain.steps.extend(steps)
            
            # Track usage
            self.performance_metrics["reasoning_type_usage"][reasoning_type] += 1
        
        # Generate final conclusion and confidence
        if chain.steps:
            chain.final_conclusion = chain.steps[-1].conclusion
            chain.overall_confidence = self._calculate_overall_confidence(chain.steps)
            chain.reasoning_path = [step.step_id for step in chain.steps]
            
            self.performance_metrics["successful_reasonings"] += 1
        else:
            chain.final_conclusion = "Unable to reach a conclusion"
            chain.overall_confidence = 0.0
        
        # Update average confidence
        self._update_average_confidence(chain.overall_confidence)
        
        # Store reasoning chain
        self.reasoning_chains[chain.chain_id] = chain
        
        return chain
    
    async def explain_reasoning(self, chain_id: str) -> str:
        """Generate explanation for a reasoning chain"""
        if chain_id not in self.reasoning_chains:
            return "Reasoning chain not found"
        
        chain = self.reasoning_chains[chain_id]
        return await self.explanation_generator.generate_chain_explanation(chain)
    
    async def find_similar_reasonings(self, query: str, limit: int = 5) -> List[ReasoningChain]:
        """Find similar past reasonings"""
        similar_chains = []
        
        for chain in self.reasoning_chains.values():
            similarity = self._calculate_query_similarity(query, chain.query)
            if similarity > 0.3:  # Threshold for similarity
                similar_chains.append((chain, similarity))
        
        # Sort by similarity and return top results
        similar_chains.sort(key=lambda x: x[1], reverse=True)
        return [chain for chain, _ in similar_chains[:limit]]
    
    async def _apply_reasoning_strategy(self, reasoning_type: ReasoningType, 
                                      query: str, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Apply a specific reasoning strategy"""
        steps = []
        
        if reasoning_type == ReasoningType.DEDUCTIVE:
            if "premises" in context:
                steps = await self.logical_engine.deductive_reasoning(context["premises"])
        
        elif reasoning_type == ReasoningType.INDUCTIVE:
            if "observations" in context:
                steps = await self.logical_engine.inductive_reasoning(context["observations"])
        
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            if "observation" in context and "explanations" in context:
                steps = await self.logical_engine.abductive_reasoning(
                    context["observation"], context["explanations"]
                )
        
        elif reasoning_type == ReasoningType.ANALOGICAL:
            # Try to find analogies in the query
            patterns = await self.pattern_matcher.match_patterns(query)
            for pattern, variables in patterns:
                if pattern.pattern_type == "analogical":
                    step = ReasoningStep(
                        step_id=str(uuid4()),
                        reasoning_type=ReasoningType.ANALOGICAL,
                        premise=[query],
                        conclusion=f"By analogy: {pattern.template}",
                        rule_applied=pattern.pattern_id,
                        confidence=pattern.confidence,
                        explanation="Applied analogical reasoning pattern",
                        evidence=[str(variables)],
                        timestamp=datetime.utcnow()
                    )
                    steps.append(step)
        
        elif reasoning_type == ReasoningType.TEMPORAL:
            if "events" in context:
                temporal_conclusions = await self.temporal_reasoner.temporal_sequence_reasoning(
                    context["events"]
                )
                for conclusion in temporal_conclusions:
                    step = ReasoningStep(
                        step_id=str(uuid4()),
                        reasoning_type=ReasoningType.TEMPORAL,
                        premise=["temporal_events"],
                        conclusion=conclusion,
                        rule_applied="temporal_sequence",
                        confidence=0.7,
                        explanation="Temporal sequence analysis",
                        evidence=[str(context["events"])],
                        timestamp=datetime.utcnow()
                    )
                    steps.append(step)
        
        return steps
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence for a reasoning chain"""
        if not steps:
            return 0.0
        
        # Weighted average based on step importance
        weights = [1.0 / (i + 1) for i in range(len(steps))]  # Later steps have higher weight
        weighted_confidences = [step.confidence * weight for step, weight in zip(steps, weights)]
        
        return sum(weighted_confidences) / sum(weights)
    
    def _update_average_confidence(self, new_confidence: float):
        """Update average confidence metric"""
        total_queries = self.performance_metrics["total_queries"]
        current_avg = self.performance_metrics["average_confidence"]
        
        self.performance_metrics["average_confidence"] = (
            (current_avg * (total_queries - 1) + new_confidence) / total_queries
        )
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        # Simple word overlap similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _initialize_default_patterns(self):
        """Initialize with some default reasoning patterns"""
        # Causal pattern
        causal_pattern = Pattern(
            pattern_id="causal_basic",
            pattern_type="causal",
            template="If {cause} then {effect}",
            variables=["cause", "effect"],
            constraints={},
            confidence=0.8,
            usage_count=0,
            examples=["If it rains then the ground gets wet"]
        )
        self.pattern_matcher.add_pattern(causal_pattern)
        
        # Analogical pattern
        analogical_pattern = Pattern(
            pattern_id="analogical_basic",
            pattern_type="analogical",
            template="{source} is like {target} because {reason}",
            variables=["source", "target", "reason"],
            constraints={},
            confidence=0.6,
            usage_count=0,
            examples=["An atom is like a solar system because electrons orbit the nucleus"]
        )
        self.pattern_matcher.add_pattern(analogical_pattern)
    
    async def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        return {
            "performance_metrics": dict(self.performance_metrics),
            "total_reasoning_chains": len(self.reasoning_chains),
            "patterns_available": len(self.pattern_matcher.patterns),
            "analogies_available": len(self.analogical_reasoner.analogies),
            "inference_rules": len(self.logical_engine.inference_rules),
            "temporal_facts": len(self.temporal_reasoner.temporal_facts)
        }