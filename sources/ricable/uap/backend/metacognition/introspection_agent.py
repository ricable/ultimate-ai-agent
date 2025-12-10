"""
Agent 40: Self-Improving AI Metacognition System - Introspection Agent
Implements deep system self-analysis and introspective capabilities.
"""

import asyncio
import json
import logging
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import uuid4
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class IntrospectionDomain(Enum):
    """Domains of introspective analysis"""
    COGNITIVE_PATTERNS = "cognitive_patterns"
    DECISION_MAKING = "decision_making"
    LEARNING_EFFICIENCY = "learning_efficiency"
    COMMUNICATION_STYLE = "communication_style"
    PROBLEM_SOLVING = "problem_solving"
    RESOURCE_UTILIZATION = "resource_utilization"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"
    KNOWLEDGE_GAPS = "knowledge_gaps"


class AnalysisDepth(Enum):
    """Depth levels for introspective analysis"""
    SURFACE = "surface"  # Immediate patterns
    BEHAVIORAL = "behavioral"  # Behavior patterns over time
    COGNITIVE = "cognitive"  # Underlying reasoning patterns
    ARCHITECTURAL = "architectural"  # Deep system architecture analysis


@dataclass
class IntrospectiveInsight:
    """Represents an insight gained through introspection"""
    id: str
    timestamp: datetime
    domain: IntrospectionDomain
    depth: AnalysisDepth
    pattern_description: str
    evidence: List[Dict[str, Any]]
    confidence: float
    implications: List[str]
    actionable_recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class CognitivePattern:
    """Represents a detected cognitive pattern"""
    pattern_id: str
    pattern_type: str
    frequency: int
    accuracy_correlation: float
    performance_impact: float
    contexts: List[str]
    triggers: List[str]
    outcomes: List[str]
    first_detected: datetime
    last_observed: datetime


@dataclass
class SystemCapability:
    """Represents a system capability assessment"""
    capability_name: str
    current_level: float  # 0.0 to 1.0
    potential_level: float  # Estimated maximum potential
    utilization_rate: float  # How often it's used
    improvement_rate: float  # Rate of improvement over time
    bottlenecks: List[str]
    enhancement_opportunities: List[str]
    dependency_map: Dict[str, float]  # Dependencies on other capabilities


class CognitivePatternAnalyzer:
    """Analyzes cognitive patterns in system behavior"""
    
    def __init__(self):
        self.pattern_history: Dict[str, List[CognitivePattern]] = defaultdict(list)
        self.interaction_logs: deque = deque(maxlen=10000)
        self.decision_patterns: Dict[str, Dict] = {}
        self.learning_patterns: Dict[str, Dict] = {}
        self.communication_patterns: Dict[str, Dict] = {}
    
    async def analyze_interaction(self, interaction_data: Dict[str, Any]) -> List[CognitivePattern]:
        """Analyze a single interaction for cognitive patterns"""
        self.interaction_logs.append({
            'timestamp': datetime.utcnow(),
            'data': interaction_data
        })
        
        patterns = []
        
        # Analyze decision-making patterns
        decision_patterns = await self._analyze_decision_patterns(interaction_data)
        patterns.extend(decision_patterns)
        
        # Analyze communication patterns
        communication_patterns = await self._analyze_communication_patterns(interaction_data)
        patterns.extend(communication_patterns)
        
        # Analyze problem-solving approaches
        problem_solving_patterns = await self._analyze_problem_solving_patterns(interaction_data)
        patterns.extend(problem_solving_patterns)
        
        return patterns
    
    async def _analyze_decision_patterns(self, interaction_data: Dict[str, Any]) -> List[CognitivePattern]:
        """Analyze decision-making patterns"""
        patterns = []
        
        if 'decision_points' in interaction_data:
            for decision in interaction_data['decision_points']:
                decision_type = decision.get('type', 'unknown')
                decision_time = decision.get('processing_time', 0)
                decision_confidence = decision.get('confidence', 0.5)
                
                # Check for pattern in decision timing
                if decision_type not in self.decision_patterns:
                    self.decision_patterns[decision_type] = {
                        'times': [],
                        'confidences': [],
                        'outcomes': []
                    }
                
                self.decision_patterns[decision_type]['times'].append(decision_time)
                self.decision_patterns[decision_type]['confidences'].append(decision_confidence)
                
                # Detect patterns if we have enough data
                if len(self.decision_patterns[decision_type]['times']) >= 10:
                    pattern = await self._detect_decision_timing_pattern(decision_type)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    async def _analyze_communication_patterns(self, interaction_data: Dict[str, Any]) -> List[CognitivePattern]:
        """Analyze communication patterns"""
        patterns = []
        
        if 'response_data' in interaction_data:
            response = interaction_data['response_data']
            
            # Analyze response length patterns
            response_length = len(str(response.get('content', '')))
            response_complexity = self._calculate_response_complexity(response)
            
            # Store communication data
            comm_key = 'general_communication'
            if comm_key not in self.communication_patterns:
                self.communication_patterns[comm_key] = {
                    'lengths': [],
                    'complexities': [],
                    'response_times': []
                }
            
            self.communication_patterns[comm_key]['lengths'].append(response_length)
            self.communication_patterns[comm_key]['complexities'].append(response_complexity)
            
            if 'response_time' in interaction_data:
                self.communication_patterns[comm_key]['response_times'].append(
                    interaction_data['response_time']
                )
            
            # Detect patterns if we have enough data
            if len(self.communication_patterns[comm_key]['lengths']) >= 20:
                pattern = await self._detect_communication_pattern(comm_key)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_problem_solving_patterns(self, interaction_data: Dict[str, Any]) -> List[CognitivePattern]:
        """Analyze problem-solving approach patterns"""
        patterns = []
        
        if 'problem_type' in interaction_data and 'solution_approach' in interaction_data:
            problem_type = interaction_data['problem_type']
            approach = interaction_data['solution_approach']
            success = interaction_data.get('success', False)
            
            # Track problem-solving patterns
            pattern_key = f"{problem_type}_{approach}"
            
            if pattern_key not in self.pattern_history:
                self.pattern_history[pattern_key] = []
            
            # Create pattern record
            pattern_record = CognitivePattern(
                pattern_id=str(uuid4()),
                pattern_type='problem_solving_approach',
                frequency=1,
                accuracy_correlation=1.0 if success else 0.0,
                performance_impact=interaction_data.get('efficiency', 0.5),
                contexts=[problem_type],
                triggers=[interaction_data.get('trigger', 'unknown')],
                outcomes=['success' if success else 'failure'],
                first_detected=datetime.utcnow(),
                last_observed=datetime.utcnow()
            )
            
            # Check if this pattern exists and update it
            existing_pattern = await self._find_similar_pattern(pattern_record)
            if existing_pattern:
                existing_pattern.frequency += 1
                existing_pattern.accuracy_correlation = (
                    existing_pattern.accuracy_correlation * (existing_pattern.frequency - 1) + 
                    (1.0 if success else 0.0)
                ) / existing_pattern.frequency
                existing_pattern.last_observed = datetime.utcnow()
            else:
                self.pattern_history[pattern_key].append(pattern_record)
                patterns.append(pattern_record)
        
        return patterns
    
    async def _detect_decision_timing_pattern(self, decision_type: str) -> Optional[CognitivePattern]:
        """Detect patterns in decision timing"""
        data = self.decision_patterns[decision_type]
        times = data['times'][-20:]  # Last 20 decisions
        
        if len(times) < 10:
            return None
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Check for consistent timing pattern
        if std_time / avg_time < 0.3:  # Low variance indicates consistent pattern
            return CognitivePattern(
                pattern_id=str(uuid4()),
                pattern_type='consistent_decision_timing',
                frequency=len(times),
                accuracy_correlation=np.mean(data['confidences'][-20:]),
                performance_impact=1.0 / avg_time if avg_time > 0 else 0,
                contexts=[decision_type],
                triggers=['decision_required'],
                outcomes=['consistent_timing'],
                first_detected=datetime.utcnow() - timedelta(minutes=30),
                last_observed=datetime.utcnow()
            )
        
        return None
    
    async def _detect_communication_pattern(self, comm_key: str) -> Optional[CognitivePattern]:
        """Detect patterns in communication style"""
        data = self.communication_patterns[comm_key]
        lengths = data['lengths'][-50:]  # Last 50 responses
        
        if len(lengths) < 20:
            return None
        
        # Analyze length consistency
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        if std_length / avg_length < 0.5:  # Consistent length pattern
            return CognitivePattern(
                pattern_id=str(uuid4()),
                pattern_type='consistent_response_length',
                frequency=len(lengths),
                accuracy_correlation=0.8,  # Assume consistent communication is generally good
                performance_impact=avg_length / 1000,  # Normalize by expected length
                contexts=['communication'],
                triggers=['user_query'],
                outcomes=['consistent_communication_style'],
                first_detected=datetime.utcnow() - timedelta(hours=1),
                last_observed=datetime.utcnow()
            )
        
        return None
    
    async def _find_similar_pattern(self, new_pattern: CognitivePattern) -> Optional[CognitivePattern]:
        """Find similar existing pattern"""
        for pattern_list in self.pattern_history.values():
            for existing_pattern in pattern_list:
                if (existing_pattern.pattern_type == new_pattern.pattern_type and
                    set(existing_pattern.contexts) & set(new_pattern.contexts)):
                    return existing_pattern
        return None
    
    def _calculate_response_complexity(self, response_data: Dict[str, Any]) -> float:
        """Calculate complexity score for a response"""
        content = str(response_data.get('content', ''))
        
        # Simple complexity metrics
        word_count = len(content.split())
        unique_words = len(set(content.lower().split()))
        avg_word_length = np.mean([len(word) for word in content.split()]) if content.split() else 0
        
        # Normalize complexity score
        complexity = (
            min(1.0, word_count / 200) * 0.4 +  # Length component
            min(1.0, unique_words / 100) * 0.4 +  # Vocabulary diversity
            min(1.0, avg_word_length / 8) * 0.2   # Word complexity
        )
        
        return complexity


class CapabilityAssessment:
    """Assesses and tracks system capabilities"""
    
    def __init__(self):
        self.capabilities: Dict[str, SystemCapability] = {}
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self.capability_interactions: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    async def assess_capability(self, capability_name: str, 
                               performance_data: Dict[str, Any]) -> SystemCapability:
        """Assess a specific system capability"""
        current_performance = performance_data.get('current_score', 0.5)
        
        # Initialize capability if not exists
        if capability_name not in self.capabilities:
            self.capabilities[capability_name] = SystemCapability(
                capability_name=capability_name,
                current_level=current_performance,
                potential_level=min(1.0, current_performance * 1.5),  # Estimate potential
                utilization_rate=0.5,
                improvement_rate=0.0,
                bottlenecks=[],
                enhancement_opportunities=[],
                dependency_map={}
            )
        
        capability = self.capabilities[capability_name]
        
        # Update performance history
        self.performance_history[capability_name].append({
            'timestamp': datetime.utcnow(),
            'performance': current_performance,
            'context': performance_data.get('context', {})
        })
        
        # Calculate improvement rate
        if len(self.performance_history[capability_name]) >= 2:
            recent_scores = [entry['performance'] for entry in 
                           self.performance_history[capability_name][-10:]]
            if len(recent_scores) >= 2:
                capability.improvement_rate = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        # Update current level
        capability.current_level = current_performance
        
        # Assess bottlenecks
        capability.bottlenecks = await self._identify_bottlenecks(capability_name, performance_data)
        
        # Identify enhancement opportunities
        capability.enhancement_opportunities = await self._identify_enhancements(
            capability_name, performance_data
        )
        
        # Update utilization rate
        capability.utilization_rate = performance_data.get('utilization', capability.utilization_rate)
        
        return capability
    
    async def _identify_bottlenecks(self, capability_name: str, 
                                   performance_data: Dict[str, Any]) -> List[str]:
        """Identify bottlenecks limiting capability performance"""
        bottlenecks = []
        
        # Check for performance constraints
        if performance_data.get('response_time', 0) > 2.0:
            bottlenecks.append('slow_response_time')
        
        if performance_data.get('accuracy', 1.0) < 0.8:
            bottlenecks.append('low_accuracy')
        
        if performance_data.get('resource_usage', 0) > 0.9:
            bottlenecks.append('high_resource_usage')
        
        # Check historical performance trends
        if capability_name in self.performance_history:
            recent_performance = [entry['performance'] for entry in 
                                self.performance_history[capability_name][-5:]]
            if len(recent_performance) >= 3 and all(p < 0.7 for p in recent_performance):
                bottlenecks.append('consistently_low_performance')
        
        return bottlenecks
    
    async def _identify_enhancements(self, capability_name: str, 
                                    performance_data: Dict[str, Any]) -> List[str]:
        """Identify opportunities for capability enhancement"""
        enhancements = []
        
        current_score = performance_data.get('current_score', 0.5)
        
        # Suggest enhancements based on current performance
        if current_score < 0.8:
            enhancements.append('performance_optimization')
        
        if performance_data.get('utilization', 0.5) < 0.6:
            enhancements.append('increase_utilization')
        
        # Check for specific improvement areas
        if 'weaknesses' in performance_data:
            for weakness in performance_data['weaknesses']:
                enhancements.append(f'address_{weakness}')
        
        # Suggest cross-capability improvements
        related_capabilities = await self._find_related_capabilities(capability_name)
        for related_cap in related_capabilities:
            if self.capabilities[related_cap].current_level > current_score + 0.2:
                enhancements.append(f'learn_from_{related_cap}')
        
        return enhancements
    
    async def _find_related_capabilities(self, capability_name: str) -> List[str]:
        """Find capabilities related to the given capability"""
        related = []
        
        # Simple heuristic based on capability names
        capability_groups = {
            'language': ['text_processing', 'communication', 'translation'],
            'analysis': ['data_analysis', 'pattern_recognition', 'reasoning'],
            'problem_solving': ['decision_making', 'planning', 'optimization']
        }
        
        for group, capabilities in capability_groups.items():
            if any(keyword in capability_name.lower() for keyword in capabilities):
                related.extend([cap for cap in self.capabilities.keys() 
                              if any(keyword in cap.lower() for keyword in capabilities)
                              and cap != capability_name])
        
        return related[:3]  # Return top 3 related capabilities


class IntrospectionAgent:
    """Main agent for system introspection and self-analysis"""
    
    def __init__(self):
        self.agent_id = str(uuid4())
        self.cognitive_analyzer = CognitivePatternAnalyzer()
        self.capability_assessor = CapabilityAssessment()
        self.insights_history: List[IntrospectiveInsight] = []
        self.system_metrics: Dict[str, Any] = {}
        self.introspection_schedule = {
            'cognitive_analysis': timedelta(minutes=10),
            'capability_assessment': timedelta(minutes=30),
            'deep_introspection': timedelta(hours=2)
        }
        self.last_introspection = {
            'cognitive_analysis': datetime.utcnow(),
            'capability_assessment': datetime.utcnow(),
            'deep_introspection': datetime.utcnow()
        }
    
    async def initialize(self) -> bool:
        """Initialize the introspection agent"""
        try:
            logger.info(f"Initializing Introspection Agent {self.agent_id}")
            
            # Start introspection monitoring loop
            asyncio.create_task(self._introspection_loop())
            
            logger.info("Introspection Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Introspection Agent: {e}")
            return False
    
    async def analyze_system_interaction(self, interaction_data: Dict[str, Any]) -> List[IntrospectiveInsight]:
        """Analyze a system interaction for insights"""
        insights = []
        
        # Cognitive pattern analysis
        cognitive_patterns = await self.cognitive_analyzer.analyze_interaction(interaction_data)
        
        for pattern in cognitive_patterns:
            insight = IntrospectiveInsight(
                id=str(uuid4()),
                timestamp=datetime.utcnow(),
                domain=IntrospectionDomain.COGNITIVE_PATTERNS,
                depth=AnalysisDepth.COGNITIVE,
                pattern_description=f"Detected {pattern.pattern_type} pattern with {pattern.frequency} occurrences",
                evidence=[asdict(pattern)],
                confidence=min(1.0, pattern.frequency / 10),  # Confidence based on frequency
                implications=[
                    f"This pattern impacts performance by {pattern.performance_impact:.2f}",
                    f"Pattern accuracy correlation: {pattern.accuracy_correlation:.2f}"
                ],
                actionable_recommendations=await self._generate_pattern_recommendations(pattern),
                metadata={'pattern_id': pattern.pattern_id, 'analysis_type': 'cognitive'}
            )
            insights.append(insight)
        
        # Capability assessment
        if 'capability_data' in interaction_data:
            for capability_name, performance_data in interaction_data['capability_data'].items():
                capability = await self.capability_assessor.assess_capability(
                    capability_name, performance_data
                )
                
                insight = IntrospectiveInsight(
                    id=str(uuid4()),
                    timestamp=datetime.utcnow(),
                    domain=IntrospectionDomain.RESOURCE_UTILIZATION,
                    depth=AnalysisDepth.BEHAVIORAL,
                    pattern_description=f"Capability '{capability_name}' assessment",
                    evidence=[asdict(capability)],
                    confidence=0.8,
                    implications=[
                        f"Current level: {capability.current_level:.2f}",
                        f"Potential level: {capability.potential_level:.2f}",
                        f"Utilization rate: {capability.utilization_rate:.2f}"
                    ],
                    actionable_recommendations=capability.enhancement_opportunities,
                    metadata={'capability_name': capability_name, 'analysis_type': 'capability'}
                )
                insights.append(insight)
        
        # Store insights
        self.insights_history.extend(insights)
        
        # Maintain insights history size
        if len(self.insights_history) > 1000:
            self.insights_history = self.insights_history[-500:]
        
        return insights
    
    async def perform_deep_introspection(self) -> List[IntrospectiveInsight]:
        """Perform deep introspective analysis"""
        insights = []
        
        # Analyze learning efficiency
        learning_insight = await self._analyze_learning_efficiency()
        if learning_insight:
            insights.append(learning_insight)
        
        # Analyze decision-making patterns
        decision_insight = await self._analyze_decision_making_patterns()
        if decision_insight:
            insights.append(decision_insight)
        
        # Analyze communication effectiveness
        communication_insight = await self._analyze_communication_effectiveness()
        if communication_insight:
            insights.append(communication_insight)
        
        # Analyze system resource utilization
        resource_insight = await self._analyze_resource_utilization()
        if resource_insight:
            insights.append(resource_insight)
        
        # Store insights
        self.insights_history.extend(insights)
        
        return insights
    
    async def _analyze_learning_efficiency(self) -> Optional[IntrospectiveInsight]:
        """Analyze learning efficiency patterns"""
        # Get recent capability improvements
        recent_improvements = {}
        for capability_name, capability in self.capability_assessor.capabilities.items():
            if capability.improvement_rate > 0:
                recent_improvements[capability_name] = capability.improvement_rate
        
        if not recent_improvements:
            return None
        
        avg_improvement = np.mean(list(recent_improvements.values()))
        
        return IntrospectiveInsight(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            domain=IntrospectionDomain.LEARNING_EFFICIENCY,
            depth=AnalysisDepth.COGNITIVE,
            pattern_description=f"Learning efficiency analysis: {len(recent_improvements)} capabilities improving",
            evidence=[{'improvements': recent_improvements, 'average_rate': avg_improvement}],
            confidence=0.85,
            implications=[
                f"Average improvement rate: {avg_improvement:.3f}",
                f"Learning is occurring across {len(recent_improvements)} capabilities",
                "System demonstrates adaptive learning behavior"
            ],
            actionable_recommendations=[
                "Focus learning resources on underperforming capabilities",
                "Implement cross-capability knowledge transfer",
                "Increase learning rate for capabilities with high potential"
            ],
            metadata={'analysis_type': 'learning_efficiency', 'capability_count': len(recent_improvements)}
        )
    
    async def _analyze_decision_making_patterns(self) -> Optional[IntrospectiveInsight]:
        """Analyze decision-making patterns"""
        decision_data = self.cognitive_analyzer.decision_patterns
        
        if not decision_data:
            return None
        
        # Analyze decision consistency
        consistency_scores = []
        for decision_type, data in decision_data.items():
            if len(data['times']) >= 5:
                times = data['times'][-10:]
                consistency = 1.0 - (np.std(times) / max(np.mean(times), 0.1))
                consistency_scores.append(consistency)
        
        if not consistency_scores:
            return None
        
        avg_consistency = np.mean(consistency_scores)
        
        return IntrospectiveInsight(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            domain=IntrospectionDomain.DECISION_MAKING,
            depth=AnalysisDepth.BEHAVIORAL,
            pattern_description=f"Decision-making consistency analysis across {len(decision_data)} decision types",
            evidence=[{'decision_types': list(decision_data.keys()), 'consistency_score': avg_consistency}],
            confidence=0.8,
            implications=[
                f"Decision consistency score: {avg_consistency:.2f}",
                "Consistent decision-making indicates stable reasoning patterns",
                "High consistency suggests reliable cognitive processes"
            ],
            actionable_recommendations=[
                "Maintain decision consistency for reliable behavior",
                "Monitor for decision pattern changes that might indicate issues",
                "Document successful decision patterns for replication"
            ] if avg_consistency > 0.7 else [
                "Investigate causes of decision inconsistency",
                "Implement decision validation mechanisms",
                "Review decision-making algorithms for stability"
            ],
            metadata={'analysis_type': 'decision_making', 'consistency_score': avg_consistency}
        )
    
    async def _analyze_communication_effectiveness(self) -> Optional[IntrospectiveInsight]:
        """Analyze communication effectiveness"""
        comm_data = self.cognitive_analyzer.communication_patterns
        
        if not comm_data:
            return None
        
        # Analyze communication metrics
        effectiveness_metrics = {}
        for comm_type, data in comm_data.items():
            if data['lengths'] and data['complexities']:
                avg_length = np.mean(data['lengths'][-20:])
                avg_complexity = np.mean(data['complexities'][-20:])
                
                # Calculate effectiveness score (balance of length and complexity)
                effectiveness = min(1.0, (avg_complexity * 0.7 + (1.0 - min(1.0, avg_length / 500)) * 0.3))
                effectiveness_metrics[comm_type] = {
                    'effectiveness': effectiveness,
                    'avg_length': avg_length,
                    'avg_complexity': avg_complexity
                }
        
        if not effectiveness_metrics:
            return None
        
        overall_effectiveness = np.mean([m['effectiveness'] for m in effectiveness_metrics.values()])
        
        return IntrospectiveInsight(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            domain=IntrospectionDomain.COMMUNICATION_STYLE,
            depth=AnalysisDepth.BEHAVIORAL,
            pattern_description=f"Communication effectiveness analysis across {len(effectiveness_metrics)} communication types",
            evidence=[effectiveness_metrics],
            confidence=0.75,
            implications=[
                f"Overall communication effectiveness: {overall_effectiveness:.2f}",
                "Communication style demonstrates consistent patterns",
                "Response complexity is appropriate for content"
            ],
            actionable_recommendations=[
                "Maintain current communication style effectiveness",
                "Continue balancing complexity with clarity",
                "Monitor user feedback for communication improvements"
            ] if overall_effectiveness > 0.7 else [
                "Improve communication clarity and conciseness",
                "Adjust response complexity to user needs",
                "Implement communication style optimization"
            ],
            metadata={'analysis_type': 'communication', 'effectiveness_score': overall_effectiveness}
        )
    
    async def _analyze_resource_utilization(self) -> Optional[IntrospectiveInsight]:
        """Analyze system resource utilization patterns"""
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Store metrics
            self.system_metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'timestamp': datetime.utcnow()
            }
            
            # Calculate resource efficiency
            resource_efficiency = (
                min(1.0, (100 - cpu_percent) / 100) * 0.5 +  # CPU efficiency
                min(1.0, (100 - memory_percent) / 100) * 0.5   # Memory efficiency
            )
            
            return IntrospectiveInsight(
                id=str(uuid4()),
                timestamp=datetime.utcnow(),
                domain=IntrospectionDomain.RESOURCE_UTILIZATION,
                depth=AnalysisDepth.SURFACE,
                pattern_description=f"System resource utilization analysis",
                evidence=[self.system_metrics],
                confidence=0.9,
                implications=[
                    f"CPU usage: {cpu_percent:.1f}%",
                    f"Memory usage: {memory_percent:.1f}%",
                    f"Resource efficiency: {resource_efficiency:.2f}"
                ],
                actionable_recommendations=[
                    "Monitor resource usage trends",
                    "Optimize resource-intensive processes",
                    "Implement resource usage alerts"
                ] if resource_efficiency < 0.7 else [
                    "Resource utilization is optimal",
                    "Continue current resource management practices",
                    "Monitor for any degradation in efficiency"
                ],
                metadata={'analysis_type': 'resource_utilization', 'efficiency_score': resource_efficiency}
            )
            
        except Exception as e:
            logger.error(f"Error analyzing resource utilization: {e}")
            return None
    
    async def _generate_pattern_recommendations(self, pattern: CognitivePattern) -> List[str]:
        """Generate recommendations based on a cognitive pattern"""
        recommendations = []
        
        if pattern.accuracy_correlation > 0.8:
            recommendations.append(f"Reinforce {pattern.pattern_type} pattern - it correlates with high accuracy")
        elif pattern.accuracy_correlation < 0.5:
            recommendations.append(f"Review {pattern.pattern_type} pattern - low accuracy correlation detected")
        
        if pattern.performance_impact > 0.8:
            recommendations.append(f"Leverage {pattern.pattern_type} pattern for performance optimization")
        elif pattern.performance_impact < 0.3:
            recommendations.append(f"Investigate {pattern.pattern_type} pattern - low performance impact")
        
        if pattern.frequency > 50:
            recommendations.append(f"Monitor {pattern.pattern_type} pattern - high frequency detected")
        
        return recommendations
    
    async def _introspection_loop(self):
        """Main introspection processing loop"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Perform scheduled introspection
                for introspection_type, interval in self.introspection_schedule.items():
                    if current_time - self.last_introspection[introspection_type] > interval:
                        if introspection_type == 'deep_introspection':
                            await self.perform_deep_introspection()
                        
                        self.last_introspection[introspection_type] = current_time
                
                # Sleep before next iteration
                await asyncio.sleep(60)  # 1-minute processing cycle
                
            except Exception as e:
                logger.error(f"Error in introspection loop: {e}")
                await asyncio.sleep(300)  # 5-minute sleep on error
    
    async def get_insights_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent introspective insights"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_insights = [i for i in self.insights_history if i.timestamp > cutoff_time]
        
        # Group insights by domain
        insights_by_domain = defaultdict(list)
        for insight in recent_insights:
            insights_by_domain[insight.domain.value].append(insight)
        
        # Calculate average confidence by domain
        domain_confidence = {}
        for domain, insights in insights_by_domain.items():
            domain_confidence[domain] = np.mean([i.confidence for i in insights])
        
        return {
            'total_insights': len(recent_insights),
            'insights_by_domain': {domain: len(insights) for domain, insights in insights_by_domain.items()},
            'average_confidence': np.mean([i.confidence for i in recent_insights]) if recent_insights else 0,
            'domain_confidence': domain_confidence,
            'top_recommendations': self._get_top_recommendations(recent_insights),
            'system_metrics': self.system_metrics
        }
    
    def _get_top_recommendations(self, insights: List[IntrospectiveInsight]) -> List[str]:
        """Get top actionable recommendations from insights"""
        all_recommendations = []
        for insight in insights:
            all_recommendations.extend(insight.actionable_recommendations)
        
        # Count recommendation frequency
        recommendation_counts = defaultdict(int)
        for rec in all_recommendations:
            recommendation_counts[rec] += 1
        
        # Return top 5 most frequent recommendations
        return sorted(recommendation_counts.keys(), 
                     key=lambda x: recommendation_counts[x], reverse=True)[:5]
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info(f"Introspection Agent {self.agent_id} cleaned up")
