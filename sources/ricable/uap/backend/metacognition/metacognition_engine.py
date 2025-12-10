"""
Agent 40: Self-Improving AI Metacognition System - Core Engine
Implements self-reflective AI systems with metacognitive awareness and recursive improvement.
"""

import asyncio
import json
import logging
import time
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from uuid import uuid4
import hashlib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MetacognitionState(Enum):
    """States of metacognitive processing"""
    OBSERVING = "observing"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    LEARNING = "learning"


class ImprovementType(Enum):
    """Types of self-improvement operations"""
    PARAMETER_TUNING = "parameter_tuning"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    KNOWLEDGE_EXPANSION = "knowledge_expansion"
    STRATEGY_REFINEMENT = "strategy_refinement"
    PERFORMANCE_ENHANCEMENT = "performance_enhancement"
    SAFETY_REINFORCEMENT = "safety_reinforcement"


class ReflectionLevel(Enum):
    """Levels of metacognitive reflection"""
    SURFACE = "surface"  # Basic performance metrics
    INTERMEDIATE = "intermediate"  # Strategy analysis
    DEEP = "deep"  # Fundamental reasoning patterns
    META = "meta"  # Reflection on reflection itself


@dataclass
class MetacognitiveObservation:
    """Represents an observation made by the metacognitive system"""
    id: str
    timestamp: datetime
    observation_type: str
    data: Dict[str, Any]
    confidence: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class SelfReflection:
    """Represents a self-reflective analysis"""
    id: str
    timestamp: datetime
    level: ReflectionLevel
    trigger: str
    analysis: Dict[str, Any]
    insights: List[str]
    improvement_opportunities: List[str]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class ImprovementPlan:
    """Represents a plan for self-improvement"""
    id: str
    timestamp: datetime
    improvement_type: ImprovementType
    target_metrics: Dict[str, float]
    actions: List[Dict[str, Any]]
    expected_impact: Dict[str, float]
    risk_assessment: Dict[str, float]
    safety_constraints: List[str]
    execution_timeline: Dict[str, datetime]
    metadata: Dict[str, Any]


@dataclass
class MetacognitiveState:
    """Represents the current state of metacognitive processing"""
    current_state: MetacognitionState
    confidence: float
    focus_areas: List[str]
    active_reflections: List[str]
    pending_improvements: List[str]
    last_update: datetime
    processing_capacity: Dict[str, float]
    metadata: Dict[str, Any]


class MetacognitiveModule(ABC):
    """Abstract base class for metacognitive modules"""
    
    def __init__(self):
        self.module_id = str(uuid4())
        self.is_active = False
        self.performance_history = []
        self.last_execution = None
    
    @abstractmethod
    async def process(self, observation: MetacognitiveObservation) -> Dict[str, Any]:
        """Process a metacognitive observation"""
        pass
    
    @abstractmethod
    async def reflect(self, context: Dict[str, Any]) -> SelfReflection:
        """Perform self-reflection"""
        pass
    
    @abstractmethod
    async def improve(self, reflection: SelfReflection) -> ImprovementPlan:
        """Generate improvement plan based on reflection"""
        pass


class PerformanceMonitoringModule(MetacognitiveModule):
    """Monitors system performance and generates observations"""
    
    def __init__(self):
        super().__init__()
        self.metrics_history = {}
        self.baseline_metrics = {}
        self.performance_thresholds = {
            'response_time': 2.0,  # seconds
            'accuracy': 0.95,
            'efficiency': 0.90,
            'user_satisfaction': 0.85
        }
    
    async def process(self, observation: MetacognitiveObservation) -> Dict[str, Any]:
        """Process performance-related observations"""
        if observation.observation_type == "performance_metric":
            metric_name = observation.data.get('metric_name')
            metric_value = observation.data.get('metric_value')
            
            # Store historical data
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            self.metrics_history[metric_name].append({
                'timestamp': observation.timestamp,
                'value': metric_value,
                'context': observation.context
            })
            
            # Detect anomalies
            anomaly_score = await self._detect_performance_anomaly(metric_name, metric_value)
            
            # Generate insights
            insights = await self._generate_performance_insights(metric_name)
            
            return {
                'anomaly_score': anomaly_score,
                'insights': insights,
                'trend': await self._calculate_trend(metric_name),
                'recommendations': await self._generate_recommendations(metric_name)
            }
        
        return {}
    
    async def reflect(self, context: Dict[str, Any]) -> SelfReflection:
        """Reflect on overall performance patterns"""
        analysis = {
            'overall_performance': await self._calculate_overall_performance(),
            'performance_trends': await self._analyze_performance_trends(),
            'bottlenecks': await self._identify_bottlenecks(),
            'improvement_areas': await self._identify_improvement_areas()
        }
        
        insights = [
            f"Overall performance score: {analysis['overall_performance']:.2f}",
            f"Identified {len(analysis['bottlenecks'])} performance bottlenecks",
            f"Found {len(analysis['improvement_areas'])} areas for improvement"
        ]
        
        improvement_opportunities = [
            f"Optimize {area['name']}: Expected {area['impact']:.1%} improvement"
            for area in analysis['improvement_areas'][:3]
        ]
        
        return SelfReflection(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            level=ReflectionLevel.INTERMEDIATE,
            trigger="performance_analysis",
            analysis=analysis,
            insights=insights,
            improvement_opportunities=improvement_opportunities,
            confidence=0.85,
            metadata={'module': 'performance_monitoring'}
        )
    
    async def improve(self, reflection: SelfReflection) -> ImprovementPlan:
        """Generate performance improvement plan"""
        analysis = reflection.analysis
        
        actions = []
        for area in analysis.get('improvement_areas', [])[:3]:
            actions.append({
                'action_type': 'optimize_performance',
                'target': area['name'],
                'method': area['optimization_method'],
                'expected_improvement': area['impact'],
                'timeline': 'immediate'
            })
        
        target_metrics = {
            'response_time': max(0.5, analysis['overall_performance'] * 0.8),
            'efficiency': min(1.0, analysis['overall_performance'] * 1.2),
            'user_satisfaction': min(1.0, analysis['overall_performance'] * 1.1)
        }
        
        return ImprovementPlan(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            improvement_type=ImprovementType.PERFORMANCE_ENHANCEMENT,
            target_metrics=target_metrics,
            actions=actions,
            expected_impact={'overall_performance': 0.15},
            risk_assessment={'performance_degradation': 0.1, 'system_instability': 0.05},
            safety_constraints=['maintain_minimum_performance', 'gradual_rollout'],
            execution_timeline={
                'start': datetime.utcnow(),
                'completion': datetime.utcnow() + timedelta(hours=2)
            },
            metadata={'module': 'performance_monitoring'}
        )
    
    async def _detect_performance_anomaly(self, metric_name: str, metric_value: float) -> float:
        """Detect performance anomalies using statistical analysis"""
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < 10:
            return 0.0
        
        history = [entry['value'] for entry in self.metrics_history[metric_name][-50:]]
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:
            return 0.0
        
        z_score = abs((metric_value - mean) / std)
        return min(1.0, z_score / 3.0)  # Normalize to 0-1 range
    
    async def _generate_performance_insights(self, metric_name: str) -> List[str]:
        """Generate insights about performance metrics"""
        insights = []
        
        if metric_name in self.metrics_history and len(self.metrics_history[metric_name]) >= 5:
            recent_values = [entry['value'] for entry in self.metrics_history[metric_name][-5:]]
            avg_recent = np.mean(recent_values)
            
            if metric_name in self.performance_thresholds:
                threshold = self.performance_thresholds[metric_name]
                if avg_recent < threshold:
                    insights.append(f"{metric_name} is below optimal threshold ({avg_recent:.2f} < {threshold})")
                else:
                    insights.append(f"{metric_name} is performing well ({avg_recent:.2f} >= {threshold})")
        
        return insights
    
    async def _calculate_trend(self, metric_name: str) -> str:
        """Calculate performance trend"""
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < 5:
            return "insufficient_data"
        
        recent_values = [entry['value'] for entry in self.metrics_history[metric_name][-10:]]
        
        # Simple linear trend calculation
        if len(recent_values) >= 2:
            slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
        
        return "stable"
    
    async def _generate_recommendations(self, metric_name: str) -> List[str]:
        """Generate recommendations for performance improvement"""
        recommendations = []
        trend = await self._calculate_trend(metric_name)
        
        if trend == "declining":
            recommendations.append(f"Investigate {metric_name} degradation and implement corrective measures")
        elif trend == "stable" and metric_name in self.performance_thresholds:
            recent_avg = np.mean([entry['value'] for entry in self.metrics_history[metric_name][-5:]])
            if recent_avg < self.performance_thresholds[metric_name]:
                recommendations.append(f"Optimize {metric_name} to reach target threshold")
        
        return recommendations
    
    async def _calculate_overall_performance(self) -> float:
        """Calculate overall system performance score"""
        scores = []
        for metric_name, threshold in self.performance_thresholds.items():
            if metric_name in self.metrics_history and self.metrics_history[metric_name]:
                recent_avg = np.mean([entry['value'] for entry in self.metrics_history[metric_name][-5:]])
                score = min(1.0, recent_avg / threshold)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    async def _analyze_performance_trends(self) -> Dict[str, str]:
        """Analyze trends across all performance metrics"""
        trends = {}
        for metric_name in self.metrics_history.keys():
            trends[metric_name] = await self._calculate_trend(metric_name)
        return trends
    
    async def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        for metric_name, threshold in self.performance_thresholds.items():
            if metric_name in self.metrics_history and self.metrics_history[metric_name]:
                recent_avg = np.mean([entry['value'] for entry in self.metrics_history[metric_name][-5:]])
                if recent_avg < threshold * 0.8:  # 20% below threshold
                    bottlenecks.append({
                        'metric': metric_name,
                        'current_value': recent_avg,
                        'threshold': threshold,
                        'severity': (threshold - recent_avg) / threshold
                    })
        
        return sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)
    
    async def _identify_improvement_areas(self) -> List[Dict[str, Any]]:
        """Identify areas for improvement"""
        improvement_areas = []
        
        for metric_name, threshold in self.performance_thresholds.items():
            if metric_name in self.metrics_history and self.metrics_history[metric_name]:
                recent_avg = np.mean([entry['value'] for entry in self.metrics_history[metric_name][-5:]])
                if recent_avg < threshold:
                    potential_improvement = (threshold - recent_avg) / threshold
                    improvement_areas.append({
                        'name': metric_name,
                        'current_value': recent_avg,
                        'target_value': threshold,
                        'impact': potential_improvement,
                        'optimization_method': self._suggest_optimization_method(metric_name)
                    })
        
        return sorted(improvement_areas, key=lambda x: x['impact'], reverse=True)
    
    def _suggest_optimization_method(self, metric_name: str) -> str:
        """Suggest optimization method for a metric"""
        optimization_methods = {
            'response_time': 'caching_and_parallelization',
            'accuracy': 'model_fine_tuning',
            'efficiency': 'algorithm_optimization',
            'user_satisfaction': 'ux_improvement'
        }
        return optimization_methods.get(metric_name, 'general_optimization')


class MetacognitionEngine:
    """Main engine for metacognitive processing"""
    
    def __init__(self):
        self.engine_id = str(uuid4())
        self.modules: Dict[str, MetacognitiveModule] = {}
        self.observations_buffer: List[MetacognitiveObservation] = []
        self.reflections_history: List[SelfReflection] = []
        self.improvement_plans: List[ImprovementPlan] = []
        self.current_state = MetacognitiveState(
            current_state=MetacognitionState.OBSERVING,
            confidence=0.5,
            focus_areas=[],
            active_reflections=[],
            pending_improvements=[],
            last_update=datetime.utcnow(),
            processing_capacity={'cpu': 0.5, 'memory': 0.3, 'attention': 0.7},
            metadata={}
        )
        
        # Initialize core modules
        self.modules['performance_monitoring'] = PerformanceMonitoringModule()
        
        # Metacognitive processing parameters
        self.observation_window = timedelta(minutes=5)
        self.reflection_interval = timedelta(minutes=15)
        self.improvement_interval = timedelta(hours=1)
        self.last_reflection = datetime.utcnow()
        self.last_improvement = datetime.utcnow()
    
    async def initialize(self) -> bool:
        """Initialize the metacognition engine"""
        try:
            logger.info(f"Initializing Metacognition Engine {self.engine_id}")
            
            # Initialize all modules
            for module_name, module in self.modules.items():
                module.is_active = True
                logger.info(f"Activated metacognitive module: {module_name}")
            
            # Start metacognitive processing loop
            asyncio.create_task(self._metacognitive_processing_loop())
            
            logger.info("Metacognition Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Metacognition Engine: {e}")
            return False
    
    async def add_observation(self, observation_type: str, data: Dict[str, Any], 
                            context: Dict[str, Any] = None, confidence: float = 1.0) -> str:
        """Add a new observation to the metacognitive system"""
        observation = MetacognitiveObservation(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            observation_type=observation_type,
            data=data,
            confidence=confidence,
            context=context or {},
            metadata={'source': 'external'}
        )
        
        self.observations_buffer.append(observation)
        
        # Trigger immediate processing if high-priority observation
        if observation_type in ['critical_error', 'performance_anomaly', 'safety_violation']:
            await self._process_observation(observation)
        
        # Maintain buffer size
        if len(self.observations_buffer) > 1000:
            self.observations_buffer = self.observations_buffer[-500:]
        
        return observation.id
    
    async def _process_observation(self, observation: MetacognitiveObservation) -> Dict[str, Any]:
        """Process a single observation through all relevant modules"""
        results = {}
        
        for module_name, module in self.modules.items():
            if module.is_active:
                try:
                    result = await module.process(observation)
                    results[module_name] = result
                    
                    # Update module performance
                    module.last_execution = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(f"Error processing observation in module {module_name}: {e}")
                    results[module_name] = {'error': str(e)}
        
        return results
    
    async def trigger_reflection(self, level: ReflectionLevel = ReflectionLevel.INTERMEDIATE) -> List[str]:
        """Trigger self-reflection across all modules"""
        reflection_ids = []
        
        for module_name, module in self.modules.items():
            if module.is_active:
                try:
                    reflection = await module.reflect({
                        'level': level,
                        'trigger_time': datetime.utcnow(),
                        'recent_observations': self.observations_buffer[-50:],
                        'system_state': self.current_state
                    })
                    
                    self.reflections_history.append(reflection)
                    reflection_ids.append(reflection.id)
                    
                except Exception as e:
                    logger.error(f"Error during reflection in module {module_name}: {e}")
        
        # Update system state
        self.current_state.current_state = MetacognitionState.REFLECTING
        self.current_state.active_reflections = reflection_ids
        self.current_state.last_update = datetime.utcnow()
        self.last_reflection = datetime.utcnow()
        
        # Maintain reflection history size
        if len(self.reflections_history) > 100:
            self.reflections_history = self.reflections_history[-50:]
        
        return reflection_ids
    
    async def generate_improvement_plans(self) -> List[str]:
        """Generate improvement plans based on recent reflections"""
        plan_ids = []
        
        # Get recent reflections for improvement planning
        recent_reflections = [r for r in self.reflections_history 
                            if r.timestamp > datetime.utcnow() - timedelta(hours=2)]
        
        for reflection in recent_reflections[-5:]:  # Process last 5 reflections
            if reflection.improvement_opportunities:
                for module_name, module in self.modules.items():
                    if module.is_active:
                        try:
                            plan = await module.improve(reflection)
                            self.improvement_plans.append(plan)
                            plan_ids.append(plan.id)
                            
                        except Exception as e:
                            logger.error(f"Error generating improvement plan in module {module_name}: {e}")
        
        # Update system state
        self.current_state.current_state = MetacognitionState.PLANNING
        self.current_state.pending_improvements = plan_ids
        self.current_state.last_update = datetime.utcnow()
        self.last_improvement = datetime.utcnow()
        
        # Maintain improvement plans history
        if len(self.improvement_plans) > 50:
            self.improvement_plans = self.improvement_plans[-25:]
        
        return plan_ids
    
    async def _metacognitive_processing_loop(self):
        """Main metacognitive processing loop"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Process pending observations
                if self.observations_buffer:
                    pending_observations = [obs for obs in self.observations_buffer 
                                          if obs.timestamp > current_time - self.observation_window]
                    
                    for observation in pending_observations[-10:]:  # Process last 10
                        await self._process_observation(observation)
                
                # Trigger periodic reflection
                if current_time - self.last_reflection > self.reflection_interval:
                    await self.trigger_reflection()
                
                # Generate improvement plans
                if current_time - self.last_improvement > self.improvement_interval:
                    await self.generate_improvement_plans()
                
                # Update system state
                await self._update_system_state()
                
                # Sleep before next iteration
                await asyncio.sleep(30)  # 30-second processing cycle
                
            except Exception as e:
                logger.error(f"Error in metacognitive processing loop: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _update_system_state(self):
        """Update the current metacognitive system state"""
        # Calculate processing capacity utilization
        active_modules = sum(1 for module in self.modules.values() if module.is_active)
        total_modules = len(self.modules)
        
        # Update processing capacity
        self.current_state.processing_capacity = {
            'cpu': min(1.0, active_modules / max(total_modules, 1)),
            'memory': min(1.0, len(self.observations_buffer) / 1000),
            'attention': min(1.0, len(self.current_state.active_reflections) / 10)
        }
        
        # Update focus areas based on recent observations
        recent_observations = [obs for obs in self.observations_buffer 
                             if obs.timestamp > datetime.utcnow() - timedelta(hours=1)]
        
        focus_areas = {}
        for obs in recent_observations:
            obs_type = obs.observation_type
            focus_areas[obs_type] = focus_areas.get(obs_type, 0) + 1
        
        # Top 3 focus areas
        self.current_state.focus_areas = sorted(focus_areas.keys(), 
                                              key=lambda x: focus_areas[x], reverse=True)[:3]
        
        # Calculate overall confidence
        recent_reflections = [r for r in self.reflections_history 
                            if r.timestamp > datetime.utcnow() - timedelta(hours=2)]
        
        if recent_reflections:
            avg_confidence = np.mean([r.confidence for r in recent_reflections])
            self.current_state.confidence = avg_confidence
        
        self.current_state.last_update = datetime.utcnow()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'engine_id': self.engine_id,
            'current_state': asdict(self.current_state),
            'modules': {
                name: {
                    'active': module.is_active,
                    'last_execution': module.last_execution.isoformat() if module.last_execution else None,
                    'performance_entries': len(module.performance_history)
                }
                for name, module in self.modules.items()
            },
            'statistics': {
                'total_observations': len(self.observations_buffer),
                'total_reflections': len(self.reflections_history),
                'total_improvement_plans': len(self.improvement_plans),
                'recent_observations': len([obs for obs in self.observations_buffer 
                                          if obs.timestamp > datetime.utcnow() - timedelta(hours=1)])
            },
            'performance_metrics': {
                'observation_processing_rate': len(self.observations_buffer) / max(1, 
                    (datetime.utcnow() - self.observations_buffer[0].timestamp).total_seconds() / 3600
                ) if self.observations_buffer else 0,
                'reflection_frequency': len(self.reflections_history) / max(1,
                    (datetime.utcnow() - self.reflections_history[0].timestamp).total_seconds() / 3600
                ) if self.reflections_history else 0
            }
        }
    
    async def get_recent_insights(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent insights from reflections"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_reflections = [r for r in self.reflections_history if r.timestamp > cutoff_time]
        
        insights = []
        for reflection in recent_reflections:
            insights.extend([
                {
                    'insight': insight,
                    'timestamp': reflection.timestamp,
                    'level': reflection.level.value,
                    'confidence': reflection.confidence,
                    'source': reflection.metadata.get('module', 'unknown')
                }
                for insight in reflection.insights
            ])
        
        return sorted(insights, key=lambda x: x['timestamp'], reverse=True)
    
    async def cleanup(self):
        """Clean up resources"""
        for module in self.modules.values():
            module.is_active = False
        
        logger.info(f"Metacognition Engine {self.engine_id} cleaned up")
