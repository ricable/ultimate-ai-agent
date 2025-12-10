"""
Meta-Cognitive Engine for SAFLA - Self-Aware Feedback Loop Algorithm

This package implements the core meta-cognitive capabilities including:
- Self-Awareness Module: System state monitoring and self-reflection
- Goal Management: Dynamic goal setting, tracking, and adaptation
- Strategy Selection: Context-aware strategy selection and optimization
- Performance Monitoring: Real-time performance tracking and analysis
- Adaptation Engine: Continuous learning and self-modification capabilities

The Meta-Cognitive Engine serves as the central coordination layer that enables
SAFLA to be self-aware, adaptive, and continuously improving.
"""

from .engine import MetaCognitiveEngine, PerformanceMonitor
from .awareness import SelfAwarenessModule, SystemState
from .strategies import StrategySelector, Strategy
from .learning import AdaptationEngine, AdaptationResult
from .metrics import Goal, PerformanceMetrics, GoalManager

__all__ = [
    'MetaCognitiveEngine',
    'SelfAwarenessModule',
    'SystemState',
    'StrategySelector',
    'Strategy',
    'AdaptationEngine',
    'AdaptationResult',
    'PerformanceMonitor',
    'PerformanceMetrics',
    'Goal',
    'GoalManager'
]