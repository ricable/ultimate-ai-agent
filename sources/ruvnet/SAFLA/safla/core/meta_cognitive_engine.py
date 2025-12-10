"""
Meta-Cognitive Engine for SAFLA - Self-Aware Feedback Loop Algorithm

This module serves as a compatibility layer that imports components from the new
modular metacognitive package structure.

The actual implementation has been refactored into:
- safla/core/metacognitive/engine.py (400 lines) - Main coordination engine
- safla/core/metacognitive/awareness.py (300 lines) - Self-awareness capabilities
- safla/core/metacognitive/strategies.py (400 lines) - Strategy selection system
- safla/core/metacognitive/learning.py (400 lines) - Adaptation and learning
- safla/core/metacognitive/metrics.py (169 lines) - Goals and performance monitoring

Usage:
    from safla.core.meta_cognitive_engine import MetaCognitiveEngine
    # or
    from safla.core.metacognitive import MetaCognitiveEngine
    
    engine = MetaCognitiveEngine()
    await engine.start()
"""

# Import all components from the new modular structure
from safla.core.metacognitive import (
    MetaCognitiveEngine,
    SelfAwarenessModule,
    SystemState,
    StrategySelector,
    Strategy,
    AdaptationEngine,
    AdaptationResult,
    PerformanceMonitor,
    PerformanceMetrics,
    Goal,
    GoalManager
)

# Re-export for backward compatibility
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