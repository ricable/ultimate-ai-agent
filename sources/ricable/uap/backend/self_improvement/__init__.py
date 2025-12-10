"""
Agent 40: Self-Improving AI Metacognition System - Self-Improvement Utilities
Additional utilities and tools for the self-improvement engine.
"""

# Import core self-improvement components from metacognition module
from ..metacognition.self_improvement import (
    SelfImprovementEngine,
    SafetyValidator,
    ImprovementProposal,
    ImprovementExecution,
    ImprovementCategory,
    ImprovementRisk,
    SafetyConstraint,
    SafetyConstraintType
)

# Additional utility imports
from .improvement_analytics import ImprovementAnalytics
from .learning_optimizer import MetaLearningOptimizer
from .recursive_enhancer import RecursiveEnhancer

__all__ = [
    'SelfImprovementEngine',
    'SafetyValidator', 
    'ImprovementProposal',
    'ImprovementExecution',
    'ImprovementCategory',
    'ImprovementRisk',
    'SafetyConstraint',
    'SafetyConstraintType',
    'ImprovementAnalytics',
    'MetaLearningOptimizer',
    'RecursiveEnhancer'
]
