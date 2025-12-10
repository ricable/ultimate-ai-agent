# File: backend/ai/__init__.py
"""
AI-Powered Platform Intelligence System

This module contains the core AI components for platform intelligence,
including predictive modeling, automated optimization, and self-healing capabilities.
"""

from .platform_ai import (
    PlatformAI,
    platform_ai,
    PlatformIntelligenceReport,
    AIDecision,
    IntelligenceLevel,
    OptimizationStrategy,
    PlatformState,
    generate_intelligence_report,
    execute_self_healing,
    provide_decision_feedback
)

__all__ = [
    'PlatformAI',
    'platform_ai',
    'PlatformIntelligenceReport',
    'AIDecision',
    'IntelligenceLevel',
    'OptimizationStrategy',
    'PlatformState',
    'generate_intelligence_report',
    'execute_self_healing',
    'provide_decision_feedback'
]