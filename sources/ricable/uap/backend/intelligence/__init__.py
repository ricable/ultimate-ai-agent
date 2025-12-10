# File: backend/intelligence/__init__.py
"""
AI-Powered Platform Intelligence System for UAP

This module implements advanced AI capabilities for platform optimization,
resource allocation, performance tuning, and self-healing capabilities.
"""

from .usage_prediction import PlatformUsagePredictor
from .resource_allocation import IntelligentResourceAllocator
from .performance_tuning import AutomatedPerformanceTuner
from .predictive_maintenance import PredictiveMaintenanceSystem
from .platform_intelligence import PlatformIntelligenceOrchestrator

__all__ = [
    'PlatformUsagePredictor',
    'IntelligentResourceAllocator', 
    'AutomatedPerformanceTuner',
    'PredictiveMaintenanceSystem',
    'PlatformIntelligenceOrchestrator'
]