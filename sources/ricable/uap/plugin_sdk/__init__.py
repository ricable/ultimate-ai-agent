# UAP Plugin SDK
"""
Plugin Software Development Kit for UAP.

This SDK provides tools and utilities for developing UAP plugins:
- Base classes for different plugin types
- Development utilities and helpers
- Testing framework for plugins
- Plugin packaging tools
- Documentation generators
"""

from .base import (
    PluginBase, IntegrationPlugin, ProcessorPlugin, 
    AIAgentPlugin, WorkflowPlugin
)
from .manifest import ManifestBuilder, PluginManifest
from .testing import PluginTestCase, MockContext
from .packaging import PluginPackager
from .decorators import action, permission_required, rate_limit
from .utils import PluginLogger, ConfigValidator

__version__ = "1.0.0"
__all__ = [
    # Base classes
    'PluginBase',
    'IntegrationPlugin', 
    'ProcessorPlugin',
    'AIAgentPlugin',
    'WorkflowPlugin',
    
    # Development tools
    'ManifestBuilder',
    'PluginManifest',
    'PluginTestCase',
    'MockContext',
    'PluginPackager',
    
    # Decorators
    'action',
    'permission_required',
    'rate_limit',
    
    # Utilities
    'PluginLogger',
    'ConfigValidator'
]