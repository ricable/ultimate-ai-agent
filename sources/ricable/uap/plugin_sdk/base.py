# File: plugin_sdk/base.py
"""
Base classes for plugin development.

This module re-exports the core plugin base classes with additional
development utilities and helpers.
"""

# Re-export core plugin base classes
from ..backend.plugins.plugin_base import (
    PluginBase,
    IntegrationPlugin,
    ProcessorPlugin, 
    AIAgentPlugin,
    WorkflowPlugin,
    PluginManifest,
    PluginConfig,
    PluginContext,
    PluginResponse,
    PluginEvent,
    PluginStatus,
    PluginType,
    PluginError
)

# Export for convenience
__all__ = [
    'PluginBase',
    'IntegrationPlugin',
    'ProcessorPlugin',
    'AIAgentPlugin', 
    'WorkflowPlugin',
    'PluginManifest',
    'PluginConfig',
    'PluginContext',
    'PluginResponse',
    'PluginEvent',
    'PluginStatus',
    'PluginType',
    'PluginError'
]