# UAP Plugin Marketplace
"""
Marketplace API for plugin discovery, installation, and management.

This module provides:
- Plugin marketplace browsing
- Plugin installation and configuration
- Integration with the existing integration system
- Plugin development tools
"""

from .api import MarketplaceAPI
from .store import PluginStore
from .installer import PluginInstaller

__all__ = [
    'MarketplaceAPI',
    'PluginStore', 
    'PluginInstaller'
]