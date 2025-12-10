# UAP Third-Party Integrations Module
"""
Third-party integrations and API marketplace for UAP.

This module provides:
- OAuth2 provider for secure third-party access
- Integration framework for external tools
- Webhook system for real-time communication
- API marketplace for discovering and managing integrations
"""

from .base import IntegrationBase, IntegrationError, IntegrationStatus
from .oauth_provider import OAuth2Provider, OAuth2Error
from .manager import IntegrationManager
from .registry import IntegrationRegistry

__all__ = [
    'IntegrationBase',
    'IntegrationError', 
    'IntegrationStatus',
    'OAuth2Provider',
    'OAuth2Error',
    'IntegrationManager',
    'IntegrationRegistry'
]