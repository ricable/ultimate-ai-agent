"""UAP Edge Computing Backend Services

This module provides edge computing capabilities for the UAP platform,
including WebAssembly runtime integration, edge deployment, and mobile
synchronization services.
"""

from .edge_manager import EdgeManager
from .edge_api import EdgeAPI
from .deployment_service import EdgeDeploymentService
from .sync_service import EdgeSyncService
from .mobile_bridge import MobileBridge

__all__ = [
    'EdgeManager',
    'EdgeAPI',
    'EdgeDeploymentService',
    'EdgeSyncService',
    'MobileBridge',
]