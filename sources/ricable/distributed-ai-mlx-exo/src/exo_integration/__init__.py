"""
Exo Integration Package
Integration layer between MLX distributed and Exo P2P framework
"""

from .cluster_manager import ExoClusterManager, ExoNodeSpec, create_cluster_manager, auto_detect_node_id

__version__ = "1.0.0"
__all__ = ["ExoClusterManager", "ExoNodeSpec", "create_cluster_manager", "auto_detect_node_id"]