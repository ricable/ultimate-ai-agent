"""
MLX Distributed Package
Distributed computing support for MLX on Apple Silicon clusters
"""

from .config import MLXDistributedConfig, NodeConfig, ClusterConfig

__version__ = "1.0.0"
__all__ = ["MLXDistributedConfig", "NodeConfig", "ClusterConfig"]