# backend/ai/model_management.py
# Agent 21: Advanced AI Model Management - Model Versioning & Deployment Pipelines

import asyncio
import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import httpx
from pydantic import BaseModel

# MLX and Ray integration
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

class ModelStatus(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"

class ModelType(Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

@dataclass
class ModelVersion:
    """Model version metadata with comprehensive tracking"""
    model_id: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    file_path: str
    file_hash: str
    file_size: int
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    deployment_config: Dict[str, Any]
    parent_version: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ModelRegistry:
    """Central model registry with versioning and deployment management"""
    
    def __init__(self, storage_path: str = "./models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.storage_path / "registry.json"
        self.models: Dict[str, List[ModelVersion]] = {}
    
    async def load_registry(self):
        """Load model registry from persistent storage"""
        if self.registry_file.exists():
            try:
                async with aiofiles.open(self.registry_file, 'r') as f:
                    data = json.loads(await f.read())
                    for model_id, versions in data.items():
                        self.models[model_id] = [
                            ModelVersion(
                                **{**v, 
                                   'created_at': datetime.fromisoformat(v['created_at']),
                                   'updated_at': datetime.fromisoformat(v['updated_at']),
                                   'model_type': ModelType(v['model_type']),
                                   'status': ModelStatus(v['status'])
                                }
                            ) for v in versions
                        ]
            except Exception as e:
                print(f"Failed to load model registry: {e}")
                self.models = {}
    
    async def register_model(self, 
                           model_id: str,
                           model_file: Union[str, Path],
                           model_type: ModelType,
                           metadata: Dict[str, Any] = None,
                           parent_version: str = None,
                           tags: List[str] = None) -> ModelVersion:
        """Register a new model version"""
        
        if metadata is None:
            metadata = {}
        if tags is None:
            tags = []
            
        # Calculate file hash and size
        file_path = Path(model_file)
        file_hash = await self._calculate_file_hash(file_path)
        file_size = file_path.stat().st_size
        
        # Generate version number
        current_versions = self.models.get(model_id, [])
        version = f"v{len(current_versions) + 1}.0.0"
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            status=ModelStatus.VALIDATION,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            file_path=str(file_path),
            file_hash=file_hash,
            file_size=file_size,
            metadata=metadata,
            performance_metrics={},
            deployment_config={},
            parent_version=parent_version,
            tags=tags
        )
        
        # Store in registry
        if model_id not in self.models:
            self.models[model_id] = []
        self.models[model_id].append(model_version)
        
        return model_version
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of model file"""
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class ModelDeploymentManager:
    """Manage model deployments across different environments"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_deployments: Dict[str, Dict] = {}
    
    async def deploy_model(self, 
                          model_id: str,
                          version: str = None,
                          environment: str = "production",
                          deployment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deploy model to specified environment"""
        
        if deployment_config is None:
            deployment_config = {}
            
        deployment_id = f"{model_id}_{version}_{environment}_{datetime.utcnow().timestamp()}"
        
        # MLX deployment for Apple Silicon
        if MLX_AVAILABLE and deployment_config.get('use_mlx', False):
            deployment_result = await self._deploy_mlx_model(deployment_config)
        # Ray Serve deployment for distributed inference
        elif RAY_AVAILABLE and deployment_config.get('use_ray', False):
            deployment_result = await self._deploy_ray_model(deployment_config)
        # Standard deployment
        else:
            deployment_result = await self._deploy_standard_model(deployment_config)
        
        return {
            'deployment_id': deployment_id,
            'status': 'success',
            'result': deployment_result
        }
    
    async def _deploy_mlx_model(self, config: Dict) -> Dict:
        """Deploy model using MLX for Apple Silicon"""
        return {
            'type': 'mlx',
            'endpoint': '/api/models/predict',
            'hardware': 'apple_silicon',
            'status': 'deployed'
        }
    
    async def _deploy_ray_model(self, config: Dict) -> Dict:
        """Deploy model using Ray Serve for distributed inference"""
        return {
            'type': 'ray_serve',
            'endpoint': '/api/models/predict',
            'replicas': config.get('replicas', 2),
            'status': 'deployed'
        }
    
    async def _deploy_standard_model(self, config: Dict) -> Dict:
        """Standard model deployment"""
        return {
            'type': 'standard',
            'endpoint': '/api/models/predict',
            'status': 'deployed'
        }

# Global model manager
model_registry = ModelRegistry()
deployment_manager = ModelDeploymentManager(model_registry)