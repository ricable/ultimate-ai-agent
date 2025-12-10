# backend/ml/model_manager.py
# Advanced Model Lifecycle Management System

import asyncio
import json
import uuid
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import numpy as np

# Integrations
from ..ai.model_management import ModelRegistry, ModelVersion, ModelType, ModelStatus
from ..ai.ab_testing import ABTestManager, get_ab_test_manager
from ..distributed.ray_manager import submit_distributed_task
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.prometheus_metrics import record_pipeline_event

class ModelLifecycleStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class ModelMetric(Enum):
    """Model evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    LOSS = "loss"
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    ROUGE = "rouge"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    bleu: Optional[float] = None
    rouge: Optional[float] = None
    custom_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}

@dataclass
class ModelDeployment:
    """Model deployment configuration"""
    deployment_id: str
    model_id: str
    version: str
    stage: ModelLifecycleStage
    endpoint_url: str
    traffic_percentage: float
    auto_scaling: bool
    min_replicas: int
    max_replicas: int
    deployment_time: datetime
    status: str
    health_check_url: Optional[str] = None
    monitoring_enabled: bool = True
    rollback_version: Optional[str] = None

@dataclass
class ModelApproval:
    """Model approval workflow"""
    approval_id: str
    model_id: str
    version: str
    source_stage: ModelLifecycleStage
    target_stage: ModelLifecycleStage
    approved_by: str
    approval_time: datetime
    approval_notes: str
    required_approvers: List[str]
    current_approvers: List[str]
    status: str  # pending, approved, rejected

class ModelManager:
    """
    Advanced Model Lifecycle Management System.
    
    Provides comprehensive model management including:
    - Model versioning and registry
    - Deployment management
    - A/B testing integration
    - Approval workflows
    - Performance monitoring
    - Rollback capabilities
    """
    
    def __init__(self, storage_path: str = "./models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_registry = ModelRegistry(str(self.storage_path))
        self.ab_test_manager = get_ab_test_manager()
        
        # Model lifecycle tracking
        self.deployments: Dict[str, ModelDeployment] = {}
        self.approvals: Dict[str, ModelApproval] = {}
        self.metrics_history: Dict[str, List[Dict]] = {}
        
        # Configuration
        self.approval_workflows = {
            ModelLifecycleStage.STAGING: ["ml_engineer"],
            ModelLifecycleStage.PRODUCTION: ["ml_engineer", "data_scientist", "product_manager"]
        }
    
    async def initialize(self) -> bool:
        """Initialize the model manager"""
        try:
            await self.model_registry.load_registry()
            await self._load_deployments()
            await self._load_approvals()
            
            # Start monitoring
            asyncio.create_task(self._start_monitoring())
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Model Manager initialized",
                EventType.AGENT,
                {"storage_path": str(self.storage_path)},
                "model_manager"
            )
            
            return True
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to initialize Model Manager: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "model_manager"
            )
            return False
    
    async def register_model(self,
                           model_id: str,
                           model_file: Union[str, Path],
                           model_type: ModelType,
                           stage: ModelLifecycleStage = ModelLifecycleStage.DEVELOPMENT,
                           metadata: Dict[str, Any] = None,
                           metrics: ModelMetrics = None,
                           tags: List[str] = None) -> ModelVersion:
        """Register a new model version"""
        
        # Register with model registry
        model_version = await self.model_registry.register_model(
            model_id=model_id,
            model_file=model_file,
            model_type=model_type,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Add lifecycle stage to metadata
        model_version.metadata["lifecycle_stage"] = stage.value
        
        # Store metrics if provided
        if metrics:
            model_version.metadata["metrics"] = asdict(metrics)
            await self._store_metrics(model_id, model_version.version, metrics)
        
        # Log model registration
        uap_logger.log_event(
            LogLevel.INFO,
            f"Model registered: {model_id} v{model_version.version}",
            EventType.AGENT,
            {
                "model_id": model_id,
                "version": model_version.version,
                "stage": stage.value,
                "model_type": model_type.value
            },
            "model_manager"
        )
        
        return model_version
    
    async def deploy_model(self,
                          model_id: str,
                          version: str,
                          stage: ModelLifecycleStage,
                          traffic_percentage: float = 100.0,
                          auto_scaling: bool = True,
                          min_replicas: int = 1,
                          max_replicas: int = 10,
                          approval_required: bool = True) -> str:
        """Deploy a model to specified stage"""
        
        # Check if approval is required
        if approval_required and stage in self.approval_workflows:
            approval_id = await self._create_approval_workflow(
                model_id, version, ModelLifecycleStage.DEVELOPMENT, stage
            )
            return f"approval_required:{approval_id}"
        
        # Create deployment
        deployment_id = str(uuid.uuid4())
        endpoint_url = f"/api/models/{model_id}/v{version}/predict"
        
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            model_id=model_id,
            version=version,
            stage=stage,
            endpoint_url=endpoint_url,
            traffic_percentage=traffic_percentage,
            auto_scaling=auto_scaling,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            deployment_time=datetime.utcnow(),
            status="deploying",
            health_check_url=f"/api/models/{model_id}/v{version}/health"
        )
        
        self.deployments[deployment_id] = deployment
        
        # Submit deployment task
        task_id = await submit_distributed_task(
            "model_deployment",
            self._execute_deployment,
            {"deployment_id": deployment_id},
            priority=8
        )
        
        # Log deployment
        uap_logger.log_event(
            LogLevel.INFO,
            f"Model deployment started: {model_id} v{version}",
            EventType.AGENT,
            {
                "deployment_id": deployment_id,
                "model_id": model_id,
                "version": version,
                "stage": stage.value,
                "task_id": task_id
            },
            "model_manager"
        )
        
        return deployment_id
    
    async def _execute_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Execute model deployment (distributed task)"""
        try:
            deployment = self.deployments[deployment_id]
            
            # Simulate deployment process
            await asyncio.sleep(2)  # Deployment time
            
            # Update deployment status
            deployment.status = "active"
            
            # Record metrics
            record_pipeline_event(deployment_id, deployment.model_id, "deployed")
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "endpoint_url": deployment.endpoint_url,
                "status": deployment.status
            }
            
        except Exception as e:
            if deployment_id in self.deployments:
                self.deployments[deployment_id].status = "failed"
            
            return {
                "success": False,
                "error": str(e),
                "deployment_id": deployment_id
            }
    
    async def rollback_deployment(self,
                                deployment_id: str,
                                target_version: str = None) -> bool:
        """Rollback a deployment to previous version"""
        
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        current_version = deployment.version
        
        # Use rollback version if target not specified
        rollback_version = target_version or deployment.rollback_version
        if not rollback_version:
            return False
        
        # Create rollback deployment
        rollback_deployment_id = await self.deploy_model(
            model_id=deployment.model_id,
            version=rollback_version,
            stage=deployment.stage,
            traffic_percentage=deployment.traffic_percentage,
            approval_required=False
        )
        
        # Mark original deployment as rolled back
        deployment.status = "rolled_back"
        deployment.rollback_version = current_version
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Model rolled back: {deployment.model_id} from v{current_version} to v{rollback_version}",
            EventType.AGENT,
            {
                "deployment_id": deployment_id,
                "rollback_deployment_id": rollback_deployment_id,
                "model_id": deployment.model_id,
                "from_version": current_version,
                "to_version": rollback_version
            },
            "model_manager"
        )
        
        return True
    
    async def create_ab_test(self,
                           model_a_id: str,
                           model_a_version: str,
                           model_b_id: str,
                           model_b_version: str,
                           traffic_split: float = 50.0,
                           test_name: str = None) -> str:
        """Create A/B test between two model versions"""
        
        test_name = test_name or f"AB_Test_{model_a_id}_vs_{model_b_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        variants = [
            {
                "model_id": model_a_id,
                "model_version": model_a_version,
                "traffic_percentage": traffic_split,
                "name": f"Control_{model_a_id}",
                "description": f"Control variant: {model_a_id} v{model_a_version}"
            },
            {
                "model_id": model_b_id,
                "model_version": model_b_version,
                "traffic_percentage": 100.0 - traffic_split,
                "name": f"Treatment_{model_b_id}",
                "description": f"Treatment variant: {model_b_id} v{model_b_version}"
            }
        ]
        
        metrics = [
            {
                "metric_name": "accuracy",
                "metric_type": "numeric",
                "target_direction": "increase",
                "minimum_detectable_effect": 0.05
            },
            {
                "metric_name": "response_time",
                "metric_type": "latency",
                "target_direction": "decrease",
                "minimum_detectable_effect": 0.1
            }
        ]
        
        experiment = await self.ab_test_manager.create_experiment(
            name=test_name,
            description=f"A/B test comparing {model_a_id} v{model_a_version} vs {model_b_id} v{model_b_version}",
            variants=variants,
            metrics=metrics
        )
        
        return experiment.experiment_id
    
    async def _create_approval_workflow(self,
                                      model_id: str,
                                      version: str,
                                      source_stage: ModelLifecycleStage,
                                      target_stage: ModelLifecycleStage) -> str:
        """Create approval workflow for model promotion"""
        
        approval_id = str(uuid.uuid4())
        required_approvers = self.approval_workflows.get(target_stage, [])
        
        approval = ModelApproval(
            approval_id=approval_id,
            model_id=model_id,
            version=version,
            source_stage=source_stage,
            target_stage=target_stage,
            approved_by="",
            approval_time=datetime.utcnow(),
            approval_notes="",
            required_approvers=required_approvers,
            current_approvers=[],
            status="pending"
        )
        
        self.approvals[approval_id] = approval
        
        return approval_id
    
    async def approve_model(self,
                          approval_id: str,
                          approver: str,
                          notes: str = "") -> bool:
        """Approve model for promotion"""
        
        if approval_id not in self.approvals:
            return False
        
        approval = self.approvals[approval_id]
        
        if approver not in approval.required_approvers:
            return False
        
        if approver not in approval.current_approvers:
            approval.current_approvers.append(approver)
        
        approval.approval_notes += f"\n{approver}: {notes}"
        
        # Check if all approvals received
        if set(approval.current_approvers) >= set(approval.required_approvers):
            approval.status = "approved"
            approval.approved_by = ", ".join(approval.current_approvers)
            
            # Trigger deployment
            await self.deploy_model(
                model_id=approval.model_id,
                version=approval.version,
                stage=approval.target_stage,
                approval_required=False
            )
        
        return True
    
    async def get_model_metrics(self,
                              model_id: str,
                              version: str = None,
                              days: int = 30) -> Dict[str, Any]:
        """Get model performance metrics"""
        
        key = f"{model_id}_{version}" if version else model_id
        
        if key not in self.metrics_history:
            return {"error": "No metrics found"}
        
        # Filter metrics by date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_metrics = [
            m for m in self.metrics_history[key]
            if datetime.fromisoformat(m["timestamp"]) > cutoff_date
        ]
        
        if not recent_metrics:
            return {"error": "No recent metrics found"}
        
        # Calculate aggregated metrics
        aggregated = {
            "model_id": model_id,
            "version": version,
            "period_days": days,
            "total_evaluations": len(recent_metrics),
            "metrics": {}
        }
        
        # Calculate averages for each metric
        all_metrics = {}
        for eval_result in recent_metrics:
            for metric_name, value in eval_result.get("metrics", {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        for metric_name, values in all_metrics.items():
            aggregated["metrics"][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values)
            }
        
        return aggregated
    
    async def _store_metrics(self,
                           model_id: str,
                           version: str,
                           metrics: ModelMetrics):
        """Store model metrics in history"""
        
        key = f"{model_id}_{version}"
        
        if key not in self.metrics_history:
            self.metrics_history[key] = []
        
        metrics_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "version": version,
            "metrics": {k: v for k, v in asdict(metrics).items() if v is not None}
        }
        
        self.metrics_history[key].append(metrics_record)
        
        # Keep only last 1000 records per model
        if len(self.metrics_history[key]) > 1000:
            self.metrics_history[key] = self.metrics_history[key][-1000:]
    
    async def _start_monitoring(self):
        """Start background monitoring"""
        while True:
            try:
                await self._monitor_deployments()
                await self._cleanup_old_data()
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Model manager monitoring error: {e}",
                    EventType.AGENT,
                    {"error": str(e)},
                    "model_manager"
                )
                await asyncio.sleep(600)
    
    async def _monitor_deployments(self):
        """Monitor active deployments"""
        for deployment in self.deployments.values():
            if deployment.status == "active":
                # Health check simulation
                deployment.status = "healthy"
    
    async def _cleanup_old_data(self):
        """Clean up old data"""
        # Clean up completed approvals older than 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        approvals_to_remove = []
        for approval_id, approval in self.approvals.items():
            if (approval.status in ["approved", "rejected"] and 
                approval.approval_time < cutoff_date):
                approvals_to_remove.append(approval_id)
        
        for approval_id in approvals_to_remove:
            del self.approvals[approval_id]
    
    async def _load_deployments(self):
        """Load deployments from storage"""
        try:
            deployments_file = self.storage_path / "deployments.json"
            if deployments_file.exists():
                async with aiofiles.open(deployments_file, 'r') as f:
                    data = json.loads(await f.read())
                    # Reconstruct deployments (simplified)
                    pass
        except Exception as e:
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Failed to load deployments: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "model_manager"
            )
    
    async def _load_approvals(self):
        """Load approvals from storage"""
        try:
            approvals_file = self.storage_path / "approvals.json"
            if approvals_file.exists():
                async with aiofiles.open(approvals_file, 'r') as f:
                    data = json.loads(await f.read())
                    # Reconstruct approvals (simplified)
                    pass
        except Exception as e:
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Failed to load approvals: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "model_manager"
            )
    
    async def get_manager_status(self) -> Dict[str, Any]:
        """Get model manager status"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "storage_path": str(self.storage_path),
            "total_models": len(self.model_registry.models),
            "active_deployments": len([d for d in self.deployments.values() if d.status == "active"]),
            "pending_approvals": len([a for a in self.approvals.values() if a.status == "pending"]),
            "metrics_tracked": len(self.metrics_history),
            "system_status": "operational"
        }

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager