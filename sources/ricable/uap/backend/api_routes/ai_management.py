# File: backend/api_routes/ai_management.py
# API Routes for Advanced AI Model Management
# Provides REST API endpoints for model management, versioning, A/B testing, and training

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Import AI management components
from ..ai import (
    get_model_manager,
    get_version_manager,
    get_ab_test_manager,
    get_fine_tuning_manager,
    get_training_pipeline,
    ModelManager,
    ModelVersionManager,
    ABTestManager,
    FineTuningManager,
    TrainingPipeline,
    ModelType,
    VersionType,
    TestStatus,
    FineTuningMethod,
    DatasetType,
    FineTuningStatus,
    PipelineStatus,
    TrafficSplitStrategy
)

# Import authentication
from ..services.auth import get_current_user
from ..models.user import User

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/ai", tags=["AI Management"])

# Model Management Endpoints

@router.get("/models/status")
async def get_model_manager_status(current_user: User = Depends(get_current_user)):
    """Get model manager status."""
    try:
        model_manager = get_model_manager()
        status = await model_manager.get_manager_status()
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(f"Failed to get model manager status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/deploy")
async def deploy_model(
    model_id: str,
    version: str,
    model_type: str,
    model_path: str,
    traffic_percentage: float = 0.0,
    auto_deploy: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Deploy a model version."""
    try:
        model_manager = get_model_manager()
        
        # Convert string to enum
        model_type_enum = ModelType(model_type)
        
        deployment_id = await model_manager.deploy_model(
            model_id=model_id,
            version=version,
            model_type=model_type_enum,
            model_path=model_path,
            traffic_percentage=traffic_percentage,
            auto_deploy=auto_deploy
        )
        
        return {"success": True, "deployment_id": deployment_id}
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/deployments")
async def list_deployments(
    model_id: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List model deployments."""
    try:
        model_manager = get_model_manager()
        
        # Convert status string to enum if provided
        status_filter = None
        if status:
            from ..ai.model_management import DeploymentStatus
            status_filter = DeploymentStatus(status)
        
        deployments = await model_manager.list_deployments(
            model_id=model_id,
            status=status_filter
        )
        
        return {"success": True, "deployments": deployments}
    except Exception as e:
        logger.error(f"Failed to list deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/deployments/{deployment_id}")
async def get_deployment_status(
    deployment_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get deployment status."""
    try:
        model_manager = get_model_manager()
        status = await model_manager.get_deployment_status(deployment_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        return {"success": True, "deployment": status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/deployments/{deployment_id}/rollback")
async def rollback_deployment(
    deployment_id: str,
    target_version: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Rollback a deployment."""
    try:
        model_manager = get_model_manager()
        success = await model_manager.rollback_deployment(deployment_id, target_version)
        
        if not success:
            raise HTTPException(status_code=400, detail="Rollback failed")
        
        return {"success": True, "message": "Deployment rolled back successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rollback deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/inference/{deployment_id}")
async def run_inference(
    deployment_id: str,
    input_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """Run inference using a deployed model."""
    try:
        model_manager = get_model_manager()
        result = await model_manager.inference(
            deployment_id=deployment_id,
            input_data=input_data,
            context=context or {}
        )
        
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Versioning Endpoints

@router.get("/versions/status")
async def get_version_manager_status(current_user: User = Depends(get_current_user)):
    """Get version manager status."""
    try:
        version_manager = get_version_manager()
        status = await version_manager.get_manager_status()
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(f"Failed to get version manager status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/versions/create")
async def create_version(
    model_id: str,
    model_path: str,
    version_type: str,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    changelog: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user)
):
    """Create a new model version."""
    try:
        version_manager = get_version_manager()
        
        # Convert string to enum
        version_type_enum = VersionType(version_type)
        
        version_id = await version_manager.create_version(
            model_id=model_id,
            model_path=model_path,
            version_type=version_type_enum,
            config=config,
            metadata=metadata,
            changelog=changelog,
            created_by=current_user.username,
            tags=tags
        )
        
        return {"success": True, "version_id": version_id}
    except Exception as e:
        logger.error(f"Failed to create version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions")
async def list_versions(
    model_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """List model versions."""
    try:
        version_manager = get_version_manager()
        
        # Convert status string to enum if provided
        status_filter = None
        if status:
            from ..ai.model_versioning import VersionStatus
            status_filter = VersionStatus(status)
        
        versions = await version_manager.list_versions(
            model_id=model_id,
            status=status_filter,
            limit=limit
        )
        
        return {"success": True, "versions": versions}
    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions/{version_id}")
async def get_version(
    version_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get version details."""
    try:
        version_manager = get_version_manager()
        version = await version_manager.get_version(version_id)
        
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {"success": True, "version": version}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/versions/{version_id}/promote")
async def promote_version(
    version_id: str,
    target_status: str,
    current_user: User = Depends(get_current_user)
):
    """Promote a version to a higher status."""
    try:
        version_manager = get_version_manager()
        
        # Convert string to enum
        from ..ai.model_versioning import VersionStatus
        target_status_enum = VersionStatus(target_status)
        
        success = await version_manager.promote_version(version_id, target_status_enum)
        
        if not success:
            raise HTTPException(status_code=400, detail="Version promotion failed")
        
        return {"success": True, "message": "Version promoted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to promote version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/versions/compare")
async def compare_versions(
    version_a_id: str,
    version_b_id: str,
    current_user: User = Depends(get_current_user)
):
    """Compare two model versions."""
    try:
        version_manager = get_version_manager()
        comparison = await version_manager.compare_versions(version_a_id, version_b_id)
        
        return {"success": True, "comparison": comparison.to_dict()}
    except Exception as e:
        logger.error(f"Failed to compare versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# A/B Testing Endpoints

@router.get("/ab-tests/status")
async def get_ab_test_manager_status(current_user: User = Depends(get_current_user)):
    """Get A/B test manager status."""
    try:
        ab_test_manager = get_ab_test_manager()
        status = await ab_test_manager.get_manager_status()
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(f"Failed to get A/B test manager status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests/create")
async def create_ab_test(
    test_name: str,
    description: str,
    model_id: str,
    variant_a_version: str,
    variant_b_version: str,
    traffic_split_strategy: str = "equal",
    traffic_split_weights: Optional[Dict[str, float]] = None,
    metrics_to_track: Optional[List[str]] = None,
    success_criteria: Optional[Dict[str, Any]] = None,
    minimum_sample_size: int = 1000,
    test_duration_hours: int = 168,
    current_user: User = Depends(get_current_user)
):
    """Create a new A/B test."""
    try:
        ab_test_manager = get_ab_test_manager()
        
        # Convert string to enum
        strategy_enum = TrafficSplitStrategy(traffic_split_strategy)
        
        test_id = await ab_test_manager.create_test(
            test_name=test_name,
            description=description,
            model_id=model_id,
            variant_a_version=variant_a_version,
            variant_b_version=variant_b_version,
            traffic_split_strategy=strategy_enum,
            traffic_split_weights=traffic_split_weights,
            metrics_to_track=metrics_to_track,
            success_criteria=success_criteria,
            minimum_sample_size=minimum_sample_size,
            test_duration_hours=test_duration_hours,
            created_by=current_user.username
        )
        
        return {"success": True, "test_id": test_id}
    except Exception as e:
        logger.error(f"Failed to create A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests/{test_id}/start")
async def start_ab_test(
    test_id: str,
    current_user: User = Depends(get_current_user)
):
    """Start an A/B test."""
    try:
        ab_test_manager = get_ab_test_manager()
        success = await ab_test_manager.start_test(test_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start A/B test")
        
        return {"success": True, "message": "A/B test started successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-tests")
async def list_ab_tests(
    model_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """List A/B tests."""
    try:
        ab_test_manager = get_ab_test_manager()
        
        # Convert status string to enum if provided
        status_filter = None
        if status:
            status_filter = TestStatus(status)
        
        tests = await ab_test_manager.list_tests(
            model_id=model_id,
            status=status_filter,
            limit=limit
        )
        
        return {"success": True, "tests": tests}
    except Exception as e:
        logger.error(f"Failed to list A/B tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-tests/{test_id}")
async def get_ab_test_status(
    test_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get A/B test status."""
    try:
        ab_test_manager = get_ab_test_manager()
        test_status = await ab_test_manager.get_test_status(test_id)
        
        if not test_status:
            raise HTTPException(status_code=404, detail="A/B test not found")
        
        return {"success": True, "test": test_status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get A/B test status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fine-tuning Endpoints

@router.get("/fine-tuning/status")
async def get_fine_tuning_manager_status(current_user: User = Depends(get_current_user)):
    """Get fine-tuning manager status."""
    try:
        fine_tuning_manager = get_fine_tuning_manager()
        status = await fine_tuning_manager.get_manager_status()
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(f"Failed to get fine-tuning manager status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fine-tuning/jobs")
async def create_fine_tuning_job(
    job_name: str,
    base_model_id: str,
    base_model_path: str,
    dataset_path: str,
    dataset_type: str,
    method: str = "lora",
    current_user: User = Depends(get_current_user)
):
    """Create a fine-tuning job."""
    try:
        fine_tuning_manager = get_fine_tuning_manager()
        
        # Convert strings to enums
        dataset_type_enum = DatasetType(dataset_type)
        method_enum = FineTuningMethod(method)
        
        job_id = await fine_tuning_manager.create_fine_tuning_job(
            job_name=job_name,
            base_model_id=base_model_id,
            base_model_path=base_model_path,
            dataset_path=dataset_path,
            dataset_type=dataset_type_enum,
            method=method_enum,
            created_by=current_user.username
        )
        
        return {"success": True, "job_id": job_id}
    except Exception as e:
        logger.error(f"Failed to create fine-tuning job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fine-tuning/jobs/{job_id}/start")
async def start_fine_tuning_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Start a fine-tuning job."""
    try:
        fine_tuning_manager = get_fine_tuning_manager()
        success = await fine_tuning_manager.start_fine_tuning(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start fine-tuning job")
        
        return {"success": True, "message": "Fine-tuning job started successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start fine-tuning job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fine-tuning/jobs")
async def list_fine_tuning_jobs(
    status: Optional[str] = None,
    method: Optional[str] = None,
    limit: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """List fine-tuning jobs."""
    try:
        fine_tuning_manager = get_fine_tuning_manager()
        
        # Convert strings to enums if provided
        status_filter = None
        method_filter = None
        
        if status:
            status_filter = FineTuningStatus(status)
        if method:
            method_filter = FineTuningMethod(method)
        
        jobs = await fine_tuning_manager.list_jobs(
            status=status_filter,
            method=method_filter,
            limit=limit
        )
        
        return {"success": True, "jobs": jobs}
    except Exception as e:
        logger.error(f"Failed to list fine-tuning jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fine-tuning/jobs/{job_id}")
async def get_fine_tuning_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get fine-tuning job status."""
    try:
        fine_tuning_manager = get_fine_tuning_manager()
        job_status = await fine_tuning_manager.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Fine-tuning job not found")
        
        return {"success": True, "job": job_status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fine-tuning job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/fine-tuning/jobs/{job_id}")
async def cancel_fine_tuning_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a fine-tuning job."""
    try:
        fine_tuning_manager = get_fine_tuning_manager()
        success = await fine_tuning_manager.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to cancel fine-tuning job")
        
        return {"success": True, "message": "Fine-tuning job cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel fine-tuning job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Training Pipeline Endpoints

@router.get("/pipelines/status")
async def get_training_pipeline_status(current_user: User = Depends(get_current_user)):
    """Get training pipeline manager status."""
    try:
        training_pipeline = get_training_pipeline()
        status = await training_pipeline.get_manager_status()
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(f"Failed to get training pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipelines")
async def create_training_pipeline(
    pipeline_name: str,
    description: str,
    model_id: str,
    base_model_path: str,
    dataset_path: str,
    dataset_type: str,
    fine_tuning_method: str = "lora",
    version_type: str = "minor",
    auto_deploy: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Create a training pipeline."""
    try:
        training_pipeline = get_training_pipeline()
        
        # Convert strings to enums
        dataset_type_enum = DatasetType(dataset_type)
        method_enum = FineTuningMethod(fine_tuning_method)
        version_type_enum = VersionType(version_type)
        
        pipeline_id = await training_pipeline.create_pipeline(
            pipeline_name=pipeline_name,
            description=description,
            model_id=model_id,
            base_model_path=base_model_path,
            dataset_path=dataset_path,
            dataset_type=dataset_type_enum,
            fine_tuning_method=method_enum,
            version_type=version_type_enum,
            auto_deploy=auto_deploy,
            created_by=current_user.username
        )
        
        return {"success": True, "pipeline_id": pipeline_id}
    except Exception as e:
        logger.error(f"Failed to create training pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipelines/{pipeline_id}/execute")
async def execute_training_pipeline(
    pipeline_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """Execute a training pipeline."""
    try:
        training_pipeline = get_training_pipeline()
        
        execution_id = await training_pipeline.execute_pipeline(
            pipeline_id=pipeline_id,
            triggered_by=current_user.username,
            parameters=parameters
        )
        
        return {"success": True, "execution_id": execution_id}
    except Exception as e:
        logger.error(f"Failed to execute training pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines")
async def list_training_pipelines(
    created_by: Optional[str] = None,
    limit: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """List training pipelines."""
    try:
        training_pipeline = get_training_pipeline()
        
        pipelines = await training_pipeline.list_pipelines(
            created_by=created_by,
            limit=limit
        )
        
        return {"success": True, "pipelines": pipelines}
    except Exception as e:
        logger.error(f"Failed to list training pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines/{pipeline_id}")
async def get_training_pipeline_details(
    pipeline_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get training pipeline details."""
    try:
        training_pipeline = get_training_pipeline()
        pipeline_status = await training_pipeline.get_pipeline_status(pipeline_id)
        
        if not pipeline_status:
            raise HTTPException(status_code=404, detail="Training pipeline not found")
        
        return {"success": True, "pipeline": pipeline_status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training pipeline details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines/executions/{execution_id}")
async def get_pipeline_execution_status(
    execution_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get pipeline execution status."""
    try:
        training_pipeline = get_training_pipeline()
        execution_status = await training_pipeline.get_execution_status(execution_id)
        
        if not execution_status:
            raise HTTPException(status_code=404, detail="Pipeline execution not found")
        
        return {"success": True, "execution": execution_status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline execution status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for AI management systems."""
    try:
        # Initialize all managers
        model_manager = get_model_manager()
        version_manager = get_version_manager()
        ab_test_manager = get_ab_test_manager()
        fine_tuning_manager = get_fine_tuning_manager()
        training_pipeline = get_training_pipeline()
        
        # Get status from each manager
        model_status = await model_manager.get_manager_status()
        version_status = await version_manager.get_manager_status()
        ab_test_status = await ab_test_manager.get_manager_status()
        fine_tuning_status = await fine_tuning_manager.get_manager_status()
        pipeline_status = await training_pipeline.get_manager_status()
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "model_manager": "healthy",
                "version_manager": "healthy",
                "ab_test_manager": "healthy",
                "fine_tuning_manager": "healthy",
                "training_pipeline": "healthy"
            },
            "summary": {
                "total_deployments": model_status.get("total_deployments", 0),
                "total_versions": version_status.get("total_versions", 0),
                "active_tests": ab_test_status.get("active_tests", 0),
                "active_jobs": fine_tuning_status.get("active_jobs", 0),
                "total_pipelines": pipeline_status.get("total_pipelines", 0)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

print("AI Management API routes loaded.")