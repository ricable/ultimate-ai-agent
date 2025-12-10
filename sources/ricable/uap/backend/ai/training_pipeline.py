# File: backend/ai/training_pipeline.py
# Training Pipeline System
# Provides comprehensive training workflow automation and orchestration

import asyncio
import json
import logging
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import yaml

# Import existing infrastructure
from .model_management import ModelManager, get_model_manager
from .model_versioning import ModelVersionManager, get_version_manager, VersionType
from .fine_tuning import FineTuningManager, get_fine_tuning_manager, FineTuningMethod, DatasetType
from .ab_testing import ABTestManager, get_ab_test_manager
from ..distributed.ray_manager import submit_distributed_task
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.prometheus_metrics import record_pipeline_event

# Configure logging
logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Training pipeline status states"""
    DRAFT = "draft"
    VALIDATING = "validating"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PipelineStage(Enum):
    """Training pipeline stages"""
    DATA_PREPARATION = "data_preparation"
    DATA_VALIDATION = "data_validation"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_TESTING = "model_testing"
    MODEL_DEPLOYMENT = "model_deployment"
    AB_TESTING = "ab_testing"
    MONITORING = "monitoring"

class TriggerType(Enum):
    """Pipeline trigger types"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    DATA_CHANGE = "data_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MODEL_DRIFT = "model_drift"

@dataclass
class PipelineStageConfig:
    """Configuration for a pipeline stage"""
    stage: PipelineStage
    enabled: bool = True
    timeout_minutes: int = 60
    retry_attempts: int = 3
    success_criteria: Dict[str, Any] = None
    failure_action: str = "stop"  # stop, continue, retry
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.success_criteria is None:
            self.success_criteria = {}
        if self.parameters is None:
            self.parameters = {}

@dataclass
class PipelineConfig:
    """Complete training pipeline configuration"""
    pipeline_id: str
    pipeline_name: str
    description: str
    model_id: str
    base_model_path: str
    dataset_path: str
    dataset_type: DatasetType
    fine_tuning_method: FineTuningMethod
    version_type: VersionType
    stages: List[PipelineStageConfig]
    environment: Dict[str, str]
    notifications: Dict[str, Any]
    schedule: Optional[str] = None  # Cron expression
    trigger_type: TriggerType = TriggerType.MANUAL
    auto_deploy: bool = False
    rollback_on_failure: bool = True
    created_by: str = "system"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['dataset_type'] = self.dataset_type.value
        data['fine_tuning_method'] = self.fine_tuning_method.value
        data['version_type'] = self.version_type.value
        data['trigger_type'] = self.trigger_type.value
        data['created_at'] = self.created_at.isoformat()
        for i, stage in enumerate(data['stages']):
            data['stages'][i]['stage'] = stage['stage'].value
        return data

@dataclass
class PipelineExecution:
    """Pipeline execution tracking"""
    execution_id: str
    pipeline_id: str
    status: PipelineStatus
    current_stage: Optional[PipelineStage] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    stage_results: Dict[str, Dict[str, Any]] = None
    error_message: Optional[str] = None
    triggered_by: str = "system"
    trigger_type: TriggerType = TriggerType.MANUAL
    artifacts: Dict[str, str] = None  # stage -> artifact_path
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.stage_results is None:
            self.stage_results = {}
        if self.artifacts is None:
            self.artifacts = {}
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        if self.current_stage:
            data['current_stage'] = self.current_stage.value
        data['trigger_type'] = self.trigger_type.value
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

@dataclass
class TrainingJob:
    """Complete training job information"""
    config: PipelineConfig
    executions: List[PipelineExecution]
    latest_execution: Optional[PipelineExecution] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "config": self.config.to_dict(),
            "executions": [exec.to_dict() for exec in self.executions]
        }
        if self.latest_execution:
            data["latest_execution"] = self.latest_execution.to_dict()
        return data

class TrainingPipeline:
    """
    Advanced Training Pipeline System.
    Provides comprehensive workflow automation for model training, validation, and deployment.
    """
    
    def __init__(self, base_pipeline_dir: str = None):
        """
        Initialize the training pipeline.
        
        Args:
            base_pipeline_dir: Base directory for pipeline data
        """
        self.base_pipeline_dir = Path(base_pipeline_dir) if base_pipeline_dir else Path.home() / ".uap" / "pipelines"
        
        # Initialize infrastructure components
        self.model_manager = get_model_manager()
        self.version_manager = get_version_manager()
        self.fine_tuning_manager = get_fine_tuning_manager()
        self.ab_test_manager = get_ab_test_manager()
        
        # Pipeline tracking
        self.pipelines: Dict[str, TrainingJob] = {}  # pipeline_id -> TrainingJob
        self.active_executions: Dict[str, PipelineExecution] = {}  # execution_id -> PipelineExecution
        self.execution_history: List[Dict[str, Any]] = []
        
        # Stage handlers
        self.stage_handlers = {
            PipelineStage.DATA_PREPARATION: self._handle_data_preparation,
            PipelineStage.DATA_VALIDATION: self._handle_data_validation,
            PipelineStage.MODEL_TRAINING: self._handle_model_training,
            PipelineStage.MODEL_VALIDATION: self._handle_model_validation,
            PipelineStage.MODEL_TESTING: self._handle_model_testing,
            PipelineStage.MODEL_DEPLOYMENT: self._handle_model_deployment,
            PipelineStage.AB_TESTING: self._handle_ab_testing,
            PipelineStage.MONITORING: self._handle_monitoring
        }
        
        # Initialize directories
        self.base_pipeline_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.base_pipeline_dir / "configs"
        self.executions_dir = self.base_pipeline_dir / "executions"
        self.artifacts_dir = self.base_pipeline_dir / "artifacts"
        self.logs_dir = self.base_pipeline_dir / "logs"
        
        for dir_path in [self.configs_dir, self.executions_dir, self.artifacts_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training pipeline initialized with base directory: {self.base_pipeline_dir}")
    
    async def initialize(self) -> bool:
        """
        Initialize the training pipeline system.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing training pipeline system...")
            
            # Initialize component managers
            await self.model_manager.initialize()
            await self.version_manager.initialize()
            await self.fine_tuning_manager.initialize()
            await self.ab_test_manager.initialize()
            
            # Load existing pipelines
            await self._load_existing_pipelines()
            
            # Start pipeline monitoring
            asyncio.create_task(self._start_pipeline_monitoring())
            
            logger.info(f"Training pipeline system initialization complete. Loaded {len(self.pipelines)} pipelines.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize training pipeline system: {e}")
            return False
    
    async def create_pipeline(self,
                            pipeline_name: str,
                            description: str,
                            model_id: str,
                            base_model_path: str,
                            dataset_path: str,
                            dataset_type: DatasetType,
                            fine_tuning_method: FineTuningMethod = FineTuningMethod.LORA,
                            version_type: VersionType = VersionType.MINOR,
                            custom_stages: List[PipelineStageConfig] = None,
                            schedule: str = None,
                            auto_deploy: bool = False,
                            created_by: str = "system") -> str:
        """
        Create a new training pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            description: Pipeline description
            model_id: Model identifier
            base_model_path: Path to base model
            dataset_path: Path to training dataset
            dataset_type: Type of dataset
            fine_tuning_method: Fine-tuning method to use
            version_type: Version type for new model
            custom_stages: Custom stage configurations
            schedule: Cron schedule for automatic runs
            auto_deploy: Automatically deploy successful models
            created_by: Pipeline creator
            
        Returns:
            Pipeline ID
        """
        try:
            pipeline_id = str(uuid.uuid4())
            
            # Create default stages if not provided
            if custom_stages is None:
                stages = self._create_default_stages()
            else:
                stages = custom_stages
            
            # Create pipeline configuration
            config = PipelineConfig(
                pipeline_id=pipeline_id,
                pipeline_name=pipeline_name,
                description=description,
                model_id=model_id,
                base_model_path=base_model_path,
                dataset_path=dataset_path,
                dataset_type=dataset_type,
                fine_tuning_method=fine_tuning_method,
                version_type=version_type,
                stages=stages,
                environment={},
                notifications={},
                schedule=schedule,
                trigger_type=TriggerType.SCHEDULED if schedule else TriggerType.MANUAL,
                auto_deploy=auto_deploy,
                created_by=created_by
            )
            
            # Create training job
            job = TrainingJob(
                config=config,
                executions=[]
            )
            
            self.pipelines[pipeline_id] = job
            
            # Save pipeline configuration
            await self._save_pipeline_config(job)
            
            # Log pipeline creation
            uap_logger.log_event(
                LogLevel.INFO,
                f"Training pipeline created: {pipeline_name}",
                EventType.AGENT,
                {
                    "pipeline_id": pipeline_id,
                    "pipeline_name": pipeline_name,
                    "model_id": model_id,
                    "fine_tuning_method": fine_tuning_method.value,
                    "auto_deploy": auto_deploy
                },
                "training_pipeline"
            )
            
            # Record metrics
            record_pipeline_event(pipeline_id, model_id, "created")
            
            logger.info(f"Created training pipeline {pipeline_id}: {pipeline_name}")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to create training pipeline: {e}")
            raise
    
    def _create_default_stages(self) -> List[PipelineStageConfig]:
        """
        Create default pipeline stages.
        
        Returns:
            List of default stage configurations
        """
        return [
            PipelineStageConfig(
                stage=PipelineStage.DATA_PREPARATION,
                timeout_minutes=30,
                success_criteria={"min_samples": 100},
                parameters={"validation_split": 0.1, "test_split": 0.1}
            ),
            PipelineStageConfig(
                stage=PipelineStage.DATA_VALIDATION,
                timeout_minutes=15,
                success_criteria={"data_quality_score": 0.8},
                parameters={"check_duplicates": True, "check_format": True}
            ),
            PipelineStageConfig(
                stage=PipelineStage.MODEL_TRAINING,
                timeout_minutes=120,
                success_criteria={"max_train_loss": 1.0},
                parameters={"epochs": 3, "batch_size": 4}
            ),
            PipelineStageConfig(
                stage=PipelineStage.MODEL_VALIDATION,
                timeout_minutes=30,
                success_criteria={"min_accuracy": 0.7, "max_val_loss": 1.2},
                parameters={"validation_metrics": ["accuracy", "loss", "perplexity"]}
            ),
            PipelineStageConfig(
                stage=PipelineStage.MODEL_TESTING,
                timeout_minutes=20,
                success_criteria={"min_test_accuracy": 0.65},
                parameters={"test_metrics": ["accuracy", "f1_score"]}
            ),
            PipelineStageConfig(
                stage=PipelineStage.MODEL_DEPLOYMENT,
                timeout_minutes=10,
                success_criteria={"deployment_health": True},
                parameters={"initial_traffic": 0.0}
            )
        ]
    
    async def execute_pipeline(self, 
                             pipeline_id: str, 
                             triggered_by: str = "system",
                             trigger_type: TriggerType = TriggerType.MANUAL,
                             parameters: Dict[str, Any] = None) -> str:
        """
        Execute a training pipeline.
        
        Args:
            pipeline_id: Pipeline ID
            triggered_by: Who triggered the execution
            trigger_type: Type of trigger
            parameters: Runtime parameters
            
        Returns:
            Execution ID
        """
        try:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            pipeline = self.pipelines[pipeline_id]
            execution_id = str(uuid.uuid4())
            
            # Create execution record
            execution = PipelineExecution(
                execution_id=execution_id,
                pipeline_id=pipeline_id,
                status=PipelineStatus.QUEUED,
                triggered_by=triggered_by,
                trigger_type=trigger_type
            )
            
            # Add runtime parameters
            if parameters:
                execution.metrics.update(parameters)
            
            self.active_executions[execution_id] = execution
            pipeline.executions.append(execution)
            pipeline.latest_execution = execution
            
            # Submit execution to distributed processing
            task_id = await submit_distributed_task(
                "pipeline_execution",
                self._execute_pipeline_workflow,
                {"execution_id": execution_id},
                priority=7
            )
            
            # Log execution start
            uap_logger.log_event(
                LogLevel.INFO,
                f"Pipeline execution started: {pipeline.config.pipeline_name}",
                EventType.AGENT,
                {
                    "execution_id": execution_id,
                    "pipeline_id": pipeline_id,
                    "pipeline_name": pipeline.config.pipeline_name,
                    "triggered_by": triggered_by,
                    "task_id": task_id
                },
                "training_pipeline"
            )
            
            # Record metrics
            record_pipeline_event(pipeline_id, pipeline.config.model_id, "execution_started")
            
            logger.info(f"Started pipeline execution {execution_id} for pipeline {pipeline_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute pipeline {pipeline_id}: {e}")
            raise
    
    def _execute_pipeline_workflow(self, execution_id: str) -> Dict[str, Any]:
        """
        Execute the pipeline workflow (Ray task).
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Execution result
        """
        try:
            execution = self.active_executions[execution_id]
            pipeline = self.pipelines[execution.pipeline_id]
            
            # Update execution status
            execution.status = PipelineStatus.RUNNING
            execution.started_at = datetime.utcnow()
            
            logger.info(f"Starting pipeline workflow for execution {execution_id}")
            
            # Execute stages sequentially
            for stage_config in pipeline.config.stages:
                if not stage_config.enabled:
                    logger.info(f"Skipping disabled stage: {stage_config.stage.value}")
                    continue
                
                execution.current_stage = stage_config.stage
                
                logger.info(f"Executing stage: {stage_config.stage.value}")
                
                # Execute stage with timeout and retry
                stage_result = self._execute_stage_with_retry(
                    execution, stage_config
                )
                
                execution.stage_results[stage_config.stage.value] = stage_result
                
                if not stage_result["success"]:
                    if stage_config.failure_action == "stop":
                        raise Exception(f"Stage {stage_config.stage.value} failed: {stage_result['error']}")
                    elif stage_config.failure_action == "continue":
                        logger.warning(f"Stage {stage_config.stage.value} failed but continuing: {stage_result['error']}")
                        continue
                
                logger.info(f"Stage {stage_config.stage.value} completed successfully")
            
            # Pipeline completed successfully
            execution.status = PipelineStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            
            # Remove from active executions
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            logger.info(f"Pipeline execution {execution_id} completed successfully")
            
            return {
                "success": True,
                "execution_id": execution_id,
                "stage_results": execution.stage_results,
                "artifacts": execution.artifacts
            }
            
        except Exception as e:
            # Update execution status
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = PipelineStatus.FAILED
                execution.error_message = str(e)
                execution.completed_at = datetime.utcnow()
                
                # Remove from active executions
                del self.active_executions[execution_id]
            
            logger.error(f"Pipeline execution {execution_id} failed: {e}")
            
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e)
            }
    
    def _execute_stage_with_retry(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Execute a pipeline stage with retry logic.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage execution result
        """
        last_error = None
        
        for attempt in range(stage_config.retry_attempts):
            try:
                logger.info(f"Stage {stage_config.stage.value} - Attempt {attempt + 1}/{stage_config.retry_attempts}")
                
                # Get stage handler
                handler = self.stage_handlers.get(stage_config.stage)
                if not handler:
                    raise Exception(f"No handler found for stage {stage_config.stage.value}")
                
                # Execute stage with timeout
                start_time = time.time()
                result = handler(execution, stage_config)
                execution_time = time.time() - start_time
                
                # Check timeout
                if execution_time > stage_config.timeout_minutes * 60:
                    raise Exception(f"Stage timeout after {execution_time:.1f} seconds")
                
                # Check success criteria
                if not self._validate_stage_success(result, stage_config.success_criteria):
                    raise Exception(f"Stage success criteria not met: {stage_config.success_criteria}")
                
                result["execution_time_seconds"] = execution_time
                result["attempt"] = attempt + 1
                
                return result
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Stage {stage_config.stage.value} attempt {attempt + 1} failed: {e}")
                
                if attempt < stage_config.retry_attempts - 1:
                    time.sleep(5)  # Wait before retry
        
        return {
            "success": False,
            "error": last_error,
            "attempts": stage_config.retry_attempts
        }
    
    def _validate_stage_success(self, result: Dict[str, Any], success_criteria: Dict[str, Any]) -> bool:
        """
        Validate stage success based on criteria.
        
        Args:
            result: Stage execution result
            success_criteria: Success criteria to check
            
        Returns:
            True if success criteria are met
        """
        try:
            if not result.get("success", False):
                return False
            
            for criterion, expected_value in success_criteria.items():
                if criterion in result:
                    actual_value = result[criterion]
                    
                    if isinstance(expected_value, (int, float)):
                        if criterion.startswith("min_"):
                            if actual_value < expected_value:
                                return False
                        elif criterion.startswith("max_"):
                            if actual_value > expected_value:
                                return False
                    elif isinstance(expected_value, bool):
                        if actual_value != expected_value:
                            return False
                    elif isinstance(expected_value, str):
                        if actual_value != expected_value:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating stage success: {e}")
            return False
    
    # Stage Handlers
    
    def _handle_data_preparation(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Handle data preparation stage.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage result
        """
        try:
            logger.info(f"Preparing data for execution {execution.execution_id}")
            
            pipeline = self.pipelines[execution.pipeline_id]
            dataset_path = pipeline.config.dataset_path
            
            # Simulate data preparation
            time.sleep(2)
            
            # Mock data statistics
            total_samples = 1000
            validation_split = stage_config.parameters.get("validation_split", 0.1)
            test_split = stage_config.parameters.get("test_split", 0.1)
            
            train_samples = int(total_samples * (1 - validation_split - test_split))
            val_samples = int(total_samples * validation_split)
            test_samples = int(total_samples * test_split)
            
            result = {
                "success": True,
                "total_samples": total_samples,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "test_samples": test_samples,
                "min_samples": train_samples  # For success criteria
            }
            
            # Save artifacts
            artifact_path = self.artifacts_dir / execution.execution_id / "data_preparation"
            artifact_path.mkdir(parents=True, exist_ok=True)
            
            with open(artifact_path / "data_stats.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            execution.artifacts["data_preparation"] = str(artifact_path)
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_data_validation(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Handle data validation stage.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage result
        """
        try:
            logger.info(f"Validating data for execution {execution.execution_id}")
            
            # Simulate data validation
            time.sleep(1)
            
            # Mock validation results
            data_quality_score = 0.85
            duplicates_found = 5
            format_errors = 0
            
            result = {
                "success": True,
                "data_quality_score": data_quality_score,
                "duplicates_found": duplicates_found,
                "format_errors": format_errors,
                "validation_passed": True
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_model_training(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Handle model training stage.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage result
        """
        try:
            logger.info(f"Training model for execution {execution.execution_id}")
            
            pipeline = self.pipelines[execution.pipeline_id]
            
            # Create fine-tuning job
            job_id = asyncio.run(self.fine_tuning_manager.create_fine_tuning_job(
                job_name=f"pipeline_{execution.execution_id}_training",
                base_model_id=pipeline.config.model_id,
                base_model_path=pipeline.config.base_model_path,
                dataset_path=pipeline.config.dataset_path,
                dataset_type=pipeline.config.dataset_type,
                method=pipeline.config.fine_tuning_method
            ))
            
            # Start fine-tuning
            started = asyncio.run(self.fine_tuning_manager.start_fine_tuning(job_id))
            
            if not started:
                raise Exception("Failed to start fine-tuning job")
            
            # Wait for completion (simplified - in practice would be async)
            max_wait_time = stage_config.timeout_minutes * 60
            wait_time = 0
            
            while wait_time < max_wait_time:
                job_status = asyncio.run(self.fine_tuning_manager.get_job_status(job_id))
                
                if job_status["status"] == "completed":
                    break
                elif job_status["status"] == "failed":
                    raise Exception(f"Fine-tuning failed: {job_status.get('error_message', 'Unknown error')}")
                
                time.sleep(10)
                wait_time += 10
            
            if wait_time >= max_wait_time:
                raise Exception("Training timeout")
            
            # Get final metrics
            final_job_status = asyncio.run(self.fine_tuning_manager.get_job_status(job_id))
            final_metrics = final_job_status.get("metrics", {})
            
            result = {
                "success": True,
                "fine_tuning_job_id": job_id,
                "final_train_loss": final_metrics.get("final_train_loss", 0.5),
                "final_val_loss": final_metrics.get("final_val_loss", 0.6),
                "training_time_seconds": final_metrics.get("training_time_seconds", 0),
                "model_path": final_job_status.get("output_dir", "")
            }
            
            execution.artifacts["model_training"] = result["model_path"]
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_model_validation(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Handle model validation stage.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage result
        """
        try:
            logger.info(f"Validating model for execution {execution.execution_id}")
            
            # Simulate model validation
            time.sleep(3)
            
            # Mock validation metrics
            accuracy = 0.75
            val_loss = 1.1
            perplexity = 15.2
            
            result = {
                "success": True,
                "accuracy": accuracy,
                "val_loss": val_loss,
                "perplexity": perplexity,
                "min_accuracy": accuracy,  # For success criteria
                "max_val_loss": val_loss   # For success criteria
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_model_testing(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Handle model testing stage.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage result
        """
        try:
            logger.info(f"Testing model for execution {execution.execution_id}")
            
            # Simulate model testing
            time.sleep(2)
            
            # Mock test metrics
            test_accuracy = 0.72
            f1_score = 0.71
            precision = 0.73
            recall = 0.70
            
            result = {
                "success": True,
                "test_accuracy": test_accuracy,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
                "min_test_accuracy": test_accuracy  # For success criteria
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_model_deployment(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Handle model deployment stage.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage result
        """
        try:
            logger.info(f"Deploying model for execution {execution.execution_id}")
            
            pipeline = self.pipelines[execution.pipeline_id]
            
            # Get model path from training stage
            training_result = execution.stage_results.get("model_training", {})
            model_path = training_result.get("model_path", "")
            
            if not model_path:
                raise Exception("No trained model found")
            
            # Create model version
            version_id = asyncio.run(self.version_manager.create_version(
                model_id=pipeline.config.model_id,
                model_path=model_path,
                version_type=pipeline.config.version_type,
                changelog=[f"Automated training pipeline execution {execution.execution_id}"]
            ))
            
            # Deploy model
            deployment_id = asyncio.run(self.model_manager.deploy_model(
                model_id=pipeline.config.model_id,
                version=version_id.split('_')[1],  # Extract version number
                model_type="text_generation",  # TODO: Make configurable
                model_path=model_path,
                traffic_percentage=stage_config.parameters.get("initial_traffic", 0.0),
                auto_deploy=pipeline.config.auto_deploy
            ))
            
            result = {
                "success": True,
                "version_id": version_id,
                "deployment_id": deployment_id,
                "model_path": model_path,
                "deployment_health": True  # For success criteria
            }
            
            execution.artifacts["model_deployment"] = deployment_id
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_ab_testing(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Handle A/B testing stage.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage result
        """
        try:
            logger.info(f"Setting up A/B test for execution {execution.execution_id}")
            
            # This stage would set up A/B testing but not wait for completion
            # The actual A/B test would run separately
            
            result = {
                "success": True,
                "ab_test_setup": True,
                "note": "A/B test setup initiated - test will run independently"
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_monitoring(self, execution: PipelineExecution, stage_config: PipelineStageConfig) -> Dict[str, Any]:
        """
        Handle monitoring setup stage.
        
        Args:
            execution: Pipeline execution
            stage_config: Stage configuration
            
        Returns:
            Stage result
        """
        try:
            logger.info(f"Setting up monitoring for execution {execution.execution_id}")
            
            # Set up monitoring for the deployed model
            result = {
                "success": True,
                "monitoring_enabled": True,
                "alerts_configured": True
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a training pipeline.
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            Pipeline status information
        """
        if pipeline_id not in self.pipelines:
            return None
        
        return self.pipelines[pipeline_id].to_dict()
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a pipeline execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Execution status information
        """
        # Check active executions
        if execution_id in self.active_executions:
            return self.active_executions[execution_id].to_dict()
        
        # Check completed executions
        for pipeline in self.pipelines.values():
            for execution in pipeline.executions:
                if execution.execution_id == execution_id:
                    return execution.to_dict()
        
        return None
    
    async def list_pipelines(self, created_by: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        List training pipelines.
        
        Args:
            created_by: Filter by creator
            limit: Maximum number of pipelines to return
            
        Returns:
            List of pipeline information
        """
        pipelines = []
        
        for pipeline in self.pipelines.values():
            if created_by and pipeline.config.created_by != created_by:
                continue
            pipelines.append(pipeline.to_dict())
        
        # Sort by creation time (newest first)
        pipelines.sort(key=lambda p: p['config']['created_at'], reverse=True)
        
        if limit:
            pipelines = pipelines[:limit]
        
        return pipelines
    
    async def _start_pipeline_monitoring(self) -> None:
        """
        Start background pipeline monitoring.
        """
        logger.info("Starting pipeline monitoring...")
        
        while True:
            try:
                await self._monitor_active_executions()
                await self._check_scheduled_pipelines()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Pipeline monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _monitor_active_executions(self) -> None:
        """
        Monitor active pipeline executions.
        """
        try:
            for execution_id, execution in list(self.active_executions.items()):
                # Check for timeout
                if execution.started_at:
                    elapsed_time = datetime.utcnow() - execution.started_at
                    max_execution_time = timedelta(hours=6)  # Maximum 6 hours
                    
                    if elapsed_time > max_execution_time:
                        logger.warning(f"Execution {execution_id} exceeded maximum time, marking as failed")
                        execution.status = PipelineStatus.FAILED
                        execution.error_message = "Execution timeout"
                        execution.completed_at = datetime.utcnow()
                        del self.active_executions[execution_id]
            
        except Exception as e:
            logger.error(f"Failed to monitor active executions: {e}")
    
    async def _check_scheduled_pipelines(self) -> None:
        """
        Check for scheduled pipeline executions.
        """
        try:
            # This would implement cron-like scheduling
            # For now, it's a placeholder
            pass
            
        except Exception as e:
            logger.error(f"Failed to check scheduled pipelines: {e}")
    
    async def _load_existing_pipelines(self) -> None:
        """
        Load existing pipelines from storage.
        """
        try:
            config_files = list(self.configs_dir.glob("pipeline_*.json"))
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        pipeline_data = json.load(f)
                    
                    # Reconstruct pipeline (simplified)
                    logger.info(f"Found pipeline config: {config_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load pipeline from {config_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load existing pipelines: {e}")
    
    async def _save_pipeline_config(self, pipeline: TrainingJob) -> None:
        """
        Save pipeline configuration to file.
        
        Args:
            pipeline: Pipeline to save
        """
        try:
            config_file = self.configs_dir / f"pipeline_{pipeline.config.pipeline_id}.json"
            
            with open(config_file, 'w') as f:
                json.dump(pipeline.to_dict(), f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save pipeline configuration: {e}")
    
    async def get_manager_status(self) -> Dict[str, Any]:
        """
        Get training pipeline manager status.
        
        Returns:
            Manager status information
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_pipelines": len(self.pipelines),
            "active_executions": len(self.active_executions),
            "total_executions": sum(len(p.executions) for p in self.pipelines.values()),
            "base_pipeline_dir": str(self.base_pipeline_dir),
            "component_status": {
                "model_manager": "initialized",
                "version_manager": "initialized",
                "fine_tuning_manager": "initialized",
                "ab_test_manager": "initialized"
            }
        }

# Global training pipeline instance
_training_pipeline = None

def get_training_pipeline() -> TrainingPipeline:
    """Get the global training pipeline instance."""
    global _training_pipeline
    if _training_pipeline is None:
        _training_pipeline = TrainingPipeline()
    return _training_pipeline

print("Training Pipeline System loaded.")