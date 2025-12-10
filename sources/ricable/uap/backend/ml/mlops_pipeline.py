# backend/ml/mlops_pipeline.py
# MLOps CI/CD Pipeline for Model Development and Deployment

import asyncio
import json
import uuid
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import yaml

# Integrations
from .model_manager import ModelManager, get_model_manager, ModelLifecycleStage
from .model_validator import ModelValidator, get_model_validator
from ..ai.training_pipeline import TrainingPipeline, get_training_pipeline
from ..distributed.ray_manager import submit_distributed_task
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.prometheus_metrics import record_pipeline_event

class PipelineType(Enum):
    """MLOps pipeline types"""
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    ROLLBACK = "rollback"

class PipelineStatus(Enum):
    """Pipeline execution status"""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TriggerType(Enum):
    """Pipeline trigger types"""
    MANUAL = "manual"
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CODE_COMMIT = "code_commit"

@dataclass
class PipelineStep:
    """Individual pipeline step configuration"""
    step_id: str
    step_name: str
    step_type: str
    command: str
    environment: Dict[str, str]
    dependencies: List[str]
    timeout_seconds: int
    retry_attempts: int
    continue_on_failure: bool
    artifacts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PipelineConfig:
    """MLOps pipeline configuration"""
    pipeline_id: str
    pipeline_name: str
    pipeline_type: PipelineType
    description: str
    steps: List[PipelineStep]
    triggers: List[TriggerType]
    environment_variables: Dict[str, str]
    notifications: Dict[str, Any]
    schedule: Optional[str]
    timeout_minutes: int
    retry_policy: Dict[str, Any]
    created_by: str
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['pipeline_type'] = self.pipeline_type.value
        data['triggers'] = [t.value for t in self.triggers]
        data['created_at'] = self.created_at.isoformat()
        data['steps'] = [step.to_dict() for step in self.steps]
        return data

@dataclass
class PipelineExecution:
    """Pipeline execution record"""
    execution_id: str
    pipeline_id: str
    status: PipelineStatus
    trigger_type: TriggerType
    triggered_by: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    step_results: Dict[str, Dict[str, Any]]
    artifacts: Dict[str, str]
    logs: List[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['trigger_type'] = self.trigger_type.value
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

class PipelineOrchestrator:
    """Orchestrate pipeline step execution"""
    
    def __init__(self, workspace_dir: str = "./pipelines"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
    async def execute_pipeline(self, 
                             config: PipelineConfig,
                             execution: PipelineExecution) -> Dict[str, Any]:
        """Execute pipeline steps"""
        
        execution.status = PipelineStatus.RUNNING
        execution.started_at = datetime.utcnow()
        
        try:
            # Create execution workspace
            exec_workspace = self.workspace_dir / execution.execution_id
            exec_workspace.mkdir(exist_ok=True)
            
            # Execute steps in order
            for step in config.steps:
                # Check dependencies
                if not await self._check_step_dependencies(step, execution.step_results):
                    raise Exception(f"Step {step.step_name} dependencies not met")
                
                # Execute step
                step_result = await self._execute_step(step, config, execution, exec_workspace)
                execution.step_results[step.step_id] = step_result
                
                if not step_result.get("success", False) and not step.continue_on_failure:
                    raise Exception(f"Step {step.step_name} failed: {step_result.get('error', 'Unknown error')}")
            
            # Pipeline completed successfully
            execution.status = PipelineStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            
            return {
                "success": True,
                "execution_id": execution.execution_id,
                "step_results": execution.step_results,
                "artifacts": execution.artifacts
            }
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            return {
                "success": False,
                "execution_id": execution.execution_id,
                "error": str(e)
            }
    
    async def _check_step_dependencies(self, 
                                     step: PipelineStep,
                                     step_results: Dict[str, Dict[str, Any]]) -> bool:
        """Check if step dependencies are satisfied"""
        for dependency in step.dependencies:
            if dependency not in step_results:
                return False
            if not step_results[dependency].get("success", False):
                return False
        return True
    
    async def _execute_step(self,
                          step: PipelineStep,
                          config: PipelineConfig,
                          execution: PipelineExecution,
                          workspace: Path) -> Dict[str, Any]:
        """Execute individual pipeline step"""
        
        start_time = datetime.utcnow()
        
        try:
            # Prepare environment
            env = {**config.environment_variables, **step.environment}
            env["PIPELINE_ID"] = config.pipeline_id
            env["EXECUTION_ID"] = execution.execution_id
            env["WORKSPACE_DIR"] = str(workspace)
            
            # Log step start
            execution.logs.append(f"[{start_time.isoformat()}] Starting step: {step.step_name}")
            
            # Execute command based on step type
            if step.step_type == "shell":
                result = await self._execute_shell_command(step, env, workspace)
            elif step.step_type == "python":
                result = await self._execute_python_script(step, env, workspace)
            elif step.step_type == "docker":
                result = await self._execute_docker_command(step, env, workspace)
            elif step.step_type == "training":
                result = await self._execute_training_step(step, env, execution)
            elif step.step_type == "validation":
                result = await self._execute_validation_step(step, env, execution)
            elif step.step_type == "deployment":
                result = await self._execute_deployment_step(step, env, execution)
            else:
                result = {"success": False, "error": f"Unknown step type: {step.step_type}"}
            
            # Collect artifacts
            await self._collect_step_artifacts(step, workspace, execution)
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            result["execution_time_seconds"] = execution_time
            result["start_time"] = start_time.isoformat()
            result["end_time"] = end_time.isoformat()
            
            execution.logs.append(f"[{end_time.isoformat()}] Step completed: {step.step_name} ({execution_time:.2f}s)")
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            execution.logs.append(f"[{end_time.isoformat()}] Step failed: {step.step_name} - {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time_seconds": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
    
    async def _execute_shell_command(self,
                                   step: PipelineStep,
                                   env: Dict[str, str],
                                   workspace: Path) -> Dict[str, Any]:
        """Execute shell command"""
        
        process = await asyncio.create_subprocess_shell(
            step.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=workspace
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=step.timeout_seconds
            )
            
            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else ""
            }
            
        except asyncio.TimeoutError:
            process.kill()
            return {
                "success": False,
                "error": f"Command timeout after {step.timeout_seconds} seconds"
            }
    
    async def _execute_python_script(self,
                                   step: PipelineStep,
                                   env: Dict[str, str],
                                   workspace: Path) -> Dict[str, Any]:
        """Execute Python script"""
        
        # Assume command is a Python script path
        python_command = f"python {step.command}"
        
        process = await asyncio.create_subprocess_shell(
            python_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=workspace
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=step.timeout_seconds
            )
            
            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else ""
            }
            
        except asyncio.TimeoutError:
            process.kill()
            return {
                "success": False,
                "error": f"Python script timeout after {step.timeout_seconds} seconds"
            }
    
    async def _execute_docker_command(self,
                                    step: PipelineStep,
                                    env: Dict[str, str],
                                    workspace: Path) -> Dict[str, Any]:
        """Execute Docker command"""
        
        # Build Docker command with environment variables
        env_args = " ".join([f"-e {k}={v}" for k, v in env.items()])
        docker_command = f"docker run --rm -v {workspace}:/workspace {env_args} {step.command}"
        
        process = await asyncio.create_subprocess_shell(
            docker_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=step.timeout_seconds
            )
            
            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else ""
            }
            
        except asyncio.TimeoutError:
            process.kill()
            return {
                "success": False,
                "error": f"Docker command timeout after {step.timeout_seconds} seconds"
            }
    
    async def _execute_training_step(self,
                                   step: PipelineStep,
                                   env: Dict[str, str],
                                   execution: PipelineExecution) -> Dict[str, Any]:
        """Execute model training step"""
        
        try:
            training_pipeline = get_training_pipeline()
            
            # Parse training parameters from environment
            model_id = env.get("MODEL_ID", "default_model")
            dataset_path = env.get("DATASET_PATH", "")
            
            # Execute training
            pipeline_id = await training_pipeline.create_pipeline(
                pipeline_name=f"MLOps_Training_{execution.execution_id}",
                description=f"Training initiated by MLOps pipeline {execution.pipeline_id}",
                model_id=model_id,
                base_model_path=env.get("BASE_MODEL_PATH", ""),
                dataset_path=dataset_path,
                dataset_type=env.get("DATASET_TYPE", "text"),
                created_by="mlops_pipeline"
            )
            
            # Start training execution
            training_execution_id = await training_pipeline.execute_pipeline(
                pipeline_id=pipeline_id,
                triggered_by="mlops_pipeline"
            )
            
            return {
                "success": True,
                "training_pipeline_id": pipeline_id,
                "training_execution_id": training_execution_id,
                "model_id": model_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_validation_step(self,
                                     step: PipelineStep,
                                     env: Dict[str, str],
                                     execution: PipelineExecution) -> Dict[str, Any]:
        """Execute model validation step"""
        
        try:
            model_validator = get_model_validator()
            
            model_id = env.get("MODEL_ID", "")
            model_version = env.get("MODEL_VERSION", "")
            model_path = env.get("MODEL_PATH", "")
            
            # Execute validation
            validation_result = await model_validator.validate_model(
                model_id=model_id,
                model_version=model_version,
                model_path=model_path,
                validation_config={
                    "test_dataset": env.get("TEST_DATASET", ""),
                    "metrics": ["accuracy", "precision", "recall"],
                    "thresholds": {
                        "min_accuracy": float(env.get("MIN_ACCURACY", "0.8")),
                        "max_latency_ms": float(env.get("MAX_LATENCY_MS", "100"))
                    }
                }
            )
            
            return {
                "success": validation_result.get("passed", False),
                "validation_results": validation_result,
                "model_id": model_id,
                "model_version": model_version
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_deployment_step(self,
                                     step: PipelineStep,
                                     env: Dict[str, str],
                                     execution: PipelineExecution) -> Dict[str, Any]:
        """Execute model deployment step"""
        
        try:
            model_manager = get_model_manager()
            
            model_id = env.get("MODEL_ID", "")
            model_version = env.get("MODEL_VERSION", "")
            target_stage = ModelLifecycleStage(env.get("TARGET_STAGE", "staging"))
            
            # Execute deployment
            deployment_id = await model_manager.deploy_model(
                model_id=model_id,
                version=model_version,
                stage=target_stage,
                traffic_percentage=float(env.get("TRAFFIC_PERCENTAGE", "0.0")),
                approval_required=env.get("APPROVAL_REQUIRED", "true").lower() == "true"
            )
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "model_id": model_id,
                "model_version": model_version,
                "target_stage": target_stage.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _collect_step_artifacts(self,
                                    step: PipelineStep,
                                    workspace: Path,
                                    execution: PipelineExecution):
        """Collect step artifacts"""
        
        for artifact_pattern in step.artifacts:
            try:
                # Find matching files
                artifact_files = list(workspace.glob(artifact_pattern))
                
                for artifact_file in artifact_files:
                    # Store artifact reference
                    artifact_key = f"{step.step_id}_{artifact_file.name}"
                    execution.artifacts[artifact_key] = str(artifact_file)
                    
            except Exception as e:
                execution.logs.append(f"Failed to collect artifact {artifact_pattern}: {str(e)}")

class MLOpsPipeline:
    """
    MLOps CI/CD Pipeline for Model Development and Deployment.
    
    Provides:
    - Automated training pipelines
    - Model validation and testing
    - Deployment automation
    - Rollback capabilities
    - Multi-environment support
    """
    
    def __init__(self, config_dir: str = "./mlops_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.orchestrator = PipelineOrchestrator()
        self.model_manager = get_model_manager()
        
        # Pipeline management
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.active_executions: Dict[str, PipelineExecution] = {}
        
        # Monitoring
        self.monitoring_active = False
    
    async def initialize(self) -> bool:
        """Initialize MLOps pipeline"""
        try:
            await self._load_pipeline_configs()
            
            # Start monitoring
            asyncio.create_task(self._start_monitoring())
            
            uap_logger.log_event(
                LogLevel.INFO,
                "MLOps Pipeline initialized",
                EventType.AGENT,
                {"config_dir": str(self.config_dir)},
                "mlops_pipeline"
            )
            
            return True
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to initialize MLOps Pipeline: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "mlops_pipeline"
            )
            return False
    
    async def create_pipeline(self,
                            pipeline_name: str,
                            pipeline_type: PipelineType,
                            description: str,
                            steps: List[Dict[str, Any]],
                            triggers: List[TriggerType] = None,
                            environment_variables: Dict[str, str] = None,
                            created_by: str = "system") -> str:
        """Create MLOps pipeline"""
        
        pipeline_id = str(uuid.uuid4())
        
        # Create pipeline steps
        pipeline_steps = []
        for i, step_data in enumerate(steps):
            step = PipelineStep(
                step_id=step_data.get("step_id", f"step_{i}"),
                step_name=step_data["step_name"],
                step_type=step_data["step_type"],
                command=step_data["command"],
                environment=step_data.get("environment", {}),
                dependencies=step_data.get("dependencies", []),
                timeout_seconds=step_data.get("timeout_seconds", 3600),
                retry_attempts=step_data.get("retry_attempts", 1),
                continue_on_failure=step_data.get("continue_on_failure", False),
                artifacts=step_data.get("artifacts", [])
            )
            pipeline_steps.append(step)
        
        # Create pipeline configuration
        config = PipelineConfig(
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            pipeline_type=pipeline_type,
            description=description,
            steps=pipeline_steps,
            triggers=triggers or [TriggerType.MANUAL],
            environment_variables=environment_variables or {},
            notifications={},
            schedule=None,
            timeout_minutes=60,
            retry_policy={"max_attempts": 3, "backoff_seconds": 60},
            created_by=created_by,
            created_at=datetime.utcnow()
        )
        
        self.pipelines[pipeline_id] = config
        
        # Save configuration
        await self._save_pipeline_config(config)
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"MLOps pipeline created: {pipeline_name}",
            EventType.AGENT,
            {
                "pipeline_id": pipeline_id,
                "pipeline_type": pipeline_type.value,
                "steps_count": len(pipeline_steps)
            },
            "mlops_pipeline"
        )
        
        return pipeline_id
    
    async def execute_pipeline(self,
                             pipeline_id: str,
                             trigger_type: TriggerType = TriggerType.MANUAL,
                             triggered_by: str = "system",
                             parameters: Dict[str, Any] = None) -> str:
        """Execute MLOps pipeline"""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        config = self.pipelines[pipeline_id]
        execution_id = str(uuid.uuid4())
        
        # Create execution record
        execution = PipelineExecution(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            status=PipelineStatus.QUEUED,
            trigger_type=trigger_type,
            triggered_by=triggered_by,
            started_at=None,
            completed_at=None,
            step_results={},
            artifacts={},
            logs=[],
            error_message=None,
            metadata=parameters or {}
        )
        
        self.executions[execution_id] = execution
        self.active_executions[execution_id] = execution
        
        # Submit execution as distributed task
        task_id = await submit_distributed_task(
            "mlops_pipeline_execution",
            self.orchestrator.execute_pipeline,
            {"config": config, "execution": execution},
            priority=8
        )
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"MLOps pipeline execution started: {config.pipeline_name}",
            EventType.AGENT,
            {
                "execution_id": execution_id,
                "pipeline_id": pipeline_id,
                "trigger_type": trigger_type.value,
                "task_id": task_id
            },
            "mlops_pipeline"
        )
        
        return execution_id
    
    async def create_training_pipeline(self,
                                     model_id: str,
                                     dataset_path: str,
                                     base_model_path: str = "",
                                     validation_split: float = 0.1,
                                     min_accuracy: float = 0.8) -> str:
        """Create standard training pipeline"""
        
        steps = [
            {
                "step_name": "Data Validation",
                "step_type": "python",
                "command": "scripts/validate_data.py",
                "environment": {
                    "DATASET_PATH": dataset_path,
                    "VALIDATION_SPLIT": str(validation_split)
                },
                "artifacts": ["data_validation_report.json"]
            },
            {
                "step_name": "Model Training",
                "step_type": "training",
                "command": "train_model",
                "environment": {
                    "MODEL_ID": model_id,
                    "DATASET_PATH": dataset_path,
                    "BASE_MODEL_PATH": base_model_path
                },
                "dependencies": ["step_0"],
                "timeout_seconds": 7200,
                "artifacts": ["model_*.pkl", "training_logs.txt"]
            },
            {
                "step_name": "Model Validation",
                "step_type": "validation",
                "command": "validate_model",
                "environment": {
                    "MODEL_ID": model_id,
                    "MIN_ACCURACY": str(min_accuracy)
                },
                "dependencies": ["step_1"],
                "artifacts": ["validation_report.json"]
            },
            {
                "step_name": "Deploy to Staging",
                "step_type": "deployment",
                "command": "deploy_model",
                "environment": {
                    "MODEL_ID": model_id,
                    "TARGET_STAGE": "staging",
                    "APPROVAL_REQUIRED": "false"
                },
                "dependencies": ["step_2"],
                "continue_on_failure": True
            }
        ]
        
        return await self.create_pipeline(
            pipeline_name=f"Training_Pipeline_{model_id}",
            pipeline_type=PipelineType.TRAINING,
            description=f"Automated training pipeline for {model_id}",
            steps=steps,
            triggers=[TriggerType.MANUAL, TriggerType.CODE_COMMIT]
        )
    
    async def create_deployment_pipeline(self,
                                       model_id: str,
                                       source_stage: str = "staging",
                                       target_stage: str = "production",
                                       traffic_percentage: float = 0.0) -> str:
        """Create standard deployment pipeline"""
        
        steps = [
            {
                "step_name": "Pre-deployment Validation",
                "step_type": "validation",
                "command": "validate_deployment",
                "environment": {
                    "MODEL_ID": model_id,
                    "SOURCE_STAGE": source_stage
                },
                "artifacts": ["deployment_validation.json"]
            },
            {
                "step_name": "Deploy to Production",
                "step_type": "deployment",
                "command": "deploy_model",
                "environment": {
                    "MODEL_ID": model_id,
                    "TARGET_STAGE": target_stage,
                    "TRAFFIC_PERCENTAGE": str(traffic_percentage),
                    "APPROVAL_REQUIRED": "true"
                },
                "dependencies": ["step_0"]
            },
            {
                "step_name": "Post-deployment Monitoring",
                "step_type": "shell",
                "command": "scripts/setup_monitoring.sh",
                "environment": {
                    "MODEL_ID": model_id,
                    "DEPLOYMENT_STAGE": target_stage
                },
                "dependencies": ["step_1"],
                "continue_on_failure": True
            }
        ]
        
        return await self.create_pipeline(
            pipeline_name=f"Deployment_Pipeline_{model_id}",
            pipeline_type=PipelineType.DEPLOYMENT,
            description=f"Automated deployment pipeline for {model_id}",
            steps=steps,
            triggers=[TriggerType.MANUAL]
        )
    
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status"""
        if pipeline_id not in self.pipelines:
            return None
        
        config = self.pipelines[pipeline_id]
        
        # Get recent executions
        recent_executions = [
            exec.to_dict() for exec in self.executions.values()
            if exec.pipeline_id == pipeline_id
        ]
        recent_executions.sort(key=lambda x: x['started_at'] or '', reverse=True)
        
        return {
            "config": config.to_dict(),
            "recent_executions": recent_executions[:10],
            "active_executions": len([e for e in self.active_executions.values() if e.pipeline_id == pipeline_id])
        }
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        if execution_id not in self.executions:
            return None
        
        return self.executions[execution_id].to_dict()
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel pipeline execution"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = PipelineStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        execution.error_message = "Execution cancelled by user"
        
        # Remove from active executions
        del self.active_executions[execution_id]
        
        return True
    
    async def _start_monitoring(self):
        """Start background monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                await self._monitor_executions()
                await self._cleanup_old_executions()
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"MLOps pipeline monitoring error: {e}",
                    EventType.AGENT,
                    {"error": str(e)},
                    "mlops_pipeline"
                )
                await asyncio.sleep(300)
    
    async def _monitor_executions(self):
        """Monitor active executions"""
        for execution_id, execution in list(self.active_executions.items()):
            # Check for timeouts
            if execution.started_at:
                elapsed = datetime.utcnow() - execution.started_at
                if elapsed > timedelta(hours=2):  # 2 hour timeout
                    execution.status = PipelineStatus.FAILED
                    execution.error_message = "Execution timeout"
                    execution.completed_at = datetime.utcnow()
                    del self.active_executions[execution_id]
    
    async def _cleanup_old_executions(self):
        """Clean up old execution records"""
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        executions_to_remove = []
        for execution_id, execution in self.executions.items():
            if (execution.completed_at and 
                execution.completed_at < cutoff_date and
                execution_id not in self.active_executions):
                executions_to_remove.append(execution_id)
        
        for execution_id in executions_to_remove:
            del self.executions[execution_id]
    
    async def _load_pipeline_configs(self):
        """Load pipeline configurations from files"""
        try:
            config_files = list(self.config_dir.glob("*.yaml"))
            
            for config_file in config_files:
                try:
                    async with aiofiles.open(config_file, 'r') as f:
                        config_data = yaml.safe_load(await f.read())
                        # Reconstruct pipeline config (simplified)
                        pass
                except Exception as e:
                    uap_logger.log_event(
                        LogLevel.WARNING,
                        f"Failed to load pipeline config {config_file}: {e}",
                        EventType.AGENT,
                        {"config_file": str(config_file), "error": str(e)},
                        "mlops_pipeline"
                    )
                    
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to load pipeline configurations: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "mlops_pipeline"
            )
    
    async def _save_pipeline_config(self, config: PipelineConfig):
        """Save pipeline configuration to file"""
        try:
            config_file = self.config_dir / f"{config.pipeline_id}.yaml"
            
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(yaml.dump(config.to_dict(), default_flow_style=False))
                
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to save pipeline configuration: {e}",
                EventType.AGENT,
                {"pipeline_id": config.pipeline_id, "error": str(e)},
                "mlops_pipeline"
            )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get MLOps system status"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_pipelines": len(self.pipelines),
            "active_executions": len(self.active_executions),
            "total_executions": len(self.executions),
            "monitoring_active": self.monitoring_active,
            "config_dir": str(self.config_dir)
        }

# Global MLOps pipeline instance
_mlops_pipeline = None

def get_mlops_pipeline() -> MLOpsPipeline:
    """Get the global MLOps pipeline instance"""
    global _mlops_pipeline
    if _mlops_pipeline is None:
        _mlops_pipeline = MLOpsPipeline()
    return _mlops_pipeline