"""
Workflow Execution Engine

Core execution engine for running workflows with support for different step types,
parallel execution, error handling, and monitoring.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from sqlalchemy.orm import Session
from ..database.session import get_db
from ..services.agent_orchestrator import AgentOrchestrator
from ..integrations.manager import IntegrationManager
from ..monitoring.metrics.prometheus_metrics import workflow_metrics
from .executors.platform_executor import CrossPlatformOrchestrator, PlatformExecutorFactory
from .models import (
    Workflow, WorkflowExecution, StepExecution, 
    ExecutionStatus, StepType, TriggerType
)
import uuid
import traceback

logger = logging.getLogger(__name__)


class WorkflowExecutionEngine:
    """Advanced workflow execution engine with comprehensive capabilities."""
    
    def __init__(self, agent_orchestrator: AgentOrchestrator, integration_manager: IntegrationManager):
        self.agent_orchestrator = agent_orchestrator
        self.integration_manager = integration_manager
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.step_handlers: Dict[StepType, Callable] = self._init_step_handlers()
        
        # Initialize cross-platform orchestrator
        self.cross_platform_orchestrator = CrossPlatformOrchestrator()
        self._setup_default_platforms()
        
    def _init_step_handlers(self) -> Dict[StepType, Callable]:
        """Initialize step type handlers."""
        return {
            StepType.AGENT: self._execute_agent_step,
            StepType.CONDITION: self._execute_condition_step,
            StepType.PARALLEL: self._execute_parallel_step,
            StepType.TRANSFORM: self._execute_transform_step,
            StepType.DELAY: self._execute_delay_step,
            StepType.WEBHOOK: self._execute_webhook_step,
            StepType.API_CALL: self._execute_api_call_step,
            StepType.EMAIL: self._execute_email_step,
            StepType.NOTIFICATION: self._execute_notification_step
        }
    
    def _setup_default_platforms(self):
        """Setup default execution platforms."""
        try:
            # Register local execution platform
            self.cross_platform_orchestrator.register_platform("local")
            
            # Register Docker if available
            try:
                import docker
                self.cross_platform_orchestrator.register_platform("docker")
                logger.info("Docker platform registered")
            except ImportError:
                logger.info("Docker not available, skipping Docker platform")
            
            # Register Kubernetes if available
            try:
                from kubernetes import client, config
                self.cross_platform_orchestrator.register_platform("kubernetes")
                logger.info("Kubernetes platform registered")
            except ImportError:
                logger.info("Kubernetes not available, skipping Kubernetes platform")
            
            # Register AWS if credentials available
            try:
                import boto3
                # Test AWS credentials
                boto3.client('sts').get_caller_identity()
                self.cross_platform_orchestrator.register_platform("aws")
                logger.info("AWS platform registered")
            except Exception:
                logger.info("AWS credentials not available, skipping AWS platform")
                
        except Exception as e:
            logger.error(f"Error setting up platforms: {str(e)}")
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        input_data: Dict[str, Any] = None,
        trigger_type: TriggerType = TriggerType.MANUAL,
        triggered_by: str = None,
        trigger_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a workflow with comprehensive monitoring and error handling."""
        
        db = next(get_db())
        execution_id = str(uuid.uuid4())
        
        try:
            # Load workflow
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create execution record
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow_id,
                execution_id=f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{execution_id[:8]}",
                status=ExecutionStatus.RUNNING,
                input_data=input_data or {},
                trigger_type=trigger_type,
                triggered_by=triggered_by,
                trigger_data=trigger_data or {},
                context={"workflow_version": workflow.version}
            )
            
            db.add(execution)
            db.commit()
            
            # Track active execution
            self.active_executions[execution_id] = {
                "workflow_id": workflow_id,
                "execution": execution,
                "start_time": datetime.utcnow(),
                "current_step": None
            }
            
            # Start metrics tracking
            workflow_metrics.workflow_executions_total.labels(
                workflow_id=workflow_id, 
                trigger_type=trigger_type.value
            ).inc()
            
            # Execute workflow
            result = await self._execute_workflow_steps(workflow, execution, input_data or {}, db)
            
            # Update execution status
            execution.status = ExecutionStatus.COMPLETED if result["success"] else ExecutionStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.duration_ms = int((execution.completed_at - execution.started_at).total_seconds() * 1000)
            execution.output_data = result.get("output_data", {})
            
            if not result["success"]:
                execution.error_message = result.get("error", "Unknown error")
                execution.error_details = result.get("error_details", {})
            
            # Update workflow statistics
            workflow.execution_count += 1
            if result["success"]:
                workflow.success_count += 1
            else:
                workflow.failure_count += 1
            
            # Update average duration
            if workflow.execution_count > 0:
                total_duration = (workflow.avg_duration_ms * (workflow.execution_count - 1) + 
                                execution.duration_ms)
                workflow.avg_duration_ms = int(total_duration / workflow.execution_count)
            
            db.commit()
            
            # Update metrics
            workflow_metrics.workflow_execution_duration.labels(
                workflow_id=workflow_id
            ).observe(execution.duration_ms / 1000.0)
            
            if result["success"]:
                workflow_metrics.workflow_executions_successful.labels(workflow_id=workflow_id).inc()
            else:
                workflow_metrics.workflow_executions_failed.labels(workflow_id=workflow_id).inc()
            
            logger.info(f"Workflow {workflow_id} execution {execution_id} completed: {execution.status.value}")
            
            return {
                "execution_id": execution_id,
                "status": execution.status.value,
                "duration_ms": execution.duration_ms,
                "success": result["success"],
                "output_data": execution.output_data,
                "error": execution.error_message,
                "steps_executed": len(result.get("step_results", [])),
                "workflow_name": workflow.name
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update execution record
            if execution_id in self.active_executions:
                try:
                    execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
                    if execution:
                        execution.status = ExecutionStatus.FAILED
                        execution.completed_at = datetime.utcnow()
                        execution.duration_ms = int((execution.completed_at - execution.started_at).total_seconds() * 1000)
                        execution.error_message = str(e)
                        execution.error_details = {"traceback": traceback.format_exc()}
                        db.commit()
                except Exception as db_error:
                    logger.error(f"Failed to update execution record: {db_error}")
            
            workflow_metrics.workflow_executions_failed.labels(workflow_id=workflow_id).inc()
            
            return {
                "execution_id": execution_id,
                "status": "failed",
                "success": False,
                "error": str(e),
                "steps_executed": 0
            }
        
        finally:
            # Clean up active execution tracking
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            if db:
                db.close()
    
    async def _execute_workflow_steps(
        self, 
        workflow: Workflow, 
        execution: WorkflowExecution, 
        context_data: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Execute all steps in a workflow."""
        
        try:
            # Parse workflow definition
            definition = workflow.definition or {}
            steps = definition.get("steps", [])
            
            if not steps:
                return {"success": True, "output_data": context_data, "step_results": []}
            
            step_results = []
            
            # Execute steps sequentially (unless parallel step)
            for step_index, step_def in enumerate(steps):
                step_id = step_def.get("id", f"step_{step_index}")
                
                logger.info(f"Executing step {step_index + 1}/{len(steps)}: {step_id}")
                
                # Create step execution record
                step_execution = StepExecution(
                    workflow_execution_id=execution.id,
                    step_id=step_id,
                    status=ExecutionStatus.RUNNING,
                    input_data=context_data.copy()
                )
                
                db.add(step_execution)
                db.commit()
                
                try:
                    # Execute step
                    step_start = datetime.utcnow()
                    result = await self._execute_single_step(step_def, context_data, execution.id)
                    step_end = datetime.utcnow()
                    
                    # Update step execution
                    step_execution.status = ExecutionStatus.COMPLETED if result.get("success", True) else ExecutionStatus.FAILED
                    step_execution.completed_at = step_end
                    step_execution.duration_ms = int((step_end - step_start).total_seconds() * 1000)
                    step_execution.output_data = result.get("data", {})
                    
                    if not result.get("success", True):
                        step_execution.error_message = result.get("error", "Step failed")
                        step_execution.error_details = result.get("error_details", {})
                    
                    db.commit()
                    
                    # Add step result
                    step_results.append({
                        "step_id": step_id,
                        "step_index": step_index,
                        "duration_ms": step_execution.duration_ms,
                        "success": result.get("success", True),
                        "result": result
                    })
                    
                    # Update context with step results
                    if result.get("data"):
                        context_data.update(result["data"])
                    
                    # Handle step failure
                    if not result.get("success", True):
                        error_handling = step_def.get("error_handling", {})
                        if error_handling.get("stop_on_error", True):
                            return {
                                "success": False,
                                "error": f"Step {step_id} failed: {result.get('error', 'Unknown error')}",
                                "step_results": step_results,
                                "output_data": context_data
                            }
                    
                    # Check for workflow termination
                    if result.get("terminate_workflow"):
                        return {
                            "success": True,
                            "terminated": True,
                            "termination_reason": result.get("termination_reason", "Workflow terminated by step"),
                            "step_results": step_results,
                            "output_data": context_data
                        }
                
                except Exception as e:
                    logger.error(f"Step {step_id} execution failed: {str(e)}")
                    
                    # Update step execution
                    step_execution.status = ExecutionStatus.FAILED
                    step_execution.completed_at = datetime.utcnow()
                    step_execution.duration_ms = int((step_execution.completed_at - step_execution.started_at).total_seconds() * 1000)
                    step_execution.error_message = str(e)
                    step_execution.error_details = {"traceback": traceback.format_exc()}
                    db.commit()
                    
                    step_results.append({
                        "step_id": step_id,
                        "step_index": step_index,
                        "duration_ms": step_execution.duration_ms,
                        "success": False,
                        "error": str(e)
                    })
                    
                    # Check error handling
                    error_handling = step_def.get("error_handling", {})
                    if error_handling.get("stop_on_error", True):
                        return {
                            "success": False,
                            "error": f"Step {step_id} failed: {str(e)}",
                            "step_results": step_results,
                            "output_data": context_data
                        }
            
            return {
                "success": True,
                "step_results": step_results,
                "output_data": context_data
            }
            
        except Exception as e:
            logger.error(f"Workflow step execution failed: {str(e)}")
            raise
    
    async def _execute_single_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute a single workflow step."""
        
        step_type_str = step_def.get("type", "agent")
        
        # Check if this is a cross-platform step
        if "platform" in step_def:
            return await self._execute_cross_platform_step(step_def, context_data, execution_id)
        
        try:
            step_type = StepType(step_type_str)
        except ValueError:
            return {
                "success": False,
                "error": f"Unknown step type: {step_type_str}"
            }
        
        handler = self.step_handlers.get(step_type)
        if not handler:
            return {
                "success": False,
                "error": f"No handler for step type: {step_type_str}"
            }
        
        try:
            return await handler(step_def, context_data, execution_id)
        except Exception as e:
            logger.error(f"Step handler failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_details": {"traceback": traceback.format_exc()}
            }
    
    async def _execute_cross_platform_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute a step on a specific platform."""
        
        platform = step_def.get("platform", "local")
        step_config = step_def.get("config", {})
        
        try:
            # Add execution context to step config
            enhanced_config = {
                **step_config,
                "execution_id": execution_id,
                "step_id": step_def.get("id", "unknown"),
                "step_name": step_def.get("name", "Unnamed Step")
            }
            
            result = await self.cross_platform_orchestrator.execute_on_platform(
                platform=platform,
                step_config=enhanced_config,
                context_data=context_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Cross-platform step execution failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "platform": platform,
                "error_details": {"traceback": traceback.format_exc()}
            }
    
    async def _execute_agent_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute an agent step."""
        
        agent_id = step_def.get("agent_id", "auto-agent")
        framework = step_def.get("framework", "auto")
        message_template = step_def.get("message", "")
        
        try:
            # Format message with context data
            message = message_template.format(**context_data)
        except KeyError as e:
            return {
                "success": False,
                "error": f"Message template error: Missing variable {e}"
            }
        
        try:
            # Call agent
            response = await self.agent_orchestrator.process_message(
                message=message,
                agent_id=agent_id,
                framework=framework,
                context={"workflow_execution_id": execution_id, "workflow_data": context_data}
            )
            
            # Extract output data
            result_data = {}
            output_mapping = step_def.get("output_mapping", {})
            
            for output_key, source_path in output_mapping.items():
                try:
                    # Simple path extraction (e.g., "metadata.result")
                    value = response
                    for path_part in source_path.split("."):
                        value = value[path_part]
                    result_data[output_key] = value
                except (KeyError, TypeError):
                    logger.warning(f"Could not extract {source_path} from agent response")
            
            return {
                "success": True,
                "agent_response": response,
                "data": result_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Agent call failed: {str(e)}"
            }
    
    async def _execute_condition_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute a conditional step."""
        
        condition = step_def.get("condition", "")
        true_action = step_def.get("true_action", {})
        false_action = step_def.get("false_action", {})
        
        try:
            # Evaluate condition safely
            # In production, use a safe expression evaluator
            result = self._evaluate_condition(condition, context_data)
            
            action = true_action if result else false_action
            
            if action:
                return await self._execute_single_step(action, context_data, execution_id)
            else:
                return {
                    "success": True,
                    "condition_result": result,
                    "action_taken": "none"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Condition evaluation error: {str(e)}"
            }
    
    async def _execute_parallel_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute multiple steps in parallel."""
        
        parallel_steps = step_def.get("steps", [])
        
        if not parallel_steps:
            return {
                "success": True,
                "results": []
            }
        
        # Execute all steps concurrently
        tasks = [self._execute_single_step(parallel_step, context_data.copy(), execution_id) 
                for parallel_step in parallel_steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success = True
        processed_results = []
        combined_data = {}
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "step_index": i
                })
                success = False
            else:
                processed_results.append(result)
                if result.get("data"):
                    combined_data.update(result["data"])
                if not result.get("success", True):
                    success = False
        
        return {
            "success": success,
            "results": processed_results,
            "data": combined_data
        }
    
    async def _execute_transform_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute a data transformation step."""
        
        transformations = step_def.get("transformations", {})
        result_data = {}
        
        for output_key, transformation in transformations.items():
            try:
                # Safe transformation evaluation
                if isinstance(transformation, str):
                    # Simple expression evaluation
                    result = self._evaluate_expression(transformation, context_data)
                else:
                    # Direct value assignment
                    result = transformation
                
                result_data[output_key] = result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Transformation error for '{output_key}': {str(e)}"
                }
        
        return {
            "success": True,
            "data": result_data
        }
    
    async def _execute_delay_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute a delay step."""
        
        delay_seconds = step_def.get("delay_seconds", 1)
        delay_type = step_def.get("delay_type", "fixed")  # fixed, random, exponential
        
        try:
            if delay_type == "fixed":
                await asyncio.sleep(delay_seconds)
            elif delay_type == "random":
                import random
                max_delay = step_def.get("max_delay_seconds", delay_seconds * 2)
                actual_delay = random.uniform(delay_seconds, max_delay)
                await asyncio.sleep(actual_delay)
            elif delay_type == "exponential":
                # Exponential backoff based on retry count or step index
                retry_count = context_data.get("retry_count", 0)
                actual_delay = delay_seconds * (2 ** retry_count)
                await asyncio.sleep(min(actual_delay, step_def.get("max_delay_seconds", 60)))
            
            return {
                "success": True,
                "data": {"delay_completed": True, "delay_seconds": delay_seconds}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Delay step failed: {str(e)}"
            }
    
    async def _execute_webhook_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute a webhook step."""
        
        import aiohttp
        
        url = step_def.get("url", "")
        method = step_def.get("method", "POST").upper()
        headers = step_def.get("headers", {})
        payload = step_def.get("payload", {})
        
        try:
            # Format URL and payload with context data
            formatted_url = url.format(**context_data)
            formatted_payload = self._format_json_template(payload, context_data)
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=formatted_url,
                    json=formatted_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                    
                    return {
                        "success": response.status < 400,
                        "data": {
                            "status_code": response.status,
                            "response_data": response_data,
                            "headers": dict(response.headers)
                        }
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Webhook call failed: {str(e)}"
            }
    
    async def _execute_api_call_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute an API call step."""
        
        # Use integration manager for API calls
        integration_name = step_def.get("integration", "")
        action = step_def.get("action", "")
        parameters = step_def.get("parameters", {})
        
        try:
            # Format parameters with context data
            formatted_params = self._format_json_template(parameters, context_data)
            
            # Call integration
            result = await self.integration_manager.call_integration(
                integration_name=integration_name,
                action=action,
                parameters=formatted_params
            )
            
            return {
                "success": True,
                "data": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }
    
    async def _execute_email_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute an email step."""
        
        # Email functionality would integrate with email service
        to_emails = step_def.get("to", [])
        subject = step_def.get("subject", "")
        body = step_def.get("body", "")
        
        try:
            # Format email content
            formatted_subject = subject.format(**context_data)
            formatted_body = body.format(**context_data)
            
            # TODO: Integrate with email service (SendGrid, AWS SES, etc.)
            logger.info(f"Email step: Sending to {to_emails}, Subject: {formatted_subject}")
            
            return {
                "success": True,
                "data": {
                    "email_sent": True,
                    "recipients": to_emails,
                    "subject": formatted_subject
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Email step failed: {str(e)}"
            }
    
    async def _execute_notification_step(self, step_def: Dict[str, Any], context_data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Execute a notification step."""
        
        notification_type = step_def.get("type", "info")  # info, warning, error, success
        message = step_def.get("message", "")
        recipients = step_def.get("recipients", [])
        
        try:
            # Format message
            formatted_message = message.format(**context_data)
            
            # TODO: Integrate with notification service
            logger.info(f"Notification: {notification_type} - {formatted_message}")
            
            return {
                "success": True,
                "data": {
                    "notification_sent": True,
                    "type": notification_type,
                    "message": formatted_message,
                    "recipients": recipients
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Notification step failed: {str(e)}"
            }
    
    def _evaluate_condition(self, condition: str, context_data: Dict[str, Any]) -> bool:
        """Safely evaluate a condition."""
        
        # Simple condition evaluation
        # In production, use a proper expression evaluator like py-expression-eval
        try:
            # Basic safety check
            if any(forbidden in condition for forbidden in ['import', '__', 'exec', 'eval', 'open', 'file']):
                raise ValueError("Unsafe condition expression")
            
            # Create safe namespace
            safe_namespace = {
                '__builtins__': {},
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round
            }
            safe_namespace.update(context_data)
            
            return bool(eval(condition, safe_namespace))
        
        except Exception as e:
            logger.error(f"Condition evaluation failed: {str(e)}")
            return False
    
    def _evaluate_expression(self, expression: str, context_data: Dict[str, Any]) -> Any:
        """Safely evaluate an expression."""
        
        try:
            # Basic safety check
            if any(forbidden in expression for forbidden in ['import', '__', 'exec', 'eval', 'open', 'file']):
                raise ValueError("Unsafe expression")
            
            # Create safe namespace
            safe_namespace = {
                '__builtins__': {},
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'datetime': datetime,
                'json': json
            }
            safe_namespace.update(context_data)
            
            return eval(expression, safe_namespace)
        
        except Exception as e:
            logger.error(f"Expression evaluation failed: {str(e)}")
            raise
    
    def _format_json_template(self, template: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format a JSON template with context data."""
        
        def format_value(value):
            if isinstance(value, str):
                try:
                    return value.format(**context_data)
                except:
                    return value
            elif isinstance(value, dict):
                return {k: format_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [format_value(item) for item in value]
            else:
                return value
        
        return format_value(template)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        
        if execution_id not in self.active_executions:
            return False
        
        try:
            db = next(get_db())
            execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
            
            if execution:
                execution.status = ExecutionStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                execution.duration_ms = int((execution.completed_at - execution.started_at).total_seconds() * 1000)
                db.commit()
            
            # Remove from active executions
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            workflow_metrics.workflow_executions_cancelled.labels(
                workflow_id=self.active_executions[execution_id]["workflow_id"]
            ).inc()
            
            logger.info(f"Workflow execution {execution_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id}: {str(e)}")
            return False
        finally:
            db.close()
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active workflow executions."""
        
        return {
            exec_id: {
                "workflow_id": info["workflow_id"],
                "started_at": info["start_time"].isoformat(),
                "current_step": info["current_step"],
                "duration_ms": int((datetime.utcnow() - info["start_time"]).total_seconds() * 1000)
            }
            for exec_id, info in self.active_executions.items()
        }
    
    async def get_platform_status(self) -> Dict[str, Any]:
        """Get status of all registered execution platforms."""
        
        health_status = await self.cross_platform_orchestrator.health_check_all_platforms()
        platform_info = self.cross_platform_orchestrator.get_platform_info()
        
        return {
            "platforms": {
                platform: {
                    "healthy": health_status.get(platform, False),
                    "info": platform_info.get(platform, {})
                }
                for platform in PlatformExecutorFactory.get_supported_platforms()
            },
            "total_platforms": len(platform_info),
            "healthy_platforms": sum(1 for healthy in health_status.values() if healthy)
        }
    
    async def execute_multi_platform_workflow(
        self, 
        workflow_id: str,
        platform_steps: List[Dict[str, Any]],
        input_data: Dict[str, Any] = None,
        trigger_type: TriggerType = TriggerType.MANUAL,
        triggered_by: str = None
    ) -> Dict[str, Any]:
        """Execute a workflow across multiple platforms simultaneously."""
        
        execution_id = str(uuid.uuid4())
        
        try:
            # Execute steps across platforms
            results = await self.cross_platform_orchestrator.execute_multi_platform(
                platform_steps=platform_steps,
                context_data=input_data or {}
            )
            
            # Analyze results
            total_steps = len(results)
            successful_steps = sum(1 for result in results if result.get("success", False))
            failed_steps = total_steps - successful_steps
            
            overall_success = failed_steps == 0
            
            # Calculate total duration
            total_duration = max(
                result.get("execution_duration_ms", 0) for result in results
            ) if results else 0
            
            return {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "success": overall_success,
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "total_duration_ms": total_duration,
                "platform_results": results,
                "trigger_type": trigger_type.value,
                "triggered_by": triggered_by
            }
            
        except Exception as e:
            logger.error(f"Multi-platform workflow execution failed: {str(e)}")
            return {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "success": False,
                "error": str(e),
                "platform_results": []
            }
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow execution."""
        
        try:
            db = next(get_db())
            execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
            
            if not execution:
                return None
            
            return execution.to_dict()
            
        except Exception as e:
            logger.error(f"Failed to get execution status: {str(e)}")
            return None
        finally:
            db.close()