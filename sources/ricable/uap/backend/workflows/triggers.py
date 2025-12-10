"""
Workflow Trigger Manager

Advanced trigger system for workflows supporting webhooks, events, conditions,
file uploads, API calls, and custom triggers.
"""

import asyncio
import logging
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from sqlalchemy.orm import Session
from ..database.session import get_db
from ..monitoring.metrics.prometheus_metrics import workflow_metrics
from .models import Workflow, WorkflowTrigger, TriggerType, WorkflowStatus
from .execution_engine import WorkflowExecutionEngine
import uuid
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)


class TriggerManager:
    """Advanced trigger management system for workflows."""
    
    def __init__(self, execution_engine: WorkflowExecutionEngine):
        self.execution_engine = execution_engine
        self.event_listeners: Dict[str, Set[str]] = {}  # event_type -> set of trigger_ids
        self.condition_triggers: Dict[str, Dict[str, Any]] = {}  # trigger_id -> condition info
        self.webhook_secrets: Dict[str, str] = {}  # webhook_url -> secret
        self.custom_handlers: Dict[str, Callable] = {}
        self.file_watchers: Dict[str, Dict[str, Any]] = {}  # path -> watcher info
        
    async def create_webhook_trigger(
        self, 
        workflow_id: str, 
        trigger_name: str,
        webhook_path: str = None,
        secret: str = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a webhook trigger for a workflow."""
        
        # Generate webhook URL
        if not webhook_path:
            webhook_path = f"/webhooks/workflow/{str(uuid.uuid4())}"
        
        webhook_url = f"http://localhost:8000{webhook_path}"  # Should be configurable
        
        # Generate secret if not provided
        if not secret:
            secret = str(uuid.uuid4())
        
        db = next(get_db())
        try:
            # Check if workflow exists
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create trigger
            trigger = WorkflowTrigger(
                workflow_id=workflow_id,
                name=trigger_name,
                trigger_type=TriggerType.WEBHOOK,
                webhook_url=webhook_url,
                webhook_secret=secret,
                config=config or {},
                is_active=True
            )
            
            db.add(trigger)
            db.commit()
            
            # Store webhook secret for validation
            self.webhook_secrets[webhook_url] = secret
            
            logger.info(f"Created webhook trigger {trigger.id} for workflow {workflow_id}")
            
            return {
                "trigger_id": trigger.id,
                "webhook_url": webhook_url,
                "webhook_path": webhook_path,
                "secret": secret
            }
            
        finally:
            db.close()
    
    async def handle_webhook(self, webhook_path: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle incoming webhook requests."""
        
        webhook_url = f"http://localhost:8000{webhook_path}"
        
        db = next(get_db())
        try:
            # Find trigger by webhook URL
            trigger = db.query(WorkflowTrigger).filter(
                WorkflowTrigger.webhook_url == webhook_url,
                WorkflowTrigger.trigger_type == TriggerType.WEBHOOK,
                WorkflowTrigger.is_active == True
            ).first()
            
            if not trigger:
                return {"success": False, "error": "Webhook trigger not found"}
            
            # Validate webhook signature if secret is provided
            if trigger.webhook_secret:
                signature = headers.get('x-hub-signature-256') or headers.get('x-signature')
                if not self._validate_webhook_signature(payload, trigger.webhook_secret, signature):
                    return {"success": False, "error": "Invalid webhook signature"}
            
            # Get workflow
            workflow = db.query(Workflow).filter(Workflow.id == trigger.workflow_id).first()
            if not workflow or workflow.status != WorkflowStatus.ACTIVE:
                return {"success": False, "error": "Workflow not found or not active"}
            
            # Prepare trigger data
            trigger_data = {
                "trigger_id": trigger.id,
                "trigger_name": trigger.name,
                "webhook_url": webhook_url,
                "webhook_payload": payload,
                "webhook_headers": dict(headers),
                "received_at": datetime.utcnow().isoformat()
            }
            
            # Add trigger configuration
            if trigger.config:
                trigger_data.update(trigger.config)
            
            # Execute workflow asynchronously
            asyncio.create_task(self._execute_webhook_workflow(trigger, workflow, trigger_data, payload))
            
            # Update trigger statistics
            trigger.execution_count += 1
            trigger.last_executed_at = datetime.utcnow()
            db.commit()
            
            return {
                "success": True,
                "trigger_id": trigger.id,
                "workflow_id": workflow.id,
                "message": "Webhook received and workflow execution started"
            }
            
        finally:
            db.close()
    
    async def _execute_webhook_workflow(
        self, 
        trigger: WorkflowTrigger, 
        workflow: Workflow, 
        trigger_data: Dict[str, Any], 
        payload: Dict[str, Any]
    ):
        """Execute workflow from webhook trigger."""
        
        try:
            result = await self.execution_engine.execute_workflow(
                workflow_id=workflow.id,
                input_data=payload,
                trigger_type=TriggerType.WEBHOOK,
                triggered_by="webhook",
                trigger_data=trigger_data
            )
            
            workflow_metrics.webhook_executions_total.labels(
                workflow_id=workflow.id,
                trigger_id=trigger.id
            ).inc()
            
            if result["success"]:
                workflow_metrics.webhook_executions_successful.labels(
                    workflow_id=workflow.id
                ).inc()
            else:
                workflow_metrics.webhook_executions_failed.labels(
                    workflow_id=workflow.id
                ).inc()
            
            logger.info(f"Webhook workflow {workflow.id} executed: {result['execution_id']}")
            
        except Exception as e:
            logger.error(f"Failed to execute webhook workflow {workflow.id}: {str(e)}")
            workflow_metrics.webhook_executions_failed.labels(
                workflow_id=workflow.id
            ).inc()
    
    def _validate_webhook_signature(self, payload: Dict[str, Any], secret: str, signature: str) -> bool:
        """Validate webhook signature."""
        
        if not signature:
            return False
        
        try:
            # Convert payload to JSON string
            payload_str = json.dumps(payload, sort_keys=True)
            
            # Calculate expected signature
            expected_signature = hmac.new(
                secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Handle different signature formats
            if signature.startswith('sha256='):
                signature = signature[7:]
            
            return hmac.compare_digest(expected_signature, signature)
            
        except Exception as e:
            logger.error(f"Webhook signature validation error: {str(e)}")
            return False
    
    async def create_event_trigger(
        self, 
        workflow_id: str, 
        trigger_name: str,
        event_type: str,
        conditions: Dict[str, Any] = None,
        config: Dict[str, Any] = None
    ) -> str:
        """Create an event-based trigger for a workflow."""
        
        db = next(get_db())
        try:
            # Check if workflow exists
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create trigger
            trigger = WorkflowTrigger(
                workflow_id=workflow_id,
                name=trigger_name,
                trigger_type=TriggerType.EVENT,
                conditions=conditions or {},
                config=config or {},
                is_active=True
            )
            
            # Store event type in config
            trigger.config["event_type"] = event_type
            
            db.add(trigger)
            db.commit()
            
            # Register event listener
            if event_type not in self.event_listeners:
                self.event_listeners[event_type] = set()
            self.event_listeners[event_type].add(trigger.id)
            
            logger.info(f"Created event trigger {trigger.id} for event type {event_type}")
            return trigger.id
            
        finally:
            db.close()
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Emit an event and trigger matching workflows."""
        
        if event_type not in self.event_listeners:
            return []
        
        results = []
        db = next(get_db())
        
        try:
            trigger_ids = list(self.event_listeners[event_type])
            triggers = db.query(WorkflowTrigger).filter(
                WorkflowTrigger.id.in_(trigger_ids),
                WorkflowTrigger.is_active == True
            ).all()
            
            for trigger in triggers:
                try:
                    # Check conditions
                    if trigger.conditions and not self._check_event_conditions(trigger.conditions, event_data):
                        continue
                    
                    # Get workflow
                    workflow = db.query(Workflow).filter(Workflow.id == trigger.workflow_id).first()
                    if not workflow or workflow.status != WorkflowStatus.ACTIVE:
                        continue
                    
                    # Prepare trigger data
                    trigger_data = {
                        "trigger_id": trigger.id,
                        "trigger_name": trigger.name,
                        "event_type": event_type,
                        "event_data": event_data,
                        "emitted_at": datetime.utcnow().isoformat()
                    }
                    
                    # Execute workflow
                    result = await self.execution_engine.execute_workflow(
                        workflow_id=workflow.id,
                        input_data=event_data,
                        trigger_type=TriggerType.EVENT,
                        triggered_by="event_system",
                        trigger_data=trigger_data
                    )
                    
                    results.append({
                        "trigger_id": trigger.id,
                        "workflow_id": workflow.id,
                        "execution_id": result["execution_id"],
                        "success": result["success"]
                    })
                    
                    # Update trigger statistics
                    trigger.execution_count += 1
                    trigger.last_executed_at = datetime.utcnow()
                    
                    workflow_metrics.event_executions_total.labels(
                        event_type=event_type,
                        workflow_id=workflow.id
                    ).inc()
                    
                except Exception as e:
                    logger.error(f"Failed to execute event trigger {trigger.id}: {str(e)}")
                    results.append({
                        "trigger_id": trigger.id,
                        "workflow_id": trigger.workflow_id,
                        "success": False,
                        "error": str(e)
                    })
            
            db.commit()
            
        finally:
            db.close()
        
        return results
    
    def _check_event_conditions(self, conditions: Dict[str, Any], event_data: Dict[str, Any]) -> bool:
        """Check if event data matches trigger conditions."""
        
        try:
            for condition_key, condition_value in conditions.items():
                if isinstance(condition_value, dict):
                    # Complex condition (operators like eq, gt, lt, contains, etc.)
                    event_value = event_data.get(condition_key)
                    
                    for operator, expected_value in condition_value.items():
                        if operator == "eq" and event_value != expected_value:
                            return False
                        elif operator == "ne" and event_value == expected_value:
                            return False
                        elif operator == "gt" and (event_value is None or event_value <= expected_value):
                            return False
                        elif operator == "gte" and (event_value is None or event_value < expected_value):
                            return False
                        elif operator == "lt" and (event_value is None or event_value >= expected_value):
                            return False
                        elif operator == "lte" and (event_value is None or event_value > expected_value):
                            return False
                        elif operator == "contains" and (event_value is None or expected_value not in str(event_value)):
                            return False
                        elif operator == "not_contains" and (event_value is not None and expected_value in str(event_value)):
                            return False
                        elif operator == "in" and (event_value is None or event_value not in expected_value):
                            return False
                        elif operator == "not_in" and (event_value is not None and event_value in expected_value):
                            return False
                else:
                    # Simple equality condition
                    if event_data.get(condition_key) != condition_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking event conditions: {str(e)}")
            return False
    
    async def create_condition_trigger(
        self, 
        workflow_id: str, 
        trigger_name: str,
        condition_expression: str,
        check_interval_seconds: int = 60,
        config: Dict[str, Any] = None
    ) -> str:
        """Create a condition-based trigger that periodically checks a condition."""
        
        db = next(get_db())
        try:
            # Check if workflow exists
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create trigger
            trigger = WorkflowTrigger(
                workflow_id=workflow_id,
                name=trigger_name,
                trigger_type=TriggerType.CONDITION,
                conditions={"expression": condition_expression},
                config=config or {},
                is_active=True
            )
            
            # Store check interval in config
            trigger.config["check_interval_seconds"] = check_interval_seconds
            
            db.add(trigger)
            db.commit()
            
            # Start condition monitoring
            await self._start_condition_monitoring(trigger.id, condition_expression, check_interval_seconds)
            
            logger.info(f"Created condition trigger {trigger.id} with expression: {condition_expression}")
            return trigger.id
            
        finally:
            db.close()
    
    async def _start_condition_monitoring(self, trigger_id: str, condition_expression: str, check_interval: int):
        """Start monitoring a condition trigger."""
        
        async def monitor_condition():
            while True:
                try:
                    db = next(get_db())
                    trigger = db.query(WorkflowTrigger).filter(WorkflowTrigger.id == trigger_id).first()
                    
                    if not trigger or not trigger.is_active:
                        break
                    
                    # Evaluate condition
                    if await self._evaluate_condition(condition_expression, trigger.workflow_id):
                        # Get workflow
                        workflow = db.query(Workflow).filter(Workflow.id == trigger.workflow_id).first()
                        if workflow and workflow.status == WorkflowStatus.ACTIVE:
                            
                            # Prepare trigger data
                            trigger_data = {
                                "trigger_id": trigger.id,
                                "trigger_name": trigger.name,
                                "condition_expression": condition_expression,
                                "condition_met_at": datetime.utcnow().isoformat()
                            }
                            
                            # Execute workflow
                            await self.execution_engine.execute_workflow(
                                workflow_id=workflow.id,
                                input_data={"condition_result": True},
                                trigger_type=TriggerType.CONDITION,
                                triggered_by="condition_monitor",
                                trigger_data=trigger_data
                            )
                            
                            # Update trigger statistics
                            trigger.execution_count += 1
                            trigger.last_executed_at = datetime.utcnow()
                            db.commit()
                    
                    db.close()
                    await asyncio.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Condition monitoring error for trigger {trigger_id}: {str(e)}")
                    await asyncio.sleep(check_interval)
        
        # Store monitoring task
        self.condition_triggers[trigger_id] = {
            "task": asyncio.create_task(monitor_condition()),
            "expression": condition_expression,
            "interval": check_interval
        }
    
    async def _evaluate_condition(self, condition_expression: str, workflow_id: str) -> bool:
        """Evaluate a condition expression for a workflow."""
        
        try:
            # This is a simplified condition evaluator
            # In production, use a proper expression evaluator with sandboxing
            
            # Get workflow context data (variables, recent execution results, etc.)
            context_data = await self._get_workflow_context(workflow_id)
            
            # Basic safety check
            if any(forbidden in condition_expression for forbidden in ['import', '__', 'exec', 'eval', 'open', 'file']):
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
                'round': round,
                'datetime': datetime,
                'now': datetime.utcnow()
            }
            safe_namespace.update(context_data)
            
            return bool(eval(condition_expression, safe_namespace))
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {str(e)}")
            return False
    
    async def _get_workflow_context(self, workflow_id: str) -> Dict[str, Any]:
        """Get context data for workflow condition evaluation."""
        
        db = next(get_db())
        try:
            # Get workflow
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                return {}
            
            # Get workflow variables
            context = workflow.variables or {}
            
            # Add workflow statistics
            context.update({
                "execution_count": workflow.execution_count,
                "success_count": workflow.success_count,
                "failure_count": workflow.failure_count,
                "avg_duration_ms": workflow.avg_duration_ms
            })
            
            # Add recent execution results (last 5)
            from .models import WorkflowExecution, ExecutionStatus
            recent_executions = db.query(WorkflowExecution).filter(
                WorkflowExecution.workflow_id == workflow_id
            ).order_by(WorkflowExecution.started_at.desc()).limit(5).all()
            
            context["recent_executions"] = [
                {
                    "status": exec.status.value,
                    "duration_ms": exec.duration_ms,
                    "started_at": exec.started_at.isoformat() if exec.started_at else None,
                    "success": exec.status == ExecutionStatus.COMPLETED
                }
                for exec in recent_executions
            ]
            
            return context
            
        finally:
            db.close()
    
    async def create_file_upload_trigger(
        self, 
        workflow_id: str, 
        trigger_name: str,
        file_extensions: List[str] = None,
        max_file_size_mb: int = 100,
        config: Dict[str, Any] = None
    ) -> str:
        """Create a file upload trigger for a workflow."""
        
        db = next(get_db())
        try:
            # Check if workflow exists
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create trigger
            trigger = WorkflowTrigger(
                workflow_id=workflow_id,
                name=trigger_name,
                trigger_type=TriggerType.FILE_UPLOAD,
                conditions={
                    "file_extensions": file_extensions or [],
                    "max_file_size_mb": max_file_size_mb
                },
                config=config or {},
                is_active=True
            )
            
            db.add(trigger)
            db.commit()
            
            logger.info(f"Created file upload trigger {trigger.id} for workflow {workflow_id}")
            return trigger.id
            
        finally:
            db.close()
    
    async def handle_file_upload(self, file_path: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle file upload and trigger matching workflows."""
        
        results = []
        db = next(get_db())
        
        try:
            # Get all file upload triggers
            triggers = db.query(WorkflowTrigger).filter(
                WorkflowTrigger.trigger_type == TriggerType.FILE_UPLOAD,
                WorkflowTrigger.is_active == True
            ).all()
            
            for trigger in triggers:
                try:
                    # Check file conditions
                    conditions = trigger.conditions or {}
                    
                    # Check file extension
                    file_extensions = conditions.get("file_extensions", [])
                    if file_extensions:
                        file_ext = Path(file_path).suffix.lower()
                        if file_ext not in [ext.lower() for ext in file_extensions]:
                            continue
                    
                    # Check file size
                    max_size_mb = conditions.get("max_file_size_mb", 100)
                    file_size_mb = file_info.get("size_bytes", 0) / (1024 * 1024)
                    if file_size_mb > max_size_mb:
                        continue
                    
                    # Get workflow
                    workflow = db.query(Workflow).filter(Workflow.id == trigger.workflow_id).first()
                    if not workflow or workflow.status != WorkflowStatus.ACTIVE:
                        continue
                    
                    # Prepare trigger data
                    trigger_data = {
                        "trigger_id": trigger.id,
                        "trigger_name": trigger.name,
                        "file_path": file_path,
                        "file_info": file_info,
                        "uploaded_at": datetime.utcnow().isoformat()
                    }
                    
                    # Prepare input data
                    input_data = {
                        "file_path": file_path,
                        "file_name": file_info.get("name", Path(file_path).name),
                        "file_size_bytes": file_info.get("size_bytes", 0),
                        "file_type": file_info.get("type", ""),
                        "upload_timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Execute workflow
                    result = await self.execution_engine.execute_workflow(
                        workflow_id=workflow.id,
                        input_data=input_data,
                        trigger_type=TriggerType.FILE_UPLOAD,
                        triggered_by="file_upload",
                        trigger_data=trigger_data
                    )
                    
                    results.append({
                        "trigger_id": trigger.id,
                        "workflow_id": workflow.id,
                        "execution_id": result["execution_id"],
                        "success": result["success"]
                    })
                    
                    # Update trigger statistics
                    trigger.execution_count += 1
                    trigger.last_executed_at = datetime.utcnow()
                    
                    workflow_metrics.file_upload_executions_total.labels(
                        workflow_id=workflow.id
                    ).inc()
                    
                except Exception as e:
                    logger.error(f"Failed to execute file upload trigger {trigger.id}: {str(e)}")
                    results.append({
                        "trigger_id": trigger.id,
                        "workflow_id": trigger.workflow_id,
                        "success": False,
                        "error": str(e)
                    })
            
            db.commit()
            
        finally:
            db.close()
        
        return results
    
    async def register_custom_trigger_handler(self, trigger_name: str, handler: Callable):
        """Register a custom trigger handler."""
        
        self.custom_handlers[trigger_name] = handler
        logger.info(f"Registered custom trigger handler: {trigger_name}")
    
    async def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a workflow trigger."""
        
        db = next(get_db())
        try:
            trigger = db.query(WorkflowTrigger).filter(WorkflowTrigger.id == trigger_id).first()
            if not trigger:
                return False
            
            # Clean up based on trigger type
            if trigger.trigger_type == TriggerType.EVENT:
                event_type = trigger.config.get("event_type")
                if event_type and event_type in self.event_listeners:
                    self.event_listeners[event_type].discard(trigger_id)
            
            elif trigger.trigger_type == TriggerType.CONDITION:
                if trigger_id in self.condition_triggers:
                    self.condition_triggers[trigger_id]["task"].cancel()
                    del self.condition_triggers[trigger_id]
            
            elif trigger.trigger_type == TriggerType.WEBHOOK:
                if trigger.webhook_url in self.webhook_secrets:
                    del self.webhook_secrets[trigger.webhook_url]
            
            # Remove trigger from database
            db.delete(trigger)
            db.commit()
            
            logger.info(f"Removed trigger {trigger_id}")
            return True
            
        finally:
            db.close()
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get trigger system statistics."""
        
        db = next(get_db())
        try:
            triggers = db.query(WorkflowTrigger).all()
            
            stats = {
                "total_triggers": len(triggers),
                "triggers_by_type": {},
                "active_triggers": 0,
                "webhook_endpoints": len(self.webhook_secrets),
                "event_listeners": len(self.event_listeners),
                "condition_monitors": len(self.condition_triggers),
                "custom_handlers": len(self.custom_handlers)
            }
            
            for trigger in triggers:
                trigger_type = trigger.trigger_type.value
                stats["triggers_by_type"][trigger_type] = stats["triggers_by_type"].get(trigger_type, 0) + 1
                
                if trigger.is_active:
                    stats["active_triggers"] += 1
            
            return stats
            
        finally:
            db.close()