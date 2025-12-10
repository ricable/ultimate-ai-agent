"""
Workflow API Routes

FastAPI routes for workflow automation system including CRUD operations,
execution, scheduling, triggers, and marketplace functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json
import logging

from ..database.session import get_db
from ..services.auth import get_current_user
from ..models.user import User
from .execution_engine import WorkflowExecutionEngine
from .scheduler import WorkflowScheduler
from .triggers import TriggerManager
from .marketplace import WorkflowMarketplace
from .models import (
    Workflow, WorkflowStatus, TriggerType, StepType,
    WorkflowExecution, ExecutionStatus
)

logger = logging.getLogger(__name__)

# Initialize components (these would be injected in a real implementation)
execution_engine = None
scheduler = None
trigger_manager = None
marketplace = None

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


# Pydantic models for request/response
class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    definition: Dict[str, Any] = Field(default_factory=dict)
    variables: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class WorkflowUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None


class WorkflowExecuteRequest(BaseModel):
    input_data: Optional[Dict[str, Any]] = None
    trigger_data: Optional[Dict[str, Any]] = None


class ScheduleCreateRequest(BaseModel):
    schedule: str = Field(..., description="Cron expression")
    trigger_name: str = Field(..., min_length=1, max_length=255)
    config: Optional[Dict[str, Any]] = None
    timezone: str = Field(default="UTC")


class TriggerCreateRequest(BaseModel):
    trigger_name: str = Field(..., min_length=1, max_length=255)
    trigger_type: str
    config: Optional[Dict[str, Any]] = None


class WebhookTriggerRequest(TriggerCreateRequest):
    webhook_path: Optional[str] = None
    secret: Optional[str] = None


class EventTriggerRequest(TriggerCreateRequest):
    event_type: str
    conditions: Optional[Dict[str, Any]] = None


class ConditionTriggerRequest(TriggerCreateRequest):
    condition_expression: str
    check_interval_seconds: int = Field(default=60, ge=10, le=3600)


class FileUploadTriggerRequest(TriggerCreateRequest):
    file_extensions: Optional[List[str]] = None
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)


class TemplatePublishRequest(BaseModel):
    template_name: str = Field(..., min_length=1, max_length=255)
    description: str
    category: str
    subcategory: Optional[str] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    documentation: Optional[str] = ""
    price: int = Field(default=0, ge=0)


class TemplateInstallRequest(BaseModel):
    workflow_name: str = Field(..., min_length=1, max_length=255)
    customizations: Optional[Dict[str, Any]] = None


class TemplateRatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    review: Optional[str] = None


# Dependency injection helpers
def get_execution_engine() -> WorkflowExecutionEngine:
    global execution_engine
    if not execution_engine:
        raise HTTPException(status_code=500, detail="Execution engine not initialized")
    return execution_engine


def get_scheduler() -> WorkflowScheduler:
    global scheduler
    if not scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")
    return scheduler


def get_trigger_manager() -> TriggerManager:
    global trigger_manager
    if not trigger_manager:
        raise HTTPException(status_code=500, detail="Trigger manager not initialized")
    return trigger_manager


def get_marketplace() -> WorkflowMarketplace:
    global marketplace
    if not marketplace:
        raise HTTPException(status_code=500, detail="Marketplace not initialized")
    return marketplace


# Workflow CRUD endpoints
@router.post("/", response_model=Dict[str, Any])
async def create_workflow(
    request: WorkflowCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new workflow."""
    try:
        workflow = Workflow(
            name=request.name,
            description=request.description,
            definition=request.definition,
            variables=request.variables,
            tags=request.tags,
            created_by=current_user.id,
            organization_id=getattr(current_user, 'organization_id', None)
        )
        
        db.add(workflow)
        db.commit()
        
        return {
            "success": True,
            "workflow_id": workflow.id,
            "message": f"Workflow '{workflow.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=Dict[str, Any])
async def list_workflows(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List workflows for the current user/organization."""
    try:
        query = db.query(Workflow)
        
        # Filter by user/organization
        if hasattr(current_user, 'organization_id') and current_user.organization_id:
            query = query.filter(Workflow.organization_id == current_user.organization_id)
        else:
            query = query.filter(Workflow.created_by == current_user.id)
        
        # Apply filters
        if status:
            try:
                status_enum = WorkflowStatus(status)
                query = query.filter(Workflow.status == status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        if search:
            query = query.filter(
                Workflow.name.ilike(f"%{search}%") |
                Workflow.description.ilike(f"%{search}%")
            )
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        workflows = query.offset(skip).limit(limit).all()
        
        return {
            "workflows": [workflow.to_dict() for workflow in workflows],
            "total_count": total_count,
            "page_count": (total_count + limit - 1) // limit,
            "current_page": skip // limit + 1
        }
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}", response_model=Dict[str, Any])
async def get_workflow(
    workflow_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific workflow."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if (workflow.created_by != current_user.id and 
            getattr(workflow, 'organization_id', None) != getattr(current_user, 'organization_id', None)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return workflow.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{workflow_id}", response_model=Dict[str, Any])
async def update_workflow(
    workflow_id: str,
    request: WorkflowUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a workflow."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if workflow.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Apply updates
        update_data = request.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == "status":
                try:
                    value = WorkflowStatus(value)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid status: {value}")
            
            if hasattr(workflow, field):
                setattr(workflow, field, value)
        
        workflow.updated_at = datetime.utcnow()
        db.commit()
        
        return {
            "success": True,
            "message": f"Workflow '{workflow.name}' updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{workflow_id}", response_model=Dict[str, Any])
async def delete_workflow(
    workflow_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a workflow."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if workflow.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        db.delete(workflow)
        db.commit()
        
        return {
            "success": True,
            "message": f"Workflow '{workflow.name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Workflow execution endpoints
@router.post("/{workflow_id}/execute", response_model=Dict[str, Any])
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    engine: WorkflowExecutionEngine = Depends(get_execution_engine)
):
    """Execute a workflow."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if (workflow.created_by != current_user.id and 
            getattr(workflow, 'organization_id', None) != getattr(current_user, 'organization_id', None)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if workflow.status != WorkflowStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="Workflow is not active")
        
        # Execute workflow
        result = await engine.execute_workflow(
            workflow_id=workflow_id,
            input_data=request.input_data,
            triggered_by=current_user.id,
            trigger_data=request.trigger_data
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/executions", response_model=Dict[str, Any])
async def get_workflow_executions(
    workflow_id: str,
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get workflow execution history."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if (workflow.created_by != current_user.id and 
            getattr(workflow, 'organization_id', None) != getattr(current_user, 'organization_id', None)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        query = db.query(WorkflowExecution).filter(WorkflowExecution.workflow_id == workflow_id)
        
        if status:
            try:
                status_enum = ExecutionStatus(status)
                query = query.filter(WorkflowExecution.status == status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        total_count = query.count()
        executions = query.order_by(WorkflowExecution.started_at.desc()).offset(skip).limit(limit).all()
        
        return {
            "executions": [execution.to_dict() for execution in executions],
            "total_count": total_count,
            "page_count": (total_count + limit - 1) // limit,
            "current_page": skip // limit + 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get executions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{execution_id}", response_model=Dict[str, Any])
async def get_execution_details(
    execution_id: str,
    current_user: User = Depends(get_current_user),
    engine: WorkflowExecutionEngine = Depends(get_execution_engine)
):
    """Get detailed execution information."""
    try:
        execution = await engine.get_execution_status(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return execution
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/executions/{execution_id}/cancel", response_model=Dict[str, Any])
async def cancel_execution(
    execution_id: str,
    current_user: User = Depends(get_current_user),
    engine: WorkflowExecutionEngine = Depends(get_execution_engine)
):
    """Cancel a running workflow execution."""
    try:
        success = await engine.cancel_execution(execution_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")
        
        return {
            "success": True,
            "message": f"Execution {execution_id} cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Scheduling endpoints
@router.post("/{workflow_id}/schedule", response_model=Dict[str, Any])
async def schedule_workflow(
    workflow_id: str,
    request: ScheduleCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    scheduler: WorkflowScheduler = Depends(get_scheduler)
):
    """Schedule a workflow to run on a schedule."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if workflow.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        trigger_id = await scheduler.schedule_workflow(
            workflow_id=workflow_id,
            schedule=request.schedule,
            trigger_name=request.trigger_name,
            config=request.config,
            timezone=request.timezone
        )
        
        return {
            "success": True,
            "trigger_id": trigger_id,
            "message": f"Workflow scheduled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule workflow: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{workflow_id}/schedules", response_model=List[Dict[str, Any]])
async def get_workflow_schedules(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    scheduler: WorkflowScheduler = Depends(get_scheduler)
):
    """Get all schedules for a workflow."""
    try:
        schedules = await scheduler.get_scheduled_workflows(workflow_id)
        return schedules
        
    except Exception as e:
        logger.error(f"Failed to get schedules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedules/{trigger_id}", response_model=Dict[str, Any])
async def remove_schedule(
    trigger_id: str,
    current_user: User = Depends(get_current_user),
    scheduler: WorkflowScheduler = Depends(get_scheduler)
):
    """Remove a scheduled workflow."""
    try:
        success = await scheduler.unschedule_workflow(trigger_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {
            "success": True,
            "message": "Schedule removed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Trigger endpoints
@router.post("/{workflow_id}/triggers/webhook", response_model=Dict[str, Any])
async def create_webhook_trigger(
    workflow_id: str,
    request: WebhookTriggerRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    trigger_manager: TriggerManager = Depends(get_trigger_manager)
):
    """Create a webhook trigger for a workflow."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if workflow.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        result = await trigger_manager.create_webhook_trigger(
            workflow_id=workflow_id,
            trigger_name=request.trigger_name,
            webhook_path=request.webhook_path,
            secret=request.secret,
            config=request.config
        )
        
        return {
            "success": True,
            **result,
            "message": "Webhook trigger created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create webhook trigger: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{workflow_id}/triggers/event", response_model=Dict[str, Any])
async def create_event_trigger(
    workflow_id: str,
    request: EventTriggerRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    trigger_manager: TriggerManager = Depends(get_trigger_manager)
):
    """Create an event trigger for a workflow."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if workflow.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        trigger_id = await trigger_manager.create_event_trigger(
            workflow_id=workflow_id,
            trigger_name=request.trigger_name,
            event_type=request.event_type,
            conditions=request.conditions,
            config=request.config
        )
        
        return {
            "success": True,
            "trigger_id": trigger_id,
            "message": "Event trigger created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create event trigger: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Webhook handling endpoint
@router.post("/webhooks/{webhook_path:path}", response_model=Dict[str, Any])
async def handle_webhook(
    webhook_path: str,
    payload: Dict[str, Any],
    request,
    trigger_manager: TriggerManager = Depends(get_trigger_manager)
):
    """Handle incoming webhook requests."""
    try:
        headers = dict(request.headers)
        result = await trigger_manager.handle_webhook(
            webhook_path=f"/{webhook_path}",
            payload=payload,
            headers=headers
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to handle webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Event emission endpoint
@router.post("/events/{event_type}", response_model=Dict[str, Any])
async def emit_event(
    event_type: str,
    event_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    trigger_manager: TriggerManager = Depends(get_trigger_manager)
):
    """Emit an event to trigger workflows."""
    try:
        results = await trigger_manager.emit_event(event_type, event_data)
        
        return {
            "success": True,
            "triggered_workflows": len(results),
            "results": results,
            "message": f"Event '{event_type}' emitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to emit event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Marketplace endpoints
@router.get("/marketplace/templates", response_model=Dict[str, Any])
async def search_templates(
    query: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    tags: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[int] = None,
    is_free_only: bool = False,
    is_featured: bool = False,
    is_verified: bool = False,
    sort_by: str = "relevance",
    limit: int = 20,
    offset: int = 0,
    marketplace: WorkflowMarketplace = Depends(get_marketplace)
):
    """Search workflow templates in the marketplace."""
    try:
        tag_list = tags.split(",") if tags else None
        
        result = await marketplace.search_templates(
            query=query,
            category=category,
            subcategory=subcategory,
            tags=tag_list,
            min_rating=min_rating,
            max_price=max_price,
            is_free_only=is_free_only,
            is_featured=is_featured,
            is_verified=is_verified,
            sort_by=sort_by,
            limit=limit,
            offset=offset
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to search templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marketplace/templates/{template_id}", response_model=Dict[str, Any])
async def get_template(
    template_id: str,
    marketplace: WorkflowMarketplace = Depends(get_marketplace)
):
    """Get detailed template information."""
    try:
        template = await marketplace.get_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/publish", response_model=Dict[str, Any])
async def publish_template(
    workflow_id: str,
    request: TemplatePublishRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    marketplace: WorkflowMarketplace = Depends(get_marketplace)
):
    """Publish a workflow as a template to the marketplace."""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check permissions
        if workflow.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        template_id = await marketplace.publish_template(
            workflow_id=workflow_id,
            template_name=request.template_name,
            description=request.description,
            category=request.category,
            subcategory=request.subcategory,
            tags=request.tags,
            keywords=request.keywords,
            documentation=request.documentation,
            price=request.price,
            created_by=current_user.id,
            organization_id=getattr(current_user, 'organization_id', None)
        )
        
        return {
            "success": True,
            "template_id": template_id,
            "message": "Template published successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish template: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/marketplace/templates/{template_id}/install", response_model=Dict[str, Any])
async def install_template(
    template_id: str,
    request: TemplateInstallRequest,
    current_user: User = Depends(get_current_user),
    marketplace: WorkflowMarketplace = Depends(get_marketplace)
):
    """Install a template as a new workflow."""
    try:
        workflow_id = await marketplace.install_template(
            template_id=template_id,
            workflow_name=request.workflow_name,
            user_id=current_user.id,
            organization_id=getattr(current_user, 'organization_id', None),
            customizations=request.customizations
        )
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": f"Template installed as workflow '{request.workflow_name}'"
        }
        
    except Exception as e:
        logger.error(f"Failed to install template: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/marketplace/categories", response_model=Dict[str, List[str]])
async def get_categories(
    marketplace: WorkflowMarketplace = Depends(get_marketplace)
):
    """Get all marketplace categories."""
    try:
        return await marketplace.get_categories()
        
    except Exception as e:
        logger.error(f"Failed to get categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marketplace/stats", response_model=Dict[str, Any])
async def get_marketplace_stats(
    marketplace: WorkflowMarketplace = Depends(get_marketplace)
):
    """Get marketplace statistics."""
    try:
        return marketplace.get_marketplace_stats()
        
    except Exception as e:
        logger.error(f"Failed to get marketplace stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# System status endpoints
@router.get("/system/status", response_model=Dict[str, Any])
async def get_system_status(
    engine: WorkflowExecutionEngine = Depends(get_execution_engine),
    scheduler: WorkflowScheduler = Depends(get_scheduler),
    trigger_manager: TriggerManager = Depends(get_trigger_manager)
):
    """Get workflow system status."""
    try:
        return {
            "execution_engine": {
                "active_executions": len(engine.get_active_executions()),
                "active_executions_details": engine.get_active_executions()
            },
            "scheduler": scheduler.get_scheduler_stats(),
            "triggers": trigger_manager.get_trigger_stats()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def initialize_workflow_system(
    exec_engine: WorkflowExecutionEngine,
    work_scheduler: WorkflowScheduler, 
    trig_manager: TriggerManager,
    work_marketplace: WorkflowMarketplace
):
    """Initialize the workflow system components."""
    global execution_engine, scheduler, trigger_manager, marketplace
    execution_engine = exec_engine
    scheduler = work_scheduler
    trigger_manager = trig_manager
    marketplace = work_marketplace