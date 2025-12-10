"""
Workflow Models

Data models for workflow automation system including workflows, steps, executions,
triggers, and templates.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
import uuid

Base = declarative_base()


class WorkflowStatus(Enum):
    """Workflow execution status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class ExecutionStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TriggerType(Enum):
    """Workflow trigger types."""
    MANUAL = "manual"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EVENT = "event"
    CONDITION = "condition"
    FILE_UPLOAD = "file_upload"
    API_CALL = "api_call"


class StepType(Enum):
    """Workflow step types."""
    AGENT = "agent"
    CONDITION = "condition"
    PARALLEL = "parallel"
    TRANSFORM = "transform"
    DELAY = "delay"
    WEBHOOK = "webhook"
    API_CALL = "api_call"
    EMAIL = "email"
    NOTIFICATION = "notification"


class Workflow(Base):
    """Workflow definition model."""
    __tablename__ = "workflows"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    status = Column(SQLEnum(WorkflowStatus), default=WorkflowStatus.DRAFT)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(36), ForeignKey("users.id"))
    organization_id = Column(String(36), ForeignKey("organizations.id"))
    
    # Workflow definition
    definition = Column(JSON)  # Full workflow definition
    variables = Column(JSON)  # Default variables/parameters
    settings = Column(JSON)   # Workflow-specific settings
    
    # Access control
    is_public = Column(Boolean, default=False)
    is_template = Column(Boolean, default=False)
    tags = Column(JSON)  # List of tags for categorization
    
    # Statistics
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    avg_duration_ms = Column(Integer, default=0)
    
    # Relationships
    steps = relationship("WorkflowStep", back_populates="workflow", cascade="all, delete-orphan")
    executions = relationship("WorkflowExecution", back_populates="workflow")
    triggers = relationship("WorkflowTrigger", back_populates="workflow", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "organization_id": self.organization_id,
            "definition": self.definition,
            "variables": self.variables,
            "settings": self.settings,
            "is_public": self.is_public,
            "is_template": self.is_template,
            "tags": self.tags,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_duration_ms": self.avg_duration_ms,
            "steps": [step.to_dict() for step in self.steps] if self.steps else [],
            "triggers": [trigger.to_dict() for trigger in self.triggers] if self.triggers else []
        }


class WorkflowStep(Base):
    """Workflow step model."""
    __tablename__ = "workflow_steps"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(36), ForeignKey("workflows.id"), nullable=False)
    
    # Step identification
    step_id = Column(String(255), nullable=False)  # Unique within workflow
    name = Column(String(255), nullable=False)
    description = Column(Text)
    step_type = Column(SQLEnum(StepType), nullable=False)
    
    # Step configuration
    config = Column(JSON)  # Step-specific configuration
    position = Column(JSON)  # UI position for visual designer
    
    # Flow control
    order_index = Column(Integer, default=0)
    parent_step_id = Column(String(36), ForeignKey("workflow_steps.id"))
    condition = Column(Text)  # Condition for conditional steps
    
    # Relationships
    workflow = relationship("Workflow", back_populates="steps")
    parent_step = relationship("WorkflowStep", remote_side=[id])
    child_steps = relationship("WorkflowStep")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "step_type": self.step_type.value if self.step_type else None,
            "config": self.config,
            "position": self.position,
            "order_index": self.order_index,
            "parent_step_id": self.parent_step_id,
            "condition": self.condition
        }


class WorkflowExecution(Base):
    """Workflow execution model."""
    __tablename__ = "workflow_executions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(36), ForeignKey("workflows.id"), nullable=False)
    
    # Execution metadata
    execution_id = Column(String(255), nullable=False)  # Human-readable ID
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    
    # Execution data
    input_data = Column(JSON)   # Input parameters
    output_data = Column(JSON)  # Final output
    context = Column(JSON)      # Execution context
    
    # Trigger information
    triggered_by = Column(String(36))  # User ID or system
    trigger_type = Column(SQLEnum(TriggerType))
    trigger_data = Column(JSON)
    
    # Error handling
    error_message = Column(Text)
    error_details = Column(JSON)
    retry_count = Column(Integer, default=0)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    step_executions = relationship("StepExecution", back_populates="workflow_execution")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "status": self.status.value if self.status else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "context": self.context,
            "triggered_by": self.triggered_by,
            "trigger_type": self.trigger_type.value if self.trigger_type else None,
            "trigger_data": self.trigger_data,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "retry_count": self.retry_count
        }


class StepExecution(Base):
    """Step execution model."""
    __tablename__ = "step_executions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_execution_id = Column(String(36), ForeignKey("workflow_executions.id"), nullable=False)
    step_id = Column(String(255), nullable=False)
    
    # Execution details
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    
    # Data
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    error_details = Column(JSON)
    
    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="step_executions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step execution to dictionary."""
        return {
            "id": self.id,
            "workflow_execution_id": self.workflow_execution_id,
            "step_id": self.step_id,
            "status": self.status.value if self.status else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "error_details": self.error_details
        }


class WorkflowTrigger(Base):
    """Workflow trigger model."""
    __tablename__ = "workflow_triggers"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(36), ForeignKey("workflows.id"), nullable=False)
    
    # Trigger details
    name = Column(String(255), nullable=False)
    trigger_type = Column(SQLEnum(TriggerType), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Configuration
    config = Column(JSON)  # Trigger-specific configuration
    schedule = Column(String(255))  # Cron expression for scheduled triggers
    conditions = Column(JSON)  # Conditions for condition-based triggers
    
    # Webhook specific
    webhook_url = Column(String(255))
    webhook_secret = Column(String(255))
    
    # Statistics
    execution_count = Column(Integer, default=0)
    last_executed_at = Column(DateTime)
    next_execution_at = Column(DateTime)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="triggers")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trigger to dictionary."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "name": self.name,
            "trigger_type": self.trigger_type.value if self.trigger_type else None,
            "is_active": self.is_active,
            "config": self.config,
            "schedule": self.schedule,
            "conditions": self.conditions,
            "webhook_url": self.webhook_url,
            "webhook_secret": self.webhook_secret,
            "execution_count": self.execution_count,
            "last_executed_at": self.last_executed_at.isoformat() if self.last_executed_at else None,
            "next_execution_at": self.next_execution_at.isoformat() if self.next_execution_at else None
        }


class WorkflowTemplate(Base):
    """Workflow template model for marketplace."""
    __tablename__ = "workflow_templates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Template metadata
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    subcategory = Column(String(100))
    version = Column(String(50), default="1.0.0")
    
    # Template content
    definition = Column(JSON)  # Workflow definition
    variables = Column(JSON)   # Template variables
    documentation = Column(Text)  # Usage documentation
    
    # Authorship
    created_by = Column(String(36), ForeignKey("users.id"))
    organization_id = Column(String(36), ForeignKey("organizations.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Marketplace
    is_featured = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    price = Column(Integer, default=0)  # Price in cents
    
    # Statistics
    download_count = Column(Integer, default=0)
    rating_average = Column(Integer, default=0)  # 1-5 stars * 100
    rating_count = Column(Integer, default=0)
    
    # Tags and search
    tags = Column(JSON)
    keywords = Column(JSON)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "subcategory": self.subcategory,
            "version": self.version,
            "definition": self.definition,
            "variables": self.variables,
            "documentation": self.documentation,
            "created_by": self.created_by,
            "organization_id": self.organization_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_featured": self.is_featured,
            "is_verified": self.is_verified,
            "price": self.price,
            "download_count": self.download_count,
            "rating_average": self.rating_average / 100.0 if self.rating_average else 0,
            "rating_count": self.rating_count,
            "tags": self.tags,
            "keywords": self.keywords
        }