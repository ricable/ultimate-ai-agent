"""
Workflow Scheduler

Advanced scheduling system for workflows with cron expressions, recurring patterns,
and event-based scheduling.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from croniter import croniter
from sqlalchemy.orm import Session
from ..database.session import get_db
from ..monitoring.metrics.prometheus_metrics import workflow_metrics
from .models import Workflow, WorkflowTrigger, TriggerType, WorkflowStatus
from .execution_engine import WorkflowExecutionEngine
import uuid
import pytz

logger = logging.getLogger(__name__)


class WorkflowScheduler:
    """Advanced workflow scheduler with comprehensive scheduling capabilities."""
    
    def __init__(self, execution_engine: WorkflowExecutionEngine):
        self.execution_engine = execution_engine
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.scheduled_triggers: Dict[str, Dict[str, Any]] = {}
        self.timezone = pytz.UTC
        
    async def start(self):
        """Start the workflow scheduler."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Workflow scheduler started")
    
    async def stop(self):
        """Stop the workflow scheduler."""
        if not self.running:
            return
        
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Workflow scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                await self._process_scheduled_triggers()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _process_scheduled_triggers(self):
        """Process all scheduled triggers that are due for execution."""
        
        db = next(get_db())
        try:
            # Get all active scheduled triggers
            triggers = db.query(WorkflowTrigger).filter(
                WorkflowTrigger.trigger_type == TriggerType.SCHEDULE,
                WorkflowTrigger.is_active == True
            ).all()
            
            current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
            
            for trigger in triggers:
                try:
                    if await self._should_execute_trigger(trigger, current_time):
                        await self._execute_scheduled_workflow(trigger, db)
                except Exception as e:
                    logger.error(f"Error processing trigger {trigger.id}: {str(e)}")
        
        finally:
            db.close()
    
    async def _should_execute_trigger(self, trigger: WorkflowTrigger, current_time: datetime) -> bool:
        """Determine if a trigger should be executed now."""
        
        # Check if next execution time is due
        if trigger.next_execution_at and trigger.next_execution_at.replace(tzinfo=pytz.UTC) > current_time:
            return False
        
        # Check schedule
        if not trigger.schedule:
            return False
        
        try:
            # Parse cron expression
            cron = croniter(trigger.schedule, current_time)
            next_execution = cron.get_next(datetime)
            
            # Update next execution time
            db = next(get_db())
            trigger.next_execution_at = next_execution
            db.commit()
            db.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Invalid cron expression for trigger {trigger.id}: {trigger.schedule}")
            return False
    
    async def _execute_scheduled_workflow(self, trigger: WorkflowTrigger, db: Session):
        """Execute a workflow from a scheduled trigger."""
        
        try:
            # Get workflow
            workflow = db.query(Workflow).filter(Workflow.id == trigger.workflow_id).first()
            if not workflow or workflow.status != WorkflowStatus.ACTIVE:
                logger.warning(f"Workflow {trigger.workflow_id} not found or not active")
                return
            
            # Prepare trigger data
            trigger_data = {
                "trigger_id": trigger.id,
                "trigger_name": trigger.name,
                "schedule": trigger.schedule,
                "scheduled_time": datetime.utcnow().isoformat()
            }
            
            # Add trigger configuration
            if trigger.config:
                trigger_data.update(trigger.config)
            
            # Execute workflow
            result = await self.execution_engine.execute_workflow(
                workflow_id=workflow.id,
                input_data=trigger_data.get("input_data", {}),
                trigger_type=TriggerType.SCHEDULE,
                triggered_by="scheduler",
                trigger_data=trigger_data
            )
            
            # Update trigger statistics
            trigger.execution_count += 1
            trigger.last_executed_at = datetime.utcnow()
            db.commit()
            
            # Update metrics
            workflow_metrics.scheduled_executions_total.labels(
                workflow_id=workflow.id,
                trigger_id=trigger.id
            ).inc()
            
            logger.info(f"Scheduled workflow {workflow.id} executed successfully: {result['execution_id']}")
            
        except Exception as e:
            logger.error(f"Failed to execute scheduled workflow {trigger.workflow_id}: {str(e)}")
            workflow_metrics.scheduled_executions_failed.labels(
                workflow_id=trigger.workflow_id,
                trigger_id=trigger.id
            ).inc()
    
    async def schedule_workflow(
        self, 
        workflow_id: str, 
        schedule: str, 
        trigger_name: str,
        config: Dict[str, Any] = None,
        timezone: str = "UTC"
    ) -> str:
        """Schedule a workflow with a cron expression."""
        
        # Validate cron expression
        try:
            croniter(schedule)
        except Exception as e:
            raise ValueError(f"Invalid cron expression: {schedule}")
        
        # Validate timezone
        try:
            tz = pytz.timezone(timezone)
        except:
            raise ValueError(f"Invalid timezone: {timezone}")
        
        db = next(get_db())
        try:
            # Check if workflow exists
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Calculate next execution time
            current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
            if timezone != "UTC":
                # Convert to target timezone for cron calculation
                tz_time = current_time.astimezone(tz)
                cron = croniter(schedule, tz_time)
                next_execution = cron.get_next(datetime).astimezone(pytz.UTC)
            else:
                cron = croniter(schedule, current_time)
                next_execution = cron.get_next(datetime)
            
            # Create trigger
            trigger = WorkflowTrigger(
                workflow_id=workflow_id,
                name=trigger_name,
                trigger_type=TriggerType.SCHEDULE,
                schedule=schedule,
                config=config or {},
                next_execution_at=next_execution,
                is_active=True
            )
            
            db.add(trigger)
            db.commit()
            
            logger.info(f"Scheduled workflow {workflow_id} with trigger {trigger.id}")
            return trigger.id
            
        finally:
            db.close()
    
    async def unschedule_workflow(self, trigger_id: str) -> bool:
        """Remove a scheduled workflow trigger."""
        
        db = next(get_db())
        try:
            trigger = db.query(WorkflowTrigger).filter(WorkflowTrigger.id == trigger_id).first()
            if not trigger:
                return False
            
            # Deactivate trigger
            trigger.is_active = False
            db.commit()
            
            # Remove from scheduled triggers cache if present
            if trigger_id in self.scheduled_triggers:
                del self.scheduled_triggers[trigger_id]
            
            logger.info(f"Unscheduled workflow trigger {trigger_id}")
            return True
            
        finally:
            db.close()
    
    async def update_schedule(self, trigger_id: str, schedule: str, config: Dict[str, Any] = None) -> bool:
        """Update an existing scheduled trigger."""
        
        # Validate cron expression
        try:
            croniter(schedule)
        except Exception as e:
            raise ValueError(f"Invalid cron expression: {schedule}")
        
        db = next(get_db())
        try:
            trigger = db.query(WorkflowTrigger).filter(WorkflowTrigger.id == trigger_id).first()
            if not trigger:
                return False
            
            # Update trigger
            trigger.schedule = schedule
            if config:
                trigger.config = config
            
            # Calculate next execution time
            current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
            cron = croniter(schedule, current_time)
            trigger.next_execution_at = cron.get_next(datetime)
            
            db.commit()
            
            logger.info(f"Updated schedule for trigger {trigger_id}")
            return True
            
        finally:
            db.close()
    
    async def get_scheduled_workflows(self, workflow_id: str = None) -> List[Dict[str, Any]]:
        """Get all scheduled workflows or specific workflow schedules."""
        
        db = next(get_db())
        try:
            query = db.query(WorkflowTrigger).filter(
                WorkflowTrigger.trigger_type == TriggerType.SCHEDULE
            )
            
            if workflow_id:
                query = query.filter(WorkflowTrigger.workflow_id == workflow_id)
            
            triggers = query.all()
            
            return [trigger.to_dict() for trigger in triggers]
            
        finally:
            db.close()
    
    async def get_next_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the next scheduled workflow executions."""
        
        db = next(get_db())
        try:
            triggers = db.query(WorkflowTrigger).filter(
                WorkflowTrigger.trigger_type == TriggerType.SCHEDULE,
                WorkflowTrigger.is_active == True,
                WorkflowTrigger.next_execution_at != None
            ).order_by(WorkflowTrigger.next_execution_at).limit(limit).all()
            
            result = []
            for trigger in triggers:
                workflow = db.query(Workflow).filter(Workflow.id == trigger.workflow_id).first()
                result.append({
                    "trigger_id": trigger.id,
                    "workflow_id": trigger.workflow_id,
                    "workflow_name": workflow.name if workflow else "Unknown",
                    "trigger_name": trigger.name,
                    "schedule": trigger.schedule,
                    "next_execution_at": trigger.next_execution_at.isoformat() if trigger.next_execution_at else None,
                    "is_active": trigger.is_active
                })
            
            return result
            
        finally:
            db.close()
    
    def parse_human_schedule(self, human_schedule: str) -> str:
        """Parse human-readable schedule to cron expression."""
        
        human_schedule = human_schedule.lower().strip()
        
        # Common patterns
        patterns = {
            "every minute": "* * * * *",
            "every 5 minutes": "*/5 * * * *",
            "every 10 minutes": "*/10 * * * *",
            "every 15 minutes": "*/15 * * * *",
            "every 30 minutes": "*/30 * * * *",
            "every hour": "0 * * * *",
            "every 2 hours": "0 */2 * * *",
            "every 4 hours": "0 */4 * * *",
            "every 6 hours": "0 */6 * * *",
            "every 12 hours": "0 */12 * * *",
            "daily": "0 0 * * *",
            "every day": "0 0 * * *",
            "weekly": "0 0 * * 0",
            "every week": "0 0 * * 0",
            "monthly": "0 0 1 * *",
            "every month": "0 0 1 * *",
            "yearly": "0 0 1 1 *",
            "every year": "0 0 1 1 *",
            "weekdays": "0 9 * * 1-5",
            "weekends": "0 9 * * 0,6",
            "business hours": "0 9-17 * * 1-5"
        }
        
        if human_schedule in patterns:
            return patterns[human_schedule]
        
        # Time-based patterns
        import re
        
        # Pattern: "at 9:30 AM"
        time_match = re.match(r"at (\d{1,2}):(\d{2})\s*(am|pm)?", human_schedule)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            ampm = time_match.group(3)
            
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            
            return f"{minute} {hour} * * *"
        
        # Pattern: "every X minutes/hours/days"
        interval_match = re.match(r"every (\d+) (minutes?|hours?|days?)", human_schedule)
        if interval_match:
            interval = int(interval_match.group(1))
            unit = interval_match.group(2)
            
            if "minute" in unit:
                return f"*/{interval} * * * *"
            elif "hour" in unit:
                return f"0 */{interval} * * *"
            elif "day" in unit:
                return f"0 0 */{interval} * *"
        
        # Pattern: "on weekdays at 9 AM"
        weekday_time_match = re.match(r"on (weekdays|weekends) at (\d{1,2})\s*(am|pm)?", human_schedule)
        if weekday_time_match:
            day_type = weekday_time_match.group(1)
            hour = int(weekday_time_match.group(2))
            ampm = weekday_time_match.group(3)
            
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            
            days = "1-5" if day_type == "weekdays" else "0,6"
            return f"0 {hour} * * {days}"
        
        raise ValueError(f"Could not parse human schedule: {human_schedule}")
    
    async def create_recurring_schedule(
        self, 
        workflow_id: str,
        trigger_name: str,
        pattern: str,
        start_date: datetime = None,
        end_date: datetime = None,
        max_executions: int = None,
        config: Dict[str, Any] = None
    ) -> str:
        """Create a recurring schedule with advanced options."""
        
        # Parse pattern to cron expression
        if pattern.startswith("cron:"):
            cron_expr = pattern[5:]
        else:
            cron_expr = self.parse_human_schedule(pattern)
        
        # Validate cron expression
        try:
            croniter(cron_expr)
        except Exception as e:
            raise ValueError(f"Invalid schedule pattern: {pattern}")
        
        # Enhanced configuration
        enhanced_config = config or {}
        enhanced_config.update({
            "recurring": True,
            "pattern": pattern,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "max_executions": max_executions,
            "execution_count": 0
        })
        
        return await self.schedule_workflow(
            workflow_id=workflow_id,
            schedule=cron_expr,
            trigger_name=trigger_name,
            config=enhanced_config
        )
    
    async def pause_schedule(self, trigger_id: str) -> bool:
        """Pause a scheduled workflow."""
        
        db = next(get_db())
        try:
            trigger = db.query(WorkflowTrigger).filter(WorkflowTrigger.id == trigger_id).first()
            if not trigger:
                return False
            
            trigger.is_active = False
            db.commit()
            
            logger.info(f"Paused schedule {trigger_id}")
            return True
            
        finally:
            db.close()
    
    async def resume_schedule(self, trigger_id: str) -> bool:
        """Resume a paused scheduled workflow."""
        
        db = next(get_db())
        try:
            trigger = db.query(WorkflowTrigger).filter(WorkflowTrigger.id == trigger_id).first()
            if not trigger:
                return False
            
            trigger.is_active = True
            
            # Recalculate next execution time
            if trigger.schedule:
                current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
                cron = croniter(trigger.schedule, current_time)
                trigger.next_execution_at = cron.get_next(datetime)
            
            db.commit()
            
            logger.info(f"Resumed schedule {trigger_id}")
            return True
            
        finally:
            db.close()
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        
        db = next(get_db())
        try:
            total_triggers = db.query(WorkflowTrigger).filter(
                WorkflowTrigger.trigger_type == TriggerType.SCHEDULE
            ).count()
            
            active_triggers = db.query(WorkflowTrigger).filter(
                WorkflowTrigger.trigger_type == TriggerType.SCHEDULE,
                WorkflowTrigger.is_active == True
            ).count()
            
            return {
                "running": self.running,
                "total_scheduled_triggers": total_triggers,
                "active_scheduled_triggers": active_triggers,
                "timezone": str(self.timezone),
                "next_check_in_seconds": 30 if self.running else None
            }
            
        finally:
            db.close()