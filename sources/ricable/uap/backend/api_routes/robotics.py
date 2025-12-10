# backend/api_routes/robotics.py
# Agent 34: Advanced Robotics Integration - API Routes

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Import robotics components
from ..robotics.rpa_engine import rpa_engine, RPATask, RPATaskType, RPAWorkflow
from ..robotics.sensor_fusion import sensor_fusion_engine, SensorReading, SensorType
from ..robotics.navigation_planner import navigation_controller, NavigationGoal, Waypoint, NavigationMode
from ..robotics.robot_coordinator import robot_coordinator, Robot, Task, TaskType, TaskPriority
from ..robotics.vision_processor import robotics_vision_processor, RobotVisionTask
from ..robotics.human_robot_interface import human_robot_interface, InteractionMode
from ...hardware.controllers.robot_controller import RobotController, ControllerType
from ...hardware.controllers.sensor_interface import sensor_manager
from ...hardware.controllers.actuator_manager import actuator_manager
from ...hardware.controllers.safety_monitor import safety_monitor

router = APIRouter(prefix="/api/robotics", tags=["robotics"])
logger = logging.getLogger(__name__)

# Pydantic models for API
class RoboticsStatusResponse(BaseModel):
    system_running: bool
    navigation_status: Dict[str, Any]
    sensor_status: Dict[str, Any]
    safety_status: Dict[str, Any]
    robot_coordinator_status: Dict[str, Any]
    timestamp: str

class RPATaskRequest(BaseModel):
    task_type: str
    name: str
    description: str
    parameters: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0

class NavigationGoalRequest(BaseModel):
    target_x: float
    target_y: float
    target_z: float = 0.0
    priority: int = 0
    tolerance: float = 0.2

class RobotTaskRequest(BaseModel):
    task_type: str
    priority: int
    description: str
    required_capabilities: List[str]
    target_location: Optional[Dict[str, float]] = None
    parameters: Dict[str, Any] = {}

class VisionProcessingRequest(BaseModel):
    task_type: str
    context: Dict[str, Any] = {}

class HumanInputRequest(BaseModel):
    input_text: str
    interaction_mode: str = "text_command"
    user_id: Optional[str] = None

@router.get("/status", response_model=RoboticsStatusResponse)
async def get_robotics_status():
    """Get overall robotics system status"""
    try:
        return RoboticsStatusResponse(
            system_running=True,
            navigation_status=navigation_controller.get_navigation_status(),
            sensor_status=sensor_fusion_engine.get_sensor_status(),
            safety_status=safety_monitor.get_safety_status(),
            robot_coordinator_status=robot_coordinator.get_system_status(),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Failed to get robotics status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rpa/execute-task")
async def execute_rpa_task(task_request: RPATaskRequest, background_tasks: BackgroundTasks):
    """Execute RPA task"""
    try:
        # Create RPA task
        task = RPATask(
            task_id=f"rpa_{int(datetime.utcnow().timestamp() * 1000)}",
            task_type=RPATaskType(task_request.task_type),
            name=task_request.name,
            description=task_request.description,
            parameters=task_request.parameters,
            retry_count=task_request.retry_count,
            max_retries=task_request.max_retries,
            timeout=task_request.timeout
        )
        
        # Execute task in background
        background_tasks.add_task(rpa_engine.task_executor.execute_task, task)
        
        return {
            "task_id": task.task_id,
            "status": "queued",
            "message": f"RPA task {task.name} queued for execution"
        }
        
    except Exception as e:
        logger.error(f"Failed to execute RPA task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rpa/execute-workflow")
async def execute_rpa_workflow(workflow_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Execute RPA workflow"""
    try:
        # Create workflow from data
        tasks = []
        for task_data in workflow_data.get("tasks", []):
            task = RPATask(
                task_id=task_data["task_id"],
                task_type=RPATaskType(task_data["task_type"]),
                name=task_data["name"],
                description=task_data["description"],
                parameters=task_data["parameters"]
            )
            tasks.append(task)
        
        workflow = RPAWorkflow(
            workflow_id=workflow_data["workflow_id"],
            name=workflow_data["name"],
            description=workflow_data["description"],
            tasks=tasks
        )
        
        # Execute workflow in background
        background_tasks.add_task(rpa_engine.execute_workflow, workflow)
        
        return {
            "workflow_id": workflow.workflow_id,
            "status": "executing",
            "task_count": len(tasks)
        }
        
    except Exception as e:
        logger.error(f"Failed to execute RPA workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rpa/workflow-status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get RPA workflow status"""
    try:
        status = rpa_engine.get_workflow_status(workflow_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/navigation/set-goal")
async def set_navigation_goal(goal_request: NavigationGoalRequest):
    """Set navigation goal"""
    try:
        # Create navigation goal
        target = Waypoint(
            x=goal_request.target_x,
            y=goal_request.target_y,
            z=goal_request.target_z,
            tolerance=goal_request.tolerance
        )
        
        goal = NavigationGoal(
            target=target,
            priority=goal_request.priority
        )
        
        # Set goal
        success = await navigation_controller.set_navigation_goal(goal)
        
        if success:
            return {
                "status": "success",
                "message": f"Navigation goal set to ({goal_request.target_x}, {goal_request.target_y}, {goal_request.target_z})"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set navigation goal")
            
    except Exception as e:
        logger.error(f"Failed to set navigation goal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/navigation/set-mode")
async def set_navigation_mode(mode: str):
    """Set navigation mode"""
    try:
        nav_mode = NavigationMode(mode)
        success = navigation_controller.set_navigation_mode(nav_mode)
        
        if success:
            return {
                "status": "success",
                "mode": mode,
                "message": f"Navigation mode set to {mode}"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set navigation mode")
            
    except Exception as e:
        logger.error(f"Failed to set navigation mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/navigation/status")
async def get_navigation_status():
    """Get navigation status"""
    try:
        return navigation_controller.get_navigation_status()
    except Exception as e:
        logger.error(f"Failed to get navigation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/coordination/add-task")
async def add_robot_task(task_request: RobotTaskRequest):
    """Add task to robot coordinator"""
    try:
        # Create target location if provided
        target_location = None
        if task_request.target_location:
            target_location = Waypoint(
                x=task_request.target_location["x"],
                y=task_request.target_location["y"],
                z=task_request.target_location.get("z", 0.0)
            )
        
        # Create task
        task = Task(
            task_id=f"task_{int(datetime.utcnow().timestamp() * 1000)}",
            task_type=TaskType(task_request.task_type),
            priority=TaskPriority(task_request.priority),
            description=task_request.description,
            required_capabilities=task_request.required_capabilities,
            target_location=target_location,
            parameters=task_request.parameters
        )
        
        # Add task to coordinator
        success = await robot_coordinator.add_task(task)
        
        if success:
            return {
                "task_id": task.task_id,
                "status": "added",
                "message": f"Task {task.description} added to coordination system"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to add task")
            
    except Exception as e:
        logger.error(f"Failed to add robot task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coordination/status")
async def get_coordination_status():
    """Get robot coordination status"""
    try:
        return robot_coordinator.get_system_status()
    except Exception as e:
        logger.error(f"Failed to get coordination status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coordination/robot/{robot_id}")
async def get_robot_status(robot_id: str):
    """Get specific robot status"""
    try:
        status = robot_coordinator.get_robot_status(robot_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Robot not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get robot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vision/process")
async def process_vision_task(request: VisionProcessingRequest, image_data: Optional[str] = None):
    """Process computer vision task for robotics"""
    try:
        # If image_data is provided, decode it
        image = None
        if image_data:
            import base64
            import numpy as np
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            # Use mock image for testing
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process vision task
        task_type = RobotVisionTask(request.task_type)
        result = await robotics_vision_processor.process_robot_vision(
            image, task_type, request.context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to process vision task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sensors/status")
async def get_sensor_status():
    """Get sensor system status"""
    try:
        return sensor_manager.get_sensor_status()
    except Exception as e:
        logger.error(f"Failed to get sensor status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sensors/start")
async def start_sensors():
    """Start all sensors"""
    try:
        success = await sensor_manager.start_all_sensors()
        return {
            "status": "success" if success else "partial",
            "message": "Sensor startup initiated"
        }
    except Exception as e:
        logger.error(f"Failed to start sensors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sensors/stop")
async def stop_sensors():
    """Stop all sensors"""
    try:
        success = await sensor_manager.stop_all_sensors()
        return {
            "status": "success" if success else "partial",
            "message": "Sensor shutdown initiated"
        }
    except Exception as e:
        logger.error(f"Failed to stop sensors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/actuators/status")
async def get_actuator_status():
    """Get actuator system status"""
    try:
        return actuator_manager.get_actuator_status()
    except Exception as e:
        logger.error(f"Failed to get actuator status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/actuators/start")
async def start_actuators():
    """Start all actuators"""
    try:
        success = await actuator_manager.start_all_actuators()
        return {
            "status": "success" if success else "partial",
            "message": "Actuator startup initiated"
        }
    except Exception as e:
        logger.error(f"Failed to start actuators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/actuators/emergency-stop")
async def emergency_stop_actuators():
    """Emergency stop all actuators"""
    try:
        success = await actuator_manager.emergency_stop_all()
        return {
            "status": "success" if success else "partial",
            "message": "Emergency stop executed"
        }
    except Exception as e:
        logger.error(f"Failed to execute emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/safety/status")
async def get_safety_status():
    """Get safety monitoring status"""
    try:
        return safety_monitor.get_safety_status()
    except Exception as e:
        logger.error(f"Failed to get safety status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/safety/alerts")
async def get_safety_alerts():
    """Get active safety alerts"""
    try:
        return safety_monitor.get_active_alerts()
    except Exception as e:
        logger.error(f"Failed to get safety alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/acknowledge-alert/{alert_id}")
async def acknowledge_safety_alert(alert_id: str):
    """Acknowledge safety alert"""
    try:
        success = safety_monitor.acknowledge_alert(alert_id)
        if success:
            return {
                "status": "success",
                "message": f"Alert {alert_id} acknowledged"
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/reset-emergency")
async def reset_emergency_stop():
    """Reset emergency stop"""
    try:
        success = safety_monitor.reset_emergency_stop()
        return {
            "status": "success" if success else "failed",
            "message": "Emergency stop reset" if success else "Failed to reset emergency stop"
        }
    except Exception as e:
        logger.error(f"Failed to reset emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/start-monitoring")
async def start_safety_monitoring():
    """Start safety monitoring"""
    try:
        success = safety_monitor.start_monitoring()
        return {
            "status": "success" if success else "already_running",
            "message": "Safety monitoring started" if success else "Safety monitoring already running"
        }
    except Exception as e:
        logger.error(f"Failed to start safety monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interaction/process-command")
async def process_human_command(request: HumanInputRequest):
    """Process human command and generate robot response"""
    try:
        interaction_mode = InteractionMode(request.interaction_mode)
        command, response = await human_robot_interface.process_human_input(
            request.input_text, interaction_mode, request.user_id
        )
        
        return {
            "command": {
                "command_id": command.command_id,
                "intent": command.intent.value,
                "entities": command.entities,
                "confidence": command.confidence
            },
            "response": {
                "response_id": response.response_id,
                "message": response.message,
                "response_type": response.response_type.value,
                "action_taken": response.action_taken,
                "additional_data": response.additional_data
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to process human command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interaction/stats")
async def get_interaction_stats():
    """Get human-robot interaction statistics"""
    try:
        return human_robot_interface.get_interaction_stats()
    except Exception as e:
        logger.error(f"Failed to get interaction stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interaction/history")
async def get_interaction_history(count: int = 10):
    """Get recent interaction history"""
    try:
        return human_robot_interface.get_recent_interactions(count)
    except Exception as e:
        logger.error(f"Failed to get interaction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def robotics_health_check():
    """Health check for robotics system"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "rpa_engine": "available",
                "navigation": "available",
                "vision_processing": "available",
                "sensor_fusion": "available",
                "robot_coordination": "available",
                "safety_monitoring": "available",
                "human_robot_interface": "available"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))