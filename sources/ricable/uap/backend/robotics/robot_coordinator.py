# backend/robotics/robot_coordinator.py
# Agent 34: Advanced Robotics Integration - Multi-Robot Coordination System

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import math
import threading
from collections import deque, defaultdict
import uuid

# Import robotics components
from .sensor_fusion import FusedPose, sensor_fusion_engine
from .navigation_planner import NavigationGoal, Waypoint, Path, NavigationMode, navigation_controller
from .vision_processor import robotics_vision_processor

class RobotType(Enum):
    """Types of robots in the system"""
    MOBILE_BASE = "mobile_base"
    MANIPULATOR = "manipulator"
    AERIAL_DRONE = "aerial_drone"
    HUMANOID = "humanoid"
    INDUSTRIAL_ARM = "industrial_arm"
    CLEANING_ROBOT = "cleaning_robot"
    DELIVERY_ROBOT = "delivery_robot"
    SECURITY_ROBOT = "security_robot"
    INSPECTION_ROBOT = "inspection_robot"
    CUSTOM = "custom"

class RobotStatus(Enum):
    """Robot operational status"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OFFLINE = "offline"
    EMERGENCY_STOP = "emergency_stop"

class TaskType(Enum):
    """Types of tasks robots can perform"""
    NAVIGATION = "navigation"
    PICKUP = "pickup"
    DELIVERY = "delivery"
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    SURVEILLANCE = "surveillance"
    MANIPULATION = "manipulation"
    FORMATION = "formation"
    PATROL = "patrol"
    CUSTOM = "custom"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    EMERGENCY = 5

class CoordinationMode(Enum):
    """Multi-robot coordination modes"""
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    HIERARCHICAL = "hierarchical"
    SWARM = "swarm"
    AUCTION = "auction"
    CONSENSUS = "consensus"

@dataclass
class RobotCapability:
    """Robot capability specification"""
    capability_id: str
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reliability: float = 1.0  # 0.0 to 1.0
    energy_cost: float = 1.0  # Relative energy cost
    execution_time: float = 1.0  # Estimated time in seconds

@dataclass
class RobotState:
    """Current state of a robot"""
    robot_id: str
    pose: Optional[FusedPose] = None
    status: RobotStatus = RobotStatus.OFFLINE
    battery_level: float = 100.0  # Percentage
    current_task_id: Optional[str] = None
    capabilities: List[RobotCapability] = field(default_factory=list)
    load_capacity: float = 0.0  # kg
    current_load: float = 0.0  # kg
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Robot:
    """Robot definition"""
    robot_id: str
    name: str
    robot_type: RobotType
    capabilities: List[RobotCapability]
    max_speed: float = 1.0  # m/s
    payload_capacity: float = 5.0  # kg
    battery_capacity: float = 100.0  # Wh
    communication_range: float = 100.0  # meters
    home_position: Waypoint = field(default_factory=lambda: Waypoint(0, 0, 0))
    state: RobotState = None
    
    def __post_init__(self):
        if self.state is None:
            self.state = RobotState(
                robot_id=self.robot_id,
                capabilities=self.capabilities.copy()
            )

@dataclass
class Task:
    """Task definition for robots"""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    description: str
    required_capabilities: List[str]
    target_location: Optional[Waypoint] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    estimated_duration: float = 60.0  # seconds
    assigned_robot_id: Optional[str] = None
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class Formation:
    """Multi-robot formation definition"""
    formation_id: str
    name: str
    robot_positions: Dict[str, Tuple[float, float, float]]  # robot_id -> relative position
    leader_robot_id: str
    formation_type: str = "line"  # line, circle, diamond, custom
    spacing: float = 2.0  # meters
    orientation: float = 0.0  # radians
    active: bool = False

@dataclass
class Conflict:
    """Resource or path conflict between robots"""
    conflict_id: str
    involved_robots: List[str]
    conflict_type: str  # path, resource, communication
    location: Optional[Waypoint] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: int = 1  # 1-5, higher is more severe
    resolution_strategy: Optional[str] = None
    resolved: bool = False

class TaskAllocator:
    """Allocate tasks to robots using various strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def allocate_task(self, task: Task, available_robots: List[Robot], 
                           allocation_strategy: str = "optimal") -> Optional[str]:
        """Allocate task to best available robot"""
        if not available_robots:
            return None
        
        suitable_robots = self._filter_suitable_robots(task, available_robots)
        if not suitable_robots:
            self.logger.warning(f"No suitable robots found for task {task.task_id}")
            return None
        
        if allocation_strategy == "optimal":
            return await self._optimal_allocation(task, suitable_robots)
        elif allocation_strategy == "greedy":
            return self._greedy_allocation(task, suitable_robots)
        elif allocation_strategy == "auction":
            return await self._auction_allocation(task, suitable_robots)
        else:
            return self._greedy_allocation(task, suitable_robots)
    
    def _filter_suitable_robots(self, task: Task, robots: List[Robot]) -> List[Robot]:
        """Filter robots that can perform the task"""
        suitable_robots = []
        
        for robot in robots:
            # Check if robot is available
            if robot.state.status != RobotStatus.IDLE:
                continue
            
            # Check capabilities
            robot_capabilities = {cap.capability_id for cap in robot.capabilities}
            required_capabilities = set(task.required_capabilities)
            
            if not required_capabilities.issubset(robot_capabilities):
                continue
            
            # Check battery level
            if robot.state.battery_level < 20.0:  # Minimum 20% battery
                continue
            
            # Check payload if required
            if 'payload_weight' in task.parameters:
                required_payload = task.parameters['payload_weight']
                if robot.state.current_load + required_payload > robot.payload_capacity:
                    continue
            
            suitable_robots.append(robot)
        
        return suitable_robots
    
    async def _optimal_allocation(self, task: Task, robots: List[Robot]) -> str:
        """Optimal allocation considering multiple factors"""
        best_robot = None
        best_score = float('-inf')
        
        for robot in robots:
            score = await self._calculate_robot_score(task, robot)
            if score > best_score:
                best_score = score
                best_robot = robot
        
        return best_robot.robot_id if best_robot else None
    
    async def _calculate_robot_score(self, task: Task, robot: Robot) -> float:
        """Calculate suitability score for robot-task pair"""
        score = 0.0
        
        # Distance factor (closer is better)
        if task.target_location and robot.state.pose:
            distance = math.sqrt(
                (task.target_location.x - robot.state.pose.position[0])**2 +
                (task.target_location.y - robot.state.pose.position[1])**2
            )
            score += 100.0 / (1.0 + distance)  # Inverse distance
        
        # Battery level factor
        score += robot.state.battery_level / 100.0 * 20.0
        
        # Capability match factor
        robot_capabilities = {cap.capability_id: cap for cap in robot.capabilities}
        capability_score = 0.0
        for req_cap in task.required_capabilities:
            if req_cap in robot_capabilities:
                capability_score += robot_capabilities[req_cap].reliability * 10.0
        score += capability_score
        
        # Load factor (less loaded is better)
        load_factor = 1.0 - (robot.state.current_load / robot.payload_capacity)
        score += load_factor * 10.0
        
        # Priority factor
        priority_multiplier = {
            TaskPriority.LOW: 1.0,
            TaskPriority.NORMAL: 1.2,
            TaskPriority.HIGH: 1.5,
            TaskPriority.URGENT: 2.0,
            TaskPriority.EMERGENCY: 3.0
        }
        score *= priority_multiplier.get(task.priority, 1.0)
        
        return score
    
    def _greedy_allocation(self, task: Task, robots: List[Robot]) -> str:
        """Simple greedy allocation (first available)"""
        return robots[0].robot_id if robots else None
    
    async def _auction_allocation(self, task: Task, robots: List[Robot]) -> str:
        """Auction-based allocation"""
        # Mock auction - robots "bid" based on their suitability
        bids = {}
        
        for robot in robots:
            # Simple bid based on distance and battery
            bid = robot.state.battery_level
            
            if task.target_location and robot.state.pose:
                distance = math.sqrt(
                    (task.target_location.x - robot.state.pose.position[0])**2 +
                    (task.target_location.y - robot.state.pose.position[1])**2
                )
                bid += 100.0 / (1.0 + distance)
            
            bids[robot.robot_id] = bid
        
        # Select highest bidder
        if bids:
            return max(bids, key=bids.get)
        
        return None

class ConflictResolver:
    """Resolve conflicts between robots"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_conflicts: Dict[str, Conflict] = {}
    
    async def detect_conflicts(self, robots: List[Robot]) -> List[Conflict]:
        """Detect potential conflicts between robots"""
        conflicts = []
        
        # Path conflicts
        path_conflicts = await self._detect_path_conflicts(robots)
        conflicts.extend(path_conflicts)
        
        # Resource conflicts
        resource_conflicts = await self._detect_resource_conflicts(robots)
        conflicts.extend(resource_conflicts)
        
        # Communication conflicts
        comm_conflicts = await self._detect_communication_conflicts(robots)
        conflicts.extend(comm_conflicts)
        
        return conflicts
    
    async def _detect_path_conflicts(self, robots: List[Robot]) -> List[Conflict]:
        """Detect path conflicts between robots"""
        conflicts = []
        active_robots = [r for r in robots if r.state.status == RobotStatus.ACTIVE]
        
        for i, robot1 in enumerate(active_robots):
            for robot2 in active_robots[i+1:]:
                if robot1.state.pose and robot2.state.pose:
                    distance = math.sqrt(
                        (robot1.state.pose.position[0] - robot2.state.pose.position[0])**2 +
                        (robot1.state.pose.position[1] - robot2.state.pose.position[1])**2
                    )
                    
                    # Check if robots are too close
                    min_distance = 1.0  # meters
                    if distance < min_distance:
                        conflict = Conflict(
                            conflict_id=f"path_{robot1.robot_id}_{robot2.robot_id}_{int(datetime.utcnow().timestamp())}",
                            involved_robots=[robot1.robot_id, robot2.robot_id],
                            conflict_type="path",
                            location=Waypoint(
                                x=(robot1.state.pose.position[0] + robot2.state.pose.position[0]) / 2,
                                y=(robot1.state.pose.position[1] + robot2.state.pose.position[1]) / 2,
                                z=0
                            ),
                            severity=3 if distance < min_distance / 2 else 2
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_resource_conflicts(self, robots: List[Robot]) -> List[Conflict]:
        """Detect resource conflicts (e.g., charging stations)"""
        # Mock implementation - would check for shared resources
        return []
    
    async def _detect_communication_conflicts(self, robots: List[Robot]) -> List[Conflict]:
        """Detect communication interference"""
        # Mock implementation - would check for communication interference
        return []
    
    async def resolve_conflict(self, conflict: Conflict, robots: List[Robot]) -> bool:
        """Resolve a specific conflict"""
        try:
            if conflict.conflict_type == "path":
                return await self._resolve_path_conflict(conflict, robots)
            elif conflict.conflict_type == "resource":
                return await self._resolve_resource_conflict(conflict, robots)
            elif conflict.conflict_type == "communication":
                return await self._resolve_communication_conflict(conflict, robots)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to resolve conflict {conflict.conflict_id}: {e}")
            return False
    
    async def _resolve_path_conflict(self, conflict: Conflict, robots: List[Robot]) -> bool:
        """Resolve path conflict between robots"""
        involved_robots = [r for r in robots if r.robot_id in conflict.involved_robots]
        
        if len(involved_robots) < 2:
            return False
        
        # Strategy: Have lower priority robot wait or reroute
        robot1, robot2 = involved_robots[0], involved_robots[1]
        
        # Determine which robot should yield (based on priority, battery, etc.)
        if robot1.state.battery_level < robot2.state.battery_level:
            yielding_robot = robot1
        elif robot2.state.battery_level < robot1.state.battery_level:
            yielding_robot = robot2
        else:
            # Random choice if equal
            yielding_robot = robot1
        
        # Command yielding robot to wait or find alternative path
        conflict.resolution_strategy = f"robot_{yielding_robot.robot_id}_yields"
        conflict.resolved = True
        
        self.logger.info(f"Resolved path conflict: {yielding_robot.robot_id} yields")
        return True
    
    async def _resolve_resource_conflict(self, conflict: Conflict, robots: List[Robot]) -> bool:
        """Resolve resource conflict"""
        # Mock implementation
        conflict.resolved = True
        return True
    
    async def _resolve_communication_conflict(self, conflict: Conflict, robots: List[Robot]) -> bool:
        """Resolve communication conflict"""
        # Mock implementation
        conflict.resolved = True
        return True

class RobotCoordinator:
    """Main multi-robot coordination system"""
    
    def __init__(self):
        self.robots: Dict[str, Robot] = {}
        self.tasks: Dict[str, Task] = {}
        self.formations: Dict[str, Formation] = {}
        self.task_allocator = TaskAllocator()
        self.conflict_resolver = ConflictResolver()
        self.logger = logging.getLogger(__name__)
        
        # Coordination settings
        self.coordination_mode = CoordinationMode.CENTRALIZED
        self.heartbeat_timeout = 30.0  # seconds
        self.coordination_frequency = 2.0  # Hz
        
        # Threading
        self.coordination_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Task queue
        self.pending_tasks: deque = deque()
        self.completed_tasks: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'conflicts_resolved': 0,
            'total_distance_traveled': 0.0,
            'total_operation_time': 0.0
        }
    
    def start_coordination(self) -> bool:
        """Start robot coordination system"""
        if self.running:
            return False
        
        self.running = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
        
        self.logger.info("Started robot coordination system")
        return True
    
    def stop_coordination(self) -> bool:
        """Stop robot coordination system"""
        if not self.running:
            return False
        
        self.running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=1.0)
        
        self.logger.info("Stopped robot coordination system")
        return True
    
    def _coordination_loop(self):
        """Main coordination loop"""
        loop_interval = 1.0 / self.coordination_frequency
        
        while self.running:
            try:
                # Update robot states
                asyncio.run(self._update_robot_states())
                
                # Process pending tasks
                asyncio.run(self._process_pending_tasks())
                
                # Monitor active tasks
                asyncio.run(self._monitor_active_tasks())
                
                # Detect and resolve conflicts
                asyncio.run(self._handle_conflicts())
                
                # Update formations
                asyncio.run(self._update_formations())
                
                # Health checks
                asyncio.run(self._perform_health_checks())
                
                # Sleep for next iteration
                asyncio.run(asyncio.sleep(loop_interval))
                
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")
                asyncio.run(asyncio.sleep(loop_interval))
    
    async def _update_robot_states(self):
        """Update states of all robots"""
        current_time = datetime.utcnow()
        
        for robot in self.robots.values():
            # Check for heartbeat timeout
            time_since_heartbeat = (current_time - robot.state.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.heartbeat_timeout:
                if robot.state.status != RobotStatus.OFFLINE:
                    self.logger.warning(f"Robot {robot.robot_id} heartbeat timeout")
                    robot.state.status = RobotStatus.OFFLINE
            
            # Update pose from sensor fusion if available
            # This would typically come from the robot's own sensors
            # For now, we'll use mock data
    
    async def _process_pending_tasks(self):
        """Process tasks in the pending queue"""
        while self.pending_tasks:
            task = self.pending_tasks.popleft()
            
            # Check if task is still valid
            if task.deadline and datetime.utcnow() > task.deadline:
                task.status = "expired"
                self.completed_tasks.append(task)
                continue
            
            # Allocate task to robot
            available_robots = [r for r in self.robots.values() 
                              if r.state.status == RobotStatus.IDLE]
            
            assigned_robot_id = await self.task_allocator.allocate_task(task, available_robots)
            
            if assigned_robot_id:
                task.assigned_robot_id = assigned_robot_id
                task.status = "assigned"
                task.started_at = datetime.utcnow()
                
                # Update robot state
                self.robots[assigned_robot_id].state.status = RobotStatus.BUSY
                self.robots[assigned_robot_id].state.current_task_id = task.task_id
                
                self.logger.info(f"Assigned task {task.task_id} to robot {assigned_robot_id}")
            else:
                # Put task back in queue for later
                self.pending_tasks.append(task)
                break  # Stop processing to avoid infinite loop
    
    async def _monitor_active_tasks(self):
        """Monitor progress of active tasks"""
        active_tasks = [t for t in self.tasks.values() if t.status in ["assigned", "in_progress"]]
        
        for task in active_tasks:
            if task.assigned_robot_id:
                robot = self.robots.get(task.assigned_robot_id)
                if robot:
                    # Check task progress (mock implementation)
                    # In reality, this would query the robot for task status
                    
                    # Check for task timeout
                    if task.started_at:
                        elapsed_time = (datetime.utcnow() - task.started_at).total_seconds()
                        if elapsed_time > task.estimated_duration * 2:  # 2x timeout
                            task.status = "timeout"
                            task.completed_at = datetime.utcnow()
                            
                            # Free up robot
                            robot.state.status = RobotStatus.IDLE
                            robot.state.current_task_id = None
                            
                            self.stats['tasks_failed'] += 1
                            self.logger.warning(f"Task {task.task_id} timed out")
    
    async def _handle_conflicts(self):
        """Detect and resolve conflicts between robots"""
        active_robots = [r for r in self.robots.values() 
                        if r.state.status in [RobotStatus.ACTIVE, RobotStatus.BUSY]]
        
        if len(active_robots) < 2:
            return
        
        # Detect conflicts
        conflicts = await self.conflict_resolver.detect_conflicts(active_robots)
        
        # Resolve conflicts
        for conflict in conflicts:
            if conflict.conflict_id not in self.conflict_resolver.active_conflicts:
                self.conflict_resolver.active_conflicts[conflict.conflict_id] = conflict
                
                success = await self.conflict_resolver.resolve_conflict(conflict, active_robots)
                if success:
                    self.stats['conflicts_resolved'] += 1
                    self.logger.info(f"Resolved conflict {conflict.conflict_id}")
    
    async def _update_formations(self):
        """Update robot formations"""
        for formation in self.formations.values():
            if formation.active:
                await self._maintain_formation(formation)
    
    async def _maintain_formation(self, formation: Formation):
        """Maintain a specific robot formation"""
        leader_robot = self.robots.get(formation.leader_robot_id)
        if not leader_robot or not leader_robot.state.pose:
            return
        
        leader_position = leader_robot.state.pose.position
        
        for robot_id, relative_pos in formation.robot_positions.items():
            if robot_id == formation.leader_robot_id:
                continue
                
            robot = self.robots.get(robot_id)
            if robot and robot.state.status == RobotStatus.ACTIVE:
                # Calculate target position in formation
                target_x = leader_position[0] + relative_pos[0]
                target_y = leader_position[1] + relative_pos[1]
                target_z = leader_position[2] + relative_pos[2]
                
                # Send navigation goal to robot
                goal = NavigationGoal(
                    target=Waypoint(x=target_x, y=target_y, z=target_z),
                    priority=2  # Medium priority for formation maintenance
                )
                
                # This would send the goal to the specific robot's navigation system
                self.logger.debug(f"Formation: {robot_id} -> ({target_x:.2f}, {target_y:.2f})")
    
    async def _perform_health_checks(self):
        """Perform health checks on all robots"""
        for robot in self.robots.values():
            # Check battery level
            if robot.state.battery_level < 15.0 and robot.state.status != RobotStatus.CHARGING:
                self.logger.warning(f"Robot {robot.robot_id} low battery: {robot.state.battery_level}%")
                
                # If robot is working on a task, may need to abort or handover
                if robot.state.current_task_id:
                    task = self.tasks.get(robot.state.current_task_id)
                    if task and task.priority.value < TaskPriority.URGENT.value:
                        # Abort non-urgent task and send robot to charge
                        await self._abort_task(task, "low_battery")
                        await self._send_robot_to_charge(robot)
            
            # Check for errors
            if robot.state.error_message and robot.state.status != RobotStatus.ERROR:
                robot.state.status = RobotStatus.ERROR
                self.logger.error(f"Robot {robot.robot_id} error: {robot.state.error_message}")
    
    async def _abort_task(self, task: Task, reason: str):
        """Abort a task and reassign if possible"""
        task.status = "aborted"
        task.completed_at = datetime.utcnow()
        task.result = {"aborted": True, "reason": reason}
        
        # Free up robot
        if task.assigned_robot_id:
            robot = self.robots.get(task.assigned_robot_id)
            if robot:
                robot.state.status = RobotStatus.IDLE
                robot.state.current_task_id = None
        
        # Try to reassign task if it's important and has retries left
        if (task.priority.value >= TaskPriority.HIGH.value and 
            task.retry_count < task.max_retries):
            task.retry_count += 1
            task.status = "pending"
            task.assigned_robot_id = None
            task.started_at = None
            self.pending_tasks.append(task)
            
            self.logger.info(f"Reassigning aborted task {task.task_id} (retry {task.retry_count})")
    
    async def _send_robot_to_charge(self, robot: Robot):
        """Send robot to charging station"""
        # Create charging task
        charging_task = Task(
            task_id=f"charge_{robot.robot_id}_{int(datetime.utcnow().timestamp())}",
            task_type=TaskType.CUSTOM,
            priority=TaskPriority.HIGH,
            description=f"Charge robot {robot.robot_id}",
            required_capabilities=["navigation"],
            target_location=robot.home_position,  # Assume home is charging station
            parameters={"action": "charge"}
        )
        
        await self.add_task(charging_task)
        robot.state.status = RobotStatus.CHARGING
    
    # Public API methods
    
    async def register_robot(self, robot: Robot) -> bool:
        """Register a new robot with the coordination system"""
        if robot.robot_id in self.robots:
            self.logger.warning(f"Robot {robot.robot_id} already registered")
            return False
        
        self.robots[robot.robot_id] = robot
        robot.state.last_heartbeat = datetime.utcnow()
        robot.state.status = RobotStatus.IDLE
        
        self.logger.info(f"Registered robot {robot.robot_id} ({robot.robot_type.value})")
        return True
    
    async def unregister_robot(self, robot_id: str) -> bool:
        """Unregister robot from system"""
        if robot_id not in self.robots:
            return False
        
        robot = self.robots[robot_id]
        
        # Abort current task if any
        if robot.state.current_task_id:
            task = self.tasks.get(robot.state.current_task_id)
            if task:
                await self._abort_task(task, "robot_unregistered")
        
        del self.robots[robot_id]
        self.logger.info(f"Unregistered robot {robot_id}")
        return True
    
    async def add_task(self, task: Task) -> bool:
        """Add new task to the system"""
        if task.task_id in self.tasks:
            self.logger.warning(f"Task {task.task_id} already exists")
            return False
        
        self.tasks[task.task_id] = task
        self.pending_tasks.append(task)
        
        self.logger.info(f"Added task {task.task_id} ({task.task_type.value})")
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in ["completed", "failed", "cancelled"]:
            return False
        
        # Remove from pending queue if there
        try:
            self.pending_tasks.remove(task)
        except ValueError:
            pass
        
        # Abort if in progress
        if task.assigned_robot_id:
            await self._abort_task(task, "cancelled")
        
        task.status = "cancelled"
        task.completed_at = datetime.utcnow()
        
        self.logger.info(f"Cancelled task {task_id}")
        return True
    
    async def create_formation(self, formation: Formation) -> bool:
        """Create robot formation"""
        if formation.formation_id in self.formations:
            return False
        
        # Validate all robots exist and are available
        for robot_id in formation.robot_positions.keys():
            if robot_id not in self.robots:
                self.logger.error(f"Robot {robot_id} not found for formation")
                return False
        
        self.formations[formation.formation_id] = formation
        formation.active = True
        
        self.logger.info(f"Created formation {formation.formation_id} with {len(formation.robot_positions)} robots")
        return True
    
    async def update_robot_heartbeat(self, robot_id: str, state_update: Dict[str, Any] = None) -> bool:
        """Update robot heartbeat and state"""
        robot = self.robots.get(robot_id)
        if not robot:
            return False
        
        robot.state.last_heartbeat = datetime.utcnow()
        
        if state_update:
            # Update robot state
            if 'battery_level' in state_update:
                robot.state.battery_level = state_update['battery_level']
            
            if 'status' in state_update:
                try:
                    robot.state.status = RobotStatus(state_update['status'])
                except ValueError:
                    pass
            
            if 'pose' in state_update:
                # This would typically be a full FusedPose object
                pass
            
            if 'error_message' in state_update:
                robot.state.error_message = state_update['error_message']
        
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        robot_status_counts = defaultdict(int)
        for robot in self.robots.values():
            robot_status_counts[robot.state.status.value] += 1
        
        task_status_counts = defaultdict(int)
        for task in self.tasks.values():
            task_status_counts[task.status] += 1
        
        return {
            'coordination_mode': self.coordination_mode.value,
            'total_robots': len(self.robots),
            'robot_status': dict(robot_status_counts),
            'total_tasks': len(self.tasks),
            'pending_tasks': len(self.pending_tasks),
            'task_status': dict(task_status_counts),
            'active_formations': sum(1 for f in self.formations.values() if f.active),
            'active_conflicts': len(self.conflict_resolver.active_conflicts),
            'statistics': self.stats.copy(),
            'system_running': self.running
        }
    
    def get_robot_status(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific robot"""
        robot = self.robots.get(robot_id)
        if not robot:
            return None
        
        return {
            'robot_id': robot.robot_id,
            'name': robot.name,
            'type': robot.robot_type.value,
            'status': robot.state.status.value,
            'battery_level': robot.state.battery_level,
            'current_task': robot.state.current_task_id,
            'pose': {
                'position': robot.state.pose.position if robot.state.pose else None,
                'orientation': robot.state.pose.orientation if robot.state.pose else None
            } if robot.state.pose else None,
            'capabilities': [cap.capability_id for cap in robot.capabilities],
            'last_heartbeat': robot.state.last_heartbeat.isoformat(),
            'error_message': robot.state.error_message
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific task"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            'task_id': task.task_id,
            'type': task.task_type.value,
            'priority': task.priority.value,
            'status': task.status,
            'description': task.description,
            'assigned_robot': task.assigned_robot_id,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'retry_count': task.retry_count,
            'result': task.result
        }

# Global robot coordinator
robot_coordinator = RobotCoordinator()
