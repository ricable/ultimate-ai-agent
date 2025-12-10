# backend/robotics/navigation_planner.py
# Agent 34: Advanced Robotics Integration - Autonomous Navigation System

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import math
import heapq
from collections import deque
import threading

# Numerical libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Import robotics components
from .sensor_fusion import FusedPose, sensor_fusion_engine
from .vision_processor import Obstacle, robotics_vision_processor

class NavigationMode(Enum):
    """Navigation modes for different scenarios"""
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    ASSISTED = "assisted"
    WAYPOINT = "waypoint"
    PATROL = "patrol"
    RETURN_TO_BASE = "return_to_base"
    EMERGENCY_STOP = "emergency_stop"

class PathPlanningAlgorithm(Enum):
    """Path planning algorithms"""
    A_STAR = "a_star"
    RRT = "rrt"  # Rapidly-exploring Random Tree
    RRT_STAR = "rrt_star"
    DIJKSTRA = "dijkstra"
    POTENTIAL_FIELD = "potential_field"
    DWA = "dwa"  # Dynamic Window Approach
    TEB = "teb"  # Time Elastic Band

class NavigationStatus(Enum):
    """Navigation system status"""
    IDLE = "idle"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    REPLANNING = "replanning"
    GOAL_REACHED = "goal_reached"
    STUCK = "stuck"
    ERROR = "error"

@dataclass
class Waypoint:
    """Navigation waypoint"""
    x: float
    y: float
    z: float = 0.0
    orientation: Optional[float] = None  # yaw angle in radians
    tolerance: float = 0.1  # meters
    speed: Optional[float] = None  # m/s
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Path:
    """Navigation path"""
    waypoints: List[Waypoint]
    total_distance: float
    estimated_time: float  # seconds
    created_at: datetime
    algorithm_used: PathPlanningAlgorithm
    validity_duration: float = 60.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if path is still valid"""
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed < self.validity_duration

@dataclass
class NavigationGoal:
    """Navigation goal with constraints"""
    target: Waypoint
    priority: int = 0  # Higher values = higher priority
    deadline: Optional[datetime] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Map2D:
    """2D occupancy grid map"""
    width: int  # cells
    height: int  # cells
    resolution: float  # meters per cell
    origin_x: float  # meters
    origin_y: float  # meters
    data: np.ndarray  # 0=free, 1=occupied, -1=unknown
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x = self.origin_x + grid_x * self.resolution
        y = self.origin_y + grid_y * self.resolution
        return x, y
    
    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid cell is valid"""
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height
    
    def is_free_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid cell is free"""
        if not self.is_valid_cell(grid_x, grid_y):
            return False
        return self.data[grid_y, grid_x] == 0
    
    def set_obstacle(self, x: float, y: float, radius: float = 0.1):
        """Set obstacle in map"""
        grid_x, grid_y = self.world_to_grid(x, y)
        radius_cells = int(radius / self.resolution)
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                gx, gy = grid_x + dx, grid_y + dy
                if self.is_valid_cell(gx, gy):
                    if dx*dx + dy*dy <= radius_cells*radius_cells:
                        self.data[gy, gx] = 1

class AStarPlanner:
    """A* path planning algorithm"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def plan_path(self, start: Waypoint, goal: Waypoint, 
                       occupancy_map: Map2D) -> Optional[Path]:
        """Plan path using A* algorithm"""
        if not NUMPY_AVAILABLE:
            return self._mock_path(start, goal)
        
        try:
            # Convert to grid coordinates
            start_grid = occupancy_map.world_to_grid(start.x, start.y)
            goal_grid = occupancy_map.world_to_grid(goal.x, goal.y)
            
            # Check if start and goal are valid
            if not occupancy_map.is_free_cell(*start_grid):
                self.logger.error("Start position is not free")
                return None
            
            if not occupancy_map.is_free_cell(*goal_grid):
                self.logger.error("Goal position is not free")
                return None
            
            # A* search
            open_list = [(0, start_grid)]  # (f_score, cell)
            closed_set: Set[Tuple[int, int]] = set()
            
            g_score = {start_grid: 0}
            f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
            came_from = {}
            
            while open_list:
                current = heapq.heappop(open_list)[1]
                
                if current == goal_grid:
                    # Reconstruct path
                    path_cells = self._reconstruct_path(came_from, current)
                    waypoints = self._cells_to_waypoints(path_cells, occupancy_map)
                    
                    # Calculate path metrics
                    total_distance = self._calculate_path_distance(waypoints)
                    estimated_time = total_distance / 1.0  # Assuming 1 m/s default speed
                    
                    return Path(
                        waypoints=waypoints,
                        total_distance=total_distance,
                        estimated_time=estimated_time,
                        created_at=datetime.utcnow(),
                        algorithm_used=PathPlanningAlgorithm.A_STAR
                    )
                
                closed_set.add(current)
                
                # Check neighbors
                for neighbor in self._get_neighbors(current, occupancy_map):
                    if neighbor in closed_set:
                        continue
                    
                    tentative_g_score = g_score[current] + self._distance(current, neighbor)
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                        
                        if neighbor not in [item[1] for item in open_list]:
                            heapq.heappush(open_list, (f_score[neighbor], neighbor))
            
            self.logger.warning("No path found")
            return None
            
        except Exception as e:
            self.logger.error(f"A* planning failed: {e}")
            return self._mock_path(start, goal)
    
    def _heuristic(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        dx = cell1[0] - cell2[0]
        dy = cell1[1] - cell2[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _distance(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> float:
        """Distance between adjacent cells"""
        dx = abs(cell1[0] - cell2[0])
        dy = abs(cell1[1] - cell2[1])
        
        if dx == 1 and dy == 1:
            return math.sqrt(2)  # Diagonal movement
        else:
            return 1.0  # Straight movement
    
    def _get_neighbors(self, cell: Tuple[int, int], occupancy_map: Map2D) -> List[Tuple[int, int]]:
        """Get valid neighboring cells"""
        neighbors = []
        x, y = cell
        
        # 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if occupancy_map.is_free_cell(nx, ny):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _cells_to_waypoints(self, cells: List[Tuple[int, int]], occupancy_map: Map2D) -> List[Waypoint]:
        """Convert grid cells to waypoints"""
        waypoints = []
        for cell in cells:
            x, y = occupancy_map.grid_to_world(cell[0], cell[1])
            waypoints.append(Waypoint(x=x, y=y))
        return waypoints
    
    def _calculate_path_distance(self, waypoints: List[Waypoint]) -> float:
        """Calculate total path distance"""
        total_distance = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i-1].x
            dy = waypoints[i].y - waypoints[i-1].y
            dz = waypoints[i].z - waypoints[i-1].z
            total_distance += math.sqrt(dx*dx + dy*dy + dz*dz)
        return total_distance
    
    def _mock_path(self, start: Waypoint, goal: Waypoint) -> Path:
        """Create mock path for testing"""
        # Simple straight line path
        waypoints = [start, goal]
        distance = math.sqrt(
            (goal.x - start.x)**2 + 
            (goal.y - start.y)**2 + 
            (goal.z - start.z)**2
        )
        
        return Path(
            waypoints=waypoints,
            total_distance=distance,
            estimated_time=distance / 1.0,  # 1 m/s
            created_at=datetime.utcnow(),
            algorithm_used=PathPlanningAlgorithm.A_STAR
        )

class DynamicWindowApproach:
    """Dynamic Window Approach for local path planning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # DWA parameters
        self.max_linear_vel = 2.0  # m/s
        self.max_angular_vel = 1.0  # rad/s
        self.linear_acc = 1.0  # m/s^2
        self.angular_acc = 1.0  # rad/s^2
        self.dt = 0.1  # time step
        self.predict_time = 2.0  # prediction time
        
        # Weights for optimization
        self.heading_weight = 0.2
        self.distance_weight = 0.1
        self.velocity_weight = 0.2
        self.obstacle_weight = 0.5
    
    async def compute_velocity_command(self, current_pose: FusedPose, 
                                     current_vel: Tuple[float, float],
                                     goal: Waypoint, 
                                     obstacles: List[Obstacle]) -> Tuple[float, float]:
        """Compute optimal velocity command using DWA"""
        if not NUMPY_AVAILABLE:
            return self._mock_velocity_command(current_pose, goal)
        
        try:
            # Current state
            robot_x, robot_y = current_pose.position[:2]
            robot_yaw = self._quaternion_to_yaw(current_pose.orientation)
            v_current, w_current = current_vel
            
            # Dynamic window
            v_min = max(0, v_current - self.linear_acc * self.dt)
            v_max = min(self.max_linear_vel, v_current + self.linear_acc * self.dt)
            w_min = max(-self.max_angular_vel, w_current - self.angular_acc * self.dt)
            w_max = min(self.max_angular_vel, w_current + self.angular_acc * self.dt)
            
            best_v, best_w = 0.0, 0.0
            best_score = float('-inf')
            
            # Sample velocity space
            v_samples = np.linspace(v_min, v_max, 10)
            w_samples = np.linspace(w_min, w_max, 20)
            
            for v in v_samples:
                for w in w_samples:
                    # Predict trajectory
                    trajectory = self._predict_trajectory(robot_x, robot_y, robot_yaw, v, w)
                    
                    # Check collision
                    if self._check_collision(trajectory, obstacles):
                        continue
                    
                    # Calculate score
                    score = self._calculate_score(trajectory, goal, v, w)
                    
                    if score > best_score:
                        best_score = score
                        best_v, best_w = v, w
            
            return best_v, best_w
            
        except Exception as e:
            self.logger.error(f"DWA computation failed: {e}")
            return self._mock_velocity_command(current_pose, goal)
    
    def _predict_trajectory(self, x: float, y: float, yaw: float, 
                           v: float, w: float) -> List[Tuple[float, float, float]]:
        """Predict robot trajectory"""
        trajectory = []
        current_x, current_y, current_yaw = x, y, yaw
        
        steps = int(self.predict_time / self.dt)
        for _ in range(steps):
            current_x += v * math.cos(current_yaw) * self.dt
            current_y += v * math.sin(current_yaw) * self.dt
            current_yaw += w * self.dt
            trajectory.append((current_x, current_y, current_yaw))
        
        return trajectory
    
    def _check_collision(self, trajectory: List[Tuple[float, float, float]], 
                        obstacles: List[Obstacle]) -> bool:
        """Check if trajectory collides with obstacles"""
        robot_radius = 0.3  # meters
        
        for x, y, _ in trajectory:
            for obstacle in obstacles:
                obs_x, obs_y, obs_z = obstacle.position
                distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                # Simple circular collision check
                if distance < robot_radius + max(obstacle.dimensions[:2]) / 2:
                    return True
        
        return False
    
    def _calculate_score(self, trajectory: List[Tuple[float, float, float]], 
                        goal: Waypoint, v: float, w: float) -> float:
        """Calculate trajectory score"""
        if not trajectory:
            return float('-inf')
        
        final_x, final_y, final_yaw = trajectory[-1]
        
        # Heading score (how well aligned with goal)
        goal_angle = math.atan2(goal.y - final_y, goal.x - final_x)
        heading_diff = abs(self._angle_diff(final_yaw, goal_angle))
        heading_score = math.pi - heading_diff
        
        # Distance score (closer to goal is better)
        distance_to_goal = math.sqrt(
            (goal.x - final_x)**2 + (goal.y - final_y)**2
        )
        distance_score = 1.0 / (1.0 + distance_to_goal)
        
        # Velocity score (higher velocity preferred)
        velocity_score = v / self.max_linear_vel
        
        # Combined score
        total_score = (
            self.heading_weight * heading_score +
            self.distance_weight * distance_score +
            self.velocity_weight * velocity_score
        )
        
        return total_score
    
    def _quaternion_to_yaw(self, quaternion: Tuple[float, float, float, float]) -> float:
        """Convert quaternion to yaw angle"""
        w, x, y, z = quaternion
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    def _angle_diff(self, angle1: float, angle2: float) -> float:
        """Calculate smallest angle difference"""
        diff = angle1 - angle2
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    def _mock_velocity_command(self, current_pose: FusedPose, goal: Waypoint) -> Tuple[float, float]:
        """Mock velocity command for testing"""
        robot_x, robot_y = current_pose.position[:2]
        
        # Simple proportional controller
        dx = goal.x - robot_x
        dy = goal.y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:
            return 0.0, 0.0
        
        # Target angle
        target_angle = math.atan2(dy, dx)
        robot_yaw = self._quaternion_to_yaw(current_pose.orientation)
        angle_error = self._angle_diff(target_angle, robot_yaw)
        
        # Simple control
        linear_vel = min(0.5, distance * 0.5)
        angular_vel = angle_error * 0.5
        
        return linear_vel, angular_vel

class NavigationController:
    """Main navigation controller"""
    
    def __init__(self):
        self.astar_planner = AStarPlanner()
        self.dwa_planner = DynamicWindowApproach()
        self.logger = logging.getLogger(__name__)
        
        # Navigation state
        self.current_mode = NavigationMode.MANUAL
        self.navigation_status = NavigationStatus.IDLE
        self.current_goal: Optional[NavigationGoal] = None
        self.current_path: Optional[Path] = None
        self.current_waypoint_index = 0
        
        # Navigation parameters
        self.waypoint_tolerance = 0.1  # meters
        self.goal_tolerance = 0.2  # meters
        self.replanning_interval = 5.0  # seconds
        self.max_stuck_time = 10.0  # seconds
        
        # State tracking
        self.last_replan_time = datetime.utcnow()
        self.stuck_start_time: Optional[datetime] = None
        self.last_pose: Optional[FusedPose] = None
        
        # Threading
        self.control_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Map
        self.occupancy_map: Optional[Map2D] = None
        self.obstacles: List[Obstacle] = []
        
        # Initialize default map
        self._initialize_default_map()
    
    def _initialize_default_map(self):
        """Initialize default occupancy map"""
        if NUMPY_AVAILABLE:
            # 20x20 meter map with 0.1m resolution
            width, height = 200, 200
            resolution = 0.1
            origin_x, origin_y = -10.0, -10.0
            
            # Start with free space
            data = np.zeros((height, width), dtype=np.int8)
            
            # Add some obstacles for testing
            self.occupancy_map = Map2D(
                width=width,
                height=height,
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
                data=data
            )
            
            # Add some test obstacles
            self.occupancy_map.set_obstacle(2.0, 2.0, 0.5)
            self.occupancy_map.set_obstacle(-3.0, 1.0, 0.3)
            self.occupancy_map.set_obstacle(0.0, -2.0, 0.4)
    
    def start_navigation(self) -> bool:
        """Start navigation controller"""
        if self.running:
            return False
        
        self.running = True
        self.control_thread = threading.Thread(target=self._navigation_loop, daemon=True)
        self.control_thread.start()
        
        self.logger.info("Started navigation controller")
        return True
    
    def stop_navigation(self) -> bool:
        """Stop navigation controller"""
        if not self.running:
            return False
        
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        
        self.navigation_status = NavigationStatus.IDLE
        self.logger.info("Stopped navigation controller")
        return True
    
    def _navigation_loop(self):
        """Main navigation control loop"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Get current pose
                current_pose = sensor_fusion_engine.get_current_pose()
                if current_pose is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Update navigation state
                self._update_navigation_state(current_pose, current_time)
                
                # Process based on current mode
                if self.current_mode == NavigationMode.AUTONOMOUS:
                    await self._process_autonomous_navigation(current_pose, current_time)
                elif self.current_mode == NavigationMode.WAYPOINT:
                    await self._process_waypoint_navigation(current_pose, current_time)
                
                # Update obstacles from vision system
                await self._update_obstacles()
                
                # Sleep for control loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Navigation loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_autonomous_navigation(self, current_pose: FusedPose, current_time: datetime):
        """Process autonomous navigation"""
        if self.current_goal is None:
            self.navigation_status = NavigationStatus.IDLE
            return
        
        # Check if goal is reached
        if self._is_goal_reached(current_pose):
            self.navigation_status = NavigationStatus.GOAL_REACHED
            self.current_goal = None
            self.current_path = None
            return
        
        # Check if replanning is needed
        if (self.current_path is None or 
            not self.current_path.is_valid() or
            (current_time - self.last_replan_time).total_seconds() > self.replanning_interval):
            
            await self._replan_path(current_pose)
            self.last_replan_time = current_time
        
        # Execute current path
        if self.current_path:
            await self._execute_path(current_pose)
    
    async def _process_waypoint_navigation(self, current_pose: FusedPose, current_time: datetime):
        """Process waypoint-based navigation"""
        if not self.current_path or not self.current_path.waypoints:
            self.navigation_status = NavigationStatus.IDLE
            return
        
        waypoints = self.current_path.waypoints
        
        # Check if current waypoint is reached
        if self.current_waypoint_index < len(waypoints):
            current_waypoint = waypoints[self.current_waypoint_index]
            
            if self._is_waypoint_reached(current_pose, current_waypoint):
                self.current_waypoint_index += 1
                
                if self.current_waypoint_index >= len(waypoints):
                    self.navigation_status = NavigationStatus.GOAL_REACHED
                    self.current_path = None
                    self.current_waypoint_index = 0
                    return
        
        # Navigate to current waypoint
        if self.current_waypoint_index < len(waypoints):
            target_waypoint = waypoints[self.current_waypoint_index]
            await self._navigate_to_waypoint(current_pose, target_waypoint)
    
    async def _replan_path(self, current_pose: FusedPose):
        """Replan path to current goal"""
        if self.current_goal is None:
            return
        
        self.navigation_status = NavigationStatus.PLANNING
        
        # Create start waypoint from current pose
        start_waypoint = Waypoint(
            x=current_pose.position[0],
            y=current_pose.position[1],
            z=current_pose.position[2]
        )
        
        # Plan new path
        new_path = await self.astar_planner.plan_path(
            start_waypoint, 
            self.current_goal.target, 
            self.occupancy_map
        )
        
        if new_path:
            self.current_path = new_path
            self.current_waypoint_index = 0
            self.navigation_status = NavigationStatus.NAVIGATING
            self.logger.info(f"Replanned path with {len(new_path.waypoints)} waypoints")
        else:
            self.navigation_status = NavigationStatus.ERROR
            self.logger.error("Failed to replan path")
    
    async def _execute_path(self, current_pose: FusedPose):
        """Execute current path using local planner"""
        if not self.current_path or not self.current_path.waypoints:
            return
        
        # Get next waypoint
        if self.current_waypoint_index < len(self.current_path.waypoints):
            target_waypoint = self.current_path.waypoints[self.current_waypoint_index]
            
            # Use DWA for local navigation
            current_vel = current_pose.linear_velocity[:2]
            
            velocity_command = await self.dwa_planner.compute_velocity_command(
                current_pose, current_vel, target_waypoint, self.obstacles
            )
            
            # Send velocity command (would interface with robot hardware)
            self._send_velocity_command(velocity_command)
            
            # Check if waypoint reached
            if self._is_waypoint_reached(current_pose, target_waypoint):
                self.current_waypoint_index += 1
    
    async def _navigate_to_waypoint(self, current_pose: FusedPose, waypoint: Waypoint):
        """Navigate to specific waypoint"""
        current_vel = current_pose.linear_velocity[:2]
        
        velocity_command = await self.dwa_planner.compute_velocity_command(
            current_pose, current_vel, waypoint, self.obstacles
        )
        
        self._send_velocity_command(velocity_command)
    
    async def _update_obstacles(self):
        """Update obstacle information from vision system"""
        # This would typically get obstacles from vision processor
        # For now, we'll use a mock implementation
        self.obstacles = [
            Obstacle(
                obstacle_id="static_1",
                position=(1.0, 1.0, 0.0),
                dimensions=(0.5, 0.5, 1.0),
                obstacle_type="static",
                confidence=0.9
            )
        ]
        
        # Update occupancy map with new obstacles
        if self.occupancy_map:
            for obstacle in self.obstacles:
                self.occupancy_map.set_obstacle(
                    obstacle.position[0], 
                    obstacle.position[1], 
                    max(obstacle.dimensions[:2]) / 2
                )
    
    def _update_navigation_state(self, current_pose: FusedPose, current_time: datetime):
        """Update navigation state based on current conditions"""
        # Check if robot is stuck
        if self.last_pose is not None:
            distance_moved = math.sqrt(
                (current_pose.position[0] - self.last_pose.position[0])**2 +
                (current_pose.position[1] - self.last_pose.position[1])**2
            )
            
            if distance_moved < 0.01:  # Very small movement
                if self.stuck_start_time is None:
                    self.stuck_start_time = current_time
                elif (current_time - self.stuck_start_time).total_seconds() > self.max_stuck_time:
                    self.navigation_status = NavigationStatus.STUCK
            else:
                self.stuck_start_time = None
        
        self.last_pose = current_pose
    
    def _is_goal_reached(self, current_pose: FusedPose) -> bool:
        """Check if navigation goal is reached"""
        if self.current_goal is None:
            return False
        
        distance = math.sqrt(
            (current_pose.position[0] - self.current_goal.target.x)**2 +
            (current_pose.position[1] - self.current_goal.target.y)**2 +
            (current_pose.position[2] - self.current_goal.target.z)**2
        )
        
        return distance < self.goal_tolerance
    
    def _is_waypoint_reached(self, current_pose: FusedPose, waypoint: Waypoint) -> bool:
        """Check if waypoint is reached"""
        distance = math.sqrt(
            (current_pose.position[0] - waypoint.x)**2 +
            (current_pose.position[1] - waypoint.y)**2 +
            (current_pose.position[2] - waypoint.z)**2
        )
        
        return distance < waypoint.tolerance
    
    def _send_velocity_command(self, velocity_command: Tuple[float, float]):
        """Send velocity command to robot hardware"""
        linear_vel, angular_vel = velocity_command
        
        # This would interface with actual robot hardware
        # For now, just log the command
        self.logger.debug(f"Velocity command: linear={linear_vel:.2f}, angular={angular_vel:.2f}")
    
    async def set_navigation_goal(self, goal: NavigationGoal) -> bool:
        """Set new navigation goal"""
        self.current_goal = goal
        self.current_path = None
        self.current_waypoint_index = 0
        self.navigation_status = NavigationStatus.PLANNING
        
        self.logger.info(f"Set navigation goal: ({goal.target.x}, {goal.target.y}, {goal.target.z})")
        return True
    
    async def set_waypoint_path(self, waypoints: List[Waypoint]) -> bool:
        """Set waypoint-based path"""
        if not waypoints:
            return False
        
        total_distance = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i-1].x
            dy = waypoints[i].y - waypoints[i-1].y
            dz = waypoints[i].z - waypoints[i-1].z
            total_distance += math.sqrt(dx*dx + dy*dy + dz*dz)
        
        self.current_path = Path(
            waypoints=waypoints,
            total_distance=total_distance,
            estimated_time=total_distance / 1.0,  # Assume 1 m/s
            created_at=datetime.utcnow(),
            algorithm_used=PathPlanningAlgorithm.A_STAR
        )
        
        self.current_waypoint_index = 0
        self.current_mode = NavigationMode.WAYPOINT
        self.navigation_status = NavigationStatus.NAVIGATING
        
        self.logger.info(f"Set waypoint path with {len(waypoints)} waypoints")
        return True
    
    def set_navigation_mode(self, mode: NavigationMode) -> bool:
        """Set navigation mode"""
        self.current_mode = mode
        
        if mode == NavigationMode.EMERGENCY_STOP:
            self.navigation_status = NavigationStatus.IDLE
            self.current_goal = None
            self.current_path = None
            self._send_velocity_command((0.0, 0.0))
        
        self.logger.info(f"Set navigation mode: {mode.value}")
        return True
    
    def get_navigation_status(self) -> Dict[str, Any]:
        """Get current navigation status"""
        status = {
            'mode': self.current_mode.value,
            'status': self.navigation_status.value,
            'has_goal': self.current_goal is not None,
            'has_path': self.current_path is not None,
            'obstacles_detected': len(self.obstacles)
        }
        
        if self.current_goal:
            status['goal'] = {
                'x': self.current_goal.target.x,
                'y': self.current_goal.target.y,
                'z': self.current_goal.target.z,
                'priority': self.current_goal.priority
            }
        
        if self.current_path:
            status['path'] = {
                'waypoints_count': len(self.current_path.waypoints),
                'current_waypoint': self.current_waypoint_index,
                'total_distance': self.current_path.total_distance,
                'estimated_time': self.current_path.estimated_time,
                'is_valid': self.current_path.is_valid()
            }
        
        return status

# Global navigation controller
navigation_controller = NavigationController()
