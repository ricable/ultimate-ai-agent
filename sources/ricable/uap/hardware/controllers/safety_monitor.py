# hardware/controllers/safety_monitor.py
# Agent 34: Advanced Robotics Integration - Safety Monitoring System

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import math
from collections import deque, defaultdict

class SafetyLevel(Enum):
    """Safety alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SafetyState(Enum):
    """Overall safety state"""
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"

class SafetyZoneType(Enum):
    """Types of safety zones"""
    FORBIDDEN = "forbidden"  # Robot must not enter
    RESTRICTED = "restricted"  # Robot needs permission
    SLOW_ZONE = "slow_zone"  # Reduced speed
    HUMAN_ZONE = "human_zone"  # Human detection required
    WORKSPACE = "workspace"  # Normal operation

@dataclass
class SafetyZone:
    """Safety zone definition"""
    zone_id: str
    zone_type: SafetyZoneType
    boundaries: List[Tuple[float, float, float]]  # 3D polygon vertices
    max_velocity: Optional[float] = None  # m/s
    max_acceleration: Optional[float] = None  # m/s^2
    human_detection_required: bool = False
    emergency_stop_required: bool = False
    description: str = ""
    active: bool = True

@dataclass
class SafetyLimit:
    """Safety limit definition"""
    limit_id: str
    parameter_name: str  # e.g., "velocity", "temperature", "current"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_threshold: float = 0.8  # Fraction of limit before warning
    critical_threshold: float = 0.95  # Fraction of limit before critical
    description: str = ""
    active: bool = True

@dataclass
class SafetyAlert:
    """Safety alert/violation"""
    alert_id: str
    level: SafetyLevel
    source: str  # sensor_id, actuator_id, zone_id, etc.
    message: str
    timestamp: datetime
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    acknowledged: bool = False
    resolved: bool = False
    auto_resolve: bool = True
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class EmergencyStop:
    """Emergency stop event"""
    stop_id: str
    trigger_source: str
    reason: str
    timestamp: datetime
    robots_affected: List[str]
    manual_reset_required: bool = True
    resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class CollisionDetector:
    """Detect potential collisions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.robot_positions: Dict[str, Tuple[float, float, float]] = {}
        self.robot_velocities: Dict[str, Tuple[float, float, float]] = {}
        self.robot_dimensions: Dict[str, Tuple[float, float, float]] = {}  # width, height, depth
        self.safety_distance = 1.0  # meters
        self.prediction_time = 2.0  # seconds
    
    def update_robot_state(self, robot_id: str, position: Tuple[float, float, float], 
                          velocity: Tuple[float, float, float], 
                          dimensions: Tuple[float, float, float] = (0.5, 0.5, 1.0)):
        """Update robot state for collision detection"""
        self.robot_positions[robot_id] = position
        self.robot_velocities[robot_id] = velocity
        self.robot_dimensions[robot_id] = dimensions
    
    def check_collisions(self) -> List[SafetyAlert]:
        """Check for potential collisions between robots"""
        alerts = []
        robot_ids = list(self.robot_positions.keys())
        
        for i, robot1_id in enumerate(robot_ids):
            for robot2_id in robot_ids[i+1:]:
                alert = self._check_robot_collision(robot1_id, robot2_id)
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def _check_robot_collision(self, robot1_id: str, robot2_id: str) -> Optional[SafetyAlert]:
        """Check collision between two specific robots"""
        if robot1_id not in self.robot_positions or robot2_id not in self.robot_positions:
            return None
        
        pos1 = self.robot_positions[robot1_id]
        pos2 = self.robot_positions[robot2_id]
        vel1 = self.robot_velocities.get(robot1_id, (0, 0, 0))
        vel2 = self.robot_velocities.get(robot2_id, (0, 0, 0))
        dim1 = self.robot_dimensions.get(robot1_id, (0.5, 0.5, 1.0))
        dim2 = self.robot_dimensions.get(robot2_id, (0.5, 0.5, 1.0))
        
        # Current distance
        current_distance = math.sqrt(
            (pos1[0] - pos2[0])**2 + 
            (pos1[1] - pos2[1])**2 + 
            (pos1[2] - pos2[2])**2
        )
        
        # Required safety distance (sum of robot radii + safety margin)
        required_distance = (max(dim1[:2]) + max(dim2[:2])) / 2 + self.safety_distance
        
        # Check current collision
        if current_distance < required_distance:
            level = SafetyLevel.CRITICAL if current_distance < required_distance * 0.5 else SafetyLevel.WARNING
            
            return SafetyAlert(
                alert_id=f"collision_{robot1_id}_{robot2_id}_{int(time.time() * 1000)}",
                level=level,
                source=f"collision_detector",
                message=f"Collision risk between {robot1_id} and {robot2_id}",
                timestamp=datetime.utcnow(),
                current_value=current_distance,
                limit_value=required_distance
            )
        
        # Predict future collision
        predicted_distance = self._predict_future_distance(pos1, vel1, pos2, vel2, self.prediction_time)
        
        if predicted_distance < required_distance:
            return SafetyAlert(
                alert_id=f"collision_pred_{robot1_id}_{robot2_id}_{int(time.time() * 1000)}",
                level=SafetyLevel.WARNING,
                source=f"collision_predictor",
                message=f"Predicted collision between {robot1_id} and {robot2_id} in {self.prediction_time}s",
                timestamp=datetime.utcnow(),
                current_value=predicted_distance,
                limit_value=required_distance
            )
        
        return None
    
    def _predict_future_distance(self, pos1: Tuple[float, float, float], vel1: Tuple[float, float, float],
                                pos2: Tuple[float, float, float], vel2: Tuple[float, float, float],
                                time_ahead: float) -> float:
        """Predict distance between robots after given time"""
        # Future positions
        future_pos1 = (
            pos1[0] + vel1[0] * time_ahead,
            pos1[1] + vel1[1] * time_ahead,
            pos1[2] + vel1[2] * time_ahead
        )
        
        future_pos2 = (
            pos2[0] + vel2[0] * time_ahead,
            pos2[1] + vel2[1] * time_ahead,
            pos2[2] + vel2[2] * time_ahead
        )
        
        # Calculate future distance
        return math.sqrt(
            (future_pos1[0] - future_pos2[0])**2 + 
            (future_pos1[1] - future_pos2[1])**2 + 
            (future_pos1[2] - future_pos2[2])**2
        )

class ZoneMonitor:
    """Monitor safety zones"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.safety_zones: Dict[str, SafetyZone] = {}
        self.robot_positions: Dict[str, Tuple[float, float, float]] = {}
        
        # Initialize default zones
        self._create_default_zones()
    
    def _create_default_zones(self):
        """Create default safety zones"""
        # Forbidden zone (example: dangerous machinery area)
        forbidden_zone = SafetyZone(
            zone_id="forbidden_1",
            zone_type=SafetyZoneType.FORBIDDEN,
            boundaries=[(5.0, 5.0, 0.0), (5.0, 7.0, 0.0), (7.0, 7.0, 0.0), (7.0, 5.0, 0.0)],
            emergency_stop_required=True,
            description="High-voltage equipment area"
        )
        self.safety_zones[forbidden_zone.zone_id] = forbidden_zone
        
        # Human work zone
        human_zone = SafetyZone(
            zone_id="human_1",
            zone_type=SafetyZoneType.HUMAN_ZONE,
            boundaries=[(-2.0, -2.0, 0.0), (-2.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, -2.0, 0.0)],
            max_velocity=0.5,  # Slow down near humans
            human_detection_required=True,
            description="Human workspace"
        )
        self.safety_zones[human_zone.zone_id] = human_zone
        
        # Slow zone
        slow_zone = SafetyZone(
            zone_id="slow_1",
            zone_type=SafetyZoneType.SLOW_ZONE,
            boundaries=[(-10.0, -10.0, 0.0), (-10.0, 10.0, 0.0), (10.0, 10.0, 0.0), (10.0, -10.0, 0.0)],
            max_velocity=1.0,
            description="General slow zone"
        )
        self.safety_zones[slow_zone.zone_id] = slow_zone
    
    def add_safety_zone(self, zone: SafetyZone) -> bool:
        """Add safety zone"""
        self.safety_zones[zone.zone_id] = zone
        self.logger.info(f"Added safety zone {zone.zone_id} ({zone.zone_type.value})")
        return True
    
    def remove_safety_zone(self, zone_id: str) -> bool:
        """Remove safety zone"""
        if zone_id in self.safety_zones:
            del self.safety_zones[zone_id]
            self.logger.info(f"Removed safety zone {zone_id}")
            return True
        return False
    
    def update_robot_position(self, robot_id: str, position: Tuple[float, float, float]):
        """Update robot position for zone monitoring"""
        self.robot_positions[robot_id] = position
    
    def check_zone_violations(self) -> List[SafetyAlert]:
        """Check for safety zone violations"""
        alerts = []
        
        for robot_id, position in self.robot_positions.items():
            for zone in self.safety_zones.values():
                if not zone.active:
                    continue
                
                if self._point_in_zone(position, zone):
                    alert = self._check_zone_compliance(robot_id, position, zone)
                    if alert:
                        alerts.append(alert)
        
        return alerts
    
    def _point_in_zone(self, point: Tuple[float, float, float], zone: SafetyZone) -> bool:
        """Check if point is inside zone (simplified 2D polygon)"""
        x, y, z = point
        
        # Simple bounding box check for now
        if not zone.boundaries:
            return False
        
        min_x = min(p[0] for p in zone.boundaries)
        max_x = max(p[0] for p in zone.boundaries)
        min_y = min(p[1] for p in zone.boundaries)
        max_y = max(p[1] for p in zone.boundaries)
        
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def _check_zone_compliance(self, robot_id: str, position: Tuple[float, float, float], 
                              zone: SafetyZone) -> Optional[SafetyAlert]:
        """Check if robot complies with zone rules"""
        if zone.zone_type == SafetyZoneType.FORBIDDEN:
            return SafetyAlert(
                alert_id=f"zone_violation_{robot_id}_{zone.zone_id}_{int(time.time() * 1000)}",
                level=SafetyLevel.EMERGENCY,
                source=f"zone_monitor",
                message=f"Robot {robot_id} entered forbidden zone {zone.zone_id}",
                timestamp=datetime.utcnow()
            )
        
        elif zone.zone_type == SafetyZoneType.HUMAN_ZONE and zone.human_detection_required:
            # This would typically check for human presence
            # For now, we'll simulate human detection
            import random
            if random.random() < 0.1:  # 10% chance of human detection
                return SafetyAlert(
                    alert_id=f"human_zone_{robot_id}_{zone.zone_id}_{int(time.time() * 1000)}",
                    level=SafetyLevel.WARNING,
                    source=f"zone_monitor",
                    message=f"Robot {robot_id} in human zone {zone.zone_id} - human detected",
                    timestamp=datetime.utcnow()
                )
        
        return None
    
    def get_zone_restrictions(self, position: Tuple[float, float, float]) -> Dict[str, Any]:
        """Get restrictions for current position"""
        restrictions = {
            'max_velocity': float('inf'),
            'max_acceleration': float('inf'),
            'emergency_stop_required': False,
            'active_zones': []
        }
        
        for zone in self.safety_zones.values():
            if zone.active and self._point_in_zone(position, zone):
                restrictions['active_zones'].append(zone.zone_id)
                
                if zone.max_velocity is not None:
                    restrictions['max_velocity'] = min(restrictions['max_velocity'], zone.max_velocity)
                
                if zone.max_acceleration is not None:
                    restrictions['max_acceleration'] = min(restrictions['max_acceleration'], zone.max_acceleration)
                
                if zone.emergency_stop_required:
                    restrictions['emergency_stop_required'] = True
        
        return restrictions

class LimitMonitor:
    """Monitor safety limits for various parameters"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.safety_limits: Dict[str, SafetyLimit] = {}
        self.current_values: Dict[str, Dict[str, float]] = defaultdict(dict)  # source -> parameter -> value
        
        # Initialize default limits
        self._create_default_limits()
    
    def _create_default_limits(self):
        """Create default safety limits"""
        # Velocity limits
        velocity_limit = SafetyLimit(
            limit_id="max_velocity",
            parameter_name="velocity",
            max_value=2.0,  # m/s
            warning_threshold=0.8,
            critical_threshold=0.95,
            description="Maximum robot velocity"
        )
        self.safety_limits[velocity_limit.limit_id] = velocity_limit
        
        # Acceleration limits
        acceleration_limit = SafetyLimit(
            limit_id="max_acceleration",
            parameter_name="acceleration",
            max_value=5.0,  # m/s^2
            warning_threshold=0.8,
            critical_threshold=0.95,
            description="Maximum robot acceleration"
        )
        self.safety_limits[acceleration_limit.limit_id] = acceleration_limit
        
        # Temperature limits
        temperature_limit = SafetyLimit(
            limit_id="max_temperature",
            parameter_name="temperature",
            max_value=80.0,  # Celsius
            warning_threshold=0.8,
            critical_threshold=0.9,
            description="Maximum motor temperature"
        )
        self.safety_limits[temperature_limit.limit_id] = temperature_limit
        
        # Current limits
        current_limit = SafetyLimit(
            limit_id="max_current",
            parameter_name="current",
            max_value=10.0,  # Amperes
            warning_threshold=0.8,
            critical_threshold=0.9,
            description="Maximum motor current"
        )
        self.safety_limits[current_limit.limit_id] = current_limit
    
    def add_safety_limit(self, limit: SafetyLimit) -> bool:
        """Add safety limit"""
        self.safety_limits[limit.limit_id] = limit
        self.logger.info(f"Added safety limit {limit.limit_id} for {limit.parameter_name}")
        return True
    
    def update_parameter_value(self, source: str, parameter: str, value: float):
        """Update parameter value from source"""
        self.current_values[source][parameter] = value
    
    def check_limit_violations(self) -> List[SafetyAlert]:
        """Check for safety limit violations"""
        alerts = []
        
        for source, parameters in self.current_values.items():
            for parameter, value in parameters.items():
                for limit in self.safety_limits.values():
                    if limit.parameter_name == parameter and limit.active:
                        alert = self._check_limit(source, parameter, value, limit)
                        if alert:
                            alerts.append(alert)
        
        return alerts
    
    def _check_limit(self, source: str, parameter: str, value: float, 
                    limit: SafetyLimit) -> Optional[SafetyAlert]:
        """Check if value violates limit"""
        violation_type = None
        violation_level = None
        
        if limit.min_value is not None and value < limit.min_value:
            violation_type = "minimum"
            if value < limit.min_value * limit.critical_threshold:
                violation_level = SafetyLevel.CRITICAL
            elif value < limit.min_value * limit.warning_threshold:
                violation_level = SafetyLevel.WARNING
        
        elif limit.max_value is not None and value > limit.max_value:
            violation_type = "maximum"
            if value > limit.max_value * limit.critical_threshold:
                violation_level = SafetyLevel.CRITICAL
            elif value > limit.max_value * limit.warning_threshold:
                violation_level = SafetyLevel.WARNING
        
        if violation_type and violation_level:
            return SafetyAlert(
                alert_id=f"limit_{source}_{parameter}_{int(time.time() * 1000)}",
                level=violation_level,
                source=source,
                message=f"{parameter} {violation_type} limit violation: {value:.2f} (limit: {limit.max_value or limit.min_value:.2f})",
                timestamp=datetime.utcnow(),
                current_value=value,
                limit_value=limit.max_value or limit.min_value
            )
        
        return None

class SafetyMonitor:
    """Main safety monitoring system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.collision_detector = CollisionDetector()
        self.zone_monitor = ZoneMonitor()
        self.limit_monitor = LimitMonitor()
        
        # Safety state
        self.safety_state = SafetyState.SAFE
        self.emergency_stop_active = False
        
        # Alert management
        self.active_alerts: Dict[str, SafetyAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.emergency_stops: Dict[str, EmergencyStop] = {}
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        self.monitoring_frequency = 10.0  # Hz
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_level': defaultdict(int),
            'emergency_stops': 0,
            'uptime_hours': 0.0
        }
        
        self.start_time = datetime.utcnow()
    
    def start_monitoring(self) -> bool:
        """Start safety monitoring"""
        if self.running:
            return False
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started safety monitoring system")
        return True
    
    def stop_monitoring(self) -> bool:
        """Stop safety monitoring"""
        if not self.running:
            return False
        
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        self.logger.info("Stopped safety monitoring system")
        return True
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        loop_time = 1.0 / self.monitoring_frequency
        
        while self.running:
            start_time = time.time()
            
            try:
                # Check all safety conditions
                asyncio.run(self._check_safety_conditions())
                
                # Update safety state
                self._update_safety_state()
                
                # Auto-resolve alerts
                self._auto_resolve_alerts()
                
                # Update statistics
                self._update_statistics()
                
            except Exception as e:
                self.logger.error(f"Safety monitoring error: {e}")
            
            # Maintain loop timing
            elapsed = time.time() - start_time
            sleep_time = max(0, loop_time - elapsed)
            time.sleep(sleep_time)
    
    async def _check_safety_conditions(self):
        """Check all safety conditions"""
        new_alerts = []
        
        # Check collisions
        collision_alerts = self.collision_detector.check_collisions()
        new_alerts.extend(collision_alerts)
        
        # Check zone violations
        zone_alerts = self.zone_monitor.check_zone_violations()
        new_alerts.extend(zone_alerts)
        
        # Check limit violations
        limit_alerts = self.limit_monitor.check_limit_violations()
        new_alerts.extend(limit_alerts)
        
        # Process new alerts
        for alert in new_alerts:
            await self._process_alert(alert)
    
    async def _process_alert(self, alert: SafetyAlert):
        """Process new safety alert"""
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Update statistics
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_level'][alert.level.value] += 1
        
        # Handle emergency alerts
        if alert.level == SafetyLevel.EMERGENCY:
            await self._trigger_emergency_stop(f"Safety alert: {alert.message}", alert.source)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        self.logger.warning(f"Safety alert ({alert.level.value}): {alert.message}")
    
    async def _trigger_emergency_stop(self, reason: str, source: str):
        """Trigger emergency stop"""
        if self.emergency_stop_active:
            return  # Already in emergency stop
        
        self.emergency_stop_active = True
        self.safety_state = SafetyState.EMERGENCY_STOP
        
        # Create emergency stop record
        stop_id = f"estop_{int(time.time() * 1000)}"
        emergency_stop = EmergencyStop(
            stop_id=stop_id,
            trigger_source=source,
            reason=reason,
            timestamp=datetime.utcnow(),
            robots_affected=list(self.collision_detector.robot_positions.keys())
        )
        
        self.emergency_stops[stop_id] = emergency_stop
        self.stats['emergency_stops'] += 1
        
        # Call emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                await callback(emergency_stop)
            except Exception as e:
                self.logger.error(f"Emergency callback failed: {e}")
        
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
    
    def _update_safety_state(self):
        """Update overall safety state"""
        if self.emergency_stop_active:
            self.safety_state = SafetyState.EMERGENCY_STOP
            return
        
        # Count alerts by level
        critical_count = sum(1 for a in self.active_alerts.values() if a.level == SafetyLevel.CRITICAL)
        warning_count = sum(1 for a in self.active_alerts.values() if a.level == SafetyLevel.WARNING)
        
        if critical_count > 0:
            self.safety_state = SafetyState.UNSAFE
        elif warning_count > 0:
            self.safety_state = SafetyState.CAUTION
        else:
            self.safety_state = SafetyState.SAFE
    
    def _auto_resolve_alerts(self):
        """Auto-resolve alerts that are no longer active"""
        current_time = datetime.utcnow()
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.auto_resolve:
                # Auto-resolve alerts older than 30 seconds
                if (current_time - alert.timestamp).total_seconds() > 30.0:
                    alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id)
    
    def _update_statistics(self):
        """Update safety statistics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600.0
        self.stats['uptime_hours'] = uptime
    
    # Public API methods
    
    def update_robot_state(self, robot_id: str, position: Tuple[float, float, float], 
                          velocity: Tuple[float, float, float], 
                          dimensions: Tuple[float, float, float] = (0.5, 0.5, 1.0)):
        """Update robot state for safety monitoring"""
        # Update collision detector
        self.collision_detector.update_robot_state(robot_id, position, velocity, dimensions)
        
        # Update zone monitor
        self.zone_monitor.update_robot_position(robot_id, position)
        
        # Update velocity limits
        velocity_magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
        self.limit_monitor.update_parameter_value(robot_id, "velocity", velocity_magnitude)
    
    def update_actuator_state(self, actuator_id: str, temperature: float, current: float):
        """Update actuator state for safety monitoring"""
        self.limit_monitor.update_parameter_value(actuator_id, "temperature", temperature)
        self.limit_monitor.update_parameter_value(actuator_id, "current", current)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for safety alerts"""
        self.alert_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable):
        """Add callback for emergency stops"""
        self.emergency_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge safety alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Acknowledged alert {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve safety alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            self.logger.info(f"Resolved alert {alert_id}")
            return True
        return False
    
    def reset_emergency_stop(self, stop_id: str = None) -> bool:
        """Reset emergency stop"""
        if not self.emergency_stop_active:
            return True
        
        # Clear all emergency stops if no specific ID provided
        if stop_id is None:
            self.emergency_stop_active = False
            for estop in self.emergency_stops.values():
                estop.resolved = True
        else:
            if stop_id in self.emergency_stops:
                self.emergency_stops[stop_id].resolved = True
                # Check if all emergency stops are resolved
                all_resolved = all(es.resolved for es in self.emergency_stops.values())
                if all_resolved:
                    self.emergency_stop_active = False
        
        if not self.emergency_stop_active:
            self.safety_state = SafetyState.SAFE
            self.logger.info("Emergency stop reset")
        
        return not self.emergency_stop_active
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get overall safety status"""
        return {
            'safety_state': self.safety_state.value,
            'emergency_stop_active': self.emergency_stop_active,
            'active_alerts': len(self.active_alerts),
            'alerts_by_level': {
                level.value: sum(1 for a in self.active_alerts.values() if a.level == level)
                for level in SafetyLevel
            },
            'monitoring_active': self.running,
            'uptime_hours': self.stats['uptime_hours'],
            'total_alerts': self.stats['total_alerts'],
            'emergency_stops': self.stats['emergency_stops']
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        return [asdict(alert) for alert in self.active_alerts.values()]
    
    def get_safety_zones(self) -> List[Dict[str, Any]]:
        """Get list of safety zones"""
        return [asdict(zone) for zone in self.zone_monitor.safety_zones.values()]
    
    def get_zone_restrictions(self, position: Tuple[float, float, float]) -> Dict[str, Any]:
        """Get safety restrictions for position"""
        return self.zone_monitor.get_zone_restrictions(position)
    
    def add_safety_zone(self, zone: SafetyZone) -> bool:
        """Add safety zone"""
        return self.zone_monitor.add_safety_zone(zone)
    
    def add_safety_limit(self, limit: SafetyLimit) -> bool:
        """Add safety limit"""
        return self.limit_monitor.add_safety_limit(limit)

# Global safety monitor instance
safety_monitor = SafetyMonitor()
