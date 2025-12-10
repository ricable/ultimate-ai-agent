# hardware/controllers/actuator_manager.py
# Agent 34: Advanced Robotics Integration - Actuator Management System

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import math

class ActuatorType(Enum):
    """Types of actuators"""
    SERVO_MOTOR = "servo_motor"
    STEPPER_MOTOR = "stepper_motor"
    DC_MOTOR = "dc_motor"
    BRUSHLESS_MOTOR = "brushless_motor"
    LINEAR_ACTUATOR = "linear_actuator"
    PNEUMATIC_CYLINDER = "pneumatic_cylinder"
    HYDRAULIC_CYLINDER = "hydraulic_cylinder"
    GRIPPER = "gripper"
    LED = "led"
    SPEAKER = "speaker"
    RELAY = "relay"
    SOLENOID = "solenoid"

class ActuatorStatus(Enum):
    """Actuator status"""
    OFFLINE = "offline"
    IDLE = "idle"
    ACTIVE = "active"
    ERROR = "error"
    CALIBRATING = "calibrating"
    HOMING = "homing"

@dataclass
class ActuatorCommand:
    """Command for actuator"""
    command_id: str
    actuator_id: str
    command_type: str  # position, velocity, torque, power, etc.
    value: float
    duration: Optional[float] = None  # seconds
    timestamp: datetime = None
    priority: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class ActuatorState:
    """Current state of actuator"""
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    current: float = 0.0  # Amperes
    temperature: float = 25.0  # Celsius
    voltage: float = 12.0  # Volts
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.utcnow()

class BaseActuator:
    """Base actuator class"""
    
    def __init__(self, actuator_id: str, actuator_type: ActuatorType):
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type
        self.logger = logging.getLogger(__name__)
        
        # Status and state
        self.status = ActuatorStatus.OFFLINE
        self.state = ActuatorState()
        
        # Limits and parameters
        self.min_position = -180.0  # degrees or mm
        self.max_position = 180.0
        self.max_velocity = 100.0  # deg/s or mm/s
        self.max_torque = 10.0  # Nm or N
        self.max_current = 5.0  # Amperes
        
        # Control parameters
        self.position_tolerance = 1.0  # degrees or mm
        self.velocity_tolerance = 5.0  # deg/s or mm/s
        
        # Command queue
        self.command_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        
        # Control loop
        self.control_thread: Optional[threading.Thread] = None
        self.running = False
        self.control_frequency = 100.0  # Hz
        
        # Current command
        self.current_command: Optional[ActuatorCommand] = None
        self.command_start_time: Optional[datetime] = None
    
    async def start(self) -> bool:
        """Start actuator"""
        if self.running:
            return False
        
        self.running = True
        self.status = ActuatorStatus.IDLE
        
        # Start control loop
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        self.logger.info(f"Started actuator {self.actuator_id}")
        return True
    
    async def stop(self) -> bool:
        """Stop actuator"""
        if not self.running:
            return False
        
        self.running = False
        self.status = ActuatorStatus.OFFLINE
        
        # Stop current command
        await self._execute_stop()
        
        # Wait for control thread
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        
        self.logger.info(f"Stopped actuator {self.actuator_id}")
        return True
    
    def _control_loop(self):
        """Main control loop"""
        loop_time = 1.0 / self.control_frequency
        
        while self.running:
            start_time = time.time()
            
            try:
                # Process commands
                asyncio.run(self._process_commands())
                
                # Update state
                asyncio.run(self._update_state())
                
                # Safety checks
                self._safety_checks()
                
            except Exception as e:
                self.logger.error(f"Actuator control loop error: {e}")
            
            # Maintain loop timing
            elapsed = time.time() - start_time
            sleep_time = max(0, loop_time - elapsed)
            time.sleep(sleep_time)
    
    async def _process_commands(self):
        """Process queued commands"""
        # Check if current command is complete
        if self.current_command:
            if await self._is_command_complete():
                self.logger.debug(f"Command {self.current_command.command_id} completed")
                self.current_command = None
                self.command_start_time = None
                self.status = ActuatorStatus.IDLE
        
        # Get next command if idle
        if not self.current_command and not self.command_queue.empty():
            try:
                self.current_command = await asyncio.wait_for(self.command_queue.get(), timeout=0.001)
                self.command_start_time = datetime.utcnow()
                self.status = ActuatorStatus.ACTIVE
                
                await self._execute_command(self.current_command)
                
            except asyncio.TimeoutError:
                pass  # No commands available
    
    async def _is_command_complete(self) -> bool:
        """Check if current command is complete"""
        if not self.current_command or not self.command_start_time:
            return True
        
        # Check duration timeout
        if self.current_command.duration:
            elapsed = (datetime.utcnow() - self.command_start_time).total_seconds()
            if elapsed >= self.current_command.duration:
                return True
        
        # Check position/velocity tolerance
        if self.current_command.command_type == "position":
            error = abs(self.state.position - self.current_command.value)
            return error <= self.position_tolerance
        
        elif self.current_command.command_type == "velocity":
            error = abs(self.state.velocity - self.current_command.value)
            return error <= self.velocity_tolerance
        
        return False
    
    async def _execute_command(self, command: ActuatorCommand):
        """Execute actuator command - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_command")
    
    async def _execute_stop(self):
        """Stop actuator immediately"""
        # Clear command queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Stop current motion
        self.current_command = None
        self.command_start_time = None
    
    async def _update_state(self):
        """Update actuator state - to be implemented by subclasses"""
        self.state.last_update = datetime.utcnow()
    
    def _safety_checks(self):
        """Perform safety checks"""
        # Check position limits
        if self.state.position < self.min_position or self.state.position > self.max_position:
            self.logger.warning(f"Actuator {self.actuator_id} position out of limits: {self.state.position}")
            self.status = ActuatorStatus.ERROR
        
        # Check current limits
        if self.state.current > self.max_current:
            self.logger.warning(f"Actuator {self.actuator_id} overcurrent: {self.state.current}A")
            self.status = ActuatorStatus.ERROR
        
        # Check temperature
        if self.state.temperature > 80.0:  # Celsius
            self.logger.warning(f"Actuator {self.actuator_id} overtemperature: {self.state.temperature}°C")
            self.status = ActuatorStatus.ERROR
    
    # Public API methods
    
    async def send_command(self, command: ActuatorCommand) -> bool:
        """Send command to actuator"""
        try:
            await self.command_queue.put(command)
            self.logger.debug(f"Queued command {command.command_id} for {self.actuator_id}")
            return True
        except asyncio.QueueFull:
            self.logger.warning(f"Command queue full for actuator {self.actuator_id}")
            return False
    
    async def set_position(self, position: float, duration: Optional[float] = None) -> bool:
        """Set actuator position"""
        command = ActuatorCommand(
            command_id=f"pos_{int(time.time() * 1000)}",
            actuator_id=self.actuator_id,
            command_type="position",
            value=max(self.min_position, min(self.max_position, position)),
            duration=duration
        )
        return await self.send_command(command)
    
    async def set_velocity(self, velocity: float, duration: Optional[float] = None) -> bool:
        """Set actuator velocity"""
        command = ActuatorCommand(
            command_id=f"vel_{int(time.time() * 1000)}",
            actuator_id=self.actuator_id,
            command_type="velocity",
            value=max(-self.max_velocity, min(self.max_velocity, velocity)),
            duration=duration
        )
        return await self.send_command(command)
    
    async def set_torque(self, torque: float, duration: Optional[float] = None) -> bool:
        """Set actuator torque"""
        command = ActuatorCommand(
            command_id=f"tor_{int(time.time() * 1000)}",
            actuator_id=self.actuator_id,
            command_type="torque",
            value=max(-self.max_torque, min(self.max_torque, torque)),
            duration=duration
        )
        return await self.send_command(command)
    
    async def emergency_stop(self) -> bool:
        """Emergency stop actuator"""
        await self._execute_stop()
        self.status = ActuatorStatus.ERROR
        self.logger.warning(f"Emergency stop activated for {self.actuator_id}")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get actuator status"""
        return {
            'actuator_id': self.actuator_id,
            'type': self.actuator_type.value,
            'status': self.status.value,
            'state': asdict(self.state),
            'limits': {
                'min_position': self.min_position,
                'max_position': self.max_position,
                'max_velocity': self.max_velocity,
                'max_torque': self.max_torque,
                'max_current': self.max_current
            },
            'queue_size': self.command_queue.qsize(),
            'current_command': self.current_command.command_id if self.current_command else None
        }

class ServoMotor(BaseActuator):
    """Servo motor actuator"""
    
    def __init__(self, actuator_id: str):
        super().__init__(actuator_id, ActuatorType.SERVO_MOTOR)
        
        # Servo-specific parameters
        self.min_position = -90.0  # degrees
        self.max_position = 90.0
        self.max_velocity = 180.0  # deg/s
        self.max_torque = 2.0  # Nm
        
        # Control gains (PID)
        self.kp = 5.0
        self.ki = 0.1
        self.kd = 0.5
        
        # PID state
        self.integral_error = 0.0
        self.last_error = 0.0
        self.target_position = 0.0
    
    async def _execute_command(self, command: ActuatorCommand):
        """Execute servo command"""
        if command.command_type == "position":
            self.target_position = command.value
            self.logger.debug(f"Servo {self.actuator_id} target position: {self.target_position}")
        
        elif command.command_type == "velocity":
            # For velocity control, update target position incrementally
            dt = 1.0 / self.control_frequency
            self.target_position += command.value * dt
            self.target_position = max(self.min_position, min(self.max_position, self.target_position))
    
    async def _update_state(self):
        """Update servo state using PID control"""
        dt = 1.0 / self.control_frequency
        
        # PID control to reach target position
        error = self.target_position - self.state.position
        self.integral_error += error * dt
        derivative_error = (error - self.last_error) / dt
        
        # Calculate control output
        control_output = (self.kp * error + 
                         self.ki * self.integral_error + 
                         self.kd * derivative_error)
        
        # Update velocity based on control output
        self.state.velocity = max(-self.max_velocity, min(self.max_velocity, control_output))
        
        # Update position based on velocity
        self.state.position += self.state.velocity * dt
        self.state.position = max(self.min_position, min(self.max_position, self.state.position))
        
        # Simulate torque based on error
        self.state.torque = error * 0.1  # Simple proportional
        
        # Simulate current based on torque
        self.state.current = abs(self.state.torque) * 0.5
        
        # Simulate temperature
        self.state.temperature += (self.state.current - 1.0) * 0.01
        self.state.temperature = max(20.0, min(100.0, self.state.temperature))
        
        self.last_error = error
        await super()._update_state()

class StepperMotor(BaseActuator):
    """Stepper motor actuator"""
    
    def __init__(self, actuator_id: str):
        super().__init__(actuator_id, ActuatorType.STEPPER_MOTOR)
        
        # Stepper-specific parameters
        self.steps_per_revolution = 200
        self.microsteps = 16
        self.step_angle = 360.0 / (self.steps_per_revolution * self.microsteps)
        
        self.min_position = -3600.0  # degrees (10 revolutions)
        self.max_position = 3600.0
        self.max_velocity = 360.0  # deg/s
        
        # Stepper state
        self.target_position = 0.0
        self.current_step = 0
    
    async def _execute_command(self, command: ActuatorCommand):
        """Execute stepper command"""
        if command.command_type == "position":
            self.target_position = command.value
            self.logger.debug(f"Stepper {self.actuator_id} target position: {self.target_position}")
        
        elif command.command_type == "velocity":
            # For velocity control, update target position incrementally
            dt = 1.0 / self.control_frequency
            self.target_position += command.value * dt
            self.target_position = max(self.min_position, min(self.max_position, self.target_position))
    
    async def _update_state(self):
        """Update stepper state"""
        dt = 1.0 / self.control_frequency
        
        # Calculate position error
        error = self.target_position - self.state.position
        
        if abs(error) > self.position_tolerance:
            # Move towards target
            direction = 1 if error > 0 else -1
            step_velocity = min(self.max_velocity, abs(error) / dt)
            
            self.state.velocity = direction * step_velocity
            self.state.position += self.state.velocity * dt
            
            # Quantize to step positions
            step_position = round(self.state.position / self.step_angle) * self.step_angle
            self.state.position = step_position
        else:
            self.state.velocity = 0.0
        
        # Simulate current based on movement
        self.state.current = 2.0 if abs(self.state.velocity) > 0 else 0.5
        
        # Simulate temperature
        if self.state.current > 1.0:
            self.state.temperature += 0.01
        else:
            self.state.temperature = max(25.0, self.state.temperature - 0.005)
        
        await super()._update_state()

class LinearActuator(BaseActuator):
    """Linear actuator"""
    
    def __init__(self, actuator_id: str):
        super().__init__(actuator_id, ActuatorType.LINEAR_ACTUATOR)
        
        # Linear actuator parameters
        self.min_position = 0.0   # mm
        self.max_position = 100.0 # mm
        self.max_velocity = 50.0  # mm/s
        self.max_force = 1000.0   # N
        
        # State
        self.target_position = 0.0
        self.load_force = 0.0  # External load
    
    async def _execute_command(self, command: ActuatorCommand):
        """Execute linear actuator command"""
        if command.command_type == "position":
            self.target_position = command.value
            self.logger.debug(f"Linear actuator {self.actuator_id} target position: {self.target_position}mm")
        
        elif command.command_type == "velocity":
            dt = 1.0 / self.control_frequency
            self.target_position += command.value * dt
            self.target_position = max(self.min_position, min(self.max_position, self.target_position))
        
        elif command.command_type == "force":
            # Force control mode
            self.state.torque = command.value  # Using torque field for force
    
    async def _update_state(self):
        """Update linear actuator state"""
        dt = 1.0 / self.control_frequency
        
        # Position control
        error = self.target_position - self.state.position
        
        if abs(error) > self.position_tolerance:
            direction = 1 if error > 0 else -1
            velocity = min(self.max_velocity, abs(error) * 5.0)  # Proportional control
            
            self.state.velocity = direction * velocity
            self.state.position += self.state.velocity * dt
            self.state.position = max(self.min_position, min(self.max_position, self.state.position))
        else:
            self.state.velocity = 0.0
        
        # Simulate force/torque
        if abs(self.state.velocity) > 0:
            self.state.torque = abs(self.state.velocity) * 10.0 + self.load_force
        else:
            self.state.torque = self.load_force
        
        # Current based on force
        self.state.current = self.state.torque / 100.0  # Simple model
        
        await super()._update_state()

class Gripper(BaseActuator):
    """Robotic gripper"""
    
    def __init__(self, actuator_id: str):
        super().__init__(actuator_id, ActuatorType.GRIPPER)
        
        # Gripper parameters
        self.min_position = 0.0   # 0 = fully open
        self.max_position = 100.0 # 100 = fully closed
        self.max_velocity = 50.0  # %/s
        self.max_grip_force = 50.0 # N
        
        # Gripper state
        self.target_position = 0.0
        self.grip_force = 0.0
        self.object_detected = False
    
    async def _execute_command(self, command: ActuatorCommand):
        """Execute gripper command"""
        if command.command_type == "position":
            self.target_position = command.value
            self.logger.debug(f"Gripper {self.actuator_id} target position: {self.target_position}%")
        
        elif command.command_type == "grip":
            # Grip with specified force
            self.grip_force = min(command.value, self.max_grip_force)
            self.target_position = 100.0  # Close gripper
        
        elif command.command_type == "release":
            # Release grip
            self.grip_force = 0.0
            self.target_position = 0.0  # Open gripper
    
    async def _update_state(self):
        """Update gripper state"""
        dt = 1.0 / self.control_frequency
        
        # Position control
        error = self.target_position - self.state.position
        
        if abs(error) > self.position_tolerance:
            direction = 1 if error > 0 else -1
            velocity = min(self.max_velocity, abs(error) * 2.0)
            
            self.state.velocity = direction * velocity
            new_position = self.state.position + self.state.velocity * dt
            
            # Simulate object detection
            if new_position > 70.0 and not self.object_detected:
                # Simulate object contact
                import random
                if random.random() < 0.1:  # 10% chance per update
                    self.object_detected = True
                    self.logger.info(f"Gripper {self.actuator_id} detected object")
            
            if self.object_detected and new_position > 75.0:
                # Stop at object
                self.state.position = 75.0
                self.state.velocity = 0.0
                self.state.torque = self.grip_force
            else:
                self.state.position = max(self.min_position, min(self.max_position, new_position))
        else:
            self.state.velocity = 0.0
            if self.object_detected:
                self.state.torque = self.grip_force
        
        # Reset object detection when opening
        if self.state.position < 30.0:
            self.object_detected = False
        
        # Current based on force
        self.state.current = self.state.torque / 10.0
        
        await super()._update_state()
    
    async def grip(self, force: float = 20.0) -> bool:
        """Grip with specified force"""
        command = ActuatorCommand(
            command_id=f"grip_{int(time.time() * 1000)}",
            actuator_id=self.actuator_id,
            command_type="grip",
            value=force
        )
        return await self.send_command(command)
    
    async def release(self) -> bool:
        """Release grip"""
        command = ActuatorCommand(
            command_id=f"release_{int(time.time() * 1000)}",
            actuator_id=self.actuator_id,
            command_type="release",
            value=0.0
        )
        return await self.send_command(command)

class ActuatorManager:
    """Manage multiple actuators"""
    
    def __init__(self):
        self.actuators: Dict[str, BaseActuator] = {}
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.total_commands = 0
        self.commands_per_actuator: Dict[str, int] = {}
    
    def add_actuator(self, actuator: BaseActuator) -> bool:
        """Add actuator to manager"""
        if actuator.actuator_id in self.actuators:
            self.logger.warning(f"Actuator {actuator.actuator_id} already exists")
            return False
        
        self.actuators[actuator.actuator_id] = actuator
        self.commands_per_actuator[actuator.actuator_id] = 0
        
        self.logger.info(f"Added actuator {actuator.actuator_id} ({actuator.actuator_type.value})")
        return True
    
    def remove_actuator(self, actuator_id: str) -> bool:
        """Remove actuator from manager"""
        if actuator_id not in self.actuators:
            return False
        
        actuator = self.actuators[actuator_id]
        if actuator.running:
            asyncio.run(actuator.stop())
        
        del self.actuators[actuator_id]
        del self.commands_per_actuator[actuator_id]
        
        self.logger.info(f"Removed actuator {actuator_id}")
        return True
    
    async def start_all_actuators(self) -> bool:
        """Start all actuators"""
        success_count = 0
        
        for actuator in self.actuators.values():
            try:
                success = await actuator.start()
                if success:
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to start actuator {actuator.actuator_id}: {e}")
        
        self.logger.info(f"Started {success_count}/{len(self.actuators)} actuators")
        return success_count == len(self.actuators)
    
    async def stop_all_actuators(self) -> bool:
        """Stop all actuators"""
        success_count = 0
        
        for actuator in self.actuators.values():
            try:
                success = await actuator.stop()
                if success:
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to stop actuator {actuator.actuator_id}: {e}")
        
        self.logger.info(f"Stopped {success_count}/{len(self.actuators)} actuators")
        return success_count == len(self.actuators)
    
    async def emergency_stop_all(self) -> bool:
        """Emergency stop all actuators"""
        success_count = 0
        
        for actuator in self.actuators.values():
            try:
                success = await actuator.emergency_stop()
                if success:
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to emergency stop actuator {actuator.actuator_id}: {e}")
        
        self.logger.warning(f"Emergency stopped {success_count}/{len(self.actuators)} actuators")
        return success_count == len(self.actuators)
    
    async def send_command_to_actuator(self, actuator_id: str, command: ActuatorCommand) -> bool:
        """Send command to specific actuator"""
        actuator = self.actuators.get(actuator_id)
        if not actuator:
            self.logger.warning(f"Actuator {actuator_id} not found")
            return False
        
        success = await actuator.send_command(command)
        if success:
            self.total_commands += 1
            self.commands_per_actuator[actuator_id] += 1
        
        return success
    
    def get_actuator_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all actuators"""
        status = {}
        
        for actuator_id, actuator in self.actuators.items():
            status[actuator_id] = actuator.get_status()
            status[actuator_id]['total_commands'] = self.commands_per_actuator[actuator_id]
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get actuator manager statistics"""
        active_actuators = sum(1 for a in self.actuators.values() if a.running)
        
        return {
            'total_actuators': len(self.actuators),
            'active_actuators': active_actuators,
            'total_commands': self.total_commands,
            'commands_per_actuator': self.commands_per_actuator.copy(),
            'actuator_types': {a.actuator_type.value: a.actuator_id for a in self.actuators.values()}
        }

# Create default actuator configuration
def create_default_actuator_suite() -> ActuatorManager:
    """Create a default suite of actuators for robotics"""
    manager = ActuatorManager()
    
    # Add robotic arm joints (6-DOF)
    for i in range(6):
        servo = ServoMotor(f"joint_{i+1}")
        servo.min_position = -180.0
        servo.max_position = 180.0
        manager.add_actuator(servo)
    
    # Add gripper
    gripper = Gripper("gripper_main")
    manager.add_actuator(gripper)
    
    # Add linear actuator for lift
    linear = LinearActuator("lift_actuator")
    linear.max_position = 500.0  # 50cm range
    manager.add_actuator(linear)
    
    # Add stepper motors for precise positioning
    for i in range(2):
        stepper = StepperMotor(f"stepper_{i+1}")
        manager.add_actuator(stepper)
    
    return manager

# Global actuator manager instance
actuator_manager = create_default_actuator_suite()
