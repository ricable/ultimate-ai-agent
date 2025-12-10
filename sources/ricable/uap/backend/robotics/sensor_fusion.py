# backend/robotics/sensor_fusion.py
# Agent 34: Advanced Robotics Integration - Multi-Sensor Data Fusion

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import math
import threading
from collections import deque

# Numerical libraries
try:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from scipy import signal
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class SensorType(Enum):
    """Types of sensors in robotics system"""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"  # Inertial Measurement Unit
    GPS = "gps"
    ULTRASONIC = "ultrasonic"
    ODOMETRY = "odometry"
    ENCODER = "encoder"
    GYROSCOPE = "gyroscope"
    ACCELEROMETER = "accelerometer"
    MAGNETOMETER = "magnetometer"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FORCE_TORQUE = "force_torque"
    PROXIMITY = "proximity"

class SensorStatus(Enum):
    """Status of individual sensors"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    CALIBRATING = "calibrating"
    TIMEOUT = "timeout"
    MAINTENANCE = "maintenance"

@dataclass
class SensorReading:
    """Individual sensor reading"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    data: Dict[str, Any]
    quality: float  # 0.0 to 1.0
    uncertainty: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.uncertainty is None:
            self.uncertainty = {}

@dataclass
class IMUData:
    """Inertial Measurement Unit data"""
    acceleration: Tuple[float, float, float]  # m/s^2 (x, y, z)
    angular_velocity: Tuple[float, float, float]  # rad/s (roll, pitch, yaw)
    orientation: Optional[Tuple[float, float, float, float]] = None  # Quaternion (w, x, y, z)
    magnetic_field: Optional[Tuple[float, float, float]] = None  # Tesla (x, y, z)
    temperature: Optional[float] = None  # Celsius
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GPSData:
    """GPS positioning data"""
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters above sea level
    accuracy: float  # meters (horizontal accuracy)
    speed: Optional[float] = None  # m/s
    heading: Optional[float] = None  # degrees from north
    satellites: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class LidarPoint:
    """Single LiDAR point"""
    x: float
    y: float
    z: float
    intensity: Optional[float] = None
    ring: Optional[int] = None

@dataclass
class LidarData:
    """LiDAR point cloud data"""
    points: List[LidarPoint]
    frame_id: str
    timestamp: datetime
    scan_time: float  # seconds
    range_min: float  # meters
    range_max: float  # meters
    angle_min: float  # radians
    angle_max: float  # radians
    angle_increment: float  # radians

@dataclass
class FusedPose:
    """Fused robot pose estimate"""
    position: Tuple[float, float, float]  # x, y, z in meters
    orientation: Tuple[float, float, float, float]  # quaternion (w, x, y, z)
    linear_velocity: Tuple[float, float, float]  # m/s
    angular_velocity: Tuple[float, float, float]  # rad/s
    covariance: Optional[np.ndarray] = None  # 6x6 covariance matrix
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    contributing_sensors: List[str] = field(default_factory=list)

class KalmanFilter:
    """Extended Kalman Filter for sensor fusion"""
    
    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.logger = logging.getLogger(__name__)
        
        if NUMPY_AVAILABLE:
            # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
            # Position, velocity, orientation (quaternion), angular velocity
            self.state = np.zeros(state_dim)
            self.state[6] = 1.0  # Initialize quaternion w component
            
            # State covariance matrix
            self.P = np.eye(state_dim) * 0.1
            
            # Process noise covariance
            self.Q = np.eye(state_dim) * 0.01
            
            # Measurement noise covariance
            self.R = np.eye(measurement_dim) * 0.1
            
            self.initialized = True
        else:
            self.initialized = False
            self.logger.warning("NumPy not available, using mock Kalman filter")
    
    def predict(self, dt: float, control_input: Optional[np.ndarray] = None):
        """Prediction step of Kalman filter"""
        if not self.initialized:
            return
        
        try:
            # State transition model (constant velocity)
            F = self._get_state_transition_matrix(dt)
            
            # Predict state
            self.state = F @ self.state
            
            # Add control input if available
            if control_input is not None:
                B = self._get_control_matrix(dt)
                self.state += B @ control_input
            
            # Predict covariance
            self.P = F @ self.P @ F.T + self.Q
            
            # Normalize quaternion
            self._normalize_quaternion()
            
        except Exception as e:
            self.logger.error(f"Kalman filter prediction failed: {e}")
    
    def update(self, measurement: np.ndarray, measurement_function: callable):
        """Update step of Kalman filter"""
        if not self.initialized:
            return
        
        try:
            # Predicted measurement
            h = measurement_function(self.state)
            
            # Innovation (measurement residual)
            y = measurement - h
            
            # Measurement Jacobian
            H = self._numerical_jacobian(measurement_function, self.state)
            
            # Innovation covariance
            S = H @ self.P @ H.T + self.R
            
            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.state += K @ y
            
            # Update covariance
            I = np.eye(self.state_dim)
            self.P = (I - K @ H) @ self.P
            
            # Normalize quaternion
            self._normalize_quaternion()
            
        except Exception as e:
            self.logger.error(f"Kalman filter update failed: {e}")
    
    def _get_state_transition_matrix(self, dt: float) -> np.ndarray:
        """Get state transition matrix for constant velocity model"""
        F = np.eye(self.state_dim)
        
        # Position update from velocity
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        
        # Quaternion update from angular velocity (simplified)
        # In practice, this would be more complex
        F[6, 10] = -dt * 0.5  # qw += -wx * dt * 0.5
        F[7, 11] = -dt * 0.5  # qx += -wy * dt * 0.5
        F[8, 12] = -dt * 0.5  # qy += -wz * dt * 0.5
        
        return F
    
    def _get_control_matrix(self, dt: float) -> np.ndarray:
        """Get control input matrix"""
        B = np.zeros((self.state_dim, 6))  # Assuming 6 control inputs
        
        # Acceleration inputs affect velocity
        B[3, 0] = dt  # vx += ax * dt
        B[4, 1] = dt  # vy += ay * dt
        B[5, 2] = dt  # vz += az * dt
        
        # Angular acceleration inputs affect angular velocity
        B[10, 3] = dt  # wx += alpha_x * dt
        B[11, 4] = dt  # wy += alpha_y * dt
        B[12, 5] = dt  # wz += alpha_z * dt
        
        return B
    
    def _normalize_quaternion(self):
        """Normalize quaternion components in state vector"""
        if self.initialized:
            quat = self.state[6:10]
            norm = np.linalg.norm(quat)
            if norm > 0:
                self.state[6:10] = quat / norm
    
    def _numerical_jacobian(self, func: callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute numerical Jacobian of function"""
        f0 = func(x)
        jac = np.zeros((len(f0), len(x)))
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = func(x_plus)
            
            x_minus = x.copy()
            x_minus[i] -= eps
            f_minus = func(x_minus)
            
            jac[:, i] = (f_plus - f_minus) / (2 * eps)
        
        return jac
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        return self.state.copy() if self.initialized else np.zeros(self.state_dim)
    
    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix"""
        return self.P.copy() if self.initialized else np.eye(self.state_dim)

class SensorManager:
    """Manage multiple sensors and their data"""
    
    def __init__(self):
        self.sensors: Dict[str, Dict[str, Any]] = {}
        self.sensor_data: Dict[str, deque] = {}
        self.sensor_status: Dict[str, SensorStatus] = {}
        self.data_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.max_history_size = 1000
    
    def register_sensor(self, sensor_id: str, sensor_type: SensorType, 
                       config: Dict[str, Any] = None) -> bool:
        """Register a new sensor"""
        with self.data_lock:
            if config is None:
                config = {}
            
            self.sensors[sensor_id] = {
                'type': sensor_type,
                'config': config,
                'last_reading': None,
                'total_readings': 0,
                'error_count': 0
            }
            
            self.sensor_data[sensor_id] = deque(maxlen=self.max_history_size)
            self.sensor_status[sensor_id] = SensorStatus.INACTIVE
            
            self.logger.info(f"Registered sensor {sensor_id} of type {sensor_type.value}")
            return True
    
    def unregister_sensor(self, sensor_id: str) -> bool:
        """Unregister a sensor"""
        with self.data_lock:
            if sensor_id in self.sensors:
                del self.sensors[sensor_id]
                del self.sensor_data[sensor_id]
                del self.sensor_status[sensor_id]
                self.logger.info(f"Unregistered sensor {sensor_id}")
                return True
            return False
    
    def add_reading(self, reading: SensorReading) -> bool:
        """Add new sensor reading"""
        with self.data_lock:
            sensor_id = reading.sensor_id
            
            if sensor_id not in self.sensors:
                self.logger.warning(f"Reading from unregistered sensor {sensor_id}")
                return False
            
            # Validate reading
            if not self._validate_reading(reading):
                self.sensors[sensor_id]['error_count'] += 1
                self.sensor_status[sensor_id] = SensorStatus.ERROR
                return False
            
            # Add to history
            self.sensor_data[sensor_id].append(reading)
            
            # Update sensor info
            self.sensors[sensor_id]['last_reading'] = reading.timestamp
            self.sensors[sensor_id]['total_readings'] += 1
            self.sensor_status[sensor_id] = SensorStatus.ACTIVE
            
            return True
    
    def get_latest_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """Get latest reading from sensor"""
        with self.data_lock:
            if sensor_id in self.sensor_data and self.sensor_data[sensor_id]:
                return self.sensor_data[sensor_id][-1]
            return None
    
    def get_readings_in_window(self, sensor_id: str, start_time: datetime, 
                              end_time: datetime) -> List[SensorReading]:
        """Get readings within time window"""
        with self.data_lock:
            if sensor_id not in self.sensor_data:
                return []
            
            readings = []
            for reading in self.sensor_data[sensor_id]:
                if start_time <= reading.timestamp <= end_time:
                    readings.append(reading)
            
            return readings
    
    def get_sensor_status(self, sensor_id: str) -> Optional[SensorStatus]:
        """Get status of specific sensor"""
        return self.sensor_status.get(sensor_id)
    
    def get_all_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sensors"""
        with self.data_lock:
            status_info = {}
            
            for sensor_id, sensor_info in self.sensors.items():
                status_info[sensor_id] = {
                    'type': sensor_info['type'].value,
                    'status': self.sensor_status[sensor_id].value,
                    'last_reading': sensor_info['last_reading'].isoformat() if sensor_info['last_reading'] else None,
                    'total_readings': sensor_info['total_readings'],
                    'error_count': sensor_info['error_count'],
                    'data_queue_size': len(self.sensor_data[sensor_id])
                }
            
            return status_info
    
    def _validate_reading(self, reading: SensorReading) -> bool:
        """Validate sensor reading"""
        # Basic validation
        if not reading.data:
            return False
        
        if reading.quality < 0.0 or reading.quality > 1.0:
            return False
        
        # Sensor-specific validation
        if reading.sensor_type == SensorType.GPS:
            return self._validate_gps_reading(reading)
        elif reading.sensor_type == SensorType.IMU:
            return self._validate_imu_reading(reading)
        elif reading.sensor_type == SensorType.LIDAR:
            return self._validate_lidar_reading(reading)
        
        return True
    
    def _validate_gps_reading(self, reading: SensorReading) -> bool:
        """Validate GPS reading"""
        data = reading.data
        
        # Check required fields
        required_fields = ['latitude', 'longitude', 'altitude']
        if not all(field in data for field in required_fields):
            return False
        
        # Check value ranges
        if not (-90 <= data['latitude'] <= 90):
            return False
        if not (-180 <= data['longitude'] <= 180):
            return False
        
        return True
    
    def _validate_imu_reading(self, reading: SensorReading) -> bool:
        """Validate IMU reading"""
        data = reading.data
        
        # Check required fields
        required_fields = ['acceleration', 'angular_velocity']
        if not all(field in data for field in required_fields):
            return False
        
        # Check if acceleration and angular velocity are 3-element lists/tuples
        if len(data['acceleration']) != 3 or len(data['angular_velocity']) != 3:
            return False
        
        return True
    
    def _validate_lidar_reading(self, reading: SensorReading) -> bool:
        """Validate LiDAR reading"""
        data = reading.data
        
        # Check required fields
        if 'points' not in data:
            return False
        
        # Check if points is a list
        if not isinstance(data['points'], list):
            return False
        
        return True

class SensorFusionEngine:
    """Main sensor fusion engine"""
    
    def __init__(self):
        self.sensor_manager = SensorManager()
        self.kalman_filter = KalmanFilter(state_dim=13, measurement_dim=6)  # Adjustable
        self.fusion_rate = 50.0  # Hz
        self.logger = logging.getLogger(__name__)
        
        self.current_pose: Optional[FusedPose] = None
        self.fusion_history: deque = deque(maxlen=1000)
        
        self.running = False
        self.fusion_thread: Optional[threading.Thread] = None
        
        # Measurement functions for different sensors
        self.measurement_functions = {
            SensorType.GPS: self._gps_measurement_function,
            SensorType.IMU: self._imu_measurement_function,
            SensorType.ODOMETRY: self._odometry_measurement_function
        }
    
    def start_fusion(self) -> bool:
        """Start sensor fusion process"""
        if self.running:
            return False
        
        self.running = True
        self.fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self.fusion_thread.start()
        
        self.logger.info("Started sensor fusion engine")
        return True
    
    def stop_fusion(self) -> bool:
        """Stop sensor fusion process"""
        if not self.running:
            return False
        
        self.running = False
        if self.fusion_thread:
            self.fusion_thread.join(timeout=1.0)
        
        self.logger.info("Stopped sensor fusion engine")
        return True
    
    def _fusion_loop(self):
        """Main fusion loop"""
        dt = 1.0 / self.fusion_rate
        last_time = datetime.utcnow()
        
        while self.running:
            try:
                current_time = datetime.utcnow()
                actual_dt = (current_time - last_time).total_seconds()
                
                # Prediction step
                self.kalman_filter.predict(actual_dt)
                
                # Update with available sensor measurements
                self._process_sensor_updates(current_time)
                
                # Generate fused pose estimate
                self._update_fused_pose(current_time)
                
                last_time = current_time
                
                # Sleep to maintain fusion rate
                asyncio.run(asyncio.sleep(max(0, dt - actual_dt)))
                
            except Exception as e:
                self.logger.error(f"Fusion loop error: {e}")
                asyncio.run(asyncio.sleep(dt))
    
    def _process_sensor_updates(self, current_time: datetime):
        """Process updates from all sensors"""
        # Get recent readings from all sensors
        window_start = current_time - timedelta(seconds=0.1)  # 100ms window
        
        for sensor_id, sensor_info in self.sensor_manager.sensors.items():
            sensor_type = sensor_info['type']
            
            if sensor_type in self.measurement_functions:
                # Get recent readings
                readings = self.sensor_manager.get_readings_in_window(
                    sensor_id, window_start, current_time
                )
                
                # Process most recent reading
                if readings:
                    latest_reading = readings[-1]
                    measurement_func = self.measurement_functions[sensor_type]
                    
                    try:
                        measurement = measurement_func(latest_reading)
                        if measurement is not None:
                            self.kalman_filter.update(measurement, 
                                                    lambda state: measurement_func(latest_reading, predict=True))
                    except Exception as e:
                        self.logger.error(f"Sensor update failed for {sensor_id}: {e}")
    
    def _update_fused_pose(self, timestamp: datetime):
        """Update fused pose estimate"""
        if NUMPY_AVAILABLE:
            state = self.kalman_filter.get_state()
            covariance = self.kalman_filter.get_covariance()
            
            # Extract pose components from state
            position = tuple(state[0:3])
            velocity = tuple(state[3:6])
            orientation = tuple(state[6:10])  # quaternion
            angular_velocity = tuple(state[10:13])
            
            # Calculate confidence from covariance trace
            confidence = 1.0 / (1.0 + np.trace(covariance[:6, :6]) / 6.0)
            
            self.current_pose = FusedPose(
                position=position,
                orientation=orientation,
                linear_velocity=velocity,
                angular_velocity=angular_velocity,
                covariance=covariance,
                confidence=min(max(confidence, 0.0), 1.0),
                timestamp=timestamp,
                contributing_sensors=list(self.sensor_manager.sensors.keys())
            )
        else:
            # Mock fused pose
            self.current_pose = FusedPose(
                position=(0.0, 0.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                linear_velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                confidence=0.8,
                timestamp=timestamp,
                contributing_sensors=['mock_sensor']
            )
        
        # Add to history
        self.fusion_history.append(self.current_pose)
    
    def _gps_measurement_function(self, reading: SensorReading, predict: bool = False) -> Optional[np.ndarray]:
        """Convert GPS reading to measurement vector"""
        if not NUMPY_AVAILABLE:
            return None
        
        try:
            data = reading.data
            # Convert GPS to local coordinates (simplified)
            # In practice, you'd use proper coordinate transformations
            x = data['longitude'] * 111320.0  # Rough conversion to meters
            y = data['latitude'] * 110540.0
            z = data['altitude']
            
            # Measurement vector: [x, y, z, vx, vy, vz]
            measurement = np.array([x, y, z, 0.0, 0.0, 0.0])
            
            if 'speed' in data and 'heading' in data:
                speed = data['speed']
                heading_rad = math.radians(data['heading'])
                measurement[3] = speed * math.cos(heading_rad)  # vx
                measurement[4] = speed * math.sin(heading_rad)  # vy
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"GPS measurement function error: {e}")
            return None
    
    def _imu_measurement_function(self, reading: SensorReading, predict: bool = False) -> Optional[np.ndarray]:
        """Convert IMU reading to measurement vector"""
        if not NUMPY_AVAILABLE:
            return None
        
        try:
            data = reading.data
            
            # For IMU, we primarily get acceleration and angular velocity
            # These are used to update velocity and orientation indirectly
            accel = np.array(data['acceleration'])
            angular_vel = np.array(data['angular_velocity'])
            
            # Create measurement vector (acceleration and angular velocity)
            measurement = np.concatenate([accel, angular_vel])
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"IMU measurement function error: {e}")
            return None
    
    def _odometry_measurement_function(self, reading: SensorReading, predict: bool = False) -> Optional[np.ndarray]:
        """Convert odometry reading to measurement vector"""
        if not NUMPY_AVAILABLE:
            return None
        
        try:
            data = reading.data
            
            # Odometry typically provides position and velocity
            position = np.array(data.get('position', [0.0, 0.0, 0.0]))
            velocity = np.array(data.get('velocity', [0.0, 0.0, 0.0]))
            
            measurement = np.concatenate([position, velocity])
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"Odometry measurement function error: {e}")
            return None
    
    def get_current_pose(self) -> Optional[FusedPose]:
        """Get current fused pose estimate"""
        return self.current_pose
    
    def get_pose_history(self, duration: timedelta = None) -> List[FusedPose]:
        """Get pose history within specified duration"""
        if duration is None:
            return list(self.fusion_history)
        
        cutoff_time = datetime.utcnow() - duration
        return [pose for pose in self.fusion_history if pose.timestamp >= cutoff_time]
    
    def add_sensor_reading(self, reading: SensorReading) -> bool:
        """Add sensor reading for fusion"""
        return self.sensor_manager.add_reading(reading)
    
    def register_sensor(self, sensor_id: str, sensor_type: SensorType, 
                       config: Dict[str, Any] = None) -> bool:
        """Register sensor with fusion engine"""
        return self.sensor_manager.register_sensor(sensor_id, sensor_type, config)
    
    def get_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sensors"""
        return self.sensor_manager.get_all_sensor_status()
    
    async def process_sensor_data_batch(self, readings: List[SensorReading]) -> Dict[str, Any]:
        """Process batch of sensor readings"""
        results = {
            'processed_readings': 0,
            'failed_readings': 0,
            'sensor_updates': {},
            'fusion_status': 'active' if self.running else 'inactive'
        }
        
        for reading in readings:
            try:
                success = self.add_sensor_reading(reading)
                if success:
                    results['processed_readings'] += 1
                    
                    # Track per-sensor statistics
                    sensor_id = reading.sensor_id
                    if sensor_id not in results['sensor_updates']:
                        results['sensor_updates'][sensor_id] = {
                            'readings': 0,
                            'last_update': None,
                            'quality_avg': 0.0
                        }
                    
                    sensor_stats = results['sensor_updates'][sensor_id]
                    sensor_stats['readings'] += 1
                    sensor_stats['last_update'] = reading.timestamp.isoformat()
                    
                    # Update quality average
                    old_avg = sensor_stats['quality_avg']
                    new_count = sensor_stats['readings']
                    sensor_stats['quality_avg'] = (old_avg * (new_count - 1) + reading.quality) / new_count
                    
                else:
                    results['failed_readings'] += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process reading: {e}")
                results['failed_readings'] += 1
        
        # Add current pose information
        if self.current_pose:
            results['current_pose'] = {
                'position': self.current_pose.position,
                'orientation': self.current_pose.orientation,
                'confidence': self.current_pose.confidence,
                'timestamp': self.current_pose.timestamp.isoformat()
            }
        
        return results

# Global sensor fusion engine
sensor_fusion_engine = SensorFusionEngine()
