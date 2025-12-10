# hardware/controllers/sensor_interface.py
# Agent 34: Advanced Robotics Integration - Sensor Interface Controller

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import math
import random

# Import robotics components
from ...backend.robotics.sensor_fusion import SensorReading, SensorType, IMUData, GPSData, LidarData, LidarPoint

class SensorInterface:
    """Base sensor interface class"""
    
    def __init__(self, sensor_id: str, sensor_type: SensorType):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.logger = logging.getLogger(__name__)
        
        self.is_active = False
        self.sample_rate = 10.0  # Hz
        self.last_reading_time = datetime.utcnow()
        
        # Threading
        self.read_thread: Optional[threading.Thread] = None
        self.data_callback: Optional[Callable] = None
        
        # Sensor parameters
        self.calibration_data = {}
        self.noise_level = 0.01
        self.bias = {}
    
    async def start(self, data_callback: Callable = None) -> bool:
        """Start sensor data collection"""
        if self.is_active:
            return False
        
        self.data_callback = data_callback
        self.is_active = True
        
        # Start reading thread
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        
        self.logger.info(f"Started sensor {self.sensor_id} ({self.sensor_type.value})")
        return True
    
    async def stop(self) -> bool:
        """Stop sensor data collection"""
        if not self.is_active:
            return False
        
        self.is_active = False
        
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
        
        self.logger.info(f"Stopped sensor {self.sensor_id}")
        return True
    
    def _read_loop(self):
        """Main sensor reading loop"""
        loop_time = 1.0 / self.sample_rate
        
        while self.is_active:
            start_time = time.time()
            
            try:
                # Read sensor data
                reading = asyncio.run(self._read_sensor_data())
                
                if reading and self.data_callback:
                    asyncio.run(self.data_callback(reading))
                    
            except Exception as e:
                self.logger.error(f"Sensor read error: {e}")
            
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, loop_time - elapsed)
            time.sleep(sleep_time)
    
    async def _read_sensor_data(self) -> Optional[SensorReading]:
        """Read data from sensor - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _read_sensor_data")
    
    def set_sample_rate(self, rate: float):
        """Set sensor sample rate in Hz"""
        self.sample_rate = max(0.1, min(1000.0, rate))
        self.logger.info(f"Set sample rate for {self.sensor_id} to {self.sample_rate} Hz")
    
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate sensor with provided data"""
        self.calibration_data = calibration_data
        self.logger.info(f"Calibrated sensor {self.sensor_id}")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get sensor status"""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'is_active': self.is_active,
            'sample_rate': self.sample_rate,
            'last_reading': self.last_reading_time.isoformat(),
            'calibration_status': bool(self.calibration_data)
        }

class IMUSensor(SensorInterface):
    """Inertial Measurement Unit sensor"""
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SensorType.IMU)
        
        # IMU-specific parameters
        self.accelerometer_range = 16.0  # g
        self.gyroscope_range = 2000.0  # deg/s
        self.magnetometer_range = 4800.0  # uT
        
        # Simulation state
        self.sim_orientation = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self.sim_angular_velocity = [0.0, 0.0, 0.0]
        self.sim_acceleration = [0.0, 0.0, 9.81]  # Include gravity
    
    async def _read_sensor_data(self) -> Optional[SensorReading]:
        """Read IMU data"""
        try:
            # Simulate IMU data with some noise
            accel_noise = [random.gauss(0, 0.1) for _ in range(3)]
            gyro_noise = [random.gauss(0, 0.01) for _ in range(3)]
            
            # Add some motion simulation
            current_time = time.time()
            motion_freq = 0.5  # Hz
            
            # Simulate some motion
            motion_amplitude = 0.1
            self.sim_angular_velocity[2] = motion_amplitude * math.sin(current_time * motion_freq * 2 * math.pi)
            
            # Update orientation based on angular velocity
            dt = 1.0 / self.sample_rate
            for i in range(3):
                self.sim_orientation[i] += self.sim_angular_velocity[i] * dt
            
            # Create IMU data
            imu_data = IMUData(
                acceleration=(
                    self.sim_acceleration[0] + accel_noise[0],
                    self.sim_acceleration[1] + accel_noise[1],
                    self.sim_acceleration[2] + accel_noise[2]
                ),
                angular_velocity=(
                    self.sim_angular_velocity[0] + gyro_noise[0],
                    self.sim_angular_velocity[1] + gyro_noise[1],
                    self.sim_angular_velocity[2] + gyro_noise[2]
                ),
                orientation=self._euler_to_quaternion(*self.sim_orientation),
                magnetic_field=(10.0, 20.0, 45.0),  # Mock magnetic field
                temperature=25.0 + random.gauss(0, 2.0)
            )
            
            # Create sensor reading
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                data=asdict(imu_data),
                quality=0.9 + random.uniform(-0.1, 0.1)
            )
            
            self.last_reading_time = reading.timestamp
            return reading
            
        except Exception as e:
            self.logger.error(f"IMU read failed: {e}")
            return None
    
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion (w, x, y, z)"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        
        return (w, x, y, z)

class GPSSensor(SensorInterface):
    """Global Positioning System sensor"""
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SensorType.GPS)
        
        # GPS-specific parameters
        self.accuracy = 3.0  # meters
        self.fix_type = 3  # 3D fix
        
        # Simulation parameters
        self.base_latitude = 37.7749  # San Francisco
        self.base_longitude = -122.4194
        self.base_altitude = 50.0  # meters
        
        # Motion simulation
        self.sim_position = [self.base_latitude, self.base_longitude, self.base_altitude]
        self.sim_velocity = 0.0  # m/s
        self.sim_heading = 0.0  # degrees
    
    async def _read_sensor_data(self) -> Optional[SensorReading]:
        """Read GPS data"""
        try:
            # Simulate GPS noise and drift
            lat_noise = random.gauss(0, self.accuracy / 111000.0)  # Convert meters to degrees
            lon_noise = random.gauss(0, self.accuracy / (111000.0 * math.cos(math.radians(self.sim_position[0]))))
            alt_noise = random.gauss(0, self.accuracy)
            
            # Simulate slow movement
            current_time = time.time()
            movement_speed = 0.5  # m/s
            movement_freq = 0.1  # Hz
            
            # Update position with slow circular motion
            radius = 10.0  # meters
            angle = current_time * movement_freq * 2 * math.pi
            
            lat_offset = (radius * math.cos(angle)) / 111000.0  # Convert to degrees
            lon_offset = (radius * math.sin(angle)) / (111000.0 * math.cos(math.radians(self.base_latitude)))
            
            current_lat = self.base_latitude + lat_offset + lat_noise
            current_lon = self.base_longitude + lon_offset + lon_noise
            current_alt = self.base_altitude + alt_noise
            
            # Calculate speed and heading
            self.sim_velocity = movement_speed + random.gauss(0, 0.1)
            self.sim_heading = math.degrees(angle) + random.gauss(0, 5.0)
            
            # Create GPS data
            gps_data = GPSData(
                latitude=current_lat,
                longitude=current_lon,
                altitude=current_alt,
                accuracy=self.accuracy + random.uniform(-1.0, 1.0),
                speed=self.sim_velocity,
                heading=self.sim_heading,
                satellites=8 + random.randint(-2, 2)
            )
            
            # Create sensor reading
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                data=asdict(gps_data),
                quality=0.8 + random.uniform(-0.2, 0.2)
            )
            
            self.last_reading_time = reading.timestamp
            return reading
            
        except Exception as e:
            self.logger.error(f"GPS read failed: {e}")
            return None

class LidarSensor(SensorInterface):
    """Light Detection and Ranging sensor"""
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SensorType.LIDAR)
        
        # LiDAR-specific parameters
        self.range_min = 0.1  # meters
        self.range_max = 100.0  # meters
        self.angle_min = -math.pi  # radians
        self.angle_max = math.pi  # radians
        self.angle_increment = math.pi / 180.0  # 1 degree
        self.scan_time = 0.1  # seconds
        
        # Reduce sample rate for LiDAR
        self.sample_rate = 2.0  # Hz
    
    async def _read_sensor_data(self) -> Optional[SensorReading]:
        """Read LiDAR data"""
        try:
            points = []
            
            # Generate simulated LiDAR points
            num_points = int((self.angle_max - self.angle_min) / self.angle_increment)
            
            for i in range(min(num_points, 100)):  # Limit points for performance
                angle = self.angle_min + i * self.angle_increment
                
                # Simulate distance measurement with obstacles
                distance = self._simulate_range_measurement(angle)
                
                if self.range_min <= distance <= self.range_max:
                    # Convert polar to Cartesian coordinates
                    x = distance * math.cos(angle)
                    y = distance * math.sin(angle)
                    z = 0.0  # 2D LiDAR
                    
                    intensity = 100.0 + random.uniform(-20.0, 20.0)
                    
                    points.append(LidarPoint(
                        x=x,
                        y=y,
                        z=z,
                        intensity=intensity
                    ))
            
            # Create LiDAR data
            lidar_data = LidarData(
                points=points,
                frame_id="laser",
                timestamp=datetime.utcnow(),
                scan_time=self.scan_time,
                range_min=self.range_min,
                range_max=self.range_max,
                angle_min=self.angle_min,
                angle_max=self.angle_max,
                angle_increment=self.angle_increment
            )
            
            # Create sensor reading
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                data={
                    'points': [asdict(point) for point in lidar_data.points],
                    'frame_id': lidar_data.frame_id,
                    'scan_time': lidar_data.scan_time,
                    'range_min': lidar_data.range_min,
                    'range_max': lidar_data.range_max,
                    'angle_min': lidar_data.angle_min,
                    'angle_max': lidar_data.angle_max,
                    'angle_increment': lidar_data.angle_increment
                },
                quality=0.9 + random.uniform(-0.1, 0.1)
            )
            
            self.last_reading_time = reading.timestamp
            return reading
            
        except Exception as e:
            self.logger.error(f"LiDAR read failed: {e}")
            return None
    
    def _simulate_range_measurement(self, angle: float) -> float:
        """Simulate range measurement with virtual obstacles"""
        # Default range
        base_range = 10.0 + 5.0 * math.sin(angle * 2)
        
        # Add some virtual obstacles
        obstacles = [
            {'angle': 0.0, 'distance': 5.0, 'width': 0.5},
            {'angle': math.pi/4, 'distance': 8.0, 'width': 0.3},
            {'angle': -math.pi/3, 'distance': 6.0, 'width': 0.4}
        ]
        
        min_distance = base_range
        
        for obstacle in obstacles:
            angle_diff = abs(angle - obstacle['angle'])
            if angle_diff < obstacle['width']:
                # Hit obstacle
                obstacle_distance = obstacle['distance'] + random.gauss(0, 0.1)
                min_distance = min(min_distance, obstacle_distance)
        
        # Add noise
        min_distance += random.gauss(0, 0.05)
        
        return max(self.range_min, min(self.range_max, min_distance))

class UltrasonicSensor(SensorInterface):
    """Ultrasonic distance sensor"""
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SensorType.ULTRASONIC)
        
        # Ultrasonic-specific parameters
        self.range_min = 0.02  # meters (2 cm)
        self.range_max = 4.0   # meters (400 cm)
        self.beam_angle = math.radians(15)  # 15 degrees
        
        # Higher sample rate for ultrasonic
        self.sample_rate = 20.0  # Hz
    
    async def _read_sensor_data(self) -> Optional[SensorReading]:
        """Read ultrasonic distance data"""
        try:
            # Simulate distance measurement
            base_distance = 1.5  # meters
            
            # Add some variation
            current_time = time.time()
            variation = 0.3 * math.sin(current_time * 2.0)
            
            distance = base_distance + variation + random.gauss(0, 0.02)
            distance = max(self.range_min, min(self.range_max, distance))
            
            # Create sensor reading
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                data={
                    'distance': distance,
                    'range_min': self.range_min,
                    'range_max': self.range_max,
                    'beam_angle': self.beam_angle
                },
                quality=0.8 if self.range_min < distance < self.range_max else 0.3
            )
            
            self.last_reading_time = reading.timestamp
            return reading
            
        except Exception as e:
            self.logger.error(f"Ultrasonic read failed: {e}")
            return None

class CameraSensor(SensorInterface):
    """Camera sensor for vision data"""
    
    def __init__(self, sensor_id: str):
        super().__init__(sensor_id, SensorType.CAMERA)
        
        # Camera-specific parameters
        self.resolution = (640, 480)
        self.fps = 30
        self.compression = "jpeg"
        
        # Lower sample rate to avoid overwhelming the system
        self.sample_rate = 5.0  # Hz
    
    async def _read_sensor_data(self) -> Optional[SensorReading]:
        """Read camera image data"""
        try:
            # Simulate camera data (in real implementation, would capture from camera)
            image_data = {
                'width': self.resolution[0],
                'height': self.resolution[1],
                'format': self.compression,
                'timestamp': datetime.utcnow().isoformat(),
                'frame_id': f"frame_{int(time.time() * 1000)}",
                'exposure_time': 1.0/60.0,  # seconds
                'gain': 1.0
            }
            
            # In real implementation, would include base64 encoded image data
            # image_data['image_data'] = base64_encoded_image
            
            # Create sensor reading
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                data=image_data,
                quality=0.9 + random.uniform(-0.1, 0.1)
            )
            
            self.last_reading_time = reading.timestamp
            return reading
            
        except Exception as e:
            self.logger.error(f"Camera read failed: {e}")
            return None

class SensorManager:
    """Manage multiple sensor interfaces"""
    
    def __init__(self):
        self.sensors: Dict[str, SensorInterface] = {}
        self.logger = logging.getLogger(__name__)
        self.data_callbacks: List[Callable] = []
        
        # Statistics
        self.total_readings = 0
        self.readings_per_sensor: Dict[str, int] = {}
    
    def add_sensor(self, sensor: SensorInterface) -> bool:
        """Add sensor to manager"""
        if sensor.sensor_id in self.sensors:
            self.logger.warning(f"Sensor {sensor.sensor_id} already exists")
            return False
        
        self.sensors[sensor.sensor_id] = sensor
        self.readings_per_sensor[sensor.sensor_id] = 0
        
        self.logger.info(f"Added sensor {sensor.sensor_id} ({sensor.sensor_type.value})")
        return True
    
    def remove_sensor(self, sensor_id: str) -> bool:
        """Remove sensor from manager"""
        if sensor_id not in self.sensors:
            return False
        
        sensor = self.sensors[sensor_id]
        if sensor.is_active:
            asyncio.run(sensor.stop())
        
        del self.sensors[sensor_id]
        del self.readings_per_sensor[sensor_id]
        
        self.logger.info(f"Removed sensor {sensor_id}")
        return True
    
    def add_data_callback(self, callback: Callable):
        """Add callback for sensor data"""
        self.data_callbacks.append(callback)
    
    async def start_all_sensors(self) -> bool:
        """Start all sensors"""
        success_count = 0
        
        for sensor in self.sensors.values():
            try:
                success = await sensor.start(self._on_sensor_data)
                if success:
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to start sensor {sensor.sensor_id}: {e}")
        
        self.logger.info(f"Started {success_count}/{len(self.sensors)} sensors")
        return success_count == len(self.sensors)
    
    async def stop_all_sensors(self) -> bool:
        """Stop all sensors"""
        success_count = 0
        
        for sensor in self.sensors.values():
            try:
                success = await sensor.stop()
                if success:
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to stop sensor {sensor.sensor_id}: {e}")
        
        self.logger.info(f"Stopped {success_count}/{len(self.sensors)} sensors")
        return success_count == len(self.sensors)
    
    async def _on_sensor_data(self, reading: SensorReading):
        """Handle sensor data reading"""
        # Update statistics
        self.total_readings += 1
        self.readings_per_sensor[reading.sensor_id] += 1
        
        # Call all registered callbacks
        for callback in self.data_callbacks:
            try:
                await callback(reading)
            except Exception as e:
                self.logger.error(f"Data callback failed: {e}")
    
    def get_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sensors"""
        status = {}
        
        for sensor_id, sensor in self.sensors.items():
            status[sensor_id] = sensor.get_status()
            status[sensor_id]['total_readings'] = self.readings_per_sensor[sensor_id]
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor manager statistics"""
        active_sensors = sum(1 for s in self.sensors.values() if s.is_active)
        
        return {
            'total_sensors': len(self.sensors),
            'active_sensors': active_sensors,
            'total_readings': self.total_readings,
            'readings_per_sensor': self.readings_per_sensor.copy(),
            'sensor_types': {s.sensor_type.value: s.sensor_id for s in self.sensors.values()}
        }

# Create default sensor configuration
def create_default_sensor_suite() -> SensorManager:
    """Create a default suite of sensors for robotics"""
    manager = SensorManager()
    
    # Add common sensors
    imu = IMUSensor("imu_main")
    imu.set_sample_rate(50.0)  # 50 Hz
    manager.add_sensor(imu)
    
    gps = GPSSensor("gps_main")
    gps.set_sample_rate(5.0)  # 5 Hz
    manager.add_sensor(gps)
    
    lidar = LidarSensor("lidar_main")
    lidar.set_sample_rate(2.0)  # 2 Hz
    manager.add_sensor(lidar)
    
    # Add ultrasonic sensors (multiple for 360-degree coverage)
    for i, angle in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
        ultrasonic = UltrasonicSensor(f"ultrasonic_{i}")
        ultrasonic.set_sample_rate(10.0)  # 10 Hz
        manager.add_sensor(ultrasonic)
    
    # Add camera
    camera = CameraSensor("camera_main")
    camera.set_sample_rate(5.0)  # 5 Hz
    manager.add_sensor(camera)
    
    return manager

# Global sensor manager instance
sensor_manager = create_default_sensor_suite()
