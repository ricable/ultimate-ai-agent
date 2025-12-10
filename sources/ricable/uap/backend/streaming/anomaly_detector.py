# backend/streaming/anomaly_detector.py
"""
Real-Time Anomaly Detection
Advanced anomaly detection algorithms for streaming data with sub-millisecond detection.
"""

import asyncio
import time
import json
import logging
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import math
from abc import ABC, abstractmethod

# Import stream processing components
from .stream_processor import StreamEvent, EventType, ProcessingPriority
from ..monitoring.logs.logger import get_logger
from ..monitoring.metrics.prometheus_metrics import metrics_collector

# Optional ML libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = get_logger(__name__)

class AnomalyType(Enum):
    """Types of anomalies"""
    STATISTICAL = "statistical"  # Statistical outliers
    PATTERN = "pattern"  # Pattern-based anomalies
    TEMPORAL = "temporal"  # Time-based anomalies
    VOLUME = "volume"  # Volume-based anomalies
    BEHAVIORAL = "behavioral"  # Behavioral anomalies
    THRESHOLD = "threshold"  # Threshold-based anomalies
    CORRELATION = "correlation"  # Correlation anomalies

class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionMethod(Enum):
    """Anomaly detection methods"""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    CUSTOM = "custom"

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    detector_id: str
    name: str
    method: DetectionMethod
    sensitivity: float = 0.95  # Detection sensitivity (0.0 to 1.0)
    window_size: int = 100  # Number of data points for analysis
    update_interval_ms: int = 1000  # Model update interval
    threshold_config: Dict[str, Any] = None
    feature_columns: List[str] = None
    enable_online_learning: bool = True
    alert_cooldown_ms: int = 5000  # Minimum time between alerts
    
    def __post_init__(self):
        if self.threshold_config is None:
            self.threshold_config = {}
        if self.feature_columns is None:
            self.feature_columns = []

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    event_id: str
    detector_id: str
    is_anomaly: bool
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float  # 0.0 to 1.0
    anomaly_score: float
    threshold: float
    explanation: str
    detected_at: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'detector_id': self.detector_id,
            'is_anomaly': self.is_anomaly,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'anomaly_score': self.anomaly_score,
            'threshold': self.threshold,
            'explanation': self.explanation,
            'detected_at': self.detected_at,
            'metadata': self.metadata
        }

class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.detector_id = config.detector_id
        self.is_trained = False
        self.last_update = 0.0
        self.data_buffer = deque(maxlen=config.window_size)
        self.metrics = {
            'total_detections': 0,
            'anomalies_detected': 0,
            'false_positives': 0,
            'last_detection_time': 0.0,
            'avg_detection_time_ms': 0.0
        }
    
    @abstractmethod
    async def detect(self, data_point: Any) -> AnomalyResult:
        """Detect anomalies in a single data point"""
        pass
    
    @abstractmethod
    async def update_model(self, data_points: List[Any]) -> None:
        """Update the anomaly detection model"""
        pass
    
    def add_data_point(self, data_point: Any) -> None:
        """Add data point to buffer"""
        self.data_buffer.append(data_point)
    
    def get_buffer_data(self) -> List[Any]:
        """Get data from buffer"""
        return list(self.data_buffer)
    
    def should_update_model(self) -> bool:
        """Check if model should be updated"""
        current_time = time.time() * 1000
        return (current_time - self.last_update) >= self.config.update_interval_ms
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detector metrics"""
        return {
            'detector_id': self.detector_id,
            'method': self.config.method.value,
            'is_trained': self.is_trained,
            'buffer_size': len(self.data_buffer),
            'last_update': self.last_update,
            **self.metrics
        }

class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detector using Z-score and IQR methods"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        super().__init__(config)
        self.mean = 0.0
        self.std = 0.0
        self.q1 = 0.0
        self.q3 = 0.0
        self.iqr = 0.0
        self.z_threshold = 3.0  # Standard Z-score threshold
        self.iqr_multiplier = 1.5  # IQR outlier multiplier
    
    async def detect(self, data_point: float) -> AnomalyResult:
        """Detect statistical anomalies"""
        start_time = time.perf_counter()
        
        # Add to buffer
        self.add_data_point(data_point)
        
        # Check if model needs updating
        if self.should_update_model() or not self.is_trained:
            await self.update_model(self.get_buffer_data())
        
        # Perform detection
        is_anomaly = False
        anomaly_score = 0.0
        threshold = 0.0
        explanation = "Normal data point"
        
        if self.is_trained:
            if self.config.method == DetectionMethod.Z_SCORE:
                z_score = abs((data_point - self.mean) / self.std) if self.std > 0 else 0
                anomaly_score = z_score
                threshold = self.z_threshold
                is_anomaly = z_score > self.z_threshold
                explanation = f"Z-score: {z_score:.3f}, threshold: {threshold}"
                
            elif self.config.method == DetectionMethod.IQR:
                lower_bound = self.q1 - self.iqr_multiplier * self.iqr
                upper_bound = self.q3 + self.iqr_multiplier * self.iqr
                is_anomaly = data_point < lower_bound or data_point > upper_bound
                anomaly_score = max(
                    abs(data_point - lower_bound) / self.iqr if data_point < lower_bound else 0,
                    abs(data_point - upper_bound) / self.iqr if data_point > upper_bound else 0
                )
                threshold = self.iqr_multiplier
                explanation = f"IQR bounds: [{lower_bound:.3f}, {upper_bound:.3f}], value: {data_point:.3f}"
        
        # Determine severity
        severity = self._calculate_severity(anomaly_score, threshold)
        
        # Calculate confidence
        confidence = min(anomaly_score / threshold, 1.0) if threshold > 0 else 0.0
        
        # Update metrics
        detection_time_ms = (time.perf_counter() - start_time) * 1000
        self.metrics['total_detections'] += 1
        if is_anomaly:
            self.metrics['anomalies_detected'] += 1
            self.metrics['last_detection_time'] = time.time()
        
        self.metrics['avg_detection_time_ms'] = (
            (self.metrics['avg_detection_time_ms'] * (self.metrics['total_detections'] - 1) + detection_time_ms) /
            self.metrics['total_detections']
        )
        
        return AnomalyResult(
            event_id=str(uuid.uuid4()),
            detector_id=self.detector_id,
            is_anomaly=is_anomaly,
            anomaly_type=AnomalyType.STATISTICAL,
            severity=severity,
            confidence=confidence,
            anomaly_score=anomaly_score,
            threshold=threshold,
            explanation=explanation,
            detected_at=time.time(),
            metadata={
                'method': self.config.method.value,
                'mean': self.mean,
                'std': self.std,
                'data_point': data_point
            }
        )
    
    async def update_model(self, data_points: List[float]) -> None:
        """Update statistical model"""
        if len(data_points) < 10:  # Need minimum data points
            return
        
        try:
            # Calculate statistics
            self.mean = statistics.mean(data_points)
            self.std = statistics.stdev(data_points) if len(data_points) > 1 else 0.0
            
            # Calculate quartiles for IQR
            sorted_data = sorted(data_points)
            n = len(sorted_data)
            self.q1 = sorted_data[n // 4] if n >= 4 else sorted_data[0]
            self.q3 = sorted_data[3 * n // 4] if n >= 4 else sorted_data[-1]
            self.iqr = self.q3 - self.q1
            
            self.is_trained = True
            self.last_update = time.time() * 1000
            
            logger.debug(f"Updated statistical model: mean={self.mean:.3f}, std={self.std:.3f}, IQR={self.iqr:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating statistical model: {e}")
    
    def _calculate_severity(self, anomaly_score: float, threshold: float) -> AnomalySeverity:
        """Calculate anomaly severity based on score"""
        if not threshold or anomaly_score <= threshold:
            return AnomalySeverity.LOW
        
        ratio = anomaly_score / threshold
        if ratio > 3.0:
            return AnomalySeverity.CRITICAL
        elif ratio > 2.0:
            return AnomalySeverity.HIGH
        elif ratio > 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

class MovingAverageAnomalyDetector(AnomalyDetector):
    """Moving average based anomaly detector"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        super().__init__(config)
        self.moving_average = 0.0
        self.moving_std = 0.0
        self.alpha = 0.1  # Exponential smoothing factor
        self.deviation_threshold = 2.0  # Standard deviations for anomaly
    
    async def detect(self, data_point: float) -> AnomalyResult:
        """Detect anomalies using moving average"""
        start_time = time.perf_counter()
        
        # Add to buffer
        self.add_data_point(data_point)
        
        # Update moving statistics
        if not self.is_trained:
            self.moving_average = data_point
            self.moving_std = 0.0
            self.is_trained = True
        else:
            # Exponential moving average
            self.moving_average = self.alpha * data_point + (1 - self.alpha) * self.moving_average
            
            # Update standard deviation estimate
            deviation = abs(data_point - self.moving_average)
            self.moving_std = self.alpha * deviation + (1 - self.alpha) * self.moving_std
        
        # Detect anomaly
        threshold = self.deviation_threshold * self.moving_std
        deviation = abs(data_point - self.moving_average)
        is_anomaly = deviation > threshold if threshold > 0 else False
        anomaly_score = deviation / self.moving_std if self.moving_std > 0 else 0
        
        # Calculate severity and confidence
        severity = self._calculate_severity(anomaly_score, self.deviation_threshold)
        confidence = min(anomaly_score / self.deviation_threshold, 1.0) if self.deviation_threshold > 0 else 0.0
        
        explanation = f"Moving avg: {self.moving_average:.3f}, deviation: {deviation:.3f}, threshold: {threshold:.3f}"
        
        # Update metrics
        detection_time_ms = (time.perf_counter() - start_time) * 1000
        self.metrics['total_detections'] += 1
        if is_anomaly:
            self.metrics['anomalies_detected'] += 1
        
        return AnomalyResult(
            event_id=str(uuid.uuid4()),
            detector_id=self.detector_id,
            is_anomaly=is_anomaly,
            anomaly_type=AnomalyType.TEMPORAL,
            severity=severity,
            confidence=confidence,
            anomaly_score=anomaly_score,
            threshold=self.deviation_threshold,
            explanation=explanation,
            detected_at=time.time(),
            metadata={
                'moving_average': self.moving_average,
                'moving_std': self.moving_std,
                'data_point': data_point
            }
        )
    
    async def update_model(self, data_points: List[float]) -> None:
        """Update moving average model"""
        # Moving average updates in real-time, no batch update needed
        self.last_update = time.time() * 1000
    
    def _calculate_severity(self, anomaly_score: float, threshold: float) -> AnomalySeverity:
        """Calculate anomaly severity"""
        if anomaly_score <= threshold:
            return AnomalySeverity.LOW
        
        ratio = anomaly_score / threshold
        if ratio > 4.0:
            return AnomalySeverity.CRITICAL
        elif ratio > 3.0:
            return AnomalySeverity.HIGH
        elif ratio > 2.0:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

class ThresholdAnomalyDetector(AnomalyDetector):
    """Simple threshold-based anomaly detector"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        super().__init__(config)
        self.upper_threshold = config.threshold_config.get('upper', float('inf'))
        self.lower_threshold = config.threshold_config.get('lower', float('-inf'))
        self.is_trained = True  # No training needed for threshold detection
    
    async def detect(self, data_point: float) -> AnomalyResult:
        """Detect threshold-based anomalies"""
        start_time = time.perf_counter()
        
        # Check thresholds
        exceeds_upper = data_point > self.upper_threshold
        exceeds_lower = data_point < self.lower_threshold
        is_anomaly = exceeds_upper or exceeds_lower
        
        # Calculate anomaly score
        if exceeds_upper:
            anomaly_score = (data_point - self.upper_threshold) / self.upper_threshold if self.upper_threshold != 0 else 1.0
            explanation = f"Value {data_point:.3f} exceeds upper threshold {self.upper_threshold:.3f}"
        elif exceeds_lower:
            anomaly_score = (self.lower_threshold - data_point) / abs(self.lower_threshold) if self.lower_threshold != 0 else 1.0
            explanation = f"Value {data_point:.3f} below lower threshold {self.lower_threshold:.3f}"
        else:
            anomaly_score = 0.0
            explanation = f"Value {data_point:.3f} within thresholds [{self.lower_threshold:.3f}, {self.upper_threshold:.3f}]"
        
        # Determine severity based on how much threshold is exceeded
        severity = AnomalySeverity.LOW
        if is_anomaly:
            if anomaly_score > 0.5:
                severity = AnomalySeverity.CRITICAL
            elif anomaly_score > 0.3:
                severity = AnomalySeverity.HIGH
            elif anomaly_score > 0.1:
                severity = AnomalySeverity.MEDIUM
        
        confidence = min(anomaly_score, 1.0) if is_anomaly else 1.0
        
        # Update metrics
        detection_time_ms = (time.perf_counter() - start_time) * 1000
        self.metrics['total_detections'] += 1
        if is_anomaly:
            self.metrics['anomalies_detected'] += 1
        
        return AnomalyResult(
            event_id=str(uuid.uuid4()),
            detector_id=self.detector_id,
            is_anomaly=is_anomaly,
            anomaly_type=AnomalyType.THRESHOLD,
            severity=severity,
            confidence=confidence,
            anomaly_score=anomaly_score,
            threshold=max(self.upper_threshold, abs(self.lower_threshold)),
            explanation=explanation,
            detected_at=time.time(),
            metadata={
                'upper_threshold': self.upper_threshold,
                'lower_threshold': self.lower_threshold,
                'data_point': data_point
            }
        )
    
    async def update_model(self, data_points: List[float]) -> None:
        """Update thresholds (no training needed)"""
        self.last_update = time.time() * 1000

class MLAnomalyDetector(AnomalyDetector):
    """Machine learning based anomaly detector"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.contamination = 1.0 - config.sensitivity  # Convert sensitivity to contamination
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, ML anomaly detection will use fallback")
    
    async def detect(self, data_point: Union[float, List[float]]) -> AnomalyResult:
        """Detect anomalies using ML model"""
        start_time = time.perf_counter()
        
        # Convert single value to array
        if isinstance(data_point, (int, float)):
            data_array = np.array([[data_point]])
        else:
            data_array = np.array([data_point])
        
        self.add_data_point(data_point)
        
        # Check if model needs updating
        if self.should_update_model() or not self.is_trained:
            await self.update_model(self.get_buffer_data())
        
        is_anomaly = False
        anomaly_score = 0.0
        explanation = "ML model not available"
        
        if self.is_trained and self.model and SKLEARN_AVAILABLE:
            try:
                # Scale data
                scaled_data = self.scaler.transform(data_array)
                
                # Predict anomaly
                prediction = self.model.predict(scaled_data)[0]
                is_anomaly = prediction == -1  # -1 indicates anomaly in sklearn
                
                # Get anomaly score
                if hasattr(self.model, 'decision_function'):
                    anomaly_score = abs(self.model.decision_function(scaled_data)[0])
                elif hasattr(self.model, 'score_samples'):
                    anomaly_score = -self.model.score_samples(scaled_data)[0]  # Negative for anomaly
                else:
                    anomaly_score = 1.0 if is_anomaly else 0.0
                
                explanation = f"ML prediction: {'Anomaly' if is_anomaly else 'Normal'}, score: {anomaly_score:.3f}"
                
            except Exception as e:
                logger.error(f"ML detection error: {e}")
                explanation = f"ML detection error: {str(e)}"
        
        # Calculate severity and confidence
        severity = self._calculate_severity(anomaly_score)
        confidence = min(anomaly_score, 1.0) if is_anomaly else 1.0 - anomaly_score
        
        # Update metrics
        detection_time_ms = (time.perf_counter() - start_time) * 1000
        self.metrics['total_detections'] += 1
        if is_anomaly:
            self.metrics['anomalies_detected'] += 1
        
        return AnomalyResult(
            event_id=str(uuid.uuid4()),
            detector_id=self.detector_id,
            is_anomaly=is_anomaly,
            anomaly_type=AnomalyType.PATTERN,
            severity=severity,
            confidence=confidence,
            anomaly_score=anomaly_score,
            threshold=0.5,  # ML models typically use 0.5 as threshold
            explanation=explanation,
            detected_at=time.time(),
            metadata={
                'method': self.config.method.value,
                'contamination': self.contamination,
                'model_type': type(self.model).__name__ if self.model else None
            }
        )
    
    async def update_model(self, data_points: List[Any]) -> None:
        """Update ML model"""
        if not SKLEARN_AVAILABLE or len(data_points) < 20:
            return
        
        try:
            # Prepare data
            if isinstance(data_points[0], (list, np.ndarray)):
                X = np.array(data_points)
            else:
                X = np.array(data_points).reshape(-1, 1)
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Create and train model based on method
            if self.config.method == DetectionMethod.ISOLATION_FOREST:
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
            elif self.config.method == DetectionMethod.ONE_CLASS_SVM:
                self.model = OneClassSVM(
                    nu=self.contamination,
                    kernel='rbf',
                    gamma='scale'
                )
            else:
                # Default to Isolation Forest
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42
                )
            
            # Train model
            self.model.fit(X_scaled)
            self.is_trained = True
            self.last_update = time.time() * 1000
            
            logger.debug(f"Updated ML model: {type(self.model).__name__} with {len(data_points)} samples")
            
        except Exception as e:
            logger.error(f"Error updating ML model: {e}")
    
    def _calculate_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Calculate severity based on anomaly score"""
        if anomaly_score > 0.8:
            return AnomalySeverity.CRITICAL
        elif anomaly_score > 0.6:
            return AnomalySeverity.HIGH
        elif anomaly_score > 0.4:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

class AnomalyDetectionManager:
    """Manager for multiple anomaly detectors"""
    
    def __init__(self):
        self.detectors: Dict[str, AnomalyDetector] = {}
        self.alert_handlers: List[Callable] = []
        self.last_alerts: Dict[str, float] = {}  # Track last alert times for cooldown
        self.is_running = False
        self.processing_tasks = []
    
    def create_detector(self, config: AnomalyDetectionConfig) -> AnomalyDetector:
        """Create and register an anomaly detector"""
        if config.method in [DetectionMethod.Z_SCORE, DetectionMethod.IQR]:
            detector = StatisticalAnomalyDetector(config)
        elif config.method == DetectionMethod.MOVING_AVERAGE:
            detector = MovingAverageAnomalyDetector(config)
        elif config.method == DetectionMethod.THRESHOLD:
            detector = ThresholdAnomalyDetector(config)
        elif config.method in [DetectionMethod.ISOLATION_FOREST, DetectionMethod.ONE_CLASS_SVM]:
            detector = MLAnomalyDetector(config)
        else:
            raise ValueError(f"Unsupported detection method: {config.method}")
        
        self.detectors[config.detector_id] = detector
        logger.info(f"Created anomaly detector: {config.detector_id} ({config.method.value})")
        
        return detector
    
    async def detect_anomalies(self, detector_id: str, data_point: Any) -> Optional[AnomalyResult]:
        """Detect anomalies using specified detector"""
        detector = self.detectors.get(detector_id)
        if not detector:
            logger.warning(f"Detector {detector_id} not found")
            return None
        
        try:
            result = await detector.detect(data_point)
            
            # Handle alerts
            if result.is_anomaly:
                await self._handle_anomaly_alert(result)
            
            # Update Prometheus metrics
            metrics_collector.anomaly_detections.labels(
                detector_id=detector_id,
                anomaly_type=result.anomaly_type.value
            ).inc()
            
            if result.is_anomaly:
                metrics_collector.anomalies_detected.labels(
                    detector_id=detector_id,
                    severity=result.severity.value
                ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return None
    
    async def detect_stream_event(self, event: StreamEvent, detector_configs: List[str] = None) -> List[AnomalyResult]:
        """Detect anomalies in a stream event using multiple detectors"""
        results = []
        
        # Use all detectors if none specified
        if detector_configs is None:
            detector_configs = list(self.detectors.keys())
        
        # Extract numeric data from event
        data_point = self._extract_numeric_data(event)
        if data_point is None:
            return results
        
        # Run detection with each detector
        for detector_id in detector_configs:
            if detector_id in self.detectors:
                result = await self.detect_anomalies(detector_id, data_point)
                if result:
                    result.event_id = event.event_id  # Link to original event
                    results.append(result)
        
        return results
    
    def _extract_numeric_data(self, event: StreamEvent) -> Optional[Union[float, List[float]]]:
        """Extract numeric data from stream event"""
        try:
            # Try different ways to extract numeric data
            if isinstance(event.data, (int, float)):
                return float(event.data)
            elif isinstance(event.data, dict):
                # Look for common numeric fields
                for field in ['value', 'amount', 'count', 'size', 'duration', 'timestamp']:
                    if field in event.data and isinstance(event.data[field], (int, float)):
                        return float(event.data[field])
                
                # Extract all numeric values
                numeric_values = []
                for key, value in event.data.items():
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
                
                if numeric_values:
                    return numeric_values if len(numeric_values) > 1 else numeric_values[0]
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting numeric data: {e}")
            return None
    
    async def _handle_anomaly_alert(self, result: AnomalyResult) -> None:
        """Handle anomaly alert with cooldown"""
        detector_id = result.detector_id
        current_time = time.time() * 1000
        
        # Check cooldown period
        last_alert = self.last_alerts.get(detector_id, 0)
        detector = self.detectors.get(detector_id)
        cooldown_ms = detector.config.alert_cooldown_ms if detector else 5000
        
        if current_time - last_alert < cooldown_ms:
            logger.debug(f"Anomaly alert suppressed due to cooldown: {detector_id}")
            return
        
        # Update last alert time
        self.last_alerts[detector_id] = current_time
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(result)
                else:
                    handler(result)
            except Exception as e:
                logger.error(f"Error in anomaly alert handler: {e}")
        
        # Log alert
        logger.warning(
            f"ANOMALY DETECTED - {result.severity.value.upper()}: "
            f"{result.explanation} (Confidence: {result.confidence:.3f})"
        )
    
    def register_alert_handler(self, handler: Callable) -> None:
        """Register handler for anomaly alerts"""
        self.alert_handlers.append(handler)
        logger.info("Registered anomaly alert handler")
    
    def get_detector(self, detector_id: str) -> Optional[AnomalyDetector]:
        """Get detector by ID"""
        return self.detectors.get(detector_id)
    
    def list_detectors(self) -> List[str]:
        """List all detector IDs"""
        return list(self.detectors.keys())
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all detectors"""
        return {
            detector_id: detector.get_metrics()
            for detector_id, detector in self.detectors.items()
        }
    
    async def start(self) -> None:
        """Start anomaly detection manager"""
        self.is_running = True
        logger.info(f"Started anomaly detection manager with {len(self.detectors)} detectors")
    
    async def stop(self) -> None:
        """Stop anomaly detection manager"""
        self.is_running = False
        
        # Wait for processing tasks
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        logger.info("Anomaly detection manager stopped")

# Global anomaly detection manager
anomaly_manager = AnomalyDetectionManager()

# Convenience functions
async def create_anomaly_detector(detector_id: str, method: str, **kwargs) -> AnomalyDetector:
    """Create an anomaly detector"""
    config = AnomalyDetectionConfig(
        detector_id=detector_id,
        name=f"Detector {detector_id}",
        method=DetectionMethod(method),
        **kwargs
    )
    return anomaly_manager.create_detector(config)

async def detect_anomaly(detector_id: str, data_point: Any) -> Optional[AnomalyResult]:
    """Detect anomaly using specified detector"""
    return await anomaly_manager.detect_anomalies(detector_id, data_point)

async def detect_stream_anomalies(event: StreamEvent) -> List[AnomalyResult]:
    """Detect anomalies in stream event"""
    return await anomaly_manager.detect_stream_event(event)
