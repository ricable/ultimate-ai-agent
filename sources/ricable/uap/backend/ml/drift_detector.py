# backend/ml/drift_detector.py
# Model Drift Detection and Monitoring System

import asyncio
import json
import uuid
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
from scipy import stats
from collections import deque, defaultdict
import hashlib

# Integrations
from ..distributed.ray_manager import submit_distributed_task
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.prometheus_metrics import record_pipeline_event

class DriftType(Enum):
    """Types of drift detection"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"

class DriftSeverity(Enum):
    """Drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriftStatus(Enum):
    """Drift detection status"""
    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    MONITORING = "monitoring"
    ERROR = "error"

@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    model_id: str
    drift_type: DriftType
    severity: DriftSeverity
    detected_at: datetime
    drift_score: float
    threshold: float
    affected_features: List[str]
    description: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['drift_type'] = self.drift_type.value
        data['severity'] = self.severity.value
        data['detected_at'] = self.detected_at.isoformat()
        return data

@dataclass
class DriftMetrics:
    """Drift detection metrics"""
    metric_id: str
    model_id: str
    timestamp: datetime
    drift_type: DriftType
    drift_scores: Dict[str, float]
    feature_statistics: Dict[str, Any]
    performance_metrics: Dict[str, float]
    baseline_comparison: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['drift_type'] = self.drift_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class DriftMonitorConfig:
    """Configuration for drift monitoring"""
    model_id: str
    enabled: bool
    detection_window_hours: int
    alert_thresholds: Dict[DriftType, float]
    baseline_update_frequency_hours: int
    feature_list: List[str]
    performance_metrics: List[str]
    notification_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['alert_thresholds'] = {k.value: v for k, v in self.alert_thresholds.items()}
        return data

class StatisticalDriftDetector:
    """Statistical drift detection algorithms"""
    
    @staticmethod
    def kolmogorov_smirnov_test(reference_data: np.ndarray, 
                               current_data: np.ndarray,
                               alpha: float = 0.05) -> Tuple[float, bool]:
        """Kolmogorov-Smirnov two-sample test for drift detection"""
        try:
            if len(reference_data) == 0 or len(current_data) == 0:
                return 0.0, False
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
            
            # Drift detected if p-value < alpha
            drift_detected = p_value < alpha
            
            # Return KS statistic as drift score (higher = more drift)
            return float(ks_statistic), drift_detected
            
        except Exception:
            return 0.0, False
    
    @staticmethod
    def wasserstein_distance(reference_data: np.ndarray,
                           current_data: np.ndarray,
                           threshold: float = 0.1) -> Tuple[float, bool]:
        """Wasserstein distance for drift detection"""
        try:
            if len(reference_data) == 0 or len(current_data) == 0:
                return 0.0, False
            
            # Calculate Wasserstein distance
            distance = stats.wasserstein_distance(reference_data, current_data)
            
            # Normalize by reference data range
            ref_range = np.max(reference_data) - np.min(reference_data)
            if ref_range > 0:
                normalized_distance = distance / ref_range
            else:
                normalized_distance = 0.0
            
            drift_detected = normalized_distance > threshold
            
            return float(normalized_distance), drift_detected
            
        except Exception:
            return 0.0, False
    
    @staticmethod
    def jensen_shannon_divergence(reference_data: np.ndarray,
                                current_data: np.ndarray,
                                bins: int = 50,
                                threshold: float = 0.1) -> Tuple[float, bool]:
        """Jensen-Shannon divergence for drift detection"""
        try:
            if len(reference_data) == 0 or len(current_data) == 0:
                return 0.0, False
            
            # Create histograms
            data_range = (
                min(np.min(reference_data), np.min(current_data)),
                max(np.max(reference_data), np.max(current_data))
            )
            
            ref_hist, _ = np.histogram(reference_data, bins=bins, range=data_range, density=True)
            curr_hist, _ = np.histogram(current_data, bins=bins, range=data_range, density=True)
            
            # Normalize to probabilities
            ref_hist = ref_hist / np.sum(ref_hist)
            curr_hist = curr_hist / np.sum(curr_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_hist += epsilon
            curr_hist += epsilon
            
            # Calculate Jensen-Shannon divergence
            m = 0.5 * (ref_hist + curr_hist)
            js_divergence = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(curr_hist, m)
            
            drift_detected = js_divergence > threshold
            
            return float(js_divergence), drift_detected
            
        except Exception:
            return 0.0, False
    
    @staticmethod
    def population_stability_index(reference_data: np.ndarray,
                                 current_data: np.ndarray,
                                 bins: int = 10,
                                 threshold: float = 0.2) -> Tuple[float, bool]:
        """Population Stability Index for drift detection"""
        try:
            if len(reference_data) == 0 or len(current_data) == 0:
                return 0.0, False
            
            # Create quantile-based bins
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(reference_data, quantiles)
            
            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                return 0.0, False
            
            # Calculate proportions in each bin
            ref_counts, _ = np.histogram(reference_data, bins=bin_edges)
            curr_counts, _ = np.histogram(current_data, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / np.sum(ref_counts)
            curr_props = curr_counts / np.sum(curr_counts)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            ref_props += epsilon
            curr_props += epsilon
            
            # Calculate PSI
            psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
            
            drift_detected = psi > threshold
            
            return float(psi), drift_detected
            
        except Exception:
            return 0.0, False

class PerformanceDriftDetector:
    """Performance-based drift detection"""
    
    @staticmethod
    def performance_degradation_test(baseline_metrics: Dict[str, float],
                                   current_metrics: Dict[str, float],
                                   degradation_threshold: float = 0.05) -> Tuple[float, bool]:
        """Detect performance degradation drift"""
        try:
            if not baseline_metrics or not current_metrics:
                return 0.0, False
            
            degradations = []
            
            for metric_name in baseline_metrics:
                if metric_name in current_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    current_value = current_metrics[metric_name]
                    
                    if baseline_value > 0:
                        # Calculate relative degradation
                        degradation = (baseline_value - current_value) / baseline_value
                        degradations.append(max(0, degradation))  # Only count degradations
            
            if not degradations:
                return 0.0, False
            
            # Use average degradation as drift score
            avg_degradation = np.mean(degradations)
            drift_detected = avg_degradation > degradation_threshold
            
            return float(avg_degradation), drift_detected
            
        except Exception:
            return 0.0, False

class DriftDetector:
    """
    Model Drift Detection and Monitoring System.
    
    Provides comprehensive drift detection including:
    - Data drift detection (input distribution changes)
    - Concept drift detection (input-output relationship changes)
    - Prediction drift detection (output distribution changes)
    - Performance drift detection (model performance degradation)
    """
    
    def __init__(self, storage_dir: str = "./drift_monitoring"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Drift detectors
        self.statistical_detector = StatisticalDriftDetector()
        self.performance_detector = PerformanceDriftDetector()
        
        # Monitoring state
        self.monitor_configs: Dict[str, DriftMonitorConfig] = {}
        self.drift_alerts: Dict[str, List[DriftAlert]] = defaultdict(list)
        self.drift_metrics: Dict[str, List[DriftMetrics]] = defaultdict(list)
        
        # Data storage for drift detection
        self.reference_data: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        self.current_data_buffer: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self.performance_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Configuration
        self.default_thresholds = {
            DriftType.DATA_DRIFT: 0.1,
            DriftType.CONCEPT_DRIFT: 0.15,
            DriftType.PREDICTION_DRIFT: 0.1,
            DriftType.PERFORMANCE_DRIFT: 0.05
        }
        
        self.monitoring_active = False
    
    async def initialize(self) -> bool:
        """Initialize drift detector"""
        try:
            await self._load_monitor_configs()
            await self._load_reference_data()
            
            # Start monitoring
            asyncio.create_task(self._start_monitoring())
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Drift Detector initialized",
                EventType.AGENT,
                {"storage_dir": str(self.storage_dir)},
                "drift_detector"
            )
            
            return True
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to initialize Drift Detector: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "drift_detector"
            )
            return False
    
    async def configure_monitoring(self,
                                 model_id: str,
                                 feature_list: List[str],
                                 performance_metrics: List[str] = None,
                                 detection_window_hours: int = 24,
                                 alert_thresholds: Dict[DriftType, float] = None) -> bool:
        """Configure drift monitoring for a model"""
        
        config = DriftMonitorConfig(
            model_id=model_id,
            enabled=True,
            detection_window_hours=detection_window_hours,
            alert_thresholds=alert_thresholds or self.default_thresholds,
            baseline_update_frequency_hours=168,  # Weekly
            feature_list=feature_list,
            performance_metrics=performance_metrics or ["accuracy", "precision", "recall"],
            notification_config={}
        )
        
        self.monitor_configs[model_id] = config
        
        # Save configuration
        await self._save_monitor_config(config)
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Drift monitoring configured for model: {model_id}",
            EventType.AGENT,
            {
                "model_id": model_id,
                "features": len(feature_list),
                "detection_window_hours": detection_window_hours
            },
            "drift_detector"
        )
        
        return True
    
    async def set_reference_baseline(self,
                                   model_id: str,
                                   reference_data: Dict[str, List[float]],
                                   performance_metrics: Dict[str, float] = None) -> bool:
        """Set reference baseline for drift detection"""
        
        try:
            # Store reference data
            for feature_name, feature_data in reference_data.items():
                self.reference_data[model_id][feature_name] = np.array(feature_data)
            
            # Store reference performance metrics
            if performance_metrics:
                self.reference_data[model_id]["_performance_baseline"] = performance_metrics
            
            # Save to storage
            await self._save_reference_data(model_id)
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Reference baseline set for model: {model_id}",
                EventType.AGENT,
                {
                    "model_id": model_id,
                    "features": len(reference_data),
                    "data_points": sum(len(data) for data in reference_data.values())
                },
                "drift_detector"
            )
            
            return True
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to set reference baseline for {model_id}: {e}",
                EventType.AGENT,
                {"model_id": model_id, "error": str(e)},
                "drift_detector"
            )
            return False
    
    async def record_prediction(self,
                              model_id: str,
                              input_features: Dict[str, float],
                              prediction: Any,
                              prediction_metadata: Dict[str, Any] = None) -> bool:
        """Record a model prediction for drift monitoring"""
        
        try:
            if model_id not in self.monitor_configs:
                return False
            
            config = self.monitor_configs[model_id]
            if not config.enabled:
                return False
            
            # Record input features
            for feature_name, feature_value in input_features.items():
                if feature_name in config.feature_list:
                    self.current_data_buffer[model_id][feature_name].append(feature_value)
            
            # Record prediction (for prediction drift detection)
            if isinstance(prediction, (int, float)):
                self.current_data_buffer[model_id]["_predictions"].append(prediction)
            
            # Record performance metrics if provided
            if prediction_metadata and "performance_metrics" in prediction_metadata:
                self.performance_buffer[model_id].append(prediction_metadata["performance_metrics"])
            
            return True
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to record prediction for {model_id}: {e}",
                EventType.AGENT,
                {"model_id": model_id, "error": str(e)},
                "drift_detector"
            )
            return False
    
    async def detect_drift(self, model_id: str) -> Dict[str, Any]:
        """Perform drift detection for a model"""
        
        if model_id not in self.monitor_configs:
            return {"error": "Model not configured for drift monitoring"}
        
        config = self.monitor_configs[model_id]
        
        try:
            # Collect drift detection results
            drift_results = {
                "model_id": model_id,
                "detection_timestamp": datetime.utcnow().isoformat(),
                "drift_detected": False,
                "drift_scores": {},
                "alerts": []
            }
            
            # Data drift detection
            data_drift_results = await self._detect_data_drift(model_id, config)
            drift_results["drift_scores"]["data_drift"] = data_drift_results
            
            # Prediction drift detection
            prediction_drift_results = await self._detect_prediction_drift(model_id, config)
            drift_results["drift_scores"]["prediction_drift"] = prediction_drift_results
            
            # Performance drift detection
            performance_drift_results = await self._detect_performance_drift(model_id, config)
            drift_results["drift_scores"]["performance_drift"] = performance_drift_results
            
            # Generate alerts
            alerts = await self._generate_drift_alerts(model_id, config, {
                DriftType.DATA_DRIFT: data_drift_results,
                DriftType.PREDICTION_DRIFT: prediction_drift_results,
                DriftType.PERFORMANCE_DRIFT: performance_drift_results
            })
            
            drift_results["alerts"] = [alert.to_dict() for alert in alerts]
            drift_results["drift_detected"] = len(alerts) > 0
            
            # Store metrics
            await self._store_drift_metrics(model_id, drift_results)
            
            # Record Prometheus metrics
            record_pipeline_event(model_id, model_id, "drift_detection_completed")
            
            return drift_results
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to detect drift for {model_id}: {e}",
                EventType.AGENT,
                {"model_id": model_id, "error": str(e)},
                "drift_detector"
            )
            
            return {
                "error": str(e),
                "model_id": model_id,
                "detection_timestamp": datetime.utcnow().isoformat()
            }
    
    async def _detect_data_drift(self,
                               model_id: str,
                               config: DriftMonitorConfig) -> Dict[str, Any]:
        """Detect data drift in input features"""
        
        drift_results = {
            "overall_drift_detected": False,
            "feature_drift_scores": {},
            "affected_features": []
        }
        
        try:
            if model_id not in self.reference_data:
                return {"error": "No reference data available"}
            
            reference_data = self.reference_data[model_id]
            current_buffers = self.current_data_buffer[model_id]
            
            for feature_name in config.feature_list:
                if feature_name not in reference_data or feature_name not in current_buffers:
                    continue
                
                if len(current_buffers[feature_name]) < 30:  # Minimum sample size
                    continue
                
                ref_data = reference_data[feature_name]
                curr_data = np.array(list(current_buffers[feature_name]))
                
                # Run multiple drift detection algorithms
                ks_score, ks_drift = self.statistical_detector.kolmogorov_smirnov_test(ref_data, curr_data)
                ws_score, ws_drift = self.statistical_detector.wasserstein_distance(ref_data, curr_data)
                js_score, js_drift = self.statistical_detector.jensen_shannon_divergence(ref_data, curr_data)
                psi_score, psi_drift = self.statistical_detector.population_stability_index(ref_data, curr_data)
                
                # Combine scores (weighted average)
                combined_score = (ks_score * 0.25 + ws_score * 0.25 + js_score * 0.25 + psi_score * 0.25)
                feature_drift_detected = any([ks_drift, ws_drift, js_drift, psi_drift])
                
                drift_results["feature_drift_scores"][feature_name] = {
                    "combined_score": combined_score,
                    "kolmogorov_smirnov": {"score": ks_score, "drift_detected": ks_drift},
                    "wasserstein_distance": {"score": ws_score, "drift_detected": ws_drift},
                    "jensen_shannon": {"score": js_score, "drift_detected": js_drift},
                    "population_stability_index": {"score": psi_score, "drift_detected": psi_drift},
                    "feature_drift_detected": feature_drift_detected
                }
                
                if feature_drift_detected:
                    drift_results["affected_features"].append(feature_name)
            
            # Overall drift detection
            drift_results["overall_drift_detected"] = len(drift_results["affected_features"]) > 0
            
            return drift_results
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _detect_prediction_drift(self,
                                     model_id: str,
                                     config: DriftMonitorConfig) -> Dict[str, Any]:
        """Detect drift in model predictions"""
        
        try:
            if model_id not in self.reference_data:
                return {"error": "No reference data available"}
            
            current_buffers = self.current_data_buffer[model_id]
            
            if "_predictions" not in current_buffers or len(current_buffers["_predictions"]) < 30:
                return {"error": "Insufficient prediction data"}
            
            # Use historical predictions as reference if available
            if "_predictions" in self.reference_data[model_id]:
                ref_predictions = self.reference_data[model_id]["_predictions"]
            else:
                # Use older predictions from buffer as reference
                ref_predictions = np.array(list(current_buffers["_predictions"])[:len(current_buffers["_predictions"])//2])
            
            curr_predictions = np.array(list(current_buffers["_predictions"]))
            
            # Detect prediction drift
            ks_score, ks_drift = self.statistical_detector.kolmogorov_smirnov_test(ref_predictions, curr_predictions)
            ws_score, ws_drift = self.statistical_detector.wasserstein_distance(ref_predictions, curr_predictions)
            
            combined_score = (ks_score + ws_score) / 2
            
            return {
                "prediction_drift_detected": ks_drift or ws_drift,
                "combined_score": combined_score,
                "kolmogorov_smirnov": {"score": ks_score, "drift_detected": ks_drift},
                "wasserstein_distance": {"score": ws_score, "drift_detected": ws_drift}
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _detect_performance_drift(self,
                                      model_id: str,
                                      config: DriftMonitorConfig) -> Dict[str, Any]:
        """Detect performance drift"""
        
        try:
            if model_id not in self.reference_data or "_performance_baseline" not in self.reference_data[model_id]:
                return {"error": "No performance baseline available"}
            
            baseline_metrics = self.reference_data[model_id]["_performance_baseline"]
            
            if model_id not in self.performance_buffer or len(self.performance_buffer[model_id]) < 10:
                return {"error": "Insufficient performance data"}
            
            # Calculate current performance metrics
            recent_performance = list(self.performance_buffer[model_id])[-10:]  # Last 10 evaluations
            current_metrics = {}
            
            for metric_name in config.performance_metrics:
                metric_values = [perf.get(metric_name, 0) for perf in recent_performance if isinstance(perf, dict)]
                if metric_values:
                    current_metrics[metric_name] = np.mean(metric_values)
            
            # Detect performance drift
            drift_score, drift_detected = self.performance_detector.performance_degradation_test(
                baseline_metrics, current_metrics
            )
            
            return {
                "performance_drift_detected": drift_detected,
                "drift_score": drift_score,
                "baseline_metrics": baseline_metrics,
                "current_metrics": current_metrics,
                "metric_comparisons": {
                    metric: {
                        "baseline": baseline_metrics.get(metric, 0),
                        "current": current_metrics.get(metric, 0),
                        "degradation": max(0, (baseline_metrics.get(metric, 0) - current_metrics.get(metric, 0)) / baseline_metrics.get(metric, 1))
                    }
                    for metric in config.performance_metrics
                    if metric in baseline_metrics and metric in current_metrics
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _generate_drift_alerts(self,
                                   model_id: str,
                                   config: DriftMonitorConfig,
                                   drift_results: Dict[DriftType, Dict[str, Any]]) -> List[DriftAlert]:
        """Generate drift alerts based on detection results"""
        
        alerts = []
        
        for drift_type, results in drift_results.items():
            if results.get("error"):
                continue
            
            threshold = config.alert_thresholds.get(drift_type, 0.1)
            
            # Check if drift exceeds threshold
            if drift_type == DriftType.DATA_DRIFT:
                if results.get("overall_drift_detected", False):
                    severity = self._determine_severity(len(results.get("affected_features", [])) / len(config.feature_list))
                    
                    alert = DriftAlert(
                        alert_id=str(uuid.uuid4()),
                        model_id=model_id,
                        drift_type=drift_type,
                        severity=severity,
                        detected_at=datetime.utcnow(),
                        drift_score=len(results.get("affected_features", [])) / len(config.feature_list),
                        threshold=threshold,
                        affected_features=results.get("affected_features", []),
                        description=f"Data drift detected in {len(results.get('affected_features', []))} features",
                        metadata={"drift_details": results}
                    )
                    alerts.append(alert)
            
            elif drift_type == DriftType.PREDICTION_DRIFT:
                if results.get("prediction_drift_detected", False):
                    drift_score = results.get("combined_score", 0)
                    if drift_score > threshold:
                        severity = self._determine_severity(drift_score / threshold)
                        
                        alert = DriftAlert(
                            alert_id=str(uuid.uuid4()),
                            model_id=model_id,
                            drift_type=drift_type,
                            severity=severity,
                            detected_at=datetime.utcnow(),
                            drift_score=drift_score,
                            threshold=threshold,
                            affected_features=["predictions"],
                            description="Prediction distribution drift detected",
                            metadata={"drift_details": results}
                        )
                        alerts.append(alert)
            
            elif drift_type == DriftType.PERFORMANCE_DRIFT:
                if results.get("performance_drift_detected", False):
                    drift_score = results.get("drift_score", 0)
                    if drift_score > threshold:
                        severity = self._determine_severity(drift_score / threshold)
                        
                        alert = DriftAlert(
                            alert_id=str(uuid.uuid4()),
                            model_id=model_id,
                            drift_type=drift_type,
                            severity=severity,
                            detected_at=datetime.utcnow(),
                            drift_score=drift_score,
                            threshold=threshold,
                            affected_features=list(results.get("current_metrics", {}).keys()),
                            description="Model performance degradation detected",
                            metadata={"drift_details": results}
                        )
                        alerts.append(alert)
        
        # Store alerts
        self.drift_alerts[model_id].extend(alerts)
        
        # Keep only recent alerts (last 100)
        if len(self.drift_alerts[model_id]) > 100:
            self.drift_alerts[model_id] = self.drift_alerts[model_id][-100:]
        
        return alerts
    
    def _determine_severity(self, ratio: float) -> DriftSeverity:
        """Determine drift severity based on score ratio"""
        if ratio < 1.5:
            return DriftSeverity.LOW
        elif ratio < 2.0:
            return DriftSeverity.MEDIUM
        elif ratio < 3.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    async def get_drift_status(self, model_id: str, days: int = 7) -> Dict[str, Any]:
        """Get drift monitoring status for a model"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get recent alerts
        recent_alerts = []
        if model_id in self.drift_alerts:
            recent_alerts = [
                alert.to_dict() for alert in self.drift_alerts[model_id]
                if alert.detected_at > cutoff_date
            ]
        
        # Get recent metrics
        recent_metrics = []
        if model_id in self.drift_metrics:
            recent_metrics = [
                metric.to_dict() for metric in self.drift_metrics[model_id]
                if metric.timestamp > cutoff_date
            ]
        
        # Calculate summary statistics
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert["drift_type"]] += 1
        
        return {
            "model_id": model_id,
            "monitoring_enabled": model_id in self.monitor_configs and self.monitor_configs[model_id].enabled,
            "recent_alerts": recent_alerts,
            "recent_metrics": recent_metrics,
            "alert_summary": dict(alert_counts),
            "total_alerts": len(recent_alerts),
            "has_baseline": model_id in self.reference_data and len(self.reference_data[model_id]) > 0,
            "current_data_points": {
                feature: len(buffer) for feature, buffer in self.current_data_buffer[model_id].items()
            } if model_id in self.current_data_buffer else {}
        }
    
    async def _start_monitoring(self):
        """Start background drift monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Run drift detection for all configured models
                for model_id in self.monitor_configs:
                    if self.monitor_configs[model_id].enabled:
                        await submit_distributed_task(
                            "drift_detection",
                            self.detect_drift,
                            {"model_id": model_id},
                            priority=5
                        )
                
                # Clean up old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Drift monitoring error: {e}",
                    EventType.AGENT,
                    {"error": str(e)},
                    "drift_detector"
                )
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # Clean up old alerts
        for model_id in self.drift_alerts:
            self.drift_alerts[model_id] = [
                alert for alert in self.drift_alerts[model_id]
                if alert.detected_at > cutoff_date
            ]
        
        # Clean up old metrics
        for model_id in self.drift_metrics:
            self.drift_metrics[model_id] = [
                metric for metric in self.drift_metrics[model_id]
                if metric.timestamp > cutoff_date
            ]
    
    async def _store_drift_metrics(self, model_id: str, drift_results: Dict[str, Any]):
        """Store drift detection metrics"""
        try:
            metric = DriftMetrics(
                metric_id=str(uuid.uuid4()),
                model_id=model_id,
                timestamp=datetime.utcnow(),
                drift_type=DriftType.DATA_DRIFT,  # Primary type
                drift_scores=drift_results.get("drift_scores", {}),
                feature_statistics={},
                performance_metrics={},
                baseline_comparison={}
            )
            
            self.drift_metrics[model_id].append(metric)
            
            # Keep only recent metrics
            if len(self.drift_metrics[model_id]) > 1000:
                self.drift_metrics[model_id] = self.drift_metrics[model_id][-1000:]
                
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to store drift metrics: {e}",
                EventType.AGENT,
                {"model_id": model_id, "error": str(e)},
                "drift_detector"
            )
    
    async def _load_monitor_configs(self):
        """Load monitoring configurations from storage"""
        try:
            config_file = self.storage_dir / "monitor_configs.json"
            if config_file.exists():
                async with aiofiles.open(config_file, 'r') as f:
                    data = json.loads(await f.read())
                    # Reconstruct configs (simplified)
                    pass
        except Exception as e:
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Failed to load monitor configs: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "drift_detector"
            )
    
    async def _load_reference_data(self):
        """Load reference data from storage"""
        try:
            for model_dir in self.storage_dir.glob("model_*"):
                if model_dir.is_dir():
                    model_id = model_dir.name.replace("model_", "")
                    ref_file = model_dir / "reference_data.json"
                    
                    if ref_file.exists():
                        async with aiofiles.open(ref_file, 'r') as f:
                            data = json.loads(await f.read())
                            # Reconstruct reference data (simplified)
                            pass
        except Exception as e:
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Failed to load reference data: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "drift_detector"
            )
    
    async def _save_monitor_config(self, config: DriftMonitorConfig):
        """Save monitoring configuration"""
        try:
            config_file = self.storage_dir / f"config_{config.model_id}.json"
            
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(json.dumps(config.to_dict(), indent=2))
                
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to save monitor config: {e}",
                EventType.AGENT,
                {"model_id": config.model_id, "error": str(e)},
                "drift_detector"
            )
    
    async def _save_reference_data(self, model_id: str):
        """Save reference data for a model"""
        try:
            model_dir = self.storage_dir / f"model_{model_id}"
            model_dir.mkdir(exist_ok=True)
            
            ref_file = model_dir / "reference_data.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for feature_name, feature_data in self.reference_data[model_id].items():
                if isinstance(feature_data, np.ndarray):
                    serializable_data[feature_name] = feature_data.tolist()
                else:
                    serializable_data[feature_name] = feature_data
            
            async with aiofiles.open(ref_file, 'w') as f:
                await f.write(json.dumps(serializable_data, indent=2))
                
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to save reference data: {e}",
                EventType.AGENT,
                {"model_id": model_id, "error": str(e)},
                "drift_detector"
            )
    
    async def get_detector_status(self) -> Dict[str, Any]:
        """Get drift detector status"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": self.monitoring_active,
            "configured_models": len(self.monitor_configs),
            "total_alerts": sum(len(alerts) for alerts in self.drift_alerts.values()),
            "storage_dir": str(self.storage_dir)
        }

# Global drift detector instance
_drift_detector = None

def get_drift_detector() -> DriftDetector:
    """Get the global drift detector instance"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector