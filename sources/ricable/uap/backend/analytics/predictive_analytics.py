# File: backend/analytics/predictive_analytics.py
"""
Predictive Analytics System for UAP Platform

Provides predictive capabilities for resource planning, usage forecasting,
and proactive issue detection using machine learning models.
"""

import asyncio
import json
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import uuid
import warnings
from threading import Lock

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback to simple statistical methods

from .usage_analytics import usage_analytics, EventType
from ..monitoring.metrics.performance import performance_monitor

class PredictionType(Enum):
    """Types of predictions that can be made"""
    USAGE_FORECAST = "usage_forecast"
    RESOURCE_DEMAND = "resource_demand"
    ANOMALY_DETECTION = "anomaly_detection"
    CAPACITY_PLANNING = "capacity_planning"
    PERFORMANCE_TREND = "performance_trend"
    ERROR_PREDICTION = "error_prediction"
    USER_BEHAVIOR = "user_behavior"
    COST_PROJECTION = "cost_projection"

class ModelType(Enum):
    """Machine learning model types"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    ISOLATION_FOREST = "isolation_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"

class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    HIGH = "high"      # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"        # <70% confidence

@dataclass
class PredictionResult:
    """Container for prediction results"""
    prediction_id: str
    prediction_type: PredictionType
    model_type: ModelType
    predicted_value: Union[float, int, Dict[str, Any]]
    confidence: PredictionConfidence
    confidence_score: float
    time_horizon: timedelta
    created_at: datetime
    metadata: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "prediction_type": self.prediction_type.value,
            "model_type": self.model_type.value,
            "predicted_value": self.predicted_value,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "time_horizon_hours": self.time_horizon.total_seconds() / 3600,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "feature_importance": self.feature_importance
        }

@dataclass
class TimeSeriesPoint:
    """Time series data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "metadata": self.metadata or {}
        }

class ModelPerformanceMetrics(NamedTuple):
    """Model performance evaluation metrics"""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float # Root Mean Squared Error
    r2: float   # R-squared
    accuracy: Optional[float] = None  # For classification models

class PredictiveModel:
    """Base class for predictive models"""
    
    def __init__(self, model_type: ModelType, prediction_type: PredictionType):
        self.model_type = model_type
        self.prediction_type = prediction_type
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.last_trained = None
        self.performance_metrics = None
        self.feature_names = []
    
    def prepare_features(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets from raw data"""
        raise NotImplementedError
    
    def train(self, data: List[Dict[str, Any]]) -> ModelPerformanceMetrics:
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, features: np.ndarray) -> Tuple[Union[float, np.ndarray], float]:
        """Make predictions and return confidence score"""
        raise NotImplementedError
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        return None

class UsageForecastModel(PredictiveModel):
    """Model for forecasting usage patterns"""
    
    def __init__(self):
        super().__init__(ModelType.RANDOM_FOREST, PredictionType.USAGE_FORECAST)
        if SKLEARN_AVAILABLE:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
    
    def prepare_features(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for usage forecasting"""
        features = []
        targets = []
        
        for point in data:
            timestamp = datetime.fromisoformat(point['timestamp'])
            
            # Time-based features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            day_of_month = timestamp.day
            month = timestamp.month
            
            # Usage features
            active_users = point.get('active_users', 0)
            total_requests = point.get('total_requests', 0)
            error_rate = point.get('error_rate', 0)
            avg_response_time = point.get('avg_response_time', 0)
            
            feature_vector = [
                hour, day_of_week, day_of_month, month,
                active_users, total_requests, error_rate, avg_response_time
            ]
            
            features.append(feature_vector)
            targets.append(point.get('target_value', total_requests))
        
        self.feature_names = [
            'hour', 'day_of_week', 'day_of_month', 'month',
            'active_users', 'total_requests', 'error_rate', 'avg_response_time'
        ]
        
        return np.array(features), np.array(targets)
    
    def train(self, data: List[Dict[str, Any]]) -> ModelPerformanceMetrics:
        """Train the usage forecasting model"""
        if not SKLEARN_AVAILABLE or len(data) < 10:
            # Fallback to simple statistical method
            self.is_trained = True
            self.last_trained = datetime.utcnow()
            return ModelPerformanceMetrics(0.1, 0.01, 0.1, 0.8)
        
        X, y = self.prepare_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        self.performance_metrics = ModelPerformanceMetrics(mae, mse, rmse, r2)
        self.is_trained = True
        self.last_trained = datetime.utcnow()
        
        return self.performance_metrics
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make usage forecast prediction"""
        if not self.is_trained:
            return 0.0, 0.0
        
        if not SKLEARN_AVAILABLE:
            # Simple statistical fallback
            return float(np.mean(features)), 0.8
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on model performance
        confidence = min(0.95, max(0.5, self.performance_metrics.r2)) if self.performance_metrics else 0.7
        
        return float(prediction), confidence
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance"""
        if not self.is_trained or not SKLEARN_AVAILABLE or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

class AnomalyDetectionModel(PredictiveModel):
    """Model for detecting anomalies in system behavior"""
    
    def __init__(self):
        super().__init__(ModelType.ISOLATION_FOREST, PredictionType.ANOMALY_DETECTION)
        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
    
    def prepare_features(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for anomaly detection"""
        features = []
        
        for point in data:
            feature_vector = [
                point.get('cpu_usage', 0),
                point.get('memory_usage', 0),
                point.get('active_connections', 0),
                point.get('request_rate', 0),
                point.get('error_rate', 0),
                point.get('response_time', 0)
            ]
            features.append(feature_vector)
        
        self.feature_names = [
            'cpu_usage', 'memory_usage', 'active_connections',
            'request_rate', 'error_rate', 'response_time'
        ]
        
        return np.array(features), np.array([0] * len(features))  # Unsupervised
    
    def train(self, data: List[Dict[str, Any]]) -> ModelPerformanceMetrics:
        """Train the anomaly detection model"""
        if not SKLEARN_AVAILABLE or len(data) < 20:
            self.is_trained = True
            self.last_trained = datetime.utcnow()
            return ModelPerformanceMetrics(0.1, 0.01, 0.1, 0.8)
        
        X, _ = self.prepare_features(data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Evaluate on training data (for unsupervised learning)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        self.performance_metrics = ModelPerformanceMetrics(
            mae=np.std(anomaly_scores),
            mse=np.var(anomaly_scores),
            rmse=np.std(anomaly_scores),
            r2=0.8  # Placeholder for unsupervised
        )
        
        self.is_trained = True
        self.last_trained = datetime.utcnow()
        
        return self.performance_metrics
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict anomaly score"""
        if not self.is_trained:
            return 0.0, 0.0
        
        if not SKLEARN_AVAILABLE:
            # Simple statistical fallback - check if values are outside 2 standard deviations
            z_scores = np.abs((features - np.mean(features)) / (np.std(features) + 1e-8))
            anomaly_score = float(np.max(z_scores))
            return anomaly_score, 0.7
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        anomaly_score = -self.model.decision_function(features_scaled)[0]  # Higher = more anomalous
        
        confidence = 0.8 if abs(anomaly_score) > 0.5 else 0.6
        
        return float(anomaly_score), confidence

class PredictiveAnalytics:
    """Main predictive analytics system"""
    
    def __init__(self):
        self.models: Dict[PredictionType, PredictiveModel] = {}
        self.prediction_history: deque = deque(maxlen=10000)
        self.training_data: Dict[PredictionType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = Lock()
        
        # Initialize models
        self._initialize_models()
        
        # Prediction settings
        self.min_training_samples = 50
        self.retrain_interval_hours = 24
        self.prediction_horizons = {
            PredictionType.USAGE_FORECAST: timedelta(hours=24),
            PredictionType.RESOURCE_DEMAND: timedelta(hours=12),
            PredictionType.ANOMALY_DETECTION: timedelta(minutes=15),
            PredictionType.CAPACITY_PLANNING: timedelta(days=7),
            PredictionType.PERFORMANCE_TREND: timedelta(hours=6)
        }
    
    def _initialize_models(self):
        """Initialize all predictive models"""
        self.models[PredictionType.USAGE_FORECAST] = UsageForecastModel()
        self.models[PredictionType.ANOMALY_DETECTION] = AnomalyDetectionModel()
        # Additional models would be initialized here
    
    def add_training_data(self, prediction_type: PredictionType, 
                         data_point: Dict[str, Any]):
        """Add new data point for model training"""
        with self.lock:
            self.training_data[prediction_type].append(data_point)
    
    async def train_model(self, prediction_type: PredictionType) -> Optional[ModelPerformanceMetrics]:
        """Train a specific model"""
        if prediction_type not in self.models:
            return None
        
        model = self.models[prediction_type]
        training_data = list(self.training_data[prediction_type])
        
        if len(training_data) < self.min_training_samples:
            return None
        
        try:
            performance = model.train(training_data)
            return performance
        except Exception as e:
            print(f"Error training {prediction_type.value} model: {e}")
            return None
    
    async def make_prediction(self, prediction_type: PredictionType,
                            features: Dict[str, Any],
                            time_horizon: Optional[timedelta] = None) -> Optional[PredictionResult]:
        """Make a prediction using the specified model"""
        if prediction_type not in self.models:
            return None
        
        model = self.models[prediction_type]
        
        if not model.is_trained:
            # Try to train the model first
            await self.train_model(prediction_type)
            
            if not model.is_trained:
                return None
        
        try:
            # Convert features to numpy array
            if prediction_type == PredictionType.USAGE_FORECAST:
                feature_array = np.array([
                    features.get('hour', 0),
                    features.get('day_of_week', 0),
                    features.get('day_of_month', 1),
                    features.get('month', 1),
                    features.get('active_users', 0),
                    features.get('total_requests', 0),
                    features.get('error_rate', 0),
                    features.get('avg_response_time', 0)
                ])
            elif prediction_type == PredictionType.ANOMALY_DETECTION:
                feature_array = np.array([
                    features.get('cpu_usage', 0),
                    features.get('memory_usage', 0),
                    features.get('active_connections', 0),
                    features.get('request_rate', 0),
                    features.get('error_rate', 0),
                    features.get('response_time', 0)
                ])
            else:
                return None
            
            predicted_value, confidence_score = model.predict(feature_array)
            
            # Determine confidence level
            if confidence_score >= 0.9:
                confidence = PredictionConfidence.HIGH
            elif confidence_score >= 0.7:
                confidence = PredictionConfidence.MEDIUM
            else:
                confidence = PredictionConfidence.LOW
            
            # Create prediction result
            prediction = PredictionResult(
                prediction_id=str(uuid.uuid4()),
                prediction_type=prediction_type,
                model_type=model.model_type,
                predicted_value=predicted_value,
                confidence=confidence,
                confidence_score=confidence_score,
                time_horizon=time_horizon or self.prediction_horizons.get(prediction_type, timedelta(hours=1)),
                created_at=datetime.utcnow(),
                metadata={
                    'input_features': features,
                    'model_last_trained': model.last_trained.isoformat() if model.last_trained else None,
                    'model_performance': model.performance_metrics._asdict() if model.performance_metrics else None
                },
                feature_importance=model.get_feature_importance()
            )
            
            # Store prediction
            with self.lock:
                self.prediction_history.append(prediction)
            
            return prediction
            
        except Exception as e:
            print(f"Error making {prediction_type.value} prediction: {e}")
            return None
    
    async def predict_usage_forecast(self, target_time: datetime) -> Optional[PredictionResult]:
        """Predict usage at a specific future time"""
        features = {
            'hour': target_time.hour,
            'day_of_week': target_time.weekday(),
            'day_of_month': target_time.day,
            'month': target_time.month,
            'active_users': len(usage_analytics.active_sessions),
            'total_requests': sum(stats.total_requests for stats in usage_analytics.agent_stats.values()),
            'error_rate': 0,  # Would calculate from recent data
            'avg_response_time': 0  # Would calculate from recent data
        }
        
        return await self.make_prediction(PredictionType.USAGE_FORECAST, features)
    
    async def detect_anomalies(self) -> Optional[PredictionResult]:
        """Detect current system anomalies"""
        # Get current system metrics
        health = performance_monitor.get_system_health()
        current_stats = health.get('current_stats', {})
        
        features = {
            'cpu_usage': current_stats.get('cpu_percent', 0),
            'memory_usage': current_stats.get('memory_percent', 0),
            'active_connections': current_stats.get('active_connections', 0),
            'request_rate': 0,  # Would calculate from recent metrics
            'error_rate': 0,  # Would calculate from recent metrics
            'response_time': 0  # Would calculate from recent metrics
        }
        
        return await self.make_prediction(PredictionType.ANOMALY_DETECTION, features)
    
    async def predict_resource_demand(self, hours_ahead: int = 12) -> Dict[str, Any]:
        """Predict resource demand for capacity planning"""
        predictions = {}
        
        target_time = datetime.utcnow() + timedelta(hours=hours_ahead)
        
        # Predict usage
        usage_prediction = await self.predict_usage_forecast(target_time)
        
        if usage_prediction:
            predicted_requests = usage_prediction.predicted_value
            
            # Estimate resource requirements based on predicted usage
            # These would be calibrated based on actual system behavior
            estimated_cpu = min(100, max(10, predicted_requests * 0.01))  # 1% CPU per 100 requests
            estimated_memory = min(90, max(20, predicted_requests * 0.005))  # 0.5% memory per 100 requests
            estimated_connections = min(1000, max(10, predicted_requests * 0.1))  # 10 connections per 100 requests
            
            predictions = {
                'predicted_requests': predicted_requests,
                'estimated_cpu_usage': estimated_cpu,
                'estimated_memory_usage': estimated_memory,
                'estimated_connections': estimated_connections,
                'confidence': usage_prediction.confidence.value,
                'time_horizon_hours': hours_ahead,
                'recommendations': self._generate_capacity_recommendations(
                    estimated_cpu, estimated_memory, estimated_connections
                )
            }
        
        return predictions
    
    def _generate_capacity_recommendations(self, cpu: float, memory: float, 
                                         connections: int) -> List[str]:
        """Generate capacity planning recommendations"""
        recommendations = []
        
        if cpu > 80:
            recommendations.append("Consider scaling CPU resources - high utilization predicted")
        
        if memory > 85:
            recommendations.append("Consider scaling memory resources - high utilization predicted")
        
        if connections > 800:
            recommendations.append("Consider scaling connection capacity - high load predicted")
        
        if cpu < 20 and memory < 30:
            recommendations.append("Resources may be over-provisioned - consider cost optimization")
        
        return recommendations
    
    async def get_prediction_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get insights from recent predictions"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        recent_predictions = [p for p in self.prediction_history if p.created_at >= cutoff_time]
        
        if not recent_predictions:
            return {'message': 'No recent predictions available'}
        
        # Analyze predictions by type
        predictions_by_type = defaultdict(list)
        for prediction in recent_predictions:
            predictions_by_type[prediction.prediction_type].append(prediction)
        
        insights = {
            'total_predictions': len(recent_predictions),
            'time_period_days': days,
            'predictions_by_type': {},
            'confidence_distribution': defaultdict(int),
            'model_performance': {},
            'trends': {}
        }
        
        for pred_type, predictions in predictions_by_type.items():
            # Calculate statistics for each prediction type
            confidence_scores = [p.confidence_score for p in predictions]
            
            insights['predictions_by_type'][pred_type.value] = {
                'count': len(predictions),
                'avg_confidence': statistics.mean(confidence_scores),
                'high_confidence_ratio': len([p for p in predictions if p.confidence == PredictionConfidence.HIGH]) / len(predictions)
            }
        
        # Overall confidence distribution
        for prediction in recent_predictions:
            insights['confidence_distribution'][prediction.confidence.value] += 1
        
        return insights
    
    async def collect_training_data(self):
        """Collect current system data for model training"""
        current_time = datetime.utcnow()
        
        # Collect usage data
        usage_summary = usage_analytics.get_usage_summary(1)  # Last hour
        usage_data = {
            'timestamp': current_time.isoformat(),
            'active_users': len(usage_analytics.active_sessions),
            'total_requests': usage_summary['summary'].get('total_events', 0),
            'error_rate': usage_summary['summary'].get('error_rate_percent', 0),
            'avg_response_time': usage_summary['summary'].get('avg_response_time_ms', 0),
            'target_value': usage_summary['summary'].get('total_events', 0)  # For supervised learning
        }
        
        self.add_training_data(PredictionType.USAGE_FORECAST, usage_data)
        
        # Collect system health data for anomaly detection
        health = performance_monitor.get_system_health()
        current_stats = health.get('current_stats', {})
        
        anomaly_data = {
            'timestamp': current_time.isoformat(),
            'cpu_usage': current_stats.get('cpu_percent', 0),
            'memory_usage': current_stats.get('memory_percent', 0),
            'active_connections': current_stats.get('active_connections', 0),
            'request_rate': usage_summary['summary'].get('total_events', 0),
            'error_rate': usage_summary['summary'].get('error_rate_percent', 0),
            'response_time': usage_summary['summary'].get('avg_response_time_ms', 0)
        }
        
        self.add_training_data(PredictionType.ANOMALY_DETECTION, anomaly_data)
    
    async def auto_retrain_models(self):
        """Automatically retrain models based on schedule"""
        current_time = datetime.utcnow()
        
        for prediction_type, model in self.models.items():
            if (model.last_trained is None or 
                current_time - model.last_trained > timedelta(hours=self.retrain_interval_hours)):
                
                print(f"Auto-retraining {prediction_type.value} model...")
                performance = await self.train_model(prediction_type)
                
                if performance:
                    print(f"Model {prediction_type.value} retrained with R² score: {performance.r2:.3f}")
    
    async def start_background_tasks(self):
        """Start background data collection and model training"""
        while True:
            try:
                # Collect training data every 15 minutes
                await self.collect_training_data()
                
                # Check for model retraining every hour
                await self.auto_retrain_models()
                
                await asyncio.sleep(900)  # 15 minutes
                
            except Exception as e:
                print(f"Error in predictive analytics background tasks: {e}")
                await asyncio.sleep(900)

# Global predictive analytics instance
predictive_analytics = PredictiveAnalytics()

# Convenience functions
async def forecast_usage(hours_ahead: int = 24) -> Optional[PredictionResult]:
    """Forecast usage for specified hours ahead"""
    target_time = datetime.utcnow() + timedelta(hours=hours_ahead)
    return await predictive_analytics.predict_usage_forecast(target_time)

async def detect_system_anomalies() -> Optional[PredictionResult]:
    """Detect current system anomalies"""
    return await predictive_analytics.detect_anomalies()

async def plan_capacity(hours_ahead: int = 12) -> Dict[str, Any]:
    """Plan capacity for specified hours ahead"""
    return await predictive_analytics.predict_resource_demand(hours_ahead)

async def get_prediction_insights(days: int = 7) -> Dict[str, Any]:
    """Get prediction insights"""
    return await predictive_analytics.get_prediction_insights(days)