# File: backend/intelligence/usage_prediction.py
"""
Platform Usage Prediction System

Advanced AI-powered usage forecasting with multi-dimensional analysis,
seasonal pattern recognition, and demand spike prediction.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import statistics

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..analytics.predictive_analytics import predictive_analytics
from ..monitoring.metrics.performance import performance_monitor
from ..analytics.usage_analytics import usage_analytics

logger = logging.getLogger(__name__)

class UsagePattern(Enum):
    """Different usage patterns that can be detected"""
    STEADY = "steady"
    GROWING = "growing"
    DECLINING = "declining"
    SEASONAL = "seasonal"
    SPIKY = "spiky"
    CYCLICAL = "cyclical"

class DemandLevel(Enum):
    """Demand level classifications"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UsageForecast:
    """Container for usage forecast results"""
    forecast_id: str
    timestamp: datetime
    prediction_horizon: timedelta
    predicted_usage: Dict[str, float]
    confidence_score: float
    usage_pattern: UsagePattern
    demand_level: DemandLevel
    seasonal_factors: Dict[str, float]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['prediction_horizon_hours'] = self.prediction_horizon.total_seconds() / 3600
        data['usage_pattern'] = self.usage_pattern.value
        data['demand_level'] = self.demand_level.value
        return data

@dataclass
class UsageMetrics:
    """Real-time usage metrics"""
    timestamp: datetime
    active_users: int
    total_requests: int
    requests_per_minute: float
    unique_sessions: int
    avg_session_duration: float
    peak_concurrent_users: int
    resource_utilization: Dict[str, float]
    agent_usage: Dict[str, int]
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class PlatformUsagePredictor:
    """Advanced platform usage prediction system"""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.usage_history = deque(maxlen=history_size)
        self.pattern_models = {}
        self.scalers = {}
        self.last_model_update = None
        
        # Prediction parameters
        self.prediction_horizons = [
            timedelta(hours=1),
            timedelta(hours=6),
            timedelta(hours=24),
            timedelta(days=7),
            timedelta(days=30)
        ]
        
        # Pattern detection parameters
        self.pattern_detection_window = timedelta(days=7)
        self.seasonal_periods = {
            'hourly': 24,      # Daily cycle
            'daily': 7,        # Weekly cycle
            'weekly': 4,       # Monthly cycle
            'monthly': 12      # Yearly cycle
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Platform Usage Predictor initialized")
    
    def _initialize_models(self):
        """Initialize ML models for different prediction horizons"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, using statistical methods")
            return
        
        try:
            # Short-term prediction (1-6 hours)
            self.pattern_models['short_term'] = {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'scaler': StandardScaler(),
                'trained': False
            }
            
            # Medium-term prediction (1-7 days)
            self.pattern_models['medium_term'] = {
                'model': RandomForestRegressor(
                    n_estimators=150,
                    max_depth=10,
                    random_state=42
                ),
                'scaler': StandardScaler(),
                'trained': False
            }
            
            # Long-term prediction (weeks-months)
            self.pattern_models['long_term'] = {
                'model': LinearRegression(),
                'scaler': StandardScaler(),
                'trained': False
            }
            
            logger.info("Usage prediction models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def collect_current_usage_metrics(self) -> UsageMetrics:
        """Collect current platform usage metrics"""
        try:
            # Get system health and performance data
            system_health = performance_monitor.get_system_health()
            current_stats = system_health.get('current_stats', {})
            
            # Get usage analytics data
            usage_summary = usage_analytics.get_usage_summary(1)  # Last hour
            summary_data = usage_summary.get('summary', {})
            
            # Get agent statistics
            agent_stats = performance_monitor.get_agent_statistics()
            
            # Calculate derived metrics
            total_requests = summary_data.get('total_events', 0)
            requests_per_minute = total_requests / 60.0 if total_requests > 0 else 0
            
            # Agent usage breakdown
            agent_usage = {}
            if isinstance(agent_stats, dict):
                for agent_id, stats in agent_stats.items():
                    if isinstance(stats, dict):
                        agent_usage[agent_id] = stats.get('total_requests', 0)
            
            metrics = UsageMetrics(
                timestamp=datetime.utcnow(),
                active_users=len(usage_analytics.active_sessions),
                total_requests=total_requests,
                requests_per_minute=requests_per_minute,
                unique_sessions=len(usage_analytics.active_sessions),
                avg_session_duration=summary_data.get('avg_session_duration_minutes', 0),
                peak_concurrent_users=summary_data.get('peak_concurrent_users', 0),
                resource_utilization={
                    'cpu_percent': current_stats.get('cpu_percent', 0),
                    'memory_percent': current_stats.get('memory_percent', 0),
                    'active_connections': current_stats.get('active_connections', 0)
                },
                agent_usage=agent_usage,
                error_rate=summary_data.get('error_rate_percent', 0)
            )
            
            # Store in history
            self.usage_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting usage metrics: {e}")
            # Return default metrics
            return UsageMetrics(
                timestamp=datetime.utcnow(),
                active_users=0,
                total_requests=0,
                requests_per_minute=0,
                unique_sessions=0,
                avg_session_duration=0,
                peak_concurrent_users=0,
                resource_utilization={'cpu_percent': 0, 'memory_percent': 0, 'active_connections': 0},
                agent_usage={},
                error_rate=0
            )
    
    def _detect_usage_pattern(self, metrics_history: List[UsageMetrics]) -> UsagePattern:
        """Detect usage patterns from historical data"""
        if len(metrics_history) < 10:
            return UsagePattern.STEADY
        
        try:
            # Extract request rates for pattern analysis
            request_rates = [m.requests_per_minute for m in metrics_history[-24:]]  # Last 24 data points
            
            if len(request_rates) < 5:
                return UsagePattern.STEADY
            
            # Calculate trend
            x = np.arange(len(request_rates))
            coeffs = np.polyfit(x, request_rates, 1)
            trend_slope = coeffs[0]
            
            # Calculate variability
            mean_rate = np.mean(request_rates)
            std_rate = np.std(request_rates)
            cv = std_rate / mean_rate if mean_rate > 0 else 0
            
            # Detect patterns
            if abs(trend_slope) < 0.1 and cv < 0.3:
                return UsagePattern.STEADY
            elif trend_slope > 0.5:
                return UsagePattern.GROWING
            elif trend_slope < -0.5:
                return UsagePattern.DECLINING
            elif cv > 0.8:
                return UsagePattern.SPIKY
            else:
                # Check for seasonal patterns
                if self._detect_seasonality(request_rates):
                    return UsagePattern.SEASONAL
                else:
                    return UsagePattern.CYCLICAL
            
        except Exception as e:
            logger.error(f"Error detecting usage pattern: {e}")
            return UsagePattern.STEADY
    
    def _detect_seasonality(self, data: List[float]) -> bool:
        """Detect seasonal patterns in usage data"""
        if len(data) < 12:
            return False
        
        try:
            # Simple seasonality detection using autocorrelation
            # Check for daily patterns (assuming hourly data)
            if len(data) >= 24:
                daily_correlation = self._calculate_autocorrelation(data, 24)
                if daily_correlation > 0.3:
                    return True
            
            # Check for weekly patterns
            if len(data) >= 168:  # 7 days * 24 hours
                weekly_correlation = self._calculate_autocorrelation(data, 168)
                if weekly_correlation > 0.3:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            return False
    
    def _calculate_autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        if len(data) <= lag:
            return 0.0
        
        try:
            series = np.array(data)
            mean = np.mean(series)
            
            # Calculate autocorrelation
            c0 = np.mean((series - mean) ** 2)
            c_lag = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
            
            return c_lag / c0 if c0 > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating autocorrelation: {e}")
            return 0.0
    
    def _classify_demand_level(self, predicted_usage: Dict[str, float], 
                              historical_avg: Dict[str, float]) -> DemandLevel:
        """Classify demand level based on predictions"""
        try:
            # Compare predicted requests with historical average
            predicted_requests = predicted_usage.get('total_requests', 0)
            historical_requests = historical_avg.get('total_requests', 0)
            
            if historical_requests == 0:
                return DemandLevel.NORMAL
            
            demand_ratio = predicted_requests / historical_requests
            
            if demand_ratio >= 2.0:
                return DemandLevel.CRITICAL
            elif demand_ratio >= 1.5:
                return DemandLevel.HIGH
            elif demand_ratio <= 0.5:
                return DemandLevel.LOW
            else:
                return DemandLevel.NORMAL
                
        except Exception as e:
            logger.error(f"Error classifying demand level: {e}")
            return DemandLevel.NORMAL
    
    def _prepare_features(self, metrics: UsageMetrics) -> np.ndarray:
        """Prepare features for ML models"""
        timestamp = metrics.timestamp
        
        # Time-based features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        day_of_month = timestamp.day
        month = timestamp.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Usage features
        active_users = metrics.active_users
        total_requests = metrics.total_requests
        requests_per_minute = metrics.requests_per_minute
        unique_sessions = metrics.unique_sessions
        error_rate = metrics.error_rate
        
        # Resource features
        cpu_usage = metrics.resource_utilization.get('cpu_percent', 0)
        memory_usage = metrics.resource_utilization.get('memory_percent', 0)
        active_connections = metrics.resource_utilization.get('active_connections', 0)
        
        # Agent usage features (top 3 agents)
        sorted_agents = sorted(metrics.agent_usage.items(), key=lambda x: x[1], reverse=True)
        agent1_usage = sorted_agents[0][1] if len(sorted_agents) > 0 else 0
        agent2_usage = sorted_agents[1][1] if len(sorted_agents) > 1 else 0
        agent3_usage = sorted_agents[2][1] if len(sorted_agents) > 2 else 0
        
        return np.array([
            hour, day_of_week, day_of_month, month, is_weekend,
            active_users, total_requests, requests_per_minute, unique_sessions, error_rate,
            cpu_usage, memory_usage, active_connections,
            agent1_usage, agent2_usage, agent3_usage
        ])
    
    async def train_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Train prediction models with historical data"""
        if not ML_AVAILABLE:
            return {'status': 'skipped', 'reason': 'ML libraries not available'}
        
        if len(self.usage_history) < 50:
            return {'status': 'skipped', 'reason': 'Insufficient training data'}
        
        # Check if retraining is needed
        if not force_retrain and self.last_model_update:
            time_since_update = datetime.utcnow() - self.last_model_update
            if time_since_update < timedelta(hours=6):
                return {'status': 'skipped', 'reason': 'Models recently updated'}
        
        try:
            results = {}
            
            # Prepare training data
            features = []
            targets = []
            
            for i, metrics in enumerate(list(self.usage_history)[:-1]):  # Exclude last item
                next_metrics = list(self.usage_history)[i + 1]
                
                feature_vector = self._prepare_features(metrics)
                target_value = next_metrics.total_requests
                
                features.append(feature_vector)
                targets.append(target_value)
            
            if len(features) < 20:
                return {'status': 'failed', 'reason': 'Insufficient feature data'}
            
            X = np.array(features)
            y = np.array(targets)
            
            # Train each model
            for model_name, model_info in self.pattern_models.items():
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Scale features
                    X_train_scaled = model_info['scaler'].fit_transform(X_train)
                    X_test_scaled = model_info['scaler'].transform(X_test)
                    
                    # Train model
                    model_info['model'].fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model_info['model'].predict(X_test_scaled)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    model_info['trained'] = True
                    
                    results[model_name] = {
                        'mae': mae,
                        'r2_score': r2,
                        'training_samples': len(X_train),
                        'test_samples': len(X_test)
                    }
                    
                    logger.info(f"Model {model_name} trained - MAE: {mae:.2f}, R²: {r2:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training model {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            
            self.last_model_update = datetime.utcnow()
            
            return {
                'status': 'completed',
                'timestamp': self.last_model_update.isoformat(),
                'models': results
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def predict_usage(self, horizon: timedelta) -> UsageForecast:
        """Predict platform usage for given time horizon"""
        forecast_id = f"forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Get current metrics
            current_metrics = await self.collect_current_usage_metrics()
            
            # Determine which model to use based on horizon
            if horizon <= timedelta(hours=6):
                model_type = 'short_term'
            elif horizon <= timedelta(days=7):
                model_type = 'medium_term'
            else:
                model_type = 'long_term'
            
            # Get predicted usage
            predicted_usage = await self._generate_prediction(current_metrics, horizon, model_type)
            
            # Analyze patterns
            usage_pattern = self._detect_usage_pattern(list(self.usage_history)[-100:])
            
            # Calculate historical averages for comparison
            historical_avg = self._calculate_historical_averages()
            
            # Classify demand level
            demand_level = self._classify_demand_level(predicted_usage, historical_avg)
            
            # Calculate seasonal factors
            seasonal_factors = self._calculate_seasonal_factors(current_metrics.timestamp)
            
            # Generate trend analysis
            trend_analysis = self._analyze_trends()
            
            # Generate recommendations
            recommendations = self._generate_usage_recommendations(
                predicted_usage, demand_level, usage_pattern
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_prediction_confidence(model_type, predicted_usage)
            
            forecast = UsageForecast(
                forecast_id=forecast_id,
                timestamp=datetime.utcnow(),
                prediction_horizon=horizon,
                predicted_usage=predicted_usage,
                confidence_score=confidence_score,
                usage_pattern=usage_pattern,
                demand_level=demand_level,
                seasonal_factors=seasonal_factors,
                trend_analysis=trend_analysis,
                recommendations=recommendations,
                metadata={
                    'model_type': model_type,
                    'historical_data_points': len(self.usage_history),
                    'current_metrics': current_metrics.to_dict()
                }
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error predicting usage: {e}")
            # Return default forecast
            return UsageForecast(
                forecast_id=forecast_id,
                timestamp=datetime.utcnow(),
                prediction_horizon=horizon,
                predicted_usage={'total_requests': 0, 'active_users': 0},
                confidence_score=0.0,
                usage_pattern=UsagePattern.STEADY,
                demand_level=DemandLevel.NORMAL,
                seasonal_factors={},
                trend_analysis={},
                recommendations=['Unable to generate forecast due to error'],
                metadata={'error': str(e)}
            )
    
    async def _generate_prediction(self, current_metrics: UsageMetrics, 
                                  horizon: timedelta, model_type: str) -> Dict[str, float]:
        """Generate usage prediction using appropriate model"""
        try:
            if ML_AVAILABLE and model_type in self.pattern_models:
                model_info = self.pattern_models[model_type]
                
                if model_info['trained']:
                    # Prepare features for prediction
                    target_time = current_metrics.timestamp + horizon
                    
                    # Create future metrics template
                    future_metrics = UsageMetrics(
                        timestamp=target_time,
                        active_users=current_metrics.active_users,
                        total_requests=current_metrics.total_requests,
                        requests_per_minute=current_metrics.requests_per_minute,
                        unique_sessions=current_metrics.unique_sessions,
                        avg_session_duration=current_metrics.avg_session_duration,
                        peak_concurrent_users=current_metrics.peak_concurrent_users,
                        resource_utilization=current_metrics.resource_utilization,
                        agent_usage=current_metrics.agent_usage,
                        error_rate=current_metrics.error_rate
                    )
                    
                    features = self._prepare_features(future_metrics)
                    features_scaled = model_info['scaler'].transform(features.reshape(1, -1))
                    
                    predicted_requests = model_info['model'].predict(features_scaled)[0]
                    
                    # Estimate other metrics based on predicted requests
                    request_ratio = predicted_requests / max(current_metrics.total_requests, 1)
                    
                    return {
                        'total_requests': max(0, predicted_requests),
                        'active_users': max(0, current_metrics.active_users * request_ratio),
                        'requests_per_minute': max(0, predicted_requests / 60),
                        'unique_sessions': max(0, current_metrics.unique_sessions * request_ratio),
                        'estimated_cpu_usage': min(100, current_metrics.resource_utilization.get('cpu_percent', 0) * request_ratio),
                        'estimated_memory_usage': min(100, current_metrics.resource_utilization.get('memory_percent', 0) * request_ratio)
                    }
            
            # Fallback to statistical prediction
            return self._statistical_prediction(current_metrics, horizon)
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return self._statistical_prediction(current_metrics, horizon)
    
    def _statistical_prediction(self, current_metrics: UsageMetrics, 
                               horizon: timedelta) -> Dict[str, float]:
        """Fallback statistical prediction when ML models aren't available"""
        try:
            # Use historical averages and trends
            if len(self.usage_history) < 5:
                # No historical data, use current values
                return {
                    'total_requests': current_metrics.total_requests,
                    'active_users': current_metrics.active_users,
                    'requests_per_minute': current_metrics.requests_per_minute,
                    'unique_sessions': current_metrics.unique_sessions,
                    'estimated_cpu_usage': current_metrics.resource_utilization.get('cpu_percent', 0),
                    'estimated_memory_usage': current_metrics.resource_utilization.get('memory_percent', 0)
                }
            
            # Calculate trends from recent data
            recent_metrics = list(self.usage_history)[-min(24, len(self.usage_history)):]
            
            if len(recent_metrics) >= 2:
                # Simple linear trend
                request_values = [m.total_requests for m in recent_metrics]
                user_values = [m.active_users for m in recent_metrics]
                
                request_trend = (request_values[-1] - request_values[0]) / len(request_values)
                user_trend = (user_values[-1] - user_values[0]) / len(user_values)
                
                # Project trends forward
                hours_ahead = horizon.total_seconds() / 3600
                
                predicted_requests = max(0, current_metrics.total_requests + request_trend * hours_ahead)
                predicted_users = max(0, current_metrics.active_users + user_trend * hours_ahead)
                
                return {
                    'total_requests': predicted_requests,
                    'active_users': predicted_users,
                    'requests_per_minute': predicted_requests / 60,
                    'unique_sessions': predicted_users * 0.8,  # Estimate
                    'estimated_cpu_usage': min(100, predicted_requests * 0.01),  # Simple estimation
                    'estimated_memory_usage': min(100, predicted_requests * 0.005)  # Simple estimation
                }
            
            # Fallback to current values
            return {
                'total_requests': current_metrics.total_requests,
                'active_users': current_metrics.active_users,
                'requests_per_minute': current_metrics.requests_per_minute,
                'unique_sessions': current_metrics.unique_sessions,
                'estimated_cpu_usage': current_metrics.resource_utilization.get('cpu_percent', 0),
                'estimated_memory_usage': current_metrics.resource_utilization.get('memory_percent', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical prediction: {e}")
            return {
                'total_requests': 0,
                'active_users': 0,
                'requests_per_minute': 0,
                'unique_sessions': 0,
                'estimated_cpu_usage': 0,
                'estimated_memory_usage': 0
            }
    
    def _calculate_historical_averages(self) -> Dict[str, float]:
        """Calculate historical averages for comparison"""
        if not self.usage_history:
            return {'total_requests': 0, 'active_users': 0}
        
        try:
            recent_metrics = list(self.usage_history)[-min(168, len(self.usage_history)):]  # Last week
            
            return {
                'total_requests': statistics.mean(m.total_requests for m in recent_metrics),
                'active_users': statistics.mean(m.active_users for m in recent_metrics),
                'requests_per_minute': statistics.mean(m.requests_per_minute for m in recent_metrics),
                'unique_sessions': statistics.mean(m.unique_sessions for m in recent_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error calculating historical averages: {e}")
            return {'total_requests': 0, 'active_users': 0}
    
    def _calculate_seasonal_factors(self, timestamp: datetime) -> Dict[str, float]:
        """Calculate seasonal adjustment factors"""
        factors = {}
        
        try:
            # Hour of day factor
            hour = timestamp.hour
            if 6 <= hour <= 10:  # Morning peak
                factors['hour_factor'] = 1.3
            elif 14 <= hour <= 18:  # Afternoon peak
                factors['hour_factor'] = 1.2
            elif 22 <= hour or hour <= 6:  # Night low
                factors['hour_factor'] = 0.7
            else:
                factors['hour_factor'] = 1.0
            
            # Day of week factor
            day_of_week = timestamp.weekday()
            if day_of_week < 5:  # Weekday
                factors['day_factor'] = 1.1
            else:  # Weekend
                factors['day_factor'] = 0.8
            
            # Month factor (simple seasonal adjustment)
            month = timestamp.month
            if month in [12, 1, 2]:  # Winter
                factors['month_factor'] = 0.9
            elif month in [6, 7, 8]:  # Summer
                factors['month_factor'] = 1.1
            else:
                factors['month_factor'] = 1.0
            
        except Exception as e:
            logger.error(f"Error calculating seasonal factors: {e}")
            factors = {'hour_factor': 1.0, 'day_factor': 1.0, 'month_factor': 1.0}
        
        return factors
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze usage trends from historical data"""
        if len(self.usage_history) < 10:
            return {'trend': 'insufficient_data'}
        
        try:
            recent_metrics = list(self.usage_history)[-min(168, len(self.usage_history)):]
            
            # Request trend
            request_values = [m.total_requests for m in recent_metrics]
            x = np.arange(len(request_values))
            request_coeffs = np.polyfit(x, request_values, 1)
            request_trend = request_coeffs[0]
            
            # User trend
            user_values = [m.active_users for m in recent_metrics]
            user_coeffs = np.polyfit(x, user_values, 1)
            user_trend = user_coeffs[0]
            
            # Determine trend direction
            if request_trend > 0.5:
                trend_direction = 'increasing'
            elif request_trend < -0.5:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            return {
                'trend_direction': trend_direction,
                'request_trend_per_hour': request_trend,
                'user_trend_per_hour': user_trend,
                'analysis_period_hours': len(recent_metrics),
                'volatility': np.std(request_values) / np.mean(request_values) if np.mean(request_values) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {'trend': 'analysis_error', 'error': str(e)}
    
    def _generate_usage_recommendations(self, predicted_usage: Dict[str, float],
                                       demand_level: DemandLevel,
                                       usage_pattern: UsagePattern) -> List[str]:
        """Generate recommendations based on usage predictions"""
        recommendations = []
        
        try:
            predicted_requests = predicted_usage.get('total_requests', 0)
            predicted_cpu = predicted_usage.get('estimated_cpu_usage', 0)
            predicted_memory = predicted_usage.get('estimated_memory_usage', 0)
            
            # Demand-based recommendations
            if demand_level == DemandLevel.CRITICAL:
                recommendations.append("Critical demand spike predicted - prepare for immediate scaling")
                recommendations.append("Enable auto-scaling policies and monitor system resources closely")
                recommendations.append("Consider implementing request throttling if needed")
            elif demand_level == DemandLevel.HIGH:
                recommendations.append("High demand period expected - consider preemptive resource scaling")
                recommendations.append("Review cache warming strategies for peak performance")
            elif demand_level == DemandLevel.LOW:
                recommendations.append("Low demand period - opportunity for maintenance and cost optimization")
                recommendations.append("Consider scheduled maintenance tasks during low usage")
            
            # Pattern-based recommendations
            if usage_pattern == UsagePattern.SPIKY:
                recommendations.append("Spiky usage pattern detected - implement burst handling capabilities")
                recommendations.append("Consider queue-based processing for peak load management")
            elif usage_pattern == UsagePattern.SEASONAL:
                recommendations.append("Seasonal usage pattern - implement predictive scaling based on historical patterns")
            elif usage_pattern == UsagePattern.GROWING:
                recommendations.append("Growing usage trend - plan for long-term capacity expansion")
            elif usage_pattern == UsagePattern.DECLINING:
                recommendations.append("Declining usage trend - review cost optimization opportunities")
            
            # Resource-based recommendations
            if predicted_cpu > 80:
                recommendations.append(f"High CPU usage predicted ({predicted_cpu:.1f}%) - consider CPU scaling")
            
            if predicted_memory > 85:
                recommendations.append(f"High memory usage predicted ({predicted_memory:.1f}%) - consider memory optimization")
            
            # Request volume recommendations
            if predicted_requests > 10000:
                recommendations.append("High request volume predicted - ensure load balancing is optimized")
                recommendations.append("Review API rate limiting and caching strategies")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Usage levels appear normal - continue monitoring")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - manual review suggested")
        
        return recommendations
    
    def _calculate_prediction_confidence(self, model_type: str, 
                                       predicted_usage: Dict[str, float]) -> float:
        """Calculate confidence score for predictions"""
        try:
            base_confidence = 0.7  # Base confidence level
            
            # Adjust based on model type and training status
            if ML_AVAILABLE and model_type in self.pattern_models:
                model_info = self.pattern_models[model_type]
                if model_info['trained']:
                    base_confidence = 0.85
            
            # Adjust based on data availability
            data_factor = min(1.0, len(self.usage_history) / 100)  # More data = higher confidence
            
            # Adjust based on prediction reasonableness
            predicted_requests = predicted_usage.get('total_requests', 0)
            if predicted_requests < 0 or predicted_requests > 1000000:  # Unreasonable values
                base_confidence *= 0.5
            
            final_confidence = base_confidence * data_factor
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def get_usage_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive usage insights and analytics"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            recent_metrics = [m for m in self.usage_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {'message': 'No recent usage data available'}
            
            # Calculate statistics
            request_values = [m.total_requests for m in recent_metrics]
            user_values = [m.active_users for m in recent_metrics]
            
            insights = {
                'analysis_period_days': days,
                'total_data_points': len(recent_metrics),
                'usage_statistics': {
                    'avg_requests_per_hour': statistics.mean(request_values),
                    'max_requests_per_hour': max(request_values),
                    'min_requests_per_hour': min(request_values),
                    'avg_active_users': statistics.mean(user_values),
                    'peak_concurrent_users': max(m.peak_concurrent_users for m in recent_metrics)
                },
                'patterns': {
                    'detected_pattern': self._detect_usage_pattern(recent_metrics).value,
                    'trend_analysis': self._analyze_trends()
                },
                'peak_usage_hours': self._identify_peak_hours(recent_metrics),
                'agent_usage_distribution': self._analyze_agent_usage(recent_metrics),
                'resource_correlation': self._analyze_resource_correlation(recent_metrics)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting usage insights: {e}")
            return {'error': str(e)}
    
    def _identify_peak_hours(self, metrics: List[UsageMetrics]) -> List[int]:
        """Identify peak usage hours from historical data"""
        try:
            hourly_usage = defaultdict(list)
            
            for metric in metrics:
                hour = metric.timestamp.hour
                hourly_usage[hour].append(metric.total_requests)
            
            # Calculate average usage per hour
            hourly_averages = {}
            for hour, requests in hourly_usage.items():
                hourly_averages[hour] = statistics.mean(requests)
            
            # Find top 25% of hours
            sorted_hours = sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)
            peak_count = max(1, len(sorted_hours) // 4)
            
            return [hour for hour, _ in sorted_hours[:peak_count]]
            
        except Exception as e:
            logger.error(f"Error identifying peak hours: {e}")
            return []
    
    def _analyze_agent_usage(self, metrics: List[UsageMetrics]) -> Dict[str, Any]:
        """Analyze agent usage distribution"""
        try:
            agent_totals = defaultdict(int)
            
            for metric in metrics:
                for agent_id, usage in metric.agent_usage.items():
                    agent_totals[agent_id] += usage
            
            total_usage = sum(agent_totals.values())
            
            if total_usage == 0:
                return {'message': 'No agent usage data available'}
            
            # Calculate percentages
            agent_percentages = {}
            for agent_id, usage in agent_totals.items():
                agent_percentages[agent_id] = (usage / total_usage) * 100
            
            # Sort by usage
            sorted_agents = sorted(agent_percentages.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'total_agent_requests': total_usage,
                'agent_distribution': dict(sorted_agents),
                'top_agents': dict(sorted_agents[:5]),
                'agent_count': len(agent_totals)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing agent usage: {e}")
            return {'error': str(e)}
    
    def _analyze_resource_correlation(self, metrics: List[UsageMetrics]) -> Dict[str, float]:
        """Analyze correlation between usage and resource utilization"""
        try:
            if len(metrics) < 5:
                return {'message': 'Insufficient data for correlation analysis'}
            
            request_values = [m.total_requests for m in metrics]
            cpu_values = [m.resource_utilization.get('cpu_percent', 0) for m in metrics]
            memory_values = [m.resource_utilization.get('memory_percent', 0) for m in metrics]
            
            # Calculate correlations
            correlations = {}
            
            if len(set(request_values)) > 1 and len(set(cpu_values)) > 1:
                correlations['requests_cpu_correlation'] = np.corrcoef(request_values, cpu_values)[0, 1]
            
            if len(set(request_values)) > 1 and len(set(memory_values)) > 1:
                correlations['requests_memory_correlation'] = np.corrcoef(request_values, memory_values)[0, 1]
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing resource correlation: {e}")
            return {'error': str(e)}

# Global usage predictor instance
usage_predictor = PlatformUsagePredictor()

# Convenience functions
async def predict_platform_usage(hours_ahead: int = 24) -> UsageForecast:
    """Predict platform usage for specified hours ahead"""
    horizon = timedelta(hours=hours_ahead)
    return await usage_predictor.predict_usage(horizon)

async def get_current_usage_metrics() -> UsageMetrics:
    """Get current platform usage metrics"""
    return await usage_predictor.collect_current_usage_metrics()

async def train_usage_models(force_retrain: bool = False) -> Dict[str, Any]:
    """Train usage prediction models"""
    return await usage_predictor.train_models(force_retrain)

async def get_usage_insights(days: int = 7) -> Dict[str, Any]:
    """Get usage insights and analytics"""
    return await usage_predictor.get_usage_insights(days)

__all__ = [
    'PlatformUsagePredictor',
    'usage_predictor',
    'UsageForecast',
    'UsageMetrics',
    'UsagePattern',
    'DemandLevel',
    'predict_platform_usage',
    'get_current_usage_metrics',
    'train_usage_models',
    'get_usage_insights'
]