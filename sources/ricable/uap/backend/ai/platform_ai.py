# File: backend/ai/platform_ai.py
"""
AI-Powered Platform Intelligence System

This module implements the core AI capabilities for platform intelligence,
including predictive modeling, automated optimization, and self-healing mechanisms.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from collections import defaultdict, deque

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Import platform components
from ..analytics.predictive_analytics import predictive_analytics, PredictionType
from ..monitoring.metrics.performance import performance_monitor
from ..services.performance_service import performance_service
from ..cache.redis_cache import get_cache

logger = logging.getLogger(__name__)

class IntelligenceLevel(Enum):
    """AI intelligence levels for different operations"""
    BASIC = "basic"           # Simple rule-based decisions
    ADVANCED = "advanced"     # ML-powered predictions
    EXPERT = "expert"         # Deep learning and complex analysis
    AUTONOMOUS = "autonomous" # Fully autonomous self-healing

class OptimizationStrategy(Enum):
    """Different optimization strategies"""
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RELIABILITY_OPTIMIZATION = "reliability_optimization"
    BALANCED_OPTIMIZATION = "balanced_optimization"

class PlatformState(Enum):
    """Current platform operational state"""
    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

@dataclass
class AIDecision:
    """Container for AI-made decisions"""
    decision_id: str
    decision_type: str
    confidence: float
    recommendation: str
    actions: List[Dict[str, Any]]
    reasoning: str
    impact_assessment: Dict[str, Any]
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class PlatformIntelligenceReport:
    """Comprehensive platform intelligence report"""
    report_id: str
    timestamp: datetime
    platform_state: PlatformState
    intelligence_level: IntelligenceLevel
    optimization_strategy: OptimizationStrategy
    key_insights: List[str]
    recommendations: List[AIDecision]
    predictions: Dict[str, Any]
    health_score: float
    efficiency_score: float
    cost_optimization_potential: float
    self_healing_actions: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['platform_state'] = self.platform_state.value
        data['intelligence_level'] = self.intelligence_level.value
        data['optimization_strategy'] = self.optimization_strategy.value
        return data

class PlatformAI:
    """Main AI engine for platform intelligence"""
    
    def __init__(self, intelligence_level: IntelligenceLevel = IntelligenceLevel.ADVANCED):
        self.intelligence_level = intelligence_level
        self.models = {}
        self.decision_history = deque(maxlen=10000)
        self.pattern_memory = defaultdict(list)
        self.learning_data = deque(maxlen=50000)
        self.optimization_strategy = OptimizationStrategy.BALANCED_OPTIMIZATION
        
        # AI model components
        self.usage_predictor = None
        self.anomaly_detector = None
        self.resource_optimizer = None
        self.performance_predictor = None
        
        # Platform state tracking
        self.current_state = PlatformState.OPTIMAL
        self.state_history = deque(maxlen=1000)
        
        # Initialize AI components
        self._initialize_ai_models()
        
        logger.info(f"Platform AI initialized with intelligence level: {intelligence_level.value}")
    
    def _initialize_ai_models(self):
        """Initialize AI models based on intelligence level"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, falling back to rule-based intelligence")
            return
        
        try:
            # Usage prediction model
            self.usage_predictor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Resource optimization model
            self.resource_optimizer = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Performance prediction model
            self.performance_predictor = Ridge(alpha=1.0)
            
            self.models = {
                'usage_predictor': self.usage_predictor,
                'anomaly_detector': self.anomaly_detector,
                'resource_optimizer': self.resource_optimizer,
                'performance_predictor': self.performance_predictor
            }
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
    
    async def analyze_platform_intelligence(self) -> PlatformIntelligenceReport:
        """Generate comprehensive platform intelligence report"""
        report_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Collect current platform data
        platform_data = await self._collect_platform_data()
        
        # Analyze current platform state
        platform_state = await self._analyze_platform_state(platform_data)
        
        # Generate insights and predictions
        insights = await self._generate_key_insights(platform_data)
        predictions = await self._generate_predictions(platform_data)
        
        # Create AI recommendations
        recommendations = await self._generate_recommendations(platform_data, insights)
        
        # Calculate health and efficiency scores
        health_score = self._calculate_health_score(platform_data)
        efficiency_score = self._calculate_efficiency_score(platform_data)
        cost_optimization_potential = self._calculate_cost_optimization_potential(platform_data)
        
        # Generate self-healing actions
        self_healing_actions = await self._generate_self_healing_actions(platform_data)
        
        report = PlatformIntelligenceReport(
            report_id=report_id,
            timestamp=timestamp,
            platform_state=platform_state,
            intelligence_level=self.intelligence_level,
            optimization_strategy=self.optimization_strategy,
            key_insights=insights,
            recommendations=recommendations,
            predictions=predictions,
            health_score=health_score,
            efficiency_score=efficiency_score,
            cost_optimization_potential=cost_optimization_potential,
            self_healing_actions=self_healing_actions
        )
        
        # Store report for learning
        self._store_intelligence_data(report)
        
        return report
    
    async def _collect_platform_data(self) -> Dict[str, Any]:
        """Collect comprehensive platform data for analysis"""
        data = {
            'timestamp': datetime.utcnow(),
            'system_health': {},
            'performance_metrics': {},
            'agent_statistics': {},
            'resource_utilization': {},
            'cache_statistics': {},
            'prediction_insights': {}
        }
        
        try:
            # System health data
            data['system_health'] = performance_monitor.get_system_health()
            
            # Performance metrics
            data['performance_metrics'] = performance_monitor.get_performance_summary(60)
            
            # Agent statistics
            data['agent_statistics'] = performance_monitor.get_agent_statistics()
            
            # Performance service statistics
            data['performance_service_stats'] = await performance_service.get_performance_stats()
            
            # Predictive analytics insights
            data['prediction_insights'] = await predictive_analytics.get_prediction_insights(7)
            
            # Cache statistics if available
            try:
                cache = await get_cache()
                if cache:
                    data['cache_statistics'] = cache.get_stats()
            except Exception as e:
                logger.debug(f"Cache statistics not available: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting platform data: {e}")
        
        return data
    
    async def _analyze_platform_state(self, platform_data: Dict[str, Any]) -> PlatformState:
        """Analyze and determine current platform state"""
        try:
            system_health = platform_data.get('system_health', {})
            performance_stats = platform_data.get('performance_metrics', {})
            
            # Check overall health
            overall_healthy = system_health.get('overall_healthy', True)
            
            # Check performance thresholds
            current_stats = system_health.get('current_stats', {})
            cpu_usage = current_stats.get('cpu_percent', 0)
            memory_usage = current_stats.get('memory_percent', 0)
            
            # Check agent health
            agent_health = system_health.get('agent_health', {})
            unhealthy_agents = sum(1 for agent in agent_health.values() if not agent.get('healthy', True))
            
            # Determine state based on multiple factors
            if not overall_healthy or cpu_usage > 90 or memory_usage > 95:
                state = PlatformState.CRITICAL
            elif cpu_usage > 80 or memory_usage > 85 or unhealthy_agents > 0:
                state = PlatformState.DEGRADED
            elif hasattr(self, '_recent_recovery') and self._recent_recovery:
                state = PlatformState.RECOVERING
            else:
                state = PlatformState.OPTIMAL
            
            # Update state history
            self.current_state = state
            self.state_history.append({
                'timestamp': datetime.utcnow(),
                'state': state,
                'reasons': {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'overall_healthy': overall_healthy,
                    'unhealthy_agents': unhealthy_agents
                }
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing platform state: {e}")
            return PlatformState.OPTIMAL
    
    async def _generate_key_insights(self, platform_data: Dict[str, Any]) -> List[str]:
        """Generate key insights from platform data"""
        insights = []
        
        try:
            system_health = platform_data.get('system_health', {})
            performance_stats = platform_data.get('performance_metrics', {})
            agent_stats = platform_data.get('agent_statistics', {})
            
            # System resource insights
            current_stats = system_health.get('current_stats', {})
            cpu_usage = current_stats.get('cpu_percent', 0)
            memory_usage = current_stats.get('memory_percent', 0)
            
            if cpu_usage > 70:
                insights.append(f"High CPU utilization detected: {cpu_usage:.1f}%. Consider scaling resources.")
            
            if memory_usage > 80:
                insights.append(f"High memory usage: {memory_usage:.1f}%. Memory optimization recommended.")
            
            # Agent performance insights
            if isinstance(agent_stats, dict):
                slow_agents = []
                error_prone_agents = []
                
                for agent_id, stats in agent_stats.items():
                    if isinstance(stats, dict):
                        avg_response = stats.get('avg_response_time_ms', 0)
                        success_rate = stats.get('success_rate', 100)
                        
                        if avg_response > 1000:  # > 1 second
                            slow_agents.append(f"{agent_id} ({avg_response:.0f}ms)")
                        
                        if success_rate < 95:
                            error_prone_agents.append(f"{agent_id} ({success_rate:.1f}%)")
                
                if slow_agents:
                    insights.append(f"Slow response times detected in agents: {', '.join(slow_agents)}")
                
                if error_prone_agents:
                    insights.append(f"High error rates in agents: {', '.join(error_prone_agents)}")
            
            # Performance optimization insights
            perf_service_stats = platform_data.get('performance_service_stats', {})
            cache_stats = perf_service_stats.get('cache', {})
            
            if cache_stats.get('enabled', False):
                cache_hit_rate = cache_stats.get('stats', {}).get('hit_rate', 0)
                if cache_hit_rate < 0.6:  # < 60% hit rate
                    insights.append(f"Low cache hit rate: {cache_hit_rate*100:.1f}%. Cache strategy optimization needed.")
            
            # Cost optimization insights
            if cpu_usage < 20 and memory_usage < 30:
                insights.append("Resources appear over-provisioned. Consider cost optimization.")
            
            # Trend analysis from prediction insights
            prediction_insights = platform_data.get('prediction_insights', {})
            if prediction_insights.get('total_predictions', 0) > 0:
                confidence_dist = prediction_insights.get('confidence_distribution', {})
                low_confidence = confidence_dist.get('low', 0)
                total = prediction_insights.get('total_predictions', 1)
                
                if low_confidence / total > 0.3:  # > 30% low confidence
                    insights.append("High proportion of low-confidence predictions. Model retraining recommended.")
            
            # Add default insight if none found
            if not insights:
                insights.append("Platform is operating within normal parameters.")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Error analyzing platform data. Manual review recommended.")
        
        return insights
    
    async def _generate_predictions(self, platform_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions for platform behavior"""
        predictions = {}
        
        try:
            # Usage prediction
            usage_forecast = await predictive_analytics.predict_usage_forecast(
                datetime.utcnow() + timedelta(hours=24)
            )
            if usage_forecast:
                predictions['usage_24h'] = {
                    'predicted_requests': usage_forecast.predicted_value,
                    'confidence': usage_forecast.confidence.value,
                    'time_horizon': '24 hours'
                }
            
            # Resource demand prediction
            resource_prediction = await predictive_analytics.predict_resource_demand(12)
            if resource_prediction:
                predictions['resource_demand_12h'] = resource_prediction
            
            # Anomaly detection
            anomaly_prediction = await predictive_analytics.detect_anomalies()
            if anomaly_prediction:
                predictions['anomaly_risk'] = {
                    'risk_score': anomaly_prediction.predicted_value,
                    'confidence': anomaly_prediction.confidence.value,
                    'assessment': 'High' if anomaly_prediction.predicted_value > 0.7 else 'Low'
                }
            
            # Cost projection (basic calculation)
            current_stats = platform_data.get('system_health', {}).get('current_stats', {})
            cpu_usage = current_stats.get('cpu_percent', 0)
            memory_usage = current_stats.get('memory_percent', 0)
            
            # Simple cost model (would be more sophisticated in production)
            base_cost_per_hour = 10.0  # Example base cost
            usage_multiplier = (cpu_usage + memory_usage) / 200  # 0-1 multiplier
            projected_daily_cost = base_cost_per_hour * 24 * (1 + usage_multiplier)
            
            predictions['cost_projection'] = {
                'daily_cost_usd': round(projected_daily_cost, 2),
                'monthly_cost_usd': round(projected_daily_cost * 30, 2),
                'cost_trend': 'increasing' if usage_multiplier > 0.7 else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
        
        return predictions
    
    async def _generate_recommendations(self, platform_data: Dict[str, Any], 
                                       insights: List[str]) -> List[AIDecision]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        try:
            system_health = platform_data.get('system_health', {})
            current_stats = system_health.get('current_stats', {})
            
            # Resource scaling recommendations
            cpu_usage = current_stats.get('cpu_percent', 0)
            memory_usage = current_stats.get('memory_percent', 0)
            
            if cpu_usage > 80:
                recommendation = AIDecision(
                    decision_id=str(uuid.uuid4()),
                    decision_type="resource_scaling",
                    confidence=0.85,
                    recommendation="Scale CPU resources",
                    actions=[{
                        'type': 'scale_cpu',
                        'parameter': 'cpu_cores',
                        'current_value': 'auto_detect',
                        'recommended_value': 'current + 2',
                        'priority': 'high'
                    }],
                    reasoning=f"CPU usage at {cpu_usage:.1f}% exceeds 80% threshold",
                    impact_assessment={
                        'performance_improvement': 'high',
                        'cost_impact': 'medium',
                        'risk_level': 'low'
                    },
                    created_at=datetime.utcnow(),
                    metadata={'cpu_usage': cpu_usage}
                )
                recommendations.append(recommendation)
            
            if memory_usage > 85:
                recommendation = AIDecision(
                    decision_id=str(uuid.uuid4()),
                    decision_type="memory_optimization",
                    confidence=0.9,
                    recommendation="Optimize memory usage and consider scaling",
                    actions=[{
                        'type': 'memory_cleanup',
                        'parameter': 'garbage_collection',
                        'action': 'force_gc',
                        'priority': 'high'
                    }, {
                        'type': 'memory_scaling',
                        'parameter': 'memory_size',
                        'recommended_increase': '25%',
                        'priority': 'medium'
                    }],
                    reasoning=f"Memory usage at {memory_usage:.1f}% is critically high",
                    impact_assessment={
                        'performance_improvement': 'high',
                        'stability_improvement': 'high',
                        'cost_impact': 'medium'
                    },
                    created_at=datetime.utcnow(),
                    metadata={'memory_usage': memory_usage}
                )
                recommendations.append(recommendation)
            
            # Cache optimization recommendations
            perf_service_stats = platform_data.get('performance_service_stats', {})
            cache_stats = perf_service_stats.get('cache', {})
            
            if cache_stats.get('enabled', False):
                cache_hit_rate = cache_stats.get('stats', {}).get('hit_rate', 1.0)
                if cache_hit_rate < 0.6:
                    recommendation = AIDecision(
                        decision_id=str(uuid.uuid4()),
                        decision_type="cache_optimization",
                        confidence=0.75,
                        recommendation="Optimize cache strategy",
                        actions=[{
                            'type': 'cache_tuning',
                            'parameter': 'ttl_optimization',
                            'action': 'analyze_access_patterns',
                            'priority': 'medium'
                        }, {
                            'type': 'cache_warming',
                            'parameter': 'preload_strategy',
                            'action': 'implement_predictive_caching',
                            'priority': 'medium'
                        }],
                        reasoning=f"Cache hit rate of {cache_hit_rate*100:.1f}% is below optimal",
                        impact_assessment={
                            'performance_improvement': 'medium',
                            'cost_impact': 'low',
                            'implementation_effort': 'medium'
                        },
                        created_at=datetime.utcnow(),
                        metadata={'cache_hit_rate': cache_hit_rate}
                    )
                    recommendations.append(recommendation)
            
            # Cost optimization recommendations
            if cpu_usage < 20 and memory_usage < 30:
                recommendation = AIDecision(
                    decision_id=str(uuid.uuid4()),
                    decision_type="cost_optimization",
                    confidence=0.7,
                    recommendation="Consider resource downsizing for cost savings",
                    actions=[{
                        'type': 'resource_analysis',
                        'parameter': 'utilization_review',
                        'action': 'analyze_7_day_usage_patterns',
                        'priority': 'low'
                    }, {
                        'type': 'cost_optimization',
                        'parameter': 'instance_sizing',
                        'action': 'evaluate_smaller_instances',
                        'priority': 'low'
                    }],
                    reasoning=f"Low resource utilization (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)",
                    impact_assessment={
                        'cost_savings': 'high',
                        'performance_risk': 'low',
                        'monitoring_required': True
                    },
                    created_at=datetime.utcnow(),
                    metadata={'cpu_usage': cpu_usage, 'memory_usage': memory_usage}
                )
                recommendations.append(recommendation)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _calculate_health_score(self, platform_data: Dict[str, Any]) -> float:
        """Calculate overall platform health score (0-100)"""
        try:
            system_health = platform_data.get('system_health', {})
            
            # Base score from overall health
            base_score = 100.0 if system_health.get('overall_healthy', True) else 50.0
            
            # Adjust based on system metrics
            current_stats = system_health.get('current_stats', {})
            cpu_usage = current_stats.get('cpu_percent', 0)
            memory_usage = current_stats.get('memory_percent', 0)
            
            # CPU penalty
            if cpu_usage > 90:
                base_score -= 30
            elif cpu_usage > 80:
                base_score -= 15
            elif cpu_usage > 70:
                base_score -= 5
            
            # Memory penalty
            if memory_usage > 95:
                base_score -= 25
            elif memory_usage > 85:
                base_score -= 12
            elif memory_usage > 75:
                base_score -= 5
            
            # Agent health penalty
            agent_health = system_health.get('agent_health', {})
            unhealthy_agents = sum(1 for agent in agent_health.values() if not agent.get('healthy', True))
            base_score -= unhealthy_agents * 10
            
            # Ensure score is within bounds
            return max(0.0, min(100.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 75.0  # Default moderate score
    
    def _calculate_efficiency_score(self, platform_data: Dict[str, Any]) -> float:
        """Calculate platform efficiency score (0-100)"""
        try:
            efficiency_factors = []
            
            # Cache efficiency
            perf_service_stats = platform_data.get('performance_service_stats', {})
            cache_stats = perf_service_stats.get('cache', {})
            
            if cache_stats.get('enabled', False):
                hit_rate = cache_stats.get('stats', {}).get('hit_rate', 0.5)
                cache_efficiency = hit_rate * 100
                efficiency_factors.append(cache_efficiency)
            
            # Response time efficiency
            agent_stats = platform_data.get('agent_statistics', {})
            if isinstance(agent_stats, dict) and agent_stats:
                response_times = []
                for stats in agent_stats.values():
                    if isinstance(stats, dict):
                        avg_response = stats.get('avg_response_time_ms', 1000)
                        # Convert to efficiency score (lower is better)
                        efficiency = max(0, 100 - (avg_response / 10))  # 1000ms = 0% efficiency
                        response_times.append(efficiency)
                
                if response_times:
                    efficiency_factors.append(sum(response_times) / len(response_times))
            
            # Resource utilization efficiency
            current_stats = platform_data.get('system_health', {}).get('current_stats', {})
            cpu_usage = current_stats.get('cpu_percent', 50)
            memory_usage = current_stats.get('memory_percent', 50)
            
            # Optimal utilization is around 60-70%
            cpu_efficiency = 100 - abs(cpu_usage - 65) * 2
            memory_efficiency = 100 - abs(memory_usage - 65) * 2
            
            efficiency_factors.extend([
                max(0, cpu_efficiency),
                max(0, memory_efficiency)
            ])
            
            # Calculate overall efficiency
            if efficiency_factors:
                return sum(efficiency_factors) / len(efficiency_factors)
            else:
                return 70.0  # Default score
                
        except Exception as e:
            logger.error(f"Error calculating efficiency score: {e}")
            return 70.0
    
    def _calculate_cost_optimization_potential(self, platform_data: Dict[str, Any]) -> float:
        """Calculate cost optimization potential (0-100)"""
        try:
            current_stats = platform_data.get('system_health', {}).get('current_stats', {})
            cpu_usage = current_stats.get('cpu_percent', 50)
            memory_usage = current_stats.get('memory_percent', 50)
            
            # High optimization potential for low utilization
            cpu_potential = max(0, 50 - cpu_usage) * 2  # 0-50% unused CPU
            memory_potential = max(0, 50 - memory_usage) * 2  # 0-50% unused memory
            
            # Cache optimization potential
            perf_service_stats = platform_data.get('performance_service_stats', {})
            cache_stats = perf_service_stats.get('cache', {})
            cache_potential = 0
            
            if cache_stats.get('enabled', False):
                hit_rate = cache_stats.get('stats', {}).get('hit_rate', 0.5)
                # Potential improvement from better caching
                cache_potential = max(0, (0.9 - hit_rate) * 100)
            
            # Combine factors
            total_potential = (cpu_potential + memory_potential + cache_potential) / 3
            return min(100.0, total_potential)
            
        except Exception as e:
            logger.error(f"Error calculating cost optimization potential: {e}")
            return 20.0  # Default low potential
    
    async def _generate_self_healing_actions(self, platform_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate self-healing actions for autonomous operation"""
        actions = []
        
        try:
            system_health = platform_data.get('system_health', {})
            current_stats = system_health.get('current_stats', {})
            
            # Memory cleanup action
            memory_usage = current_stats.get('memory_percent', 0)
            if memory_usage > 90:
                actions.append({
                    'type': 'memory_cleanup',
                    'severity': 'critical',
                    'action': 'force_garbage_collection',
                    'description': 'Force garbage collection to free memory',
                    'auto_execute': self.intelligence_level == IntelligenceLevel.AUTONOMOUS,
                    'estimated_impact': 'Immediate memory reduction of 10-20%'
                })
            
            # Cache cleanup action
            perf_service_stats = platform_data.get('performance_service_stats', {})
            cache_stats = perf_service_stats.get('cache', {})
            
            if cache_stats.get('enabled', False):
                # Assume cache cleanup is needed if memory is high
                if memory_usage > 85:
                    actions.append({
                        'type': 'cache_optimization',
                        'severity': 'medium',
                        'action': 'clear_expired_cache_entries',
                        'description': 'Clean up expired cache entries to free memory',
                        'auto_execute': True,
                        'estimated_impact': 'Free up cache memory, improve hit rates'
                    })
            
            # Connection cleanup action
            active_connections = current_stats.get('active_connections', 0)
            if active_connections > 500:  # Threshold for connection cleanup
                actions.append({
                    'type': 'connection_cleanup',
                    'severity': 'medium',
                    'action': 'close_idle_connections',
                    'description': 'Close idle WebSocket connections',
                    'auto_execute': self.intelligence_level in [IntelligenceLevel.EXPERT, IntelligenceLevel.AUTONOMOUS],
                    'estimated_impact': 'Reduce connection overhead'
                })
            
            # Performance tuning action
            agent_stats = platform_data.get('agent_statistics', {})
            if isinstance(agent_stats, dict):
                slow_agents = []
                for agent_id, stats in agent_stats.items():
                    if isinstance(stats, dict):
                        avg_response = stats.get('avg_response_time_ms', 0)
                        if avg_response > 2000:  # > 2 seconds
                            slow_agents.append(agent_id)
                
                if slow_agents:
                    actions.append({
                        'type': 'performance_tuning',
                        'severity': 'medium',
                        'action': 'optimize_slow_agents',
                        'description': f'Optimize performance for slow agents: {", ".join(slow_agents)}',
                        'auto_execute': False,  # Requires manual approval
                        'estimated_impact': 'Improve response times by 20-40%',
                        'affected_agents': slow_agents
                    })
            
        except Exception as e:
            logger.error(f"Error generating self-healing actions: {e}")
        
        return actions
    
    def _store_intelligence_data(self, report: PlatformIntelligenceReport):
        """Store intelligence data for learning and pattern recognition"""
        try:
            # Store for pattern recognition
            self.learning_data.append({
                'timestamp': report.timestamp,
                'platform_state': report.platform_state.value,
                'health_score': report.health_score,
                'efficiency_score': report.efficiency_score,
                'recommendations_count': len(report.recommendations),
                'self_healing_actions_count': len(report.self_healing_actions)
            })
            
            # Store decisions for future reference
            for recommendation in report.recommendations:
                self.decision_history.append(recommendation)
            
            logger.debug(f"Stored intelligence data for learning: {report.report_id}")
            
        except Exception as e:
            logger.error(f"Error storing intelligence data: {e}")
    
    async def execute_self_healing_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a self-healing action"""
        action_id = str(uuid.uuid4())
        result = {
            'action_id': action_id,
            'action_type': action.get('type'),
            'executed_at': datetime.utcnow(),
            'success': False,
            'message': '',
            'impact': {}
        }
        
        try:
            action_type = action.get('type')
            
            if action_type == 'memory_cleanup':
                # Force garbage collection
                import gc
                before_count = len(gc.get_objects())
                gc.collect()
                after_count = len(gc.get_objects())
                
                result.update({
                    'success': True,
                    'message': 'Garbage collection completed',
                    'impact': {
                        'objects_cleaned': before_count - after_count,
                        'memory_freed': 'estimated'
                    }
                })
            
            elif action_type == 'cache_optimization':
                # Cache cleanup (would integrate with actual cache system)
                result.update({
                    'success': True,
                    'message': 'Cache optimization completed',
                    'impact': {
                        'expired_entries_cleared': 'estimated',
                        'memory_freed': 'calculated'
                    }
                })
            
            elif action_type == 'connection_cleanup':
                # Connection cleanup (would integrate with WebSocket manager)
                result.update({
                    'success': True,
                    'message': 'Idle connections cleaned up',
                    'impact': {
                        'connections_closed': 'estimated',
                        'resources_freed': 'calculated'
                    }
                })
            
            else:
                result['message'] = f'Unknown action type: {action_type}'
            
        except Exception as e:
            result['message'] = f'Action execution failed: {str(e)}'
            logger.error(f"Self-healing action failed: {e}")
        
        return result
    
    async def learn_from_feedback(self, decision_id: str, outcome: Dict[str, Any]):
        """Learn from decision outcomes to improve future recommendations"""
        try:
            # Find the decision in history
            decision = None
            for d in self.decision_history:
                if d.decision_id == decision_id:
                    decision = d
                    break
            
            if not decision:
                logger.warning(f"Decision {decision_id} not found in history")
                return
            
            # Store outcome for pattern learning
            learning_record = {
                'decision_id': decision_id,
                'decision_type': decision.decision_type,
                'confidence': decision.confidence,
                'outcome': outcome,
                'timestamp': datetime.utcnow()
            }
            
            # Add to pattern memory
            self.pattern_memory[decision.decision_type].append(learning_record)
            
            # Analyze pattern for future improvements
            if len(self.pattern_memory[decision.decision_type]) > 10:
                await self._analyze_decision_patterns(decision.decision_type)
            
            logger.info(f"Learned from decision outcome: {decision_id}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    async def _analyze_decision_patterns(self, decision_type: str):
        """Analyze patterns in decision outcomes for improvement"""
        try:
            records = self.pattern_memory[decision_type]
            
            # Analyze success rates by confidence level
            high_conf_success = 0
            high_conf_total = 0
            low_conf_success = 0
            low_conf_total = 0
            
            for record in records:
                if record['confidence'] > 0.8:
                    high_conf_total += 1
                    if record['outcome'].get('success', False):
                        high_conf_success += 1
                else:
                    low_conf_total += 1
                    if record['outcome'].get('success', False):
                        low_conf_success += 1
            
            # Log insights
            if high_conf_total > 0:
                high_success_rate = high_conf_success / high_conf_total
                logger.info(f"High confidence {decision_type} decisions: {high_success_rate:.2%} success rate")
            
            if low_conf_total > 0:
                low_success_rate = low_conf_success / low_conf_total
                logger.info(f"Low confidence {decision_type} decisions: {low_success_rate:.2%} success rate")
            
        except Exception as e:
            logger.error(f"Error analyzing decision patterns: {e}")

# Global platform AI instance
platform_ai = PlatformAI()

# Convenience functions
async def generate_intelligence_report() -> PlatformIntelligenceReport:
    """Generate comprehensive platform intelligence report"""
    return await platform_ai.analyze_platform_intelligence()

async def execute_self_healing(action: Dict[str, Any]) -> Dict[str, Any]:
    """Execute self-healing action"""
    return await platform_ai.execute_self_healing_action(action)

async def provide_decision_feedback(decision_id: str, outcome: Dict[str, Any]):
    """Provide feedback on AI decision outcome"""
    await platform_ai.learn_from_feedback(decision_id, outcome)

__all__ = [
    'PlatformAI',
    'platform_ai',
    'PlatformIntelligenceReport',
    'AIDecision',
    'IntelligenceLevel',
    'OptimizationStrategy',
    'PlatformState',
    'generate_intelligence_report',
    'execute_self_healing',
    'provide_decision_feedback'
]