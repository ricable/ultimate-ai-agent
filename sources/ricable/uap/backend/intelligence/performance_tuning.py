# File: backend/intelligence/performance_tuning.py
"""
Automated Performance Tuning System

AI-powered performance optimization with automated tuning, bottleneck detection,
and system optimization recommendations for the UAP platform.
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
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..monitoring.metrics.performance import performance_monitor
from ..services.performance_service import performance_service
from ..analytics.predictive_analytics import predictive_analytics

logger = logging.getLogger(__name__)

class PerformanceIssueType(Enum):
    """Types of performance issues that can be detected"""
    HIGH_LATENCY = "high_latency"
    HIGH_CPU_USAGE = "high_cpu_usage"
    HIGH_MEMORY_USAGE = "high_memory_usage"
    SLOW_DATABASE_QUERIES = "slow_database_queries"
    CACHE_INEFFICIENCY = "cache_inefficiency"
    NETWORK_BOTTLENECK = "network_bottleneck"
    RESOURCE_CONTENTION = "resource_contention"
    MEMORY_LEAK = "memory_leak"
    INEFFICIENT_ALGORITHM = "inefficient_algorithm"

class TuningStrategy(Enum):
    """Performance tuning strategies"""
    AGGRESSIVE = "aggressive"      # Maximum performance, higher resource usage
    BALANCED = "balanced"          # Balance performance and resources
    CONSERVATIVE = "conservative"  # Gradual improvements, minimal risk
    COST_AWARE = "cost_aware"     # Optimize performance within cost constraints

class OptimizationLevel(Enum):
    """Optimization impact levels"""
    MINOR = "minor"        # <5% improvement
    MODERATE = "moderate"  # 5-15% improvement
    MAJOR = "major"        # 15-30% improvement
    CRITICAL = "critical"  # >30% improvement

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    response_times: Dict[str, float]  # Agent ID -> response time
    throughput: float  # requests per second
    error_rate: float
    cache_hit_rate: float
    database_connection_pool_usage: float
    active_connections: int
    queue_lengths: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class PerformanceIssue:
    """Detected performance issue"""
    issue_id: str
    issue_type: PerformanceIssueType
    severity: str  # critical, high, medium, low
    description: str
    affected_components: List[str]
    metrics: Dict[str, float]
    first_detected: datetime
    last_detected: datetime
    frequency: int
    impact_assessment: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['issue_type'] = self.issue_type.value
        data['first_detected'] = self.first_detected.isoformat()
        data['last_detected'] = self.last_detected.isoformat()
        return data

@dataclass
class TuningRecommendation:
    """Performance tuning recommendation"""
    recommendation_id: str
    timestamp: datetime
    issue_type: PerformanceIssueType
    optimization_level: OptimizationLevel
    title: str
    description: str
    implementation_steps: List[str]
    expected_improvement: Dict[str, float]
    resource_impact: Dict[str, float]
    risk_level: str
    confidence: float
    estimated_effort: str
    rollback_procedure: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['issue_type'] = self.issue_type.value
        data['optimization_level'] = self.optimization_level.value
        return data

@dataclass
class TuningPlan:
    """Comprehensive performance tuning plan"""
    plan_id: str
    timestamp: datetime
    strategy: TuningStrategy
    detected_issues: List[PerformanceIssue]
    recommendations: List[TuningRecommendation]
    implementation_order: List[str]  # recommendation IDs in order
    total_expected_improvement: Dict[str, float]
    estimated_implementation_time: timedelta
    risk_assessment: Dict[str, Any]
    success_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['strategy'] = self.strategy.value
        data['estimated_implementation_hours'] = self.estimated_implementation_time.total_seconds() / 3600
        return data

class AutomatedPerformanceTuner:
    """AI-powered automated performance tuning system"""
    
    def __init__(self, strategy: TuningStrategy = TuningStrategy.BALANCED):
        self.strategy = strategy
        self.performance_history = deque(maxlen=10000)
        self.issue_history = deque(maxlen=1000)
        self.tuning_plans = deque(maxlen=100)
        
        # ML models for performance analysis
        self.models = {}
        self.anomaly_detector = None
        self.performance_classifier = None
        
        # Performance thresholds
        self.thresholds = {
            'response_time_ms': 2000,  # 2 seconds
            'cpu_usage_percent': 80,
            'memory_usage_percent': 85,
            'error_rate_percent': 5,
            'cache_hit_rate_percent': 60,
            'database_pool_usage_percent': 80
        }
        
        # Issue detection patterns
        self.issue_patterns = {}
        self.baseline_metrics = None
        
        # Initialize models and patterns
        self._initialize_models()
        self._initialize_issue_patterns()
        
        logger.info(f"Automated Performance Tuner initialized with strategy: {strategy.value}")
    
    def _initialize_models(self):
        """Initialize ML models for performance analysis"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, using rule-based tuning")
            return
        
        try:
            # Anomaly detection for performance metrics
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Performance issue classifier
            self.performance_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.models = {
                'anomaly_detector': self.anomaly_detector,
                'performance_classifier': self.performance_classifier,
                'scaler': StandardScaler()
            }
            
            logger.info("Performance tuning models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _initialize_issue_patterns(self):
        """Initialize performance issue detection patterns"""
        self.issue_patterns = {
            PerformanceIssueType.HIGH_LATENCY: {
                'conditions': [
                    lambda m: any(rt > self.thresholds['response_time_ms'] for rt in m.response_times.values()),
                    lambda m: m.throughput < 10  # Low throughput indicator
                ],
                'severity_thresholds': {
                    'critical': 5000,  # >5s response time
                    'high': 3000,      # >3s response time
                    'medium': 2000,    # >2s response time
                    'low': 1000        # >1s response time
                }
            },
            PerformanceIssueType.HIGH_CPU_USAGE: {
                'conditions': [
                    lambda m: m.cpu_usage > self.thresholds['cpu_usage_percent']
                ],
                'severity_thresholds': {
                    'critical': 95,
                    'high': 90,
                    'medium': 80,
                    'low': 70
                }
            },
            PerformanceIssueType.HIGH_MEMORY_USAGE: {
                'conditions': [
                    lambda m: m.memory_usage > self.thresholds['memory_usage_percent']
                ],
                'severity_thresholds': {
                    'critical': 95,
                    'high': 90,
                    'medium': 85,
                    'low': 75
                }
            },
            PerformanceIssueType.CACHE_INEFFICIENCY: {
                'conditions': [
                    lambda m: m.cache_hit_rate < self.thresholds['cache_hit_rate_percent']
                ],
                'severity_thresholds': {
                    'critical': 30,  # <30% hit rate
                    'high': 40,      # <40% hit rate
                    'medium': 50,    # <50% hit rate
                    'low': 60        # <60% hit rate
                }
            }
        }
    
    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        try:
            # Get system health and performance data
            system_health = performance_monitor.get_system_health()
            current_stats = system_health.get('current_stats', {})
            
            # Get agent statistics
            agent_stats = performance_monitor.get_agent_statistics()
            
            # Get performance service stats
            perf_service_stats = await performance_service.get_performance_stats()
            
            # Extract response times from agent stats
            response_times = {}
            if isinstance(agent_stats, dict):
                for agent_id, stats in agent_stats.items():
                    if isinstance(stats, dict):
                        response_times[agent_id] = stats.get('avg_response_time_ms', 0)
            
            # Calculate throughput (simplified)
            total_requests = sum(
                stats.get('total_requests', 0) for stats in agent_stats.values()
                if isinstance(stats, dict)
            ) if isinstance(agent_stats, dict) else 0
            throughput = total_requests / 3600.0  # Rough requests per hour to per second
            
            # Get cache statistics
            cache_stats = perf_service_stats.get('cache', {}).get('stats', {})
            cache_hit_rate = cache_stats.get('hit_rate', 0.5) * 100
            
            # Get database stats
            db_stats = perf_service_stats.get('database', {}).get('stats', {})
            db_pool_usage = 50.0  # Placeholder - would get from actual pool
            
            # Calculate error rate
            total_errors = sum(
                stats.get('error_count', 0) for stats in agent_stats.values()
                if isinstance(stats, dict)
            ) if isinstance(agent_stats, dict) else 0
            error_rate = (total_errors / max(1, total_requests)) * 100
            
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=current_stats.get('cpu_percent', 0),
                memory_usage=current_stats.get('memory_percent', 0),
                response_times=response_times,
                throughput=throughput,
                error_rate=error_rate,
                cache_hit_rate=cache_hit_rate,
                database_connection_pool_usage=db_pool_usage,
                active_connections=current_stats.get('active_connections', 0),
                queue_lengths={}  # Would get from actual queue systems
            )
            
            # Store in history
            self.performance_history.append(metrics)
            
            # Update baseline if needed
            if self.baseline_metrics is None:
                self.baseline_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            # Return default metrics
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=0,
                memory_usage=0,
                response_times={},
                throughput=0,
                error_rate=0,
                cache_hit_rate=50,
                database_connection_pool_usage=50,
                active_connections=0,
                queue_lengths={}
            )
    
    async def detect_performance_issues(self, metrics: PerformanceMetrics = None) -> List[PerformanceIssue]:
        """Detect performance issues from current or provided metrics"""
        if metrics is None:
            metrics = await self.collect_performance_metrics()
        
        detected_issues = []
        
        try:
            # Rule-based issue detection
            for issue_type, pattern in self.issue_patterns.items():
                conditions = pattern['conditions']
                severity_thresholds = pattern['severity_thresholds']
                
                # Check if any condition is met
                if any(condition(metrics) for condition in conditions):
                    # Determine severity
                    severity = self._determine_issue_severity(issue_type, metrics, severity_thresholds)
                    
                    # Create issue
                    issue = PerformanceIssue(
                        issue_id=f"{issue_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        issue_type=issue_type,
                        severity=severity,
                        description=self._generate_issue_description(issue_type, metrics),
                        affected_components=self._identify_affected_components(issue_type, metrics),
                        metrics=self._extract_relevant_metrics(issue_type, metrics),
                        first_detected=datetime.utcnow(),
                        last_detected=datetime.utcnow(),
                        frequency=1,
                        impact_assessment=self._assess_issue_impact(issue_type, metrics)
                    )
                    
                    detected_issues.append(issue)
            
            # ML-based anomaly detection
            if ML_AVAILABLE and len(self.performance_history) > 50:
                anomalies = await self._detect_ml_anomalies(metrics)
                detected_issues.extend(anomalies)
            
            # Store detected issues
            for issue in detected_issues:
                self.issue_history.append(issue)
            
            return detected_issues
            
        except Exception as e:
            logger.error(f"Error detecting performance issues: {e}")
            return []
    
    def _determine_issue_severity(self, issue_type: PerformanceIssueType,
                                 metrics: PerformanceMetrics,
                                 severity_thresholds: Dict[str, float]) -> str:
        """Determine severity of detected issue"""
        try:
            if issue_type == PerformanceIssueType.HIGH_LATENCY:
                max_response_time = max(metrics.response_times.values()) if metrics.response_times else 0
                value = max_response_time
            elif issue_type == PerformanceIssueType.HIGH_CPU_USAGE:
                value = metrics.cpu_usage
            elif issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE:
                value = metrics.memory_usage
            elif issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                value = metrics.cache_hit_rate
            else:
                return 'medium'
            
            for severity, threshold in severity_thresholds.items():
                if issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                    # Lower is worse for cache hit rate
                    if value <= threshold:
                        return severity
                else:
                    # Higher is worse for other metrics
                    if value >= threshold:
                        return severity
            
            return 'low'
            
        except Exception as e:
            logger.error(f"Error determining severity: {e}")
            return 'medium'
    
    def _generate_issue_description(self, issue_type: PerformanceIssueType,
                                   metrics: PerformanceMetrics) -> str:
        """Generate human-readable issue description"""
        try:
            if issue_type == PerformanceIssueType.HIGH_LATENCY:
                max_rt = max(metrics.response_times.values()) if metrics.response_times else 0
                return f"High response times detected. Maximum: {max_rt:.0f}ms, Threshold: {self.thresholds['response_time_ms']}ms"
            
            elif issue_type == PerformanceIssueType.HIGH_CPU_USAGE:
                return f"High CPU utilization: {metrics.cpu_usage:.1f}%, Threshold: {self.thresholds['cpu_usage_percent']}%"
            
            elif issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE:
                return f"High memory utilization: {metrics.memory_usage:.1f}%, Threshold: {self.thresholds['memory_usage_percent']}%"
            
            elif issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                return f"Low cache hit rate: {metrics.cache_hit_rate:.1f}%, Threshold: {self.thresholds['cache_hit_rate_percent']}%"
            
            else:
                return f"Performance issue detected: {issue_type.value}"
                
        except Exception as e:
            return f"Performance issue: {issue_type.value}"
    
    def _identify_affected_components(self, issue_type: PerformanceIssueType,
                                    metrics: PerformanceMetrics) -> List[str]:
        """Identify components affected by the issue"""
        components = []
        
        try:
            if issue_type == PerformanceIssueType.HIGH_LATENCY:
                # Identify slow agents
                for agent_id, response_time in metrics.response_times.items():
                    if response_time > self.thresholds['response_time_ms']:
                        components.append(f"Agent: {agent_id}")
            
            elif issue_type == PerformanceIssueType.HIGH_CPU_USAGE:
                components.append("System CPU")
            
            elif issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE:
                components.append("System Memory")
            
            elif issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                components.append("Cache System")
            
            if not components:
                components.append("System")
            
        except Exception as e:
            logger.error(f"Error identifying affected components: {e}")
            components = ["Unknown"]
        
        return components
    
    def _extract_relevant_metrics(self, issue_type: PerformanceIssueType,
                                 metrics: PerformanceMetrics) -> Dict[str, float]:
        """Extract metrics relevant to the specific issue"""
        relevant_metrics = {
            'timestamp': metrics.timestamp.timestamp()
        }
        
        try:
            if issue_type == PerformanceIssueType.HIGH_LATENCY:
                relevant_metrics.update({
                    'max_response_time': max(metrics.response_times.values()) if metrics.response_times else 0,
                    'avg_response_time': statistics.mean(metrics.response_times.values()) if metrics.response_times else 0,
                    'throughput': metrics.throughput
                })
            
            elif issue_type == PerformanceIssueType.HIGH_CPU_USAGE:
                relevant_metrics.update({
                    'cpu_usage': metrics.cpu_usage,
                    'active_connections': metrics.active_connections
                })
            
            elif issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE:
                relevant_metrics.update({
                    'memory_usage': metrics.memory_usage,
                    'active_connections': metrics.active_connections
                })
            
            elif issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                relevant_metrics.update({
                    'cache_hit_rate': metrics.cache_hit_rate,
                    'throughput': metrics.throughput
                })
            
        except Exception as e:
            logger.error(f"Error extracting relevant metrics: {e}")
        
        return relevant_metrics
    
    def _assess_issue_impact(self, issue_type: PerformanceIssueType,
                           metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Assess the impact of the performance issue"""
        impact = {
            'user_experience_impact': 'medium',
            'system_stability_risk': 'low',
            'cost_impact': 'low',
            'estimated_affected_users': 0
        }
        
        try:
            if issue_type == PerformanceIssueType.HIGH_LATENCY:
                max_rt = max(metrics.response_times.values()) if metrics.response_times else 0
                if max_rt > 5000:  # >5s
                    impact['user_experience_impact'] = 'critical'
                    impact['estimated_affected_users'] = metrics.active_connections
                elif max_rt > 3000:  # >3s
                    impact['user_experience_impact'] = 'high'
                    impact['estimated_affected_users'] = metrics.active_connections * 0.8
            
            elif issue_type in [PerformanceIssueType.HIGH_CPU_USAGE, PerformanceIssueType.HIGH_MEMORY_USAGE]:
                if metrics.cpu_usage > 95 or metrics.memory_usage > 95:
                    impact['system_stability_risk'] = 'critical'
                    impact['user_experience_impact'] = 'high'
                elif metrics.cpu_usage > 90 or metrics.memory_usage > 90:
                    impact['system_stability_risk'] = 'high'
            
            elif issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                if metrics.cache_hit_rate < 30:
                    impact['cost_impact'] = 'high'  # More database queries
                    impact['user_experience_impact'] = 'medium'
            
        except Exception as e:
            logger.error(f"Error assessing impact: {e}")
        
        return impact
    
    async def _detect_ml_anomalies(self, metrics: PerformanceMetrics) -> List[PerformanceIssue]:
        """Detect anomalies using ML models"""
        anomalies = []
        
        try:
            if not self.anomaly_detector or len(self.performance_history) < 50:
                return anomalies
            
            # Prepare feature vector
            features = self._prepare_ml_features(metrics)
            
            # Train anomaly detector if needed
            if not hasattr(self.anomaly_detector, 'decision_function'):
                # Train on historical data
                historical_features = []
                for hist_metrics in list(self.performance_history)[-100:]:
                    hist_features = self._prepare_ml_features(hist_metrics)
                    historical_features.append(hist_features)
                
                if len(historical_features) >= 20:
                    X = np.array(historical_features)
                    self.anomaly_detector.fit(X)
            
            # Detect anomaly
            if hasattr(self.anomaly_detector, 'decision_function'):
                anomaly_score = self.anomaly_detector.decision_function([features])[0]
                
                if anomaly_score < -0.5:  # Anomaly threshold
                    issue = PerformanceIssue(
                        issue_id=f"ml_anomaly_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        issue_type=PerformanceIssueType.INEFFICIENT_ALGORITHM,  # Generic type
                        severity='medium',
                        description=f"ML-detected performance anomaly (score: {anomaly_score:.3f})",
                        affected_components=['System'],
                        metrics={'anomaly_score': anomaly_score},
                        first_detected=datetime.utcnow(),
                        last_detected=datetime.utcnow(),
                        frequency=1,
                        impact_assessment={'ml_detected': True, 'confidence': abs(anomaly_score)}
                    )
                    anomalies.append(issue)
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
        
        return anomalies
    
    def _prepare_ml_features(self, metrics: PerformanceMetrics) -> List[float]:
        """Prepare feature vector for ML models"""
        try:
            # Basic system metrics
            features = [
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.throughput,
                metrics.error_rate,
                metrics.cache_hit_rate,
                metrics.database_connection_pool_usage,
                metrics.active_connections
            ]
            
            # Response time statistics
            if metrics.response_times:
                response_values = list(metrics.response_times.values())
                features.extend([
                    statistics.mean(response_values),
                    max(response_values),
                    min(response_values),
                    statistics.stdev(response_values) if len(response_values) > 1 else 0
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Time-based features
            hour = metrics.timestamp.hour
            day_of_week = metrics.timestamp.weekday()
            features.extend([hour, day_of_week])
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return [0] * 13  # Return default feature vector
    
    async def generate_tuning_recommendations(self, issues: List[PerformanceIssue]) -> List[TuningRecommendation]:
        """Generate tuning recommendations for detected issues"""
        recommendations = []
        
        try:
            for issue in issues:
                recommendation = await self._create_tuning_recommendation(issue)
                if recommendation:
                    recommendations.append(recommendation)
            
            # Sort recommendations by optimization level and confidence
            recommendations.sort(
                key=lambda r: (r.optimization_level.value, -r.confidence),
                reverse=True
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating tuning recommendations: {e}")
            return []
    
    async def _create_tuning_recommendation(self, issue: PerformanceIssue) -> Optional[TuningRecommendation]:
        """Create tuning recommendation for a specific issue"""
        try:
            recommendation_id = f"rec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Generate recommendation based on issue type
            if issue.issue_type == PerformanceIssueType.HIGH_LATENCY:
                return self._create_latency_recommendation(recommendation_id, issue)
            elif issue.issue_type == PerformanceIssueType.HIGH_CPU_USAGE:
                return self._create_cpu_recommendation(recommendation_id, issue)
            elif issue.issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE:
                return self._create_memory_recommendation(recommendation_id, issue)
            elif issue.issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                return self._create_cache_recommendation(recommendation_id, issue)
            else:
                return self._create_generic_recommendation(recommendation_id, issue)
                
        except Exception as e:
            logger.error(f"Error creating tuning recommendation: {e}")
            return None
    
    def _create_latency_recommendation(self, rec_id: str, issue: PerformanceIssue) -> TuningRecommendation:
        """Create recommendation for high latency issues"""
        return TuningRecommendation(
            recommendation_id=rec_id,
            timestamp=datetime.utcnow(),
            issue_type=issue.issue_type,
            optimization_level=OptimizationLevel.MAJOR,
            title="Optimize Agent Response Times",
            description="Reduce response times through caching, connection pooling, and query optimization",
            implementation_steps=[
                "1. Enable response caching for frequently requested data",
                "2. Optimize database connection pooling",
                "3. Implement request queuing and load balancing",
                "4. Profile and optimize slow agent operations",
                "5. Consider horizontal scaling of slow agents"
            ],
            expected_improvement={
                'response_time_reduction_percent': 30,
                'throughput_increase_percent': 25,
                'user_satisfaction_improvement': 'high'
            },
            resource_impact={
                'cpu_increase_percent': 5,
                'memory_increase_percent': 10,
                'cost_increase_percent': 15
            },
            risk_level='medium',
            confidence=0.85,
            estimated_effort='2-4 hours',
            rollback_procedure=[
                "1. Disable new caching mechanisms",
                "2. Restore previous connection pool settings",
                "3. Remove request queuing if performance degrades",
                "4. Monitor system stability for 30 minutes"
            ]
        )
    
    def _create_cpu_recommendation(self, rec_id: str, issue: PerformanceIssue) -> TuningRecommendation:
        """Create recommendation for high CPU usage"""
        return TuningRecommendation(
            recommendation_id=rec_id,
            timestamp=datetime.utcnow(),
            issue_type=issue.issue_type,
            optimization_level=OptimizationLevel.MODERATE,
            title="Optimize CPU Utilization",
            description="Reduce CPU usage through algorithm optimization and resource scaling",
            implementation_steps=[
                "1. Profile CPU-intensive operations",
                "2. Implement more efficient algorithms where possible",
                "3. Add CPU-based auto-scaling policies",
                "4. Optimize garbage collection settings",
                "5. Consider process distribution across cores"
            ],
            expected_improvement={
                'cpu_usage_reduction_percent': 20,
                'system_stability_improvement': 'high',
                'response_time_improvement_percent': 15
            },
            resource_impact={
                'memory_increase_percent': 5,
                'cost_increase_percent': 10
            },
            risk_level='low',
            confidence=0.8,
            estimated_effort='3-6 hours',
            rollback_procedure=[
                "1. Revert algorithm changes",
                "2. Restore previous GC settings",
                "3. Disable auto-scaling if issues occur",
                "4. Monitor CPU usage for stability"
            ]
        )
    
    def _create_memory_recommendation(self, rec_id: str, issue: PerformanceIssue) -> TuningRecommendation:
        """Create recommendation for high memory usage"""
        return TuningRecommendation(
            recommendation_id=rec_id,
            timestamp=datetime.utcnow(),
            issue_type=issue.issue_type,
            optimization_level=OptimizationLevel.MAJOR,
            title="Optimize Memory Usage",
            description="Reduce memory consumption through garbage collection tuning and memory leak detection",
            implementation_steps=[
                "1. Force garbage collection and monitor memory reclamation",
                "2. Profile memory usage to identify leaks",
                "3. Optimize data structures and caching strategies",
                "4. Implement memory usage monitoring and alerts",
                "5. Consider memory scaling if optimization is insufficient"
            ],
            expected_improvement={
                'memory_usage_reduction_percent': 25,
                'system_stability_improvement': 'critical',
                'gc_efficiency_improvement_percent': 30
            },
            resource_impact={
                'cpu_increase_percent': 3,
                'cost_increase_percent': 5
            },
            risk_level='medium',
            confidence=0.9,
            estimated_effort='2-5 hours',
            rollback_procedure=[
                "1. Restore previous GC settings",
                "2. Revert data structure changes",
                "3. Disable memory monitoring if problematic",
                "4. Monitor for memory leaks post-rollback"
            ]
        )
    
    def _create_cache_recommendation(self, rec_id: str, issue: PerformanceIssue) -> TuningRecommendation:
        """Create recommendation for cache inefficiency"""
        return TuningRecommendation(
            recommendation_id=rec_id,
            timestamp=datetime.utcnow(),
            issue_type=issue.issue_type,
            optimization_level=OptimizationLevel.MODERATE,
            title="Optimize Cache Strategy",
            description="Improve cache hit rates through better TTL policies and cache warming",
            implementation_steps=[
                "1. Analyze cache access patterns and TTL effectiveness",
                "2. Implement cache warming for frequently accessed data",
                "3. Optimize cache key strategies and partitioning",
                "4. Adjust TTL values based on data access patterns",
                "5. Consider increasing cache memory allocation"
            ],
            expected_improvement={
                'cache_hit_rate_increase_percent': 40,
                'database_load_reduction_percent': 30,
                'response_time_improvement_percent': 20
            },
            resource_impact={
                'memory_increase_percent': 20,
                'cost_increase_percent': 8
            },
            risk_level='low',
            confidence=0.75,
            estimated_effort='1-3 hours',
            rollback_procedure=[
                "1. Restore previous TTL settings",
                "2. Disable cache warming if issues occur",
                "3. Revert cache memory allocation changes",
                "4. Monitor cache performance for 1 hour"
            ]
        )
    
    def _create_generic_recommendation(self, rec_id: str, issue: PerformanceIssue) -> TuningRecommendation:
        """Create generic recommendation for unknown issues"""
        return TuningRecommendation(
            recommendation_id=rec_id,
            timestamp=datetime.utcnow(),
            issue_type=issue.issue_type,
            optimization_level=OptimizationLevel.MINOR,
            title="General Performance Investigation",
            description="Investigate and address detected performance anomaly",
            implementation_steps=[
                "1. Review system logs for error patterns",
                "2. Analyze performance metrics trends",
                "3. Profile application components",
                "4. Identify bottlenecks through monitoring",
                "5. Implement targeted optimizations"
            ],
            expected_improvement={
                'performance_improvement_percent': 10,
                'system_stability_improvement': 'medium'
            },
            resource_impact={
                'resource_increase_percent': 5
            },
            risk_level='low',
            confidence=0.6,
            estimated_effort='1-2 hours',
            rollback_procedure=[
                "1. Revert any configuration changes",
                "2. Monitor system for stability",
                "3. Document findings for future reference"
            ]
        )
    
    async def create_tuning_plan(self, issues: List[PerformanceIssue] = None) -> TuningPlan:
        """Create comprehensive performance tuning plan"""
        plan_id = f"tuning_plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Detect issues if not provided
            if issues is None:
                issues = await self.detect_performance_issues()
            
            # Generate recommendations
            recommendations = await self.generate_tuning_recommendations(issues)
            
            # Create implementation order based on impact and risk
            implementation_order = self._create_implementation_order(recommendations)
            
            # Calculate total expected improvement
            total_improvement = self._calculate_total_improvement(recommendations)
            
            # Estimate implementation time
            estimated_time = self._estimate_implementation_time(recommendations)
            
            # Assess risks
            risk_assessment = self._assess_tuning_risks(recommendations)
            
            # Define success metrics
            success_metrics = self._define_success_metrics(issues, recommendations)
            
            plan = TuningPlan(
                plan_id=plan_id,
                timestamp=datetime.utcnow(),
                strategy=self.strategy,
                detected_issues=issues,
                recommendations=recommendations,
                implementation_order=implementation_order,
                total_expected_improvement=total_improvement,
                estimated_implementation_time=estimated_time,
                risk_assessment=risk_assessment,
                success_metrics=success_metrics
            )
            
            # Store plan
            self.tuning_plans.append(plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating tuning plan: {e}")
            return TuningPlan(
                plan_id=plan_id,
                timestamp=datetime.utcnow(),
                strategy=self.strategy,
                detected_issues=issues or [],
                recommendations=[],
                implementation_order=[],
                total_expected_improvement={},
                estimated_implementation_time=timedelta(0),
                risk_assessment={'error': str(e)},
                success_metrics={}
            )
    
    def _create_implementation_order(self, recommendations: List[TuningRecommendation]) -> List[str]:
        """Create optimal implementation order for recommendations"""
        try:
            # Sort by optimization level (higher impact first) and risk (lower risk first)
            optimization_weights = {
                OptimizationLevel.CRITICAL: 4,
                OptimizationLevel.MAJOR: 3,
                OptimizationLevel.MODERATE: 2,
                OptimizationLevel.MINOR: 1
            }
            
            risk_weights = {
                'low': 3,
                'medium': 2,
                'high': 1
            }
            
            def sort_key(rec):
                opt_weight = optimization_weights.get(rec.optimization_level, 1)
                risk_weight = risk_weights.get(rec.risk_level, 1)
                return opt_weight + risk_weight + rec.confidence
            
            sorted_recs = sorted(recommendations, key=sort_key, reverse=True)
            return [rec.recommendation_id for rec in sorted_recs]
            
        except Exception as e:
            logger.error(f"Error creating implementation order: {e}")
            return [rec.recommendation_id for rec in recommendations]
    
    def _calculate_total_improvement(self, recommendations: List[TuningRecommendation]) -> Dict[str, float]:
        """Calculate total expected improvement from all recommendations"""
        try:
            total_improvement = defaultdict(float)
            
            for rec in recommendations:
                for metric, improvement in rec.expected_improvement.items():
                    if isinstance(improvement, (int, float)):
                        # Use diminishing returns model for combined improvements
                        current = total_improvement[metric]
                        total_improvement[metric] = current + improvement * (1 - current / 100)
            
            return dict(total_improvement)
            
        except Exception as e:
            logger.error(f"Error calculating total improvement: {e}")
            return {}
    
    def _estimate_implementation_time(self, recommendations: List[TuningRecommendation]) -> timedelta:
        """Estimate total implementation time"""
        try:
            total_hours = 0
            
            for rec in recommendations:
                effort = rec.estimated_effort
                if 'hour' in effort:
                    # Parse effort string like "2-4 hours"
                    import re
                    matches = re.findall(r'\d+', effort)
                    if matches:
                        # Use the maximum estimated time
                        hours = int(matches[-1])
                        total_hours += hours
                else:
                    total_hours += 2  # Default estimate
            
            return timedelta(hours=total_hours)
            
        except Exception as e:
            logger.error(f"Error estimating implementation time: {e}")
            return timedelta(hours=len(recommendations) * 2)
    
    def _assess_tuning_risks(self, recommendations: List[TuningRecommendation]) -> Dict[str, Any]:
        """Assess risks of the tuning plan"""
        try:
            high_risk_count = sum(1 for rec in recommendations if rec.risk_level == 'high')
            medium_risk_count = sum(1 for rec in recommendations if rec.risk_level == 'medium')
            
            overall_risk = 'low'
            if high_risk_count > 0:
                overall_risk = 'high'
            elif medium_risk_count > 2:
                overall_risk = 'medium'
            
            return {
                'overall_risk': overall_risk,
                'high_risk_recommendations': high_risk_count,
                'medium_risk_recommendations': medium_risk_count,
                'risk_mitigation_required': high_risk_count > 0 or medium_risk_count > 2,
                'recommended_approach': 'staged_implementation' if overall_risk != 'low' else 'parallel_implementation'
            }
            
        except Exception as e:
            logger.error(f"Error assessing tuning risks: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}
    
    def _define_success_metrics(self, issues: List[PerformanceIssue],
                               recommendations: List[TuningRecommendation]) -> Dict[str, float]:
        """Define success metrics for the tuning plan"""
        try:
            success_metrics = {}
            
            # Set targets based on detected issues
            for issue in issues:
                if issue.issue_type == PerformanceIssueType.HIGH_LATENCY:
                    success_metrics['max_response_time_ms'] = self.thresholds['response_time_ms']
                elif issue.issue_type == PerformanceIssueType.HIGH_CPU_USAGE:
                    success_metrics['cpu_usage_percent'] = self.thresholds['cpu_usage_percent']
                elif issue.issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE:
                    success_metrics['memory_usage_percent'] = self.thresholds['memory_usage_percent']
                elif issue.issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                    success_metrics['cache_hit_rate_percent'] = self.thresholds['cache_hit_rate_percent']
            
            # Add general improvement targets
            success_metrics.update({
                'error_rate_percent': self.thresholds['error_rate_percent'],
                'system_stability_score': 95.0,
                'user_satisfaction_score': 90.0
            })
            
            return success_metrics
            
        except Exception as e:
            logger.error(f"Error defining success metrics: {e}")
            return {}
    
    async def get_tuning_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get insights from performance tuning history"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            recent_issues = [i for i in self.issue_history if i.first_detected >= cutoff_time]
            recent_plans = [p for p in self.tuning_plans if p.timestamp >= cutoff_time]
            
            # Analyze issue patterns
            issue_frequency = defaultdict(int)
            severity_distribution = defaultdict(int)
            
            for issue in recent_issues:
                issue_frequency[issue.issue_type.value] += 1
                severity_distribution[issue.severity] += 1
            
            # Analyze tuning effectiveness
            plan_effectiveness = {}
            if recent_plans:
                total_expected_improvement = 0
                total_recommendations = 0
                
                for plan in recent_plans:
                    total_recommendations += len(plan.recommendations)
                    for improvement in plan.total_expected_improvement.values():
                        if isinstance(improvement, (int, float)):
                            total_expected_improvement += improvement
                
                plan_effectiveness = {
                    'total_plans': len(recent_plans),
                    'total_recommendations': total_recommendations,
                    'avg_recommendations_per_plan': total_recommendations / len(recent_plans),
                    'avg_expected_improvement': total_expected_improvement / len(recent_plans) if recent_plans else 0
                }
            
            insights = {
                'analysis_period_days': days,
                'issue_analysis': {
                    'total_issues_detected': len(recent_issues),
                    'issue_type_frequency': dict(issue_frequency),
                    'severity_distribution': dict(severity_distribution),
                    'most_common_issue': max(issue_frequency.items(), key=lambda x: x[1])[0] if issue_frequency else None
                },
                'tuning_effectiveness': plan_effectiveness,
                'performance_trends': self._analyze_performance_trends(),
                'recommendations': self._generate_insights_recommendations(recent_issues, recent_plans)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting tuning insights: {e}")
            return {'error': str(e)}
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from historical data"""
        try:
            if len(self.performance_history) < 10:
                return {'trend': 'insufficient_data'}
            
            recent_metrics = list(self.performance_history)[-min(100, len(self.performance_history)):]
            
            # Analyze trends for key metrics
            cpu_values = [m.cpu_usage for m in recent_metrics]
            memory_values = [m.memory_usage for m in recent_metrics]
            response_times = []
            
            for m in recent_metrics:
                if m.response_times:
                    avg_rt = statistics.mean(m.response_times.values())
                    response_times.append(avg_rt)
            
            trends = {}
            
            # Calculate trend slopes
            if len(cpu_values) > 5:
                x = np.arange(len(cpu_values))
                cpu_trend = np.polyfit(x, cpu_values, 1)[0]
                trends['cpu_trend'] = 'increasing' if cpu_trend > 0.1 else 'decreasing' if cpu_trend < -0.1 else 'stable'
            
            if len(memory_values) > 5:
                x = np.arange(len(memory_values))
                memory_trend = np.polyfit(x, memory_values, 1)[0]
                trends['memory_trend'] = 'increasing' if memory_trend > 0.1 else 'decreasing' if memory_trend < -0.1 else 'stable'
            
            if len(response_times) > 5:
                x = np.arange(len(response_times))
                rt_trend = np.polyfit(x, response_times, 1)[0]
                trends['response_time_trend'] = 'increasing' if rt_trend > 10 else 'decreasing' if rt_trend < -10 else 'stable'
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {'error': str(e)}
    
    def _generate_insights_recommendations(self, recent_issues: List[PerformanceIssue],
                                         recent_plans: List[TuningPlan]) -> List[str]:
        """Generate insights-based recommendations"""
        recommendations = []
        
        try:
            # Analyze issue patterns
            issue_counts = defaultdict(int)
            for issue in recent_issues:
                issue_counts[issue.issue_type] += 1
            
            # Recommend proactive measures for frequent issues
            for issue_type, count in issue_counts.items():
                if count >= 3:  # Frequent issue
                    if issue_type == PerformanceIssueType.HIGH_CPU_USAGE:
                        recommendations.append("Consider implementing CPU auto-scaling due to frequent CPU issues")
                    elif issue_type == PerformanceIssueType.HIGH_MEMORY_USAGE:
                        recommendations.append("Investigate memory leaks - memory issues are recurring")
                    elif issue_type == PerformanceIssueType.CACHE_INEFFICIENCY:
                        recommendations.append("Review cache strategy - cache performance issues are frequent")
            
            # Analyze plan effectiveness
            if recent_plans:
                avg_recommendations = statistics.mean(len(p.recommendations) for p in recent_plans)
                if avg_recommendations > 5:
                    recommendations.append("Consider preventive monitoring - many tuning recommendations needed")
            
            # Add general recommendations
            if not recommendations:
                recommendations.append("Performance appears stable - continue monitoring")
            
        except Exception as e:
            logger.error(f"Error generating insights recommendations: {e}")
            recommendations.append("Error generating recommendations - manual review suggested")
        
        return recommendations

# Global performance tuner instance
performance_tuner = AutomatedPerformanceTuner()

# Convenience functions
async def detect_performance_issues() -> List[PerformanceIssue]:
    """Detect current performance issues"""
    return await performance_tuner.detect_performance_issues()

async def create_tuning_plan(issues: List[PerformanceIssue] = None) -> TuningPlan:
    """Create performance tuning plan"""
    return await performance_tuner.create_tuning_plan(issues)

async def collect_performance_metrics() -> PerformanceMetrics:
    """Collect current performance metrics"""
    return await performance_tuner.collect_performance_metrics()

async def get_tuning_insights(days: int = 7) -> Dict[str, Any]:
    """Get performance tuning insights"""
    return await performance_tuner.get_tuning_insights(days)

__all__ = [
    'AutomatedPerformanceTuner',
    'performance_tuner',
    'PerformanceIssueType',
    'TuningStrategy',
    'OptimizationLevel',
    'PerformanceMetrics',
    'PerformanceIssue',
    'TuningRecommendation',
    'TuningPlan',
    'detect_performance_issues',
    'create_tuning_plan',
    'collect_performance_metrics',
    'get_tuning_insights'
]