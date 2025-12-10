# File: backend/operations/cost_optimization.py
"""
Cost optimization and resource right-sizing for UAP platform.
Provides intelligent cost management, budget tracking, resource optimization,
and cost anomaly detection.
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import boto3
import aioredis
from prometheus_client import Counter, Gauge, Histogram

from ..cache.redis_cache import get_redis_client
from .monitoring import AnomalyDetector, AlertSeverity

# Configure logging
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of cloud resources"""
    COMPUTE = "compute"
    STORAGE = "storage" 
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    LOAD_BALANCER = "load_balancer"
    CONTAINER = "container"

class OptimizationAction(Enum):
    """Types of optimization actions"""
    SCALE_DOWN = "scale_down"
    SCALE_UP = "scale_up"
    RIGHTSIZE = "rightsize"
    TERMINATE = "terminate"
    RESERVED_INSTANCE = "reserved_instance"
    SPOT_INSTANCE = "spot_instance"
    SCHEDULE = "schedule"

class CostSeverity(Enum):
    """Cost alert severity levels"""
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_WARNING = "budget_warning"
    COST_SPIKE = "cost_spike"
    WASTE_DETECTED = "waste_detected"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"

@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    resource_id: str
    resource_type: ResourceType
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    network_utilization: float
    disk_utilization: float
    cost_per_hour: float
    uptime_hours: float
    labels: Dict[str, str]

@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    id: str
    resource_id: str
    resource_type: ResourceType
    action: OptimizationAction
    current_cost_per_hour: float
    projected_cost_per_hour: float
    monthly_savings: float
    confidence_score: float
    description: str
    implementation_effort: str
    risk_level: str
    created_at: float

@dataclass
class BudgetAlert:
    """Budget monitoring alert"""
    id: str
    budget_name: str
    severity: CostSeverity
    current_spend: float
    budget_limit: float
    percentage_used: float
    period: str
    projected_spend: float
    timestamp: float
    description: str

@dataclass
class CostAnalysis:
    """Cost analysis results"""
    period: str
    total_cost: float
    cost_by_service: Dict[str, float]
    cost_by_resource_type: Dict[str, float]
    cost_trend: str
    top_cost_drivers: List[Dict[str, Any]]
    optimization_opportunities: List[OptimizationRecommendation]
    waste_indicators: List[Dict[str, Any]]

class CostOptimizer:
    """
    Intelligent cost optimization system that analyzes resource usage,
    identifies waste, and provides actionable recommendations.
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client or get_redis_client()
        self.anomaly_detector = AnomalyDetector(redis_client)
        
        # Cloud provider clients (would be configured based on deployment)
        self.aws_client = None
        self.gcp_client = None
        self.azure_client = None
        
        # Configuration
        self.analysis_interval = 3600  # Analyze every hour
        self.utilization_threshold_low = 10  # % - Consider underutilized
        self.utilization_threshold_high = 80  # % - Consider rightsize candidate
        self.cost_spike_threshold = 1.5  # 50% increase triggers alert
        
        # Resource usage history
        self.usage_history: Dict[str, List[ResourceUsage]] = {}
        self.max_history_size = 168  # 1 week of hourly data
        
        # Prometheus metrics
        self.cost_gauge = Gauge(
            'uap_cost_current',
            'Current cost metrics',
            ['service', 'resource_type', 'period']
        )
        
        self.optimization_savings_gauge = Gauge(
            'uap_optimization_potential_savings',
            'Potential cost savings from optimizations',
            ['resource_type', 'action']
        )
        
        self.budget_utilization_gauge = Gauge(
            'uap_budget_utilization_percent',
            'Budget utilization percentage',
            ['budget_name']
        )
        
        self.waste_detection_counter = Counter(
            'uap_waste_detected_total',
            'Total waste instances detected',
            ['resource_type', 'waste_type']
        )
    
    async def start_cost_monitoring(self) -> None:
        """Start cost monitoring and optimization tasks"""
        logger.info("Starting cost optimization monitoring...")
        
        tasks = [
            asyncio.create_task(self._cost_analysis_loop()),
            asyncio.create_task(self._resource_monitoring_loop()),
            asyncio.create_task(self._budget_monitoring_loop()),
            asyncio.create_task(self._optimization_generation_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Cost monitoring error: {e}")
    
    async def _cost_analysis_loop(self) -> None:
        """Continuous cost analysis and trend monitoring"""
        while True:
            try:
                # Analyze current costs
                analysis = await self._perform_cost_analysis()
                
                # Store analysis results
                await self._store_cost_analysis(analysis)
                
                # Check for cost anomalies
                await self._detect_cost_anomalies(analysis)
                
                # Update Prometheus metrics
                await self._update_cost_metrics(analysis)
                
            except Exception as e:
                logger.error(f"Cost analysis error: {e}")
            
            await asyncio.sleep(self.analysis_interval)
    
    async def _resource_monitoring_loop(self) -> None:
        """Monitor resource utilization for optimization opportunities"""
        while True:
            try:
                # Collect resource usage data
                resources = await self._collect_resource_usage()
                
                # Store usage data
                for resource in resources:
                    await self._store_resource_usage(resource)
                
                # Identify underutilized resources
                await self._identify_waste(resources)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _budget_monitoring_loop(self) -> None:
        """Monitor budgets and send alerts"""
        while True:
            try:
                # Check all configured budgets
                budgets = await self._get_configured_budgets()
                
                for budget in budgets:
                    alert = await self._check_budget_status(budget)
                    if alert:
                        await self._handle_budget_alert(alert)
                
            except Exception as e:
                logger.error(f"Budget monitoring error: {e}")
            
            await asyncio.sleep(1800)  # Check every 30 minutes
    
    async def _optimization_generation_loop(self) -> None:
        """Generate optimization recommendations"""
        while True:
            try:
                # Generate new recommendations
                recommendations = await self._generate_optimization_recommendations()
                
                # Store recommendations
                for rec in recommendations:
                    await self._store_recommendation(rec)
                
                # Update metrics
                await self._update_optimization_metrics(recommendations)
                
            except Exception as e:
                logger.error(f"Optimization generation error: {e}")
            
            await asyncio.sleep(3600)  # Generate every hour
    
    async def _perform_cost_analysis(self) -> CostAnalysis:
        """Perform comprehensive cost analysis"""
        # This would integrate with cloud provider billing APIs
        # For now, we'll simulate the analysis
        
        current_time = time.time()
        period = "monthly"
        
        # Simulate cost data (would come from actual billing APIs)
        total_cost = 1250.75
        cost_by_service = {
            "EC2": 450.25,
            "RDS": 320.50,
            "S3": 125.75,
            "ELB": 85.25,
            "CloudWatch": 45.00,
            "Other": 224.00
        }
        
        cost_by_resource_type = {
            ResourceType.COMPUTE.value: 535.50,
            ResourceType.DATABASE.value: 320.50,
            ResourceType.STORAGE.value: 155.75,
            ResourceType.LOAD_BALANCER.value: 85.25,
            ResourceType.NETWORK.value: 75.25,
            ResourceType.CACHE.value: 78.50
        }
        
        # Analyze cost trend
        cost_trend = await self._analyze_cost_trend()
        
        # Identify top cost drivers
        top_cost_drivers = [
            {"service": "EC2", "cost": 450.25, "change": "+15%"},
            {"service": "RDS", "cost": 320.50, "change": "+5%"},
            {"service": "S3", "cost": 125.75, "change": "-2%"}
        ]
        
        # Get current optimization opportunities
        optimization_opportunities = await self._get_current_recommendations()
        
        # Identify waste indicators
        waste_indicators = await self._identify_current_waste()
        
        return CostAnalysis(
            period=period,
            total_cost=total_cost,
            cost_by_service=cost_by_service,
            cost_by_resource_type=cost_by_resource_type,
            cost_trend=cost_trend,
            top_cost_drivers=top_cost_drivers,
            optimization_opportunities=optimization_opportunities,
            waste_indicators=waste_indicators
        )
    
    async def _analyze_cost_trend(self) -> str:
        """Analyze cost trend over time"""
        # Get historical cost data
        historical_costs = await self._get_historical_costs()
        
        if len(historical_costs) < 2:
            return "insufficient_data"
        
        recent_avg = statistics.mean(historical_costs[-7:])  # Last week
        previous_avg = statistics.mean(historical_costs[-14:-7])  # Previous week
        
        if recent_avg > previous_avg * 1.1:
            return "increasing"
        elif recent_avg < previous_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    async def _collect_resource_usage(self) -> List[ResourceUsage]:
        """Collect current resource usage data"""
        # This would integrate with cloud provider monitoring APIs
        # For now, we'll simulate usage data
        
        current_time = time.time()
        resources = []
        
        # Simulate EC2 instances
        for i in range(5):
            resources.append(ResourceUsage(
                resource_id=f"i-{i:08x}",
                resource_type=ResourceType.COMPUTE,
                timestamp=current_time,
                cpu_utilization=max(5, min(95, 30 + (i * 15) + (time.time() % 20))),
                memory_utilization=max(10, min(90, 40 + (i * 10) + (time.time() % 15))),
                network_utilization=max(1, min(80, 20 + (i * 5) + (time.time() % 25))),
                disk_utilization=max(5, min(95, 35 + (i * 8) + (time.time() % 10))),
                cost_per_hour=0.096 + (i * 0.024),
                uptime_hours=24 * (7 + i),
                labels={"instance_type": f"t3.medium", "environment": "production"}
            ))
        
        return resources
    
    async def _store_resource_usage(self, usage: ResourceUsage) -> None:
        """Store resource usage data for analysis"""
        resource_id = usage.resource_id
        
        if resource_id not in self.usage_history:
            self.usage_history[resource_id] = []
        
        self.usage_history[resource_id].append(usage)
        
        # Maintain history size
        if len(self.usage_history[resource_id]) > self.max_history_size:
            self.usage_history[resource_id] = self.usage_history[resource_id][-self.max_history_size:]
        
        # Store in Redis for persistence
        key = f"resource_usage:{resource_id}:{int(usage.timestamp)}"
        await self.redis_client.setex(
            key,
            86400 * 7,  # Store for 1 week
            json.dumps(asdict(usage), default=str)
        )
    
    async def _identify_waste(self, resources: List[ResourceUsage]) -> None:
        """Identify wasteful resource usage patterns"""
        for resource in resources:
            waste_indicators = []
            
            # Low CPU utilization
            if resource.cpu_utilization < self.utilization_threshold_low:
                waste_indicators.append({
                    "type": "low_cpu_utilization",
                    "value": resource.cpu_utilization,
                    "threshold": self.utilization_threshold_low
                })
            
            # Low memory utilization
            if resource.memory_utilization < self.utilization_threshold_low:
                waste_indicators.append({
                    "type": "low_memory_utilization",
                    "value": resource.memory_utilization,
                    "threshold": self.utilization_threshold_low
                })
            
            # Check historical patterns
            if resource.resource_id in self.usage_history:
                historical_data = self.usage_history[resource.resource_id]
                if len(historical_data) >= 24:  # At least 24 hours of data
                    avg_cpu = statistics.mean([u.cpu_utilization for u in historical_data[-24:]])
                    avg_memory = statistics.mean([u.memory_utilization for u in historical_data[-24:]])
                    
                    if avg_cpu < self.utilization_threshold_low and avg_memory < self.utilization_threshold_low:
                        waste_indicators.append({
                            "type": "persistent_underutilization", 
                            "avg_cpu": avg_cpu,
                            "avg_memory": avg_memory,
                            "duration_hours": 24
                        })
            
            # Store waste indicators
            if waste_indicators:
                await self._store_waste_indicators(resource.resource_id, waste_indicators)
                
                # Update Prometheus counter
                for indicator in waste_indicators:
                    self.waste_detection_counter.labels(
                        resource_type=resource.resource_type.value,
                        waste_type=indicator["type"]
                    ).inc()
    
    async def _generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate actionable optimization recommendations"""
        recommendations = []
        current_time = time.time()
        
        # Analyze each resource with sufficient history
        for resource_id, history in self.usage_history.items():
            if len(history) < 24:  # Need at least 24 hours of data
                continue
            
            latest = history[-1]
            recent_data = history[-24:]  # Last 24 hours
            
            # Calculate average utilization
            avg_cpu = statistics.mean([u.cpu_utilization for u in recent_data])
            avg_memory = statistics.mean([u.memory_utilization for u in recent_data])
            avg_cost = statistics.mean([u.cost_per_hour for u in recent_data])
            
            # Generate recommendations based on usage patterns
            if avg_cpu < 20 and avg_memory < 30:
                # Recommend downsizing
                new_cost = avg_cost * 0.5  # Simulate 50% cost reduction
                monthly_savings = (avg_cost - new_cost) * 24 * 30
                
                recommendations.append(OptimizationRecommendation(
                    id=f"downsize_{resource_id}_{int(current_time)}",
                    resource_id=resource_id,
                    resource_type=latest.resource_type,
                    action=OptimizationAction.SCALE_DOWN,
                    current_cost_per_hour=avg_cost,
                    projected_cost_per_hour=new_cost,
                    monthly_savings=monthly_savings,
                    confidence_score=0.85,
                    description=f"Resource {resource_id} is consistently underutilized (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%). Recommend downsizing to reduce costs.",
                    implementation_effort="Low",
                    risk_level="Low",
                    created_at=current_time
                ))
            
            elif avg_cpu > 80 or avg_memory > 85:
                # Recommend upsizing
                new_cost = avg_cost * 1.5  # 50% cost increase for better performance
                monthly_additional_cost = (new_cost - avg_cost) * 24 * 30
                
                recommendations.append(OptimizationRecommendation(
                    id=f"upsize_{resource_id}_{int(current_time)}",
                    resource_id=resource_id,
                    resource_type=latest.resource_type,
                    action=OptimizationAction.SCALE_UP,
                    current_cost_per_hour=avg_cost,
                    projected_cost_per_hour=new_cost,
                    monthly_savings=-monthly_additional_cost,  # Negative savings (additional cost)
                    confidence_score=0.75,
                    description=f"Resource {resource_id} is highly utilized (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%). Recommend upsizing to improve performance.",
                    implementation_effort="Medium",
                    risk_level="Low",
                    created_at=current_time
                ))
            
            # Check for spot instance opportunities
            if latest.resource_type == ResourceType.COMPUTE and avg_cost > 0.05:
                spot_cost = avg_cost * 0.3  # Spot instances typically 70% cheaper
                monthly_savings = (avg_cost - spot_cost) * 24 * 30
                
                recommendations.append(OptimizationRecommendation(
                    id=f"spot_{resource_id}_{int(current_time)}",
                    resource_id=resource_id,
                    resource_type=latest.resource_type,
                    action=OptimizationAction.SPOT_INSTANCE,
                    current_cost_per_hour=avg_cost,
                    projected_cost_per_hour=spot_cost,
                    monthly_savings=monthly_savings,
                    confidence_score=0.60,
                    description=f"Resource {resource_id} could use spot instances for significant cost savings.",
                    implementation_effort="High",
                    risk_level="Medium",
                    created_at=current_time
                ))
        
        return recommendations
    
    async def _check_budget_status(self, budget: Dict[str, Any]) -> Optional[BudgetAlert]:
        """Check budget status and generate alerts if needed"""
        budget_name = budget["name"]
        budget_limit = budget["limit"]
        period = budget["period"]
        
        # Get current spend (would come from billing API)
        current_spend = await self._get_current_spend(budget_name, period)
        percentage_used = (current_spend / budget_limit) * 100
        
        # Predict end-of-period spend
        projected_spend = await self._project_spend(current_spend, period)
        
        # Determine if alert is needed
        severity = None
        description = ""
        
        if percentage_used >= 100:
            severity = CostSeverity.BUDGET_EXCEEDED
            description = f"Budget '{budget_name}' has exceeded its limit of ${budget_limit:.2f}"
        elif percentage_used >= 90:
            severity = CostSeverity.BUDGET_WARNING
            description = f"Budget '{budget_name}' is at {percentage_used:.1f}% of its ${budget_limit:.2f} limit"
        elif projected_spend > budget_limit:
            severity = CostSeverity.BUDGET_WARNING
            description = f"Budget '{budget_name}' is projected to exceed its limit (projected: ${projected_spend:.2f})"
        
        if severity:
            return BudgetAlert(
                id=f"budget_{budget_name}_{int(time.time())}",
                budget_name=budget_name,
                severity=severity,
                current_spend=current_spend,
                budget_limit=budget_limit,
                percentage_used=percentage_used,
                period=period,
                projected_spend=projected_spend,
                timestamp=time.time(),
                description=description
            )
        
        return None
    
    async def _detect_cost_anomalies(self, analysis: CostAnalysis) -> None:
        """Detect cost anomalies using the anomaly detector"""
        # Check total cost anomalies
        await self.anomaly_detector.add_metric_point(
            "total_cost", 
            analysis.total_cost,
            {"period": analysis.period}
        )
        
        # Check service-specific cost anomalies
        for service, cost in analysis.cost_by_service.items():
            await self.anomaly_detector.add_metric_point(
                f"service_cost_{service.lower()}", 
                cost,
                {"service": service, "period": analysis.period}
            )
    
    async def get_cost_dashboard_data(self) -> Dict[str, Any]:
        """Get data for cost optimization dashboard"""
        # Get latest cost analysis
        analysis = await self._get_latest_cost_analysis()
        
        # Get active recommendations
        recommendations = await self._get_active_recommendations()
        
        # Get budget statuses
        budget_statuses = await self._get_budget_statuses()
        
        # Calculate potential savings
        total_potential_savings = sum(r.monthly_savings for r in recommendations if r.monthly_savings > 0)
        
        return {
            "timestamp": time.time(),
            "current_costs": {
                "total": analysis.total_cost if analysis else 0,
                "by_service": analysis.cost_by_service if analysis else {},
                "by_resource_type": analysis.cost_by_resource_type if analysis else {},
                "trend": analysis.cost_trend if analysis else "unknown"
            },
            "optimization": {
                "recommendations_count": len(recommendations),
                "potential_monthly_savings": total_potential_savings,
                "top_recommendations": sorted(recommendations, key=lambda x: x.monthly_savings, reverse=True)[:5]
            },
            "budgets": budget_statuses,
            "waste_indicators": analysis.waste_indicators if analysis else []
        }
    
    # Helper methods (simplified implementations)
    async def _get_configured_budgets(self) -> List[Dict[str, Any]]:
        """Get configured budgets"""
        return [
            {"name": "monthly_total", "limit": 2500.0, "period": "monthly"},
            {"name": "compute_budget", "limit": 1000.0, "period": "monthly"},
            {"name": "storage_budget", "limit": 500.0, "period": "monthly"}
        ]
    
    async def _get_current_spend(self, budget_name: str, period: str) -> float:
        """Get current spend for budget"""
        # Simulate current spend
        return {
            "monthly_total": 1250.75,
            "compute_budget": 535.50,
            "storage_budget": 155.75
        }.get(budget_name, 0.0)
    
    async def _project_spend(self, current_spend: float, period: str) -> float:
        """Project end-of-period spend"""
        if period == "monthly":
            # Simple projection based on days elapsed
            day_of_month = datetime.now().day
            days_in_month = 30  # Simplified
            return current_spend * (days_in_month / day_of_month)
        return current_spend
    
    async def _store_cost_analysis(self, analysis: CostAnalysis) -> None:
        """Store cost analysis results"""
        key = f"cost_analysis:{analysis.period}:{int(time.time())}"
        await self.redis_client.setex(
            key,
            86400 * 30,  # Store for 30 days
            json.dumps(asdict(analysis), default=str)
        )
    
    async def _get_latest_cost_analysis(self) -> Optional[CostAnalysis]:
        """Get the most recent cost analysis"""
        # This would retrieve from Redis
        return None  # Simplified for now
    
    async def _get_active_recommendations(self) -> List[OptimizationRecommendation]:
        """Get active optimization recommendations"""
        return []  # Simplified for now
    
    async def _get_budget_statuses(self) -> List[Dict[str, Any]]:
        """Get current budget statuses"""
        return []  # Simplified for now
    
    async def _get_historical_costs(self) -> List[float]:
        """Get historical cost data"""
        return [1200, 1150, 1300, 1250, 1180, 1350, 1250]  # Simulated
    
    async def _get_current_recommendations(self) -> List[OptimizationRecommendation]:
        """Get current optimization recommendations"""
        return []  # Simplified for now
    
    async def _identify_current_waste(self) -> List[Dict[str, Any]]:
        """Identify current waste indicators"""
        return []  # Simplified for now
    
    async def _store_recommendation(self, recommendation: OptimizationRecommendation) -> None:
        """Store optimization recommendation"""
        key = f"recommendation:{recommendation.id}"
        await self.redis_client.setex(
            key,
            86400 * 7,  # Store for 1 week
            json.dumps(asdict(recommendation), default=str)
        )
    
    async def _store_waste_indicators(self, resource_id: str, indicators: List[Dict[str, Any]]) -> None:
        """Store waste indicators"""
        key = f"waste:{resource_id}:{int(time.time())}"
        await self.redis_client.setex(
            key,
            86400,  # Store for 1 day
            json.dumps(indicators)
        )
    
    async def _handle_budget_alert(self, alert: BudgetAlert) -> None:
        """Handle budget alert"""
        logger.warning(f"Budget Alert: {alert.description}")
        
        # Store alert
        key = f"budget_alert:{alert.id}"
        await self.redis_client.setex(
            key,
            86400 * 7,  # Store for 1 week
            json.dumps(asdict(alert), default=str)
        )
    
    async def _update_cost_metrics(self, analysis: CostAnalysis) -> None:
        """Update Prometheus cost metrics"""
        # Update total cost
        self.cost_gauge.labels(
            service="total",
            resource_type="all",
            period=analysis.period
        ).set(analysis.total_cost)
        
        # Update service costs
        for service, cost in analysis.cost_by_service.items():
            self.cost_gauge.labels(
                service=service,
                resource_type="all",
                period=analysis.period
            ).set(cost)
    
    async def _update_optimization_metrics(self, recommendations: List[OptimizationRecommendation]) -> None:
        """Update optimization metrics"""
        # Group savings by resource type and action
        savings_by_type_action: Dict[Tuple[str, str], float] = {}
        
        for rec in recommendations:
            key = (rec.resource_type.value, rec.action.value)
            if key not in savings_by_type_action:
                savings_by_type_action[key] = 0
            savings_by_type_action[key] += max(0, rec.monthly_savings)
        
        # Update Prometheus metrics
        for (resource_type, action), savings in savings_by_type_action.items():
            self.optimization_savings_gauge.labels(
                resource_type=resource_type,
                action=action
            ).set(savings)

class ResourceRightSizer:
    """Specialized component for resource right-sizing recommendations"""
    
    def __init__(self):
        self.rightsizing_rules = self._load_rightsizing_rules()
    
    def _load_rightsizing_rules(self) -> Dict[str, Any]:
        """Load resource right-sizing rules"""
        return {
            "compute": {
                "cpu_thresholds": {
                    "underutilized": 10,
                    "optimal_min": 30,
                    "optimal_max": 70,
                    "overutilized": 85
                },
                "memory_thresholds": {
                    "underutilized": 20,
                    "optimal_min": 40,
                    "optimal_max": 80,
                    "overutilized": 90
                }
            }
        }
    
    async def analyze_resource_sizing(self, resource_usage: List[ResourceUsage]) -> List[OptimizationRecommendation]:
        """Analyze resource sizing and generate recommendations"""
        recommendations = []
        
        # Group by resource ID
        usage_by_resource: Dict[str, List[ResourceUsage]] = {}
        for usage in resource_usage:
            if usage.resource_id not in usage_by_resource:
                usage_by_resource[usage.resource_id] = []
            usage_by_resource[usage.resource_id].append(usage)
        
        # Analyze each resource
        for resource_id, usage_list in usage_by_resource.items():
            if len(usage_list) < 24:  # Need sufficient data
                continue
            
            rec = await self._analyze_single_resource(resource_id, usage_list)
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    async def _analyze_single_resource(self, resource_id: str, usage_list: List[ResourceUsage]) -> Optional[OptimizationRecommendation]:
        """Analyze a single resource for right-sizing opportunities"""
        latest = usage_list[-1]
        
        # Calculate utilization statistics
        cpu_values = [u.cpu_utilization for u in usage_list]
        memory_values = [u.memory_utilization for u in usage_list]
        
        avg_cpu = statistics.mean(cpu_values)
        max_cpu = max(cpu_values)
        avg_memory = statistics.mean(memory_values)
        max_memory = max(memory_values)
        
        current_cost = latest.cost_per_hour
        
        # Apply right-sizing rules
        if latest.resource_type == ResourceType.COMPUTE:
            rules = self.rightsizing_rules["compute"]
            
            if (avg_cpu < rules["cpu_thresholds"]["underutilized"] and 
                avg_memory < rules["memory_thresholds"]["underutilized"]):
                # Recommend downsizing
                new_cost = current_cost * 0.5
                monthly_savings = (current_cost - new_cost) * 24 * 30
                
                return OptimizationRecommendation(
                    id=f"rightsize_down_{resource_id}_{int(time.time())}",
                    resource_id=resource_id,
                    resource_type=latest.resource_type,
                    action=OptimizationAction.RIGHTSIZE,
                    current_cost_per_hour=current_cost,
                    projected_cost_per_hour=new_cost,
                    monthly_savings=monthly_savings,
                    confidence_score=0.8,
                    description=f"Resource consistently underutilized (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%)",
                    implementation_effort="Medium",
                    risk_level="Low",
                    created_at=time.time()
                )
            
            elif (max_cpu > rules["cpu_thresholds"]["overutilized"] or 
                  max_memory > rules["memory_thresholds"]["overutilized"]):
                # Recommend upsizing
                new_cost = current_cost * 1.5
                monthly_additional_cost = (new_cost - current_cost) * 24 * 30
                
                return OptimizationRecommendation(
                    id=f"rightsize_up_{resource_id}_{int(time.time())}",
                    resource_id=resource_id,
                    resource_type=latest.resource_type,
                    action=OptimizationAction.RIGHTSIZE,
                    current_cost_per_hour=current_cost,
                    projected_cost_per_hour=new_cost,
                    monthly_savings=-monthly_additional_cost,
                    confidence_score=0.7,
                    description=f"Resource showing high utilization (Max CPU: {max_cpu:.1f}%, Max Memory: {max_memory:.1f}%)",
                    implementation_effort="Medium",
                    risk_level="Low",
                    created_at=time.time()
                )
        
        return None

class BudgetManager:
    """Manages budgets and spending alerts"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client or get_redis_client()
        self.budgets: Dict[str, Dict[str, Any]] = {}
    
    async def create_budget(self, name: str, limit: float, period: str = "monthly") -> None:
        """Create a new budget"""
        budget = {
            "name": name,
            "limit": limit,
            "period": period,
            "created_at": time.time(),
            "alerts": {
                "50": True,
                "80": True,
                "90": True,
                "100": True
            }
        }
        
        self.budgets[name] = budget
        
        # Store in Redis
        key = f"budget:{name}"
        await self.redis_client.set(key, json.dumps(budget))
    
    async def update_budget(self, name: str, limit: float) -> None:
        """Update budget limit"""
        if name in self.budgets:
            self.budgets[name]["limit"] = limit
            
            # Update in Redis
            key = f"budget:{name}"
            await self.redis_client.set(key, json.dumps(self.budgets[name]))
    
    async def check_budget_alerts(self, name: str, current_spend: float) -> List[BudgetAlert]:
        """Check if budget alerts should be triggered"""
        if name not in self.budgets:
            return []
        
        budget = self.budgets[name]
        limit = budget["limit"]
        percentage = (current_spend / limit) * 100
        
        alerts = []
        
        # Check different threshold levels
        if percentage >= 100:
            alerts.append(BudgetAlert(
                id=f"budget_exceeded_{name}_{int(time.time())}",
                budget_name=name,
                severity=CostSeverity.BUDGET_EXCEEDED,
                current_spend=current_spend,
                budget_limit=limit,
                percentage_used=percentage,
                period=budget["period"],
                projected_spend=current_spend,
                timestamp=time.time(),
                description=f"Budget '{name}' has been exceeded"
            ))
        elif percentage >= 90:
            alerts.append(BudgetAlert(
                id=f"budget_warning_90_{name}_{int(time.time())}",
                budget_name=name,
                severity=CostSeverity.BUDGET_WARNING,
                current_spend=current_spend,
                budget_limit=limit,
                percentage_used=percentage,
                period=budget["period"],
                projected_spend=current_spend,
                timestamp=time.time(),
                description=f"Budget '{name}' is at {percentage:.1f}% utilization"
            ))
        
        return alerts

# Global instances
cost_optimizer = CostOptimizer()
resource_right_sizer = ResourceRightSizer()
budget_manager = BudgetManager()