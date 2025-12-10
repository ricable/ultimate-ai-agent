# File: backend/intelligence/resource_allocation.py
"""
Intelligent Resource Allocation System

AI-powered resource allocation with cost optimization, demand prediction,
and automated scaling decisions for the UAP platform.
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

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .usage_prediction import usage_predictor, UsageForecast
from ..monitoring.metrics.performance import performance_monitor
from ..services.performance_service import performance_service

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of resources that can be allocated"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE_MEMORY = "cache_memory"
    AGENT_INSTANCES = "agent_instances"

class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    BALANCED = "balanced"
    PREDICTIVE = "predictive"
    ELASTIC = "elastic"

class ScalingAction(Enum):
    """Scaling actions that can be taken"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    OPTIMIZE = "optimize"

@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    resource_type: ResourceType
    current_allocation: float
    required_allocation: float
    priority: int  # 1 (highest) to 5 (lowest)
    cost_per_unit: float
    utilization_threshold: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['resource_type'] = self.resource_type.value
        return data

@dataclass
class AllocationDecision:
    """Resource allocation decision"""
    decision_id: str
    timestamp: datetime
    resource_type: ResourceType
    current_allocation: float
    recommended_allocation: float
    scaling_action: ScalingAction
    confidence: float
    cost_impact: float
    performance_impact: float
    justification: str
    implementation_steps: List[str]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['resource_type'] = self.resource_type.value
        data['scaling_action'] = self.scaling_action.value
        return data

@dataclass
class ResourceAllocationPlan:
    """Comprehensive resource allocation plan"""
    plan_id: str
    timestamp: datetime
    strategy: AllocationStrategy
    time_horizon: timedelta
    total_cost_impact: float
    expected_performance_improvement: float
    decisions: List[AllocationDecision]
    risk_assessment: Dict[str, Any]
    implementation_timeline: List[Dict[str, Any]]
    rollback_plan: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['strategy'] = self.strategy.value
        data['time_horizon_hours'] = self.time_horizon.total_seconds() / 3600
        return data

class IntelligentResourceAllocator:
    """AI-powered resource allocation system"""
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.BALANCED):
        self.strategy = strategy
        self.allocation_history = deque(maxlen=1000)
        self.resource_models = {}
        self.cost_models = {}
        self.scalers = {}
        
        # Resource configuration
        self.resource_configs = {
            ResourceType.CPU: {
                'min_allocation': 1,
                'max_allocation': 64,
                'step_size': 1,
                'cost_per_unit_hour': 0.05,
                'utilization_threshold': 0.8
            },
            ResourceType.MEMORY: {
                'min_allocation': 1,  # GB
                'max_allocation': 256,
                'step_size': 1,
                'cost_per_unit_hour': 0.01,
                'utilization_threshold': 0.85
            },
            ResourceType.STORAGE: {
                'min_allocation': 10,  # GB
                'max_allocation': 10000,
                'step_size': 10,
                'cost_per_unit_hour': 0.001,
                'utilization_threshold': 0.9
            },
            ResourceType.DATABASE_CONNECTIONS: {
                'min_allocation': 10,
                'max_allocation': 1000,
                'step_size': 10,
                'cost_per_unit_hour': 0.001,
                'utilization_threshold': 0.8
            },
            ResourceType.CACHE_MEMORY: {
                'min_allocation': 0.5,  # GB
                'max_allocation': 64,
                'step_size': 0.5,
                'cost_per_unit_hour': 0.02,
                'utilization_threshold': 0.75
            },
            ResourceType.AGENT_INSTANCES: {
                'min_allocation': 1,
                'max_allocation': 100,
                'step_size': 1,
                'cost_per_unit_hour': 0.1,
                'utilization_threshold': 0.7
            }
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"Intelligent Resource Allocator initialized with strategy: {strategy.value}")
    
    def _initialize_models(self):
        """Initialize ML models for resource allocation"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, using rule-based allocation")
            return
        
        try:
            for resource_type in ResourceType:
                # Resource demand prediction model
                self.resource_models[resource_type] = {
                    'demand_predictor': RandomForestRegressor(
                        n_estimators=100,
                        max_depth=8,
                        random_state=42
                    ),
                    'scaler': StandardScaler(),
                    'trained': False
                }
                
                # Cost optimization model
                self.cost_models[resource_type] = {
                    'cost_predictor': LinearRegression(),
                    'scaler': StandardScaler(),
                    'trained': False
                }
            
            logger.info("Resource allocation models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def analyze_resource_requirements(self) -> Dict[ResourceType, ResourceRequirement]:
        """Analyze current resource requirements"""
        requirements = {}
        
        try:
            # Get current system state
            system_health = performance_monitor.get_system_health()
            current_stats = system_health.get('current_stats', {})
            
            # Get usage forecast
            usage_forecast = await usage_predictor.predict_usage(timedelta(hours=4))
            
            # Analyze each resource type
            for resource_type in ResourceType:
                requirement = await self._analyze_resource_type(
                    resource_type, current_stats, usage_forecast
                )
                requirements[resource_type] = requirement
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error analyzing resource requirements: {e}")
            return {}
    
    async def _analyze_resource_type(self, resource_type: ResourceType,
                                   current_stats: Dict[str, Any],
                                   usage_forecast: UsageForecast) -> ResourceRequirement:
        """Analyze requirements for a specific resource type"""
        try:
            config = self.resource_configs[resource_type]
            
            if resource_type == ResourceType.CPU:
                current_allocation = self._estimate_current_cpu_allocation()
                current_utilization = current_stats.get('cpu_percent', 0) / 100.0
                predicted_load = usage_forecast.predicted_usage.get('estimated_cpu_usage', 0) / 100.0
                
            elif resource_type == ResourceType.MEMORY:
                current_allocation = self._estimate_current_memory_allocation()
                current_utilization = current_stats.get('memory_percent', 0) / 100.0
                predicted_load = usage_forecast.predicted_usage.get('estimated_memory_usage', 0) / 100.0
                
            elif resource_type == ResourceType.DATABASE_CONNECTIONS:
                current_allocation = self._estimate_current_db_connections()
                current_utilization = min(1.0, current_stats.get('active_connections', 0) / current_allocation)
                predicted_connections = usage_forecast.predicted_usage.get('active_users', 0) * 1.5
                predicted_load = min(1.0, predicted_connections / current_allocation)
                
            elif resource_type == ResourceType.CACHE_MEMORY:
                current_allocation = self._estimate_current_cache_memory()
                current_utilization = 0.6  # Placeholder - would get from cache system
                predicted_load = min(1.0, usage_forecast.predicted_usage.get('total_requests', 0) * 0.001)
                
            elif resource_type == ResourceType.AGENT_INSTANCES:
                current_allocation = len(performance_monitor.get_agent_statistics() or {})
                current_utilization = 0.7  # Placeholder - would calculate based on agent load
                predicted_load = min(1.0, usage_forecast.predicted_usage.get('active_users', 0) * 0.01)
                
            else:
                # Default values for other resource types
                current_allocation = config['min_allocation']
                current_utilization = 0.5
                predicted_load = 0.5
            
            # Calculate required allocation
            required_allocation = self._calculate_required_allocation(
                resource_type, current_allocation, current_utilization, predicted_load
            )
            
            # Determine priority based on utilization and criticality
            priority = self._determine_priority(resource_type, current_utilization, predicted_load)
            
            return ResourceRequirement(
                resource_type=resource_type,
                current_allocation=current_allocation,
                required_allocation=required_allocation,
                priority=priority,
                cost_per_unit=config['cost_per_unit_hour'],
                utilization_threshold=config['utilization_threshold'],
                metadata={
                    'current_utilization': current_utilization,
                    'predicted_load': predicted_load,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {resource_type.value}: {e}")
            config = self.resource_configs[resource_type]
            return ResourceRequirement(
                resource_type=resource_type,
                current_allocation=config['min_allocation'],
                required_allocation=config['min_allocation'],
                priority=3,
                cost_per_unit=config['cost_per_unit_hour'],
                utilization_threshold=config['utilization_threshold'],
                metadata={'error': str(e)}
            )
    
    def _estimate_current_cpu_allocation(self) -> float:
        """Estimate current CPU allocation (cores)"""
        try:
            import psutil
            return float(psutil.cpu_count())
        except:
            return 4.0  # Default assumption
    
    def _estimate_current_memory_allocation(self) -> float:
        """Estimate current memory allocation (GB)"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)  # Convert to GB
        except:
            return 8.0  # Default assumption
    
    def _estimate_current_db_connections(self) -> float:
        """Estimate current database connection pool size"""
        try:
            # Would integrate with actual database connection pool
            return 50.0  # Default pool size
        except:
            return 50.0
    
    def _estimate_current_cache_memory(self) -> float:
        """Estimate current cache memory allocation (GB)"""
        try:
            # Would integrate with Redis/cache system
            return 2.0  # Default cache size
        except:
            return 2.0
    
    def _calculate_required_allocation(self, resource_type: ResourceType,
                                     current_allocation: float,
                                     current_utilization: float,
                                     predicted_load: float) -> float:
        """Calculate required resource allocation"""
        try:
            config = self.resource_configs[resource_type]
            threshold = config['utilization_threshold']
            
            # Base calculation on predicted load and safety margin
            safety_margin = 0.2  # 20% safety margin
            target_utilization = threshold - safety_margin
            
            if predicted_load > target_utilization:
                # Scale up needed
                scale_factor = predicted_load / target_utilization
                required = current_allocation * scale_factor
            elif current_utilization < (target_utilization * 0.5):
                # Scale down opportunity
                scale_factor = max(0.5, current_utilization / target_utilization)
                required = current_allocation * scale_factor
            else:
                # Current allocation is appropriate
                required = current_allocation
            
            # Apply constraints
            required = max(config['min_allocation'], required)
            required = min(config['max_allocation'], required)
            
            # Align to step size
            steps = round(required / config['step_size'])
            required = steps * config['step_size']
            
            return required
            
        except Exception as e:
            logger.error(f"Error calculating required allocation: {e}")
            return current_allocation
    
    def _determine_priority(self, resource_type: ResourceType,
                          current_utilization: float,
                          predicted_load: float) -> int:
        """Determine priority for resource allocation (1=highest, 5=lowest)"""
        try:
            # Critical resources get higher priority
            critical_resources = [ResourceType.CPU, ResourceType.MEMORY]
            
            if resource_type in critical_resources:
                base_priority = 1
            else:
                base_priority = 3
            
            # Adjust based on utilization
            if current_utilization > 0.9 or predicted_load > 0.9:
                priority = 1  # Critical
            elif current_utilization > 0.8 or predicted_load > 0.8:
                priority = 2  # High
            elif current_utilization > 0.6 or predicted_load > 0.6:
                priority = 3  # Medium
            elif current_utilization < 0.3 and predicted_load < 0.3:
                priority = 5  # Low (scale down opportunity)
            else:
                priority = 4  # Normal
            
            return min(priority, base_priority)
            
        except Exception as e:
            logger.error(f"Error determining priority: {e}")
            return 3
    
    async def create_allocation_plan(self, requirements: Dict[ResourceType, ResourceRequirement],
                                   time_horizon: timedelta = timedelta(hours=12)) -> ResourceAllocationPlan:
        """Create comprehensive resource allocation plan"""
        plan_id = f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            decisions = []
            total_cost_impact = 0.0
            performance_improvement = 0.0
            
            # Sort requirements by priority
            sorted_requirements = sorted(
                requirements.items(),
                key=lambda x: x[1].priority
            )
            
            # Create allocation decisions
            for resource_type, requirement in sorted_requirements:
                decision = await self._create_allocation_decision(requirement)
                decisions.append(decision)
                
                total_cost_impact += decision.cost_impact
                performance_improvement += decision.performance_impact
            
            # Assess risks
            risk_assessment = self._assess_allocation_risks(decisions)
            
            # Create implementation timeline
            implementation_timeline = self._create_implementation_timeline(decisions)
            
            # Create rollback plan
            rollback_plan = self._create_rollback_plan(decisions)
            
            plan = ResourceAllocationPlan(
                plan_id=plan_id,
                timestamp=datetime.utcnow(),
                strategy=self.strategy,
                time_horizon=time_horizon,
                total_cost_impact=total_cost_impact,
                expected_performance_improvement=performance_improvement,
                decisions=decisions,
                risk_assessment=risk_assessment,
                implementation_timeline=implementation_timeline,
                rollback_plan=rollback_plan
            )
            
            # Store plan in history
            self.allocation_history.append(plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating allocation plan: {e}")
            # Return minimal plan
            return ResourceAllocationPlan(
                plan_id=plan_id,
                timestamp=datetime.utcnow(),
                strategy=self.strategy,
                time_horizon=time_horizon,
                total_cost_impact=0.0,
                expected_performance_improvement=0.0,
                decisions=[],
                risk_assessment={'error': str(e)},
                implementation_timeline=[],
                rollback_plan=[]
            )
    
    async def _create_allocation_decision(self, requirement: ResourceRequirement) -> AllocationDecision:
        """Create allocation decision for a resource requirement"""
        try:
            decision_id = f"decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            current = requirement.current_allocation
            required = requirement.required_allocation
            
            # Determine scaling action
            if required > current * 1.1:  # >10% increase
                scaling_action = ScalingAction.SCALE_UP
            elif required < current * 0.9:  # <10% decrease
                scaling_action = ScalingAction.SCALE_DOWN
            elif abs(required - current) < current * 0.05:  # <5% change
                scaling_action = ScalingAction.MAINTAIN
            else:
                scaling_action = ScalingAction.OPTIMIZE
            
            # Calculate cost impact
            cost_difference = (required - current) * requirement.cost_per_unit
            
            # Calculate performance impact (simplified)
            if scaling_action == ScalingAction.SCALE_UP:
                performance_impact = min(20.0, (required - current) / current * 100)
            elif scaling_action == ScalingAction.SCALE_DOWN:
                performance_impact = max(-10.0, (required - current) / current * 50)
            else:
                performance_impact = 0.0
            
            # Calculate confidence based on data availability and prediction accuracy
            confidence = self._calculate_decision_confidence(requirement)
            
            # Generate justification
            justification = self._generate_justification(requirement, scaling_action)
            
            # Generate implementation steps
            implementation_steps = self._generate_implementation_steps(
                requirement.resource_type, scaling_action, current, required
            )
            
            return AllocationDecision(
                decision_id=decision_id,
                timestamp=datetime.utcnow(),
                resource_type=requirement.resource_type,
                current_allocation=current,
                recommended_allocation=required,
                scaling_action=scaling_action,
                confidence=confidence,
                cost_impact=cost_difference,
                performance_impact=performance_impact,
                justification=justification,
                implementation_steps=implementation_steps,
                metadata={
                    'priority': requirement.priority,
                    'utilization_threshold': requirement.utilization_threshold,
                    'analysis_data': requirement.metadata
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating allocation decision: {e}")
            return AllocationDecision(
                decision_id="error",
                timestamp=datetime.utcnow(),
                resource_type=requirement.resource_type,
                current_allocation=requirement.current_allocation,
                recommended_allocation=requirement.current_allocation,
                scaling_action=ScalingAction.MAINTAIN,
                confidence=0.0,
                cost_impact=0.0,
                performance_impact=0.0,
                justification=f"Error creating decision: {str(e)}",
                implementation_steps=[],
                metadata={'error': str(e)}
            )
    
    def _calculate_decision_confidence(self, requirement: ResourceRequirement) -> float:
        """Calculate confidence level for allocation decision"""
        try:
            base_confidence = 0.7
            
            # Adjust based on data quality
            metadata = requirement.metadata or {}
            if 'error' in metadata:
                base_confidence *= 0.3
            
            # Adjust based on priority (higher priority = more conservative = higher confidence)
            priority_factor = (6 - requirement.priority) / 5.0  # Invert priority
            base_confidence *= priority_factor
            
            # Adjust based on resource type criticality
            if requirement.resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                base_confidence *= 1.1
            
            return max(0.0, min(1.0, base_confidence))
            
        except:
            return 0.5
    
    def _generate_justification(self, requirement: ResourceRequirement,
                              scaling_action: ScalingAction) -> str:
        """Generate human-readable justification for allocation decision"""
        try:
            resource_name = requirement.resource_type.value.replace('_', ' ').title()
            current = requirement.current_allocation
            required = requirement.required_allocation
            
            metadata = requirement.metadata or {}
            current_util = metadata.get('current_utilization', 0)
            predicted_load = metadata.get('predicted_load', 0)
            
            if scaling_action == ScalingAction.SCALE_UP:
                return (f"{resource_name} scaling up from {current:.1f} to {required:.1f} units. "
                       f"Current utilization: {current_util:.1%}, predicted load: {predicted_load:.1%}. "
                       f"Scaling needed to maintain performance under increased demand.")
            
            elif scaling_action == ScalingAction.SCALE_DOWN:
                return (f"{resource_name} scaling down from {current:.1f} to {required:.1f} units. "
                       f"Current utilization: {current_util:.1%}, predicted load: {predicted_load:.1%}. "
                       f"Cost optimization opportunity due to low utilization.")
            
            elif scaling_action == ScalingAction.OPTIMIZE:
                return (f"{resource_name} optimization from {current:.1f} to {required:.1f} units. "
                       f"Fine-tuning allocation for better efficiency.")
            
            else:
                return (f"{resource_name} maintaining current allocation of {current:.1f} units. "
                       f"Current configuration is optimal for predicted workload.")
            
        except Exception as e:
            return f"Resource allocation decision for {requirement.resource_type.value}"
    
    def _generate_implementation_steps(self, resource_type: ResourceType,
                                      scaling_action: ScalingAction,
                                      current: float, required: float) -> List[str]:
        """Generate implementation steps for allocation decision"""
        steps = []
        
        try:
            resource_name = resource_type.value.replace('_', ' ')
            
            if scaling_action == ScalingAction.SCALE_UP:
                steps.extend([
                    f"1. Validate current {resource_name} capacity and utilization",
                    f"2. Prepare scaling from {current:.1f} to {required:.1f} units",
                    "3. Implement gradual scaling to avoid service disruption",
                    "4. Monitor performance during scaling process",
                    "5. Verify scaling completion and performance improvement"
                ])
            
            elif scaling_action == ScalingAction.SCALE_DOWN:
                steps.extend([
                    f"1. Confirm low utilization of {resource_name}",
                    "2. Ensure no upcoming demand spikes predicted",
                    f"3. Gradually reduce allocation from {current:.1f} to {required:.1f} units",
                    "4. Monitor for any performance degradation",
                    "5. Complete scaling and verify cost savings"
                ])
            
            elif scaling_action == ScalingAction.OPTIMIZE:
                steps.extend([
                    f"1. Analyze current {resource_name} configuration",
                    f"2. Optimize allocation from {current:.1f} to {required:.1f} units",
                    "3. Fine-tune resource parameters",
                    "4. Validate optimization results",
                    "5. Document configuration changes"
                ])
            
            else:  # MAINTAIN
                steps.extend([
                    f"1. Continue monitoring {resource_name} utilization",
                    "2. Maintain current allocation settings",
                    "3. Review allocation in next planning cycle"
                ])
            
        except Exception as e:
            steps = [f"Error generating steps: {str(e)}"]
        
        return steps
    
    def _assess_allocation_risks(self, decisions: List[AllocationDecision]) -> Dict[str, Any]:
        """Assess risks associated with allocation plan"""
        try:
            risks = {
                'overall_risk_level': 'low',
                'risk_factors': [],
                'mitigation_strategies': []
            }
            
            # Analyze cost impact
            total_cost_increase = sum(d.cost_impact for d in decisions if d.cost_impact > 0)
            if total_cost_increase > 100:  # $100/hour threshold
                risks['risk_factors'].append('High cost increase')
                risks['mitigation_strategies'].append('Implement cost monitoring and alerts')
                risks['overall_risk_level'] = 'medium'
            
            # Analyze scale-down risks
            scale_down_count = sum(1 for d in decisions if d.scaling_action == ScalingAction.SCALE_DOWN)
            if scale_down_count > 2:
                risks['risk_factors'].append('Multiple simultaneous scale-down operations')
                risks['mitigation_strategies'].append('Stagger scale-down operations')
                risks['overall_risk_level'] = 'medium'
            
            # Analyze confidence levels
            low_confidence_count = sum(1 for d in decisions if d.confidence < 0.6)
            if low_confidence_count > 0:
                risks['risk_factors'].append('Some decisions have low confidence')
                risks['mitigation_strategies'].append('Manual review of low-confidence decisions')
                if risks['overall_risk_level'] == 'low':
                    risks['overall_risk_level'] = 'medium'
            
            # Analyze critical resource changes
            critical_changes = [
                d for d in decisions 
                if d.resource_type in [ResourceType.CPU, ResourceType.MEMORY] 
                and d.scaling_action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN]
            ]
            if critical_changes:
                risks['risk_factors'].append('Changes to critical resources (CPU/Memory)')
                risks['mitigation_strategies'].append('Careful monitoring during critical resource changes')
                risks['overall_risk_level'] = 'high' if len(critical_changes) > 2 else 'medium'
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            return {
                'overall_risk_level': 'unknown',
                'risk_factors': ['Error in risk assessment'],
                'mitigation_strategies': ['Manual review required'],
                'error': str(e)
            }
    
    def _create_implementation_timeline(self, decisions: List[AllocationDecision]) -> List[Dict[str, Any]]:
        """Create implementation timeline for allocation decisions"""
        timeline = []
        
        try:
            # Sort decisions by priority (from metadata)
            sorted_decisions = sorted(
                decisions,
                key=lambda d: d.metadata.get('priority', 5)
            )
            
            current_time = datetime.utcnow()
            
            for i, decision in enumerate(sorted_decisions):
                # Stagger implementations to avoid conflicts
                start_time = current_time + timedelta(minutes=i * 15)
                duration = timedelta(minutes=30)  # Estimated duration
                
                timeline.append({
                    'decision_id': decision.decision_id,
                    'resource_type': decision.resource_type.value,
                    'scaling_action': decision.scaling_action.value,
                    'scheduled_start': start_time.isoformat(),
                    'estimated_duration_minutes': 30,
                    'priority': decision.metadata.get('priority', 5),
                    'dependencies': []  # Could add dependencies between decisions
                })
            
        except Exception as e:
            logger.error(f"Error creating timeline: {e}")
            timeline = [{'error': str(e)}]
        
        return timeline
    
    def _create_rollback_plan(self, decisions: List[AllocationDecision]) -> List[str]:
        """Create rollback plan for allocation changes"""
        rollback_steps = []
        
        try:
            rollback_steps.extend([
                "1. Monitor all resource changes for 30 minutes after implementation",
                "2. If performance degradation detected, begin rollback procedure",
                "3. Rollback critical resources (CPU, Memory) first",
                "4. Restore previous allocation settings",
                "5. Validate system stability after rollback",
                "6. Document issues and update allocation models"
            ])
            
            # Add specific rollback steps for each decision
            for decision in decisions:
                if decision.scaling_action != ScalingAction.MAINTAIN:
                    resource_name = decision.resource_type.value.replace('_', ' ')
                    rollback_steps.append(
                        f"Rollback {resource_name}: restore to {decision.current_allocation:.1f} units"
                    )
            
        except Exception as e:
            rollback_steps = [f"Error creating rollback plan: {str(e)}"]
        
        return rollback_steps
    
    async def optimize_costs(self, target_reduction_percent: float = 10.0) -> Dict[str, Any]:
        """Optimize resource allocation for cost reduction"""
        try:
            # Analyze current resource requirements
            requirements = await self.analyze_resource_requirements()
            
            # Identify cost optimization opportunities
            optimizations = []
            total_current_cost = 0.0
            total_optimized_cost = 0.0
            
            for resource_type, requirement in requirements.items():
                current_cost = requirement.current_allocation * requirement.cost_per_unit
                total_current_cost += current_cost
                
                # Calculate minimum viable allocation
                current_util = requirement.metadata.get('current_utilization', 0.5)
                if current_util < 0.6:  # Under-utilized resource
                    # Reduce allocation while maintaining minimum thresholds
                    min_viable = requirement.current_allocation * max(0.5, current_util / 0.7)
                    min_viable = max(self.resource_configs[resource_type]['min_allocation'], min_viable)
                    
                    optimized_cost = min_viable * requirement.cost_per_unit
                    cost_savings = current_cost - optimized_cost
                    
                    if cost_savings > 0:
                        optimizations.append({
                            'resource_type': resource_type.value,
                            'current_allocation': requirement.current_allocation,
                            'optimized_allocation': min_viable,
                            'current_cost_per_hour': current_cost,
                            'optimized_cost_per_hour': optimized_cost,
                            'cost_savings_per_hour': cost_savings,
                            'utilization': current_util
                        })
                        total_optimized_cost += optimized_cost
                    else:
                        total_optimized_cost += current_cost
                else:
                    total_optimized_cost += current_cost
            
            total_savings = total_current_cost - total_optimized_cost
            savings_percent = (total_savings / total_current_cost * 100) if total_current_cost > 0 else 0
            
            return {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'target_reduction_percent': target_reduction_percent,
                'achievable_reduction_percent': savings_percent,
                'current_cost_per_hour': total_current_cost,
                'optimized_cost_per_hour': total_optimized_cost,
                'potential_savings_per_hour': total_savings,
                'potential_monthly_savings': total_savings * 24 * 30,
                'optimizations': optimizations,
                'recommendation': (
                    'Significant cost savings available' if savings_percent >= target_reduction_percent
                    else 'Limited cost optimization opportunities'
                )
            }
            
        except Exception as e:
            logger.error(f"Error optimizing costs: {e}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_allocation_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get insights from resource allocation history"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            recent_plans = [p for p in self.allocation_history if p.timestamp >= cutoff_time]
            
            if not recent_plans:
                return {'message': 'No recent allocation history available'}
            
            # Analyze allocation trends
            scaling_actions = defaultdict(int)
            cost_impacts = []
            performance_impacts = []
            
            for plan in recent_plans:
                for decision in plan.decisions:
                    scaling_actions[decision.scaling_action.value] += 1
                    cost_impacts.append(decision.cost_impact)
                    performance_impacts.append(decision.performance_impact)
            
            insights = {
                'analysis_period_days': days,
                'total_plans': len(recent_plans),
                'scaling_action_distribution': dict(scaling_actions),
                'cost_impact_analysis': {
                    'total_cost_changes': sum(cost_impacts),
                    'average_cost_impact': sum(cost_impacts) / len(cost_impacts) if cost_impacts else 0,
                    'cost_increases': len([c for c in cost_impacts if c > 0]),
                    'cost_decreases': len([c for c in cost_impacts if c < 0])
                },
                'performance_impact_analysis': {
                    'average_performance_impact': sum(performance_impacts) / len(performance_impacts) if performance_impacts else 0,
                    'positive_impacts': len([p for p in performance_impacts if p > 0]),
                    'negative_impacts': len([p for p in performance_impacts if p < 0])
                },
                'resource_type_frequency': self._analyze_resource_frequency(recent_plans),
                'strategy_effectiveness': self._analyze_strategy_effectiveness(recent_plans)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting allocation insights: {e}")
            return {'error': str(e)}
    
    def _analyze_resource_frequency(self, plans: List[ResourceAllocationPlan]) -> Dict[str, int]:
        """Analyze which resource types are modified most frequently"""
        frequency = defaultdict(int)
        
        for plan in plans:
            for decision in plan.decisions:
                if decision.scaling_action != ScalingAction.MAINTAIN:
                    frequency[decision.resource_type.value] += 1
        
        return dict(frequency)
    
    def _analyze_strategy_effectiveness(self, plans: List[ResourceAllocationPlan]) -> Dict[str, Any]:
        """Analyze effectiveness of different allocation strategies"""
        try:
            strategy_stats = defaultdict(lambda: {'count': 0, 'avg_cost_impact': 0, 'avg_performance_impact': 0})
            
            for plan in plans:
                strategy = plan.strategy.value
                strategy_stats[strategy]['count'] += 1
                strategy_stats[strategy]['avg_cost_impact'] += plan.total_cost_impact
                strategy_stats[strategy]['avg_performance_impact'] += plan.expected_performance_improvement
            
            # Calculate averages
            for strategy, stats in strategy_stats.items():
                if stats['count'] > 0:
                    stats['avg_cost_impact'] /= stats['count']
                    stats['avg_performance_impact'] /= stats['count']
            
            return dict(strategy_stats)
            
        except Exception as e:
            logger.error(f"Error analyzing strategy effectiveness: {e}")
            return {}

# Global resource allocator instance
resource_allocator = IntelligentResourceAllocator()

# Convenience functions
async def analyze_resource_requirements() -> Dict[ResourceType, ResourceRequirement]:
    """Analyze current resource requirements"""
    return await resource_allocator.analyze_resource_requirements()

async def create_allocation_plan(requirements: Dict[ResourceType, ResourceRequirement] = None,
                               time_horizon: timedelta = timedelta(hours=12)) -> ResourceAllocationPlan:
    """Create resource allocation plan"""
    if requirements is None:
        requirements = await resource_allocator.analyze_resource_requirements()
    return await resource_allocator.create_allocation_plan(requirements, time_horizon)

async def optimize_costs(target_reduction_percent: float = 10.0) -> Dict[str, Any]:
    """Optimize resource allocation for cost reduction"""
    return await resource_allocator.optimize_costs(target_reduction_percent)

async def get_allocation_insights(days: int = 7) -> Dict[str, Any]:
    """Get resource allocation insights"""
    return await resource_allocator.get_allocation_insights(days)

__all__ = [
    'IntelligentResourceAllocator',
    'resource_allocator',
    'ResourceType',
    'AllocationStrategy',
    'ScalingAction',
    'ResourceRequirement',
    'AllocationDecision',
    'ResourceAllocationPlan',
    'analyze_resource_requirements',
    'create_allocation_plan',
    'optimize_costs',
    'get_allocation_insights'
]