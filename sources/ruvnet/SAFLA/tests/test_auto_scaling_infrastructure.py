"""
Test suite for Auto-Scaling Infrastructure - Dynamic resource allocation and scaling.

This module tests the auto-scaling infrastructure capabilities for SAFLA:
- Dynamic resource allocation and deallocation
- Predictive scaling based on ML models
- Multi-dimensional scaling (CPU, memory, storage, network)
- Cost-aware scaling optimization
- Performance-based scaling decisions

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import numpy as np
import torch
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass, field
from enum import Enum

# Import the classes we'll implement
from safla.core.auto_scaling_infrastructure import (
    AutoScalingInfrastructure,
    ScalingDirection,
    ResourceType,
    ScalingPolicy,
    ResourceMetrics,
    ScalingRule,
    ScalingEvent,
    Instance,
    LoadBalancer,
    PerformanceOptimizer
)


class ResourceType(Enum):
    """Resource type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    INSTANCE = "instance"


class ScalingAction(Enum):
    """Scaling action enumeration."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"
    EMERGENCY_SCALE = "emergency_scale"


class ScalingTrigger(Enum):
    """Scaling trigger enumeration."""
    THRESHOLD_BASED = "threshold_based"
    PREDICTIVE = "predictive"
    SCHEDULE_BASED = "schedule_based"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_TARGET = "performance_target"
    MANUAL = "manual"


@dataclass
class MockResourceMetrics:
    """Mock resource metrics for testing."""
    resource_type: ResourceType
    current_usage: float
    capacity: float
    utilization: float
    cost_per_hour: float
    performance_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def usage_percentage(self) -> float:
        return (self.current_usage / self.capacity) * 100 if self.capacity > 0 else 0


@dataclass
class MockScalingDecision:
    """Mock scaling decision for testing."""
    resource_type: ResourceType
    action: ScalingAction
    target_capacity: float
    current_capacity: float
    confidence: float
    estimated_cost: float
    expected_performance: float
    trigger: ScalingTrigger
    timestamp: datetime = field(default_factory=datetime.now)


class TestAutoScalingManager:
    """Test suite for Auto-Scaling Manager functionality."""
    
    @pytest.fixture
    def scaling_config(self):
        """Create scaling configuration for testing."""
        return ScalingConfig(
            min_instances=2,
            max_instances=20,
            target_cpu_utilization=70.0,
            target_memory_utilization=80.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            scale_up_cooldown=300,  # 5 minutes
            scale_down_cooldown=600,  # 10 minutes
            prediction_window=3600,  # 1 hour
            cost_optimization_enabled=True,
            performance_targets={
                "response_time": 500,  # ms
                "throughput": 1000,  # requests/sec
                "availability": 99.9  # percentage
            },
            scaling_strategies=["reactive", "predictive", "cost_aware"]
        )
    
    @pytest.fixture
    def auto_scaling_manager(self, scaling_config):
        """Create AutoScalingInfrastructure instance for testing."""
        return AutoScalingInfrastructure(config=scaling_config)
    
    @pytest.fixture
    def mock_resource_metrics(self):
        """Create mock resource metrics for testing."""
        return [
            MockResourceMetrics(
                resource_type=ResourceType.CPU,
                current_usage=60.0,
                capacity=100.0,
                utilization=60.0,
                cost_per_hour=0.10,
                performance_score=0.8
            ),
            MockResourceMetrics(
                resource_type=ResourceType.MEMORY,
                current_usage=70.0,
                capacity=100.0,
                utilization=70.0,
                cost_per_hour=0.05,
                performance_score=0.85
            ),
            MockResourceMetrics(
                resource_type=ResourceType.STORAGE,
                current_usage=40.0,
                capacity=100.0,
                utilization=40.0,
                cost_per_hour=0.02,
                performance_score=0.9
            ),
            MockResourceMetrics(
                resource_type=ResourceType.NETWORK,
                current_usage=30.0,
                capacity=100.0,
                utilization=30.0,
                cost_per_hour=0.08,
                performance_score=0.95
            )
        ]
    
    def test_auto_scaling_manager_initialization(self, auto_scaling_manager, scaling_config):
        """Test AutoScalingInfrastructure initialization."""
        assert auto_scaling_manager.config == scaling_config
        assert auto_scaling_manager.resource_allocator is not None
        assert auto_scaling_manager.scaling_predictor is not None
        assert auto_scaling_manager.cost_optimizer is not None
        assert auto_scaling_manager.performance_monitor is not None
        assert auto_scaling_manager.load_balancer is not None
        assert auto_scaling_manager.is_scaling_enabled is True
        assert len(auto_scaling_manager.active_policies) == 0
    
    def test_scaling_policy_management(self, auto_scaling_manager):
        """Test scaling policy registration and management."""
        # Create scaling policies
        cpu_policy = ScalingPolicy(
            policy_id="cpu_scaling",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.THRESHOLD_BASED,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            min_capacity=2,
            max_capacity=20,
            cooldown_period=300
        )
        
        memory_policy = ScalingPolicy(
            policy_id="memory_scaling",
            resource_type=ResourceType.MEMORY,
            trigger=ScalingTrigger.PREDICTIVE,
            scale_up_threshold=85.0,
            scale_down_threshold=40.0,
            min_capacity=4,
            max_capacity=32,
            cooldown_period=600
        )
        
        # Register policies
        success1 = auto_scaling_manager.register_policy(cpu_policy)
        success2 = auto_scaling_manager.register_policy(memory_policy)
        
        assert success1 and success2
        assert len(auto_scaling_manager.get_active_policies()) == 2
        
        # Test policy retrieval
        retrieved_policy = auto_scaling_manager.get_policy("cpu_scaling")
        assert retrieved_policy is not None
        assert retrieved_policy.resource_type == ResourceType.CPU
        
        # Test policy deactivation
        success = auto_scaling_manager.deactivate_policy("memory_scaling")
        assert success
        assert len(auto_scaling_manager.get_active_policies()) == 1
    
    @pytest.mark.asyncio
    async def test_threshold_based_scaling(self, auto_scaling_manager, mock_resource_metrics):
        """Test threshold-based scaling decisions."""
        # Register CPU scaling policy
        cpu_policy = ScalingPolicy(
            policy_id="cpu_threshold",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.THRESHOLD_BASED,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
            min_capacity=2,
            max_capacity=20
        )
        auto_scaling_manager.register_policy(cpu_policy)
        
        # Test scale-up scenario (high CPU usage)
        high_cpu_metrics = mock_resource_metrics.copy()
        high_cpu_metrics[0].current_usage = 85.0
        high_cpu_metrics[0].utilization = 85.0
        
        scaling_decisions = await auto_scaling_manager.evaluate_scaling_decisions(high_cpu_metrics)
        
        cpu_decision = next((d for d in scaling_decisions if d.resource_type == ResourceType.CPU), None)
        assert cpu_decision is not None
        assert cpu_decision.action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]
        assert cpu_decision.trigger == ScalingTrigger.THRESHOLD_BASED
        
        # Test scale-down scenario (low CPU usage)
        low_cpu_metrics = mock_resource_metrics.copy()
        low_cpu_metrics[0].current_usage = 20.0
        low_cpu_metrics[0].utilization = 20.0
        
        scaling_decisions = await auto_scaling_manager.evaluate_scaling_decisions(low_cpu_metrics)
        
        cpu_decision = next((d for d in scaling_decisions if d.resource_type == ResourceType.CPU), None)
        if cpu_decision:  # May be None if cooldown is active
            assert cpu_decision.action in [ScalingAction.SCALE_DOWN, ScalingAction.SCALE_IN]
    
    @pytest.mark.asyncio
    async def test_predictive_scaling(self, auto_scaling_manager, mock_resource_metrics):
        """Test predictive scaling based on ML models."""
        # Register predictive scaling policy
        predictive_policy = ScalingPolicy(
            policy_id="predictive_cpu",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.PREDICTIVE,
            prediction_window=3600,  # 1 hour
            confidence_threshold=0.8
        )
        auto_scaling_manager.register_policy(predictive_policy)
        
        # Mock historical metrics for prediction
        historical_metrics = []
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=i)
            # Simulate increasing load pattern
            base_usage = 50 + (i / 100) * 30 + np.random.normal(0, 5)
            base_usage = max(0, min(100, base_usage))
            
            metrics = MockResourceMetrics(
                resource_type=ResourceType.CPU,
                current_usage=base_usage,
                capacity=100.0,
                utilization=base_usage,
                cost_per_hour=0.10,
                performance_score=0.8,
                timestamp=timestamp
            )
            historical_metrics.append(metrics)
        
        # Set historical data for prediction
        auto_scaling_manager.scaling_predictor.set_historical_data(historical_metrics)
        
        # Evaluate predictive scaling
        scaling_decisions = await auto_scaling_manager.evaluate_scaling_decisions(mock_resource_metrics)
        
        predictive_decision = next((d for d in scaling_decisions if d.trigger == ScalingTrigger.PREDICTIVE), None)
        if predictive_decision:
            assert predictive_decision.confidence > 0.0
            assert predictive_decision.action != ScalingAction.NO_ACTION
    
    def test_cost_aware_scaling_optimization(self, auto_scaling_manager, mock_resource_metrics):
        """Test cost-aware scaling optimization."""
        cost_optimizer = auto_scaling_manager.cost_optimizer
        
        # Define cost models for different resource types
        cost_models = {
            ResourceType.CPU: CostModel(
                base_cost=0.10,
                scaling_factor=1.2,
                efficiency_curve="linear"
            ),
            ResourceType.MEMORY: CostModel(
                base_cost=0.05,
                scaling_factor=1.1,
                efficiency_curve="logarithmic"
            ),
            ResourceType.STORAGE: CostModel(
                base_cost=0.02,
                scaling_factor=1.05,
                efficiency_curve="linear"
            )
        }
        
        # Test cost optimization for scaling decisions
        scaling_options = [
            {
                "resource_type": ResourceType.CPU,
                "action": ScalingAction.SCALE_UP,
                "target_capacity": 120.0,
                "expected_performance": 0.9
            },
            {
                "resource_type": ResourceType.MEMORY,
                "action": ScalingAction.SCALE_UP,
                "target_capacity": 110.0,
                "expected_performance": 0.85
            }
        ]
        
        optimized_decisions = cost_optimizer.optimize_scaling_decisions(
            current_metrics=mock_resource_metrics,
            scaling_options=scaling_options,
            cost_models=cost_models,
            performance_targets=auto_scaling_manager.config.performance_targets
        )
        
        assert len(optimized_decisions) > 0
        
        # Verify cost-effectiveness
        for decision in optimized_decisions:
            assert decision.estimated_cost > 0
            assert decision.expected_performance > 0
            # Cost-performance ratio should be reasonable
            cost_performance_ratio = decision.estimated_cost / decision.expected_performance
            assert cost_performance_ratio < 1.0  # Arbitrary threshold for testing
    
    @pytest.mark.asyncio
    async def test_multi_dimensional_scaling(self, auto_scaling_manager, mock_resource_metrics):
        """Test multi-dimensional scaling across different resource types."""
        # Register policies for multiple resource types
        policies = [
            ScalingPolicy(
                policy_id="cpu_policy",
                resource_type=ResourceType.CPU,
                trigger=ScalingTrigger.THRESHOLD_BASED,
                scale_up_threshold=75.0,
                scale_down_threshold=25.0
            ),
            ScalingPolicy(
                policy_id="memory_policy",
                resource_type=ResourceType.MEMORY,
                trigger=ScalingTrigger.THRESHOLD_BASED,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0
            ),
            ScalingPolicy(
                policy_id="storage_policy",
                resource_type=ResourceType.STORAGE,
                trigger=ScalingTrigger.PREDICTIVE,
                prediction_window=7200  # 2 hours
            )
        ]
        
        for policy in policies:
            auto_scaling_manager.register_policy(policy)
        
        # Create metrics that trigger scaling for multiple resources
        high_usage_metrics = [
            MockResourceMetrics(
                resource_type=ResourceType.CPU,
                current_usage=85.0,  # Above threshold
                capacity=100.0,
                utilization=85.0,
                cost_per_hour=0.10,
                performance_score=0.7
            ),
            MockResourceMetrics(
                resource_type=ResourceType.MEMORY,
                current_usage=90.0,  # Above threshold
                capacity=100.0,
                utilization=90.0,
                cost_per_hour=0.05,
                performance_score=0.6
            ),
            MockResourceMetrics(
                resource_type=ResourceType.STORAGE,
                current_usage=60.0,
                capacity=100.0,
                utilization=60.0,
                cost_per_hour=0.02,
                performance_score=0.8
            )
        ]
        
        # Evaluate multi-dimensional scaling
        scaling_decisions = await auto_scaling_manager.evaluate_scaling_decisions(high_usage_metrics)
        
        # Should have scaling decisions for CPU and Memory
        cpu_decision = next((d for d in scaling_decisions if d.resource_type == ResourceType.CPU), None)
        memory_decision = next((d for d in scaling_decisions if d.resource_type == ResourceType.MEMORY), None)
        
        assert cpu_decision is not None
        assert memory_decision is not None
        assert cpu_decision.action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]
        assert memory_decision.action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]
        
        # Test coordinated scaling to avoid resource imbalances
        coordinated_decisions = auto_scaling_manager.coordinate_scaling_decisions(scaling_decisions)
        
        assert len(coordinated_decisions) > 0
        # Verify no conflicting decisions
        resource_actions = {}
        for decision in coordinated_decisions:
            if decision.resource_type in resource_actions:
                # Should not have conflicting actions for same resource
                assert resource_actions[decision.resource_type] == decision.action
            resource_actions[decision.resource_type] = decision.action
    
    @pytest.mark.asyncio
    async def test_performance_target_scaling(self, auto_scaling_manager):
        """Test scaling based on performance targets."""
        # Set performance targets
        performance_targets = {
            "response_time": 500,  # ms
            "throughput": 1000,  # requests/sec
            "availability": 99.9,  # percentage
            "error_rate": 0.01  # 1%
        }
        
        auto_scaling_manager.config.performance_targets = performance_targets
        
        # Register performance-based scaling policy
        performance_policy = ScalingPolicy(
            policy_id="performance_scaling",
            resource_type=ResourceType.INSTANCE,
            trigger=ScalingTrigger.PERFORMANCE_TARGET,
            performance_targets=performance_targets
        )
        auto_scaling_manager.register_policy(performance_policy)
        
        # Simulate performance metrics that don't meet targets
        poor_performance_metrics = {
            "response_time": 800,  # Above target
            "throughput": 600,  # Below target
            "availability": 99.5,  # Below target
            "error_rate": 0.05  # Above target
        }
        
        # Evaluate performance-based scaling
        scaling_decision = await auto_scaling_manager.evaluate_performance_scaling(
            current_performance=poor_performance_metrics,
            targets=performance_targets
        )
        
        assert scaling_decision is not None
        assert scaling_decision.action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]
        assert scaling_decision.trigger == ScalingTrigger.PERFORMANCE_TARGET
        
        # Test with good performance (should not scale)
        good_performance_metrics = {
            "response_time": 300,  # Below target
            "throughput": 1200,  # Above target
            "availability": 99.95,  # Above target
            "error_rate": 0.005  # Below target
        }
        
        scaling_decision = await auto_scaling_manager.evaluate_performance_scaling(
            current_performance=good_performance_metrics,
            targets=performance_targets
        )
        
        if scaling_decision:
            assert scaling_decision.action == ScalingAction.NO_ACTION
    
    def test_cooldown_period_enforcement(self, auto_scaling_manager):
        """Test cooldown period enforcement for scaling actions."""
        # Register policy with cooldown
        policy = ScalingPolicy(
            policy_id="cooldown_test",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.THRESHOLD_BASED,
            scale_up_threshold=80.0,
            cooldown_period=300  # 5 minutes
        )
        auto_scaling_manager.register_policy(policy)
        
        # Record a recent scaling action
        recent_action = {
            "resource_type": ResourceType.CPU,
            "action": ScalingAction.SCALE_UP,
            "timestamp": datetime.now() - timedelta(minutes=2),  # 2 minutes ago
            "policy_id": "cooldown_test"
        }
        auto_scaling_manager.record_scaling_action(recent_action)
        
        # Test that scaling is blocked during cooldown
        is_in_cooldown = auto_scaling_manager.is_in_cooldown(
            resource_type=ResourceType.CPU,
            policy_id="cooldown_test"
        )
        
        assert is_in_cooldown
        
        # Test that scaling is allowed after cooldown
        old_action = {
            "resource_type": ResourceType.CPU,
            "action": ScalingAction.SCALE_UP,
            "timestamp": datetime.now() - timedelta(minutes=10),  # 10 minutes ago
            "policy_id": "cooldown_test"
        }
        auto_scaling_manager.record_scaling_action(old_action)
        
        is_in_cooldown = auto_scaling_manager.is_in_cooldown(
            resource_type=ResourceType.CPU,
            policy_id="cooldown_test"
        )
        
        assert not is_in_cooldown
    
    @pytest.mark.asyncio
    async def test_emergency_scaling(self, auto_scaling_manager, mock_resource_metrics):
        """Test emergency scaling for critical situations."""
        # Create critical resource situation
        critical_metrics = [
            MockResourceMetrics(
                resource_type=ResourceType.CPU,
                current_usage=98.0,  # Critical usage
                capacity=100.0,
                utilization=98.0,
                cost_per_hour=0.10,
                performance_score=0.2  # Poor performance
            ),
            MockResourceMetrics(
                resource_type=ResourceType.MEMORY,
                current_usage=95.0,  # Critical usage
                capacity=100.0,
                utilization=95.0,
                cost_per_hour=0.05,
                performance_score=0.3  # Poor performance
            )
        ]
        
        # Trigger emergency scaling evaluation
        emergency_decisions = await auto_scaling_manager.evaluate_emergency_scaling(critical_metrics)
        
        assert len(emergency_decisions) > 0
        
        for decision in emergency_decisions:
            assert decision.action == ScalingAction.EMERGENCY_SCALE
            assert decision.confidence > 0.9  # High confidence for emergency
            # Emergency scaling should ignore cooldowns
            assert decision.target_capacity > decision.current_capacity
    
    def test_scaling_metrics_and_monitoring(self, auto_scaling_manager):
        """Test scaling metrics collection and monitoring."""
        performance_monitor = auto_scaling_manager.performance_monitor
        
        # Record scaling events
        scaling_events = [
            {
                "timestamp": datetime.now() - timedelta(hours=i),
                "resource_type": ResourceType.CPU,
                "action": ScalingAction.SCALE_UP,
                "trigger": ScalingTrigger.THRESHOLD_BASED,
                "success": True,
                "duration": 30 + (i * 5),  # seconds
                "cost_impact": 0.05 * (i + 1)
            }
            for i in range(10)
        ]
        
        for event in scaling_events:
            performance_monitor.record_scaling_event(event)
        
        # Analyze scaling performance
        scaling_analysis = performance_monitor.analyze_scaling_performance(
            time_window=timedelta(hours=24)
        )
        
        assert "total_scaling_events" in scaling_analysis
        assert "success_rate" in scaling_analysis
        assert "average_duration" in scaling_analysis
        assert "cost_impact" in scaling_analysis
        assert "efficiency_metrics" in scaling_analysis
        
        assert scaling_analysis["total_scaling_events"] == 10
        assert 0.0 <= scaling_analysis["success_rate"] <= 1.0
        
        # Generate scaling recommendations
        recommendations = performance_monitor.generate_scaling_recommendations(scaling_analysis)
        
        assert "policy_adjustments" in recommendations
        assert "threshold_optimizations" in recommendations
        assert "cost_optimizations" in recommendations


class TestResourceAllocator:
    """Test suite for Resource Allocator functionality."""
    
    @pytest.fixture
    def resource_allocator(self):
        """Create ResourceAllocator instance for testing."""
        config = {
            "allocation_strategy": "best_fit",
            "resource_pools": {
                ResourceType.CPU: {"total_capacity": 1000, "available": 600},
                ResourceType.MEMORY: {"total_capacity": 2000, "available": 1200},
                ResourceType.STORAGE: {"total_capacity": 5000, "available": 3000}
            },
            "allocation_constraints": {
                "max_allocation_per_request": 0.3,  # 30% of total capacity
                "min_available_buffer": 0.1  # 10% buffer
            }
        }
        return ResourceAllocator(config=config)
    
    def test_resource_allocation_request(self, resource_allocator):
        """Test resource allocation requests."""
        # Test successful allocation
        allocation_request = {
            "resource_type": ResourceType.CPU,
            "requested_amount": 100,
            "priority": "high",
            "duration": 3600,  # 1 hour
            "requester_id": "test_service"
        }
        
        allocation_result = resource_allocator.allocate_resources(allocation_request)
        
        assert allocation_result["success"]
        assert allocation_result["allocated_amount"] == 100
        assert "allocation_id" in allocation_result
        
        # Test allocation that exceeds available resources
        large_request = {
            "resource_type": ResourceType.CPU,
            "requested_amount": 800,  # More than available (600)
            "priority": "medium",
            "duration": 1800,
            "requester_id": "large_service"
        }
        
        allocation_result = resource_allocator.allocate_resources(large_request)
        
        assert not allocation_result["success"]
        assert "insufficient_resources" in allocation_result["reason"]
    
    def test_resource_deallocation(self, resource_allocator):
        """Test resource deallocation."""
        # First allocate resources
        allocation_request = {
            "resource_type": ResourceType.MEMORY,
            "requested_amount": 200,
            "priority": "medium",
            "duration": 1800,
            "requester_id": "test_service"
        }
        
        allocation_result = resource_allocator.allocate_resources(allocation_request)
        allocation_id = allocation_result["allocation_id"]
        
        # Check available resources decreased
        available_before = resource_allocator.get_available_resources(ResourceType.MEMORY)
        assert available_before == 1000  # 1200 - 200
        
        # Deallocate resources
        deallocation_result = resource_allocator.deallocate_resources(allocation_id)
        
        assert deallocation_result["success"]
        
        # Check available resources increased back
        available_after = resource_allocator.get_available_resources(ResourceType.MEMORY)
        assert available_after == 1200  # Back to original
    
    def test_resource_pool_management(self, resource_allocator):
        """Test resource pool management and optimization."""
        # Test pool expansion
        expansion_result = resource_allocator.expand_resource_pool(
            resource_type=ResourceType.CPU,
            additional_capacity=200
        )
        
        assert expansion_result["success"]
        assert resource_allocator.get_total_capacity(ResourceType.CPU) == 1200  # 1000 + 200
        
        # Test pool contraction
        contraction_result = resource_allocator.contract_resource_pool(
            resource_type=ResourceType.CPU,
            reduction_amount=100
        )
        
        assert contraction_result["success"]
        assert resource_allocator.get_total_capacity(ResourceType.CPU) == 1100  # 1200 - 100
    
    def test_allocation_optimization(self, resource_allocator):
        """Test allocation optimization algorithms."""
        # Create multiple allocation requests
        requests = [
            {"resource_type": ResourceType.CPU, "requested_amount": 150, "priority": "high"},
            {"resource_type": ResourceType.CPU, "requested_amount": 100, "priority": "medium"},
            {"resource_type": ResourceType.CPU, "requested_amount": 200, "priority": "low"},
            {"resource_type": ResourceType.MEMORY, "requested_amount": 300, "priority": "high"},
            {"resource_type": ResourceType.MEMORY, "requested_amount": 250, "priority": "medium"}
        ]
        
        # Optimize allocation order
        optimized_allocation = resource_allocator.optimize_allocation_order(requests)
        
        assert len(optimized_allocation) == len(requests)
        
        # High priority requests should be first
        high_priority_indices = [i for i, req in enumerate(optimized_allocation) if req["priority"] == "high"]
        medium_priority_indices = [i for i, req in enumerate(optimized_allocation) if req["priority"] == "medium"]
        
        assert all(h < m for h in high_priority_indices for m in medium_priority_indices)
    
    def test_resource_fragmentation_handling(self, resource_allocator):
        """Test handling of resource fragmentation."""
        # Create fragmented allocation pattern
        small_allocations = []
        for i in range(10):
            request = {
                "resource_type": ResourceType.MEMORY,
                "requested_amount": 50,
                "priority": "medium",
                "duration": 1800,
                "requester_id": f"service_{i}"
            }
            result = resource_allocator.allocate_resources(request)
            if result["success"]:
                small_allocations.append(result["allocation_id"])
        
        # Deallocate every other allocation to create fragmentation
        for i in range(0, len(small_allocations), 2):
            resource_allocator.deallocate_resources(small_allocations[i])
        
        # Test defragmentation
        defrag_result = resource_allocator.defragment_resources(ResourceType.MEMORY)
        
        assert defrag_result["success"]
        assert "fragmentation_reduced" in defrag_result
        assert defrag_result["fragmentation_reduced"] > 0


class TestScalingPredictor:
    """Test suite for Scaling Predictor functionality."""
    
    @pytest.fixture
    def scaling_predictor(self):
        """Create ScalingPredictor instance for testing."""
        config = {
            "prediction_models": ["lstm", "arima", "linear_regression"],
            "prediction_window": 3600,  # 1 hour
            "training_window": 86400,  # 24 hours
            "confidence_threshold": 0.8,
            "feature_engineering": True,
            "ensemble_method": "weighted_average"
        }
        return ScalingPredictor(config=config)
    
    def test_predictor_initialization(self, scaling_predictor):
        """Test ScalingPredictor initialization."""
        assert scaling_predictor.models is not None
        assert len(scaling_predictor.models) == 3
        assert scaling_predictor.feature_engineer is not None
        assert scaling_predictor.ensemble_predictor is not None
    
    def test_time_series_prediction(self, scaling_predictor):
        """Test time series prediction for resource usage."""
        # Generate synthetic time series data
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
        
        # Create realistic usage pattern with trend and seasonality
        base_usage = 50
        trend = np.linspace(0, 20, 100)  # Increasing trend
        seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, 100))  # Seasonal pattern
        noise = np.random.normal(0, 5, 100)
        
        usage_values = base_usage + trend + seasonal + noise
        usage_values = np.clip(usage_values, 0, 100)  # Keep within bounds
        
        time_series_data = [
            {"timestamp": ts, "usage": usage, "capacity": 100}
            for ts, usage in zip(timestamps, usage_values)
        ]
        
        # Train predictor
        training_result = scaling_predictor.train_models(time_series_data)
        
        assert training_result["success"]
        assert "model_performance" in training_result
        
        # Make predictions
        prediction_result = scaling_predictor.predict_usage(
            prediction_horizon=60,  # 60 minutes
            confidence_level=0.95
        )
        
        assert "predicted_values" in prediction_result
        assert "confidence_intervals" in prediction_result
        assert "prediction_confidence" in prediction_result
        
        assert len(prediction_result["predicted_values"]) == 60
        assert prediction_result["prediction_confidence"] > 0.0
    
    def test_feature_engineering(self, scaling_predictor):
        """Test feature engineering for prediction models."""
        # Create raw time series data
        raw_data = []
        for i in range(168):  # 1 week of hourly data
            timestamp = datetime.now() - timedelta(hours=i)
            usage = 50 + 20 * np.sin(i * 2 * np.pi / 24) + np.random.normal(0, 5)  # Daily pattern
            
            raw_data.append({
                "timestamp": timestamp,
                "cpu_usage": max(0, min(100, usage)),
                "memory_usage": max(0, min(100, usage + np.random.normal(0, 10))),
                "request_count": max(0, int(1000 + 500 * np.sin(i * 2 * np.pi / 24))),
                "response_time": max(100, 500 + 200 * (usage / 100))
            })
        
        # Engineer features
        engineered_features = scaling_predictor.engineer_features(raw_data)
        
        assert "temporal_features" in engineered_features
        assert "statistical_features" in engineered_features
        assert "lag_features" in engineered_features
        
        # Check temporal features
        temporal = engineered_features["temporal_features"]
        assert "hour_of_day" in temporal
        assert "day_of_week" in temporal
        assert "is_weekend" in temporal
        
        # Check statistical features
        statistical = engineered_features["statistical_features"]
        assert "rolling_mean" in statistical
        assert "rolling_std" in statistical
        assert "trend" in statistical
    
    def test_ensemble_prediction(self, scaling_predictor):
        """Test ensemble prediction combining multiple models."""
        # Mock individual model predictions
        model_predictions = {
            "lstm": {
                "predictions": np.random.uniform(40, 80, 60),
                "confidence": 0.85,
                "mse": 25.0
            },
            "arima": {
                "predictions": np.random.uniform(45, 75, 60),
                "confidence": 0.80,
                "mse": 30.0
            },
            "linear_regression": {
                "predictions": np.random.uniform(35, 85, 60),
                "confidence": 0.75,
                "mse": 35.0
            }
        }
        
        # Combine predictions using ensemble
        ensemble_result = scaling_predictor.ensemble_predict(model_predictions)
        
        assert "ensemble_predictions" in ensemble_result
        assert "ensemble_confidence" in ensemble_result
        assert "model_weights" in ensemble_result
        
        assert len(ensemble_result["ensemble_predictions"]) == 60
        assert 0.0 <= ensemble_result["ensemble_confidence"] <= 1.0
        
        # Higher performing models should have higher weights
        weights = ensemble_result["model_weights"]
        assert weights["lstm"] >= weights["arima"]  # LSTM has lower MSE
        assert weights["arima"] >= weights["linear_regression"]  # ARIMA has lower MSE
    
    def test_anomaly_detection_in_predictions(self, scaling_predictor):
        """Test anomaly detection in prediction inputs."""
        # Create data with anomalies
        normal_data = []
        for i in range(50):
            timestamp = datetime.now() - timedelta(minutes=i)
            usage = 50 + np.random.normal(0, 5)  # Normal usage
            normal_data.append({"timestamp": timestamp, "usage": usage})
        
        # Add anomalous data points
        anomalous_data = normal_data.copy()
        anomalous_data.extend([
            {"timestamp": datetime.now(), "usage": 150},  # Spike
            {"timestamp": datetime.now() - timedelta(minutes=1), "usage": -10},  # Impossible value
            {"timestamp": datetime.now() - timedelta(minutes=2), "usage": 200}  # Another spike
        ])
        
        # Detect anomalies
        anomaly_result = scaling_predictor.detect_anomalies(anomalous_data)
        
        assert "anomalies_detected" in anomaly_result
        assert "cleaned_data" in anomaly_result
        assert "anomaly_scores" in anomaly_result
        
        assert anomaly_result["anomalies_detected"] >= 2  # Should detect at least 2 anomalies
        assert len(anomaly_result["cleaned_data"]) < len(anomalous_data)  # Some data should be removed
    
    def test_prediction_accuracy_evaluation(self, scaling_predictor):
        """Test prediction accuracy evaluation and model selection."""
        # Create test data with known future values
        historical_data = []
        future_data = []
        
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=100-i)
            # Simple linear trend for predictable testing
            usage = 30 + (i * 0.5) + np.random.normal(0, 2)
            usage = max(0, min(100, usage))
            
            if i < 80:  # Historical data
                historical_data.append({"timestamp": timestamp, "usage": usage})
            else:  # Future data for validation
                future_data.append({"timestamp": timestamp, "usage": usage})
        
        # Train on historical data
        scaling_predictor.train_models(historical_data)
        
        # Make predictions for the future period
        predictions = scaling_predictor.predict_usage(prediction_horizon=20)
        
        # Evaluate accuracy against known future values
        actual_values = [d["usage"] for d in future_data]
        predicted_values = predictions["predicted_values"]
        
        evaluation_result = scaling_predictor.evaluate_prediction_accuracy(
            actual_values=actual_values,
            predicted_values=predicted_values
        )
        
        assert "mae" in evaluation_result  # Mean Absolute Error
        assert "rmse" in evaluation_result  # Root Mean Square Error
        assert "mape" in evaluation_result  # Mean Absolute Percentage Error
        assert "r2_score" in evaluation_result  # R-squared
        
        # For a simple linear trend, accuracy should be reasonable
        assert evaluation_result["mae"] < 20  # Reasonable error threshold
        assert evaluation_result["r2_score"] > 0.5  # Reasonable correlation


class TestCostOptimizer:
    """Test suite for Cost Optimizer functionality."""
    
    @pytest.fixture
    def cost_optimizer(self):
        """Create CostOptimizer instance for testing."""
        config = {
            "optimization_objective": "minimize_cost",
            "cost_models": {
                "on_demand": {"base_rate": 0.10, "scaling_factor": 1.0},
                "reserved": {"base_rate": 0.06, "scaling_factor": 0.8, "commitment": 8760},  # 1 year
                "spot": {"base_rate": 0.03, "scaling_factor": 1.2, "availability": 0.9}
            },
            "performance_constraints": {
                "min_availability": 0.99,
                "max_response_time": 1000,
                "min_throughput": 500
            }
        }
        return CostOptimizer(config=config)
    
    def test_cost_model_evaluation(self, cost_optimizer):
        """Test cost model evaluation for different pricing strategies."""
        # Test on-demand pricing
        on_demand_cost = cost_optimizer.calculate_cost(
            pricing_model="on_demand",
            resource_usage=100,
            duration=3600  # 1 hour
        )
        
        assert on_demand_cost > 0
        
        # Test reserved pricing (should be cheaper for long-term usage)
        reserved_cost = cost_optimizer.calculate_cost(
            pricing_model="reserved",
            resource_usage=100,
            duration=3600
        )
        
        assert reserved_cost < on_demand_cost
        
        # Test spot pricing (should be cheapest but less reliable)
        spot_cost = cost_optimizer.calculate_cost(
            pricing_model="spot",
            resource_usage=100,
            duration=3600
        )
        
        assert spot_cost < reserved_cost
    
    def test_cost_performance_optimization(self, cost_optimizer):
        """Test cost-performance optimization."""
        # Define optimization scenario
        optimization_request = {
            "target_performance": {
                "response_time": 500,  # ms
                "throughput": 1000,  # requests/sec
                "availability": 99.5  # percentage
            },
            "resource_options": [
                {"type": "small", "cost_per_hour": 0.05, "performance_score": 0.6},
                {"type": "medium", "cost_per_hour": 0.10, "performance_score": 0.8},
                {"type": "large", "cost_per_hour": 0.20, "performance_score": 0.95},
                {"type": "xlarge", "cost_per_hour": 0.40, "performance_score": 0.99}
            ],
            "duration": 8760,  # 1 year
            "budget_constraint": 1000  # dollars
        }
        
        # Optimize cost-performance trade-off
        optimization_result = cost_optimizer.optimize_cost_performance(optimization_request)
        
        assert "recommended_configuration" in optimization_result
        assert "total_cost" in optimization_result
        assert "expected_performance" in optimization_result
        assert "cost_efficiency" in optimization_result
        
        # Should meet performance targets within budget
        assert optimization_result["total_cost"] <= optimization_request["budget_constraint"]
        assert optimization_result["expected_performance"]["response_time"] <= 500
    
    def test_dynamic_pricing_optimization(self, cost_optimizer):
        """Test optimization with dynamic pricing."""
        # Simulate dynamic pricing over time
        pricing_history = []
        for hour in range(24):
            # Simulate daily pricing patterns
            base_price = 0.10
            demand_multiplier = 1 + 0.5 * np.sin(hour * 2 * np.pi / 24)  # Peak during day
            price = base_price * demand_multiplier
            
            pricing_history.append({
                "hour": hour,
                "price_per_unit": price,
                "availability": 0.95 + 0.05 * np.random.random()
            })
        
        # Optimize scheduling based on dynamic pricing
        workload_requirements = {
            "total_compute_hours": 100,
            "deadline": 24,  # hours
            "priority": "cost_sensitive",
            "min_performance": 0.8
        }
        
        scheduling_result = cost_optimizer.optimize_dynamic_scheduling(
            pricing_history=pricing_history,
            workload_requirements=workload_requirements
        )
        
        assert "optimal_schedule" in scheduling_result
        assert "total_cost" in scheduling_result
        assert "cost_savings" in scheduling_result
        
        # Should schedule more work during low-price periods
        schedule = scheduling_result["optimal_schedule"]
        low_price_hours = [h for h in pricing_history if h["price_per_unit"] < 0.12]
        high_price_hours = [h for h in pricing_history if h["price_per_unit"] > 0.15]
        
        low_price_allocation = sum(schedule.get(h["hour"], 0) for h in low_price_hours)
        high_price_allocation = sum(schedule.get(h["hour"], 0) for h in high_price_hours)
        
        assert low_price_allocation > high_price_allocation
    
    def test_multi_cloud_cost_optimization(self, cost_optimizer):
        """Test cost optimization across multiple cloud providers."""
        # Define multi-cloud pricing
        cloud_providers = {
            "aws": {
                "regions": {
                    "us-east-1": {"price": 0.10, "latency": 50, "availability": 0.999},
                    "us-west-2": {"price": 0.12, "latency": 80, "availability": 0.998}
                }
            },
            "azure": {
                "regions": {
                    "east-us": {"price": 0.09, "latency": 60, "availability": 0.998},
                    "west-us": {"price": 0.11, "latency": 90, "availability": 0.997}
                }
            },
            "gcp": {
                "regions": {
                    "us-central1": {"price": 0.08, "latency": 70, "availability": 0.999},
                    "us-west1": {"price": 0.10, "latency": 100, "availability": 0.998}
                }
            }
        }
        
        # Optimize across providers
        optimization_request = {
            "workload_requirements": {
                "compute_units": 1000,
                "max_latency": 75,  # ms
                "min_availability": 0.998,
                "duration": 720  # hours (1 month)
            },
            "constraints": {
                "max_providers": 2,  # Limit complexity
                "geographic_restrictions": ["us"]
            }
        }
        
        multi_cloud_result = cost_optimizer.optimize_multi_cloud_deployment(
            cloud_providers=cloud_providers,
            optimization_request=optimization_request
        )
        
        assert "optimal_deployment" in multi_cloud_result
        assert "total_cost" in multi_cloud_result
        assert "cost_breakdown" in multi_cloud_result
        
        # Should meet latency and availability constraints
        deployment = multi_cloud_result["optimal_deployment"]
        for provider, regions in deployment.items():
            for region, allocation in regions.items():
                if allocation > 0:
                    region_info = cloud_providers[provider]["regions"][region]
                    assert region_info["latency"] <= 75
                    assert region_info["availability"] >= 0.998