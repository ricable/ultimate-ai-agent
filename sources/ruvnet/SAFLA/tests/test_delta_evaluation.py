"""
Test suite for SAFLA Delta Evaluation System
Following TDD principles: Write failing tests first, then implement to pass

Based on research findings from research/04_synthesis/01_integrated_model.md
Delta Evaluation Formula: Δ_total = α₁ × Δ_performance + α₂ × Δ_efficiency + α₃ × Δ_stability + α₄ × Δ_capability
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Optional

# Import the classes we'll implement
from safla.core.delta_evaluation import (
    DeltaEvaluator,
    DeltaMetrics,
    DeltaResult,
    PerformanceDelta,
    EfficiencyDelta,
    StabilityDelta,
    CapabilityDelta,
    AdaptiveWeights
)


class TestDeltaMetrics:
    """Test the DeltaMetrics data structure"""
    
    def test_delta_metrics_creation(self):
        """Test that DeltaMetrics can be created with required fields"""
        metrics = DeltaMetrics(
            performance=0.15,
            efficiency=0.08,
            stability=0.95,
            capability=0.12
        )
        
        assert metrics.performance == 0.15
        assert metrics.efficiency == 0.08
        assert metrics.stability == 0.95
        assert metrics.capability == 0.12
    
    def test_delta_metrics_validation(self):
        """Test that DeltaMetrics validates input ranges"""
        # Stability delta can be negative (no validation needed)
        # This should not raise an error
        metrics = DeltaMetrics(performance=0.1, efficiency=0.1, stability=-0.2, capability=0.1)
        assert metrics.stability == -0.2
        
        # Capability should be between 0 and 1
        with pytest.raises(ValueError, match="Capability must be between 0 and 1"):
            DeltaMetrics(performance=0.1, efficiency=0.1, stability=0.9, capability=-0.1)


class TestDeltaResult:
    """Test the DeltaResult output structure"""
    
    def test_delta_result_creation(self):
        """Test that DeltaResult can be created with all components"""
        metrics = DeltaMetrics(0.15, 0.08, 0.95, 0.12)
        weights = AdaptiveWeights(0.4, 0.3, 0.2, 0.1)
        
        result = DeltaResult(
            total_delta=0.142,
            component_deltas=metrics,
            weights=weights,
            timestamp=1234567890,
            confidence=0.85
        )
        
        assert result.total_delta == 0.142
        assert result.component_deltas == metrics
        assert result.weights == weights
        assert result.timestamp == 1234567890
        assert result.confidence == 0.85
    
    def test_delta_result_improvement_detection(self):
        """Test that DeltaResult can detect if improvement occurred"""
        metrics = DeltaMetrics(0.15, 0.08, 0.95, 0.12)
        weights = AdaptiveWeights(0.4, 0.3, 0.2, 0.1)
        
        # Positive delta indicates improvement
        positive_result = DeltaResult(0.05, metrics, weights, 1234567890, 0.85)
        assert positive_result.is_improvement() is True
        
        # Negative delta indicates degradation
        negative_result = DeltaResult(-0.03, metrics, weights, 1234567890, 0.85)
        assert negative_result.is_improvement() is False
        
        # Zero delta indicates no change
        zero_result = DeltaResult(0.0, metrics, weights, 1234567890, 0.85)
        assert zero_result.is_improvement() is False


class TestPerformanceDelta:
    """Test the Performance Delta calculation component"""
    
    def test_performance_delta_calculation(self):
        """Test basic performance delta calculation: (reward_i - reward_i-1) / tokens_used_i"""
        calculator = PerformanceDelta()
        
        # Test improvement case
        current_reward = 100.0
        previous_reward = 80.0
        tokens_used = 1000
        
        delta = calculator.calculate(current_reward, previous_reward, tokens_used)
        expected = (100.0 - 80.0) / 1000  # 0.02
        assert delta == expected
    
    def test_performance_delta_degradation(self):
        """Test performance delta when performance degrades"""
        calculator = PerformanceDelta()
        
        current_reward = 70.0
        previous_reward = 90.0
        tokens_used = 500
        
        delta = calculator.calculate(current_reward, previous_reward, tokens_used)
        expected = (70.0 - 90.0) / 500  # -0.04
        assert delta == expected
    
    def test_performance_delta_zero_tokens_error(self):
        """Test that zero tokens raises an error"""
        calculator = PerformanceDelta()
        
        with pytest.raises(ValueError, match="Tokens used must be greater than zero"):
            calculator.calculate(100.0, 80.0, 0)
    
    def test_performance_delta_with_history(self):
        """Test performance delta calculation with historical data"""
        calculator = PerformanceDelta()
        
        # Add historical data points
        calculator.add_data_point(reward=50.0, tokens=800, timestamp=1000)
        calculator.add_data_point(reward=75.0, tokens=900, timestamp=2000)
        calculator.add_data_point(reward=90.0, tokens=1000, timestamp=3000)
        
        # Calculate delta for new data point
        delta = calculator.calculate_with_history(reward=100.0, tokens=1100, timestamp=4000)
        
        # Should compare with most recent (90.0 reward, 1000 tokens)
        expected = (100.0 - 90.0) / 1100  # ~0.009
        assert abs(delta - expected) < 1e-6


class TestEfficiencyDelta:
    """Test the Efficiency Delta calculation component"""
    
    def test_efficiency_delta_calculation(self):
        """Test basic efficiency delta: (throughput_i - throughput_i-1) / resource_used_i"""
        calculator = EfficiencyDelta()
        
        current_throughput = 150.0  # tasks per hour
        previous_throughput = 120.0
        resource_used = 100.0  # CPU hours
        
        delta = calculator.calculate(current_throughput, previous_throughput, resource_used)
        expected = (150.0 - 120.0) / 100.0  # 0.3
        assert delta == expected
    
    def test_efficiency_delta_with_multiple_resources(self):
        """Test efficiency calculation with multiple resource types"""
        calculator = EfficiencyDelta()
        
        current_throughput = 200.0
        previous_throughput = 180.0
        resources = {
            'cpu_hours': 50.0,
            'memory_gb_hours': 100.0,
            'gpu_hours': 10.0
        }
        
        delta = calculator.calculate_multi_resource(
            current_throughput, previous_throughput, resources
        )
        
        # Should normalize across resource types
        assert isinstance(delta, float)
        assert delta > 0  # Improvement case


class TestStabilityDelta:
    """Test the Stability Delta calculation component"""
    
    def test_stability_delta_calculation(self):
        """Test stability delta: 0.5 - divergence_score (relative to baseline)"""
        calculator = StabilityDelta()
        
        # Low divergence = positive stability delta
        low_divergence = 0.1
        delta = calculator.calculate(low_divergence)
        assert abs(delta - 0.4) < 1e-10  # 0.5 - 0.1 = 0.4
        
        # High divergence = negative stability delta
        high_divergence = 0.8
        delta = calculator.calculate(high_divergence)
        assert abs(delta - (-0.3)) < 1e-10  # 0.5 - 0.8 = -0.3
    
    def test_stability_delta_validation(self):
        """Test that divergence score is properly validated"""
        calculator = StabilityDelta()
        
        # Divergence score must be between 0 and 1
        with pytest.raises(ValueError, match="Divergence score must be between 0 and 1"):
            calculator.calculate(1.5)
        
        with pytest.raises(ValueError, match="Divergence score must be between 0 and 1"):
            calculator.calculate(-0.1)
    
    def test_stability_delta_with_trend_analysis(self):
        """Test stability calculation with trend analysis"""
        calculator = StabilityDelta()
        
        # Add historical divergence scores
        historical_scores = [0.1, 0.15, 0.12, 0.18, 0.14]
        current_score = 0.16
        
        delta = calculator.calculate_with_trend(current_score, historical_scores)
        
        # Should consider trend in addition to absolute value
        assert isinstance(delta, float)
        assert 0 <= delta <= 1


class TestCapabilityDelta:
    """Test the Capability Delta calculation component"""
    
    def test_capability_delta_calculation(self):
        """Test capability delta: new_capabilities_count / total_capabilities"""
        calculator = CapabilityDelta()
        
        new_capabilities = 3
        total_capabilities = 25
        
        delta = calculator.calculate(new_capabilities, total_capabilities)
        expected = 3 / 25  # 0.12
        assert delta == expected
    
    def test_capability_delta_no_new_capabilities(self):
        """Test capability delta when no new capabilities are added"""
        calculator = CapabilityDelta()
        
        delta = calculator.calculate(0, 20)
        assert delta == 0.0
    
    def test_capability_delta_validation(self):
        """Test capability delta input validation"""
        calculator = CapabilityDelta()
        
        # New capabilities cannot exceed total
        with pytest.raises(ValueError, match="New capabilities cannot exceed total capabilities"):
            calculator.calculate(10, 5)
        
        # Total capabilities must be positive
        with pytest.raises(ValueError, match="Total capabilities must be positive"):
            calculator.calculate(2, 0)


class TestAdaptiveWeights:
    """Test the Adaptive Weights system"""
    
    def test_adaptive_weights_creation(self):
        """Test that adaptive weights can be created and sum to 1"""
        weights = AdaptiveWeights(0.4, 0.3, 0.2, 0.1)
        
        assert weights.performance == 0.4
        assert weights.efficiency == 0.3
        assert weights.stability == 0.2
        assert weights.capability == 0.1
        assert abs(weights.sum() - 1.0) < 1e-6
    
    def test_adaptive_weights_validation(self):
        """Test that weights are validated to sum to 1"""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            AdaptiveWeights(0.5, 0.3, 0.2, 0.2)  # Sums to 1.2
    
    def test_adaptive_weights_adjustment(self):
        """Test that weights can be adjusted based on context"""
        weights = AdaptiveWeights(0.25, 0.25, 0.25, 0.25)
        
        # Adjust for performance-critical context
        adjusted = weights.adjust_for_context("performance_critical")
        assert adjusted.performance > 0.25  # Should increase performance weight
        assert abs(adjusted.sum() - 1.0) < 1e-6  # Should still sum to 1
        
        # Adjust for stability-critical context
        adjusted = weights.adjust_for_context("stability_critical")
        assert adjusted.stability > 0.25  # Should increase stability weight
        assert abs(adjusted.sum() - 1.0) < 1e-6


class TestDeltaEvaluator:
    """Test the main DeltaEvaluator class"""
    
    def test_delta_evaluator_creation(self):
        """Test that DeltaEvaluator can be created with default components"""
        evaluator = DeltaEvaluator()
        
        assert evaluator.performance_calculator is not None
        assert evaluator.efficiency_calculator is not None
        assert evaluator.stability_calculator is not None
        assert evaluator.capability_calculator is not None
        assert evaluator.weights is not None
    
    def test_delta_evaluator_with_custom_components(self):
        """Test that DeltaEvaluator can be created with custom components"""
        custom_performance = Mock(spec=PerformanceDelta)
        custom_weights = AdaptiveWeights(0.5, 0.2, 0.2, 0.1)
        
        evaluator = DeltaEvaluator(
            performance_calculator=custom_performance,
            weights=custom_weights
        )
        
        assert evaluator.performance_calculator == custom_performance
        assert evaluator.weights == custom_weights
    
    def test_evaluate_delta_complete_calculation(self):
        """Test complete delta evaluation with all components"""
        evaluator = DeltaEvaluator()
        
        # Mock the individual calculators
        evaluator.performance_calculator.calculate = Mock(return_value=0.15)
        evaluator.efficiency_calculator.calculate = Mock(return_value=0.08)
        evaluator.stability_calculator.calculate = Mock(return_value=0.45)  # 0.5 - 0.05 = 0.45
        evaluator.capability_calculator.calculate = Mock(return_value=0.12)
        
        # Set weights
        evaluator.weights = AdaptiveWeights(0.4, 0.3, 0.2, 0.1)
        
        # Prepare input data
        performance_data = {'current_reward': 100, 'previous_reward': 80, 'tokens_used': 1000}
        efficiency_data = {'current_throughput': 150, 'previous_throughput': 120, 'resource_used': 100}
        stability_data = {'divergence_score': 0.05}
        capability_data = {'new_capabilities': 3, 'total_capabilities': 25}
        
        result = evaluator.evaluate_delta(
            performance_data=performance_data,
            efficiency_data=efficiency_data,
            stability_data=stability_data,
            capability_data=capability_data
        )
        
        # Verify the calculation
        expected_total = (0.4 * 0.15) + (0.3 * 0.08) + (0.2 * 0.45) + (0.1 * 0.12)
        assert abs(result.total_delta - expected_total) < 1e-6
        assert result.component_deltas.performance == 0.15
        assert result.component_deltas.efficiency == 0.08
        assert result.component_deltas.stability == 0.45
        assert result.component_deltas.capability == 0.12
    
    def test_evaluate_delta_with_missing_data(self):
        """Test delta evaluation with missing data components"""
        evaluator = DeltaEvaluator()
        
        # Only provide performance data
        performance_data = {'current_reward': 100, 'previous_reward': 80, 'tokens_used': 1000}
        
        result = evaluator.evaluate_delta(performance_data=performance_data)
        
        # Should handle missing data gracefully
        assert result is not None
        assert result.component_deltas.performance is not None
        # Other components should have default values or be None
    
    def test_evaluate_delta_trend_analysis(self):
        """Test delta evaluation with historical trend analysis"""
        evaluator = DeltaEvaluator()
        
        # Add historical data
        historical_data = [
            {'timestamp': 1000, 'total_delta': 0.05, 'performance': 0.1},
            {'timestamp': 2000, 'total_delta': 0.08, 'performance': 0.12},
            {'timestamp': 3000, 'total_delta': 0.06, 'performance': 0.09}
        ]
        
        for data in historical_data:
            evaluator.add_historical_data(data)
        
        # Current evaluation should consider trends
        performance_data = {'current_reward': 100, 'previous_reward': 80, 'tokens_used': 1000}
        result = evaluator.evaluate_delta(
            performance_data=performance_data,
            use_trend_analysis=True
        )
        
        assert result.confidence is not None
        assert 0 <= result.confidence <= 1
    
    def test_evaluate_delta_confidence_calculation(self):
        """Test that confidence scores are calculated properly"""
        evaluator = DeltaEvaluator()
        
        # High-quality data should result in high confidence
        performance_data = {'current_reward': 100, 'previous_reward': 80, 'tokens_used': 1000}
        efficiency_data = {'current_throughput': 150, 'previous_throughput': 120, 'resource_used': 100}
        stability_data = {'divergence_score': 0.05}
        capability_data = {'new_capabilities': 3, 'total_capabilities': 25}
        
        result = evaluator.evaluate_delta(
            performance_data=performance_data,
            efficiency_data=efficiency_data,
            stability_data=stability_data,
            capability_data=capability_data
        )
        
        assert result.confidence > 0.7  # Should be high confidence with complete data
        
        # Incomplete data should result in lower confidence
        result_incomplete = evaluator.evaluate_delta(performance_data=performance_data)
        assert result_incomplete.confidence < result.confidence


class TestDeltaEvaluatorIntegration:
    """Integration tests for the complete Delta Evaluation system"""
    
    def test_end_to_end_improvement_detection(self):
        """Test complete workflow from data input to improvement detection"""
        evaluator = DeltaEvaluator()
        
        # Simulate system improvement scenario
        performance_data = {
            'current_reward': 120.0,
            'previous_reward': 100.0,
            'tokens_used': 1000
        }
        
        efficiency_data = {
            'current_throughput': 180.0,
            'previous_throughput': 150.0,
            'resource_used': 100.0
        }
        
        stability_data = {
            'divergence_score': 0.1  # Low divergence = high stability
        }
        
        capability_data = {
            'new_capabilities': 2,
            'total_capabilities': 20
        }
        
        result = evaluator.evaluate_delta(
            performance_data=performance_data,
            efficiency_data=efficiency_data,
            stability_data=stability_data,
            capability_data=capability_data
        )
        
        # All components show improvement, so total should be positive
        assert result.total_delta > 0
        assert result.is_improvement() is True
        assert result.confidence > 0.5
    
    def test_end_to_end_degradation_detection(self):
        """Test complete workflow for detecting system degradation"""
        evaluator = DeltaEvaluator()
        
        # Simulate system degradation scenario
        performance_data = {
            'current_reward': 80.0,
            'previous_reward': 100.0,
            'tokens_used': 1200  # More tokens for less reward
        }
        
        efficiency_data = {
            'current_throughput': 120.0,
            'previous_throughput': 150.0,
            'resource_used': 110.0  # More resources for less throughput
        }
        
        stability_data = {
            'divergence_score': 0.7  # High divergence = low stability
        }
        
        capability_data = {
            'new_capabilities': 0,
            'total_capabilities': 20
        }
        
        result = evaluator.evaluate_delta(
            performance_data=performance_data,
            efficiency_data=efficiency_data,
            stability_data=stability_data,
            capability_data=capability_data
        )
        
        # All components show degradation, so total should be negative
        assert result.total_delta < 0
        assert result.is_improvement() is False
    
    def test_adaptive_weight_adjustment_during_evaluation(self):
        """Test that weights adapt based on system context during evaluation"""
        evaluator = DeltaEvaluator()
        
        performance_data = {'current_reward': 110, 'previous_reward': 100, 'tokens_used': 1000}
        stability_data = {'divergence_score': 0.3}  # Moderate instability
        
        # In stability-critical context, stability should be weighted higher
        result_stability_critical = evaluator.evaluate_delta(
            performance_data=performance_data,
            stability_data=stability_data,
            context="stability_critical"
        )
        
        # In performance-critical context, performance should be weighted higher
        result_performance_critical = evaluator.evaluate_delta(
            performance_data=performance_data,
            stability_data=stability_data,
            context="performance_critical"
        )
        
        # Results should differ based on context
        assert result_stability_critical.total_delta != result_performance_critical.total_delta
        assert result_stability_critical.weights.stability > result_performance_critical.weights.stability
        assert result_performance_critical.weights.performance > result_stability_critical.weights.performance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])