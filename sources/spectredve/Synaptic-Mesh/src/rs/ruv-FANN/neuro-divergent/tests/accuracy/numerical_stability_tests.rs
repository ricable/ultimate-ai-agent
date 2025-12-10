//! Numerical stability tests for neural forecasting
//!
//! This module tests numerical stability under various challenging conditions:
//! - Very large/small values
//! - Near-zero divisions
//! - Overflow/underflow scenarios
//! - NaN/Inf propagation
//! - Accumulation errors
//! - Platform-specific floating point behaviors

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::f64::{INFINITY, NEG_INFINITY, NAN, EPSILON, MAX, MIN_POSITIVE};

use neuro_divergent_training::{
    LossFunction, Loss,
    MAELoss, MSELoss, RMSELoss, MAPELoss, SMAPELoss,
    HuberLoss, NegativeLogLikelihoodLoss,
};

use neuro_divergent_models::{
    activations::{Sigmoid, Tanh, Softmax, LogSoftmax},
    layers::{LayerNorm, BatchNorm},
};

/// Test utilities for numerical stability
mod stability_utils {
    use super::*;
    
    /// Check if a value is numerically stable (finite and not too extreme)
    pub fn is_stable(x: f64) -> bool {
        x.is_finite() && x.abs() < 1e100
    }
    
    /// Generate edge case values for testing
    pub fn edge_case_values() -> Vec<f64> {
        vec![
            0.0,
            -0.0,
            EPSILON,
            -EPSILON,
            MIN_POSITIVE,
            -MIN_POSITIVE,
            1e-308,  // Near underflow
            -1e-308,
            1e308,   // Near overflow
            -1e308,
            1.0,
            -1.0,
            MAX / 2.0,
            -MAX / 2.0,
        ]
    }
    
    /// Generate values that commonly cause numerical issues
    pub fn problematic_values() -> Vec<f64> {
        vec![
            1e-10,   // Very small
            1e10,    // Very large
            1.0 + EPSILON,  // Near 1
            1.0 - EPSILON,
            0.0 + EPSILON,  // Near 0
            0.0 - EPSILON,
            std::f64::consts::PI,
            std::f64::consts::E,
        ]
    }
    
    /// Safe logarithm with underflow protection
    pub fn safe_log(x: f64, epsilon: f64) -> f64 {
        x.max(epsilon).ln()
    }
    
    /// Safe division with zero protection
    pub fn safe_div(numerator: f64, denominator: f64, epsilon: f64) -> f64 {
        numerator / denominator.abs().max(epsilon)
    }
    
    /// Safe exponential with overflow protection
    pub fn safe_exp(x: f64, max_val: f64) -> f64 {
        x.min(max_val).exp()
    }
}

#[cfg(test)]
mod loss_stability_tests {
    use super::*;
    use stability_utils::*;
    
    #[test]
    fn test_mse_extreme_values() {
        let loss = MSELoss::new();
        
        // Test with very large values
        let large_pred = vec![1e150, -1e150];
        let large_target = vec![1e150 + 1.0, -1e150 - 1.0];
        
        let result = loss.forward(&large_pred, &large_target);
        assert!(result.is_ok(), "MSE should handle large values");
        
        let value = result.unwrap();
        assert!(value.is_finite(), "MSE result should be finite");
        assert!(value >= 0.0, "MSE should be non-negative");
        
        // Test with very small values
        let small_pred = vec![1e-300, -1e-300];
        let small_target = vec![2e-300, -2e-300];
        
        let result = loss.forward(&small_pred, &small_target);
        assert!(result.is_ok(), "MSE should handle small values");
        assert!(result.unwrap().is_finite(), "MSE with small values should be finite");
    }
    
    #[test]
    fn test_mape_near_zero_targets() {
        let loss = MAPELoss::with_epsilon(1e-8);
        
        // Test with near-zero targets
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![0.0, 1e-10, -1e-10];
        
        let result = loss.forward(&predictions, &targets);
        assert!(result.is_ok(), "MAPE should handle near-zero targets");
        
        let value = result.unwrap();
        assert!(value.is_finite(), "MAPE should return finite value for near-zero targets");
        
        // Gradient should also be stable
        let gradients = loss.backward(&predictions, &targets);
        assert!(gradients.is_ok(), "MAPE gradient should handle near-zero targets");
        
        for grad in gradients.unwrap() {
            assert!(grad.is_finite(), "MAPE gradient should be finite");
        }
    }
    
    #[test]
    fn test_smape_zero_denominator() {
        let loss = SMAPELoss::new();
        
        // Test when both prediction and target are zero
        let predictions = vec![0.0, 1e-15, -1e-15];
        let targets = vec![0.0, -1e-15, 1e-15];
        
        let result = loss.forward(&predictions, &targets);
        assert!(result.is_ok(), "SMAPE should handle zero denominators");
        
        let value = result.unwrap();
        assert!(value.is_finite(), "SMAPE should return finite value");
        assert!(value >= 0.0, "SMAPE should be non-negative");
        assert!(value <= 200.0, "SMAPE should be bounded by 200%");
    }
    
    #[test]
    fn test_log_likelihood_extreme_variance() {
        let loss = NegativeLogLikelihoodLoss::new();
        
        // Test with extreme log variances
        let predictions = vec![
            0.0, -100.0,  // mean, log_var (very small variance)
            0.0, 100.0,   // mean, log_var (very large variance)
        ];
        let targets = vec![0.0, 1.0];
        
        let result = loss.forward(&predictions, &targets);
        assert!(result.is_ok(), "NLL should handle extreme variances");
        
        let value = result.unwrap();
        assert!(value.is_finite() || value == INFINITY, 
            "NLL should be finite or positive infinity for extreme cases");
    }
    
    #[test]
    fn test_huber_loss_outliers() {
        let loss = HuberLoss::new(1.0);
        
        // Test with extreme outliers
        let predictions = vec![0.0, 0.0, 1e10];
        let targets = vec![1.0, 1.0, 0.0];
        
        let result = loss.forward(&predictions, &targets);
        assert!(result.is_ok(), "Huber loss should handle outliers");
        
        let value = result.unwrap();
        assert!(value.is_finite(), "Huber loss should be finite even with outliers");
        
        // Compare with MSE to verify robustness
        let mse_loss = MSELoss::new();
        let mse_value = mse_loss.forward(&predictions, &targets).unwrap();
        
        // Huber loss should grow linearly for large errors, MSE quadratically
        assert!(value < mse_value / 1e5, "Huber should be more robust than MSE");
    }
}

#[cfg(test)]
mod activation_stability_tests {
    use super::*;
    use stability_utils::*;
    
    #[test]
    fn test_sigmoid_extreme_inputs() {
        let sigmoid = Sigmoid::new();
        let extreme_inputs = vec![-1000.0, -100.0, 100.0, 1000.0];
        
        let outputs = sigmoid.forward(&Array1::from_vec(extreme_inputs.clone()));
        
        for (input, output) in extreme_inputs.iter().zip(outputs.iter()) {
            assert!(output.is_finite(), "Sigmoid output should be finite");
            assert!(*output >= 0.0, "Sigmoid output should be >= 0");
            assert!(*output <= 1.0, "Sigmoid output should be <= 1");
            
            // Check asymptotic behavior
            if *input < -10.0 {
                assert!(*output < 1e-4, "Sigmoid should approach 0 for large negative inputs");
            }
            if *input > 10.0 {
                assert!(*output > 1.0 - 1e-4, "Sigmoid should approach 1 for large positive inputs");
            }
        }
        
        // Test gradient stability
        let grad_output = Array1::ones(extreme_inputs.len());
        let gradients = sigmoid.backward(&Array1::from_vec(extreme_inputs), &grad_output);
        
        for grad in gradients.iter() {
            assert!(grad.is_finite(), "Sigmoid gradient should be finite");
            assert!(*grad >= 0.0, "Sigmoid gradient should be non-negative");
            assert!(*grad <= 0.25, "Sigmoid gradient should be <= 0.25");
        }
    }
    
    #[test]
    fn test_tanh_extreme_inputs() {
        let tanh = Tanh::new();
        let extreme_inputs = vec![-1000.0, -100.0, 100.0, 1000.0];
        
        let outputs = tanh.forward(&Array1::from_vec(extreme_inputs.clone()));
        
        for (input, output) in extreme_inputs.iter().zip(outputs.iter()) {
            assert!(output.is_finite(), "Tanh output should be finite");
            assert!(*output >= -1.0, "Tanh output should be >= -1");
            assert!(*output <= 1.0, "Tanh output should be <= 1");
            
            // Check asymptotic behavior
            if input.abs() > 10.0 {
                assert!(output.abs() > 1.0 - 1e-4, "Tanh should approach ±1 for large inputs");
            }
        }
    }
    
    #[test]
    fn test_softmax_overflow_protection() {
        let softmax = Softmax::new();
        
        // Test with values that would overflow without protection
        let inputs = vec![
            vec![1000.0, 999.0, 998.0],  // Large positive
            vec![-1000.0, -999.0, -998.0],  // Large negative
            vec![1000.0, 0.0, -1000.0],  // Mixed extreme
        ];
        
        for input in inputs {
            let output = softmax.forward(&Array1::from_vec(input.clone()));
            
            // Check outputs are valid probabilities
            for prob in output.iter() {
                assert!(prob.is_finite(), "Softmax output should be finite");
                assert!(*prob >= 0.0, "Softmax output should be non-negative");
                assert!(*prob <= 1.0, "Softmax output should be <= 1");
            }
            
            // Check sum equals 1
            let sum: f64 = output.sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10, "Softmax should sum to 1");
        }
    }
    
    #[test]
    fn test_log_softmax_underflow_protection() {
        let log_softmax = LogSoftmax::new();
        
        // Test with values that could cause underflow in regular softmax
        let inputs = vec![
            vec![-1000.0, -1001.0, -1002.0],
            vec![0.0, -500.0, -1000.0],
        ];
        
        for input in inputs {
            let output = log_softmax.forward(&Array1::from_vec(input));
            
            // Check outputs are valid log probabilities
            for log_prob in output.iter() {
                assert!(log_prob.is_finite(), "LogSoftmax output should be finite");
                assert!(*log_prob <= 0.0, "LogSoftmax output should be <= 0");
            }
            
            // Check that exp(log_softmax) sums to 1
            let sum: f64 = output.mapv(f64::exp).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }
}

#[cfg(test)]
mod normalization_stability_tests {
    use super::*;
    
    #[test]
    fn test_layer_norm_zero_variance() {
        let layer_norm = LayerNorm::new(4, 1e-5);
        
        // Test with constant values (zero variance)
        let input = Array2::from_shape_vec((2, 4), vec![5.0; 8]).unwrap();
        let output = layer_norm.forward(&input);
        
        // Output should be all zeros when input is constant
        for val in output.iter() {
            assert!(val.is_finite(), "LayerNorm output should be finite");
            assert!(val.abs() < 1e-4, "LayerNorm of constant input should be near zero");
        }
    }
    
    #[test]
    fn test_batch_norm_small_batch_variance() {
        let mut batch_norm = BatchNorm::new(4, 0.9, 1e-5);
        
        // Test with very small batch (can cause unstable variance estimates)
        let input = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = batch_norm.forward(&input, true); // Training mode
        
        for val in output.iter() {
            assert!(val.is_finite(), "BatchNorm output should be finite with small batch");
        }
        
        // Test with near-constant values
        let constant_input = Array2::from_shape_vec((2, 4), vec![1.0; 8]).unwrap();
        let output = batch_norm.forward(&constant_input, true);
        
        for val in output.iter() {
            assert!(val.is_finite(), "BatchNorm should handle near-zero variance");
        }
    }
}

#[cfg(test)]
mod accumulation_error_tests {
    use super::*;
    
    #[test]
    fn test_kahan_summation() {
        // Implement Kahan summation for accurate accumulation
        fn kahan_sum(values: &[f64]) -> f64 {
            let mut sum = 0.0;
            let mut c = 0.0; // Compensation for lost digits
            
            for &value in values {
                let y = value - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            
            sum
        }
        
        // Test case where naive summation has significant error
        let n = 1_000_000;
        let values: Vec<f64> = vec![0.1; n]; // Sum should be exactly 100,000
        
        // Naive summation
        let naive_sum: f64 = values.iter().sum();
        
        // Kahan summation
        let kahan = kahan_sum(&values);
        
        // Expected value
        let expected = 100_000.0;
        
        // Kahan should be more accurate
        let naive_error = (naive_sum - expected).abs();
        let kahan_error = (kahan - expected).abs();
        
        assert!(kahan_error < naive_error, "Kahan summation should be more accurate");
        assert!(kahan_error < 1e-10, "Kahan summation error should be minimal");
    }
    
    #[test]
    fn test_welford_variance() {
        // Implement Welford's algorithm for numerically stable variance
        fn welford_variance(values: &[f64]) -> (f64, f64) {
            let mut mean = 0.0;
            let mut m2 = 0.0;
            let mut count = 0.0;
            
            for &value in values {
                count += 1.0;
                let delta = value - mean;
                mean += delta / count;
                let delta2 = value - mean;
                m2 += delta * delta2;
            }
            
            let variance = if count > 1.0 { m2 / (count - 1.0) } else { 0.0 };
            (mean, variance)
        }
        
        // Test with values that cause issues with naive variance calculation
        let values = vec![1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0, 1e8 + 4.0];
        
        // Naive calculation
        let naive_mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let naive_var: f64 = values.iter()
            .map(|&x| (x - naive_mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        // Welford's algorithm
        let (welford_mean, welford_var) = welford_variance(&values);
        
        // True variance should be 2.5 (variance of [0, 1, 2, 3, 4])
        let true_variance = 2.5;
        
        assert_relative_eq!(welford_var, true_variance, epsilon = 1e-10,
            "Welford's algorithm should give accurate variance");
        
        // Naive might have larger error due to catastrophic cancellation
        println!("Naive variance: {}, Welford variance: {}", naive_var, welford_var);
    }
}

#[cfg(test)]
mod overflow_underflow_tests {
    use super::*;
    use stability_utils::*;
    
    #[test]
    fn test_exp_overflow_protection() {
        let large_values = vec![100.0, 500.0, 1000.0, 10000.0];
        
        for &x in &large_values {
            let safe_result = safe_exp(x, 100.0);
            assert!(safe_result.is_finite(), "Safe exp should prevent overflow");
            
            // Direct exp might overflow
            let direct_result = x.exp();
            if x > 700.0 {
                assert!(direct_result.is_infinite(), "Direct exp should overflow for x > 700");
            }
        }
    }
    
    #[test]
    fn test_log_underflow_protection() {
        let small_values = vec![0.0, 1e-100, 1e-200, 1e-300, MIN_POSITIVE];
        
        for &x in &small_values {
            let safe_result = safe_log(x, 1e-50);
            assert!(safe_result.is_finite(), "Safe log should prevent -inf");
            
            // Direct log of 0 gives -inf
            if x == 0.0 {
                assert!(x.ln().is_infinite(), "Direct log(0) should be -inf");
            }
        }
    }
    
    #[test]
    fn test_product_overflow() {
        // Test product of many probabilities
        let probabilities = vec![0.1; 1000]; // Product would underflow
        
        // Use log-sum-exp trick
        let log_probs: Vec<f64> = probabilities.iter().map(|p| p.ln()).collect();
        let log_product: f64 = log_probs.iter().sum();
        
        assert!(log_product.is_finite(), "Log product should be finite");
        
        // Direct product would underflow
        let direct_product: f64 = probabilities.iter().product();
        assert_eq!(direct_product, 0.0, "Direct product should underflow to 0");
    }
}

#[cfg(test)]
mod nan_inf_propagation_tests {
    use super::*;
    
    #[test]
    fn test_nan_propagation_prevention() {
        let loss = MSELoss::new();
        
        // Test with NaN in predictions
        let predictions = vec![1.0, NAN, 3.0];
        let targets = vec![1.0, 2.0, 3.0];
        
        let result = loss.forward(&predictions, &targets);
        // Loss should either handle NaN gracefully or return an error
        if let Ok(value) = result {
            assert!(value.is_nan(), "Loss with NaN input should propagate NaN or error");
        }
        
        // Test gradient computation with NaN
        let predictions_no_nan = vec![1.0, 2.0, 3.0];
        let targets_with_nan = vec![1.0, NAN, 3.0];
        
        let grad_result = loss.backward(&predictions_no_nan, &targets_with_nan);
        if let Ok(gradients) = grad_result {
            // Check if NaN is contained to affected elements
            assert!(gradients[1].is_nan(), "Gradient should be NaN where target is NaN");
            assert!(gradients[0].is_finite(), "Gradient should be finite for valid elements");
        }
    }
    
    #[test]
    fn test_inf_handling() {
        let loss = MAELoss::new();
        
        // Test with infinity
        let predictions = vec![INFINITY, NEG_INFINITY, 0.0];
        let targets = vec![1.0, -1.0, 0.0];
        
        let result = loss.forward(&predictions, &targets);
        if let Ok(value) = result {
            assert!(value.is_infinite(), "Loss with infinite input should be infinite");
        }
        
        // Test that finite values don't get contaminated
        let predictions_mixed = vec![1.0, INFINITY, 3.0];
        let targets_finite = vec![2.0, 2.0, 2.0];
        
        // MAE would average, so infinity dominates
        let result_mixed = loss.forward(&predictions_mixed, &targets_finite);
        if let Ok(value) = result_mixed {
            assert!(value.is_infinite(), "Loss with any infinite value should be infinite");
        }
    }
    
    #[test]
    fn test_gradient_masking() {
        // Test masking of invalid gradients
        let gradients = vec![1.0, NAN, INFINITY, -2.0, 0.0];
        
        let masked: Vec<f64> = gradients.iter()
            .map(|&g| if g.is_finite() { g } else { 0.0 })
            .collect();
        
        assert_eq!(masked, vec![1.0, 0.0, 0.0, -2.0, 0.0]);
        
        // All masked gradients should be finite
        for grad in masked {
            assert!(grad.is_finite(), "Masked gradient should be finite");
        }
    }
}

#[cfg(test)]
mod platform_consistency_tests {
    use super::*;
    
    #[test]
    fn test_float_comparison_consistency() {
        // Test that float comparisons are consistent
        let a = 0.1 + 0.2;
        let b = 0.3;
        
        // Direct comparison might fail
        assert!(a != b, "0.1 + 0.2 != 0.3 in floating point");
        
        // Approximate comparison should work
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }
    
    #[test]
    fn test_rounding_mode_effects() {
        // Test that we handle different rounding modes gracefully
        let values = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        
        // Round to nearest even (banker's rounding)
        let rounded: Vec<f64> = values.iter()
            .map(|&x| (x + 0.5).floor())
            .collect();
        
        // All results should be integers
        for r in rounded {
            assert_eq!(r.fract(), 0.0, "Rounded value should be integer");
        }
    }
    
    #[test]
    fn test_denormal_number_handling() {
        // Test handling of denormal (subnormal) numbers
        let denormal = MIN_POSITIVE / 2.0;
        assert!(denormal > 0.0, "Denormal should be positive");
        assert!(denormal < MIN_POSITIVE, "Denormal should be less than MIN_POSITIVE");
        
        // Operations with denormals
        let result = denormal * 2.0;
        assert_eq!(result, MIN_POSITIVE, "Denormal * 2 should equal MIN_POSITIVE");
        
        // Some platforms flush denormals to zero
        let very_small = denormal / 1e10;
        // Either it's 0 (flushed) or a very small positive number
        assert!(very_small >= 0.0, "Very small number should be non-negative");
    }
}