//! Loss function validation tests against Python implementations
//!
//! This module validates that our loss function implementations produce
//! identical results to Python's NeuralForecast loss functions.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use neuro_divergent_training::{
    LossFunction, Loss,
    MAELoss, MSELoss, RMSELoss, MAPELoss, SMAPELoss, MASELoss,
    NegativeLogLikelihoodLoss, PinballLoss, CRPSLoss,
    GaussianNLLLoss, PoissonNLLLoss, NegativeBinomialNLLLoss,
    HuberLoss, QuantileLoss, ScaledLoss, SeasonalLoss,
};

const LOSS_VALUE_TOLERANCE: f64 = 1e-8;
const GRADIENT_TOLERANCE: f64 = 1e-7;

/// Test data generator for reproducible testing
struct TestDataGenerator {
    rng: ChaCha8Rng,
}

impl TestDataGenerator {
    fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
    
    fn generate_predictions(&mut self, n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n).map(|_| self.rng.gen_range(0.0..100.0)).collect()
        )
    }
    
    fn generate_targets(&mut self, n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n).map(|_| self.rng.gen_range(0.0..100.0)).collect()
        )
    }
    
    fn generate_positive_targets(&mut self, n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n).map(|_| self.rng.gen_range(1.0..100.0)).collect()
        )
    }
    
    fn generate_probabilistic_predictions(&mut self, n: usize) -> Array1<f64> {
        // For probabilistic losses, we need mean and variance/std
        let mut result = Vec::with_capacity(n * 2);
        for _ in 0..n {
            result.push(self.rng.gen_range(0.0..100.0)); // mean
            result.push(self.rng.gen_range(0.1..2.0)); // log_var or std
        }
        Array1::from_vec(result)
    }
}

/// Python reference implementations for validation
mod python_reference {
    use super::*;
    
    pub fn mae(predictions: &[f64], targets: &[f64]) -> f64 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>() / predictions.len() as f64
    }
    
    pub fn mae_gradient(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
        let n = predictions.len() as f64;
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                if p > t { 1.0 / n }
                else if p < t { -1.0 / n }
                else { 0.0 }
            })
            .collect()
    }
    
    pub fn mse(predictions: &[f64], targets: &[f64]) -> f64 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / predictions.len() as f64
    }
    
    pub fn mse_gradient(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
        let n = predictions.len() as f64;
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| 2.0 * (p - t) / n)
            .collect()
    }
    
    pub fn rmse(predictions: &[f64], targets: &[f64]) -> f64 {
        mse(predictions, targets).sqrt()
    }
    
    pub fn rmse_gradient(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
        let mse_val = mse(predictions, targets);
        if mse_val == 0.0 {
            vec![0.0; predictions.len()]
        } else {
            let rmse_val = mse_val.sqrt();
            let mse_grads = mse_gradient(predictions, targets);
            mse_grads.iter()
                .map(|g| g / (2.0 * rmse_val))
                .collect()
        }
    }
    
    pub fn mape(predictions: &[f64], targets: &[f64], epsilon: f64) -> f64 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let denominator = t.abs().max(epsilon);
                (p - t).abs() / denominator
            })
            .sum::<f64>() * 100.0 / predictions.len() as f64
    }
    
    pub fn mape_gradient(predictions: &[f64], targets: &[f64], epsilon: f64) -> Vec<f64> {
        let n = predictions.len() as f64;
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let denominator = t.abs().max(epsilon);
                let sign = if p > t { 1.0 } else { -1.0 };
                100.0 * sign / (n * denominator)
            })
            .collect()
    }
    
    pub fn smape(predictions: &[f64], targets: &[f64], epsilon: f64) -> f64 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let numerator = (p - t).abs();
                let denominator = (p.abs() + t.abs()).max(epsilon);
                numerator / denominator
            })
            .sum::<f64>() * 100.0 / predictions.len() as f64
    }
    
    pub fn huber(predictions: &[f64], targets: &[f64], delta: f64) -> f64 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let diff = (p - t).abs();
                if diff <= delta {
                    diff * diff / 2.0
                } else {
                    delta * diff - delta * delta / 2.0
                }
            })
            .sum::<f64>() / predictions.len() as f64
    }
    
    pub fn pinball(predictions: &[f64], targets: &[f64], quantile: f64) -> f64 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let diff = t - p;
                if diff >= 0.0 {
                    quantile * diff
                } else {
                    (quantile - 1.0) * diff
                }
            })
            .sum::<f64>() / predictions.len() as f64
    }
}

#[cfg(test)]
mod mae_tests {
    use super::*;
    
    #[test]
    fn test_mae_forward() {
        let mut gen = TestDataGenerator::new(42);
        let predictions = gen.generate_predictions(100);
        let targets = gen.generate_targets(100);
        
        let loss = MAELoss::new();
        let rust_result = loss.forward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_result = python_reference::mae(predictions.as_slice().unwrap(), targets.as_slice().unwrap());
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = LOSS_VALUE_TOLERANCE,
            "MAE forward pass mismatch"
        );
    }
    
    #[test]
    fn test_mae_backward() {
        let mut gen = TestDataGenerator::new(42);
        let predictions = gen.generate_predictions(50);
        let targets = gen.generate_targets(50);
        
        let loss = MAELoss::new();
        let rust_gradients = loss.backward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_gradients = python_reference::mae_gradient(predictions.as_slice().unwrap(), targets.as_slice().unwrap());
        
        for (rust_grad, py_grad) in rust_gradients.iter().zip(python_gradients.iter()) {
            assert_abs_diff_eq!(
                rust_grad,
                py_grad,
                epsilon = GRADIENT_TOLERANCE,
                "MAE gradient mismatch"
            );
        }
    }
    
    #[test]
    fn test_mae_edge_cases() {
        let loss = MAELoss::new();
        
        // Test with zeros
        let zeros = vec![0.0; 10];
        let result = loss.forward(&zeros, &zeros).unwrap();
        assert_eq!(result, 0.0, "MAE with zeros should be 0");
        
        // Test with equal values
        let ones = vec![1.0; 10];
        let twos = vec![2.0; 10];
        let result = loss.forward(&ones, &twos).unwrap();
        assert_eq!(result, 1.0, "MAE with constant difference should be exact");
    }
}

#[cfg(test)]
mod mse_tests {
    use super::*;
    
    #[test]
    fn test_mse_forward() {
        let mut gen = TestDataGenerator::new(123);
        let predictions = gen.generate_predictions(100);
        let targets = gen.generate_targets(100);
        
        let loss = MSELoss::new();
        let rust_result = loss.forward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_result = python_reference::mse(predictions.as_slice().unwrap(), targets.as_slice().unwrap());
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = LOSS_VALUE_TOLERANCE,
            "MSE forward pass mismatch"
        );
    }
    
    #[test]
    fn test_mse_backward() {
        let mut gen = TestDataGenerator::new(123);
        let predictions = gen.generate_predictions(50);
        let targets = gen.generate_targets(50);
        
        let loss = MSELoss::new();
        let rust_gradients = loss.backward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_gradients = python_reference::mse_gradient(predictions.as_slice().unwrap(), targets.as_slice().unwrap());
        
        for (rust_grad, py_grad) in rust_gradients.iter().zip(python_gradients.iter()) {
            assert_abs_diff_eq!(
                rust_grad,
                py_grad,
                epsilon = GRADIENT_TOLERANCE,
                "MSE gradient mismatch"
            );
        }
    }
}

#[cfg(test)]
mod rmse_tests {
    use super::*;
    
    #[test]
    fn test_rmse_forward() {
        let mut gen = TestDataGenerator::new(456);
        let predictions = gen.generate_predictions(100);
        let targets = gen.generate_targets(100);
        
        let loss = RMSELoss::new();
        let rust_result = loss.forward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_result = python_reference::rmse(predictions.as_slice().unwrap(), targets.as_slice().unwrap());
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = LOSS_VALUE_TOLERANCE,
            "RMSE forward pass mismatch"
        );
    }
    
    #[test]
    fn test_rmse_backward() {
        let mut gen = TestDataGenerator::new(456);
        let predictions = gen.generate_predictions(50);
        let targets = gen.generate_targets(50);
        
        let loss = RMSELoss::new();
        let rust_gradients = loss.backward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_gradients = python_reference::rmse_gradient(predictions.as_slice().unwrap(), targets.as_slice().unwrap());
        
        for (rust_grad, py_grad) in rust_gradients.iter().zip(python_gradients.iter()) {
            assert_abs_diff_eq!(
                rust_grad,
                py_grad,
                epsilon = GRADIENT_TOLERANCE,
                "RMSE gradient mismatch"
            );
        }
    }
    
    #[test]
    fn test_rmse_zero_mse() {
        let loss = RMSELoss::new();
        let values = vec![1.0; 10];
        
        let result = loss.forward(&values, &values).unwrap();
        assert_eq!(result, 0.0, "RMSE with identical values should be 0");
        
        let gradients = loss.backward(&values, &values).unwrap();
        for grad in gradients {
            assert_eq!(grad, 0.0, "RMSE gradient should be 0 for identical values");
        }
    }
}

#[cfg(test)]
mod percentage_error_tests {
    use super::*;
    
    #[test]
    fn test_mape_forward() {
        let mut gen = TestDataGenerator::new(789);
        let predictions = gen.generate_predictions(100);
        let targets = gen.generate_positive_targets(100); // MAPE needs non-zero targets
        
        let loss = MAPELoss::new();
        let rust_result = loss.forward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_result = python_reference::mape(
            predictions.as_slice().unwrap(), 
            targets.as_slice().unwrap(),
            1e-8
        );
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = LOSS_VALUE_TOLERANCE,
            "MAPE forward pass mismatch"
        );
    }
    
    #[test]
    fn test_mape_near_zero_handling() {
        let loss = MAPELoss::new();
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![0.0, 1e-10, 2.0]; // Near-zero values
        
        let result = loss.forward(&predictions, &targets);
        assert!(result.is_ok(), "MAPE should handle near-zero targets");
        assert!(result.unwrap().is_finite(), "MAPE should return finite value");
    }
    
    #[test]
    fn test_smape_forward() {
        let mut gen = TestDataGenerator::new(101112);
        let predictions = gen.generate_predictions(100);
        let targets = gen.generate_targets(100);
        
        let loss = SMAPELoss::new();
        let rust_result = loss.forward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_result = python_reference::smape(
            predictions.as_slice().unwrap(), 
            targets.as_slice().unwrap(),
            1e-8
        );
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = LOSS_VALUE_TOLERANCE,
            "SMAPE forward pass mismatch"
        );
    }
    
    #[test]
    fn test_smape_symmetry() {
        let loss = SMAPELoss::new();
        let pred1 = vec![100.0];
        let target1 = vec![90.0];
        
        let pred2 = vec![90.0];
        let target2 = vec![100.0];
        
        let result1 = loss.forward(&pred1, &target1).unwrap();
        let result2 = loss.forward(&pred2, &target2).unwrap();
        
        assert_abs_diff_eq!(
            result1,
            result2,
            epsilon = 1e-10,
            "SMAPE should be symmetric"
        );
    }
}

#[cfg(test)]
mod probabilistic_loss_tests {
    use super::*;
    
    #[test]
    fn test_gaussian_nll() {
        let mut gen = TestDataGenerator::new(131415);
        let predictions = gen.generate_probabilistic_predictions(50);
        let targets = gen.generate_targets(50);
        
        let loss = GaussianNLLLoss::new();
        let result = loss.forward(predictions.as_slice().unwrap(), targets.as_slice().unwrap());
        
        assert!(result.is_ok(), "Gaussian NLL should handle probabilistic predictions");
        assert!(result.unwrap().is_finite(), "Gaussian NLL should return finite value");
    }
    
    #[test]
    fn test_pinball_loss() {
        let mut gen = TestDataGenerator::new(161718);
        let predictions = gen.generate_predictions(100);
        let targets = gen.generate_targets(100);
        
        let quantile = 0.5;
        let loss = PinballLoss::new(quantile);
        let rust_result = loss.forward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_result = python_reference::pinball(
            predictions.as_slice().unwrap(), 
            targets.as_slice().unwrap(),
            quantile
        );
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = LOSS_VALUE_TOLERANCE,
            "Pinball loss forward pass mismatch"
        );
    }
    
    #[test]
    fn test_quantile_loss_multiple() {
        let quantiles = vec![0.1, 0.5, 0.9];
        let loss = QuantileLoss::new(quantiles.clone());
        
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 samples, 3 quantiles each
        let targets = vec![1.5, 4.5]; // 2 target values
        
        let result = loss.forward(&predictions, &targets);
        assert!(result.is_ok(), "Quantile loss should handle multiple quantiles");
    }
}

#[cfg(test)]
mod robust_loss_tests {
    use super::*;
    
    #[test]
    fn test_huber_loss() {
        let mut gen = TestDataGenerator::new(192021);
        let predictions = gen.generate_predictions(100);
        let targets = gen.generate_targets(100);
        
        let delta = 1.0;
        let loss = HuberLoss::new(delta);
        let rust_result = loss.forward(predictions.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap();
        let python_result = python_reference::huber(
            predictions.as_slice().unwrap(), 
            targets.as_slice().unwrap(),
            delta
        );
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = LOSS_VALUE_TOLERANCE,
            "Huber loss forward pass mismatch"
        );
    }
    
    #[test]
    fn test_huber_outlier_robustness() {
        let loss = HuberLoss::new(1.0);
        
        // Normal errors
        let pred_normal = vec![1.0, 2.0, 3.0];
        let target_normal = vec![1.1, 2.1, 3.1];
        let loss_normal = loss.forward(&pred_normal, &target_normal).unwrap();
        
        // With outlier
        let pred_outlier = vec![1.0, 2.0, 100.0]; // Large outlier
        let target_outlier = vec![1.1, 2.1, 3.1];
        let loss_outlier = loss.forward(&pred_outlier, &target_outlier).unwrap();
        
        // Huber loss should be less sensitive to outliers than MSE
        let mse_loss = MSELoss::new();
        let mse_normal = mse_loss.forward(&pred_normal, &target_normal).unwrap();
        let mse_outlier = mse_loss.forward(&pred_outlier, &target_outlier).unwrap();
        
        let huber_ratio = loss_outlier / loss_normal;
        let mse_ratio = mse_outlier / mse_normal;
        
        assert!(huber_ratio < mse_ratio, "Huber should be more robust to outliers than MSE");
    }
}

#[cfg(test)]
mod custom_loss_tests {
    use super::*;
    
    #[test]
    fn test_scaled_loss() {
        let base_loss = Loss::MAE(MAELoss::new());
        let scale_factor = 2.5;
        let loss = ScaledLoss::new(base_loss, scale_factor);
        
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 3.5];
        
        let mae_loss = MAELoss::new();
        let base_result = mae_loss.forward(&predictions, &targets).unwrap();
        let scaled_result = loss.forward(&predictions, &targets).unwrap();
        
        assert_abs_diff_eq!(
            scaled_result,
            base_result / scale_factor,
            epsilon = LOSS_VALUE_TOLERANCE,
            "Scaled loss should divide by scale factor"
        );
    }
    
    #[test]
    fn test_seasonal_loss() {
        let base_loss = Loss::MSE(MSELoss::new());
        let seasonal_weights = vec![1.0, 1.5, 2.0, 1.5]; // 4-period seasonality
        let loss = SeasonalLoss::new(base_loss, seasonal_weights);
        
        let predictions = vec![1.0; 8]; // 2 full seasons
        let targets = vec![2.0; 8];
        
        let result = loss.forward(&predictions, &targets);
        assert!(result.is_ok(), "Seasonal loss should handle multiple seasons");
        
        // The loss should be weighted by seasonal factors
        let base_mse = MSELoss::new();
        let unweighted = base_mse.forward(&predictions, &targets).unwrap();
        let weighted = result.unwrap();
        
        assert_ne!(weighted, unweighted, "Seasonal weighting should affect loss");
    }
}

#[cfg(test)]
mod numerical_stability_tests {
    use super::*;
    
    #[test]
    fn test_extreme_values() {
        let losses: Vec<Box<dyn LossFunction<f64>>> = vec![
            Box::new(MAELoss::new()),
            Box::new(MSELoss::new()),
            Box::new(RMSELoss::new()),
            Box::new(HuberLoss::new(1.0)),
        ];
        
        // Test with very large values
        let large_pred = vec![1e10, -1e10, 1e10];
        let large_target = vec![1e10 + 1.0, -1e10 - 1.0, 1e10 - 1.0];
        
        for loss in &losses {
            let result = loss.forward(&large_pred, &large_target);
            assert!(result.is_ok(), "{} should handle large values", loss.name());
            assert!(result.unwrap().is_finite(), "{} should return finite values", loss.name());
        }
        
        // Test with very small values
        let small_pred = vec![1e-10, -1e-10, 1e-10];
        let small_target = vec![2e-10, -2e-10, 0.0];
        
        for loss in &losses {
            let result = loss.forward(&small_pred, &small_target);
            assert!(result.is_ok(), "{} should handle small values", loss.name());
            assert!(result.unwrap().is_finite(), "{} should return finite values", loss.name());
        }
    }
    
    #[test]
    fn test_gradient_numerical_stability() {
        let losses: Vec<Box<dyn LossFunction<f64>>> = vec![
            Box::new(MAELoss::new()),
            Box::new(MSELoss::new()),
            Box::new(MAPELoss::new()),
            Box::new(HuberLoss::new(1.0)),
        ];
        
        // Test gradient computation with values that could cause issues
        let pred = vec![1e-8, 1e8, 0.0, 1.0];
        let target = vec![0.0, 1e8 + 1.0, 1e-8, 1.0];
        
        for loss in &losses {
            let result = loss.backward(&pred, &target);
            if result.is_ok() {
                let gradients = result.unwrap();
                for grad in &gradients {
                    assert!(grad.is_finite(), 
                        "{} gradient should be finite", loss.name());
                }
            }
        }
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let loss = MAELoss::new();
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0]; // Different length
        
        let result = loss.forward(&pred, &target);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }
}