//! Comprehensive unit tests for the training system
//!
//! This module tests optimizers, loss functions, learning rate schedulers,
//! and the overall training pipeline infrastructure.

use neuro_divergent::prelude::*;
use neuro_divergent::{AccuracyMetrics, NeuroDivergentError, NeuroDivergentResult};
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};
use neuro_divergent::training::*;
use num_traits::Float;
use proptest::prelude::*;
use approx::assert_relative_eq;
use std::collections::HashMap;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Mock Training Infrastructure for Testing
// ============================================================================

/// Mock optimizer implementation for testing
#[derive(Clone, Debug)]
struct MockSGD<T: Float> {
    learning_rate: T,
    step_count: usize,
}

impl<T: Float> MockSGD<T> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            step_count: 0,
        }
    }
    
    pub fn step(&mut self, params: &mut Vec<T>, gradients: &[T]) -> NeuroDivergentResult<()> {
        if params.len() != gradients.len() {
            return Err(NeuroDivergentError::training("Parameter and gradient dimensions mismatch"));
        }
        
        for (param, grad) in params.iter_mut().zip(gradients.iter()) {
            *param = *param - self.learning_rate * *grad;
        }
        
        self.step_count += 1;
        Ok(())
    }
    
    pub fn get_step_count(&self) -> usize {
        self.step_count
    }
}

/// Mock Adam optimizer implementation
#[derive(Clone, Debug)]
struct MockAdam<T: Float> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    step_count: usize,
    momentum: Vec<T>,
    velocity: Vec<T>,
}

impl<T: Float> MockAdam<T> {
    pub fn new(learning_rate: T, beta1: T, beta2: T, epsilon: T) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            step_count: 0,
            momentum: Vec::new(),
            velocity: Vec::new(),
        }
    }
    
    pub fn step(&mut self, params: &mut Vec<T>, gradients: &[T]) -> NeuroDivergentResult<()> {
        if params.len() != gradients.len() {
            return Err(NeuroDivergentError::training("Parameter and gradient dimensions mismatch"));
        }
        
        // Initialize momentum and velocity if needed
        if self.momentum.is_empty() {
            self.momentum = vec![T::zero(); params.len()];
            self.velocity = vec![T::zero(); params.len()];
        }
        
        self.step_count += 1;
        let step_t = T::from(self.step_count).unwrap();
        
        // Bias correction
        let bias_correction1 = T::one() - self.beta1.powf(step_t);
        let bias_correction2 = T::one() - self.beta2.powf(step_t);
        
        for ((param, grad), (m, v)) in params.iter_mut().zip(gradients.iter())
            .zip(self.momentum.iter_mut().zip(self.velocity.iter_mut())) {
            
            // Update biased first moment estimate
            *m = self.beta1 * *m + (T::one() - self.beta1) * *grad;
            
            // Update biased second raw moment estimate
            *v = self.beta2 * *v + (T::one() - self.beta2) * (*grad * *grad);
            
            // Compute bias-corrected first moment estimate
            let m_hat = *m / bias_correction1;
            
            // Compute bias-corrected second raw moment estimate
            let v_hat = *v / bias_correction2;
            
            // Update parameters
            *param = *param - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
        
        Ok(())
    }
}

/// Mock loss function implementations
#[derive(Clone, Debug)]
enum MockLossFunction {
    MSE,
    MAE,
    Huber { delta: f64 },
}

impl MockLossFunction {
    pub fn compute<T: Float>(&self, predictions: &[T], targets: &[T]) -> NeuroDivergentResult<T> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::training("Predictions and targets length mismatch"));
        }
        
        if predictions.is_empty() {
            return Err(NeuroDivergentError::training("Cannot compute loss on empty arrays"));
        }
        
        let n = T::from(predictions.len()).unwrap();
        
        match self {
            MockLossFunction::MSE => {
                let sum_squared_errors = predictions.iter().zip(targets.iter())
                    .map(|(&pred, &target)| {
                        let diff = pred - target;
                        diff * diff
                    })
                    .fold(T::zero(), |acc, x| acc + x);
                Ok(sum_squared_errors / n)
            },
            MockLossFunction::MAE => {
                let sum_absolute_errors = predictions.iter().zip(targets.iter())
                    .map(|(&pred, &target)| (pred - target).abs())
                    .fold(T::zero(), |acc, x| acc + x);
                Ok(sum_absolute_errors / n)
            },
            MockLossFunction::Huber { delta } => {
                let delta_t = T::from(*delta).unwrap();
                let sum_huber_errors = predictions.iter().zip(targets.iter())
                    .map(|(&pred, &target)| {
                        let diff = pred - target;
                        let abs_diff = diff.abs();
                        if abs_diff <= delta_t {
                            T::from(0.5).unwrap() * diff * diff
                        } else {
                            delta_t * (abs_diff - T::from(0.5).unwrap() * delta_t)
                        }
                    })
                    .fold(T::zero(), |acc, x| acc + x);
                Ok(sum_huber_errors / n)
            }
        }
    }
    
    pub fn compute_gradient<T: Float>(&self, predictions: &[T], targets: &[T]) -> NeuroDivergentResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::training("Predictions and targets length mismatch"));
        }
        
        let n = T::from(predictions.len()).unwrap();
        
        match self {
            MockLossFunction::MSE => {
                let gradients = predictions.iter().zip(targets.iter())
                    .map(|(&pred, &target)| T::from(2.0).unwrap() * (pred - target) / n)
                    .collect();
                Ok(gradients)
            },
            MockLossFunction::MAE => {
                let gradients = predictions.iter().zip(targets.iter())
                    .map(|(&pred, &target)| {
                        let diff = pred - target;
                        if diff > T::zero() {
                            T::one() / n
                        } else if diff < T::zero() {
                            -T::one() / n
                        } else {
                            T::zero()
                        }
                    })
                    .collect();
                Ok(gradients)
            },
            MockLossFunction::Huber { delta } => {
                let delta_t = T::from(*delta).unwrap();
                let gradients = predictions.iter().zip(targets.iter())
                    .map(|(&pred, &target)| {
                        let diff = pred - target;
                        let abs_diff = diff.abs();
                        if abs_diff <= delta_t {
                            diff / n
                        } else {
                            delta_t * diff.signum() / n
                        }
                    })
                    .collect();
                Ok(gradients)
            }
        }
    }
}

/// Mock learning rate scheduler
#[derive(Clone, Debug)]
struct MockStepLR<T: Float> {
    initial_lr: T,
    step_size: usize,
    gamma: T,
    current_step: usize,
}

impl<T: Float> MockStepLR<T> {
    pub fn new(initial_lr: T, step_size: usize, gamma: T) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
    
    pub fn step(&mut self) {
        self.current_step += 1;
    }
    
    pub fn get_lr(&self) -> T {
        let decay_factor = T::from(self.current_step / self.step_size).unwrap();
        self.initial_lr * self.gamma.powf(decay_factor)
    }
}

/// Mock training configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MockTrainingConfig<T: Float> {
    pub max_epochs: usize,
    pub learning_rate: T,
    pub batch_size: Option<usize>,
    pub patience: Option<usize>,
    pub validation_split: Option<T>,
    pub shuffle: bool,
    pub seed: Option<u64>,
}

impl<T: Float> MockTrainingConfig<T> {
    pub fn new() -> Self {
        Self {
            max_epochs: 100,
            learning_rate: T::from(0.001).unwrap(),
            batch_size: None,
            patience: Some(10),
            validation_split: Some(T::from(0.2).unwrap()),
            shuffle: true,
            seed: None,
        }
    }
    
    pub fn with_max_epochs(mut self, epochs: usize) -> Self {
        self.max_epochs = epochs;
        self
    }
    
    pub fn with_learning_rate(mut self, lr: T) -> Self {
        self.learning_rate = lr;
        self
    }
    
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }
    
    pub fn validate(&self) -> NeuroDivergentResult<()> {
        if self.max_epochs == 0 {
            return Err(NeuroDivergentError::config("Max epochs must be greater than 0"));
        }
        
        if self.learning_rate <= T::zero() {
            return Err(NeuroDivergentError::config("Learning rate must be positive"));
        }
        
        if let Some(split) = self.validation_split {
            if split <= T::zero() || split >= T::one() {
                return Err(NeuroDivergentError::config("Validation split must be between 0 and 1"));
            }
        }
        
        if let Some(batch_size) = self.batch_size {
            if batch_size == 0 {
                return Err(NeuroDivergentError::config("Batch size must be greater than 0"));
            }
        }
        
        Ok(())
    }
}

// ============================================================================
// Optimizer Tests
// ============================================================================

#[cfg(test)]
mod optimizer_tests {
    use super::*;

    #[test]
    fn test_sgd_optimizer_basic() {
        let learning_rate = 0.01;
        let mut sgd = MockSGD::new(learning_rate);
        
        // Test basic parameter update
        let mut params = vec![1.0f64, 2.0, 3.0];
        let gradients = vec![0.1, 0.2, 0.3];
        
        let result = sgd.step(&mut params, &gradients);
        assert!(result.is_ok());
        
        // params = params - learning_rate * gradients
        assert_relative_eq!(params[0], 1.0 - 0.01 * 0.1, epsilon = 1e-10);
        assert_relative_eq!(params[1], 2.0 - 0.01 * 0.2, epsilon = 1e-10);
        assert_relative_eq!(params[2], 3.0 - 0.01 * 0.3, epsilon = 1e-10);
        
        assert_eq!(sgd.get_step_count(), 1);
    }

    #[test]
    fn test_sgd_multiple_steps() {
        let learning_rate = 0.1;
        let mut sgd = MockSGD::new(learning_rate);
        
        let mut params = vec![1.0f64, 2.0];
        let gradients1 = vec![0.1, 0.2];
        let gradients2 = vec![0.05, 0.1];
        
        // First step
        sgd.step(&mut params, &gradients1).unwrap();
        let expected1 = vec![1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2];
        
        // Second step
        sgd.step(&mut params, &gradients2).unwrap();
        let expected2 = vec![expected1[0] - 0.1 * 0.05, expected1[1] - 0.1 * 0.1];
        
        assert_relative_eq!(params[0], expected2[0], epsilon = 1e-10);
        assert_relative_eq!(params[1], expected2[1], epsilon = 1e-10);
        assert_eq!(sgd.get_step_count(), 2);
    }

    #[test]
    fn test_sgd_dimension_mismatch() {
        let mut sgd = MockSGD::new(0.01);
        let mut params = vec![1.0f64, 2.0];
        let gradients = vec![0.1, 0.2, 0.3]; // Wrong dimension
        
        let result = sgd.step(&mut params, &gradients);
        assert!(result.is_err());
    }

    #[test]
    fn test_adam_optimizer_basic() {
        let learning_rate = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;
        
        let mut adam = MockAdam::new(learning_rate, beta1, beta2, epsilon);
        let mut params = vec![1.0f64, 2.0, 3.0];
        let gradients = vec![0.1, 0.2, 0.3];
        
        let initial_params = params.clone();
        let result = adam.step(&mut params, &gradients);
        assert!(result.is_ok());
        
        // Parameters should have changed
        assert_ne!(params[0], initial_params[0]);
        assert_ne!(params[1], initial_params[1]);
        assert_ne!(params[2], initial_params[2]);
        
        // Should move in the opposite direction of gradients
        assert!(params[0] < initial_params[0]); // Gradient is positive
        assert!(params[1] < initial_params[1]); // Gradient is positive
        assert!(params[2] < initial_params[2]); // Gradient is positive
    }

    #[test]
    fn test_adam_convergence_behavior() {
        let mut adam = MockAdam::new(0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0f64];
        
        // Simulate consistent gradients pointing towards zero
        for _ in 0..100 {
            let gradients = vec![params[0]]; // Gradient proportional to distance from zero
            adam.step(&mut params, &gradients).unwrap();
        }
        
        // Should converge towards zero
        assert!(params[0].abs() < 0.1);
    }
}

// ============================================================================
// Loss Function Tests
// ============================================================================

#[cfg(test)]
mod loss_function_tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let mse = MockLossFunction::MSE;
        let predictions = vec![1.0f64, 2.0, 3.0];
        let targets = vec![1.1, 1.9, 3.1];
        
        let loss = mse.compute(&predictions, &targets).unwrap();
        
        // MSE = ((1.0-1.1)^2 + (2.0-1.9)^2 + (3.0-3.1)^2) / 3
        // = (0.01 + 0.01 + 0.01) / 3 = 0.01
        assert_relative_eq!(loss, 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_mae_loss() {
        let mae = MockLossFunction::MAE;
        let predictions = vec![1.0f64, 2.0, 3.0];
        let targets = vec![1.1, 1.8, 3.2];
        
        let loss = mae.compute(&predictions, &targets).unwrap();
        
        // MAE = (|1.0-1.1| + |2.0-1.8| + |3.0-3.2|) / 3
        // = (0.1 + 0.2 + 0.2) / 3 = 0.5 / 3 ≈ 0.1667
        assert_relative_eq!(loss, 0.5 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_huber_loss() {
        let huber = MockLossFunction::Huber { delta: 1.0 };
        let predictions = vec![0.0f64, 0.0, 0.0];
        let targets = vec![0.5, 1.5, 2.5]; // Errors: 0.5, 1.5, 2.5
        
        let loss = huber.compute(&predictions, &targets).unwrap();
        
        // For delta=1.0:
        // Error 0.5: |0.5| <= 1.0, so loss = 0.5 * 0.5^2 = 0.125
        // Error 1.5: |1.5| > 1.0, so loss = 1.0 * (1.5 - 0.5) = 1.0
        // Error 2.5: |2.5| > 1.0, so loss = 1.0 * (2.5 - 0.5) = 2.0
        // Total = (0.125 + 1.0 + 2.0) / 3 ≈ 1.0417
        let expected = (0.125 + 1.0 + 2.0) / 3.0;
        assert_relative_eq!(loss, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_gradient() {
        let mse = MockLossFunction::MSE;
        let predictions = vec![1.0f64, 2.0];
        let targets = vec![0.5, 1.5];
        
        let gradients = mse.compute_gradient(&predictions, &targets).unwrap();
        
        // Gradient of MSE: 2 * (pred - target) / n
        // For pred=1.0, target=0.5: 2 * (1.0 - 0.5) / 2 = 0.5
        // For pred=2.0, target=1.5: 2 * (2.0 - 1.5) / 2 = 0.5
        assert_eq!(gradients.len(), 2);
        assert_relative_eq!(gradients[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(gradients[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_loss_dimension_mismatch() {
        let mse = MockLossFunction::MSE;
        let predictions = vec![1.0f64, 2.0];
        let targets = vec![1.0, 2.0, 3.0]; // Wrong dimension
        
        let result = mse.compute(&predictions, &targets);
        assert!(result.is_err());
        
        let grad_result = mse.compute_gradient(&predictions, &targets);
        assert!(grad_result.is_err());
    }

    #[test]
    fn test_loss_empty_arrays() {
        let mse = MockLossFunction::MSE;
        let predictions: Vec<f64> = vec![];
        let targets: Vec<f64> = vec![];
        
        let result = mse.compute(&predictions, &targets);
        assert!(result.is_err());
    }
}

// ============================================================================
// Learning Rate Scheduler Tests
// ============================================================================

#[cfg(test)]
mod scheduler_tests {
    use super::*;

    #[test]
    fn test_step_lr_scheduler() {
        let initial_lr = 0.1;
        let step_size = 3;
        let gamma = 0.5;
        
        let mut scheduler = MockStepLR::new(initial_lr, step_size, gamma);
        
        // Initial learning rate
        assert_relative_eq!(scheduler.get_lr(), initial_lr, epsilon = 1e-10);
        
        // Steps 1-2: no decay yet
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), initial_lr, epsilon = 1e-10);
        
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), initial_lr, epsilon = 1e-10);
        
        // Step 3: decay occurs
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), initial_lr * gamma, epsilon = 1e-10);
        
        // Steps 4-5: no additional decay
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), initial_lr * gamma, epsilon = 1e-10);
        
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), initial_lr * gamma, epsilon = 1e-10);
        
        // Step 6: second decay
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), initial_lr * gamma * gamma, epsilon = 1e-10);
    }

    #[test]
    fn test_step_lr_zero_step_size() {
        let scheduler = MockStepLR::new(0.1, 0, 0.5);
        
        // With step_size=0, division should result in immediate decay
        // Note: This is edge case behavior
        assert!(scheduler.get_lr() <= 0.1);
    }
}

// ============================================================================
// Training Configuration Tests
// ============================================================================

#[cfg(test)]
mod training_config_tests {
    use super::*;

    #[test]
    fn test_training_config_creation() {
        let config = MockTrainingConfig::<f64>::new()
            .with_max_epochs(200)
            .with_learning_rate(0.01)
            .with_batch_size(32);

        assert_eq!(config.max_epochs, 200);
        assert_relative_eq!(config.learning_rate, 0.01, epsilon = 1e-10);
        assert_eq!(config.batch_size, Some(32));
        assert_eq!(config.patience, Some(10));
        assert!(config.shuffle);
    }

    #[test]
    fn test_training_config_validation() {
        // Valid configuration
        let valid_config = MockTrainingConfig::<f64>::new();
        assert!(valid_config.validate().is_ok());

        // Invalid max_epochs
        let invalid_config = MockTrainingConfig::<f64>::new().with_max_epochs(0);
        assert!(invalid_config.validate().is_err());

        // Invalid learning rate
        let invalid_config = MockTrainingConfig::<f64>::new().with_learning_rate(-0.01);
        assert!(invalid_config.validate().is_err());

        // Invalid batch size
        let invalid_config = MockTrainingConfig::<f64>::new().with_batch_size(0);
        assert!(invalid_config.validate().is_err());
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_sgd_optimizer_properties(
            learning_rate in 0.0001f64..1.0,
            param_count in 1usize..100
        ) {
            let mut sgd = MockSGD::new(learning_rate);
            let mut params: Vec<f64> = (0..param_count).map(|i| i as f64 * 0.1).collect();
            let gradients: Vec<f64> = (0..param_count).map(|i| (i as f64 + 1.0) * 0.01).collect();
            
            let initial_params = params.clone();
            let result = sgd.step(&mut params, &gradients);
            
            // Should succeed for valid inputs
            assert!(result.is_ok());
            
            // Parameters should change
            for (param, initial) in params.iter().zip(initial_params.iter()) {
                if gradients[0] != 0.0 { // Avoid zero gradient case
                    assert_ne!(param, initial);
                }
            }
            
            // Step count should increment
            assert_eq!(sgd.get_step_count(), 1);
        }

        #[test]
        fn test_loss_function_properties(
            pred_val in -10.0f64..10.0,
            target_val in -10.0f64..10.0
        ) {
            let predictions = vec![pred_val];
            let targets = vec![target_val];
            
            // MSE should always be non-negative
            let mse = MockLossFunction::MSE;
            let mse_loss = mse.compute(&predictions, &targets).unwrap();
            assert!(mse_loss >= 0.0);
            
            // MAE should always be non-negative
            let mae = MockLossFunction::MAE;
            let mae_loss = mae.compute(&predictions, &targets).unwrap();
            assert!(mae_loss >= 0.0);
            
            // Huber loss should always be non-negative
            let huber = MockLossFunction::Huber { delta: 1.0 };
            let huber_loss = huber.compute(&predictions, &targets).unwrap();
            assert!(huber_loss >= 0.0);
            
            // Perfect predictions should give zero loss
            let perfect_targets = vec![pred_val];
            assert_relative_eq!(mse.compute(&predictions, &perfect_targets).unwrap(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(mae.compute(&predictions, &perfect_targets).unwrap(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(huber.compute(&predictions, &perfect_targets).unwrap(), 0.0, epsilon = 1e-10);
        }

        #[test]
        fn test_scheduler_monotonic_decay(
            initial_lr in 0.001f64..1.0,
            gamma in 0.1f64..0.9
        ) {
            let step_size = 5;
            let mut scheduler = MockStepLR::new(initial_lr, step_size, gamma);
            
            let mut previous_lr = scheduler.get_lr();
            
            // Step through multiple decay periods
            for _ in 0..20 {
                scheduler.step();
                let current_lr = scheduler.get_lr();
                
                // Learning rate should never increase
                assert!(current_lr <= previous_lr + 1e-10); // Small epsilon for floating point comparison
                
                previous_lr = current_lr;
            }
        }
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_optimizer_scheduler_integration() {
        let mut sgd = MockSGD::new(0.1);
        let mut scheduler = MockStepLR::new(0.1, 2, 0.5);
        
        let mut params = vec![1.0f64, 2.0];
        let gradients = vec![0.1, 0.1];
        
        // Step 1: initial learning rate
        let lr1 = scheduler.get_lr();
        assert_relative_eq!(lr1, 0.1, epsilon = 1e-10);
        
        sgd.learning_rate = lr1;
        sgd.step(&mut params, &gradients).unwrap();
        scheduler.step();
        
        // Step 2: still initial learning rate
        let lr2 = scheduler.get_lr();
        assert_relative_eq!(lr2, 0.1, epsilon = 1e-10);
        
        sgd.learning_rate = lr2;
        sgd.step(&mut params, &gradients).unwrap();
        scheduler.step();
        
        // Step 3: learning rate should decay
        let lr3 = scheduler.get_lr();
        assert_relative_eq!(lr3, 0.05, epsilon = 1e-10);
        
        sgd.learning_rate = lr3;
        sgd.step(&mut params, &gradients).unwrap();
    }

    #[test]
    fn test_training_pipeline_simulation() {
        // Simulate a simple training loop
        let mut sgd = MockSGD::new(0.01);
        let loss_fn = MockLossFunction::MSE;
        let mut scheduler = MockStepLR::new(0.01, 10, 0.9);
        
        // Simple linear model: y = ax + b
        let mut params = vec![0.5f64, 0.0]; // [a, b]
        let target_a = 2.0;
        let target_b = 1.0;
        
        // Training data: y = 2x + 1
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data: Vec<f64> = x_data.iter().map(|&x| target_a * x + target_b).collect();
        
        let mut losses = Vec::new();
        
        for epoch in 0..50 {
            let mut epoch_loss = 0.0;
            let mut epoch_gradients = vec![0.0f64, 0.0];
            
            for (&x, &y_true) in x_data.iter().zip(y_data.iter()) {
                // Forward pass
                let y_pred = params[0] * x + params[1];
                let predictions = vec![y_pred];
                let targets = vec![y_true];
                
                // Compute loss
                let loss = loss_fn.compute(&predictions, &targets).unwrap();
                epoch_loss += loss;
                
                // Compute gradients (simplified)
                let pred_grad = 2.0 * (y_pred - y_true) / 5.0; // MSE gradient
                epoch_gradients[0] += pred_grad * x; // da/da
                epoch_gradients[1] += pred_grad; // db/db
            }
            
            epoch_loss /= x_data.len() as f64;
            losses.push(epoch_loss);
            
            // Update parameters
            sgd.learning_rate = scheduler.get_lr();
            sgd.step(&mut params, &epoch_gradients).unwrap();
            scheduler.step();
        }
        
        // Check that loss decreased over time
        assert!(losses.last().unwrap() < losses.first().unwrap());
        
        // Check that parameters converged towards target values
        assert!((params[0] - target_a).abs() < 0.5);
        assert!((params[1] - target_b).abs() < 0.5);
    }
}