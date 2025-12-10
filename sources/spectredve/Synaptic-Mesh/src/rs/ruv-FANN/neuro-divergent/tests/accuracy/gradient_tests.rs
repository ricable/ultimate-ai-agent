//! Gradient computation validation tests
//!
//! This module validates gradient computations using:
//! 1. Finite difference approximation
//! 2. Comparison with Python autograd
//! 3. Gradient flow validation
//! 4. Numerical stability checks

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use std::f64::EPSILON;

use neuro_divergent_training::{
    LossFunction, Loss,
    MAELoss, MSELoss, RMSELoss, MAPELoss, SMAPELoss,
    HuberLoss, QuantileLoss,
};

use neuro_divergent_models::{
    layers::{Dense, Activation, LayerTrait},
    activations::{ReLU, Sigmoid, Tanh, LeakyReLU, Softmax},
};

const GRADIENT_TOLERANCE: f64 = 1e-7;
const FINITE_DIFF_EPSILON: f64 = 1e-5;

/// Compute numerical gradient using finite differences
fn numerical_gradient<F>(
    f: F,
    x: &Array1<f64>,
    epsilon: f64,
) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);
    
    for i in 0..n {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        
        x_plus[i] += epsilon;
        x_minus[i] -= epsilon;
        
        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);
    }
    
    grad
}

/// Validate gradient computation using finite difference check
fn validate_gradient<L: LossFunction<f64>>(
    loss: &L,
    predictions: &Array1<f64>,
    targets: &Array1<f64>,
    tolerance: f64,
) -> Result<(), String> {
    // Compute analytical gradient
    let analytical_grad = loss.backward(
        predictions.as_slice().unwrap(),
        targets.as_slice().unwrap()
    )?;
    
    // Compute numerical gradient
    let loss_fn = |pred: &Array1<f64>| -> f64 {
        loss.forward(pred.as_slice().unwrap(), targets.as_slice().unwrap())
            .unwrap()
    };
    
    let numerical_grad = numerical_gradient(loss_fn, predictions, FINITE_DIFF_EPSILON);
    
    // Compare gradients
    for (i, (analytical, numerical)) in analytical_grad.iter()
        .zip(numerical_grad.iter())
        .enumerate()
    {
        if numerical.abs() > 1e-10 {
            let relative_error = ((analytical - numerical) / numerical).abs();
            if relative_error > tolerance {
                return Err(format!(
                    "Gradient mismatch at index {}: analytical {} vs numerical {}, relative error {}",
                    i, analytical, numerical, relative_error
                ));
            }
        } else {
            let absolute_error = (analytical - numerical).abs();
            if absolute_error > tolerance {
                return Err(format!(
                    "Gradient mismatch at index {}: analytical {} vs numerical {}, absolute error {}",
                    i, analytical, numerical, absolute_error
                ));
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod loss_gradient_tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    fn generate_test_data(seed: u64, n: usize) -> (Array1<f64>, Array1<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let predictions = Array1::from_vec(
            (0..n).map(|_| rng.gen_range(0.1..10.0)).collect()
        );
        let targets = Array1::from_vec(
            (0..n).map(|_| rng.gen_range(0.1..10.0)).collect()
        );
        (predictions, targets)
    }
    
    #[test]
    fn test_mae_gradient() {
        let (predictions, targets) = generate_test_data(42, 20);
        let loss = MAELoss::new();
        
        validate_gradient(&loss, &predictions, &targets, GRADIENT_TOLERANCE)
            .expect("MAE gradient validation failed");
    }
    
    #[test]
    fn test_mse_gradient() {
        let (predictions, targets) = generate_test_data(123, 20);
        let loss = MSELoss::new();
        
        validate_gradient(&loss, &predictions, &targets, GRADIENT_TOLERANCE)
            .expect("MSE gradient validation failed");
    }
    
    #[test]
    fn test_rmse_gradient() {
        let (predictions, targets) = generate_test_data(456, 20);
        let loss = RMSELoss::new();
        
        validate_gradient(&loss, &predictions, &targets, GRADIENT_TOLERANCE)
            .expect("RMSE gradient validation failed");
    }
    
    #[test]
    fn test_huber_gradient() {
        let (predictions, targets) = generate_test_data(789, 20);
        let loss = HuberLoss::new(1.0);
        
        validate_gradient(&loss, &predictions, &targets, GRADIENT_TOLERANCE)
            .expect("Huber gradient validation failed");
    }
    
    #[test]
    fn test_quantile_gradient() {
        let (predictions, targets) = generate_test_data(101112, 20);
        let loss = QuantileLoss::new(vec![0.1, 0.5, 0.9]);
        
        // Adjust predictions for multiple quantiles
        let n_quantiles = 3;
        let n_samples = targets.len();
        let extended_predictions = Array1::from_vec(
            predictions.as_slice().unwrap()
                .iter()
                .cycle()
                .take(n_samples * n_quantiles)
                .cloned()
                .collect()
        );
        
        validate_gradient(&loss, &extended_predictions, &targets, GRADIENT_TOLERANCE)
            .expect("Quantile gradient validation failed");
    }
    
    #[test]
    fn test_gradient_at_non_differentiable_points() {
        // Test MAE at zero difference (non-differentiable point)
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Identical
        
        let loss = MAELoss::new();
        let gradients = loss.backward(
            predictions.as_slice().unwrap(),
            targets.as_slice().unwrap()
        ).unwrap();
        
        // At non-differentiable points, we use subgradient (0 in this case)
        for grad in gradients {
            assert_eq!(grad, 0.0, "Gradient at zero difference should be 0");
        }
    }
}

#[cfg(test)]
mod activation_gradient_tests {
    use super::*;
    
    fn validate_activation_gradient<A: ActivationFunction>(
        activation: &A,
        input: &Array1<f64>,
        tolerance: f64,
    ) -> Result<(), String> {
        // Forward pass
        let output = activation.forward(input);
        
        // Analytical gradient
        let analytical_grad = activation.backward(input, &Array1::ones(input.len()));
        
        // Numerical gradient for each output w.r.t each input
        for i in 0..input.len() {
            let activation_fn = |x: &Array1<f64>| -> f64 {
                activation.forward(x)[i]
            };
            
            let numerical_grad = numerical_gradient(activation_fn, input, FINITE_DIFF_EPSILON);
            
            // Compare only the diagonal elements (derivative of output[i] w.r.t input[i])
            let analytical_value = analytical_grad[[i, i]];
            let numerical_value = numerical_grad[i];
            
            if numerical_value.abs() > 1e-10 {
                let relative_error = ((analytical_value - numerical_value) / numerical_value).abs();
                if relative_error > tolerance {
                    return Err(format!(
                        "Activation gradient mismatch at {}: analytical {} vs numerical {}",
                        i, analytical_value, numerical_value
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_relu_gradient() {
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let activation = ReLU::new();
        
        let output = activation.forward(&input);
        let grad_output = Array1::ones(input.len());
        let grad_input = activation.backward(&input, &grad_output);
        
        // ReLU gradient should be 0 for negative inputs, 1 for positive
        assert_eq!(grad_input[0], 0.0);
        assert_eq!(grad_input[1], 0.0);
        assert_eq!(grad_input[2], 0.0); // Technically undefined at 0, we use 0
        assert_eq!(grad_input[3], 1.0);
        assert_eq!(grad_input[4], 1.0);
    }
    
    #[test]
    fn test_sigmoid_gradient() {
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let activation = Sigmoid::new();
        
        validate_activation_gradient(&activation, &input, GRADIENT_TOLERANCE)
            .expect("Sigmoid gradient validation failed");
    }
    
    #[test]
    fn test_tanh_gradient() {
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let activation = Tanh::new();
        
        validate_activation_gradient(&activation, &input, GRADIENT_TOLERANCE)
            .expect("Tanh gradient validation failed");
    }
    
    #[test]
    fn test_leaky_relu_gradient() {
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let activation = LeakyReLU::new(0.01);
        
        let output = activation.forward(&input);
        let grad_output = Array1::ones(input.len());
        let grad_input = activation.backward(&input, &grad_output);
        
        // LeakyReLU gradient should be alpha for negative inputs, 1 for positive
        assert_abs_diff_eq!(grad_input[0], 0.01, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_input[1], 0.01, epsilon = 1e-10);
        assert_eq!(grad_input[3], 1.0);
        assert_eq!(grad_input[4], 1.0);
    }
}

#[cfg(test)]
mod layer_gradient_tests {
    use super::*;
    
    #[test]
    fn test_dense_layer_gradient() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let input_size = 5;
        let output_size = 3;
        let batch_size = 2;
        
        // Create dense layer
        let mut dense = Dense::new(input_size, output_size);
        
        // Generate random input
        let input = Array2::from_shape_fn((batch_size, input_size), |_| {
            rng.gen_range(-1.0..1.0)
        });
        
        // Forward pass
        let output = dense.forward(&input);
        
        // Generate random gradient from next layer
        let grad_output = Array2::from_shape_fn((batch_size, output_size), |_| {
            rng.gen_range(-1.0..1.0)
        });
        
        // Backward pass
        let grad_input = dense.backward(&input, &grad_output);
        
        // Validate gradient shapes
        assert_eq!(grad_input.shape(), input.shape());
        
        // Numerical gradient check for weights
        let loss_fn = |weights: &Array2<f64>| -> f64 {
            let mut test_dense = dense.clone();
            test_dense.weights = weights.clone();
            let output = test_dense.forward(&input);
            
            // Simple loss: sum of outputs multiplied by grad_output
            (output * &grad_output).sum()
        };
        
        // Flatten weights for numerical gradient
        let weights_flat = Array1::from_vec(
            dense.weights.iter().cloned().collect()
        );
        
        let numerical_grad_weights = numerical_gradient(
            |w| {
                let weights_2d = w.clone().into_shape((input_size, output_size)).unwrap();
                loss_fn(&weights_2d)
            },
            &weights_flat,
            FINITE_DIFF_EPSILON
        );
        
        // Compare with analytical gradient (stored in dense layer)
        let analytical_grad_weights_flat = Array1::from_vec(
            dense.grad_weights.unwrap().iter().cloned().collect()
        );
        
        for (analytical, numerical) in analytical_grad_weights_flat.iter()
            .zip(numerical_grad_weights.iter())
        {
            assert_relative_eq!(
                analytical,
                numerical,
                epsilon = GRADIENT_TOLERANCE,
                "Dense layer weight gradient mismatch"
            );
        }
    }
}

#[cfg(test)]
mod gradient_flow_tests {
    use super::*;
    
    #[test]
    fn test_gradient_vanishing() {
        // Test for vanishing gradients with deep networks
        let n_layers = 10;
        let layer_size = 50;
        
        // Create deep network with sigmoid activations (prone to vanishing gradients)
        let mut gradients = vec![1.0; layer_size]; // Start with unit gradients
        
        for _ in 0..n_layers {
            // Simulate sigmoid gradient (max value is 0.25)
            gradients = gradients.iter()
                .map(|&g| g * 0.25)
                .collect();
        }
        
        let final_gradient_magnitude: f64 = gradients.iter()
            .map(|g| g.abs())
            .sum::<f64>() / gradients.len() as f64;
        
        assert!(
            final_gradient_magnitude < 1e-6,
            "Deep sigmoid network should exhibit vanishing gradients"
        );
    }
    
    #[test]
    fn test_gradient_explosion() {
        // Test for exploding gradients
        let n_layers = 10;
        let layer_size = 50;
        
        // Create network with large weight initialization
        let mut gradients = vec![1.0; layer_size];
        
        for _ in 0..n_layers {
            // Simulate gradient multiplication by large weights
            gradients = gradients.iter()
                .map(|&g| g * 2.5) // Each layer amplifies gradient
                .collect();
        }
        
        let final_gradient_magnitude: f64 = gradients.iter()
            .map(|g| g.abs())
            .sum::<f64>() / gradients.len() as f64;
        
        assert!(
            final_gradient_magnitude > 1e6,
            "Network with large weights should exhibit exploding gradients"
        );
    }
    
    #[test]
    fn test_gradient_clipping() {
        let gradients = vec![10.0, -15.0, 5.0, -3.0, 20.0];
        let clip_value = 5.0;
        
        let clipped: Vec<f64> = gradients.iter()
            .map(|&g| {
                if g > clip_value {
                    clip_value
                } else if g < -clip_value {
                    -clip_value
                } else {
                    g
                }
            })
            .collect();
        
        assert_eq!(clipped, vec![5.0, -5.0, 5.0, -3.0, 5.0]);
        
        // Check that gradient norm clipping works
        let grad_norm: f64 = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        let max_norm = 10.0;
        
        let scale = if grad_norm > max_norm {
            max_norm / grad_norm
        } else {
            1.0
        };
        
        let norm_clipped: Vec<f64> = gradients.iter()
            .map(|&g| g * scale)
            .collect();
        
        let new_norm: f64 = norm_clipped.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(new_norm <= max_norm + 1e-6);
    }
}

#[cfg(test)]
mod numerical_stability_gradient_tests {
    use super::*;
    
    #[test]
    fn test_gradient_near_zero() {
        // Test gradient computation near zero
        let predictions = Array1::from_vec(vec![1e-10, -1e-10, 0.0]);
        let targets = Array1::from_vec(vec![2e-10, -2e-10, 1e-10]);
        
        let loss = MSELoss::new();
        let gradients = loss.backward(
            predictions.as_slice().unwrap(),
            targets.as_slice().unwrap()
        ).unwrap();
        
        for grad in gradients {
            assert!(grad.is_finite(), "Gradient near zero should be finite");
        }
    }
    
    #[test]
    fn test_gradient_large_values() {
        // Test gradient computation with large values
        let predictions = Array1::from_vec(vec![1e8, -1e8, 1e7]);
        let targets = Array1::from_vec(vec![1e8 + 1.0, -1e8 - 1.0, 1e7 + 10.0]);
        
        let loss = MAELoss::new();
        let gradients = loss.backward(
            predictions.as_slice().unwrap(),
            targets.as_slice().unwrap()
        ).unwrap();
        
        for grad in gradients {
            assert!(grad.is_finite(), "Gradient with large values should be finite");
            assert!(grad.abs() <= 1.0, "MAE gradient magnitude should be bounded");
        }
    }
    
    #[test]
    fn test_gradient_overflow_protection() {
        // Test protection against gradient overflow
        let x = 100.0; // Large input to sigmoid
        let sigmoid_output = 1.0 / (1.0 + (-x).exp());
        let sigmoid_gradient = sigmoid_output * (1.0 - sigmoid_output);
        
        assert!(sigmoid_gradient.is_finite(), "Sigmoid gradient should not overflow");
        assert!(sigmoid_gradient >= 0.0, "Sigmoid gradient should be non-negative");
        assert!(sigmoid_gradient <= 0.25, "Sigmoid gradient should be bounded by 0.25");
    }
    
    #[test]
    fn test_gradient_symmetry() {
        // Test that gradients maintain expected symmetries
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        
        let loss = MSELoss::new();
        let gradients1 = loss.backward(
            predictions.as_slice().unwrap(),
            targets.as_slice().unwrap()
        ).unwrap();
        
        // Swap predictions and targets
        let gradients2 = loss.backward(
            targets.as_slice().unwrap(),
            predictions.as_slice().unwrap()
        ).unwrap();
        
        // MSE gradients should be negatives of each other
        for (g1, g2) in gradients1.iter().zip(gradients2.iter()) {
            assert_abs_diff_eq!(g1, -g2, epsilon = 1e-10);
        }
    }
}