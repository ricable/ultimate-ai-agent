//! Activation functions for neural networks
//!
//! This module provides common activation functions used in neural forecasting models.

use ndarray::{Array1, Array2};
use num_traits::Float;

/// Trait for activation functions
pub trait ActivationFunction {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64>;
    fn backward(&self, input: &Array1<f64>, grad_output: &Array1<f64>) -> Array1<f64>;
    fn name(&self) -> &'static str;
}

/// ReLU activation function
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl ActivationFunction for ReLU {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| x.max(0.0))
    }
    
    fn backward(&self, input: &Array1<f64>, grad_output: &Array1<f64>) -> Array1<f64> {
        input.iter()
            .zip(grad_output.iter())
            .map(|(&x, &grad)| if x > 0.0 { grad } else { 0.0 })
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "ReLU"
    }
}

/// Sigmoid activation function
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl ActivationFunction for Sigmoid {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
    
    fn backward(&self, input: &Array1<f64>, grad_output: &Array1<f64>) -> Array1<f64> {
        let output = self.forward(input);
        output.iter()
            .zip(grad_output.iter())
            .map(|(&s, &grad)| grad * s * (1.0 - s))
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "Sigmoid"
    }
}

/// Tanh activation function
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl ActivationFunction for Tanh {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| x.tanh())
    }
    
    fn backward(&self, input: &Array1<f64>, grad_output: &Array1<f64>) -> Array1<f64> {
        let output = self.forward(input);
        output.iter()
            .zip(grad_output.iter())
            .map(|(&t, &grad)| grad * (1.0 - t * t))
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "Tanh"
    }
}

/// LeakyReLU activation function
pub struct LeakyReLU {
    alpha: f64,
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl ActivationFunction for LeakyReLU {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x > 0.0 { x } else { self.alpha * x })
    }
    
    fn backward(&self, input: &Array1<f64>, grad_output: &Array1<f64>) -> Array1<f64> {
        input.iter()
            .zip(grad_output.iter())
            .map(|(&x, &grad)| if x > 0.0 { grad } else { self.alpha * grad })
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "LeakyReLU"
    }
}

/// Softmax activation function
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl ActivationFunction for Softmax {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // Subtract max for numerical stability
        let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values = input.mapv(|x| (x - max_val).exp());
        let sum = exp_values.sum();
        exp_values / sum
    }
    
    fn backward(&self, input: &Array1<f64>, grad_output: &Array1<f64>) -> Array1<f64> {
        let output = self.forward(input);
        let n = output.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    result[i] += grad_output[j] * output[i] * (1.0 - output[i]);
                } else {
                    result[i] -= grad_output[j] * output[i] * output[j];
                }
            }
        }
        
        result
    }
    
    fn name(&self) -> &'static str {
        "Softmax"
    }
}

/// LogSoftmax activation function
pub struct LogSoftmax;

impl LogSoftmax {
    pub fn new() -> Self {
        Self
    }
}

impl ActivationFunction for LogSoftmax {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // Subtract max for numerical stability
        let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let shifted = input.mapv(|x| x - max_val);
        let log_sum_exp = shifted.mapv(|x| x.exp()).sum().ln();
        shifted - log_sum_exp
    }
    
    fn backward(&self, input: &Array1<f64>, grad_output: &Array1<f64>) -> Array1<f64> {
        let output = self.forward(input);
        let softmax = output.mapv(|x| x.exp());
        let sum_grad = grad_output.sum();
        
        grad_output.iter()
            .zip(softmax.iter())
            .map(|(&grad, &s)| grad - s * sum_grad)
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "LogSoftmax"
    }
}