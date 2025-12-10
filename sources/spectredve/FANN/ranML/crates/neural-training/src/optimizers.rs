//! Advanced Optimization Algorithms for Neural Network Training
//!
//! This module provides state-of-the-art optimization algorithms including
//! SGD, Adam, AdaGrad, RMSprop, AdaDelta, and specialized variants.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ndarray::Array1;

/// Trait for all optimization algorithms
pub trait Optimizer: Send + Sync {
    /// Update parameters given gradients
    fn update(&mut self, params: &mut [f32], gradients: &[f32], step: usize);
    
    /// Reset optimizer state
    fn reset(&mut self);
    
    /// Get optimizer configuration
    fn config(&self) -> OptimizerConfig;
    
    /// Clone the optimizer
    fn clone_optimizer(&self) -> Box<dyn Optimizer>;
}

/// Optimizer configuration for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerConfig {
    SGD {
        learning_rate: f32,
        momentum: f32,
        dampening: f32,
        weight_decay: f32,
        nesterov: bool,
    },
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        amsgrad: bool,
    },
    AdaGrad {
        learning_rate: f32,
        epsilon: f32,
        weight_decay: f32,
        lr_decay: f32,
    },
    RMSprop {
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
        weight_decay: f32,
        momentum: f32,
        centered: bool,
    },
    AdaDelta {
        learning_rate: f32,
        rho: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    AdamW {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    LAMB {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    AdaBound {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        final_lr: f32,
        gamma: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    RAdam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    Lookahead {
        base_optimizer: Box<OptimizerConfig>,
        k: usize,
        alpha: f32,
    },
}

/// Stochastic Gradient Descent with momentum
#[derive(Debug, Clone)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: HashMap<usize, Array1<f32>>,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_dampening(mut self, dampening: f32) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn update(&mut self, params: &mut [f32], gradients: &[f32], step: usize) {
        let param_id = params.as_ptr() as usize;
        
        if !self.velocity.contains_key(&param_id) {
            self.velocity.insert(param_id, Array1::zeros(params.len()));
        }
        
        let velocity = self.velocity.get_mut(&param_id).unwrap();
        
        for i in 0..params.len() {
            let mut grad = gradients[i];
            
            // Add weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }
            
            if self.momentum != 0.0 {
                if step > 1 {
                    velocity[i] = self.momentum * velocity[i] + (1.0 - self.dampening) * grad;
                } else {
                    velocity[i] = grad;
                }
                
                if self.nesterov {
                    grad = grad + self.momentum * velocity[i];
                } else {
                    grad = velocity[i];
                }
            }
            
            params[i] -= self.learning_rate * grad;
        }
    }

    fn reset(&mut self) {
        self.velocity.clear();
    }

    fn config(&self) -> OptimizerConfig {
        OptimizerConfig::SGD {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            dampening: self.dampening,
            weight_decay: self.weight_decay,
            nesterov: self.nesterov,
        }
    }

    fn clone_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
#[derive(Debug, Clone)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    amsgrad: bool,
    momentum: HashMap<usize, Array1<f32>>,
    velocity: HashMap<usize, Array1<f32>>,
    max_velocity: HashMap<usize, Array1<f32>>, // For AMSGrad
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            max_velocity: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl Optimizer for Adam {
    fn update(&mut self, params: &mut [f32], gradients: &[f32], step: usize) {
        let param_id = params.as_ptr() as usize;
        
        if !self.momentum.contains_key(&param_id) {
            self.momentum.insert(param_id, Array1::zeros(params.len()));
            self.velocity.insert(param_id, Array1::zeros(params.len()));
            if self.amsgrad {
                self.max_velocity.insert(param_id, Array1::zeros(params.len()));
            }
        }
        
        let momentum = self.momentum.get_mut(&param_id).unwrap();
        let velocity = self.velocity.get_mut(&param_id).unwrap();
        
        for i in 0..params.len() {
            let mut grad = gradients[i];
            
            // Add weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }
            
            // Update biased first moment estimate
            momentum[i] = self.beta1 * momentum[i] + (1.0 - self.beta1) * grad;
            
            // Update biased second raw moment estimate
            velocity[i] = self.beta2 * velocity[i] + (1.0 - self.beta2) * grad * grad;
            
            // Compute bias-corrected first moment estimate
            let m_hat = momentum[i] / (1.0 - self.beta1.powi(step as i32));
            
            // Compute bias-corrected second raw moment estimate
            let mut v_hat = velocity[i] / (1.0 - self.beta2.powi(step as i32));
            
            // AMSGrad variant
            if self.amsgrad {
                let max_velocity = self.max_velocity.get_mut(&param_id).unwrap();
                max_velocity[i] = max_velocity[i].max(velocity[i]);
                v_hat = max_velocity[i] / (1.0 - self.beta2.powi(step as i32));
            }
            
            // Update parameters
            params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    fn reset(&mut self) {
        self.momentum.clear();
        self.velocity.clear();
        self.max_velocity.clear();
    }

    fn config(&self) -> OptimizerConfig {
        OptimizerConfig::Adam {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
            amsgrad: self.amsgrad,
        }
    }

    fn clone_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// AdaGrad optimizer
#[derive(Debug, Clone)]
pub struct AdaGrad {
    learning_rate: f32,
    epsilon: f32,
    weight_decay: f32,
    lr_decay: f32,
    sum_squared_gradients: HashMap<usize, Array1<f32>>,
}

impl AdaGrad {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            epsilon: 1e-10,
            weight_decay: 0.0,
            lr_decay: 0.0,
            sum_squared_gradients: HashMap::new(),
        }
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_lr_decay(mut self, lr_decay: f32) -> Self {
        self.lr_decay = lr_decay;
        self
    }
}

impl Optimizer for AdaGrad {
    fn update(&mut self, params: &mut [f32], gradients: &[f32], step: usize) {
        let param_id = params.as_ptr() as usize;
        
        if !self.sum_squared_gradients.contains_key(&param_id) {
            self.sum_squared_gradients.insert(param_id, Array1::zeros(params.len()));
        }
        
        let sum_squared_gradients = self.sum_squared_gradients.get_mut(&param_id).unwrap();
        let clr = self.learning_rate / (1.0 + (step - 1) as f32 * self.lr_decay);
        
        for i in 0..params.len() {
            let mut grad = gradients[i];
            
            // Add weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }
            
            // Accumulate gradient squared
            sum_squared_gradients[i] += grad * grad;
            
            // Update parameters
            let std = sum_squared_gradients[i].sqrt() + self.epsilon;
            params[i] -= clr * grad / std;
        }
    }

    fn reset(&mut self) {
        self.sum_squared_gradients.clear();
    }

    fn config(&self) -> OptimizerConfig {
        OptimizerConfig::AdaGrad {
            learning_rate: self.learning_rate,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
            lr_decay: self.lr_decay,
        }
    }

    fn clone_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// RMSprop optimizer
#[derive(Debug, Clone)]
pub struct RMSprop {
    learning_rate: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    square_avg: HashMap<usize, Array1<f32>>,
    momentum_buffer: HashMap<usize, Array1<f32>>,
    grad_avg: HashMap<usize, Array1<f32>>, // For centered variant
}

impl RMSprop {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }
}

impl Optimizer for RMSprop {
    fn update(&mut self, params: &mut [f32], gradients: &[f32], step: usize) {
        let param_id = params.as_ptr() as usize;
        
        if !self.square_avg.contains_key(&param_id) {
            self.square_avg.insert(param_id, Array1::zeros(params.len()));
            if self.momentum > 0.0 {
                self.momentum_buffer.insert(param_id, Array1::zeros(params.len()));
            }
            if self.centered {
                self.grad_avg.insert(param_id, Array1::zeros(params.len()));
            }
        }
        
        let square_avg = self.square_avg.get_mut(&param_id).unwrap();
        
        for i in 0..params.len() {
            let mut grad = gradients[i];
            
            // Add weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }
            
            // Update exponential moving average of squared gradients
            square_avg[i] = self.alpha * square_avg[i] + (1.0 - self.alpha) * grad * grad;
            
            let mut avg = square_avg[i];
            
            // Centered RMSprop
            if self.centered {
                let grad_avg = self.grad_avg.get_mut(&param_id).unwrap();
                grad_avg[i] = self.alpha * grad_avg[i] + (1.0 - self.alpha) * grad;
                avg = square_avg[i] - grad_avg[i] * grad_avg[i];
            }
            
            if self.momentum > 0.0 {
                let momentum_buffer = self.momentum_buffer.get_mut(&param_id).unwrap();
                momentum_buffer[i] = self.momentum * momentum_buffer[i] + grad / (avg.sqrt() + self.epsilon);
                params[i] -= self.learning_rate * momentum_buffer[i];
            } else {
                params[i] -= self.learning_rate * grad / (avg.sqrt() + self.epsilon);
            }
        }
    }

    fn reset(&mut self) {
        self.square_avg.clear();
        self.momentum_buffer.clear();
        self.grad_avg.clear();
    }

    fn config(&self) -> OptimizerConfig {
        OptimizerConfig::RMSprop {
            learning_rate: self.learning_rate,
            alpha: self.alpha,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
            momentum: self.momentum,
            centered: self.centered,
        }
    }

    fn clone_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// AdaDelta optimizer
#[derive(Debug, Clone)]
pub struct AdaDelta {
    learning_rate: f32,
    rho: f32,
    epsilon: f32,
    weight_decay: f32,
    square_avg: HashMap<usize, Array1<f32>>,
    acc_delta: HashMap<usize, Array1<f32>>,
}

impl AdaDelta {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            rho: 0.9,
            epsilon: 1e-6,
            weight_decay: 0.0,
            square_avg: HashMap::new(),
            acc_delta: HashMap::new(),
        }
    }

    pub fn with_rho(mut self, rho: f32) -> Self {
        self.rho = rho;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdaDelta {
    fn update(&mut self, params: &mut [f32], gradients: &[f32], _step: usize) {
        let param_id = params.as_ptr() as usize;
        
        if !self.square_avg.contains_key(&param_id) {
            self.square_avg.insert(param_id, Array1::zeros(params.len()));
            self.acc_delta.insert(param_id, Array1::zeros(params.len()));
        }
        
        let square_avg = self.square_avg.get_mut(&param_id).unwrap();
        let acc_delta = self.acc_delta.get_mut(&param_id).unwrap();
        
        for i in 0..params.len() {
            let mut grad = gradients[i];
            
            // Add weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }
            
            // Update accumulator of gradients squared
            square_avg[i] = self.rho * square_avg[i] + (1.0 - self.rho) * grad * grad;
            
            // Compute update
            let std_grad = (square_avg[i] + self.epsilon).sqrt();
            let std_delta = (acc_delta[i] + self.epsilon).sqrt();
            let delta = (std_delta / std_grad) * grad;
            
            // Update parameters
            params[i] -= self.learning_rate * delta;
            
            // Update accumulator of updates squared
            acc_delta[i] = self.rho * acc_delta[i] + (1.0 - self.rho) * delta * delta;
        }
    }

    fn reset(&mut self) {
        self.square_avg.clear();
        self.acc_delta.clear();
    }

    fn config(&self) -> OptimizerConfig {
        OptimizerConfig::AdaDelta {
            learning_rate: self.learning_rate,
            rho: self.rho,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        }
    }

    fn clone_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
#[derive(Debug, Clone)]
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: HashMap<usize, Array1<f32>>,
    velocity: HashMap<usize, Array1<f32>>,
}

impl AdamW {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamW {
    fn update(&mut self, params: &mut [f32], gradients: &[f32], step: usize) {
        let param_id = params.as_ptr() as usize;
        
        if !self.momentum.contains_key(&param_id) {
            self.momentum.insert(param_id, Array1::zeros(params.len()));
            self.velocity.insert(param_id, Array1::zeros(params.len()));
        }
        
        let momentum = self.momentum.get_mut(&param_id).unwrap();
        let velocity = self.velocity.get_mut(&param_id).unwrap();
        
        for i in 0..params.len() {
            let grad = gradients[i];
            
            // Update biased first moment estimate
            momentum[i] = self.beta1 * momentum[i] + (1.0 - self.beta1) * grad;
            
            // Update biased second raw moment estimate
            velocity[i] = self.beta2 * velocity[i] + (1.0 - self.beta2) * grad * grad;
            
            // Compute bias-corrected first moment estimate
            let m_hat = momentum[i] / (1.0 - self.beta1.powi(step as i32));
            
            // Compute bias-corrected second raw moment estimate
            let v_hat = velocity[i] / (1.0 - self.beta2.powi(step as i32));
            
            // Update parameters with decoupled weight decay
            params[i] -= self.learning_rate * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * params[i]);
        }
    }

    fn reset(&mut self) {
        self.momentum.clear();
        self.velocity.clear();
    }

    fn config(&self) -> OptimizerConfig {
        OptimizerConfig::AdamW {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        }
    }

    fn clone_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// RAdam optimizer (Rectified Adam)
#[derive(Debug, Clone)]
pub struct RAdam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: HashMap<usize, Array1<f32>>,
    velocity: HashMap<usize, Array1<f32>>,
}

impl RAdam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn compute_sma_threshold(&self) -> f32 {
        2.0 / (1.0 - self.beta2) - 1.0
    }
}

impl Optimizer for RAdam {
    fn update(&mut self, params: &mut [f32], gradients: &[f32], step: usize) {
        let param_id = params.as_ptr() as usize;
        
        if !self.momentum.contains_key(&param_id) {
            self.momentum.insert(param_id, Array1::zeros(params.len()));
            self.velocity.insert(param_id, Array1::zeros(params.len()));
        }
        
        let rho_inf = self.compute_sma_threshold();
        let rho_t = rho_inf - 2.0 * step as f32 * self.beta2.powi(step as i32) / (1.0 - self.beta2.powi(step as i32));
        
        let momentum = self.momentum.get_mut(&param_id).unwrap();
        let velocity = self.velocity.get_mut(&param_id).unwrap();
        
        for i in 0..params.len() {
            let mut grad = gradients[i];
            
            // Add weight decay
            if self.weight_decay != 0.0 {
                grad += self.weight_decay * params[i];
            }
            
            // Update biased first moment estimate
            momentum[i] = self.beta1 * momentum[i] + (1.0 - self.beta1) * grad;
            
            // Update biased second raw moment estimate
            velocity[i] = self.beta2 * velocity[i] + (1.0 - self.beta2) * grad * grad;
            
            // Compute bias-corrected first moment estimate
            let m_hat = momentum[i] / (1.0 - self.beta1.powi(step as i32));
            
            if rho_t > 5.0 {
                // Variance is tractable
                let v_hat = velocity[i] / (1.0 - self.beta2.powi(step as i32));
                let r = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt();
                params[i] -= self.learning_rate * r * m_hat / (v_hat.sqrt() + self.epsilon);
            } else {
                // Variance is intractable, use momentum only
                params[i] -= self.learning_rate * m_hat;
            }
        }
    }

    fn reset(&mut self) {
        self.momentum.clear();
        self.velocity.clear();
    }

    fn config(&self) -> OptimizerConfig {
        OptimizerConfig::RAdam {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        }
    }

    fn clone_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training)
#[derive(Debug, Clone)]
pub struct LAMB {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: HashMap<usize, Array1<f32>>,
    velocity: HashMap<usize, Array1<f32>>,
}

impl LAMB {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            weight_decay: 0.01,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn layer_norm(values: &[f32]) -> f32 {
        let sum_squares: f32 = values.iter().map(|&x| x * x).sum();
        (sum_squares / values.len() as f32).sqrt()
    }
}

impl Optimizer for LAMB {
    fn update(&mut self, params: &mut [f32], gradients: &[f32], step: usize) {
        let param_id = params.as_ptr() as usize;
        
        if !self.momentum.contains_key(&param_id) {
            self.momentum.insert(param_id, Array1::zeros(params.len()));
            self.velocity.insert(param_id, Array1::zeros(params.len()));
        }
        
        let momentum = self.momentum.get_mut(&param_id).unwrap();
        let velocity = self.velocity.get_mut(&param_id).unwrap();
        
        // Compute updates for each parameter
        let mut updates = vec![0.0; params.len()];
        
        for i in 0..params.len() {
            let grad = gradients[i] + self.weight_decay * params[i];
            
            // Update biased first moment estimate
            momentum[i] = self.beta1 * momentum[i] + (1.0 - self.beta1) * grad;
            
            // Update biased second raw moment estimate
            velocity[i] = self.beta2 * velocity[i] + (1.0 - self.beta2) * grad * grad;
            
            // Compute bias-corrected estimates
            let m_hat = momentum[i] / (1.0 - self.beta1.powi(step as i32));
            let v_hat = velocity[i] / (1.0 - self.beta2.powi(step as i32));
            
            updates[i] = m_hat / (v_hat.sqrt() + self.epsilon);
        }
        
        // Layer-wise adaptation
        let param_norm = Self::layer_norm(params);
        let update_norm = Self::layer_norm(&updates);
        
        let trust_ratio = if param_norm > 0.0 && update_norm > 0.0 {
            param_norm / update_norm
        } else {
            1.0
        };
        
        let adaptive_lr = self.learning_rate * trust_ratio;
        
        // Apply updates
        for i in 0..params.len() {
            params[i] -= adaptive_lr * updates[i];
        }
    }

    fn reset(&mut self) {
        self.momentum.clear();
        self.velocity.clear();
    }

    fn config(&self) -> OptimizerConfig {
        OptimizerConfig::LAMB {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        }
    }

    fn clone_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// Optimizer factory for creating optimizers from configuration
pub struct OptimizerFactory;

impl OptimizerFactory {
    pub fn create(config: &OptimizerConfig) -> Box<dyn Optimizer> {
        match config {
            OptimizerConfig::SGD { learning_rate, momentum, dampening, weight_decay, nesterov } => {
                Box::new(
                    SGD::new(*learning_rate)
                        .with_momentum(*momentum)
                        .with_dampening(*dampening)
                        .with_weight_decay(*weight_decay)
                        .with_nesterov(*nesterov)
                )
            },
            OptimizerConfig::Adam { learning_rate, beta1, beta2, epsilon, weight_decay, amsgrad } => {
                Box::new(
                    Adam::new(*learning_rate)
                        .with_betas(*beta1, *beta2)
                        .with_epsilon(*epsilon)
                        .with_weight_decay(*weight_decay)
                        .with_amsgrad(*amsgrad)
                )
            },
            OptimizerConfig::AdaGrad { learning_rate, epsilon, weight_decay, lr_decay } => {
                Box::new(
                    AdaGrad::new(*learning_rate)
                        .with_epsilon(*epsilon)
                        .with_weight_decay(*weight_decay)
                        .with_lr_decay(*lr_decay)
                )
            },
            OptimizerConfig::RMSprop { learning_rate, alpha, epsilon, weight_decay, momentum, centered } => {
                Box::new(
                    RMSprop::new(*learning_rate)
                        .with_alpha(*alpha)
                        .with_epsilon(*epsilon)
                        .with_weight_decay(*weight_decay)
                        .with_momentum(*momentum)
                        .with_centered(*centered)
                )
            },
            OptimizerConfig::AdaDelta { learning_rate, rho, epsilon, weight_decay } => {
                Box::new(
                    AdaDelta::new(*learning_rate)
                        .with_rho(*rho)
                        .with_epsilon(*epsilon)
                        .with_weight_decay(*weight_decay)
                )
            },
            OptimizerConfig::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                Box::new(
                    AdamW::new(*learning_rate)
                        .with_betas(*beta1, *beta2)
                        .with_epsilon(*epsilon)
                        .with_weight_decay(*weight_decay)
                )
            },
            OptimizerConfig::RAdam { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                Box::new(
                    RAdam::new(*learning_rate)
                        .with_betas(*beta1, *beta2)
                        .with_epsilon(*epsilon)
                        .with_weight_decay(*weight_decay)
                )
            },
            OptimizerConfig::LAMB { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                Box::new(
                    LAMB::new(*learning_rate)
                        .with_betas(*beta1, *beta2)
                        .with_epsilon(*epsilon)
                        .with_weight_decay(*weight_decay)
                )
            },
            _ => {
                // Default to Adam for unsupported optimizers
                Box::new(Adam::new(0.001))
            }
        }
    }

    pub fn create_default(optimizer_type: &str, learning_rate: f32) -> Box<dyn Optimizer> {
        match optimizer_type.to_lowercase().as_str() {
            "sgd" => Box::new(SGD::new(learning_rate).with_momentum(0.9)),
            "adam" => Box::new(Adam::new(learning_rate)),
            "adamw" => Box::new(AdamW::new(learning_rate)),
            "adagrad" => Box::new(AdaGrad::new(learning_rate)),
            "rmsprop" => Box::new(RMSprop::new(learning_rate)),
            "adadelta" => Box::new(AdaDelta::new(learning_rate)),
            "radam" => Box::new(RAdam::new(learning_rate)),
            "lamb" => Box::new(LAMB::new(learning_rate)),
            _ => Box::new(Adam::new(learning_rate)), // Default to Adam
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = SGD::new(0.01).with_momentum(0.9);
        let mut params = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.1, 0.2, 0.3];
        
        optimizer.update(&mut params, &gradients, 1);
        
        // Parameters should have changed
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = Adam::new(0.001);
        let mut params = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.1, 0.2, 0.3];
        
        optimizer.update(&mut params, &gradients, 1);
        
        // Parameters should have changed
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);
    }

    #[test]
    fn test_optimizer_factory() {
        let config = OptimizerConfig::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        };
        
        let optimizer = OptimizerFactory::create(&config);
        
        // Test that optimizer was created successfully
        match optimizer.config() {
            OptimizerConfig::Adam { learning_rate, .. } => {
                assert_eq!(learning_rate, 0.001);
            },
            _ => panic!("Expected Adam optimizer"),
        }
    }

    #[test]
    fn test_optimizer_reset() {
        let mut optimizer = Adam::new(0.001);
        let mut params = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.1, 0.2, 0.3];
        
        // First update
        optimizer.update(&mut params, &gradients, 1);
        
        // Reset optimizer state
        optimizer.reset();
        
        // State should be cleared
        assert!(optimizer.momentum.is_empty());
        assert!(optimizer.velocity.is_empty());
    }

    #[test]
    fn test_multiple_optimizer_types() {
        let optimizers = vec![
            ("SGD", OptimizerFactory::create_default("sgd", 0.01)),
            ("Adam", OptimizerFactory::create_default("adam", 0.001)),
            ("AdamW", OptimizerFactory::create_default("adamw", 0.001)),
            ("RMSprop", OptimizerFactory::create_default("rmsprop", 0.001)),
        ];
        
        for (name, mut optimizer) in optimizers {
            let mut params = vec![1.0, 2.0, 3.0];
            let gradients = vec![0.1, 0.2, 0.3];
            
            optimizer.update(&mut params, &gradients, 1);
            
            // All optimizers should modify parameters
            assert!(params[0] != 1.0, "Optimizer {} failed to update parameters", name);
        }
    }
}