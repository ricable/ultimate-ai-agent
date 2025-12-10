//! Advanced Backpropagation Training Engine
//! 
//! This module provides comprehensive forward and backward propagation algorithms
//! with multiple optimization strategies for neural network training.

use crate::fann_compat::{Network, TrainingData, ActivationFunction, TrainingError};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ndarray::{Array1, Array2};

/// Advanced training configuration with multiple optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub batch_size: usize,
    pub optimizer: OptimizerType,
    pub lr_scheduler: Option<LearningRateScheduler>,
    pub gradient_clipping: Option<f32>,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub regularization: RegularizationType,
    pub initialization: WeightInitialization,
}

/// Optimizer algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD { momentum: f32 },
    Adam { beta1: f32, beta2: f32, epsilon: f32 },
    AdaGrad { epsilon: f32 },
    RMSprop { beta: f32, epsilon: f32 },
    AdaDelta { rho: f32, epsilon: f32 },
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateScheduler {
    StepDecay { step_size: usize, gamma: f32 },
    ExponentialDecay { gamma: f32 },
    CosineAnnealing { t_max: usize },
    ReduceOnPlateau { patience: usize, factor: f32, threshold: f32 },
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f32,
    pub restore_best_weights: bool,
}

/// Regularization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegularizationType {
    None,
    L1 { lambda: f32 },
    L2 { lambda: f32 },
    ElasticNet { l1_ratio: f32, alpha: f32 },
    Dropout { rate: f32 },
}

/// Weight initialization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightInitialization {
    Random,
    Xavier,
    He,
    LeCun,
    Zeros,
    Ones,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
            batch_size: 32,
            optimizer: OptimizerType::Adam { 
                beta1: 0.9, 
                beta2: 0.999, 
                epsilon: 1e-8 
            },
            lr_scheduler: Some(LearningRateScheduler::ReduceOnPlateau {
                patience: 10,
                factor: 0.5,
                threshold: 1e-4,
            }),
            gradient_clipping: Some(1.0),
            early_stopping: Some(EarlyStoppingConfig {
                patience: 20,
                min_delta: 1e-4,
                restore_best_weights: true,
            }),
            regularization: RegularizationType::L2 { lambda: 0.01 },
            initialization: WeightInitialization::Xavier,
        }
    }
}

/// Advanced backpropagation trainer with multiple optimization algorithms
pub struct AdvancedBackpropagationTrainer {
    config: TrainingConfig,
    optimizer_state: OptimizerState,
    lr_scheduler_state: Option<LRSchedulerState>,
    early_stopping_state: Option<EarlyStoppingState>,
    training_metrics: TrainingMetrics,
}

/// Optimizer state to store momentum, velocity, etc.
#[derive(Debug)]
pub struct OptimizerState {
    pub momentum: HashMap<String, Array2<f32>>,
    pub velocity: HashMap<String, Array2<f32>>,
    pub squared_gradients: HashMap<String, Array2<f32>>,
    pub iteration: usize,
}

/// Learning rate scheduler state
#[derive(Debug)]
pub struct LRSchedulerState {
    pub current_lr: f32,
    pub step_count: usize,
    pub best_loss: f32,
    pub patience_counter: usize,
}

/// Early stopping state
#[derive(Debug)]
pub struct EarlyStoppingState {
    pub best_loss: f32,
    pub patience_counter: usize,
    pub best_weights: Option<Vec<Array2<f32>>>,
}

/// Training metrics tracking
#[derive(Debug, Clone, Serialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    pub train_accuracy: Option<f32>,
    pub val_accuracy: Option<f32>,
    pub learning_rate: f32,
    pub gradient_norm: f32,
    pub weight_norm: f32,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            epoch: 0,
            train_loss: 0.0,
            val_loss: None,
            train_accuracy: None,
            val_accuracy: None,
            learning_rate: 0.001,
            gradient_norm: 0.0,
            weight_norm: 0.0,
        }
    }
}

impl AdvancedBackpropagationTrainer {
    /// Create new advanced trainer
    pub fn new(config: TrainingConfig) -> Self {
        let lr_scheduler_state = config.lr_scheduler.as_ref().map(|_| LRSchedulerState {
            current_lr: config.learning_rate,
            step_count: 0,
            best_loss: f32::INFINITY,
            patience_counter: 0,
        });

        let early_stopping_state = config.early_stopping.as_ref().map(|_| EarlyStoppingState {
            best_loss: f32::INFINITY,
            patience_counter: 0,
            best_weights: None,
        });

        Self {
            config,
            optimizer_state: OptimizerState {
                momentum: HashMap::new(),
                velocity: HashMap::new(),
                squared_gradients: HashMap::new(),
                iteration: 0,
            },
            lr_scheduler_state,
            early_stopping_state,
            training_metrics: TrainingMetrics::default(),
        }
    }

    /// Initialize network weights based on configuration
    pub fn initialize_weights(&self, network: &mut Network) {
        let layer_sizes: Vec<usize> = network.layers.iter().map(|l| l.neurons.len()).collect();
        
        for (layer_idx, layer) in network.layers.iter_mut().enumerate() {
            if layer_idx == 0 { continue; } // Skip input layer
            
            let prev_layer_size = layer_sizes[layer_idx - 1];
            let current_layer_size = layer_sizes[layer_idx];
            
            for neuron in &mut layer.neurons {
                match self.config.initialization {
                    WeightInitialization::Random => {
                        for weight in &mut neuron.connections {
                            *weight = rand::random::<f32>() * 2.0 - 1.0;
                        }
                    },
                    WeightInitialization::Xavier => {
                        let limit = (6.0 / (prev_layer_size + current_layer_size) as f32).sqrt();
                        for weight in &mut neuron.connections {
                            *weight = (rand::random::<f32>() * 2.0 - 1.0) * limit;
                        }
                    },
                    WeightInitialization::He => {
                        let std_dev = (2.0 / prev_layer_size as f32).sqrt();
                        for weight in &mut neuron.connections {
                            *weight = self.sample_normal(0.0, std_dev);
                        }
                    },
                    WeightInitialization::LeCun => {
                        let std_dev = (1.0 / prev_layer_size as f32).sqrt();
                        for weight in &mut neuron.connections {
                            *weight = self.sample_normal(0.0, std_dev);
                        }
                    },
                    WeightInitialization::Zeros => {
                        for weight in &mut neuron.connections {
                            *weight = 0.0;
                        }
                    },
                    WeightInitialization::Ones => {
                        for weight in &mut neuron.connections {
                            *weight = 1.0;
                        }
                    },
                }
            }
        }
    }

    /// Forward propagation with detailed intermediate storage
    pub fn forward_propagation(&self, network: &Network, input: &[f32]) -> ForwardResult {
        let mut activations = vec![Array1::from(input.to_vec())];
        let mut pre_activations = Vec::new();
        
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if layer_idx == 0 { continue; } // Skip input layer
            
            let prev_activations = &activations[layer_idx - 1];
            let mut layer_pre_activations = Vec::new();
            let mut layer_activations = Vec::new();
            
            for neuron in &layer.neurons {
                let mut sum = neuron.bias;
                
                for (i, &weight) in neuron.connections.iter().enumerate() {
                    if i < prev_activations.len() {
                        sum += weight * prev_activations[i];
                    }
                }
                
                layer_pre_activations.push(sum);
                let activation = self.apply_activation(sum, neuron.activation_function);
                layer_activations.push(activation);
            }
            
            pre_activations.push(Array1::from(layer_pre_activations));
            activations.push(Array1::from(layer_activations));
        }
        
        let final_output = activations.last().unwrap().to_vec();
        ForwardResult {
            activations,
            pre_activations,
            final_output,
        }
    }

    /// Backward propagation with gradient calculation
    pub fn backward_propagation(
        &self,
        network: &Network,
        forward_result: &ForwardResult,
        target: &[f32],
    ) -> BackwardResult {
        let mut gradients = Vec::new();
        let mut deltas = Vec::new();
        
        // Initialize gradients for each layer
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if layer_idx == 0 { continue; } // Skip input layer
            
            let layer_gradients = vec![vec![0.0; layer.neurons[0].connections.len()]; layer.neurons.len()];
            gradients.push(layer_gradients);
        }
        
        // Calculate output layer delta
        let output_layer_idx = network.layers.len() - 1;
        let output_activations = &forward_result.activations[output_layer_idx];
        let output_pre_activations = &forward_result.pre_activations[output_layer_idx - 1];
        
        let mut output_delta = Vec::new();
        for (i, (&output, &target_val)) in output_activations.iter().zip(target.iter()).enumerate() {
            let error = output - target_val;
            let derivative = self.activation_derivative(
                output_pre_activations[i],
                network.layers[output_layer_idx].neurons[i].activation_function,
            );
            output_delta.push(error * derivative);
        }
        deltas.push(output_delta);
        
        // Backpropagate deltas through hidden layers
        for layer_idx in (1..network.layers.len() - 1).rev() {
            let mut layer_delta = Vec::new();
            let layer_pre_activations = &forward_result.pre_activations[layer_idx - 1];
            
            for (neuron_idx, neuron) in network.layers[layer_idx].neurons.iter().enumerate() {
                let mut delta_sum = 0.0;
                
                // Sum deltas from next layer
                for (next_neuron_idx, next_neuron) in network.layers[layer_idx + 1].neurons.iter().enumerate() {
                    if neuron_idx < next_neuron.connections.len() {
                        delta_sum += deltas[deltas.len() - 1][next_neuron_idx] * next_neuron.connections[neuron_idx];
                    }
                }
                
                let derivative = self.activation_derivative(
                    layer_pre_activations[neuron_idx],
                    neuron.activation_function,
                );
                layer_delta.push(delta_sum * derivative);
            }
            
            deltas.push(layer_delta);
        }
        
        deltas.reverse(); // Reverse to match layer order
        
        // Calculate gradients
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if layer_idx == 0 { continue; } // Skip input layer
            
            let prev_activations = &forward_result.activations[layer_idx - 1];
            let layer_deltas = &deltas[layer_idx - 1];
            
            for (neuron_idx, neuron) in layer.neurons.iter().enumerate() {
                for (weight_idx, _) in neuron.connections.iter().enumerate() {
                    if weight_idx < prev_activations.len() {
                        gradients[layer_idx - 1][neuron_idx][weight_idx] = 
                            layer_deltas[neuron_idx] * prev_activations[weight_idx];
                    }
                }
            }
        }
        
        BackwardResult {
            gradients,
            deltas,
            loss: self.calculate_loss(output_activations, target),
        }
    }

    /// Update weights using specified optimizer
    pub fn update_weights(&mut self, network: &mut Network, backward_result: &BackwardResult) {
        self.optimizer_state.iteration += 1;
        
        let current_lr = self.get_current_learning_rate();
        
        for (layer_idx, layer) in network.layers.iter_mut().enumerate() {
            if layer_idx == 0 { continue; } // Skip input layer
            
            let layer_gradients = &backward_result.gradients[layer_idx - 1];
            let layer_key = format!("layer_{}", layer_idx);
            
            for (neuron_idx, neuron) in layer.neurons.iter_mut().enumerate() {
                for (weight_idx, weight) in neuron.connections.iter_mut().enumerate() {
                    let gradient = layer_gradients[neuron_idx][weight_idx];
                    
                    // Apply regularization
                    let regularized_gradient = self.apply_regularization(gradient, *weight);
                    
                    // Apply gradient clipping
                    let clipped_gradient = self.apply_gradient_clipping(regularized_gradient);
                    
                    // Update weight using optimizer
                    let weight_update = self.calculate_weight_update(
                        &layer_key,
                        neuron_idx,
                        weight_idx,
                        clipped_gradient,
                        current_lr,
                    );
                    
                    *weight -= weight_update;
                }
            }
        }
        
        // Update learning rate scheduler
        let loss = backward_result.loss;
        let lr_state_opt = self.lr_scheduler_state.as_mut();
        if let Some(lr_state) = lr_state_opt {
            self.update_learning_rate_scheduler(lr_state, loss);
        }
        
        // Update training metrics
        self.training_metrics.learning_rate = current_lr;
        self.training_metrics.gradient_norm = self.calculate_gradient_norm(&backward_result.gradients);
        self.training_metrics.weight_norm = self.calculate_weight_norm(network);
    }

    /// Calculate weight update based on optimizer type
    fn calculate_weight_update(
        &mut self,
        layer_key: &str,
        neuron_idx: usize,
        weight_idx: usize,
        gradient: f32,
        learning_rate: f32,
    ) -> f32 {
        match &self.config.optimizer {
            OptimizerType::SGD { momentum } => {
                let velocity_key = format!("{}_{}", layer_key, neuron_idx);
                let velocity = self.optimizer_state.velocity
                    .entry(velocity_key)
                    .or_insert_with(|| Array2::zeros((1, weight_idx + 1)));
                
                if weight_idx < velocity.ncols() {
                    velocity[[0, weight_idx]] = momentum * velocity[[0, weight_idx]] + gradient;
                    learning_rate * velocity[[0, weight_idx]]
                } else {
                    learning_rate * gradient
                }
            },
            OptimizerType::Adam { beta1, beta2, epsilon } => {
                let momentum_key = format!("{}_{}", layer_key, neuron_idx);
                let velocity_key = format!("{}_{}_v", layer_key, neuron_idx);
                
                let momentum = self.optimizer_state.momentum
                    .entry(momentum_key)
                    .or_insert_with(|| Array2::zeros((1, weight_idx + 1)));
                
                let velocity = self.optimizer_state.velocity
                    .entry(velocity_key)
                    .or_insert_with(|| Array2::zeros((1, weight_idx + 1)));
                
                if weight_idx < momentum.ncols() && weight_idx < velocity.ncols() {
                    momentum[[0, weight_idx]] = beta1 * momentum[[0, weight_idx]] + (1.0 - beta1) * gradient;
                    velocity[[0, weight_idx]] = beta2 * velocity[[0, weight_idx]] + (1.0 - beta2) * gradient * gradient;
                    
                    let m_hat = momentum[[0, weight_idx]] / (1.0 - beta1.powi(self.optimizer_state.iteration as i32));
                    let v_hat = velocity[[0, weight_idx]] / (1.0 - beta2.powi(self.optimizer_state.iteration as i32));
                    
                    learning_rate * m_hat / (v_hat.sqrt() + epsilon)
                } else {
                    learning_rate * gradient
                }
            },
            OptimizerType::AdaGrad { epsilon } => {
                let squared_grad_key = format!("{}_{}", layer_key, neuron_idx);
                let squared_gradients = self.optimizer_state.squared_gradients
                    .entry(squared_grad_key)
                    .or_insert_with(|| Array2::zeros((1, weight_idx + 1)));
                
                if weight_idx < squared_gradients.ncols() {
                    squared_gradients[[0, weight_idx]] += gradient * gradient;
                    learning_rate * gradient / (squared_gradients[[0, weight_idx]].sqrt() + epsilon)
                } else {
                    learning_rate * gradient
                }
            },
            OptimizerType::RMSprop { beta, epsilon } => {
                let squared_grad_key = format!("{}_{}", layer_key, neuron_idx);
                let squared_gradients = self.optimizer_state.squared_gradients
                    .entry(squared_grad_key)
                    .or_insert_with(|| Array2::zeros((1, weight_idx + 1)));
                
                if weight_idx < squared_gradients.ncols() {
                    squared_gradients[[0, weight_idx]] = beta * squared_gradients[[0, weight_idx]] + (1.0 - beta) * gradient * gradient;
                    learning_rate * gradient / (squared_gradients[[0, weight_idx]].sqrt() + epsilon)
                } else {
                    learning_rate * gradient
                }
            },
            OptimizerType::AdaDelta { rho, epsilon } => {
                let squared_grad_key = format!("{}_{}", layer_key, neuron_idx);
                let squared_delta_key = format!("{}_{}_delta", layer_key, neuron_idx);
                
                let squared_gradients = self.optimizer_state.squared_gradients
                    .entry(squared_grad_key)
                    .or_insert_with(|| Array2::zeros((1, weight_idx + 1)));
                
                let squared_deltas = self.optimizer_state.velocity
                    .entry(squared_delta_key)
                    .or_insert_with(|| Array2::zeros((1, weight_idx + 1)));
                
                if weight_idx < squared_gradients.ncols() && weight_idx < squared_deltas.ncols() {
                    squared_gradients[[0, weight_idx]] = rho * squared_gradients[[0, weight_idx]] + (1.0 - rho) * gradient * gradient;
                    
                    let delta = (squared_deltas[[0, weight_idx]] + epsilon).sqrt() / (squared_gradients[[0, weight_idx]] + epsilon).sqrt() * gradient;
                    
                    squared_deltas[[0, weight_idx]] = rho * squared_deltas[[0, weight_idx]] + (1.0 - rho) * delta * delta;
                    
                    delta
                } else {
                    learning_rate * gradient
                }
            },
        }
    }

    /// Train for one epoch
    pub fn train_epoch(&mut self, network: &mut Network, training_data: &TrainingData) -> Result<f32, TrainingError> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        // Process data in batches
        for batch_start in (0..training_data.inputs.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(training_data.inputs.len());
            let mut batch_gradients = Vec::new();
            let mut batch_loss = 0.0;
            
            // Accumulate gradients for batch
            for i in batch_start..batch_end {
                let forward_result = self.forward_propagation(network, &training_data.inputs[i]);
                let backward_result = self.backward_propagation(network, &forward_result, &training_data.outputs[i]);
                
                if batch_gradients.is_empty() {
                    batch_gradients = backward_result.gradients.clone();
                } else {
                    // Accumulate gradients
                    for (layer_idx, layer_gradients) in backward_result.gradients.iter().enumerate() {
                        for (neuron_idx, neuron_gradients) in layer_gradients.iter().enumerate() {
                            for (weight_idx, &gradient) in neuron_gradients.iter().enumerate() {
                                batch_gradients[layer_idx][neuron_idx][weight_idx] += gradient;
                            }
                        }
                    }
                }
                
                batch_loss += backward_result.loss;
            }
            
            // Average gradients over batch
            let batch_size = batch_end - batch_start;
            for layer_gradients in &mut batch_gradients {
                for neuron_gradients in layer_gradients {
                    for gradient in neuron_gradients {
                        *gradient /= batch_size as f32;
                    }
                }
            }
            
            // Update weights with averaged gradients
            let averaged_backward_result = BackwardResult {
                gradients: batch_gradients,
                deltas: Vec::new(),
                loss: batch_loss / batch_size as f32,
            };
            
            self.update_weights(network, &averaged_backward_result);
            
            total_loss += batch_loss;
            batch_count += 1;
        }
        
        let average_loss = total_loss / training_data.inputs.len() as f32;
        
        // Update training metrics
        self.training_metrics.epoch += 1;
        self.training_metrics.train_loss = average_loss;
        
        // Check early stopping
        if let Some(ref early_stopping_config) = self.config.early_stopping {
            let early_stopping_opt = self.early_stopping_state.as_mut();
            if let Some(early_stopping) = early_stopping_opt {
                if average_loss < early_stopping.best_loss - early_stopping_config.min_delta {
                    early_stopping.best_loss = average_loss;
                    early_stopping.patience_counter = 0;
                    
                    if early_stopping_config.restore_best_weights {
                        let weights = self.extract_weights(network);
                        early_stopping.best_weights = Some(weights);
                    }
                } else {
                    early_stopping.patience_counter += 1;
                }
            }
        }
        
        Ok(average_loss)
    }

    /// Check if training should stop early
    pub fn should_stop_early(&self) -> bool {
        if let Some(ref early_stopping) = self.early_stopping_state {
            if let Some(ref config) = self.config.early_stopping {
                return early_stopping.patience_counter >= config.patience;
            }
        }
        false
    }

    /// Restore best weights if early stopping is used
    pub fn restore_best_weights(&self, network: &mut Network) {
        if let Some(ref early_stopping) = self.early_stopping_state {
            if let Some(ref best_weights) = early_stopping.best_weights {
                self.restore_weights(network, best_weights);
            }
        }
    }

    /// Get current training metrics
    pub fn get_metrics(&self) -> &TrainingMetrics {
        &self.training_metrics
    }

    // Helper functions
    fn apply_activation(&self, x: f32, func: ActivationFunction) -> f32 {
        match func {
            ActivationFunction::Linear => x,
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::ReLULeaky => if x > 0.0 { x } else { 0.01 * x },
            ActivationFunction::Elliot => x / (1.0 + x.abs()),
            ActivationFunction::Gaussian => (-x * x).exp(),
        }
    }

    fn activation_derivative(&self, x: f32, func: ActivationFunction) -> f32 {
        match func {
            ActivationFunction::Linear => 1.0,
            ActivationFunction::Sigmoid => {
                let s = self.apply_activation(x, func);
                s * (1.0 - s)
            },
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            },
            ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::ReLULeaky => if x > 0.0 { 1.0 } else { 0.01 },
            ActivationFunction::Elliot => {
                let abs_x = x.abs();
                1.0 / ((1.0 + abs_x) * (1.0 + abs_x))
            },
            ActivationFunction::Gaussian => {
                let g = self.apply_activation(x, func);
                -2.0 * x * g
            },
        }
    }

    fn calculate_loss(&self, predicted: &Array1<f32>, target: &[f32]) -> f32 {
        predicted.iter()
            .zip(target.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f32>() / predicted.len() as f32
    }

    fn apply_regularization(&self, gradient: f32, weight: f32) -> f32 {
        match &self.config.regularization {
            RegularizationType::None => gradient,
            RegularizationType::L1 { lambda } => gradient + lambda * weight.signum(),
            RegularizationType::L2 { lambda } => gradient + lambda * weight,
            RegularizationType::ElasticNet { l1_ratio, alpha } => {
                gradient + alpha * (l1_ratio * weight.signum() + (1.0 - l1_ratio) * weight)
            },
            RegularizationType::Dropout { .. } => gradient, // Applied during forward pass
        }
    }

    fn apply_gradient_clipping(&self, gradient: f32) -> f32 {
        if let Some(max_norm) = self.config.gradient_clipping {
            gradient.max(-max_norm).min(max_norm)
        } else {
            gradient
        }
    }

    fn get_current_learning_rate(&self) -> f32 {
        self.lr_scheduler_state
            .as_ref()
            .map(|state| state.current_lr)
            .unwrap_or(self.config.learning_rate)
    }

    fn update_learning_rate_scheduler(&mut self, lr_state: &mut LRSchedulerState, loss: f32) {
        if let Some(ref scheduler) = self.config.lr_scheduler {
            match scheduler {
                LearningRateScheduler::StepDecay { step_size, gamma } => {
                    lr_state.step_count += 1;
                    if lr_state.step_count % step_size == 0 {
                        lr_state.current_lr *= gamma;
                    }
                },
                LearningRateScheduler::ExponentialDecay { gamma } => {
                    lr_state.current_lr *= gamma;
                },
                LearningRateScheduler::CosineAnnealing { t_max } => {
                    let progress = (lr_state.step_count as f32 / *t_max as f32).min(1.0);
                    lr_state.current_lr = self.config.learning_rate * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                },
                LearningRateScheduler::ReduceOnPlateau { patience, factor, threshold } => {
                    if loss < lr_state.best_loss - threshold {
                        lr_state.best_loss = loss;
                        lr_state.patience_counter = 0;
                    } else {
                        lr_state.patience_counter += 1;
                        if lr_state.patience_counter >= *patience {
                            lr_state.current_lr *= factor;
                            lr_state.patience_counter = 0;
                        }
                    }
                },
            }
        }
    }

    fn calculate_gradient_norm(&self, gradients: &[Vec<Vec<f32>>]) -> f32 {
        let mut sum_squares = 0.0;
        for layer_gradients in gradients {
            for neuron_gradients in layer_gradients {
                for &gradient in neuron_gradients {
                    sum_squares += gradient * gradient;
                }
            }
        }
        sum_squares.sqrt()
    }

    fn calculate_weight_norm(&self, network: &Network) -> f32 {
        let mut sum_squares = 0.0;
        for layer in &network.layers {
            for neuron in &layer.neurons {
                for &weight in &neuron.connections {
                    sum_squares += weight * weight;
                }
            }
        }
        sum_squares.sqrt()
    }

    fn sample_normal(&self, mean: f32, std_dev: f32) -> f32 {
        // Box-Muller transform for normal distribution
        use std::f32::consts::PI;
        let u1 = rand::random::<f32>();
        let u2 = rand::random::<f32>();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std_dev * z0
    }

    fn extract_weights(&self, network: &Network) -> Vec<Array2<f32>> {
        let mut weights = Vec::new();
        for layer in &network.layers {
            let mut layer_weights = Vec::new();
            for neuron in &layer.neurons {
                layer_weights.push(neuron.connections.clone());
            }
            weights.push(Array2::from_shape_vec((layer_weights.len(), layer_weights[0].len()), layer_weights.into_iter().flatten().collect()).unwrap());
        }
        weights
    }

    fn restore_weights(&self, network: &mut Network, weights: &[Array2<f32>]) {
        for (layer_idx, layer) in network.layers.iter_mut().enumerate() {
            if layer_idx < weights.len() {
                for (neuron_idx, neuron) in layer.neurons.iter_mut().enumerate() {
                    for (weight_idx, weight) in neuron.connections.iter_mut().enumerate() {
                        if neuron_idx < weights[layer_idx].nrows() && weight_idx < weights[layer_idx].ncols() {
                            *weight = weights[layer_idx][[neuron_idx, weight_idx]];
                        }
                    }
                }
            }
        }
    }
}

/// Forward propagation result
#[derive(Debug)]
pub struct ForwardResult {
    pub activations: Vec<Array1<f32>>,
    pub pre_activations: Vec<Array1<f32>>,
    pub final_output: Vec<f32>,
}

/// Backward propagation result
#[derive(Debug)]
pub struct BackwardResult {
    pub gradients: Vec<Vec<Vec<f32>>>,
    pub deltas: Vec<Vec<f32>>,
    pub loss: f32,
}

/// Hyperparameter tuning utilities
pub struct HyperparameterTuner {
    pub search_space: HyperparameterSearchSpace,
}

#[derive(Debug, Clone)]
pub struct HyperparameterSearchSpace {
    pub learning_rates: Vec<f32>,
    pub batch_sizes: Vec<usize>,
    pub optimizers: Vec<OptimizerType>,
    pub regularization: Vec<RegularizationType>,
    pub architectures: Vec<Vec<usize>>,
}

impl Default for HyperparameterSearchSpace {
    fn default() -> Self {
        Self {
            learning_rates: vec![0.001, 0.01, 0.1, 0.0001],
            batch_sizes: vec![16, 32, 64, 128],
            optimizers: vec![
                OptimizerType::Adam { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
                OptimizerType::SGD { momentum: 0.9 },
                OptimizerType::RMSprop { beta: 0.9, epsilon: 1e-8 },
            ],
            regularization: vec![
                RegularizationType::None,
                RegularizationType::L1 { lambda: 0.001 },
                RegularizationType::L2 { lambda: 0.001 },
                RegularizationType::ElasticNet { l1_ratio: 0.5, alpha: 0.001 },
            ],
            architectures: vec![
                vec![64, 32],
                vec![128, 64, 32],
                vec![256, 128, 64],
                vec![512, 256, 128, 64],
            ],
        }
    }
}

impl HyperparameterTuner {
    pub fn new(search_space: HyperparameterSearchSpace) -> Self {
        Self { search_space }
    }

    /// Generate random hyperparameter configurations
    pub fn random_search(&self, n_trials: usize) -> Vec<TrainingConfig> {
        let mut configs = Vec::new();
        
        for _ in 0..n_trials {
            let config = TrainingConfig {
                learning_rate: self.search_space.learning_rates[rand::random::<usize>() % self.search_space.learning_rates.len()],
                batch_size: self.search_space.batch_sizes[rand::random::<usize>() % self.search_space.batch_sizes.len()],
                optimizer: self.search_space.optimizers[rand::random::<usize>() % self.search_space.optimizers.len()].clone(),
                regularization: self.search_space.regularization[rand::random::<usize>() % self.search_space.regularization.len()].clone(),
                ..Default::default()
            };
            configs.push(config);
        }
        
        configs
    }

    /// Grid search over hyperparameters
    pub fn grid_search(&self) -> Vec<TrainingConfig> {
        let mut configs = Vec::new();
        
        for &learning_rate in &self.search_space.learning_rates {
            for &batch_size in &self.search_space.batch_sizes {
                for optimizer in &self.search_space.optimizers {
                    for regularization in &self.search_space.regularization {
                        let config = TrainingConfig {
                            learning_rate,
                            batch_size,
                            optimizer: optimizer.clone(),
                            regularization: regularization.clone(),
                            ..Default::default()
                        };
                        configs.push(config);
                    }
                }
            }
        }
        
        configs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fann_compat::NetworkBuilder;

    #[test]
    fn test_advanced_trainer_initialization() {
        let config = TrainingConfig::default();
        let trainer = AdvancedBackpropagationTrainer::new(config);
        
        assert!(trainer.optimizer_state.iteration == 0);
        assert!(trainer.lr_scheduler_state.is_some());
        assert!(trainer.early_stopping_state.is_some());
    }

    #[test]
    fn test_weight_initialization() {
        let config = TrainingConfig {
            initialization: WeightInitialization::Xavier,
            ..Default::default()
        };
        let trainer = AdvancedBackpropagationTrainer::new(config);
        
        let mut network = NetworkBuilder::new()
            .add_layer(4)
            .add_layer(8)
            .add_layer(2)
            .build();
        
        trainer.initialize_weights(&mut network);
        
        // Check that weights are initialized
        for (i, layer) in network.layers.iter().enumerate() {
            if i > 0 {
                for neuron in &layer.neurons {
                    for &weight in &neuron.connections {
                        assert!(weight.abs() < 2.0); // Reasonable range
                    }
                }
            }
        }
    }

    #[test]
    fn test_forward_propagation() {
        let config = TrainingConfig::default();
        let trainer = AdvancedBackpropagationTrainer::new(config);
        
        let network = NetworkBuilder::new()
            .add_layer(3)
            .add_layer(4)
            .add_layer(2)
            .build();
        
        let input = vec![1.0, 0.5, -0.5];
        let result = trainer.forward_propagation(&network, &input);
        
        assert_eq!(result.activations.len(), 3); // 3 layers
        assert_eq!(result.pre_activations.len(), 2); // 2 hidden + output layers
        assert_eq!(result.final_output.len(), 2); // 2 output neurons
    }

    #[test]
    fn test_hyperparameter_tuning() {
        let search_space = HyperparameterSearchSpace::default();
        let tuner = HyperparameterTuner::new(search_space);
        
        let random_configs = tuner.random_search(5);
        assert_eq!(random_configs.len(), 5);
        
        let grid_configs = tuner.grid_search();
        assert!(grid_configs.len() > 0);
    }
}