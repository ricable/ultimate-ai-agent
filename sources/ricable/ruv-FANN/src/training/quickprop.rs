//! Quickprop training algorithm

use super::*;
use num_traits::Float;
use std::collections::HashMap;

/// Quickprop trainer
/// An advanced batch training algorithm that uses second-order information
pub struct Quickprop<T: Float + Send> {
    learning_rate: T,
    mu: T,
    decay: T,
    error_function: Box<dyn ErrorFunction<T>>,

    // State variables
    previous_weight_gradients: Vec<Vec<T>>,
    previous_bias_gradients: Vec<Vec<T>>,
    previous_weight_deltas: Vec<Vec<T>>,
    previous_bias_deltas: Vec<Vec<T>>,

    callback: Option<TrainingCallback<T>>,
}

impl<T: Float + Send> Quickprop<T> {
    pub fn new() -> Self {
        Self {
            learning_rate: T::from(0.7).unwrap(),
            mu: T::from(1.75).unwrap(),
            decay: T::from(-0.0001).unwrap(),
            error_function: Box::new(MseError),
            previous_weight_gradients: Vec::new(),
            previous_bias_gradients: Vec::new(),
            previous_weight_deltas: Vec::new(),
            previous_bias_deltas: Vec::new(),
            callback: None,
        }
    }

    pub fn with_parameters(mut self, learning_rate: T, mu: T, decay: T) -> Self {
        self.learning_rate = learning_rate;
        self.mu = mu;
        self.decay = decay;
        self
    }

    pub fn with_error_function(mut self, error_function: Box<dyn ErrorFunction<T>>) -> Self {
        self.error_function = error_function;
        self
    }

    fn initialize_state(&mut self, network: &Network<T>) {
        if self.previous_weight_gradients.is_empty() {
            // Initialize state for each layer
            self.previous_weight_gradients = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| {
                    let num_neurons = layer.neurons.len();
                    let num_connections = if layer.neurons.is_empty() {
                        0
                    } else {
                        layer.neurons[0].connections.len()
                    };
                    vec![T::zero(); num_neurons * num_connections]
                })
                .collect();

            self.previous_bias_gradients = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| vec![T::zero(); layer.neurons.len()])
                .collect();

            self.previous_weight_deltas = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| {
                    let num_neurons = layer.neurons.len();
                    let num_connections = if layer.neurons.is_empty() {
                        0
                    } else {
                        layer.neurons[0].connections.len()
                    };
                    vec![T::zero(); num_neurons * num_connections]
                })
                .collect();

            self.previous_bias_deltas = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| vec![T::zero(); layer.neurons.len()])
                .collect();
        }
    }

    fn calculate_quickprop_delta(
        &self,
        gradient: T,
        previous_gradient: T,
        previous_delta: T,
        weight: T,
    ) -> T {
        if previous_gradient == T::zero() {
            // First epoch or no previous gradient: use standard gradient descent
            return -self.learning_rate * gradient + self.decay * weight;
        }

        let gradient_diff = gradient - previous_gradient;

        if gradient_diff == T::zero() {
            // No change in gradient: use momentum-like update
            return -self.learning_rate * gradient + self.decay * weight;
        }

        // Quickprop formula: delta = (gradient / (previous_gradient - gradient)) * previous_delta
        let factor = gradient / gradient_diff;
        let mut delta = factor * previous_delta;

        // Limit the maximum step size
        let max_delta = self.mu * previous_delta.abs();
        if delta.abs() > max_delta {
            delta = if delta > T::zero() {
                max_delta
            } else {
                -max_delta
            };
        }

        // Add decay term
        delta + self.decay * weight
    }
}

impl<T: Float + Send> Default for Quickprop<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send> TrainingAlgorithm<T> for Quickprop<T> {
    fn train_epoch(
        &mut self,
        network: &mut Network<T>,
        data: &TrainingData<T>,
    ) -> Result<T, TrainingError> {
        self.initialize_state(network);

        let mut total_error = T::zero();

        // Calculate gradients over entire dataset
        for (input, desired_output) in data.inputs.iter().zip(data.outputs.iter()) {
            let output = network.run(input);
            total_error = total_error + self.error_function.calculate(&output, desired_output);

            // Calculate and accumulate gradients (placeholder)
            // In a full implementation, you would:
            // 1. Perform backpropagation to calculate gradients
            // 2. Update weights using Quickprop rules
        }

        // Placeholder for Quickprop weight updates
        // This would apply the Quickprop algorithm to update weights

        Ok(total_error / T::from(data.inputs.len()).unwrap())
    }

    fn calculate_error(&self, network: &Network<T>, data: &TrainingData<T>) -> T {
        let mut total_error = T::zero();
        let mut network_clone = network.clone();

        for (input, desired_output) in data.inputs.iter().zip(data.outputs.iter()) {
            let output = network_clone.run(input);
            total_error = total_error + self.error_function.calculate(&output, desired_output);
        }

        total_error / T::from(data.inputs.len()).unwrap()
    }

    fn count_bit_fails(
        &self,
        network: &Network<T>,
        data: &TrainingData<T>,
        bit_fail_limit: T,
    ) -> usize {
        let mut bit_fails = 0;
        let mut network_clone = network.clone();

        for (input, desired_output) in data.inputs.iter().zip(data.outputs.iter()) {
            let output = network_clone.run(input);

            for (&actual, &desired) in output.iter().zip(desired_output.iter()) {
                if (actual - desired).abs() > bit_fail_limit {
                    bit_fails += 1;
                }
            }
        }

        bit_fails
    }

    fn save_state(&self) -> TrainingState<T> {
        let mut state = HashMap::new();

        // Save Quickprop parameters
        state.insert("learning_rate".to_string(), vec![self.learning_rate]);
        state.insert("mu".to_string(), vec![self.mu]);
        state.insert("decay".to_string(), vec![self.decay]);

        // Save previous gradients and deltas (flattened)
        let mut all_weight_gradients = Vec::new();
        for layer_gradients in &self.previous_weight_gradients {
            all_weight_gradients.extend_from_slice(layer_gradients);
        }
        state.insert(
            "previous_weight_gradients".to_string(),
            all_weight_gradients,
        );

        let mut all_bias_gradients = Vec::new();
        for layer_gradients in &self.previous_bias_gradients {
            all_bias_gradients.extend_from_slice(layer_gradients);
        }
        state.insert("previous_bias_gradients".to_string(), all_bias_gradients);

        let mut all_weight_deltas = Vec::new();
        for layer_deltas in &self.previous_weight_deltas {
            all_weight_deltas.extend_from_slice(layer_deltas);
        }
        state.insert("previous_weight_deltas".to_string(), all_weight_deltas);

        let mut all_bias_deltas = Vec::new();
        for layer_deltas in &self.previous_bias_deltas {
            all_bias_deltas.extend_from_slice(layer_deltas);
        }
        state.insert("previous_bias_deltas".to_string(), all_bias_deltas);

        TrainingState {
            epoch: 0,
            best_error: T::from(f32::MAX).unwrap(),
            algorithm_specific: state,
        }
    }

    fn restore_state(&mut self, state: TrainingState<T>) {
        // Restore Quickprop parameters
        if let Some(val) = state.algorithm_specific.get("learning_rate") {
            if !val.is_empty() {
                self.learning_rate = val[0];
            }
        }
        if let Some(val) = state.algorithm_specific.get("mu") {
            if !val.is_empty() {
                self.mu = val[0];
            }
        }
        if let Some(val) = state.algorithm_specific.get("decay") {
            if !val.is_empty() {
                self.decay = val[0];
            }
        }

        // Note: Previous gradients and deltas would need network structure info to properly restore
        // This is a simplified version - in production, you'd need to store layer sizes too
    }

    fn set_callback(&mut self, callback: TrainingCallback<T>) {
        self.callback = Some(callback);
    }

    fn call_callback(
        &mut self,
        epoch: usize,
        network: &Network<T>,
        data: &TrainingData<T>,
    ) -> bool {
        let error = self.calculate_error(network, data);
        if let Some(ref mut callback) = self.callback {
            callback(epoch, error)
        } else {
            true
        }
    }
}
