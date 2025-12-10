//! Quickprop training algorithm

#![allow(clippy::needless_range_loop)]

use super::*;
use num_traits::Float;
use std::collections::HashMap;

/// Quickprop trainer
/// An advanced batch training algorithm that uses second-order information
pub struct Quickprop<T: Float + Send + Default> {
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

impl<T: Float + Send + Default> Quickprop<T> {
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

impl<T: Float + Send + Default> Default for Quickprop<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Default> TrainingAlgorithm<T> for Quickprop<T> {
    fn train_epoch(
        &mut self,
        network: &mut Network<T>,
        data: &TrainingData<T>,
    ) -> Result<T, TrainingError> {
        use super::helpers::*;

        self.initialize_state(network);

        let mut total_error = T::zero();

        // Convert network to simplified form for easier manipulation
        let simple_network = network_to_simple(network);

        // Initialize gradient accumulators
        let mut accumulated_weight_gradients = simple_network
            .weights
            .iter()
            .map(|w| vec![T::zero(); w.len()])
            .collect::<Vec<_>>();
        let mut accumulated_bias_gradients = simple_network
            .biases
            .iter()
            .map(|b| vec![T::zero(); b.len()])
            .collect::<Vec<_>>();

        // Calculate gradients over entire dataset
        for (input, desired_output) in data.inputs.iter().zip(data.outputs.iter()) {
            // Forward propagation to get all layer activations
            let activations = forward_propagate(&simple_network, input);

            // Get output from last layer
            let output = &activations[activations.len() - 1];

            // Calculate error
            total_error = total_error + self.error_function.calculate(output, desired_output);

            // Calculate gradients using backpropagation
            let (weight_gradients, bias_gradients) = calculate_gradients(
                &simple_network,
                &activations,
                desired_output,
                self.error_function.as_ref(),
            );

            // Accumulate gradients
            for layer_idx in 0..weight_gradients.len() {
                for i in 0..weight_gradients[layer_idx].len() {
                    accumulated_weight_gradients[layer_idx][i] =
                        accumulated_weight_gradients[layer_idx][i] + weight_gradients[layer_idx][i];
                }
                for i in 0..bias_gradients[layer_idx].len() {
                    accumulated_bias_gradients[layer_idx][i] =
                        accumulated_bias_gradients[layer_idx][i] + bias_gradients[layer_idx][i];
                }
            }
        }

        // Average gradients by batch size
        let batch_size = T::from(data.inputs.len()).unwrap();
        for layer_idx in 0..accumulated_weight_gradients.len() {
            for i in 0..accumulated_weight_gradients[layer_idx].len() {
                accumulated_weight_gradients[layer_idx][i] =
                    accumulated_weight_gradients[layer_idx][i] / batch_size;
            }
            for i in 0..accumulated_bias_gradients[layer_idx].len() {
                accumulated_bias_gradients[layer_idx][i] =
                    accumulated_bias_gradients[layer_idx][i] / batch_size;
            }
        }

        // Apply Quickprop updates
        let mut weight_updates = Vec::new();
        let mut bias_updates = Vec::new();

        // Update weights using Quickprop algorithm
        for layer_idx in 0..accumulated_weight_gradients.len() {
            let mut layer_weight_updates = Vec::new();

            for i in 0..accumulated_weight_gradients[layer_idx].len() {
                let current_gradient = accumulated_weight_gradients[layer_idx][i];
                let previous_gradient = self.previous_weight_gradients[layer_idx][i];
                let previous_delta = self.previous_weight_deltas[layer_idx][i];

                // Get current weight for decay term
                let weight_idx = i;
                let weight = if weight_idx < simple_network.weights[layer_idx].len() {
                    simple_network.weights[layer_idx][weight_idx]
                } else {
                    T::zero()
                };

                let delta = if previous_gradient == T::zero() {
                    // First epoch or no previous gradient: use standard gradient descent with decay
                    -self.learning_rate * current_gradient + self.decay * weight
                } else {
                    let gradient_diff = previous_gradient - current_gradient;

                    if gradient_diff.abs() < T::from(1e-15).unwrap() {
                        // Gradient difference too small: use momentum-like update with decay
                        -self.learning_rate * current_gradient + self.decay * weight
                    } else {
                        // Quickprop formula: delta = (gradient / (previous_gradient - gradient)) * previous_delta
                        let mut quickprop_delta =
                            (current_gradient / gradient_diff) * previous_delta;

                        // Apply maximum growth factor constraint
                        let max_delta = self.mu * previous_delta.abs();
                        if quickprop_delta.abs() > max_delta && previous_delta != T::zero() {
                            quickprop_delta = if quickprop_delta > T::zero() {
                                max_delta
                            } else {
                                -max_delta
                            };
                        }

                        // Conditional gradient addition (if moving in same direction)
                        if quickprop_delta * current_gradient > T::zero() {
                            quickprop_delta =
                                quickprop_delta - self.learning_rate * current_gradient;
                        }

                        // Add decay term
                        quickprop_delta + self.decay * weight
                    }
                };

                layer_weight_updates.push(delta);

                // Store gradient and delta for next iteration
                self.previous_weight_gradients[layer_idx][i] = current_gradient;
                self.previous_weight_deltas[layer_idx][i] = delta;
            }

            weight_updates.push(layer_weight_updates);
        }

        // Update biases using Quickprop algorithm (no decay for biases)
        for layer_idx in 0..accumulated_bias_gradients.len() {
            let mut layer_bias_updates = Vec::new();

            for i in 0..accumulated_bias_gradients[layer_idx].len() {
                let current_gradient = accumulated_bias_gradients[layer_idx][i];
                let previous_gradient = self.previous_bias_gradients[layer_idx][i];
                let previous_delta = self.previous_bias_deltas[layer_idx][i];

                let delta = if previous_gradient == T::zero() {
                    // First epoch or no previous gradient: use standard gradient descent
                    -self.learning_rate * current_gradient
                } else {
                    let gradient_diff = previous_gradient - current_gradient;

                    if gradient_diff.abs() < T::from(1e-15).unwrap() {
                        // Gradient difference too small: use momentum-like update
                        -self.learning_rate * current_gradient
                    } else {
                        // Quickprop formula
                        let mut quickprop_delta =
                            (current_gradient / gradient_diff) * previous_delta;

                        // Apply maximum growth factor constraint
                        let max_delta = self.mu * previous_delta.abs();
                        if quickprop_delta.abs() > max_delta && previous_delta != T::zero() {
                            quickprop_delta = if quickprop_delta > T::zero() {
                                max_delta
                            } else {
                                -max_delta
                            };
                        }

                        // Conditional gradient addition
                        if quickprop_delta * current_gradient > T::zero() {
                            quickprop_delta =
                                quickprop_delta - self.learning_rate * current_gradient;
                        }

                        quickprop_delta
                    }
                };

                layer_bias_updates.push(delta);

                // Store gradient and delta for next iteration
                self.previous_bias_gradients[layer_idx][i] = current_gradient;
                self.previous_bias_deltas[layer_idx][i] = delta;
            }

            bias_updates.push(layer_bias_updates);
        }

        // Apply the updates to the actual network
        apply_updates_to_network(network, &weight_updates, &bias_updates);

        Ok(total_error / batch_size)
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
