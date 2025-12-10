//! Resilient Propagation (RPROP) training algorithm

#![allow(clippy::needless_range_loop)]

use super::*;
use num_traits::Float;
use std::collections::HashMap;

/// RPROP (Resilient Propagation) trainer
/// An adaptive learning algorithm that only uses the sign of the gradient
pub struct Rprop<T: Float + Send + Default> {
    increase_factor: T,
    decrease_factor: T,
    delta_min: T,
    delta_max: T,
    delta_zero: T,
    error_function: Box<dyn ErrorFunction<T>>,

    // State variables
    weight_step_sizes: Vec<Vec<T>>,
    bias_step_sizes: Vec<Vec<T>>,
    previous_weight_gradients: Vec<Vec<T>>,
    previous_bias_gradients: Vec<Vec<T>>,

    callback: Option<TrainingCallback<T>>,
}

impl<T: Float + Send + Default> Rprop<T> {
    pub fn new() -> Self {
        Self {
            increase_factor: T::from(1.2).unwrap(),
            decrease_factor: T::from(0.5).unwrap(),
            delta_min: T::zero(),
            delta_max: T::from(50.0).unwrap(),
            delta_zero: T::from(0.1).unwrap(),
            error_function: Box::new(MseError),
            weight_step_sizes: Vec::new(),
            bias_step_sizes: Vec::new(),
            previous_weight_gradients: Vec::new(),
            previous_bias_gradients: Vec::new(),
            callback: None,
        }
    }

    pub fn with_parameters(
        mut self,
        increase_factor: T,
        decrease_factor: T,
        delta_min: T,
        delta_max: T,
        delta_zero: T,
    ) -> Self {
        self.increase_factor = increase_factor;
        self.decrease_factor = decrease_factor;
        self.delta_min = delta_min;
        self.delta_max = delta_max;
        self.delta_zero = delta_zero;
        self
    }

    pub fn with_error_function(mut self, error_function: Box<dyn ErrorFunction<T>>) -> Self {
        self.error_function = error_function;
        self
    }

    fn initialize_state(&mut self, network: &Network<T>) {
        if self.weight_step_sizes.is_empty() {
            // Initialize step sizes and gradients for each layer
            self.weight_step_sizes = network
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
                    vec![self.delta_zero; num_neurons * num_connections]
                })
                .collect();

            self.bias_step_sizes = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| vec![self.delta_zero; layer.neurons.len()])
                .collect();

            // Initialize previous gradients to zero
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
        }
    }

    fn update_step_size(&self, step_size: T, gradient: T, previous_gradient: T) -> T {
        let sign_change = gradient * previous_gradient;

        if sign_change > T::zero() {
            // Same sign: increase step size
            (step_size * self.increase_factor).min(self.delta_max)
        } else if sign_change < T::zero() {
            // Different sign: decrease step size
            (step_size * self.decrease_factor).max(self.delta_min)
        } else {
            // No previous gradient or zero gradient: keep step size
            step_size
        }
    }
}

impl<T: Float + Send + Default> Default for Rprop<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Default> TrainingAlgorithm<T> for Rprop<T> {
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

        // Apply RPROP updates
        let mut weight_updates = Vec::new();
        let mut bias_updates = Vec::new();

        // Update weights using RPROP algorithm
        for layer_idx in 0..accumulated_weight_gradients.len() {
            let mut layer_weight_updates = Vec::new();

            for i in 0..accumulated_weight_gradients[layer_idx].len() {
                let current_gradient = accumulated_weight_gradients[layer_idx][i];
                let previous_gradient = self.previous_weight_gradients[layer_idx][i];
                let sign_change = current_gradient * previous_gradient;

                // Update step size based on gradient sign change
                if sign_change > T::zero() {
                    // Same sign - increase step size
                    self.weight_step_sizes[layer_idx][i] = (self.weight_step_sizes[layer_idx][i]
                        * self.increase_factor)
                        .min(self.delta_max);

                    // Update weight
                    let update = if current_gradient > T::zero() {
                        -self.weight_step_sizes[layer_idx][i]
                    } else if current_gradient < T::zero() {
                        self.weight_step_sizes[layer_idx][i]
                    } else {
                        T::zero()
                    };
                    layer_weight_updates.push(update);

                    // Store gradient for next iteration
                    self.previous_weight_gradients[layer_idx][i] = current_gradient;
                } else if sign_change < T::zero() {
                    // Sign changed - decrease step size and backtrack
                    self.weight_step_sizes[layer_idx][i] = (self.weight_step_sizes[layer_idx][i]
                        * self.decrease_factor)
                        .max(self.delta_min);

                    // Don't update weight (backtrack)
                    layer_weight_updates.push(T::zero());

                    // Set gradient to zero to prevent another sign change detection
                    self.previous_weight_gradients[layer_idx][i] = T::zero();
                } else {
                    // No previous gradient or current gradient is zero
                    let update = if current_gradient > T::zero() {
                        -self.weight_step_sizes[layer_idx][i]
                    } else if current_gradient < T::zero() {
                        self.weight_step_sizes[layer_idx][i]
                    } else {
                        T::zero()
                    };
                    layer_weight_updates.push(update);

                    // Store gradient for next iteration
                    self.previous_weight_gradients[layer_idx][i] = current_gradient;
                }
            }

            weight_updates.push(layer_weight_updates);
        }

        // Update biases using RPROP algorithm
        for layer_idx in 0..accumulated_bias_gradients.len() {
            let mut layer_bias_updates = Vec::new();

            for i in 0..accumulated_bias_gradients[layer_idx].len() {
                let current_gradient = accumulated_bias_gradients[layer_idx][i];
                let previous_gradient = self.previous_bias_gradients[layer_idx][i];
                let sign_change = current_gradient * previous_gradient;

                // Update step size based on gradient sign change
                if sign_change > T::zero() {
                    // Same sign - increase step size
                    self.bias_step_sizes[layer_idx][i] = (self.bias_step_sizes[layer_idx][i]
                        * self.increase_factor)
                        .min(self.delta_max);

                    // Update bias
                    let update = if current_gradient > T::zero() {
                        -self.bias_step_sizes[layer_idx][i]
                    } else if current_gradient < T::zero() {
                        self.bias_step_sizes[layer_idx][i]
                    } else {
                        T::zero()
                    };
                    layer_bias_updates.push(update);

                    // Store gradient for next iteration
                    self.previous_bias_gradients[layer_idx][i] = current_gradient;
                } else if sign_change < T::zero() {
                    // Sign changed - decrease step size and backtrack
                    self.bias_step_sizes[layer_idx][i] = (self.bias_step_sizes[layer_idx][i]
                        * self.decrease_factor)
                        .max(self.delta_min);

                    // Don't update bias (backtrack)
                    layer_bias_updates.push(T::zero());

                    // Set gradient to zero to prevent another sign change detection
                    self.previous_bias_gradients[layer_idx][i] = T::zero();
                } else {
                    // No previous gradient or current gradient is zero
                    let update = if current_gradient > T::zero() {
                        -self.bias_step_sizes[layer_idx][i]
                    } else if current_gradient < T::zero() {
                        self.bias_step_sizes[layer_idx][i]
                    } else {
                        T::zero()
                    };
                    layer_bias_updates.push(update);

                    // Store gradient for next iteration
                    self.previous_bias_gradients[layer_idx][i] = current_gradient;
                }
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

        // Save RPROP parameters
        state.insert("increase_factor".to_string(), vec![self.increase_factor]);
        state.insert("decrease_factor".to_string(), vec![self.decrease_factor]);
        state.insert("delta_min".to_string(), vec![self.delta_min]);
        state.insert("delta_max".to_string(), vec![self.delta_max]);
        state.insert("delta_zero".to_string(), vec![self.delta_zero]);

        // Save step sizes (flattened)
        let mut all_weight_steps = Vec::new();
        for layer_steps in &self.weight_step_sizes {
            all_weight_steps.extend_from_slice(layer_steps);
        }
        state.insert("weight_step_sizes".to_string(), all_weight_steps);

        let mut all_bias_steps = Vec::new();
        for layer_steps in &self.bias_step_sizes {
            all_bias_steps.extend_from_slice(layer_steps);
        }
        state.insert("bias_step_sizes".to_string(), all_bias_steps);

        TrainingState {
            epoch: 0,
            best_error: T::from(f32::MAX).unwrap(),
            algorithm_specific: state,
        }
    }

    fn restore_state(&mut self, state: TrainingState<T>) {
        // Restore RPROP parameters
        if let Some(val) = state.algorithm_specific.get("increase_factor") {
            if !val.is_empty() {
                self.increase_factor = val[0];
            }
        }
        if let Some(val) = state.algorithm_specific.get("decrease_factor") {
            if !val.is_empty() {
                self.decrease_factor = val[0];
            }
        }
        if let Some(val) = state.algorithm_specific.get("delta_min") {
            if !val.is_empty() {
                self.delta_min = val[0];
            }
        }
        if let Some(val) = state.algorithm_specific.get("delta_max") {
            if !val.is_empty() {
                self.delta_max = val[0];
            }
        }
        if let Some(val) = state.algorithm_specific.get("delta_zero") {
            if !val.is_empty() {
                self.delta_zero = val[0];
            }
        }

        // Note: Step sizes would need network structure info to properly restore
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
