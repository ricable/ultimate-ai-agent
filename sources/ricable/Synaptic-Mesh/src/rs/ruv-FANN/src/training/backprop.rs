//! Backpropagation training algorithms

use super::*;
use num_traits::Float;
use std::collections::HashMap;

/// Incremental (online) backpropagation
/// Updates weights after each training pattern
pub struct IncrementalBackprop<T: Float + Send + Default> {
    learning_rate: T,
    momentum: T,
    error_function: Box<dyn ErrorFunction<T>>,
    previous_weight_deltas: Vec<Vec<T>>,
    previous_bias_deltas: Vec<Vec<T>>,
    callback: Option<TrainingCallback<T>>,
}

impl<T: Float + Send + Default> IncrementalBackprop<T> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            momentum: T::zero(),
            error_function: Box::new(MseError),
            previous_weight_deltas: Vec::new(),
            previous_bias_deltas: Vec::new(),
            callback: None,
        }
    }

    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_error_function(mut self, error_function: Box<dyn ErrorFunction<T>>) -> Self {
        self.error_function = error_function;
        self
    }

    fn initialize_deltas(&mut self, network: &Network<T>) {
        if self.previous_weight_deltas.is_empty() {
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
}

impl<T: Float + Send + Default> TrainingAlgorithm<T> for IncrementalBackprop<T> {
    fn train_epoch(
        &mut self,
        network: &mut Network<T>,
        data: &TrainingData<T>,
    ) -> Result<T, TrainingError> {
        use super::helpers::*;

        self.initialize_deltas(network);

        let mut total_error = T::zero();

        // Convert network to simplified form for easier manipulation
        let simple_network = network_to_simple(network);

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

            // Update weights and biases immediately (incremental/online learning)
            // Apply momentum
            for layer_idx in 0..weight_gradients.len() {
                // Update weight deltas with momentum
                for (i, &grad) in weight_gradients[layer_idx].iter().enumerate() {
                    let delta = self.learning_rate * grad
                        + self.momentum * self.previous_weight_deltas[layer_idx][i];
                    self.previous_weight_deltas[layer_idx][i] = delta;
                }

                // Update bias deltas with momentum
                for (i, &grad) in bias_gradients[layer_idx].iter().enumerate() {
                    let delta = self.learning_rate * grad
                        + self.momentum * self.previous_bias_deltas[layer_idx][i];
                    self.previous_bias_deltas[layer_idx][i] = delta;
                }
            }

            // Apply the updates to the actual network
            apply_updates_to_network(
                network,
                &self.previous_weight_deltas,
                &self.previous_bias_deltas,
            );
        }

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
        state.insert("learning_rate".to_string(), vec![self.learning_rate]);
        state.insert("momentum".to_string(), vec![self.momentum]);

        TrainingState {
            epoch: 0,
            best_error: T::from(f32::MAX).unwrap(),
            algorithm_specific: state,
        }
    }

    fn restore_state(&mut self, state: TrainingState<T>) {
        if let Some(lr) = state.algorithm_specific.get("learning_rate") {
            if !lr.is_empty() {
                self.learning_rate = lr[0];
            }
        }
        if let Some(mom) = state.algorithm_specific.get("momentum") {
            if !mom.is_empty() {
                self.momentum = mom[0];
            }
        }
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

/// Batch backpropagation
/// Accumulates gradients over entire dataset before updating weights
pub struct BatchBackprop<T: Float + Send + Default> {
    learning_rate: T,
    momentum: T,
    error_function: Box<dyn ErrorFunction<T>>,
    previous_weight_deltas: Vec<Vec<T>>,
    previous_bias_deltas: Vec<Vec<T>>,
    callback: Option<TrainingCallback<T>>,
}

impl<T: Float + Send + Default> BatchBackprop<T> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            momentum: T::zero(),
            error_function: Box::new(MseError),
            previous_weight_deltas: Vec::new(),
            previous_bias_deltas: Vec::new(),
            callback: None,
        }
    }

    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_error_function(mut self, error_function: Box<dyn ErrorFunction<T>>) -> Self {
        self.error_function = error_function;
        self
    }

    fn initialize_deltas(&mut self, network: &Network<T>) {
        if self.previous_weight_deltas.is_empty() {
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
}

impl<T: Float + Send + Default> TrainingAlgorithm<T> for BatchBackprop<T> {
    fn train_epoch(
        &mut self,
        network: &mut Network<T>,
        data: &TrainingData<T>,
    ) -> Result<T, TrainingError> {
        use super::helpers::*;

        self.initialize_deltas(network);

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

        // Accumulate gradients over all patterns
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

        // Update weights and biases using accumulated gradients with momentum
        let mut weight_updates = Vec::new();
        let mut bias_updates = Vec::new();

        for layer_idx in 0..accumulated_weight_gradients.len() {
            let mut layer_weight_updates = Vec::new();
            let mut layer_bias_updates = Vec::new();

            // Update weights with momentum
            for (i, &grad) in accumulated_weight_gradients[layer_idx].iter().enumerate() {
                let delta = self.learning_rate * grad
                    + self.momentum * self.previous_weight_deltas[layer_idx][i];
                self.previous_weight_deltas[layer_idx][i] = delta;
                layer_weight_updates.push(delta);
            }

            // Update biases with momentum
            for (i, &grad) in accumulated_bias_gradients[layer_idx].iter().enumerate() {
                let delta = self.learning_rate * grad
                    + self.momentum * self.previous_bias_deltas[layer_idx][i];
                self.previous_bias_deltas[layer_idx][i] = delta;
                layer_bias_updates.push(delta);
            }

            weight_updates.push(layer_weight_updates);
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
        state.insert("learning_rate".to_string(), vec![self.learning_rate]);
        state.insert("momentum".to_string(), vec![self.momentum]);

        TrainingState {
            epoch: 0,
            best_error: T::from(f32::MAX).unwrap(),
            algorithm_specific: state,
        }
    }

    fn restore_state(&mut self, state: TrainingState<T>) {
        if let Some(lr) = state.algorithm_specific.get("learning_rate") {
            if !lr.is_empty() {
                self.learning_rate = lr[0];
            }
        }
        if let Some(mom) = state.algorithm_specific.get("momentum") {
            if !mom.is_empty() {
                self.momentum = mom[0];
            }
        }
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
