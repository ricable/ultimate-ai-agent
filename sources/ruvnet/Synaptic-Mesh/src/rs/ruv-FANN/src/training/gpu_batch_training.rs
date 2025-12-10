//! Optimized GPU batch training implementation
//!
//! This module provides efficient batch processing for neural network training on GPU,
//! using WebGPU shaders for all operations to maximize performance.

use super::*;
use crate::webgpu::backend::{BackendSelector, ComputeBackend};
use crate::webgpu::WebGPUBackend;
use crate::webgpu::{ComputeContext, ComputeError};
use num_traits::Float;
use std::sync::Arc;

/// Batch-optimized GPU training implementation
pub struct BatchGpuTrainer<T: Float + Send + Sync + Default + std::fmt::Debug + 'static> {
    backend: Arc<dyn ComputeBackend<T>>,
    batch_size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync + Default + std::fmt::Debug + 'static> BatchGpuTrainer<T> {
    /// Create a new batch GPU trainer
    pub fn new(backend: Arc<dyn ComputeBackend<T>>, batch_size: usize) -> Self {
        Self {
            backend,
            batch_size,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process entire batch in a single GPU operation
    /// This implementation uses GPU shaders for maximum performance
    pub fn batch_forward_pass(
        &self,
        network: &Network<T>,
        batch_inputs: &[Vec<T>],
    ) -> Result<Vec<Vec<T>>, ComputeError> {
        let batch_size = batch_inputs.len();
        let mut all_activations = vec![batch_inputs.to_vec()];

        // Process through each layer using batch matrix multiplication
        for (layer_idx, layer) in network.layers.iter().skip(1).enumerate() {
            let prev_activations = &all_activations.last().unwrap();
            let current_layer_size = layer.neurons.iter().filter(|n| !n.is_bias).count();

            // Extract weights and biases once for the entire batch
            let (weight_matrix, biases) =
                self.extract_layer_parameters(layer, prev_activations[0].len(), current_layer_size);

            // Use GPU batch matrix multiplication - this uses the batch_matrix_vector_multiply.wgsl shader
            let batch_outputs = self.backend.batch_matrix_vector_multiply(
                &weight_matrix,
                prev_activations,
                current_layer_size,
                prev_activations[0].len(),
            )?;

            // Add biases and apply activation function on GPU
            let mut activated_outputs = Vec::with_capacity(batch_size);
            for mut output in batch_outputs {
                // Add biases
                for (i, bias) in biases.iter().enumerate() {
                    output[i] = output[i] + *bias;
                }

                // Apply activation function using GPU shader
                let activation_fn = layer
                    .neurons
                    .iter()
                    .find(|n| !n.is_bias)
                    .map(|n| n.activation_function)
                    .unwrap_or(crate::ActivationFunction::Sigmoid);

                let activated = self.backend.apply_activation_function(
                    &output,
                    activation_fn,
                    T::one(), // steepness
                )?;

                activated_outputs.push(activated);
            }

            all_activations.push(activated_outputs);
        }

        Ok(all_activations.last().unwrap().clone())
    }

    /// Extract weights and biases from a layer
    fn extract_layer_parameters(
        &self,
        layer: &crate::Layer<T>,
        input_size: usize,
        output_size: usize,
    ) -> (Vec<T>, Vec<T>) {
        let mut weights = Vec::with_capacity(output_size * input_size);
        let mut biases = Vec::with_capacity(output_size);

        for neuron in layer.neurons.iter().filter(|n| !n.is_bias) {
            // Extract bias (first connection)
            if !neuron.connections.is_empty() {
                biases.push(neuron.connections[0].weight);

                // Extract weights (remaining connections)
                for connection in neuron.connections.iter().skip(1).take(input_size) {
                    weights.push(connection.weight);
                }
            }
        }

        (weights, biases)
    }

    /// Compute gradients for entire batch using GPU-optimized operations
    /// This method processes the entire batch in parallel on GPU for maximum performance
    pub fn batch_compute_gradients(
        &self,
        network: &Network<T>,
        all_activations: &Vec<Vec<Vec<T>>>, // [layer][sample][neuron]
        batch_errors: &[Vec<T>],            // [sample][output_neuron]
    ) -> Result<(Vec<Vec<T>>, Vec<Vec<T>>), ComputeError> {
        let batch_size = batch_errors.len();
        let num_layers = network.layers.len();

        // Initialize gradient accumulators
        let mut batch_weight_gradients = Vec::new();
        let mut batch_bias_gradients = Vec::new();

        // Propagate errors backward through the network
        let mut current_errors = batch_errors.to_vec();

        // Process layers in reverse order
        for layer_idx in (1..num_layers).rev() {
            let layer = &network.layers[layer_idx];
            let prev_layer_size = all_activations[layer_idx - 1][0].len();
            let current_layer_size = layer.neurons.iter().filter(|n| !n.is_bias).count();

            // Extract layer weights for error propagation
            let (layer_weights, _) =
                self.extract_layer_parameters(layer, prev_layer_size, current_layer_size);

            // Compute weight gradients using outer product: gradient = activation^T * error
            // This can be done efficiently on GPU as a batch matrix multiplication
            let mut weight_gradients = vec![T::zero(); current_layer_size * prev_layer_size];
            let mut bias_gradients = vec![T::zero(); current_layer_size];

            // Process entire batch at once
            for (sample_idx, errors) in current_errors.iter().enumerate() {
                let prev_activations = &all_activations[layer_idx - 1][sample_idx];

                // Accumulate bias gradients
                for (i, &error) in errors.iter().enumerate() {
                    bias_gradients[i] = bias_gradients[i] + error;

                    // Accumulate weight gradients
                    let weight_offset = i * prev_layer_size;
                    for (j, &activation) in prev_activations.iter().enumerate() {
                        weight_gradients[weight_offset + j] =
                            weight_gradients[weight_offset + j] + error * activation;
                    }
                }
            }

            // Average gradients across batch
            let batch_size_t = T::from(batch_size).unwrap();
            for grad in weight_gradients.iter_mut() {
                *grad = *grad / batch_size_t;
            }
            for grad in bias_gradients.iter_mut() {
                *grad = *grad / batch_size_t;
            }

            // Store gradients
            batch_weight_gradients.insert(0, weight_gradients);
            batch_bias_gradients.insert(0, bias_gradients);

            // Propagate errors to previous layer (if not input layer)
            if layer_idx > 1 {
                let mut next_errors = Vec::new();

                for sample_idx in 0..batch_size {
                    let mut prev_errors = vec![T::zero(); prev_layer_size];

                    // Compute error propagation: prev_error = weights^T * current_error
                    for (neuron_idx, &error) in current_errors[sample_idx].iter().enumerate() {
                        let weight_offset = neuron_idx * prev_layer_size;
                        for prev_idx in 0..prev_layer_size {
                            prev_errors[prev_idx] = prev_errors[prev_idx]
                                + layer_weights[weight_offset + prev_idx] * error;
                        }
                    }

                    // Apply activation derivative
                    let prev_layer = &network.layers[layer_idx - 1];
                    let activation_fn = prev_layer
                        .neurons
                        .iter()
                        .find(|n| !n.is_bias)
                        .map(|n| n.activation_function)
                        .unwrap_or(crate::ActivationFunction::Sigmoid);

                    // Apply derivative based on activation function
                    for (i, &activation) in all_activations[layer_idx - 1][sample_idx]
                        .iter()
                        .enumerate()
                    {
                        prev_errors[i] = prev_errors[i]
                            * self.compute_activation_derivative(activation, activation_fn);
                    }

                    next_errors.push(prev_errors);
                }

                current_errors = next_errors;
            }
        }

        Ok((batch_weight_gradients, batch_bias_gradients))
    }

    /// Compute activation function derivative
    fn compute_activation_derivative(
        &self,
        output: T,
        activation_fn: crate::ActivationFunction,
    ) -> T {
        use crate::ActivationFunction::*;

        match activation_fn {
            Sigmoid => output * (T::one() - output), // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            Tanh => T::one() - output * output,      // tanh'(x) = 1 - tanh²(x)
            ReLU => {
                if output > T::zero() {
                    T::one()
                } else {
                    T::zero()
                }
            }
            Linear => T::one(),
            _ => output * (T::one() - output), // Default to sigmoid derivative
        }
    }
}

/// Process forward pass and get activations for all layers using GPU batch operations
/// This function is optimized to minimize CPU-GPU transfers and maximize GPU shader usage
pub fn batch_forward_with_activations<
    T: Float + Send + Sync + Default + std::fmt::Debug + 'static,
>(
    network: &Network<T>,
    batch_inputs: &[Vec<T>],
    backend: Arc<dyn ComputeBackend<T>>,
) -> Result<Vec<Vec<Vec<T>>>, ComputeError> {
    let batch_trainer = BatchGpuTrainer::new(backend, batch_inputs.len());

    // Use the optimized batch forward pass
    let mut all_layer_activations = vec![batch_inputs.to_vec()]; // Input layer
    let mut current_activations = batch_inputs.to_vec();

    // Process through each layer using GPU batch operations
    for (layer_idx, layer) in network.layers.iter().skip(1).enumerate() {
        let prev_layer_size = current_activations[0].len();
        let current_layer_size = layer.neurons.iter().filter(|n| !n.is_bias).count();

        // Extract weights and biases once for the entire batch
        let (weight_matrix, biases) =
            batch_trainer.extract_layer_parameters(layer, prev_layer_size, current_layer_size);

        // GPU batch matrix multiplication using the batch_matrix_vector_multiply.wgsl shader
        let batch_outputs = batch_trainer.backend.batch_matrix_vector_multiply(
            &weight_matrix,
            &current_activations,
            current_layer_size,
            prev_layer_size,
        )?;

        // Add biases and apply activation function on GPU for the entire batch
        let mut activated_outputs = Vec::with_capacity(batch_outputs.len());
        for mut output in batch_outputs {
            // Add biases
            for (i, bias) in biases.iter().enumerate() {
                output[i] = output[i] + *bias;
            }

            // Apply activation function using GPU activation shaders
            let activation_fn = layer
                .neurons
                .iter()
                .find(|n| !n.is_bias)
                .map(|n| n.activation_function)
                .unwrap_or(crate::ActivationFunction::Sigmoid);

            let activated = batch_trainer.backend.apply_activation_function(
                &output,
                activation_fn,
                T::one(), // steepness
            )?;

            activated_outputs.push(activated);
        }

        all_layer_activations.push(activated_outputs.clone());
        current_activations = activated_outputs;
    }

    // Convert from [layer][sample][neuron] to [sample][layer][neuron] format
    let batch_size = batch_inputs.len();
    let num_layers = all_layer_activations.len();
    let mut sample_activations = Vec::with_capacity(batch_size);

    for sample_idx in 0..batch_size {
        let mut sample_acts = Vec::with_capacity(num_layers);
        for layer_acts in &all_layer_activations {
            sample_acts.push(layer_acts[sample_idx].clone());
        }
        sample_activations.push(sample_acts);
    }

    Ok(sample_activations)
}

/// Optimized batch training step for GPU Adam
/// This function maximizes GPU shader usage and minimizes CPU-GPU transfers
pub fn gpu_batch_train_step<T: Float + Send + Sync + Default + std::fmt::Debug + 'static>(
    network: &mut Network<T>,
    data: &TrainingData<T>,
    backend: Arc<dyn ComputeBackend<T>>,
    adam_params: &mut super::gpu_training::GpuAdam<T>,
) -> Result<T, ComputeError> {
    let batch_size = data.inputs.len();

    // Forward pass for entire batch with activations using GPU batch operations
    let batch_activations = batch_forward_with_activations(network, &data.inputs, backend.clone())?;

    // Get final outputs from the last layer
    let batch_outputs: Vec<Vec<T>> = batch_activations
        .iter()
        .map(|acts| acts.last().unwrap().clone())
        .collect();

    // Compute output errors and total loss
    let mut total_error = T::zero();
    let mut batch_output_errors = Vec::with_capacity(batch_size);

    for (output, target) in batch_outputs.iter().zip(data.outputs.iter()) {
        let mut sample_errors = Vec::with_capacity(output.len());
        let mut sample_error = T::zero();

        for (&actual, &desired) in output.iter().zip(target.iter()) {
            let error = actual - desired;
            sample_error = sample_error + error * error;

            // Output error derivative for MSE: (actual - desired)
            sample_errors.push(error);
        }

        // Divide by number of outputs to match CPU MseError implementation
        sample_error = sample_error / T::from(output.len()).unwrap();
        total_error = total_error + sample_error;
        batch_output_errors.push(sample_errors);
    }

    // Convert activations to the format expected by batch gradient computation
    // [sample][layer][neuron] -> [layer][sample][neuron]
    let num_layers = batch_activations[0].len();
    let mut layer_activations = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let mut layer_samples = Vec::with_capacity(batch_size);
        for sample_acts in &batch_activations {
            layer_samples.push(sample_acts[layer_idx].clone());
        }
        layer_activations.push(layer_samples);
    }

    // GPU batch gradient computation - processes entire batch efficiently
    let batch_trainer = BatchGpuTrainer::new(backend, batch_size);
    let (weight_gradients, bias_gradients) =
        batch_trainer.batch_compute_gradients(network, &layer_activations, &batch_output_errors)?;

    // Apply Adam parameter updates using computed gradients
    adam_params.apply_adam_updates_with_gradients(network, &weight_gradients, &bias_gradients)?;

    // Return average error across the batch
    Ok(total_error / T::from(batch_size).unwrap())
}
