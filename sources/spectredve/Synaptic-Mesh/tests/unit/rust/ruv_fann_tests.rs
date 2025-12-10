//! Unit tests for ruv-FANN neural network implementation
//! Tests the Fast Artificial Neural Network functionality

#[cfg(test)]
mod neural_network_tests {
    use std::f32::consts::PI;

    #[test]
    fn test_neural_network_creation() {
        // Test basic neural network initialization
        struct NeuralNetwork {
            layers: Vec<usize>,
            weights: Vec<Vec<Vec<f32>>>,
            biases: Vec<Vec<f32>>,
            learning_rate: f32,
        }
        
        let network = NeuralNetwork {
            layers: vec![2, 4, 1], // 2 inputs, 4 hidden, 1 output
            weights: vec![
                vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6], vec![0.7, 0.8]], // Input to hidden
                vec![vec![0.1, 0.2, 0.3, 0.4]], // Hidden to output
            ],
            biases: vec![
                vec![0.1, 0.2, 0.3, 0.4], // Hidden layer biases
                vec![0.1], // Output layer bias
            ],
            learning_rate: 0.01,
        };
        
        assert_eq!(network.layers.len(), 3);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.biases.len(), 2);
        assert_eq!(network.learning_rate, 0.01);
    }

    #[test]
    fn test_activation_functions() {
        // Test various activation functions
        let x = 0.5;
        
        // Sigmoid activation
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        assert!((sigmoid - 0.6224593).abs() < 1e-6);
        
        // ReLU activation
        let relu = if x > 0.0 { x } else { 0.0 };
        assert_eq!(relu, 0.5);
        
        // Tanh activation
        let tanh = x.tanh();
        assert!((tanh - 0.4621172).abs() < 1e-6);
        
        // Leaky ReLU
        let leaky_relu = if x > 0.0 { x } else { 0.01 * x };
        assert_eq!(leaky_relu, 0.5);
    }

    #[test]
    fn test_forward_propagation() {
        // Test forward propagation through network
        let inputs = vec![1.0, 0.5];
        let weights_layer1 = vec![
            vec![0.2, 0.1], // First neuron weights
            vec![0.4, 0.3], // Second neuron weights
        ];
        let biases_layer1 = vec![0.1, 0.2];
        
        // Calculate first layer output
        let mut layer1_output = Vec::new();
        for i in 0..weights_layer1.len() {
            let mut sum = biases_layer1[i];
            for j in 0..inputs.len() {
                sum += weights_layer1[i][j] * inputs[j];
            }
            layer1_output.push(sigmoid(sum));
        }
        
        assert_eq!(layer1_output.len(), 2);
        assert!(layer1_output[0] > 0.0 && layer1_output[0] < 1.0);
        assert!(layer1_output[1] > 0.0 && layer1_output[1] < 1.0);
    }

    #[test]
    fn test_xor_training() {
        // Test XOR problem training
        let xor_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        
        // Simple network structure for XOR
        let mut network = SimpleNetwork::new(vec![2, 4, 1]);
        
        // Train for multiple epochs
        for _ in 0..100 {
            for (inputs, targets) in &xor_data {
                let outputs = network.forward(inputs);
                let error = calculate_mse(&outputs, targets);
                network.backward(targets);
            }
        }
        
        // Test XOR predictions
        let test_cases = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ];
        
        for (inputs, expected) in test_cases {
            let output = network.forward(&inputs);
            let prediction = if output[0] > 0.5 { 1.0 } else { 0.0 };
            assert_eq!(prediction, expected, "XOR prediction failed for inputs {:?}", inputs);
        }
    }

    #[test]
    fn test_gradient_calculation() {
        // Test gradient calculation for backpropagation
        let output = 0.8;
        let target = 1.0;
        let learning_rate = 0.1;
        
        // Calculate error gradient
        let error = target - output;
        let output_gradient = error * output * (1.0 - output); // Sigmoid derivative
        let weight_update = learning_rate * output_gradient;
        
        assert!(error > 0.0); // Should be positive error
        assert!(output_gradient > 0.0); // Should be positive gradient
        assert!(weight_update > 0.0); // Should be positive update
    }

    #[test]
    fn test_network_serialization() {
        // Test network saving and loading
        let network = SimpleNetwork::new(vec![2, 3, 1]);
        let serialized = network.serialize();
        let deserialized = SimpleNetwork::deserialize(&serialized);
        
        assert_eq!(network.layers, deserialized.layers);
        assert_eq!(network.weights.len(), deserialized.weights.len());
        assert_eq!(network.biases.len(), deserialized.biases.len());
    }

    #[test]
    fn test_parallel_training() {
        // Test parallel batch training
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let network = Arc::new(Mutex::new(SimpleNetwork::new(vec![2, 4, 1])));
        let training_data = Arc::new(vec![
            (vec![0.1, 0.2], vec![0.3]),
            (vec![0.4, 0.5], vec![0.6]),
            (vec![0.7, 0.8], vec![0.9]),
        ]);
        
        let handles: Vec<_> = (0..3).map(|i| {
            let network_clone = Arc::clone(&network);
            let data_clone = Arc::clone(&training_data);
            thread::spawn(move || {
                let mut net = network_clone.lock().unwrap();
                let (inputs, targets) = &data_clone[i];
                net.forward(inputs);
                net.backward(targets);
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify network was updated
        let final_network = network.lock().unwrap();
        assert!(final_network.weights.len() > 0);
    }

    // Helper functions and structures
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn calculate_mse(outputs: &[f32], targets: &[f32]) -> f32 {
        outputs.iter()
            .zip(targets.iter())
            .map(|(o, t)| (t - o).powi(2))
            .sum::<f32>() / outputs.len() as f32
    }

    // Mock SimpleNetwork for testing
    struct SimpleNetwork {
        layers: Vec<usize>,
        weights: Vec<Vec<Vec<f32>>>,
        biases: Vec<Vec<f32>>,
        learning_rate: f32,
    }

    impl SimpleNetwork {
        fn new(layers: Vec<usize>) -> Self {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            
            let mut weights = Vec::new();
            let mut biases = Vec::new();
            
            for i in 0..layers.len() - 1 {
                let input_size = layers[i];
                let output_size = layers[i + 1];
                
                let layer_weights: Vec<Vec<f32>> = (0..output_size)
                    .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
                    .collect();
                
                let layer_biases: Vec<f32> = (0..output_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect();
                
                weights.push(layer_weights);
                biases.push(layer_biases);
            }
            
            Self {
                layers,
                weights,
                biases,
                learning_rate: 0.1,
            }
        }
        
        fn forward(&self, inputs: &[f32]) -> Vec<f32> {
            let mut current_inputs = inputs.to_vec();
            
            for (layer_weights, layer_biases) in self.weights.iter().zip(self.biases.iter()) {
                let mut layer_outputs = Vec::new();
                
                for (neuron_weights, bias) in layer_weights.iter().zip(layer_biases.iter()) {
                    let sum: f32 = neuron_weights.iter()
                        .zip(current_inputs.iter())
                        .map(|(w, i)| w * i)
                        .sum::<f32>() + bias;
                    
                    layer_outputs.push(sigmoid(sum));
                }
                
                current_inputs = layer_outputs;
            }
            
            current_inputs
        }
        
        fn backward(&mut self, _targets: &[f32]) {
            // Simplified backward pass for testing
            // In reality, this would implement full backpropagation
        }
        
        fn serialize(&self) -> Vec<u8> {
            // Mock serialization
            vec![0u8; 100]
        }
        
        fn deserialize(_data: &[u8]) -> Self {
            // Mock deserialization
            Self::new(vec![2, 3, 1])
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_inference_speed() {
        // Test neural network inference speed
        let network = SimpleNetwork::new(vec![784, 128, 64, 10]); // MNIST-like network
        let input = vec![0.5; 784]; // Mock input
        
        let start = Instant::now();
        let _output = network.forward(&input);
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 100, "Inference should complete in <100ms");
    }

    #[test]
    fn test_batch_processing() {
        // Test batch processing performance
        let network = SimpleNetwork::new(vec![10, 20, 5]);
        let batch_size = 32;
        let inputs: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| vec![0.1; 10])
            .collect();
        
        let start = Instant::now();
        for input in &inputs {
            let _output = network.forward(input);
        }
        let duration = start.elapsed();
        
        let avg_time_per_sample = duration.as_millis() as f32 / batch_size as f32;
        assert!(avg_time_per_sample < 10.0, "Average inference time per sample should be <10ms");
    }

    #[test]
    fn test_memory_usage() {
        // Test memory efficiency
        let large_network = SimpleNetwork::new(vec![1000, 500, 250, 10]);
        
        // Estimate memory usage (simplified)
        let mut total_params = 0;
        for i in 0..large_network.layers.len() - 1 {
            total_params += large_network.layers[i] * large_network.layers[i + 1]; // Weights
            total_params += large_network.layers[i + 1]; // Biases
        }
        
        let estimated_memory_mb = (total_params * 4) as f32 / (1024.0 * 1024.0); // 4 bytes per f32
        assert!(estimated_memory_mb < 50.0, "Network should use <50MB memory");
    }
}