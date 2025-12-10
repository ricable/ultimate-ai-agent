use crate::{ActivationFunction, Connection, Layer, Network, NetworkBuilder, Neuron};
use approx::assert_relative_eq;

#[test]
fn test_connection_creation() {
    let conn = Connection::new(0, 1, 0.5_f32);
    assert_eq!(conn.from_neuron, 0);
    assert_eq!(conn.to_neuron, 1);
    assert_relative_eq!(conn.weight, 0.5_f32);
}

#[test]
fn test_connection_generic_f64() {
    let conn = Connection::new(0, 1, 0.5_f64);
    assert_eq!(conn.from_neuron, 0);
    assert_eq!(conn.to_neuron, 1);
    assert_relative_eq!(conn.weight, 0.5_f64);
}

#[test]
fn test_neuron_creation() {
    let neuron = Neuron::<f32>::new(ActivationFunction::Sigmoid, 1.0);
    assert_eq!(neuron.activation_function, ActivationFunction::Sigmoid);
    assert_relative_eq!(neuron.activation_steepness, 1.0);
    assert_relative_eq!(neuron.sum, 0.0);
    assert_relative_eq!(neuron.value, 0.0);
    assert!(neuron.connections.is_empty());
}

#[test]
fn test_neuron_add_connection() {
    let mut neuron = Neuron::<f32>::new(ActivationFunction::Sigmoid, 1.0);
    neuron.add_connection(0, 0.5);

    assert_eq!(neuron.connections.len(), 1);
    assert_eq!(neuron.connections[0].from_neuron, 0);
    assert_relative_eq!(neuron.connections[0].weight, 0.5);
}

#[test]
fn test_layer_creation() {
    let layer = Layer::<f32>::new(3, ActivationFunction::Sigmoid, 1.0);
    assert_eq!(layer.neurons.len(), 3);

    for neuron in &layer.neurons {
        assert_eq!(neuron.activation_function, ActivationFunction::Sigmoid);
        assert_relative_eq!(neuron.activation_steepness, 1.0);
    }
}

#[test]
fn test_layer_with_bias() {
    let layer = Layer::<f32>::with_bias(3, ActivationFunction::Sigmoid, 1.0);
    assert_eq!(layer.neurons.len(), 4); // 3 regular + 1 bias
    assert!(layer.has_bias());

    // Bias neuron should have value 1.0
    let bias_neuron = &layer.neurons[3];
    assert_relative_eq!(bias_neuron.value, 1.0);
}

#[test]
fn test_network_builder_basic() {
    let network: Network<f32> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(3)
        .output_layer(1)
        .build();

    assert_eq!(network.num_layers(), 3);
    assert_eq!(network.num_inputs(), 2);
    assert_eq!(network.num_outputs(), 1);

    // Check layer sizes (including bias neurons in input and hidden layers)
    assert_eq!(network.layers[0].neurons.len(), 3); // 2 input + 1 bias
    assert_eq!(network.layers[1].neurons.len(), 4); // 3 hidden + 1 bias
    assert_eq!(network.layers[2].neurons.len(), 1); // 1 output (no bias)
}

#[test]
fn test_network_builder_with_activation_functions() {
    let network: Network<f32> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer_with_activation(3, ActivationFunction::ReLU, 1.0)
        .output_layer_with_activation(1, ActivationFunction::Linear, 1.0)
        .build();

    // Check activation functions
    let hidden_layer = &network.layers[1];
    for neuron in &hidden_layer.neurons[..3] {
        // Exclude bias neuron
        assert_eq!(neuron.activation_function, ActivationFunction::ReLU);
    }

    let output_layer = &network.layers[2];
    assert_eq!(
        output_layer.neurons[0].activation_function,
        ActivationFunction::Linear
    );
}

#[test]
fn test_network_fully_connected() {
    let network: Network<f32> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(3)
        .output_layer(1)
        .connection_rate(1.0)
        .build();

    // Check connections
    // Input layer (2 neurons + 1 bias) should connect to all 3 hidden neurons
    let hidden_layer = &network.layers[1];
    for i in 0..3 {
        // For each hidden neuron (excluding bias)
        let neuron = &hidden_layer.neurons[i];
        assert_eq!(neuron.connections.len(), 3); // Connected to 2 input + 1 bias
    }

    // Hidden layer (3 neurons + 1 bias) should connect to output neuron
    let output_layer = &network.layers[2];
    let output_neuron = &output_layer.neurons[0];
    assert_eq!(output_neuron.connections.len(), 4); // Connected to 3 hidden + 1 bias
}

#[test]
fn test_network_sparse_connections() {
    let network: Network<f32> = NetworkBuilder::new()
        .input_layer(10)
        .hidden_layer(10)
        .output_layer(10)
        .connection_rate(0.5)
        .build();

    // With 50% connection rate, we should have roughly half the connections
    let total_connections = network.total_connections();
    let max_connections = 11 * 10 + 11 * 10; // (10+1) * 10 + (10+1) * 10

    assert!(total_connections > (max_connections as f32 * 0.3) as usize); // Allow some variance
    assert!(total_connections < (max_connections as f32 * 0.7) as usize);
}

#[test]
fn test_network_multiple_hidden_layers() {
    let network: Network<f32> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(4)
        .hidden_layer(3)
        .hidden_layer(2)
        .output_layer(1)
        .build();

    assert_eq!(network.num_layers(), 5);
    assert_eq!(network.layers[1].neurons.len(), 5); // 4 + 1 bias
    assert_eq!(network.layers[2].neurons.len(), 4); // 3 + 1 bias
    assert_eq!(network.layers[3].neurons.len(), 3); // 2 + 1 bias
}

#[test]
fn test_network_generic_f64() {
    let network: Network<f64> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(3)
        .output_layer(1)
        .build();

    assert_eq!(network.num_layers(), 3);
    assert_eq!(network.num_inputs(), 2);
    assert_eq!(network.num_outputs(), 1);
}

#[test]
fn test_network_run_forward_pass() {
    let mut network: Network<f32> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(2)
        .output_layer(1)
        .build();

    // Set some weights manually for testing
    // This is a simple test, in real usage weights would be trained
    let hidden_layer = &mut network.layers[1];
    for neuron in &mut hidden_layer.neurons[..2] {
        neuron.connections.clear();
        neuron.add_connection(0, 0.5); // From input 0
        neuron.add_connection(1, 0.5); // From input 1
        neuron.add_connection(2, 0.0); // From bias
    }

    let output_layer = &mut network.layers[2];
    output_layer.neurons[0].connections.clear();
    output_layer.neurons[0].add_connection(3, 1.0); // From hidden 0
    output_layer.neurons[0].add_connection(4, 1.0); // From hidden 1
    output_layer.neurons[0].add_connection(5, 0.0); // From bias

    let input = vec![1.0, 1.0];
    let output = network.run(&input);

    assert_eq!(output.len(), 1);
    // The actual value depends on activation function implementation
}

#[test]
fn test_activation_functions() {
    // Test that all activation function variants are available
    let _sigmoid = ActivationFunction::Sigmoid;
    let _tanh = ActivationFunction::Tanh;
    let _relu = ActivationFunction::ReLU;
    let _linear = ActivationFunction::Linear;
    let _gaussian = ActivationFunction::Gaussian;
    let _sigmoid_symmetric = ActivationFunction::SigmoidSymmetric;
    let _elliot = ActivationFunction::Elliot;
    let _elliot_symmetric = ActivationFunction::ElliotSymmetric;
    let _sin = ActivationFunction::Sin;
    let _cos = ActivationFunction::Cos;
    let _sin_symmetric = ActivationFunction::SinSymmetric;
    let _cos_symmetric = ActivationFunction::CosSymmetric;
    let _threshold = ActivationFunction::Threshold;
    let _threshold_symmetric = ActivationFunction::ThresholdSymmetric;
    let _linear_piece = ActivationFunction::LinearPiece;
    let _linear_piece_symmetric = ActivationFunction::LinearPieceSymmetric;
    let _relu_leaky = ActivationFunction::ReLULeaky;
}

#[test]
fn test_network_get_weights() {
    let network: Network<f32> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(2)
        .output_layer(1)
        .build();

    let weights = network.get_weights();
    let expected_weights = 3 * 2 + 3; // (2+1 bias) * 2 hidden + (2+1 bias) * 1 output
    assert_eq!(weights.len(), expected_weights);
}

#[test]
fn test_network_set_weights() {
    let mut network: Network<f32> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(2)
        .output_layer(1)
        .build();

    let new_weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    assert!(network.set_weights(&new_weights).is_ok());

    let retrieved_weights = network.get_weights();
    for (i, &weight) in retrieved_weights.iter().enumerate() {
        assert_relative_eq!(weight, new_weights[i]);
    }
}

#[test]
fn test_network_set_weights_wrong_size() {
    let mut network: Network<f32> = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(2)
        .output_layer(1)
        .build();

    let wrong_weights = vec![0.1, 0.2]; // Too few weights
    assert!(network.set_weights(&wrong_weights).is_err());
}
