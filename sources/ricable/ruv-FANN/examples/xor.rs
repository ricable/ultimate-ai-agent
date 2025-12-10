//! XOR problem example - Neural network demonstration
//!
//! This example shows how to create a neural network for the XOR problem.
//! Note: Training capability will be added when training algorithms are integrated.

use ruv_fann::{ActivationFunction, NetworkBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== XOR Problem Example ===\n");

    // Create a neural network with:
    // - 2 inputs (for the two binary inputs)
    // - 3 hidden neurons (sufficient for XOR)
    // - 1 output (the XOR result)
    let mut network = NetworkBuilder::<f32>::new()
        .input_layer(2)
        .hidden_layer_with_activation(3, ActivationFunction::Sigmoid, 1.0)
        .output_layer_with_activation(1, ActivationFunction::Sigmoid, 1.0)
        .build();

    println!("Network created with architecture: 2-3-1");
    println!("Total neurons: {}", network.total_neurons());
    println!("Total connections: {}", network.total_connections());

    // Configure the network activation functions
    network.set_activation_function_hidden(ActivationFunction::Sigmoid);
    network.set_activation_function_output(ActivationFunction::Sigmoid);

    println!("\nNetwork configuration:");
    println!("- Hidden activation function: Sigmoid");
    println!("- Output activation function: Sigmoid");
    println!("- Note: Training capability will be added in next integration step");

    // Test data for XOR problem (no training yet, just demonstration)
    // XOR truth table:
    // 0 XOR 0 = 0
    // 0 XOR 1 = 1
    // 1 XOR 0 = 1
    // 1 XOR 1 = 0
    let test_inputs = [
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let expected_outputs = [0.0, 1.0, 1.0, 0.0];

    println!("\nXOR Truth Table:");
    for (i, input) in test_inputs.iter().enumerate() {
        println!(
            "{:.1} XOR {:.1} should be {:.1}",
            input[0], input[1], expected_outputs[i]
        );
    }

    // Test the network with random weights (before training)
    println!("\nTesting network with random weights (untrained):");
    println!("Input\t\tExpected\tActual Output");
    println!("--------------------------------------------");

    for (i, input) in test_inputs.iter().enumerate() {
        let output = network.run(input);
        if !output.is_empty() {
            println!(
                "{:.1} XOR {:.1}\t{:.1}\t\t{:.3}",
                input[0], input[1], expected_outputs[i], output[0]
            );
        } else {
            println!("Error: network returned empty output");
        }
    }

    // Demonstrate weight manipulation
    println!("\nNetwork weights info:");
    let weights = network.get_weights();
    println!("Total weights: {}", weights.len());
    println!("Current weights: {weights:?}");

    // Try setting some manual weights that might work better for XOR
    // These are just example weights, not necessarily optimal
    let manual_weights = vec![
        2.0, -2.0, 2.0, -2.0, 1.0, 1.0, // Input to hidden connections + biases
        3.0, -3.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Hidden to output connections + biases
    ];

    if manual_weights.len() == weights.len() {
        if network.set_weights(&manual_weights).is_ok() {
            println!("\nTesting with manually set weights:");
            println!("Input\t\tExpected\tActual Output");
            println!("--------------------------------------------");

            for (i, input) in test_inputs.iter().enumerate() {
                let output = network.run(input);
                if !output.is_empty() {
                    println!(
                        "{:.1} XOR {:.1}\t{:.1}\t\t{:.3}",
                        input[0], input[1], expected_outputs[i], output[0]
                    );
                } else {
                    println!("Error: network returned empty output");
                }
            }
        }
    } else {
        println!(
            "Manual weights length mismatch: expected {}, got {}",
            weights.len(),
            manual_weights.len()
        );
    }

    Ok(())
}
