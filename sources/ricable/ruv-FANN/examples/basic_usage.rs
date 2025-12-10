use ruv_fann::{ActivationFunction, NetworkBuilder};

fn main() {
    // Create a neural network with 2 inputs, 3 hidden neurons, and 1 output
    let mut network = NetworkBuilder::<f32>::new()
        .input_layer(2)
        .hidden_layer_with_activation(3, ActivationFunction::Sigmoid, 1.0)
        .output_layer_with_activation(1, ActivationFunction::Linear, 1.0)
        .connection_rate(1.0) // Fully connected
        .build();

    println!("Created network with {} layers", network.num_layers());
    println!("Input neurons: {}", network.num_inputs());
    println!("Output neurons: {}", network.num_outputs());
    println!("Total neurons: {}", network.total_neurons());
    println!("Total connections: {}", network.total_connections());

    // Run the network with some test inputs
    let inputs = vec![0.5, 0.7];
    let outputs = network.run(&inputs);

    println!("Inputs: {inputs:?}");
    println!("Outputs: {outputs:?}");

    // Get and display current weights
    let weights = network.get_weights();
    println!("Number of weights: {}", weights.len());

    // Example of setting new weights (normally done by training algorithm)
    let new_weights: Vec<f32> = (0..weights.len()).map(|i| (i as f32) * 0.1 - 0.5).collect();

    if let Ok(()) = network.set_weights(&new_weights) {
        println!("Successfully updated weights");

        // Run again with new weights
        let new_outputs = network.run(&inputs);
        println!("New outputs: {new_outputs:?}");
    }
}
