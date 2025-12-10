//! Test optimizers with simpler problem and better parameters

use ruv_fann::training::*;
use ruv_fann::*;

fn main() {
    println!("🧪 Testing Optimizers with Improved Parameters");
    println!("===============================================");

    // Create XOR training data
    let train_data = TrainingData {
        inputs: vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        outputs: vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    };

    println!("🎯 Target: Learn XOR function");
    println!("📊 Training data:");
    for (i, (input, output)) in train_data
        .inputs
        .iter()
        .zip(train_data.outputs.iter())
        .enumerate()
    {
        println!("  {:?} -> {:?}", input, output);
    }

    // Test 1: SGD with higher learning rate
    println!("\n1️⃣ SGD with LR=1.0:");
    test_sgd_higher_lr(&train_data);

    // Test 2: Adam with higher learning rate
    println!("\n2️⃣ Adam with LR=0.1:");
    test_adam_higher_lr(&train_data);

    // Test 3: Simple linear problem (easier than XOR)
    println!("\n3️⃣ Linear problem test (y = x1 + x2):");
    test_linear_problem();
}

fn test_sgd_higher_lr(train_data: &TrainingData<f32>) {
    let mut network = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(4)
        .output_layer(1)
        .build();

    let mut trainer = IncrementalBackprop::new(1.0); // Much higher LR

    for epoch in 0..200 {
        match trainer.train_epoch(&mut network, train_data) {
            Ok(error) => {
                if epoch % 20 == 0 {
                    println!("  Epoch {:3}: Error = {:.6}", epoch, error);
                    test_network_outputs(&mut network, train_data);
                }
                if error < 0.01 {
                    println!("  ✅ Converged at epoch {} with error {:.6}", epoch, error);
                    return;
                }
            }
            Err(e) => {
                println!("  ❌ Training failed: {}", e);
                return;
            }
        }
    }
    println!("  ⚠️  Did not converge in 200 epochs");
}

fn test_adam_higher_lr(train_data: &TrainingData<f32>) {
    let mut network = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(4)
        .output_layer(1)
        .build();

    let mut trainer = Adam::new(0.1); // Higher LR for Adam

    for epoch in 0..200 {
        match trainer.train_epoch(&mut network, train_data) {
            Ok(error) => {
                if epoch % 20 == 0 {
                    println!("  Epoch {:3}: Error = {:.6}", epoch, error);
                    test_network_outputs(&mut network, train_data);
                }
                if error < 0.01 {
                    println!("  ✅ Converged at epoch {} with error {:.6}", epoch, error);
                    return;
                }
            }
            Err(e) => {
                println!("  ❌ Training failed: {}", e);
                return;
            }
        }
    }
    println!("  ⚠️  Did not converge in 200 epochs");
}

fn test_linear_problem() {
    // Simple linear problem: output = input1 + input2
    let train_data = TrainingData {
        inputs: vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        outputs: vec![
            vec![0.0], // 0 + 0 = 0
            vec![1.0], // 0 + 1 = 1
            vec![1.0], // 1 + 0 = 1
            vec![2.0], // 1 + 1 = 2
        ],
    };

    let mut network = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(3)
        .output_layer(1)
        .build();

    let mut trainer = Adam::new(0.01);

    for epoch in 0..100 {
        match trainer.train_epoch(&mut network, &train_data) {
            Ok(error) => {
                if epoch % 10 == 0 {
                    println!("  Epoch {:3}: Error = {:.6}", epoch, error);
                    test_network_outputs(&mut network, &train_data);
                }
                if error < 0.01 {
                    println!(
                        "  ✅ Linear problem converged at epoch {} with error {:.6}",
                        epoch, error
                    );
                    return;
                }
            }
            Err(e) => {
                println!("  ❌ Training failed: {}", e);
                return;
            }
        }
    }
    println!("  ⚠️  Did not converge in 100 epochs");
}

fn test_network_outputs(network: &mut Network<f32>, data: &TrainingData<f32>) {
    print!("    Outputs: ");
    for (input, expected) in data.inputs.iter().zip(data.outputs.iter()) {
        let output = network.run(input);
        print!("{:.2} ", output[0]);
    }
    print!("  (Expected: ");
    for expected in &data.outputs {
        print!("{:.2} ", expected[0]);
    }
    println!(")");
}
