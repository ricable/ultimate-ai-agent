//! Fixed neural training library with minimal dependencies

#[derive(Debug)]
pub struct SimpleNeuralNetwork {
    pub name: String,
    pub layers: Vec<usize>,
    pub training_error: f64,
}

impl SimpleNeuralNetwork {
    pub fn new(name: String, layers: Vec<usize>) -> Self {
        Self {
            name,
            layers,
            training_error: 1.0,
        }
    }
    
    pub fn train(&mut self, epochs: i32) -> f64 {
        // Simulate training
        for _epoch in 0..epochs {
            self.training_error *= 0.99; // Decay error
        }
        self.training_error
    }
}

pub fn create_demo_networks() -> Vec<SimpleNeuralNetwork> {
    vec![
        SimpleNeuralNetwork::new("Shallow".to_string(), vec![21, 32, 1]),
        SimpleNeuralNetwork::new("Deep".to_string(), vec![21, 64, 32, 16, 1]),
        SimpleNeuralNetwork::new("Wide".to_string(), vec![21, 128, 64, 1]),
    ]
}