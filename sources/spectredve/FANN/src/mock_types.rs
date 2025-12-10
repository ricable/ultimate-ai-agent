//! Mock types for testing I/O functionality

/// Mock network structure for testing
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MockNetwork {
    pub num_layers: usize,
    pub learning_rate: f32,
    pub connection_rate: f32,
    pub layer_sizes: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Mock training data structure for testing
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MockTrainingData {
    pub num_data: usize,
    pub num_input: usize,
    pub num_output: usize,
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
}
