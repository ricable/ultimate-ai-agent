use ndarray::{Array1, Array2, ArrayView1};
use std::collections::VecDeque;
use std::time::Instant;

/// Custom activation function for power curve modeling
#[inline]
fn power_activation(x: f32) -> f32 {
    // Sigmoid-like function optimized for power curves
    // Models the non-linear relationship between load and power consumption
    let alpha = 2.5;
    let beta = 0.3;
    beta + (1.0 - beta) / (1.0 + (-alpha * x).exp())
}

/// Derivative of power activation for backpropagation
#[inline]
fn power_activation_derivative(x: f32) -> f32 {
    let alpha = 2.5;
    let beta = 0.3;
    let exp_neg_alpha_x = (-alpha * x).exp();
    let denominator = (1.0 + exp_neg_alpha_x).powi(2);
    alpha * (1.0 - beta) * exp_neg_alpha_x / denominator
}

/// Quantized weight representation for edge deployment
#[derive(Clone, Copy, Debug)]
struct QuantizedWeight {
    value: i8,
    scale: f32,
}

impl QuantizedWeight {
    fn new(weight: f32, scale: f32) -> Self {
        let quantized = (weight / scale).round().clamp(-127.0, 127.0) as i8;
        Self {
            value: quantized,
            scale,
        }
    }

    #[inline]
    fn dequantize(&self) -> f32 {
        self.value as f32 * self.scale
    }
}

/// Lightweight neural network layer with quantization support
struct PowerAwareLayer {
    weights: Vec<QuantizedWeight>,
    bias: Vec<f32>,
    input_size: usize,
    output_size: usize,
    pruning_mask: Vec<bool>,
}

impl PowerAwareLayer {
    fn new(input_size: usize, output_size: usize, scale: f32) -> Self {
        let total_weights = input_size * output_size;
        let weights = vec![QuantizedWeight::new(0.0, scale); total_weights];
        let bias = vec![0.0; output_size];
        let pruning_mask = vec![true; total_weights];
        
        Self {
            weights,
            bias,
            input_size,
            output_size,
            pruning_mask,
        }
    }

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(self.output_size);
        
        for out_idx in 0..self.output_size {
            let mut sum = self.bias[out_idx];
            
            for in_idx in 0..self.input_size {
                let weight_idx = out_idx * self.input_size + in_idx;
                if self.pruning_mask[weight_idx] {
                    sum += input[in_idx] * self.weights[weight_idx].dequantize();
                }
            }
            
            output[out_idx] = power_activation(sum);
        }
        
        output
    }

    /// Prune weights below threshold for model compression
    fn prune_weights(&mut self, threshold: f32) {
        for (idx, weight) in self.weights.iter().enumerate() {
            if weight.dequantize().abs() < threshold {
                self.pruning_mask[idx] = false;
            }
        }
    }
}

/// Energy consumption prediction network
pub struct EnergyPredictionNet {
    feature_layer: PowerAwareLayer,
    hidden_layer: PowerAwareLayer,
    output_layer: PowerAwareLayer,
    feature_buffer: VecDeque<Array1<f32>>,
    buffer_size: usize,
}

impl EnergyPredictionNet {
    pub fn new(feature_dim: usize, hidden_dim: usize) -> Self {
        let scale = 0.05; // Quantization scale factor
        
        Self {
            feature_layer: PowerAwareLayer::new(feature_dim, hidden_dim, scale),
            hidden_layer: PowerAwareLayer::new(hidden_dim, hidden_dim / 2, scale),
            output_layer: PowerAwareLayer::new(hidden_dim / 2, 1, scale),
            feature_buffer: VecDeque::with_capacity(60), // 1-minute buffer at 1Hz
            buffer_size: 60,
        }
    }

    pub fn predict(&mut self, features: Array1<f32>) -> f32 {
        let start = Instant::now();
        
        // Add to buffer for temporal features
        if self.feature_buffer.len() >= self.buffer_size {
            self.feature_buffer.pop_front();
        }
        self.feature_buffer.push_back(features.clone());
        
        // Forward pass through network
        let h1 = self.feature_layer.forward(&features.view());
        let h2 = self.hidden_layer.forward(&h1.view());
        let output = self.output_layer.forward(&h2.view());
        
        let inference_time = start.elapsed();
        debug_assert!(inference_time.as_millis() < 10, "Inference took {:?}, exceeding 10ms threshold", inference_time);
        
        output[0]
    }

    /// Compress model by pruning small weights
    pub fn compress(&mut self, pruning_threshold: f32) {
        self.feature_layer.prune_weights(pruning_threshold);
        self.hidden_layer.prune_weights(pruning_threshold);
        self.output_layer.prune_weights(pruning_threshold);
    }
}

/// Feature extractor for power states
pub struct PowerStateFeatures {
    cpu_utilization: f32,
    memory_pressure: f32,
    io_wait_ratio: f32,
    thermal_state: f32,
    frequency_scaling: f32,
    voltage_level: f32,
    cache_miss_rate: f32,
    network_activity: f32,
}

impl PowerStateFeatures {
    pub fn new() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_pressure: 0.0,
            io_wait_ratio: 0.0,
            thermal_state: 0.0,
            frequency_scaling: 1.0,
            voltage_level: 1.0,
            cache_miss_rate: 0.0,
            network_activity: 0.0,
        }
    }

    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(vec![
            self.cpu_utilization,
            self.memory_pressure,
            self.io_wait_ratio,
            self.thermal_state,
            self.frequency_scaling,
            self.voltage_level,
            self.cache_miss_rate,
            self.network_activity,
        ])
    }

    /// Extract features optimized for power prediction
    pub fn extract_power_features(&self) -> Array1<f32> {
        // Non-linear feature engineering for power curves
        let cpu_power = power_activation(self.cpu_utilization);
        let memory_power = self.memory_pressure * 0.3; // Memory has lower power impact
        let io_power = self.io_wait_ratio * 0.2; // IO waiting reduces power
        let thermal_penalty = (self.thermal_state - 0.7).max(0.0) * 2.0; // Thermal throttling
        
        Array1::from_vec(vec![
            cpu_power,
            memory_power,
            io_power,
            thermal_penalty,
            self.frequency_scaling.powi(2), // Quadratic relationship with frequency
            self.voltage_level.powi(2), // Quadratic relationship with voltage
            self.cache_miss_rate * 0.5,
            self.network_activity * 0.4,
        ])
    }
}

/// Decision tree node for scheduler selection
#[derive(Clone)]
struct DecisionNode {
    feature_index: usize,
    threshold: f32,
    left_child: Option<Box<DecisionNode>>,
    right_child: Option<Box<DecisionNode>>,
    decision: Option<SchedulerDecision>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SchedulerDecision {
    MicroSleep,
    LowEnergyScheduler,
    BalancedMode,
}

/// Decision tree for Micro Sleep vs Low Energy Scheduler
pub struct SchedulerDecisionTree {
    root: DecisionNode,
    feature_importance: Vec<f32>,
}

impl SchedulerDecisionTree {
    pub fn new() -> Self {
        // Build a simple but effective decision tree
        let root = DecisionNode {
            feature_index: 0, // CPU utilization
            threshold: 0.3,
            left_child: Some(Box::new(DecisionNode {
                feature_index: 2, // IO wait ratio
                threshold: 0.5,
                left_child: Some(Box::new(DecisionNode {
                    feature_index: 0,
                    threshold: 0.0,
                    left_child: None,
                    right_child: None,
                    decision: Some(SchedulerDecision::MicroSleep),
                })),
                right_child: Some(Box::new(DecisionNode {
                    feature_index: 0,
                    threshold: 0.0,
                    left_child: None,
                    right_child: None,
                    decision: Some(SchedulerDecision::LowEnergyScheduler),
                })),
                decision: None,
            })),
            right_child: Some(Box::new(DecisionNode {
                feature_index: 3, // Thermal state
                threshold: 0.8,
                left_child: Some(Box::new(DecisionNode {
                    feature_index: 0,
                    threshold: 0.0,
                    left_child: None,
                    right_child: None,
                    decision: Some(SchedulerDecision::BalancedMode),
                })),
                right_child: Some(Box::new(DecisionNode {
                    feature_index: 0,
                    threshold: 0.0,
                    left_child: None,
                    right_child: None,
                    decision: Some(SchedulerDecision::LowEnergyScheduler),
                })),
                decision: None,
            })),
            decision: None,
        };

        let feature_importance = vec![0.4, 0.1, 0.25, 0.15, 0.05, 0.05, 0.0, 0.0];

        Self {
            root,
            feature_importance,
        }
    }

    pub fn decide(&self, features: &PowerStateFeatures) -> SchedulerDecision {
        let feature_array = features.to_array();
        self.traverse_tree(&self.root, &feature_array.view())
    }

    fn traverse_tree(&self, node: &DecisionNode, features: &ArrayView1<f32>) -> SchedulerDecision {
        if let Some(decision) = node.decision {
            return decision;
        }

        let feature_value = features[node.feature_index];
        
        if feature_value <= node.threshold {
            if let Some(ref left) = node.left_child {
                return self.traverse_tree(left, features);
            }
        } else {
            if let Some(ref right) = node.right_child {
                return self.traverse_tree(right, features);
            }
        }

        SchedulerDecision::BalancedMode
    }

    pub fn get_feature_importance(&self) -> &[f32] {
        &self.feature_importance
    }
}

/// Inter-arrival time prediction network
pub struct InterArrivalPredictor {
    lstm_hidden: Array1<f32>,
    lstm_cell: Array1<f32>,
    hidden_size: usize,
    input_gate: PowerAwareLayer,
    forget_gate: PowerAwareLayer,
    output_gate: PowerAwareLayer,
    cell_gate: PowerAwareLayer,
    output_layer: PowerAwareLayer,
    time_buffer: VecDeque<f32>,
}

impl InterArrivalPredictor {
    pub fn new(hidden_size: usize) -> Self {
        let scale = 0.05;
        
        Self {
            lstm_hidden: Array1::zeros(hidden_size),
            lstm_cell: Array1::zeros(hidden_size),
            hidden_size,
            input_gate: PowerAwareLayer::new(1 + hidden_size, hidden_size, scale),
            forget_gate: PowerAwareLayer::new(1 + hidden_size, hidden_size, scale),
            output_gate: PowerAwareLayer::new(1 + hidden_size, hidden_size, scale),
            cell_gate: PowerAwareLayer::new(1 + hidden_size, hidden_size, scale),
            output_layer: PowerAwareLayer::new(hidden_size, 1, scale),
            time_buffer: VecDeque::with_capacity(100),
        }
    }

    pub fn predict_next_arrival(&mut self, current_time: f32) -> f32 {
        let start = Instant::now();
        
        // Add to buffer
        if self.time_buffer.len() >= 100 {
            self.time_buffer.pop_front();
        }
        self.time_buffer.push_back(current_time);
        
        // Prepare input
        let mut input = Array1::zeros(1 + self.hidden_size);
        input[0] = current_time;
        for i in 0..self.hidden_size {
            input[i + 1] = self.lstm_hidden[i];
        }
        
        // LSTM forward pass
        let i_gate = self.input_gate.forward(&input.view());
        let f_gate = self.forget_gate.forward(&input.view());
        let o_gate = self.output_gate.forward(&input.view());
        let c_tilde = self.cell_gate.forward(&input.view());
        
        // Update cell state
        for i in 0..self.hidden_size {
            self.lstm_cell[i] = f_gate[i] * self.lstm_cell[i] + i_gate[i] * c_tilde[i];
            self.lstm_hidden[i] = o_gate[i] * power_activation(self.lstm_cell[i]);
        }
        
        // Output prediction
        let output = self.output_layer.forward(&self.lstm_hidden.view());
        
        let inference_time = start.elapsed();
        debug_assert!(inference_time.as_millis() < 10, "Inference took {:?}, exceeding 10ms threshold", inference_time);
        
        // Return predicted inter-arrival time
        output[0].max(0.001) // Ensure positive time
    }

    pub fn compress(&mut self, pruning_threshold: f32) {
        self.input_gate.prune_weights(pruning_threshold);
        self.forget_gate.prune_weights(pruning_threshold);
        self.output_gate.prune_weights(pruning_threshold);
        self.cell_gate.prune_weights(pruning_threshold);
        self.output_layer.prune_weights(pruning_threshold);
    }
}

/// Main DTM Power Manager
pub struct DtmPowerManager {
    energy_predictor: EnergyPredictionNet,
    scheduler_tree: SchedulerDecisionTree,
    arrival_predictor: InterArrivalPredictor,
    current_features: PowerStateFeatures,
    prediction_history: VecDeque<f32>,
}

impl DtmPowerManager {
    pub fn new() -> Self {
        Self {
            energy_predictor: EnergyPredictionNet::new(8, 16),
            scheduler_tree: SchedulerDecisionTree::new(),
            arrival_predictor: InterArrivalPredictor::new(8),
            current_features: PowerStateFeatures::new(),
            prediction_history: VecDeque::with_capacity(60),
        }
    }

    pub fn update_features(&mut self, features: PowerStateFeatures) {
        self.current_features = features;
    }

    pub fn predict_energy_consumption(&mut self) -> f32 {
        let features = self.current_features.extract_power_features();
        let prediction = self.energy_predictor.predict(features);
        
        // Store prediction history
        if self.prediction_history.len() >= 60 {
            self.prediction_history.pop_front();
        }
        self.prediction_history.push_back(prediction);
        
        prediction
    }

    pub fn get_scheduler_decision(&self) -> SchedulerDecision {
        self.scheduler_tree.decide(&self.current_features)
    }

    pub fn predict_next_event_time(&mut self, current_time: f32) -> f32 {
        self.arrival_predictor.predict_next_arrival(current_time)
    }

    pub fn compress_models(&mut self, pruning_threshold: f32) {
        self.energy_predictor.compress(pruning_threshold);
        self.arrival_predictor.compress(pruning_threshold);
    }

    pub fn get_model_stats(&self) -> ModelStats {
        ModelStats {
            energy_predictor_size: std::mem::size_of_val(&self.energy_predictor),
            arrival_predictor_size: std::mem::size_of_val(&self.arrival_predictor),
            decision_tree_size: std::mem::size_of_val(&self.scheduler_tree),
            total_parameters: self.count_parameters(),
        }
    }

    fn count_parameters(&self) -> usize {
        // Simplified parameter counting
        let energy_params = 8 * 16 + 16 * 8 + 8 * 1;
        let arrival_params = 4 * (9 * 8) + 8 * 1;
        energy_params + arrival_params
    }
}

#[derive(Debug)]
pub struct ModelStats {
    pub energy_predictor_size: usize,
    pub arrival_predictor_size: usize,
    pub decision_tree_size: usize,
    pub total_parameters: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_activation() {
        assert!((power_activation(0.0) - 0.8).abs() < 0.1);
        assert!(power_activation(1.0) > 0.9);
        assert!(power_activation(-1.0) < 0.5);
    }

    #[test]
    fn test_quantization() {
        let weight = QuantizedWeight::new(0.523, 0.01);
        let dequantized = weight.dequantize();
        assert!((dequantized - 0.52).abs() < 0.01);
    }

    #[test]
    fn test_energy_prediction() {
        let mut predictor = EnergyPredictionNet::new(8, 16);
        let features = Array1::from_vec(vec![0.5; 8]);
        let prediction = predictor.predict(features);
        assert!(prediction >= 0.0 && prediction <= 1.0);
    }

    #[test]
    fn test_scheduler_decision() {
        let tree = SchedulerDecisionTree::new();
        
        let mut features = PowerStateFeatures::new();
        features.cpu_utilization = 0.1;
        features.io_wait_ratio = 0.7;
        assert_eq!(tree.decide(&features), SchedulerDecision::LowEnergyScheduler);
        
        features.cpu_utilization = 0.8;
        features.thermal_state = 0.9;
        assert_eq!(tree.decide(&features), SchedulerDecision::LowEnergyScheduler);
    }

    #[test]
    fn test_inference_time() {
        let mut manager = DtmPowerManager::new();
        let start = Instant::now();
        
        for _ in 0..100 {
            manager.predict_energy_consumption();
        }
        
        let avg_time = start.elapsed().as_micros() / 100;
        assert!(avg_time < 10000, "Average inference time {} µs exceeds 10ms", avg_time);
    }
}