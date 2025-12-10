// Trajectory Prediction Networks
// Implements neural networks for predicting user movement patterns

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use crate::dtm_mobility::{MobilityState, CellVisit};
use crate::dtm_mobility::graph_attention::CellTransitionGraph;

/// Trajectory prediction network
pub struct TrajectoryPredictor {
    /// LSTM network for sequence prediction
    lstm_network: LSTMNetwork,
    
    /// Attention mechanism for trajectory features
    attention_layer: AttentionLayer,
    
    /// Historical trajectory data
    trajectory_history: HashMap<String, VecDeque<TrajectoryPoint>>,
    
    /// Cell transition patterns
    transition_patterns: HashMap<String, Vec<TransitionPattern>>,
    
    /// Model parameters
    model_params: TrajectoryModelParams,
}

/// LSTM network for trajectory prediction
#[derive(Debug, Clone)]
pub struct LSTMNetwork {
    /// Input dimension
    input_dim: usize,
    
    /// Hidden dimension
    hidden_dim: usize,
    
    /// Number of layers
    num_layers: usize,
    
    /// Weights for input gate
    weight_input: Vec<Vec<f64>>,
    
    /// Weights for forget gate
    weight_forget: Vec<Vec<f64>>,
    
    /// Weights for output gate
    weight_output: Vec<Vec<f64>>,
    
    /// Weights for cell state
    weight_cell: Vec<Vec<f64>>,
    
    /// Hidden state
    hidden_state: Vec<f64>,
    
    /// Cell state
    cell_state: Vec<f64>,
}

/// Attention layer for trajectory features
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    /// Query weights
    query_weights: Vec<Vec<f64>>,
    
    /// Key weights
    key_weights: Vec<Vec<f64>>,
    
    /// Value weights
    value_weights: Vec<Vec<f64>>,
    
    /// Attention dimension
    attention_dim: usize,
}

/// Trajectory point with spatial and temporal features
#[derive(Debug, Clone)]
pub struct TrajectoryPoint {
    /// Cell ID
    pub cell_id: String,
    
    /// Location coordinates
    pub location: (f64, f64),
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Signal strength
    pub signal_strength: f64,
    
    /// Speed estimate
    pub speed: f64,
    
    /// Direction (radians)
    pub direction: f64,
    
    /// Dwell time in cell
    pub dwell_time: f64,
}

/// Transition pattern between cells
#[derive(Debug, Clone)]
pub struct TransitionPattern {
    /// Source cell
    pub from_cell: String,
    
    /// Target cell
    pub to_cell: String,
    
    /// Transition probability
    pub probability: f64,
    
    /// Average transition time
    pub avg_time: f64,
    
    /// Mobility state dependency
    pub mobility_state: Option<MobilityState>,
    
    /// Time-of-day dependency
    pub time_pattern: Option<TimePattern>,
}

/// Time-based pattern
#[derive(Debug, Clone)]
pub struct TimePattern {
    /// Hour of day (0-23)
    pub hour: u8,
    
    /// Day of week (0-6)
    pub day_of_week: u8,
    
    /// Pattern strength
    pub strength: f64,
}

/// Model parameters
#[derive(Debug, Clone)]
pub struct TrajectoryModelParams {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Sequence length for prediction
    pub sequence_length: usize,
    
    /// Prediction horizon
    pub prediction_horizon: usize,
    
    /// Temperature for softmax
    pub temperature: f64,
    
    /// Regularization coefficient
    pub reg_coeff: f64,
}

impl TrajectoryPredictor {
    /// Create new trajectory predictor
    pub fn new() -> Self {
        let model_params = TrajectoryModelParams {
            learning_rate: 0.001,
            sequence_length: 10,
            prediction_horizon: 5,
            temperature: 1.0,
            reg_coeff: 0.01,
        };
        
        Self {
            lstm_network: LSTMNetwork::new(64, 128, 2),
            attention_layer: AttentionLayer::new(128, 64),
            trajectory_history: HashMap::new(),
            transition_patterns: HashMap::new(),
            model_params,
        }
    }
    
    /// Add trajectory point for user
    pub fn add_trajectory_point(&mut self, user_id: &str, point: TrajectoryPoint) {
        let history = self.trajectory_history
            .entry(user_id.to_string())
            .or_insert_with(VecDeque::new);
        
        history.push_back(point);
        
        // Keep only recent history
        if history.len() > self.model_params.sequence_length * 2 {
            history.pop_front();
        }
    }
    
    /// Predict next cells for user
    pub fn predict_next_cells(
        &self,
        user_id: &str,
        current_cell: &str,
        mobility_state: MobilityState,
        cell_graph: &CellTransitionGraph,
    ) -> Result<Vec<(String, f64)>, String> {
        // Get user's trajectory history
        let history = self.trajectory_history.get(user_id)
            .ok_or("No trajectory history for user")?;
        
        if history.is_empty() {
            return Ok(vec![]);
        }
        
        // Extract features from trajectory
        let features = self.extract_trajectory_features(history);
        
        // Use LSTM to predict next trajectory points
        let lstm_output = self.lstm_network.forward(&features)?;
        
        // Apply attention mechanism
        let attention_output = self.attention_layer.forward(&lstm_output)?;
        
        // Get candidate cells from graph
        let candidate_cells = cell_graph.get_neighbor_cells(current_cell);
        
        // Calculate probabilities for each candidate
        let mut predictions = Vec::new();
        
        for cell_id in candidate_cells {
            let probability = self.calculate_transition_probability(
                user_id,
                current_cell,
                &cell_id,
                mobility_state,
                &attention_output,
            )?;
            
            if probability > 0.01 { // Threshold for relevance
                predictions.push((cell_id, probability));
            }
        }
        
        // Sort by probability (descending)
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top predictions
        predictions.truncate(self.model_params.prediction_horizon);
        
        Ok(predictions)
    }
    
    /// Train the model with trajectory data
    pub fn train(&mut self, training_data: Vec<(String, Vec<TrajectoryPoint>)>) -> Result<(), String> {
        for (user_id, trajectory) in training_data {
            // Add trajectory points
            for point in trajectory {
                self.add_trajectory_point(&user_id, point);
            }
            
            // Update transition patterns
            self.update_transition_patterns(&user_id)?;
        }
        
        // Train LSTM network
        self.train_lstm_network()?;
        
        Ok(())
    }
    
    /// Extract features from trajectory history
    fn extract_trajectory_features(&self, history: &VecDeque<TrajectoryPoint>) -> Vec<f64> {
        let mut features = Vec::new();
        
        for point in history {
            // Spatial features
            features.push(point.location.0); // Latitude
            features.push(point.location.1); // Longitude
            
            // Temporal features
            features.push(point.timestamp as f64);
            features.push(point.dwell_time);
            
            // Signal features
            features.push(point.signal_strength);
            
            // Kinematic features
            features.push(point.speed);
            features.push(point.direction);
        }
        
        features
    }
    
    /// Calculate transition probability
    fn calculate_transition_probability(
        &self,
        user_id: &str,
        from_cell: &str,
        to_cell: &str,
        mobility_state: MobilityState,
        attention_output: &[f64],
    ) -> Result<f64, String> {
        // Base probability from transition patterns
        let base_prob = self.get_transition_pattern_probability(from_cell, to_cell, mobility_state);
        
        // Personalization factor from attention output
        let personalization_factor = self.calculate_personalization_factor(
            user_id,
            to_cell,
            attention_output,
        );
        
        // Combine probabilities
        let combined_prob = base_prob * personalization_factor;
        
        // Apply temperature scaling
        let scaled_prob = (combined_prob / self.model_params.temperature).exp();
        
        Ok(scaled_prob)
    }
    
    /// Get transition pattern probability
    fn get_transition_pattern_probability(
        &self,
        from_cell: &str,
        to_cell: &str,
        mobility_state: MobilityState,
    ) -> f64 {
        let patterns = self.transition_patterns.get(from_cell);
        
        if let Some(patterns) = patterns {
            for pattern in patterns {
                if pattern.to_cell == to_cell {
                    // Check mobility state match
                    if let Some(pattern_state) = pattern.mobility_state {
                        if pattern_state == mobility_state {
                            return pattern.probability;
                        }
                    } else {
                        return pattern.probability;
                    }
                }
            }
        }
        
        0.1 // Default probability for unknown transitions
    }
    
    /// Calculate personalization factor
    fn calculate_personalization_factor(
        &self,
        user_id: &str,
        to_cell: &str,
        attention_output: &[f64],
    ) -> f64 {
        // Use attention output to calculate personalization
        // This is a simplified implementation
        if attention_output.is_empty() {
            return 1.0;
        }
        
        let factor = attention_output.iter().sum::<f64>() / attention_output.len() as f64;
        (factor + 1.0) / 2.0 // Normalize to [0.5, 1.0]
    }
    
    /// Update transition patterns
    fn update_transition_patterns(&mut self, user_id: &str) -> Result<(), String> {
        let history = self.trajectory_history.get(user_id)
            .ok_or("No trajectory history")?;
        
        if history.len() < 2 {
            return Ok(());
        }
        
        // Extract transitions from history
        let points: Vec<&TrajectoryPoint> = history.iter().collect();
        
        for i in 1..points.len() {
            let from_cell = &points[i-1].cell_id;
            let to_cell = &points[i].cell_id;
            
            if from_cell != to_cell {
                let transition_time = points[i].timestamp - points[i-1].timestamp;
                
                // Update or create pattern
                let patterns = self.transition_patterns
                    .entry(from_cell.clone())
                    .or_insert_with(Vec::new);
                
                // Find existing pattern or create new one
                let mut found = false;
                for pattern in patterns.iter_mut() {
                    if pattern.to_cell == *to_cell {
                        // Update existing pattern
                        pattern.avg_time = (pattern.avg_time + transition_time as f64) / 2.0;
                        found = true;
                        break;
                    }
                }
                
                if !found {
                    // Create new pattern
                    patterns.push(TransitionPattern {
                        from_cell: from_cell.clone(),
                        to_cell: to_cell.clone(),
                        probability: 0.1,
                        avg_time: transition_time as f64,
                        mobility_state: None,
                        time_pattern: None,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Train LSTM network
    fn train_lstm_network(&mut self) -> Result<(), String> {
        // This is a placeholder for LSTM training
        // In a real implementation, this would use proper backpropagation
        
        for (_, history) in &self.trajectory_history {
            let features = self.extract_trajectory_features(history);
            let _output = self.lstm_network.forward(&features)?;
            
            // Update weights (simplified)
            // In practice, this would use gradients and backpropagation
        }
        
        Ok(())
    }
}

impl LSTMNetwork {
    /// Create new LSTM network
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let mut rng = fastrand::Rng::new();
        
        // Initialize weights randomly
        let weight_input = (0..num_layers)
            .map(|_| (0..hidden_dim * input_dim).map(|_| rng.f64() - 0.5).collect())
            .collect();
        
        let weight_forget = (0..num_layers)
            .map(|_| (0..hidden_dim * hidden_dim).map(|_| rng.f64() - 0.5).collect())
            .collect();
        
        let weight_output = (0..num_layers)
            .map(|_| (0..hidden_dim * hidden_dim).map(|_| rng.f64() - 0.5).collect())
            .collect();
        
        let weight_cell = (0..num_layers)
            .map(|_| (0..hidden_dim * hidden_dim).map(|_| rng.f64() - 0.5).collect())
            .collect();
        
        Self {
            input_dim,
            hidden_dim,
            num_layers,
            weight_input,
            weight_forget,
            weight_output,
            weight_cell,
            hidden_state: vec![0.0; hidden_dim],
            cell_state: vec![0.0; hidden_dim],
        }
    }
    
    /// Forward pass through LSTM
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>, String> {
        if input.len() != self.input_dim {
            return Err("Input dimension mismatch".to_string());
        }
        
        let mut hidden = self.hidden_state.clone();
        let mut cell = self.cell_state.clone();
        
        // Simplified LSTM forward pass
        for layer in 0..self.num_layers {
            // Input gate
            let input_gate = self.sigmoid(&self.linear_transform(
                input,
                &self.weight_input[layer],
                &hidden,
            ));
            
            // Forget gate
            let forget_gate = self.sigmoid(&self.linear_transform(
                input,
                &self.weight_forget[layer],
                &hidden,
            ));
            
            // Output gate
            let output_gate = self.sigmoid(&self.linear_transform(
                input,
                &self.weight_output[layer],
                &hidden,
            ));
            
            // Cell state update
            let cell_candidate = self.tanh(&self.linear_transform(
                input,
                &self.weight_cell[layer],
                &hidden,
            ));
            
            // Update cell state
            for i in 0..self.hidden_dim {
                cell[i] = forget_gate[i] * cell[i] + input_gate[i] * cell_candidate[i];
            }
            
            // Update hidden state
            let cell_tanh = self.tanh(&cell);
            for i in 0..self.hidden_dim {
                hidden[i] = output_gate[i] * cell_tanh[i];
            }
        }
        
        Ok(hidden)
    }
    
    /// Linear transformation
    fn linear_transform(&self, input: &[f64], weights: &[f64], hidden: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.hidden_dim];
        
        // Simplified linear transformation
        for i in 0..self.hidden_dim {
            for j in 0..self.input_dim {
                output[i] += input[j] * weights[i * self.input_dim + j];
            }
            for j in 0..self.hidden_dim {
                output[i] += hidden[j] * weights[i * self.hidden_dim + j];
            }
        }
        
        output
    }
    
    /// Sigmoid activation
    fn sigmoid(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&val| 1.0 / (1.0 + (-val).exp())).collect()
    }
    
    /// Tanh activation
    fn tanh(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&val| val.tanh()).collect()
    }
}

impl AttentionLayer {
    /// Create new attention layer
    pub fn new(input_dim: usize, attention_dim: usize) -> Self {
        let mut rng = fastrand::Rng::new();
        
        // Initialize weights randomly
        let query_weights = (0..attention_dim)
            .map(|_| (0..input_dim).map(|_| rng.f64() - 0.5).collect())
            .collect();
        
        let key_weights = (0..attention_dim)
            .map(|_| (0..input_dim).map(|_| rng.f64() - 0.5).collect())
            .collect();
        
        let value_weights = (0..attention_dim)
            .map(|_| (0..input_dim).map(|_| rng.f64() - 0.5).collect())
            .collect();
        
        Self {
            query_weights,
            key_weights,
            value_weights,
            attention_dim,
        }
    }
    
    /// Forward pass through attention layer
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>, String> {
        if input.is_empty() {
            return Ok(vec![]);
        }
        
        // Calculate queries, keys, and values
        let queries = self.apply_weights(input, &self.query_weights);
        let keys = self.apply_weights(input, &self.key_weights);
        let values = self.apply_weights(input, &self.value_weights);
        
        // Calculate attention scores
        let attention_scores = self.calculate_attention_scores(&queries, &keys);
        
        // Apply attention to values
        let output = self.apply_attention(&attention_scores, &values);
        
        Ok(output)
    }
    
    /// Apply weights to input
    fn apply_weights(&self, input: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
        let mut output = vec![0.0; weights.len()];
        
        for i in 0..weights.len() {
            for j in 0..input.len().min(weights[i].len()) {
                output[i] += input[j] * weights[i][j];
            }
        }
        
        output
    }
    
    /// Calculate attention scores
    fn calculate_attention_scores(&self, queries: &[f64], keys: &[f64]) -> Vec<f64> {
        let mut scores = vec![0.0; queries.len()];
        
        for i in 0..queries.len() {
            for j in 0..keys.len().min(queries.len()) {
                scores[i] += queries[i] * keys[j];
            }
        }
        
        // Apply softmax
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        
        if sum_exp > 0.0 {
            exp_scores.iter().map(|&s| s / sum_exp).collect()
        } else {
            vec![1.0 / scores.len() as f64; scores.len()]
        }
    }
    
    /// Apply attention to values
    fn apply_attention(&self, attention_scores: &[f64], values: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; values.len()];
        
        for i in 0..values.len() {
            for j in 0..attention_scores.len().min(values.len()) {
                output[i] += attention_scores[j] * values[i];
            }
        }
        
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trajectory_predictor_creation() {
        let predictor = TrajectoryPredictor::new();
        assert_eq!(predictor.model_params.learning_rate, 0.001);
        assert_eq!(predictor.model_params.sequence_length, 10);
    }
    
    #[test]
    fn test_lstm_forward() {
        let lstm = LSTMNetwork::new(10, 20, 2);
        let input = vec![0.5; 10];
        
        let result = lstm.forward(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), 20);
    }
    
    #[test]
    fn test_attention_layer() {
        let attention = AttentionLayer::new(10, 5);
        let input = vec![0.5; 10];
        
        let result = attention.forward(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), 10);
    }
}