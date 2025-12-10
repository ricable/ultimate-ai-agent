// Handover Optimization Models
// Implements intelligent handover decision algorithms

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use crate::dtm_mobility::{UserMobilityProfile, MobilityState};

/// Handover optimization engine
pub struct HandoverOptimizer {
    /// Multi-criteria decision model
    mcdm_model: MCDMModel,
    
    /// Load balancing optimizer
    load_balancer: LoadBalancer,
    
    /// Handover history for learning
    handover_history: HashMap<String, VecDeque<HandoverEvent>>,
    
    /// Cell load statistics
    cell_load_stats: HashMap<String, CellLoadStats>,
    
    /// Optimization parameters
    params: HandoverOptimizationParams,
}

/// Multi-Criteria Decision Making model
#[derive(Debug, Clone)]
pub struct MCDMModel {
    /// Criteria weights
    pub criteria_weights: CriteriaWeights,
    
    /// Normalization factors
    pub normalization_factors: NormalizationFactors,
    
    /// Decision thresholds
    pub decision_thresholds: DecisionThresholds,
}

/// Load balancing optimizer
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    /// Target load threshold
    pub target_load_threshold: f64,
    
    /// Load balancing penalty
    pub load_penalty_factor: f64,
    
    /// Interference consideration
    pub interference_factor: f64,
}

/// Handover event record
#[derive(Debug, Clone)]
pub struct HandoverEvent {
    /// User ID
    pub user_id: String,
    
    /// Source cell
    pub source_cell: String,
    
    /// Target cell
    pub target_cell: String,
    
    /// Handover trigger
    pub trigger: HandoverTrigger,
    
    /// Handover outcome
    pub outcome: HandoverOutcome,
    
    /// Signal measurements
    pub measurements: SignalMeasurements,
    
    /// Timestamp
    pub timestamp: Instant,
    
    /// Execution time
    pub execution_time: Duration,
}

/// Handover trigger reasons
#[derive(Debug, Clone, PartialEq)]
pub enum HandoverTrigger {
    /// Signal strength below threshold
    SignalStrength,
    
    /// Quality degradation
    QualityDegradation,
    
    /// Load balancing
    LoadBalancing,
    
    /// Interference avoidance
    InterferenceAvoidance,
    
    /// Predictive handover
    Predictive,
}

/// Handover outcome
#[derive(Debug, Clone, PartialEq)]
pub enum HandoverOutcome {
    /// Successful handover
    Success,
    
    /// Handover failure
    Failure,
    
    /// Ping-pong handover
    PingPong,
    
    /// Handover cancelled
    Cancelled,
}

/// Signal measurements
#[derive(Debug, Clone)]
pub struct SignalMeasurements {
    /// RSRP (Reference Signal Received Power)
    pub rsrp: f64,
    
    /// RSRQ (Reference Signal Received Quality)
    pub rsrq: f64,
    
    /// SINR (Signal-to-Interference-plus-Noise Ratio)
    pub sinr: f64,
    
    /// CQI (Channel Quality Indicator)
    pub cqi: u8,
    
    /// Path loss
    pub path_loss: f64,
}

/// Cell load statistics
#[derive(Debug, Clone)]
pub struct CellLoadStats {
    /// Current load percentage
    pub current_load: f64,
    
    /// Average load
    pub average_load: f64,
    
    /// Peak load
    pub peak_load: f64,
    
    /// Number of connected users
    pub connected_users: u32,
    
    /// Available capacity
    pub available_capacity: f64,
    
    /// Interference level
    pub interference_level: f64,
}

/// Criteria weights for handover decision
#[derive(Debug, Clone)]
pub struct CriteriaWeights {
    /// Signal strength weight
    pub signal_strength: f64,
    
    /// Signal quality weight
    pub signal_quality: f64,
    
    /// Load balancing weight
    pub load_balancing: f64,
    
    /// Interference weight
    pub interference: f64,
    
    /// Mobility prediction weight
    pub mobility_prediction: f64,
    
    /// Handover cost weight
    pub handover_cost: f64,
}

/// Normalization factors
#[derive(Debug, Clone)]
pub struct NormalizationFactors {
    /// RSRP normalization
    pub rsrp_range: (f64, f64),
    
    /// RSRQ normalization
    pub rsrq_range: (f64, f64),
    
    /// SINR normalization
    pub sinr_range: (f64, f64),
    
    /// Load normalization
    pub load_range: (f64, f64),
}

/// Decision thresholds
#[derive(Debug, Clone)]
pub struct DecisionThresholds {
    /// Minimum signal strength for handover
    pub min_rsrp: f64,
    
    /// Minimum signal quality for handover
    pub min_rsrq: f64,
    
    /// Handover hysteresis
    pub hysteresis: f64,
    
    /// Time-to-trigger
    pub time_to_trigger: Duration,
}

/// Handover optimization parameters
#[derive(Debug, Clone)]
pub struct HandoverOptimizationParams {
    /// Learning rate for adaptive algorithms
    pub learning_rate: f64,
    
    /// Discount factor for future rewards
    pub discount_factor: f64,
    
    /// Exploration rate
    pub exploration_rate: f64,
    
    /// History window size
    pub history_window: usize,
}

impl HandoverOptimizer {
    /// Create new handover optimizer
    pub fn new() -> Self {
        Self {
            mcdm_model: MCDMModel::new(),
            load_balancer: LoadBalancer::new(),
            handover_history: HashMap::new(),
            cell_load_stats: HashMap::new(),
            params: HandoverOptimizationParams::default(),
        }
    }
    
    /// Select optimal cell for handover
    pub fn select_optimal_cell(
        &self,
        user_profile: &UserMobilityProfile,
        candidate_cells: Vec<(String, f64)>,
        predicted_trajectory: Option<Vec<(String, f64)>>,
    ) -> Result<String, String> {
        if candidate_cells.is_empty() {
            return Err("No candidate cells available".to_string());
        }
        
        // Calculate decision matrix
        let decision_matrix = self.build_decision_matrix(
            user_profile,
            &candidate_cells,
            &predicted_trajectory,
        )?;
        
        // Apply MCDM algorithm
        let scores = self.calculate_mcdm_scores(&decision_matrix)?;
        
        // Select best cell
        let best_cell_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or("No valid cell found")?
            .0;
        
        Ok(candidate_cells[best_cell_idx].0.clone())
    }
    
    /// Record handover event
    pub fn record_handover_event(&mut self, event: HandoverEvent) {
        let history = self.handover_history
            .entry(event.user_id.clone())
            .or_insert_with(VecDeque::new);
        
        history.push_back(event);
        
        // Keep only recent history
        if history.len() > self.params.history_window {
            history.pop_front();
        }
    }
    
    /// Update cell load statistics
    pub fn update_cell_load(&mut self, cell_id: String, load_stats: CellLoadStats) {
        self.cell_load_stats.insert(cell_id, load_stats);
    }
    
    /// Build decision matrix for MCDM
    fn build_decision_matrix(
        &self,
        user_profile: &UserMobilityProfile,
        candidate_cells: &[(String, f64)],
        predicted_trajectory: &Option<Vec<(String, f64)>>,
    ) -> Result<Vec<Vec<f64>>, String> {
        let mut matrix = Vec::new();
        
        for (cell_id, signal_strength) in candidate_cells {
            let mut criteria_values = Vec::new();
            
            // Signal strength criterion
            criteria_values.push(*signal_strength);
            
            // Signal quality criterion (simulated)
            let signal_quality = self.estimate_signal_quality(cell_id, user_profile);
            criteria_values.push(signal_quality);
            
            // Load balancing criterion
            let load_factor = self.calculate_load_factor(cell_id);
            criteria_values.push(1.0 - load_factor); // Invert for maximization
            
            // Interference criterion
            let interference_factor = self.estimate_interference(cell_id, user_profile);
            criteria_values.push(1.0 - interference_factor); // Invert for maximization
            
            // Mobility prediction criterion
            let mobility_factor = self.calculate_mobility_factor(
                cell_id,
                user_profile,
                predicted_trajectory,
            );
            criteria_values.push(mobility_factor);
            
            // Handover cost criterion
            let handover_cost = self.calculate_handover_cost(
                &user_profile.current_cell,
                cell_id,
                user_profile,
            );
            criteria_values.push(1.0 - handover_cost); // Invert for maximization
            
            matrix.push(criteria_values);
        }
        
        Ok(matrix)
    }
    
    /// Calculate MCDM scores using weighted sum
    fn calculate_mcdm_scores(&self, decision_matrix: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        if decision_matrix.is_empty() {
            return Err("Empty decision matrix".to_string());
        }
        
        let num_criteria = decision_matrix[0].len();
        let num_alternatives = decision_matrix.len();
        
        // Normalize the decision matrix
        let normalized_matrix = self.normalize_matrix(decision_matrix)?;
        
        // Calculate weighted scores
        let weights = self.get_criteria_weights_vector();
        let mut scores = vec![0.0; num_alternatives];
        
        for i in 0..num_alternatives {
            for j in 0..num_criteria {
                scores[i] += normalized_matrix[i][j] * weights[j];
            }
        }
        
        Ok(scores)
    }
    
    /// Normalize decision matrix
    fn normalize_matrix(&self, matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if matrix.is_empty() {
            return Err("Empty matrix".to_string());
        }
        
        let num_criteria = matrix[0].len();
        let num_alternatives = matrix.len();
        
        // Calculate column sums for normalization
        let mut column_sums = vec![0.0; num_criteria];
        for row in matrix {
            for (j, &value) in row.iter().enumerate() {
                column_sums[j] += value * value;
            }
        }
        
        // Normalize
        let mut normalized = vec![vec![0.0; num_criteria]; num_alternatives];
        for i in 0..num_alternatives {
            for j in 0..num_criteria {
                if column_sums[j] > 0.0 {
                    normalized[i][j] = matrix[i][j] / column_sums[j].sqrt();
                }
            }
        }
        
        Ok(normalized)
    }
    
    /// Get criteria weights as vector
    fn get_criteria_weights_vector(&self) -> Vec<f64> {
        vec![
            self.mcdm_model.criteria_weights.signal_strength,
            self.mcdm_model.criteria_weights.signal_quality,
            self.mcdm_model.criteria_weights.load_balancing,
            self.mcdm_model.criteria_weights.interference,
            self.mcdm_model.criteria_weights.mobility_prediction,
            self.mcdm_model.criteria_weights.handover_cost,
        ]
    }
    
    /// Estimate signal quality
    fn estimate_signal_quality(&self, cell_id: &str, user_profile: &UserMobilityProfile) -> f64 {
        // Simulate signal quality based on mobility state
        match user_profile.mobility_state {
            MobilityState::Stationary => 0.9,
            MobilityState::Walking => 0.8,
            MobilityState::Vehicular => 0.7,
            MobilityState::HighSpeed => 0.6,
        }
    }
    
    /// Calculate load factor
    fn calculate_load_factor(&self, cell_id: &str) -> f64 {
        if let Some(stats) = self.cell_load_stats.get(cell_id) {
            stats.current_load / 100.0
        } else {
            0.5 // Default load
        }
    }
    
    /// Estimate interference
    fn estimate_interference(&self, cell_id: &str, user_profile: &UserMobilityProfile) -> f64 {
        if let Some(stats) = self.cell_load_stats.get(cell_id) {
            stats.interference_level / 100.0
        } else {
            0.3 // Default interference
        }
    }
    
    /// Calculate mobility factor
    fn calculate_mobility_factor(
        &self,
        cell_id: &str,
        user_profile: &UserMobilityProfile,
        predicted_trajectory: &Option<Vec<(String, f64)>>,
    ) -> f64 {
        if let Some(trajectory) = predicted_trajectory {
            // Find if this cell is in predicted trajectory
            for (predicted_cell, probability) in trajectory {
                if predicted_cell == cell_id {
                    return *probability;
                }
            }
        }
        
        // Default mobility factor based on handover history
        let history = self.handover_history.get(&user_profile.user_id);
        if let Some(history) = history {
            let successful_handovers = history.iter()
                .filter(|event| event.target_cell == *cell_id 
                    && event.outcome == HandoverOutcome::Success)
                .count();
            
            if history.len() > 0 {
                successful_handovers as f64 / history.len() as f64
            } else {
                0.5
            }
        } else {
            0.5
        }
    }
    
    /// Calculate handover cost
    fn calculate_handover_cost(
        &self,
        source_cell: &str,
        target_cell: &str,
        user_profile: &UserMobilityProfile,
    ) -> f64 {
        // Base cost for handover
        let mut cost = 0.1;
        
        // Add cost based on mobility state
        cost += match user_profile.mobility_state {
            MobilityState::Stationary => 0.05,
            MobilityState::Walking => 0.1,
            MobilityState::Vehicular => 0.15,
            MobilityState::HighSpeed => 0.2,
        };
        
        // Add cost based on handover frequency
        if let Some(history) = self.handover_history.get(&user_profile.user_id) {
            let recent_handovers = history.iter()
                .filter(|event| event.timestamp.elapsed() < Duration::from_secs(300))
                .count();
            
            cost += recent_handovers as f64 * 0.05;
        }
        
        cost.min(1.0)
    }
    
    /// Learn from handover outcomes
    pub fn learn_from_outcome(&mut self, user_id: &str, outcome: HandoverOutcome) {
        // Update model parameters based on outcome
        match outcome {
            HandoverOutcome::Success => {
                // Positive reinforcement
                self.params.exploration_rate *= 0.99;
            }
            HandoverOutcome::Failure | HandoverOutcome::PingPong => {
                // Negative reinforcement
                self.params.exploration_rate *= 1.01;
            }
            HandoverOutcome::Cancelled => {
                // Neutral
            }
        }
        
        // Keep exploration rate within bounds
        self.params.exploration_rate = self.params.exploration_rate.clamp(0.01, 0.5);
    }
}

impl MCDMModel {
    /// Create new MCDM model
    pub fn new() -> Self {
        Self {
            criteria_weights: CriteriaWeights::default(),
            normalization_factors: NormalizationFactors::default(),
            decision_thresholds: DecisionThresholds::default(),
        }
    }
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new() -> Self {
        Self {
            target_load_threshold: 0.8,
            load_penalty_factor: 0.5,
            interference_factor: 0.3,
        }
    }
}

impl Default for CriteriaWeights {
    fn default() -> Self {
        Self {
            signal_strength: 0.3,
            signal_quality: 0.25,
            load_balancing: 0.2,
            interference: 0.1,
            mobility_prediction: 0.1,
            handover_cost: 0.05,
        }
    }
}

impl Default for NormalizationFactors {
    fn default() -> Self {
        Self {
            rsrp_range: (-140.0, -40.0),
            rsrq_range: (-20.0, -3.0),
            sinr_range: (-10.0, 30.0),
            load_range: (0.0, 100.0),
        }
    }
}

impl Default for DecisionThresholds {
    fn default() -> Self {
        Self {
            min_rsrp: -110.0,
            min_rsrq: -15.0,
            hysteresis: 3.0,
            time_to_trigger: Duration::from_millis(160),
        }
    }
}

impl Default for HandoverOptimizationParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            discount_factor: 0.9,
            exploration_rate: 0.1,
            history_window: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtm_mobility::HandoverStats;
    use std::collections::VecDeque;
    
    #[test]
    fn test_handover_optimizer_creation() {
        let optimizer = HandoverOptimizer::new();
        assert_eq!(optimizer.params.learning_rate, 0.01);
        assert_eq!(optimizer.params.discount_factor, 0.9);
    }
    
    #[test]
    fn test_decision_matrix_building() {
        let optimizer = HandoverOptimizer::new();
        let user_profile = UserMobilityProfile {
            user_id: "test_user".to_string(),
            current_cell: "cell_001".to_string(),
            mobility_state: MobilityState::Walking,
            speed_estimate: 1.5,
            trajectory_history: VecDeque::new(),
            handover_stats: HandoverStats {
                total_handovers: 10,
                successful_handovers: 8,
                failed_handovers: 2,
                ping_pong_handovers: 1,
                average_handover_time: Duration::from_millis(50),
            },
        };
        
        let candidates = vec![
            ("cell_002".to_string(), -80.0),
            ("cell_003".to_string(), -85.0),
        ];
        
        let matrix = optimizer.build_decision_matrix(
            &user_profile,
            &candidates,
            &None,
        );
        
        assert!(matrix.is_ok());
        let matrix = matrix.unwrap();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 6); // 6 criteria
    }
    
    #[test]
    fn test_mcdm_score_calculation() {
        let optimizer = HandoverOptimizer::new();
        let matrix = vec![
            vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            vec![0.7, 0.8, 0.5, 0.6, 0.3, 0.4],
        ];
        
        let scores = optimizer.calculate_mcdm_scores(&matrix);
        assert!(scores.is_ok());
        
        let scores = scores.unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > 0.0);
        assert!(scores[1] > 0.0);
    }
}