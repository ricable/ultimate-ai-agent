// Ensemble Neural Coordinator for RAN Intelligence Platform
// ML-Coordinator Agent - Neural Network Optimization

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkInfo {
    pub id: String,
    pub agent_id: String,
    pub agent_name: String,
    pub accuracy: f64,
    pub response_time_ms: f64,
    pub cognitive_pattern: String,
    pub specialized_domains: Vec<String>,
    pub last_trained: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    pub prediction: f64,
    pub confidence: f64,
    pub contributor_weights: HashMap<String, f64>,
    pub consensus_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeTransferItem {
    pub source_network: String,
    pub target_network: String,
    pub feature_vector: Vec<f64>,
    pub knowledge_type: String,
    pub transfer_score: f64,
}

pub struct EnsembleNeuralCoordinator {
    neural_networks: Arc<RwLock<HashMap<String, NeuralNetworkInfo>>>,
    ensemble_weights: Arc<RwLock<HashMap<String, f64>>>,
    knowledge_base: Arc<RwLock<Vec<KnowledgeTransferItem>>>,
    performance_metrics: Arc<RwLock<HashMap<String, f64>>>,
}

impl EnsembleNeuralCoordinator {
    pub fn new() -> Self {
        Self {
            neural_networks: Arc::new(RwLock::new(HashMap::new())),
            ensemble_weights: Arc::new(RwLock::new(HashMap::new())),
            knowledge_base: Arc::new(RwLock::new(Vec::new())),
            performance_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a neural network in the ensemble
    pub async fn register_network(&self, network_info: NeuralNetworkInfo) {
        let mut networks = self.neural_networks.write().await;
        let mut weights = self.ensemble_weights.write().await;
        
        // Calculate initial weight based on accuracy and cognitive diversity
        let base_weight = network_info.accuracy;
        let diversity_bonus = self.calculate_diversity_bonus(&network_info).await;
        let final_weight = base_weight * (1.0 + diversity_bonus);
        
        networks.insert(network_info.id.clone(), network_info.clone());
        weights.insert(network_info.id.clone(), final_weight);
        
        println!("🧠 Registered neural network: {} with weight: {:.4}", 
                 network_info.agent_name, final_weight);
    }

    /// Calculate diversity bonus based on cognitive pattern uniqueness
    async fn calculate_diversity_bonus(&self, network_info: &NeuralNetworkInfo) -> f64 {
        let networks = self.neural_networks.read().await;
        let mut pattern_counts = HashMap::new();
        
        for (_, net) in networks.iter() {
            *pattern_counts.entry(net.cognitive_pattern.clone()).or_insert(0) += 1;
        }
        
        let pattern_frequency = pattern_counts.get(&network_info.cognitive_pattern).unwrap_or(&0);
        
        // Higher bonus for less common patterns
        match *pattern_frequency {
            0 => 0.2,  // 20% bonus for unique patterns
            1 => 0.1,  // 10% bonus for rare patterns
            2 => 0.05, // 5% bonus for uncommon patterns
            _ => 0.0,  // No bonus for common patterns
        }
    }

    /// Generate ensemble prediction from all networks
    pub async fn ensemble_predict(&self, input_data: &[f64]) -> EnsemblePrediction {
        let networks = self.neural_networks.read().await;
        let weights = self.ensemble_weights.read().await;
        
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut contributor_weights = HashMap::new();
        let mut individual_predictions = Vec::new();
        
        for (network_id, network_info) in networks.iter() {
            if let Some(weight) = weights.get(network_id) {
                // Simulate neural network prediction (in real implementation, call actual network)
                let prediction = self.simulate_prediction(network_info, input_data).await;
                individual_predictions.push(prediction);
                
                weighted_sum += prediction * weight;
                weight_sum += weight;
                contributor_weights.insert(network_info.agent_name.clone(), *weight);
            }
        }
        
        let ensemble_prediction = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };
        
        let consensus_level = self.calculate_consensus(&individual_predictions);
        let confidence = self.calculate_confidence(&individual_predictions, consensus_level);
        
        EnsemblePrediction {
            prediction: ensemble_prediction,
            confidence,
            contributor_weights,
            consensus_level,
        }
    }

    /// Simulate neural network prediction (placeholder for actual FANN integration)
    async fn simulate_prediction(&self, network_info: &NeuralNetworkInfo, _input_data: &[f64]) -> f64 {
        // In real implementation, this would call the actual neural network
        // For now, simulate based on network characteristics
        let base_prediction = 0.5 + (network_info.accuracy - 0.5) * 0.8;
        
        // Add some variation based on cognitive pattern
        let pattern_variation = match network_info.cognitive_pattern.as_str() {
            "convergent" => 0.1,
            "divergent" => -0.05,
            "lateral" => 0.15,
            "systems" => 0.08,
            "critical" => -0.02,
            _ => 0.0,
        };
        
        (base_prediction + pattern_variation).max(0.0).min(1.0)
    }

    /// Calculate consensus level among predictions
    fn calculate_consensus(&self, predictions: &[f64]) -> f64 {
        if predictions.len() < 2 {
            return 1.0;
        }
        
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        // Higher consensus when variance is low
        (1.0 - variance).max(0.0)
    }

    /// Calculate prediction confidence
    fn calculate_confidence(&self, predictions: &[f64], consensus_level: f64) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }
        
        let mean_prediction = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let prediction_strength = (mean_prediction - 0.5).abs() * 2.0; // 0-1 scale
        
        // Confidence is combination of prediction strength and consensus
        (prediction_strength * 0.7 + consensus_level * 0.3).min(1.0)
    }

    /// Update ensemble weights based on performance
    pub async fn update_weights(&self, performance_feedback: HashMap<String, f64>) {
        let mut weights = self.ensemble_weights.write().await;
        let networks = self.neural_networks.read().await;
        
        for (network_id, performance_score) in performance_feedback {
            if let Some(current_weight) = weights.get_mut(&network_id) {
                if let Some(network_info) = networks.get(&network_id) {
                    // Adaptive weight update based on recent performance
                    let learning_rate = 0.1;
                    let performance_adjustment = (performance_score - 0.5) * learning_rate;
                    
                    *current_weight = (*current_weight + performance_adjustment)
                        .max(0.01)  // Minimum weight
                        .min(2.0);  // Maximum weight
                    
                    println!("📊 Updated weight for {}: {:.4} (performance: {:.4})", 
                             network_info.agent_name, *current_weight, performance_score);
                }
            }
        }
    }

    /// Transfer knowledge between networks
    pub async fn transfer_knowledge(&self, source_id: &str, target_id: &str, knowledge_type: &str) {
        let networks = self.neural_networks.read().await;
        let mut knowledge_base = self.knowledge_base.write().await;
        
        if let (Some(source_net), Some(target_net)) = (networks.get(source_id), networks.get(target_id)) {
            // Simulate knowledge extraction (in real implementation, extract actual features)
            let feature_vector = self.extract_features(source_net).await;
            let transfer_score = self.calculate_transfer_compatibility(source_net, target_net).await;
            
            if transfer_score > 0.5 {
                let knowledge_item = KnowledgeTransferItem {
                    source_network: source_id.to_string(),
                    target_network: target_id.to_string(),
                    feature_vector,
                    knowledge_type: knowledge_type.to_string(),
                    transfer_score,
                };
                
                knowledge_base.push(knowledge_item);
                
                println!("🔄 Knowledge transferred from {} to {} (score: {:.4})", 
                         source_net.agent_name, target_net.agent_name, transfer_score);
            }
        }
    }

    /// Extract features from a neural network (placeholder)
    async fn extract_features(&self, network_info: &NeuralNetworkInfo) -> Vec<f64> {
        // In real implementation, extract learned features from the network
        // For now, create representative features based on network characteristics
        vec![
            network_info.accuracy,
            network_info.response_time_ms / 1000.0,
            match network_info.cognitive_pattern.as_str() {
                "convergent" => 0.2,
                "divergent" => 0.4,
                "lateral" => 0.6,
                "systems" => 0.8,
                "critical" => 1.0,
                _ => 0.0,
            },
        ]
    }

    /// Calculate knowledge transfer compatibility
    async fn calculate_transfer_compatibility(&self, source: &NeuralNetworkInfo, target: &NeuralNetworkInfo) -> f64 {
        // Check domain overlap
        let domain_overlap = source.specialized_domains.iter()
            .filter(|domain| target.specialized_domains.contains(domain))
            .count() as f64 / source.specialized_domains.len().max(1) as f64;
        
        // Check cognitive pattern compatibility
        let pattern_compatibility = if source.cognitive_pattern == target.cognitive_pattern {
            1.0
        } else {
            0.5  // Different patterns can still share some knowledge
        };
        
        // Check performance compatibility (similar performance networks transfer better)
        let performance_diff = (source.accuracy - target.accuracy).abs();
        let performance_compatibility = (1.0 - performance_diff).max(0.0);
        
        // Weighted combination
        (domain_overlap * 0.5 + pattern_compatibility * 0.3 + performance_compatibility * 0.2)
    }

    /// Get ensemble performance metrics
    pub async fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let networks = self.neural_networks.read().await;
        let weights = self.ensemble_weights.read().await;
        let knowledge_base = self.knowledge_base.read().await;
        
        let mut metrics = HashMap::new();
        
        // Overall ensemble accuracy (weighted average)
        let total_accuracy = networks.iter()
            .map(|(id, net)| net.accuracy * weights.get(id).unwrap_or(&1.0))
            .sum::<f64>() / networks.len() as f64;
        
        metrics.insert("ensemble_accuracy".to_string(), total_accuracy);
        metrics.insert("total_networks".to_string(), networks.len() as f64);
        metrics.insert("knowledge_transfers".to_string(), knowledge_base.len() as f64);
        
        // Average response time
        let avg_response_time = networks.values()
            .map(|net| net.response_time_ms)
            .sum::<f64>() / networks.len() as f64;
        metrics.insert("avg_response_time_ms".to_string(), avg_response_time);
        
        metrics
    }

    /// Generate optimization recommendations
    pub async fn generate_recommendations(&self) -> Vec<String> {
        let networks = self.neural_networks.read().await;
        let weights = self.ensemble_weights.read().await;
        let mut recommendations = Vec::new();
        
        // Find underperforming networks
        for (id, network) in networks.iter() {
            if network.accuracy < 0.85 {
                recommendations.push(format!(
                    "🔧 {} needs retraining (accuracy: {:.2}%)", 
                    network.agent_name, network.accuracy * 100.0
                ));
            }
            
            if network.response_time_ms > 500.0 {
                recommendations.push(format!(
                    "⚡ {} needs optimization (response time: {:.1}ms)", 
                    network.agent_name, network.response_time_ms
                ));
            }
        }
        
        // Check for cognitive pattern imbalance
        let mut pattern_counts = HashMap::new();
        for network in networks.values() {
            *pattern_counts.entry(network.cognitive_pattern.clone()).or_insert(0) += 1;
        }
        
        if pattern_counts.values().max().unwrap_or(&0) > &(networks.len() / 2) {
            recommendations.push("🧠 Consider adding more cognitive diversity to the ensemble".to_string());
        }
        
        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ensemble_coordinator() {
        let coordinator = EnsembleNeuralCoordinator::new();
        
        // Register test networks
        let network1 = NeuralNetworkInfo {
            id: "nn1".to_string(),
            agent_id: "agent1".to_string(),
            agent_name: "Test Agent 1".to_string(),
            accuracy: 0.95,
            response_time_ms: 200.0,
            cognitive_pattern: "convergent".to_string(),
            specialized_domains: vec!["handover".to_string()],
            last_trained: "2025-07-05".to_string(),
        };
        
        coordinator.register_network(network1).await;
        
        let input_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let prediction = coordinator.ensemble_predict(&input_data).await;
        
        assert!(prediction.prediction >= 0.0 && prediction.prediction <= 1.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }
}