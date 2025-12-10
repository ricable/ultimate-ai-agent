//! Demand Prediction Module
//! 
//! This module implements demand prediction capabilities for RAN optimization.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandPredictor {
    pub prediction_horizon: u32,
    pub historical_data: VecDeque<f32>,
    pub seasonal_patterns: Vec<f32>,
    pub confidence_level: f32,
    pub max_history_size: usize,
    pub seasonal_period: usize,
}

impl DemandPredictor {
    pub fn new(horizon: u32) -> Self {
        Self {
            prediction_horizon: horizon,
            historical_data: VecDeque::new(),
            seasonal_patterns: Vec::new(),
            confidence_level: 0.8,
            max_history_size: 1000,
            seasonal_period: 24, // 24 hours for daily patterns
        }
    }
    
    pub fn add_data_point(&mut self, value: f32) {
        self.historical_data.push_back(value);
        
        // Keep only recent data
        if self.historical_data.len() > self.max_history_size {
            self.historical_data.pop_front();
        }
        
        // Update seasonal patterns if we have enough data
        if self.historical_data.len() >= self.seasonal_period * 2 {
            self.update_seasonal_patterns();
        }
    }
    
    pub fn add_historical_data(&mut self, data: Vec<f32>) {
        for value in data {
            self.add_data_point(value);
        }
    }
    
    fn update_seasonal_patterns(&mut self) {
        let data_len = self.historical_data.len();
        if data_len < self.seasonal_period * 2 {
            return;
        }
        
        // Calculate seasonal averages
        let mut seasonal_sums = vec![0.0; self.seasonal_period];
        let mut seasonal_counts = vec![0; self.seasonal_period];
        
        for (i, &value) in self.historical_data.iter().enumerate() {
            let seasonal_index = i % self.seasonal_period;
            seasonal_sums[seasonal_index] += value;
            seasonal_counts[seasonal_index] += 1;
        }
        
        // Calculate averages
        self.seasonal_patterns = seasonal_sums.iter()
            .zip(seasonal_counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
            .collect();
    }
    
    pub fn predict_demand(&self, steps_ahead: u32) -> Result<Vec<f32>, String> {
        if self.historical_data.is_empty() {
            return Err("No historical data available for prediction".to_string());
        }
        
        let mut predictions = Vec::new();
        
        for step in 0..steps_ahead {
            let prediction = self.predict_single_step(step)?;
            predictions.push(prediction);
        }
        
        Ok(predictions)
    }
    
    fn predict_single_step(&self, step: u32) -> Result<f32, String> {
        if self.historical_data.is_empty() {
            return Err("No historical data available".to_string());
        }
        
        // Simple prediction combining trend and seasonal patterns
        let trend = self.calculate_trend();
        let seasonal = self.get_seasonal_component(step);
        
        // Get recent average as baseline
        let recent_window = std::cmp::min(self.historical_data.len(), 10);
        let recent_avg: f32 = self.historical_data.iter()
            .rev()
            .take(recent_window)
            .sum::<f32>() / recent_window as f32;
        
        let prediction = recent_avg + trend * step as f32 + seasonal;
        
        Ok(prediction.max(0.0)) // Ensure non-negative
    }
    
    fn calculate_trend(&self) -> f32 {
        let data_len = self.historical_data.len();
        if data_len < 2 {
            return 0.0;
        }
        
        // Simple linear trend calculation
        let window_size = std::cmp::min(data_len, 20);
        let recent_data: Vec<f32> = self.historical_data.iter()
            .rev()
            .take(window_size)
            .cloned()
            .collect();
        
        if recent_data.len() < 2 {
            return 0.0;
        }
        
        let n = recent_data.len() as f32;
        let sum_x = (0..recent_data.len()).map(|i| i as f32).sum::<f32>();
        let sum_y = recent_data.iter().sum::<f32>();
        let sum_xy = recent_data.iter().enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum::<f32>();
        let sum_x2 = (0..recent_data.len()).map(|i| (i as f32).powi(2)).sum::<f32>();
        
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }
    
    fn get_seasonal_component(&self, step: u32) -> f32 {
        if self.seasonal_patterns.is_empty() {
            return 0.0;
        }
        
        let seasonal_index = (step as usize) % self.seasonal_patterns.len();
        let overall_avg = self.seasonal_patterns.iter().sum::<f32>() / self.seasonal_patterns.len() as f32;
        
        // Return deviation from average
        self.seasonal_patterns[seasonal_index] - overall_avg
    }
    
    pub fn get_confidence_interval(&self, prediction: f32) -> (f32, f32) {
        if self.historical_data.is_empty() {
            return (prediction, prediction);
        }
        
        // Calculate standard deviation of recent data
        let recent_window = std::cmp::min(self.historical_data.len(), 50);
        let recent_data: Vec<f32> = self.historical_data.iter()
            .rev()
            .take(recent_window)
            .cloned()
            .collect();
        
        let mean = recent_data.iter().sum::<f32>() / recent_data.len() as f32;
        let variance = recent_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / recent_data.len() as f32;
        let std_dev = variance.sqrt();
        
        // Z-score for confidence level (approximate)
        let z_score = match self.confidence_level {
            level if level >= 0.95 => 1.96,
            level if level >= 0.9 => 1.645,
            level if level >= 0.8 => 1.282,
            _ => 1.0,
        };
        
        let margin = z_score * std_dev;
        (prediction - margin, prediction + margin)
    }
    
    pub fn evaluate_prediction_accuracy(&self, actual_values: &[f32], predicted_values: &[f32]) -> f32 {
        if actual_values.len() != predicted_values.len() || actual_values.is_empty() {
            return 0.0;
        }
        
        let mae: f32 = actual_values.iter()
            .zip(predicted_values.iter())
            .map(|(&actual, &predicted)| (actual - predicted).abs())
            .sum::<f32>() / actual_values.len() as f32;
        
        let mean_actual = actual_values.iter().sum::<f32>() / actual_values.len() as f32;
        
        // Return accuracy as percentage (higher is better)
        if mean_actual > 0.0 {
            (1.0 - mae / mean_actual).max(0.0) * 100.0
        } else {
            0.0
        }
    }
}

impl Default for DemandPredictor {
    fn default() -> Self {
        Self::new(24) // Default 24-hour prediction horizon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_demand_predictor_creation() {
        let predictor = DemandPredictor::new(12);
        assert_eq!(predictor.prediction_horizon, 12);
        assert!(predictor.historical_data.is_empty());
    }
    
    #[test]
    fn test_adding_data() {
        let mut predictor = DemandPredictor::new(24);
        predictor.add_data_point(10.0);
        predictor.add_data_point(15.0);
        
        assert_eq!(predictor.historical_data.len(), 2);
        assert_eq!(predictor.historical_data[0], 10.0);
        assert_eq!(predictor.historical_data[1], 15.0);
    }
    
    #[test]
    fn test_prediction_with_data() {
        let mut predictor = DemandPredictor::new(24);
        let test_data = vec![10.0, 12.0, 11.0, 13.0, 14.0];
        predictor.add_historical_data(test_data);
        
        let predictions = predictor.predict_demand(3).unwrap();
        assert_eq!(predictions.len(), 3);
        assert!(predictions.iter().all(|&p| p >= 0.0));
    }
}