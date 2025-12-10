use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, Optimizer, AdamW};
use std::collections::HashMap;
use crate::afm_detect::DetectionMode;

/// Dynamic threshold learning for adaptive anomaly detection
pub struct DynamicThresholdLearner {
    threshold_net: ThresholdNetwork,
    baseline_stats: BaselineStats,
    mode_thresholds: HashMap<DetectionMode, f32>,
    device: Device,
}

struct ThresholdNetwork {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

#[derive(Debug, Clone)]
struct BaselineStats {
    mean: f32,
    std: f32,
    percentiles: Vec<f32>,
    min_val: f32,
    max_val: f32,
}

impl DynamicThresholdLearner {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let threshold_net = ThresholdNetwork {
            fc1: candle_nn::linear(5, 32, vb.pp("thresh_fc1"))?,  // 5 features: mean, std, trend, volatility, time
            fc2: candle_nn::linear(32, 16, vb.pp("thresh_fc2"))?,
            fc3: candle_nn::linear(16, 1, vb.pp("thresh_fc3"))?,
        };
        
        let baseline_stats = BaselineStats {
            mean: 0.0,
            std: 1.0,
            percentiles: vec![0.5, 0.7, 0.9, 0.95, 0.99],
            min_val: 0.0,
            max_val: 1.0,
        };
        
        let mut mode_thresholds = HashMap::new();
        mode_thresholds.insert(DetectionMode::KpiKqi, 0.7);
        mode_thresholds.insert(DetectionMode::HardwareDegradation, 0.8);
        mode_thresholds.insert(DetectionMode::ThermalPower, 0.6);
        mode_thresholds.insert(DetectionMode::Combined, 0.75);
        
        Ok(Self {
            threshold_net,
            baseline_stats,
            mode_thresholds,
            device: vb.device().clone(),
        })
    }

    /// Get adaptive threshold based on current conditions
    pub fn get_threshold(&self, score: &f32, mode: DetectionMode) -> Result<f32> {
        // Get base threshold for mode
        let base_threshold = self.mode_thresholds.get(&mode).unwrap_or(&0.7);
        
        // Extract features for threshold prediction
        let features = self.extract_threshold_features(score, mode)?;
        
        // Predict threshold adjustment
        let adjustment = self.threshold_net.forward(&features)?;
        let adjustment_value = adjustment.to_scalar::<f32>()?;
        
        // Apply adjustment with bounds
        let adaptive_threshold = base_threshold + adjustment_value * 0.3;
        Ok(adaptive_threshold.clamp(0.1, 1.0))
    }

    /// Extract features for threshold prediction
    fn extract_threshold_features(&self, score: &f32, mode: DetectionMode) -> Result<Tensor> {
        let features = vec![
            *score,                           // Current score
            self.baseline_stats.mean,         // Baseline mean
            self.baseline_stats.std,          // Baseline std
            self.get_mode_factor(mode),       // Mode-specific factor
            self.get_time_factor()?,          // Time-based factor
        ];
        
        Tensor::from_slice(&features, &[1, 5], &self.device)
    }

    /// Get mode-specific adjustment factor
    fn get_mode_factor(&self, mode: DetectionMode) -> f32 {
        match mode {
            DetectionMode::KpiKqi => 0.0,
            DetectionMode::HardwareDegradation => 0.2,
            DetectionMode::ThermalPower => -0.1,
            DetectionMode::Combined => 0.1,
        }
    }

    /// Get time-based adjustment factor
    fn get_time_factor(&self) -> Result<f32> {
        // Simple time-based factor (could be more sophisticated)
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as f32;
        
        // Normalize to [0, 1] based on hour of day
        let hour_factor = (now % 86400.0) / 86400.0;
        Ok(hour_factor)
    }

    /// Update baseline statistics from normal data
    pub fn update_baseline(&mut self, data: &Tensor) -> Result<()> {
        let flat_data = data.flatten_all()?;
        let mean = flat_data.mean_all()?.to_scalar::<f32>()?;
        let std = flat_data.std_all()?.to_scalar::<f32>()?;
        let min_val = flat_data.min_all()?.to_scalar::<f32>()?;
        let max_val = flat_data.max_all()?.to_scalar::<f32>()?;
        
        // Calculate percentiles (simplified)
        let percentiles = self.calculate_percentiles(&flat_data)?;
        
        self.baseline_stats = BaselineStats {
            mean,
            std,
            percentiles,
            min_val,
            max_val,
        };
        
        Ok(())
    }

    /// Calculate percentiles from data
    fn calculate_percentiles(&self, data: &Tensor) -> Result<Vec<f32>> {
        // Simplified percentile calculation
        let mean = data.mean_all()?.to_scalar::<f32>()?;
        let std = data.std_all()?.to_scalar::<f32>()?;
        
        // Approximate percentiles using normal distribution
        let percentiles = vec![
            mean,                           // 50th percentile
            mean + 0.5 * std,              // ~70th percentile
            mean + 1.28 * std,             // ~90th percentile
            mean + 1.65 * std,             // ~95th percentile
            mean + 2.33 * std,             // ~99th percentile
        ];
        
        Ok(percentiles)
    }

    /// Adaptive threshold based on recent performance
    pub fn adaptive_threshold(&self, recent_scores: &[f32], mode: DetectionMode) -> Result<f32> {
        if recent_scores.is_empty() {
            return Ok(*self.mode_thresholds.get(&mode).unwrap_or(&0.7));
        }
        
        let mean_score = recent_scores.iter().sum::<f32>() / recent_scores.len() as f32;
        let std_score = (recent_scores.iter().map(|x| (x - mean_score).powi(2)).sum::<f32>() 
            / recent_scores.len() as f32).sqrt();
        
        // Adaptive threshold based on recent statistics
        let base_threshold = *self.mode_thresholds.get(&mode).unwrap_or(&0.7);
        let adaptive_threshold = base_threshold + 2.0 * std_score;
        
        Ok(adaptive_threshold.clamp(0.1, 1.0))
    }

    /// Get percentile-based threshold
    pub fn percentile_threshold(&self, percentile: f32, mode: DetectionMode) -> Result<f32> {
        let percentile_idx = ((percentile * 100.0) as usize).min(99);
        let base_threshold = *self.mode_thresholds.get(&mode).unwrap_or(&0.7);
        
        // Use percentile information if available
        if !self.baseline_stats.percentiles.is_empty() {
            let idx = (percentile_idx as f32 / 20.0) as usize;
            let idx = idx.min(self.baseline_stats.percentiles.len() - 1);
            let percentile_val = self.baseline_stats.percentiles[idx];
            Ok((base_threshold + percentile_val * 0.3).clamp(0.1, 1.0))
        } else {
            Ok(base_threshold)
        }
    }

    /// Train threshold network on labeled data
    pub fn train_threshold_network(
        &mut self,
        features: &Tensor,
        targets: &Tensor,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let mut opt = AdamW::new(vec![
            self.threshold_net.fc1.weight(),
            self.threshold_net.fc2.weight(),
            self.threshold_net.fc3.weight(),
        ], learning_rate)?;

        for epoch in 0..epochs {
            let predictions = self.threshold_net.forward(features)?;
            let loss = (predictions - targets)?.sqr()?.mean_all()?;
            
            opt.backward_step(&loss)?;
            
            if epoch % 10 == 0 {
                println!("Threshold network epoch {}: loss = {}", epoch, loss.to_scalar::<f32>()?);
            }
        }
        
        Ok(())
    }

    /// Update mode-specific threshold
    pub fn update_mode_threshold(&mut self, mode: DetectionMode, new_threshold: f32) {
        self.mode_thresholds.insert(mode, new_threshold.clamp(0.1, 1.0));
    }

    /// Get all mode thresholds
    pub fn get_mode_thresholds(&self) -> &HashMap<DetectionMode, f32> {
        &self.mode_thresholds
    }

    /// Calculate dynamic threshold based on system state
    pub fn system_state_threshold(
        &self,
        system_load: f32,
        error_rate: f32,
        time_since_last_anomaly: f32,
        mode: DetectionMode,
    ) -> Result<f32> {
        let base_threshold = *self.mode_thresholds.get(&mode).unwrap_or(&0.7);
        
        // Adjust threshold based on system state
        let load_adjustment = system_load * 0.2;  // Higher load = lower threshold
        let error_adjustment = error_rate * 0.3;  // Higher error rate = lower threshold
        let time_adjustment = (time_since_last_anomaly / 3600.0).min(0.2);  // Recent anomaly = lower threshold
        
        let adjusted_threshold = base_threshold - load_adjustment - error_adjustment - time_adjustment;
        Ok(adjusted_threshold.clamp(0.1, 1.0))
    }
}

impl Module for ThresholdNetwork {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.fc2.forward(&h)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.fc3.forward(&h)?;
        h.tanh()  // Output in [-1, 1] for threshold adjustment
    }
}

impl Default for BaselineStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
            percentiles: vec![0.5, 0.7, 0.9, 0.95, 0.99],
            min_val: 0.0,
            max_val: 1.0,
        }
    }
}

// Make DetectionMode hashable for HashMap
impl std::hash::Hash for DetectionMode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            DetectionMode::KpiKqi => 0.hash(state),
            DetectionMode::HardwareDegradation => 1.hash(state),
            DetectionMode::ThermalPower => 2.hash(state),
            DetectionMode::Combined => 3.hash(state),
        }
    }
}

impl std::cmp::Eq for DetectionMode {}

impl std::cmp::PartialEq for DetectionMode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DetectionMode::KpiKqi, DetectionMode::KpiKqi) => true,
            (DetectionMode::HardwareDegradation, DetectionMode::HardwareDegradation) => true,
            (DetectionMode::ThermalPower, DetectionMode::ThermalPower) => true,
            (DetectionMode::Combined, DetectionMode::Combined) => true,
            _ => false,
        }
    }
}