use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use std::collections::HashMap;

pub mod autoencoder;
pub mod vae;
pub mod ocsvm;
pub mod threshold;
pub mod contrastive;
pub mod predictor;

#[cfg(test)]
mod tests;

use autoencoder::AutoencoderDetector;
use vae::VariationalDetector;
use ocsvm::OneClassSVMDetector;
use threshold::DynamicThresholdLearner;
use contrastive::ContrastiveLearner;
use predictor::FailurePredictor;

/// Multi-modal anomaly detection system for AFM
pub struct AFMDetector {
    /// Autoencoder for reconstruction-based detection
    autoencoder: AutoencoderDetector,
    /// Variational autoencoder for probabilistic detection
    vae: VariationalDetector,
    /// One-class SVM in neural form
    ocsvm: OneClassSVMDetector,
    /// Dynamic threshold learner
    threshold_learner: DynamicThresholdLearner,
    /// Contrastive learning for representations
    contrastive: ContrastiveLearner,
    /// Failure predictor for 24-48 hour predictions
    predictor: FailurePredictor,
    device: Device,
}

/// Detection modes for different data types
#[derive(Debug, Clone, Copy)]
pub enum DetectionMode {
    /// KPI/KQI time series anomalies
    KpiKqi,
    /// Hardware degradation patterns
    HardwareDegradation,
    /// Temperature and power anomalies
    ThermalPower,
    /// Multi-modal combined detection
    Combined,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Anomaly score (0-1, higher means more anomalous)
    pub score: f32,
    /// Detection method contributions
    pub method_scores: HashMap<String, f32>,
    /// Predicted failure probability in next 24-48 hours
    pub failure_probability: Option<f32>,
    /// Anomaly type classification
    pub anomaly_type: Option<AnomalyType>,
    /// Confidence interval
    pub confidence: (f32, f32),
}

#[derive(Debug, Clone)]
pub enum AnomalyType {
    /// Sudden spike or drop
    Spike,
    /// Gradual drift from normal
    Drift,
    /// Periodic pattern disruption
    PatternBreak,
    /// Multi-metric correlation anomaly
    CorrelationAnomaly,
    /// Hardware degradation signature
    Degradation,
}

impl AFMDetector {
    /// Create new AFM detector with specified configuration
    pub fn new(
        input_dim: usize,
        latent_dim: usize,
        device: Device,
    ) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        Ok(Self {
            autoencoder: AutoencoderDetector::new(input_dim, latent_dim, vb.pp("ae"))?,
            vae: VariationalDetector::new(input_dim, latent_dim, vb.pp("vae"))?,
            ocsvm: OneClassSVMDetector::new(input_dim, latent_dim, vb.pp("ocsvm"))?,
            threshold_learner: DynamicThresholdLearner::new(vb.pp("threshold"))?,
            contrastive: ContrastiveLearner::new(latent_dim, vb.pp("contrastive"))?,
            predictor: FailurePredictor::new(input_dim, vb.pp("predictor"))?,
            device,
        })
    }

    /// Detect anomalies in input data
    pub fn detect(
        &self,
        input: &Tensor,
        mode: DetectionMode,
        history: Option<&Tensor>,
    ) -> Result<AnomalyResult> {
        let batch_size = input.dims()[0];
        
        // Get representations from contrastive learning
        let representations = self.contrastive.encode(input)?;
        
        // Collect scores from different methods
        let mut method_scores = HashMap::new();
        
        // Autoencoder reconstruction error
        let ae_score = self.autoencoder.anomaly_score(input)?;
        method_scores.insert("autoencoder".to_string(), ae_score);
        
        // VAE probabilistic score
        let vae_score = self.vae.anomaly_score(input)?;
        method_scores.insert("vae".to_string(), vae_score);
        
        // One-class SVM score
        let ocsvm_score = self.ocsvm.anomaly_score(&representations)?;
        method_scores.insert("ocsvm".to_string(), ocsvm_score);
        
        // Mode-specific processing
        let mode_weight = match mode {
            DetectionMode::KpiKqi => self.process_kpi_kqi(input)?,
            DetectionMode::HardwareDegradation => self.process_hardware(input, history)?,
            DetectionMode::ThermalPower => self.process_thermal_power(input)?,
            DetectionMode::Combined => 1.0,
        };
        
        // Combine scores with learned weights
        let combined_score = self.combine_scores(&method_scores, mode_weight)?;
        
        // Learn dynamic threshold
        let threshold = self.threshold_learner.get_threshold(&combined_score, mode)?;
        let normalized_score = (combined_score / threshold).min(1.0);
        
        // Predict failures if history is available
        let failure_probability = if let Some(hist) = history {
            Some(self.predictor.predict_failure(input, hist, 48)?)
        } else {
            None
        };
        
        // Classify anomaly type
        let anomaly_type = if normalized_score > 0.7 {
            Some(self.classify_anomaly(input, &method_scores)?)
        } else {
            None
        };
        
        // Calculate confidence interval
        let confidence = self.calculate_confidence(&method_scores);
        
        Ok(AnomalyResult {
            score: normalized_score,
            method_scores,
            failure_probability,
            anomaly_type,
            confidence,
        })
    }

    /// Process KPI/KQI specific features
    fn process_kpi_kqi(&self, input: &Tensor) -> Result<f32> {
        // Extract time series specific features
        let diff = input.narrow(1, 1, input.dims()[1] - 1)? 
            - input.narrow(1, 0, input.dims()[1] - 1)?;
        let volatility = diff.abs()?.mean_all()?.to_scalar::<f32>()?;
        
        // Weight based on volatility (more volatile = need higher threshold)
        Ok(1.0 + volatility * 0.5)
    }

    /// Process hardware degradation patterns
    fn process_hardware(&self, input: &Tensor, history: Option<&Tensor>) -> Result<f32> {
        if let Some(hist) = history {
            // Look for monotonic degradation trends
            let trend = self.calculate_trend(hist)?;
            let acceleration = self.calculate_acceleration(hist)?;
            
            // Higher weight for accelerating degradation
            Ok(1.0 + trend.abs() + acceleration * 2.0)
        } else {
            Ok(1.0)
        }
    }

    /// Process thermal/power anomalies
    fn process_thermal_power(&self, input: &Tensor) -> Result<f32> {
        // Check for thermal runaway patterns
        let max_val = input.max_all()?.to_scalar::<f32>()?;
        let mean_val = input.mean_all()?.to_scalar::<f32>()?;
        
        // Higher weight for extreme values
        let extremity = (max_val - mean_val) / mean_val.max(1e-6);
        Ok(1.0 + extremity.max(0.0))
    }

    /// Combine scores from different methods
    fn combine_scores(&self, scores: &HashMap<String, f32>, mode_weight: f32) -> Result<f32> {
        // Learned ensemble weights (could be made trainable)
        let weights = HashMap::from([
            ("autoencoder", 0.3),
            ("vae", 0.3),
            ("ocsvm", 0.4),
        ]);
        
        let weighted_sum: f32 = scores.iter()
            .map(|(method, score)| weights.get(method.as_str()).unwrap_or(&0.0) * score)
            .sum();
        
        Ok(weighted_sum * mode_weight)
    }

    /// Classify the type of anomaly detected
    fn classify_anomaly(
        &self,
        input: &Tensor,
        method_scores: &HashMap<String, f32>,
    ) -> Result<AnomalyType> {
        // Simple heuristic classification (could be replaced with learned classifier)
        let ae_score = method_scores.get("autoencoder").unwrap_or(&0.0);
        let vae_score = method_scores.get("vae").unwrap_or(&0.0);
        let ocsvm_score = method_scores.get("ocsvm").unwrap_or(&0.0);
        
        if ae_score > vae_score * 2.0 {
            Ok(AnomalyType::Spike)
        } else if ocsvm_score > ae_score * 1.5 {
            Ok(AnomalyType::PatternBreak)
        } else if (ae_score - vae_score).abs() < 0.1 {
            Ok(AnomalyType::Drift)
        } else {
            Ok(AnomalyType::CorrelationAnomaly)
        }
    }

    /// Calculate confidence interval for the detection
    fn calculate_confidence(&self, scores: &HashMap<String, f32>) -> (f32, f32) {
        let values: Vec<f32> = scores.values().copied().collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() 
            / values.len() as f32).sqrt();
        
        (mean - 2.0 * std, mean + 2.0 * std)
    }

    /// Calculate trend in time series
    fn calculate_trend(&self, data: &Tensor) -> Result<f32> {
        let n = data.dims()[1] as f32;
        let x: Vec<f32> = (0..data.dims()[1]).map(|i| i as f32).collect();
        let y = data.mean(0)?;
        
        // Simple linear regression
        let x_mean = x.iter().sum::<f32>() / n;
        let y_mean = y.mean_all()?.to_scalar::<f32>()?;
        
        let numerator: f32 = x.iter().zip(0..data.dims()[1])
            .map(|(xi, i)| {
                let yi = y.i(i).unwrap().to_scalar::<f32>().unwrap();
                (xi - x_mean) * (yi - y_mean)
            })
            .sum();
        
        let denominator: f32 = x.iter()
            .map(|xi| (xi - x_mean).powi(2))
            .sum();
        
        Ok(numerator / denominator.max(1e-6))
    }

    /// Calculate acceleration (second derivative) of trend
    fn calculate_acceleration(&self, data: &Tensor) -> Result<f32> {
        let diff1 = data.narrow(1, 1, data.dims()[1] - 1)? 
            - data.narrow(1, 0, data.dims()[1] - 1)?;
        let diff2 = diff1.narrow(1, 1, diff1.dims()[1] - 1)? 
            - diff1.narrow(1, 0, diff1.dims()[1] - 1)?;
        
        diff2.mean_all()?.to_scalar::<f32>()
    }

    /// Train the detector on normal data
    pub fn train_on_normal(
        &mut self,
        normal_data: &Tensor,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        // Train each component
        self.autoencoder.train(normal_data, epochs, learning_rate)?;
        self.vae.train(normal_data, epochs, learning_rate)?;
        self.ocsvm.train(normal_data, epochs, learning_rate)?;
        self.contrastive.train(normal_data, epochs, learning_rate)?;
        
        // Update threshold learner with normal data statistics
        self.threshold_learner.update_baseline(normal_data)?;
        
        Ok(())
    }

    /// Fine-tune on labeled anomalies
    pub fn finetune_on_anomalies(
        &mut self,
        anomaly_data: &Tensor,
        labels: &Tensor,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        // Use contrastive learning to improve representations
        self.contrastive.finetune_contrastive(anomaly_data, labels, epochs, learning_rate)?;
        
        // Update failure predictor with labeled data
        self.predictor.train_on_failures(anomaly_data, labels, epochs, learning_rate)?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_afm_detector() -> Result<()> {
        let device = Device::Cpu;
        let detector = AFMDetector::new(64, 16, device)?;
        
        // Test data
        let batch_size = 10;
        let input = Tensor::randn(0.0, 1.0, &[batch_size, 64], &device)?;
        
        // Test detection
        let result = detector.detect(&input, DetectionMode::Combined, None)?;
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(!result.method_scores.is_empty());
        
        Ok(())
    }
}