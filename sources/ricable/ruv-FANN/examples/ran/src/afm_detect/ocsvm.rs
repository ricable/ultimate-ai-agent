use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, Optimizer, AdamW};

/// One-Class SVM implemented as neural network for anomaly detection
pub struct OneClassSVMDetector {
    feature_extractor: FeatureExtractor,
    classifier: OneClassClassifier,
    device: Device,
    center: Option<Tensor>,
    radius: f32,
}

struct FeatureExtractor {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

struct OneClassClassifier {
    fc: Linear,
}

impl OneClassSVMDetector {
    pub fn new(input_dim: usize, latent_dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (input_dim + latent_dim) / 2;
        
        let feature_extractor = FeatureExtractor {
            fc1: candle_nn::linear(input_dim, hidden_dim * 2, vb.pp("feat_fc1"))?,
            fc2: candle_nn::linear(hidden_dim * 2, hidden_dim, vb.pp("feat_fc2"))?,
            fc3: candle_nn::linear(hidden_dim, latent_dim, vb.pp("feat_fc3"))?,
        };
        
        let classifier = OneClassClassifier {
            fc: candle_nn::linear(latent_dim, 1, vb.pp("clf_fc"))?,
        };
        
        Ok(Self {
            feature_extractor,
            classifier,
            device: vb.device().clone(),
            center: None,
            radius: 1.0,
        })
    }

    /// Extract features from input
    pub fn extract_features(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.feature_extractor.fc1.forward(x)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.feature_extractor.fc2.forward(&h)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.feature_extractor.fc3.forward(&h)?;
        // L2 normalization
        let norm = h.sqr()?.sum_keepdim(1)?.sqrt()?;
        h.div(&norm)
    }

    /// Forward pass through one-class classifier
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let features = self.extract_features(x)?;
        self.classifier.fc.forward(&features)
    }

    /// Calculate distance from center (for SVDD-style loss)
    pub fn distance_from_center(&self, x: &Tensor) -> Result<Tensor> {
        let features = self.extract_features(x)?;
        
        match &self.center {
            Some(center) => {
                let diff = (features - center)?;
                diff.sqr()?.sum_keepdim(1)?.sqrt()
            }
            None => {
                // If no center, compute distance from origin
                features.sqr()?.sum_keepdim(1)?.sqrt()
            }
        }
    }

    /// Calculate anomaly score
    pub fn anomaly_score(&self, x: &Tensor) -> Result<f32> {
        let distance = self.distance_from_center(x)?;
        let mean_distance = distance.mean_all()?;
        
        // Normalize by radius
        let score = mean_distance.to_scalar::<f32>()? / self.radius;
        Ok(score.max(0.0).min(1.0))
    }

    /// SVDD loss function
    pub fn svdd_loss(&self, x: &Tensor, nu: f32) -> Result<Tensor> {
        let features = self.extract_features(x)?;
        let batch_size = features.dims()[0];
        
        // Update center if not set
        if self.center.is_none() {
            // Initialize center as mean of features
            let center = features.mean(0)?;
            // Note: In a real implementation, we'd need mutable access
            // This is a simplified version
        }
        
        let distances = self.distance_from_center(x)?;
        
        // SVDD objective: minimize radius + penalty for outliers
        let radius_loss = distances.mean_all()?;
        
        // Penalty for points outside hypersphere
        let penalty = distances.clamp(0.0, f32::INFINITY)?.mean_all()?;
        
        radius_loss + nu * penalty
    }

    /// Deep SVDD loss (without pre-training)
    pub fn deep_svdd_loss(&self, x: &Tensor, nu: f32) -> Result<Tensor> {
        let features = self.extract_features(x)?;
        
        // One-class classification loss
        let scores = self.classifier.fc.forward(&features)?;
        
        // Margin loss: push normal data to positive side
        let hinge_loss = (1.0 - scores)?.clamp(0.0, f32::INFINITY)?.mean_all()?;
        
        // Regularization to prevent collapse
        let reg_loss = features.sqr()?.mean_all()?;
        
        hinge_loss + nu * reg_loss
    }

    /// Train the one-class SVM
    pub fn train(
        &mut self,
        data: &Tensor,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let mut opt = AdamW::new(vec![
            self.feature_extractor.fc1.weight(),
            self.feature_extractor.fc2.weight(),
            self.feature_extractor.fc3.weight(),
            self.classifier.fc.weight(),
        ], learning_rate)?;

        // Initialize center
        let features = self.extract_features(data)?;
        let center = features.mean(0)?;
        self.center = Some(center);

        for epoch in 0..epochs {
            let loss = self.deep_svdd_loss(data, 0.1)?;
            
            opt.backward_step(&loss)?;
            
            if epoch % 10 == 0 {
                println!("One-Class SVM epoch {}: loss = {}", epoch, loss.to_scalar::<f32>()?);
            }
        }
        
        // Update radius based on training data
        let distances = self.distance_from_center(data)?;
        self.radius = distances.mean_all()?.to_scalar::<f32>()? + 
                     distances.std_all()?.to_scalar::<f32>()?;
        
        Ok(())
    }

    /// Get decision scores for samples
    pub fn decision_scores(&self, x: &Tensor) -> Result<Tensor> {
        let features = self.extract_features(x)?;
        self.classifier.fc.forward(&features)
    }

    /// Predict outliers (1 for outlier, 0 for inlier)
    pub fn predict(&self, x: &Tensor, threshold: f32) -> Result<Tensor> {
        let scores = self.anomaly_score(x)?;
        let predictions = if scores > threshold { 1.0 } else { 0.0 };
        Ok(Tensor::from_scalar(predictions, &self.device)?)
    }

    /// Update center adaptively
    pub fn update_center(&mut self, x: &Tensor, momentum: f32) -> Result<()> {
        let features = self.extract_features(x)?;
        let new_center = features.mean(0)?;
        
        match &self.center {
            Some(old_center) => {
                let updated_center = (old_center * momentum + new_center * (1.0 - momentum))?;
                self.center = Some(updated_center);
            }
            None => {
                self.center = Some(new_center);
            }
        }
        
        Ok(())
    }

    /// Get hypersphere parameters
    pub fn get_hypersphere_params(&self) -> (Option<&Tensor>, f32) {
        (self.center.as_ref(), self.radius)
    }

    /// Calculate novelty score (for online detection)
    pub fn novelty_score(&self, x: &Tensor) -> Result<f32> {
        let features = self.extract_features(x)?;
        let score = self.classifier.fc.forward(&features)?;
        
        // Convert to probability using sigmoid
        let prob = score.sigmoid()?;
        prob.mean_all()?.to_scalar::<f32>()
    }
}