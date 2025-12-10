use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, Optimizer, AdamW};

/// Contrastive learning for better anomaly detection representations
pub struct ContrastiveLearner {
    encoder: ContrastiveEncoder,
    projection_head: ProjectionHead,
    device: Device,
    temperature: f32,
}

struct ContrastiveEncoder {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

struct ProjectionHead {
    fc1: Linear,
    fc2: Linear,
}

impl ContrastiveLearner {
    pub fn new(latent_dim: usize, vb: VarBuilder) -> Result<Self> {
        let encoder = ContrastiveEncoder {
            fc1: candle_nn::linear(latent_dim, latent_dim * 2, vb.pp("cont_enc_fc1"))?,
            fc2: candle_nn::linear(latent_dim * 2, latent_dim * 2, vb.pp("cont_enc_fc2"))?,
            fc3: candle_nn::linear(latent_dim * 2, latent_dim, vb.pp("cont_enc_fc3"))?,
        };
        
        let projection_head = ProjectionHead {
            fc1: candle_nn::linear(latent_dim, latent_dim / 2, vb.pp("proj_fc1"))?,
            fc2: candle_nn::linear(latent_dim / 2, latent_dim / 4, vb.pp("proj_fc2"))?,
        };
        
        Ok(Self {
            encoder,
            projection_head,
            device: vb.device().clone(),
            temperature: 0.07,
        })
    }

    /// Encode input to representation space
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder.fc1.forward(x)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.encoder.fc2.forward(&h)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.encoder.fc3.forward(&h)?;
        
        // L2 normalize
        let norm = h.sqr()?.sum_keepdim(1)?.sqrt()?;
        h.div(&norm)
    }

    /// Project to contrastive space
    pub fn project(&self, z: &Tensor) -> Result<Tensor> {
        let h = self.projection_head.fc1.forward(z)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.projection_head.fc2.forward(&h)?;
        
        // L2 normalize
        let norm = h.sqr()?.sum_keepdim(1)?.sqrt()?;
        h.div(&norm)
    }

    /// Generate augmented views of the data
    pub fn augment(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let batch_size = x.dims()[0];
        let feature_dim = x.dims()[1];
        
        // Augmentation 1: Add noise
        let noise1 = Tensor::randn(0.0, 0.1, &[batch_size, feature_dim], &self.device)?;
        let aug1 = (x + noise1)?;
        
        // Augmentation 2: Feature masking
        let mask = Tensor::rand(0.0, 1.0, &[batch_size, feature_dim], &self.device)?;
        let mask = mask.ge(0.1)?;  // Keep 90% of features
        let aug2 = x.mul(&mask.to_dtype(DType::F32)?)?;
        
        Ok((aug1, aug2))
    }

    /// InfoNCE contrastive loss
    pub fn infonce_loss(&self, z1: &Tensor, z2: &Tensor) -> Result<Tensor> {
        let batch_size = z1.dims()[0];
        
        // Compute similarity matrix
        let sim_matrix = z1.matmul(&z2.t()?)?;
        let sim_matrix = sim_matrix.div_scalar(self.temperature)?;
        
        // Positive pairs are on diagonal
        let labels = Tensor::arange(0, batch_size as i64, &self.device)?;
        
        // Cross-entropy loss
        let log_probs = candle_nn::ops::log_softmax(&sim_matrix, 1)?;
        let loss = candle_nn::loss::nll(&log_probs, &labels)?;
        
        Ok(loss)
    }

    /// Supervised contrastive loss
    pub fn supervised_contrastive_loss(
        &self,
        z: &Tensor,
        labels: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = z.dims()[0];
        
        // Compute similarity matrix
        let sim_matrix = z.matmul(&z.t()?)?;
        let sim_matrix = sim_matrix.div_scalar(self.temperature)?;
        
        // Create mask for positive pairs (same label)
        let labels_eq = labels.unsqueeze(1)?.eq(&labels.unsqueeze(0)?)?;
        let mask = labels_eq.to_dtype(DType::F32)?;
        
        // Remove diagonal (self-similarity)
        let eye = Tensor::eye(batch_size, DType::F32, &self.device)?;
        let mask = mask.sub(&eye)?;
        
        // Compute loss
        let exp_sim = sim_matrix.exp()?;
        let sum_exp = exp_sim.sum_keepdim(1)?;
        let log_prob = sim_matrix.sub(&sum_exp.log()?)?;
        
        let pos_loss = (log_prob.mul(&mask)?)?.sum_keepdim(1)?;
        let num_pos = mask.sum_keepdim(1)?.clamp(1.0, f32::INFINITY)?;
        
        let loss = pos_loss.div(&num_pos)?.neg()?.mean_all()?;
        Ok(loss)
    }

    /// Train contrastive representations
    pub fn train(
        &mut self,
        data: &Tensor,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let mut opt = AdamW::new(vec![
            self.encoder.fc1.weight(),
            self.encoder.fc2.weight(),
            self.encoder.fc3.weight(),
            self.projection_head.fc1.weight(),
            self.projection_head.fc2.weight(),
        ], learning_rate)?;

        for epoch in 0..epochs {
            // Generate augmented views
            let (aug1, aug2) = self.augment(data)?;
            
            // Encode both views
            let z1 = self.encode(&aug1)?;
            let z2 = self.encode(&aug2)?;
            
            // Project to contrastive space
            let p1 = self.project(&z1)?;
            let p2 = self.project(&z2)?;
            
            // Compute contrastive loss
            let loss = self.infonce_loss(&p1, &p2)?;
            
            opt.backward_step(&loss)?;
            
            if epoch % 10 == 0 {
                println!("Contrastive epoch {}: loss = {}", epoch, loss.to_scalar::<f32>()?);
            }
        }
        
        Ok(())
    }

    /// Fine-tune with labeled anomaly data
    pub fn finetune_contrastive(
        &mut self,
        data: &Tensor,
        labels: &Tensor,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let mut opt = AdamW::new(vec![
            self.encoder.fc1.weight(),
            self.encoder.fc2.weight(),
            self.encoder.fc3.weight(),
            self.projection_head.fc1.weight(),
            self.projection_head.fc2.weight(),
        ], learning_rate)?;

        for epoch in 0..epochs {
            // Encode data
            let z = self.encode(data)?;
            let p = self.project(&z)?;
            
            // Supervised contrastive loss
            let loss = self.supervised_contrastive_loss(&p, labels)?;
            
            opt.backward_step(&loss)?;
            
            if epoch % 10 == 0 {
                println!("Contrastive finetune epoch {}: loss = {}", epoch, loss.to_scalar::<f32>()?);
            }
        }
        
        Ok(())
    }

    /// Triplet loss for anomaly detection
    pub fn triplet_loss(
        &self,
        anchor: &Tensor,
        positive: &Tensor,
        negative: &Tensor,
        margin: f32,
    ) -> Result<Tensor> {
        let z_anchor = self.encode(anchor)?;
        let z_positive = self.encode(positive)?;
        let z_negative = self.encode(negative)?;
        
        // Compute distances
        let pos_dist = (z_anchor.sub(&z_positive)?).sqr()?.sum_keepdim(1)?.sqrt()?;
        let neg_dist = (z_anchor.sub(&z_negative)?).sqr()?.sum_keepdim(1)?.sqrt()?;
        
        // Triplet loss
        let loss = (pos_dist - neg_dist + margin)?.clamp(0.0, f32::INFINITY)?;
        loss.mean_all()
    }

    /// Center loss for better intra-class compactness
    pub fn center_loss(&self, features: &Tensor, labels: &Tensor, centers: &Tensor) -> Result<Tensor> {
        let batch_size = features.dims()[0];
        let mut total_loss = Tensor::zeros(&[1], DType::F32, &self.device)?;
        
        for i in 0..batch_size {
            let label = labels.i(i)?.to_scalar::<i64>()?;
            let feature = features.i(i)?;
            let center = centers.i(label as usize)?;
            
            let dist = (feature - center)?.sqr()?.sum_all()?;
            total_loss = (total_loss + dist)?;
        }
        
        total_loss.div_scalar(batch_size as f64)
    }

    /// Prototypical loss for few-shot anomaly detection
    pub fn prototypical_loss(
        &self,
        support: &Tensor,
        query: &Tensor,
        support_labels: &Tensor,
        query_labels: &Tensor,
    ) -> Result<Tensor> {
        let z_support = self.encode(support)?;
        let z_query = self.encode(query)?;
        
        // Compute prototypes (class centroids)
        let num_classes = support_labels.max_all()?.to_scalar::<i64>()? + 1;
        let mut prototypes = Vec::new();
        
        for class in 0..num_classes {
            let mask = support_labels.eq(class)?;
            let class_features = z_support.where_cond(&mask.unsqueeze(1)?, &Tensor::zeros_like(&z_support)?)?;
            let prototype = class_features.mean(0)?;
            prototypes.push(prototype);
        }
        
        // Compute distances to prototypes
        let query_size = z_query.dims()[0];
        let mut total_loss = Tensor::zeros(&[1], DType::F32, &self.device)?;
        
        for i in 0..query_size {
            let query_feat = z_query.i(i)?;
            let true_label = query_labels.i(i)?.to_scalar::<i64>()? as usize;
            
            let mut distances = Vec::new();
            for (class_idx, prototype) in prototypes.iter().enumerate() {
                let dist = (query_feat.sub(prototype)?).sqr()?.sum_all()?;
                distances.push(dist);
            }
            
            // Convert to logits (negative distances)
            let logits = Tensor::stack(&distances, 0)?.neg()?;
            let log_probs = candle_nn::ops::log_softmax(&logits, 0)?;
            let loss = log_probs.i(true_label)?.neg()?;
            
            total_loss = (total_loss + loss)?;
        }
        
        total_loss.div_scalar(query_size as f64)
    }

    /// Get similarity between two inputs
    pub fn similarity(&self, x1: &Tensor, x2: &Tensor) -> Result<f32> {
        let z1 = self.encode(x1)?;
        let z2 = self.encode(x2)?;
        
        let sim = z1.matmul(&z2.t()?)?;
        sim.mean_all()?.to_scalar::<f32>()
    }

    /// Get k-nearest neighbors in embedding space
    pub fn knn_anomaly_score(&self, query: &Tensor, support: &Tensor, k: usize) -> Result<f32> {
        let z_query = self.encode(query)?;
        let z_support = self.encode(support)?;
        
        // Compute distances
        let distances = z_query.matmul(&z_support.t()?)?;
        
        // Get k nearest neighbors (highest similarities)
        let k_values = distances.topk(k)?.0;
        let knn_score = k_values.mean_all()?.to_scalar::<f32>()?;
        
        // Lower similarity = higher anomaly score
        Ok(1.0 - knn_score)
    }

    /// Set temperature for contrastive learning
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }
}