use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, Optimizer, AdamW};

/// Variational Autoencoder for probabilistic anomaly detection
pub struct VariationalDetector {
    encoder: VariationalEncoder,
    decoder: VariationalDecoder,
    device: Device,
}

struct VariationalEncoder {
    fc1: Linear,
    fc2: Linear,
    fc_mu: Linear,
    fc_logvar: Linear,
}

struct VariationalDecoder {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl VariationalDetector {
    pub fn new(input_dim: usize, latent_dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (input_dim + latent_dim) / 2;
        
        let encoder = VariationalEncoder {
            fc1: candle_nn::linear(input_dim, hidden_dim * 2, vb.pp("enc_fc1"))?,
            fc2: candle_nn::linear(hidden_dim * 2, hidden_dim, vb.pp("enc_fc2"))?,
            fc_mu: candle_nn::linear(hidden_dim, latent_dim, vb.pp("enc_mu"))?,
            fc_logvar: candle_nn::linear(hidden_dim, latent_dim, vb.pp("enc_logvar"))?,
        };
        
        let decoder = VariationalDecoder {
            fc1: candle_nn::linear(latent_dim, hidden_dim, vb.pp("dec_fc1"))?,
            fc2: candle_nn::linear(hidden_dim, hidden_dim * 2, vb.pp("dec_fc2"))?,
            fc3: candle_nn::linear(hidden_dim * 2, input_dim, vb.pp("dec_fc3"))?,
        };
        
        Ok(Self {
            encoder,
            decoder,
            device: vb.device().clone(),
        })
    }

    /// Encode input to latent distribution parameters
    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.encoder.fc1.forward(x)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.encoder.fc2.forward(&h)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        
        let mu = self.encoder.fc_mu.forward(&h)?;
        let logvar = self.encoder.fc_logvar.forward(&h)?;
        
        Ok((mu, logvar))
    }

    /// Reparameterization trick for sampling
    pub fn reparameterize(&self, mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        let std = (logvar * 0.5)?.exp()?;
        let eps = Tensor::randn(0.0, 1.0, mu.dims(), &self.device)?;
        Ok((mu + std * eps)?)
    }

    /// Decode latent sample back to input space
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let h = self.decoder.fc1.forward(z)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.decoder.fc2.forward(&h)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        self.decoder.fc3.forward(&h)
    }

    /// Forward pass through VAE
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (mu, logvar) = self.encode(x)?;
        let z = self.reparameterize(&mu, &logvar)?;
        let reconstructed = self.decode(&z)?;
        Ok((reconstructed, mu, logvar))
    }

    /// Calculate VAE loss (reconstruction + KL divergence)
    pub fn vae_loss(&self, x: &Tensor, reconstructed: &Tensor, mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        // Reconstruction loss (MSE)
        let recon_loss = (x - reconstructed)?.sqr()?.mean_all()?;
        
        // KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        let kl_loss = -0.5 * (
            1.0 + logvar - mu.sqr()? - logvar.exp()?
        )?.mean_all()?;
        
        // Total loss
        recon_loss + kl_loss
    }

    /// Calculate anomaly score using reconstruction probability
    pub fn anomaly_score(&self, x: &Tensor) -> Result<f32> {
        let (reconstructed, mu, logvar) = self.forward(x)?;
        
        // Reconstruction error
        let recon_error = (x - reconstructed)?.sqr()?.mean_all()?;
        
        // KL divergence from standard normal
        let kl_div = -0.5 * (
            1.0 + logvar - mu.sqr()? - logvar.exp()?
        )?.mean_all()?;
        
        // Combined anomaly score
        let score = recon_error + kl_div;
        score.to_scalar::<f32>()
    }

    /// Calculate log-likelihood based anomaly score
    pub fn log_likelihood_score(&self, x: &Tensor) -> Result<f32> {
        let (reconstructed, mu, logvar) = self.forward(x)?;
        
        // Negative log-likelihood as anomaly score
        let log_likelihood = -0.5 * (
            (x - reconstructed)?.sqr()? + 
            logvar + 
            2.0 * std::f32::consts::PI.ln()
        )?.mean_all()?;
        
        (-log_likelihood).to_scalar::<f32>()
    }

    /// Train the VAE on normal data
    pub fn train(
        &mut self,
        data: &Tensor,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let mut opt = AdamW::new(vec![
            self.encoder.fc1.weight(),
            self.encoder.fc2.weight(),
            self.encoder.fc_mu.weight(),
            self.encoder.fc_logvar.weight(),
            self.decoder.fc1.weight(),
            self.decoder.fc2.weight(),
            self.decoder.fc3.weight(),
        ], learning_rate)?;

        for epoch in 0..epochs {
            let (reconstructed, mu, logvar) = self.forward(data)?;
            let loss = self.vae_loss(data, &reconstructed, &mu, &logvar)?;
            
            opt.backward_step(&loss)?;
            
            if epoch % 10 == 0 {
                println!("VAE epoch {}: loss = {}", epoch, loss.to_scalar::<f32>()?);
            }
        }
        
        Ok(())
    }

    /// Generate samples from the learned distribution
    pub fn generate(&self, batch_size: usize, latent_dim: usize) -> Result<Tensor> {
        let z = Tensor::randn(0.0, 1.0, &[batch_size, latent_dim], &self.device)?;
        self.decode(&z)
    }

    /// Get latent representation for downstream tasks
    pub fn get_latent(&self, x: &Tensor) -> Result<Tensor> {
        let (mu, logvar) = self.encode(x)?;
        self.reparameterize(&mu, &logvar)
    }

    /// Calculate anomaly score with uncertainty
    pub fn anomaly_score_with_uncertainty(&self, x: &Tensor, n_samples: usize) -> Result<(f32, f32)> {
        let mut scores = Vec::new();
        
        for _ in 0..n_samples {
            let score = self.anomaly_score(x)?;
            scores.push(score);
        }
        
        let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
        let std_score = (scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f32>() 
            / scores.len() as f32).sqrt();
        
        Ok((mean_score, std_score))
    }
}