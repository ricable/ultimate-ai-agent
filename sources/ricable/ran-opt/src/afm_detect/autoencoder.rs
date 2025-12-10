use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, Optimizer, AdamW};

/// Autoencoder for reconstruction-based anomaly detection
pub struct AutoencoderDetector {
    encoder: Encoder,
    decoder: Decoder,
    device: Device,
}

struct Encoder {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

struct Decoder {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl AutoencoderDetector {
    pub fn new(input_dim: usize, latent_dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (input_dim + latent_dim) / 2;
        
        let encoder = Encoder {
            fc1: candle_nn::linear(input_dim, hidden_dim * 2, vb.pp("enc_fc1"))?,
            fc2: candle_nn::linear(hidden_dim * 2, hidden_dim, vb.pp("enc_fc2"))?,
            fc3: candle_nn::linear(hidden_dim, latent_dim, vb.pp("enc_fc3"))?,
        };
        
        let decoder = Decoder {
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

    /// Encode input to latent representation
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder.fc1.forward(x)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.encoder.fc2.forward(&h)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        self.encoder.fc3.forward(&h)
    }

    /// Decode latent representation back to input space
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let h = self.decoder.fc1.forward(z)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        let h = self.decoder.fc2.forward(&h)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.2)?;
        self.decoder.fc3.forward(&h)
    }

    /// Forward pass through autoencoder
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.encode(x)?;
        self.decode(&z)
    }

    /// Calculate reconstruction error as anomaly score
    pub fn anomaly_score(&self, x: &Tensor) -> Result<f32> {
        let reconstructed = self.forward(x)?;
        let error = (x - reconstructed)?;
        let mse = error.sqr()?.mean_all()?;
        mse.to_scalar::<f32>()
    }

    /// Train the autoencoder on normal data
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
            self.decoder.fc1.weight(),
            self.decoder.fc2.weight(),
            self.decoder.fc3.weight(),
        ], learning_rate)?;

        for epoch in 0..epochs {
            let reconstructed = self.forward(data)?;
            let loss = (data - reconstructed)?.sqr()?.mean_all()?;
            
            opt.backward_step(&loss)?;
            
            if epoch % 10 == 0 {
                println!("Autoencoder epoch {}: loss = {}", epoch, loss.to_scalar::<f32>()?);
            }
        }
        
        Ok(())
    }

    /// Get bottleneck features for downstream tasks
    pub fn get_features(&self, x: &Tensor) -> Result<Tensor> {
        self.encode(x)
    }

    /// Compute per-feature reconstruction error
    pub fn feature_wise_error(&self, x: &Tensor) -> Result<Tensor> {
        let reconstructed = self.forward(x)?;
        (x - reconstructed)?.sqr()
    }
}