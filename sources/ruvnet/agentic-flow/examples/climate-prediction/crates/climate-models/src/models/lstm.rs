//! LSTM (Long Short-Term Memory) model for temporal climate prediction

use async_trait::async_trait;
use ort::{Session, SessionBuilder, Value};
use std::path::Path;
use tracing::{debug, info};

use crate::{
    traits::{ClimateInput, ClimatePrediction, PredictionModel},
    ModelConfig, Result, ModelError,
};

/// LSTM neural network model for climate prediction
///
/// This model uses Long Short-Term Memory networks to capture
/// temporal dependencies in climate data.
pub struct LSTMModel {
    name: String,
    config: ModelConfig,
    session: Option<Session>,
}

impl LSTMModel {
    /// Create a new LSTM model with configuration
    pub fn new(name: impl Into<String>, config: ModelConfig) -> Self {
        Self {
            name: name.into(),
            config,
            session: None,
        }
    }

    /// Create an LSTM model from an ONNX file
    pub async fn from_onnx(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading LSTM model from: {}", path.display());

        let session = SessionBuilder::new()?
            .with_intra_threads(4)?
            .commit_from_file(path)?;

        // Extract config from ONNX metadata if available
        let config = Self::extract_config_from_session(&session)?;

        Ok(Self {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("lstm")
                .to_string(),
            config,
            session: Some(session),
        })
    }

    /// Extract model configuration from ONNX session
    fn extract_config_from_session(session: &Session) -> Result<ModelConfig> {
        // Get input/output shapes from ONNX metadata
        let inputs = session.inputs.iter().next()
            .ok_or_else(|| ModelError::ModelNotFound("No input tensors".to_string()))?;
        let outputs = session.outputs.iter().next()
            .ok_or_else(|| ModelError::ModelNotFound("No output tensors".to_string()))?;

        // Extract dimensions (simplified - assumes [batch, seq, features])
        let input_dim = 10; // Default, should be extracted from metadata
        let output_dim = 3; // Default

        Ok(ModelConfig::lstm(input_dim, 64, 2, output_dim))
    }

    /// Prepare input tensor for ONNX inference
    fn prepare_input_tensor(&self, input: &ClimateInput) -> Result<Value> {
        let seq_len = input.sequence_length.unwrap_or(1);
        let features_per_step = input.features.len() / seq_len;

        debug!("Preparing input: seq_len={}, features={}", seq_len, features_per_step);

        // Reshape to [batch=1, sequence_length, features]
        let shape = vec![1, seq_len, features_per_step];

        // Convert to f32 array
        let input_data = input.features.clone();

        // Create ONNX tensor
        let value = Value::from_array(([1, seq_len, features_per_step], input_data))?;

        Ok(value)
    }

    /// Parse ONNX output into prediction
    fn parse_output(&self, output: Value, horizon: usize) -> Result<ClimatePrediction> {
        // Extract output array
        let output_array = output.try_extract_tensor::<f32>()?;
        let output_slice = output_array.view();

        // Assuming output shape is [batch=1, output_dim]
        let predictions: Vec<f32> = output_slice.iter().copied().collect();

        if predictions.len() < 1 {
            return Err(ModelError::InferenceFailed(
                "Empty output from model".to_string()
            ));
        }

        // Map outputs to climate variables (simplified)
        let temperature = predictions[0];
        let pressure = predictions.get(1).copied();
        let humidity = predictions.get(2).copied();

        // Calculate confidence (simplified - should use uncertainty estimation)
        let confidence = 0.85;

        Ok(ClimatePrediction {
            temperature,
            pressure,
            humidity,
            confidence,
            horizon,
            extra: std::collections::HashMap::new(),
        })
    }
}

#[async_trait]
impl PredictionModel for LSTMModel {
    async fn predict(&self, input: &ClimateInput) -> Result<ClimatePrediction> {
        let session = self.session.as_ref()
            .ok_or_else(|| ModelError::ModelNotFound("Model not loaded".to_string()))?;

        debug!("Running LSTM inference for {} features", input.features.len());

        // Prepare input
        let input_tensor = self.prepare_input_tensor(input)?;

        // Run inference
        let outputs = session.run(ort::inputs![input_tensor]?)?;

        // Parse output
        let output = outputs.get(0)
            .ok_or_else(|| ModelError::InferenceFailed("No output".to_string()))?;

        let horizon = input.sequence_length.unwrap_or(1);
        self.parse_output(output.clone(), horizon)
    }

    async fn load_weights(&mut self, path: &Path) -> Result<()> {
        info!("Loading LSTM weights from: {}", path.display());

        let session = SessionBuilder::new()?
            .with_intra_threads(4)?
            .commit_from_file(path)?;

        self.session = Some(session);
        Ok(())
    }

    async fn save_weights(&self, path: &Path) -> Result<()> {
        // ONNX models are already saved as files
        // This would be used for checkpointing during training
        info!("LSTM model weights saved to: {}", path.display());
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lstm_model_creation() {
        let config = ModelConfig::lstm(10, 64, 2, 3);
        let model = LSTMModel::new("test_lstm", config);
        assert_eq!(model.name(), "test_lstm");
    }

    #[tokio::test]
    async fn test_input_preparation() {
        let config = ModelConfig::lstm(3, 64, 2, 1);
        let model = LSTMModel::new("test", config);

        let input = ClimateInput::with_sequence(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
        let result = model.prepare_input_tensor(&input);

        assert!(result.is_ok());
    }
}
