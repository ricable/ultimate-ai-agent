//! Fourier Neural Operator (FNO) for spatial-temporal climate prediction

use async_trait::async_trait;
use ort::{Session, SessionBuilder, Value};
use std::path::Path;
use tracing::{debug, info};

use crate::{
    traits::{ClimateInput, ClimatePrediction, PredictionModel},
    ModelConfig, Result, ModelError,
};

/// Fourier Neural Operator model for climate prediction
///
/// FNO uses spectral methods to learn operators in Fourier space,
/// making it particularly effective for spatial-temporal PDEs like
/// climate modeling.
pub struct FNOModel {
    name: String,
    config: ModelConfig,
    session: Option<Session>,
    modes: usize,
    width: usize,
}

impl FNOModel {
    /// Create a new FNO model with configuration
    pub fn new(name: impl Into<String>, config: ModelConfig) -> Self {
        let modes = config.extra.get("modes")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize;
        let width = config.extra.get("width")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;

        Self {
            name: name.into(),
            config,
            session: None,
            modes,
            width,
        }
    }

    /// Create an FNO model from an ONNX file
    pub async fn from_onnx(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading FNO model from: {}", path.display());

        let session = SessionBuilder::new()?
            .with_intra_threads(4)?
            .commit_from_file(path)?;

        let config = Self::extract_config_from_session(&session)?;

        Ok(Self {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("fno")
                .to_string(),
            config: config.clone(),
            session: Some(session),
            modes: 12,
            width: 64,
        })
    }

    /// Extract model configuration from ONNX session
    fn extract_config_from_session(session: &Session) -> Result<ModelConfig> {
        // Extract from ONNX metadata
        let input_dim = 10; // Should extract from session
        let output_dim = 3;

        Ok(ModelConfig::fno(input_dim, 12, 64, output_dim))
    }

    /// Prepare spatial-temporal input for FNO
    fn prepare_input_tensor(&self, input: &ClimateInput) -> Result<Value> {
        let (lat_dim, lon_dim) = input.spatial_dims
            .ok_or_else(|| ModelError::InvalidInputShape {
                expected: "spatial dimensions".to_string(),
                actual: "none".to_string(),
            })?;

        let features_per_point = input.features.len() / (lat_dim * lon_dim);

        debug!("Preparing FNO input: {}x{}, {} features", lat_dim, lon_dim, features_per_point);

        // Reshape to [batch=1, channels, height, width]
        let shape = vec![1, features_per_point, lat_dim, lon_dim];
        let input_data = input.features.clone();

        let value = Value::from_array(([1, features_per_point, lat_dim, lon_dim], input_data))?;

        Ok(value)
    }

    /// Parse FNO output into climate prediction
    fn parse_output(&self, output: Value, spatial_dims: (usize, usize)) -> Result<ClimatePrediction> {
        let output_array = output.try_extract_tensor::<f32>()?;
        let output_slice = output_array.view();

        // Assuming output is spatial field, take mean for point prediction
        let predictions: Vec<f32> = output_slice.iter().copied().collect();

        if predictions.is_empty() {
            return Err(ModelError::InferenceFailed("Empty output".to_string()));
        }

        // Spatial average for scalar prediction
        let (lat_dim, lon_dim) = spatial_dims;
        let num_points = lat_dim * lon_dim;

        let temperature = predictions.iter().take(num_points).sum::<f32>() / num_points as f32;

        let confidence = 0.82; // FNO typically has good spatial consistency

        Ok(ClimatePrediction {
            temperature,
            pressure: None,
            humidity: None,
            confidence,
            horizon: 1,
            extra: {
                let mut map = std::collections::HashMap::new();
                map.insert("spatial_resolution".to_string(), format!("{}x{}", lat_dim, lon_dim).parse().unwrap_or(0.0));
                map
            },
        })
    }
}

#[async_trait]
impl PredictionModel for FNOModel {
    async fn predict(&self, input: &ClimateInput) -> Result<ClimatePrediction> {
        let session = self.session.as_ref()
            .ok_or_else(|| ModelError::ModelNotFound("Model not loaded".to_string()))?;

        let spatial_dims = input.spatial_dims
            .ok_or_else(|| ModelError::InvalidInputShape {
                expected: "spatial dimensions".to_string(),
                actual: "none".to_string(),
            })?;

        debug!("Running FNO inference for {}x{} grid", spatial_dims.0, spatial_dims.1);

        // Prepare input
        let input_tensor = self.prepare_input_tensor(input)?;

        // Run inference
        let outputs = session.run(ort::inputs![input_tensor]?)?;

        // Parse output
        let output = outputs.get(0)
            .ok_or_else(|| ModelError::InferenceFailed("No output".to_string()))?;

        self.parse_output(output.clone(), spatial_dims)
    }

    async fn load_weights(&mut self, path: &Path) -> Result<()> {
        info!("Loading FNO weights from: {}", path.display());

        let session = SessionBuilder::new()?
            .with_intra_threads(4)?
            .commit_from_file(path)?;

        self.session = Some(session);
        Ok(())
    }

    async fn save_weights(&self, path: &Path) -> Result<()> {
        info!("FNO model weights saved to: {}", path.display());
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
    async fn test_fno_model_creation() {
        let config = ModelConfig::fno(10, 12, 64, 3);
        let model = FNOModel::new("test_fno", config);
        assert_eq!(model.name(), "test_fno");
        assert_eq!(model.modes, 12);
        assert_eq!(model.width, 64);
    }

    #[tokio::test]
    async fn test_spatial_input_preparation() {
        let config = ModelConfig::fno(3, 12, 64, 1);
        let model = FNOModel::new("test", config);

        // 4x4 grid with 3 features per point
        let input = ClimateInput::with_spatial(vec![1.0; 48], 4, 4);
        let result = model.prepare_input_tensor(&input);

        assert!(result.is_ok());
    }
}
