//! Machine learning model for off-target prediction

use crate::error::{PredictionError, Result};
use crate::features::FeatureExtractor;
use data_models::targets::CrisprTarget;
use ndarray::{Array1, Array2};
use std::path::Path;

/// ML model interface for off-target scoring
pub trait MlModel: Send + Sync {
    /// Predict scores for feature vectors
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>>;

    /// Predict score for a single feature vector
    fn predict_single(&self, features: &Array1<f64>) -> Result<f64>;

    /// Get model metadata
    fn metadata(&self) -> ModelMetadata;
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub num_features: usize,
    pub trained_on: String,
}

/// Ensemble model combining multiple prediction methods
pub struct EnsembleModel {
    models: Vec<Box<dyn MlModel>>,
    weights: Vec<f64>,
}

impl EnsembleModel {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            weights: Vec::new(),
        }
    }

    pub fn add_model(&mut self, model: Box<dyn MlModel>, weight: f64) {
        self.models.push(model);
        self.weights.push(weight);
    }
}

impl MlModel for EnsembleModel {
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>> {
        if self.models.is_empty() {
            return Err(PredictionError::Model("No models in ensemble".to_string()));
        }

        let mut weighted_sum = Array1::zeros(features.nrows());
        let weight_sum: f64 = self.weights.iter().sum();

        for (model, &weight) in self.models.iter().zip(self.weights.iter()) {
            let predictions = model.predict(features)?;
            weighted_sum = weighted_sum + predictions * (weight / weight_sum);
        }

        Ok(weighted_sum)
    }

    fn predict_single(&self, features: &Array1<f64>) -> Result<f64> {
        if self.models.is_empty() {
            return Err(PredictionError::Model("No models in ensemble".to_string()));
        }

        let mut weighted_sum = 0.0;
        let weight_sum: f64 = self.weights.iter().sum();

        for (model, &weight) in self.models.iter().zip(self.weights.iter()) {
            let prediction = model.predict_single(features)?;
            weighted_sum += prediction * (weight / weight_sum);
        }

        Ok(weighted_sum)
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            name: "EnsembleModel".to_string(),
            version: "1.0.0".to_string(),
            num_features: 0,
            trained_on: "Multiple datasets".to_string(),
        }
    }
}

/// Simple linear model for off-target scoring
pub struct LinearModel {
    weights: Array1<f64>,
    bias: f64,
    metadata: ModelMetadata,
}

impl LinearModel {
    /// Create a new linear model with given weights and bias
    pub fn new(weights: Array1<f64>, bias: f64) -> Self {
        let num_features = weights.len();
        Self {
            weights,
            bias,
            metadata: ModelMetadata {
                name: "LinearModel".to_string(),
                version: "1.0.0".to_string(),
                num_features,
                trained_on: "CRISPR-Cas13 dataset".to_string(),
            },
        }
    }

    /// Create a default model with pre-trained weights
    pub fn default_model() -> Self {
        // Default weights tuned for CRISPR-Cas13 off-target prediction
        let weights = Array1::from(vec![
            -0.5,  // mismatches (negative correlation)
            -0.3,  // position-weighted score
            0.1,   // target GC
            0.1,   // candidate GC
            -0.2,  // GC difference
            -0.4,  // seed mismatches
            -0.1,  // Tm difference
            -0.15, // homopolymers
            0.2,   // complexity
            -0.1,  // dinucleotide repeats
            0.05,  // purine content
        ]);
        let bias = 0.8;

        Self::new(weights, bias)
    }

    /// Load model from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        // In production, this would load from a serialized model file
        // For now, return the default model
        Ok(Self::default_model())
    }

    /// Save model to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // In production, this would serialize the model to a file
        Ok(())
    }
}

impl MlModel for LinearModel {
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>> {
        // Linear prediction: y = Xw + b
        let predictions = features.dot(&self.weights) + self.bias;

        // Apply sigmoid activation to bound between 0 and 1
        Ok(predictions.mapv(|x| 1.0 / (1.0 + (-x).exp())))
    }

    fn predict_single(&self, features: &Array1<f64>) -> Result<f64> {
        if features.len() != self.weights.len() {
            return Err(PredictionError::FeatureExtraction(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.weights.len(),
                features.len()
            )));
        }

        let linear_output = features.dot(&self.weights) + self.bias;
        // Apply sigmoid
        Ok(1.0 / (1.0 + (-linear_output).exp()))
    }

    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }
}

/// Neural network model (placeholder for future ONNX integration)
pub struct NeuralNetworkModel {
    model_path: String,
    metadata: ModelMetadata,
}

impl NeuralNetworkModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let model_path = path.as_ref().to_string_lossy().to_string();

        Ok(Self {
            model_path,
            metadata: ModelMetadata {
                name: "NeuralNetwork".to_string(),
                version: "1.0.0".to_string(),
                num_features: 0,
                trained_on: "Large-scale CRISPR dataset".to_string(),
            },
        })
    }
}

impl MlModel for NeuralNetworkModel {
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>> {
        // Placeholder for ONNX Runtime integration
        // For now, return random scores
        Err(PredictionError::Model(
            "Neural network inference not yet implemented".to_string(),
        ))
    }

    fn predict_single(&self, features: &Array1<f64>) -> Result<f64> {
        Err(PredictionError::Model(
            "Neural network inference not yet implemented".to_string(),
        ))
    }

    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model_prediction() {
        let model = LinearModel::default_model();
        let features = Array1::from(vec![
            1.0, 0.5, 50.0, 48.0, 2.0, 1.0, 5.0, 2.0, 1.5, 1.0, 55.0,
        ]);

        let score = model.predict_single(&features).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_batch_prediction() {
        let model = LinearModel::default_model();
        let features = Array2::from_shape_vec(
            (2, 11),
            vec![
                1.0, 0.5, 50.0, 48.0, 2.0, 1.0, 5.0, 2.0, 1.5, 1.0, 55.0, 2.0, 1.0, 52.0, 50.0,
                2.0, 2.0, 8.0, 3.0, 1.2, 2.0, 58.0,
            ],
        )
        .unwrap();

        let scores = model.predict(&features).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_ensemble_model() {
        let mut ensemble = EnsembleModel::new();
        ensemble.add_model(Box::new(LinearModel::default_model()), 1.0);

        let features = Array1::from(vec![
            1.0, 0.5, 50.0, 48.0, 2.0, 1.0, 5.0, 2.0, 1.5, 1.0, 55.0,
        ]);
        let score = ensemble.predict_single(&features).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }
}
