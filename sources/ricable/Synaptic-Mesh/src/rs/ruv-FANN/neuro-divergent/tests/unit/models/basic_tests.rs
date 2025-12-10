//! Unit tests for basic model implementations and patterns
//!
//! This module tests the basic model trait implementations and common patterns
//! used across all neural forecasting models.

use neuro_divergent::prelude::*;
use neuro_divergent::{AccuracyMetrics, NeuroDivergentError, NeuroDivergentResult};
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};
use neuro_divergent::core::{BaseModel, ModelConfig};
use num_traits::Float;
use proptest::prelude::*;
use approx::assert_relative_eq;
use std::collections::HashMap;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Mock Model Implementations for Testing
// ============================================================================

/// Mock configuration for testing BaseModel trait
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MockMLPConfig<T: Float> {
    pub input_size: usize,
    pub horizon: usize,
    pub hidden_layers: Vec<usize>,
    pub learning_rate: T,
    pub max_epochs: usize,
    pub dropout: Option<T>,
    pub batch_size: Option<usize>,
}

impl<T: Float> MockMLPConfig<T> {
    pub fn new(input_size: usize, horizon: usize) -> Self {
        Self {
            input_size,
            horizon,
            hidden_layers: vec![64, 32],
            learning_rate: T::from(0.001).unwrap(),
            max_epochs: 100,
            dropout: None,
            batch_size: None,
        }
    }
    
    pub fn with_hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }
    
    pub fn with_learning_rate(mut self, lr: T) -> Self {
        self.learning_rate = lr;
        self
    }
    
    pub fn with_dropout(mut self, dropout: T) -> Self {
        self.dropout = Some(dropout);
        self
    }
}

impl<T: Float> ModelConfig<T> for MockMLPConfig<T> {
    fn horizon(&self) -> usize {
        self.horizon
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon == 0 {
            return Err(NeuroDivergentError::config("Horizon must be greater than 0"));
        }
        if self.input_size == 0 {
            return Err(NeuroDivergentError::config("Input size must be greater than 0"));
        }
        if self.learning_rate <= T::zero() {
            return Err(NeuroDivergentError::config("Learning rate must be positive"));
        }
        if self.hidden_layers.is_empty() {
            return Err(NeuroDivergentError::config("Must have at least one hidden layer"));
        }
        if let Some(dropout) = self.dropout {
            if dropout < T::zero() || dropout >= T::one() {
                return Err(NeuroDivergentError::config("Dropout must be in [0, 1)"));
            }
        }
        Ok(())
    }
    
    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("input_size".to_string(), self.input_size.to_string());
        params.insert("horizon".to_string(), self.horizon.to_string());
        params.insert("hidden_layers".to_string(), format!("{:?}", self.hidden_layers));
        params.insert("learning_rate".to_string(), format!("{:?}", self.learning_rate));
        params.insert("max_epochs".to_string(), self.max_epochs.to_string());
        if let Some(dropout) = self.dropout {
            params.insert("dropout".to_string(), format!("{:?}", dropout));
        }
        params
    }
    
    fn model_type(&self) -> &'static str {
        "MockMLP"
    }
}

/// Mock MLP model implementation for testing
#[derive(Clone)]
struct MockMLP<T: Float> {
    config: MockMLPConfig<T>,
    weights: Option<Vec<T>>,
    trained: bool,
}

impl<T: Float + Send + Sync> BaseModel<T> for MockMLP<T> {
    type Config = MockMLPConfig<T>;

    fn new(config: Self::Config) -> NeuroDivergentResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            weights: None,
            trained: false,
        })
    }

    fn fit(&mut self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()> {
        if data.shape().0 == 0 {
            return Err(NeuroDivergentError::data("No data to fit"));
        }
        
        // Simulate training by creating mock weights
        let total_params = self.config.hidden_layers.iter().sum::<usize>() + 
                          self.config.input_size + 
                          self.config.horizon;
        
        self.weights = Some((0..total_params)
            .map(|i| T::from(i as f64 * 0.01).unwrap())
            .collect());
        self.trained = true;
        
        Ok(())
    }

    fn predict(&self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        if !self.trained {
            return Err(NeuroDivergentError::prediction("Model not trained"));
        }
        
        if data.shape().0 == 0 {
            return Err(NeuroDivergentError::data("No data to predict"));
        }
        
        // Create mock forecast data
        let unique_ids = data.unique_ids()?;
        let mut forecast_data = Vec::new();
        
        for id in unique_ids {
            for step in 1..=self.config.horizon {
                forecast_data.push((
                    id.clone(),
                    step,
                    T::from(10.0 + step as f64).unwrap(), // Mock forecast value
                ));
            }
        }
        
        let df = df! {
            "unique_id" => forecast_data.iter().map(|(id, _, _)| id.clone()).collect::<Vec<_>>(),
            "ds" => forecast_data.iter().map(|(_, step, _)| *step).collect::<Vec<_>>(),
            "MockMLP" => forecast_data.iter().map(|(_, _, val)| T::to_f64(val).unwrap()).collect::<Vec<_>>(),
        }.map_err(|e| NeuroDivergentError::prediction(format!("Failed to create forecast DataFrame: {}", e)))?;
        
        Ok(ForecastDataFrame::new(
            df,
            vec!["MockMLP".to_string()],
            self.config.horizon,
            None,
            TimeSeriesSchema::default(),
        ))
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn is_fitted(&self) -> bool {
        self.trained
    }

    fn reset(&mut self) -> NeuroDivergentResult<()> {
        self.weights = None;
        self.trained = false;
        Ok(())
    }
}

// ============================================================================
// Model Configuration Tests
// ============================================================================

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_mlp_config_creation() {
        let config = MockMLPConfig::new(24, 12)
            .with_hidden_layers(vec![64, 32])
            .with_learning_rate(0.001)
            .with_dropout(0.1);

        assert_eq!(config.input_size, 24);
        assert_eq!(config.horizon, 12);
        assert_eq!(config.hidden_layers, vec![64, 32]);
        assert_relative_eq!(config.learning_rate, 0.001, epsilon = 1e-6);
        assert_eq!(config.dropout, Some(0.1));
    }

    #[test]
    fn test_config_validation_success() {
        let config = MockMLPConfig::new(10, 5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_failures() {
        // Zero horizon
        let config = MockMLPConfig::new(10, 0);
        assert!(config.validate().is_err());

        // Zero input size
        let config = MockMLPConfig::new(0, 5);
        assert!(config.validate().is_err());

        // Negative learning rate
        let mut config = MockMLPConfig::new(10, 5);
        config.learning_rate = -0.001;
        assert!(config.validate().is_err());

        // Invalid dropout
        let config = MockMLPConfig::new(10, 5).with_dropout(1.5);
        assert!(config.validate().is_err());

        let config = MockMLPConfig::new(10, 5).with_dropout(-0.1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_parameters() {
        let config = MockMLPConfig::new(24, 12)
            .with_hidden_layers(vec![64, 32])
            .with_dropout(0.2);

        let params = config.parameters();
        assert_eq!(params.get("input_size"), Some(&"24".to_string()));
        assert_eq!(params.get("horizon"), Some(&"12".to_string()));
        assert_eq!(params.get("model_type"), None); // Not included in parameters
        assert!(params.contains_key("hidden_layers"));
        assert!(params.contains_key("dropout"));
    }

    #[test]
    fn test_model_type() {
        let config = MockMLPConfig::<f64>::new(10, 5);
        assert_eq!(config.model_type(), "MockMLP");
    }
}

// ============================================================================
// Model Implementation Tests
// ============================================================================

#[cfg(test)]
mod model_tests {
    use super::*;

    fn create_test_data<T: Float>() -> NeuroDivergentResult<TimeSeriesDataFrame<T>> {
        let data = df! {
            "unique_id" => ["A", "A", "A", "B", "B", "B"],
            "ds" => [1, 2, 3, 1, 2, 3],
            "y" => [10.0, 12.0, 14.0, 20.0, 22.0, 24.0],
        }.unwrap();

        TimeSeriesDataFrame::from_polars(data, TimeSeriesSchema::default(), None)
    }

    #[test]
    fn test_model_creation() {
        let config = MockMLPConfig::new(24, 12);
        let model = MockMLP::new(config);
        
        assert!(model.is_ok());
        let model = model.unwrap();
        assert!(!model.is_fitted());
        assert_eq!(model.config().input_size, 24);
        assert_eq!(model.config().horizon, 12);
    }

    #[test]
    fn test_model_creation_with_invalid_config() {
        let config = MockMLPConfig::new(0, 12); // Invalid input size
        let model = MockMLP::new(config);
        assert!(model.is_err());
    }

    #[test]
    fn test_model_fit() {
        let config = MockMLPConfig::new(24, 12);
        let mut model = MockMLP::new(config).unwrap();
        let data = create_test_data::<f64>().unwrap();

        assert!(!model.is_fitted());
        
        let result = model.fit(&data);
        assert!(result.is_ok());
        assert!(model.is_fitted());
        assert!(model.weights.is_some());
    }

    #[test]
    fn test_model_fit_with_empty_data() {
        let config = MockMLPConfig::new(24, 12);
        let mut model = MockMLP::new(config).unwrap();
        
        // Create empty DataFrame
        let empty_data = df! {
            "unique_id" => Vec::<String>::new(),
            "ds" => Vec::<i32>::new(),
            "y" => Vec::<f64>::new(),
        }.unwrap();
        
        let ts_data = TimeSeriesDataFrame::from_polars(empty_data, TimeSeriesSchema::default(), None).unwrap();
        let result = model.fit(&ts_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_predict_before_fit() {
        let config = MockMLPConfig::new(24, 12);
        let model = MockMLP::new(config).unwrap();
        let data = create_test_data::<f64>().unwrap();

        let result = model.predict(&data);
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert!(error.is_prediction_error());
    }

    #[test]
    fn test_model_predict_after_fit() {
        let config = MockMLPConfig::new(24, 12);
        let mut model = MockMLP::new(config.clone()).unwrap();
        let data = create_test_data::<f64>().unwrap();

        model.fit(&data).unwrap();
        let forecast = model.predict(&data);
        
        assert!(forecast.is_ok());
        let forecast = forecast.unwrap();
        assert_eq!(forecast.forecast_horizon, config.horizon);
        assert_eq!(forecast.models, vec!["MockMLP".to_string()]);
        
        // Should have forecasts for both series A and B
        let forecast_shape = forecast.shape();
        assert_eq!(forecast_shape.0, config.horizon * 2); // 2 series
    }

    #[test]
    fn test_model_reset() {
        let config = MockMLPConfig::new(24, 12);
        let mut model = MockMLP::new(config).unwrap();
        let data = create_test_data::<f64>().unwrap();

        // Fit the model
        model.fit(&data).unwrap();
        assert!(model.is_fitted());

        // Reset the model
        let result = model.reset();
        assert!(result.is_ok());
        assert!(!model.is_fitted());
        assert!(model.weights.is_none());
    }

    #[test]
    fn test_model_validate_input() {
        let config = MockMLPConfig::new(24, 12);
        let model = MockMLP::new(config).unwrap();
        let data = create_test_data::<f64>().unwrap();

        // Should pass validation
        let result = model.validate_input(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_summary() {
        let config = MockMLPConfig::new(24, 12);
        let model = MockMLP::new(config).unwrap();

        let summary = model.summary();
        assert_eq!(summary.get("model_type"), Some(&"MockMLP".to_string()));
        assert_eq!(summary.get("horizon"), Some(&"12".to_string()));
        assert_eq!(summary.get("input_size"), Some(&"24".to_string()));
        assert_eq!(summary.get("fitted"), Some(&"false".to_string()));
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_config_invariants(
            input_size in 1usize..1000,
            horizon in 1usize..100,
            learning_rate in 0.0001f64..1.0,
            hidden_size in 1usize..1000
        ) {
            let config = MockMLPConfig::new(input_size, horizon)
                .with_hidden_layers(vec![hidden_size])
                .with_learning_rate(learning_rate);

            // Configuration should be valid
            assert!(config.validate().is_ok());
            
            // Properties should match
            assert_eq!(config.input_size, input_size);
            assert_eq!(config.horizon, horizon);
            assert_relative_eq!(config.learning_rate, learning_rate, epsilon = 1e-10);
            assert_eq!(config.hidden_layers, vec![hidden_size]);
        }

        #[test]
        fn test_model_lifecycle(
            input_size in 1usize..100,
            horizon in 1usize..50
        ) {
            let config = MockMLPConfig::new(input_size, horizon);
            let mut model = MockMLP::new(config.clone()).unwrap();
            
            // Initially not fitted
            assert!(!model.is_fitted());
            
            // Create test data
            let data = df! {
                "unique_id" => ["test"],
                "ds" => [1],
                "y" => [1.0],
            }.unwrap();
            let ts_data = TimeSeriesDataFrame::from_polars(data, TimeSeriesSchema::default(), None).unwrap();
            
            // Should be able to fit
            let fit_result = model.fit(&ts_data);
            assert!(fit_result.is_ok());
            assert!(model.is_fitted());
            
            // Should be able to predict after fitting
            let predict_result = model.predict(&ts_data);
            assert!(predict_result.is_ok());
            
            let forecast = predict_result.unwrap();
            assert_eq!(forecast.forecast_horizon, horizon);
            
            // Should be able to reset
            let reset_result = model.reset();
            assert!(reset_result.is_ok());
            assert!(!model.is_fitted());
        }
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_workflow() {
        // Create model configuration
        let config = MockMLPConfig::new(10, 5)
            .with_hidden_layers(vec![32, 16])
            .with_learning_rate(0.01)
            .with_dropout(0.1);

        // Create model
        let mut model = MockMLP::new(config).unwrap();

        // Create training data
        let train_data = df! {
            "unique_id" => ["series_1", "series_1", "series_1", "series_2", "series_2", "series_2"],
            "ds" => [1, 2, 3, 1, 2, 3],
            "y" => [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }.unwrap();
        
        let ts_train = TimeSeriesDataFrame::from_polars(train_data, TimeSeriesSchema::default(), None).unwrap();

        // Fit model
        let fit_result = model.fit(&ts_train);
        assert!(fit_result.is_ok());
        assert!(model.is_fitted());

        // Create prediction data
        let pred_data = df! {
            "unique_id" => ["series_1", "series_2"],
            "ds" => [4, 4],
            "y" => [4.0, 40.0],
        }.unwrap();
        
        let ts_pred = TimeSeriesDataFrame::from_polars(pred_data, TimeSeriesSchema::default(), None).unwrap();

        // Generate forecasts
        let forecast_result = model.predict(&ts_pred);
        assert!(forecast_result.is_ok());
        
        let forecast = forecast_result.unwrap();
        assert_eq!(forecast.forecast_horizon, 5);
        assert_eq!(forecast.models, vec!["MockMLP".to_string()]);
        
        // Verify forecast contains data for both series
        let model_forecasts = forecast.get_model_forecasts("MockMLP").unwrap();
        let unique_ids: Vec<&str> = model_forecasts
            .column("unique_id").unwrap()
            .str().unwrap()
            .into_no_null_iter()
            .collect();
        
        assert!(unique_ids.contains(&"series_1"));
        assert!(unique_ids.contains(&"series_2"));
    }

    #[test]
    fn test_multiple_models_comparison() {
        // Create two different model configurations
        let config1 = MockMLPConfig::new(10, 5).with_hidden_layers(vec![32]);
        let config2 = MockMLPConfig::new(10, 5).with_hidden_layers(vec![64, 32]);

        let mut model1 = MockMLP::new(config1).unwrap();
        let mut model2 = MockMLP::new(config2).unwrap();

        // Create shared training data
        let data = df! {
            "unique_id" => ["test", "test", "test"],
            "ds" => [1, 2, 3],
            "y" => [1.0, 2.0, 3.0],
        }.unwrap();
        
        let ts_data = TimeSeriesDataFrame::from_polars(data, TimeSeriesSchema::default(), None).unwrap();

        // Train both models
        model1.fit(&ts_data).unwrap();
        model2.fit(&ts_data).unwrap();

        // Generate forecasts from both
        let forecast1 = model1.predict(&ts_data).unwrap();
        let forecast2 = model2.predict(&ts_data).unwrap();

        // Both should have the same horizon
        assert_eq!(forecast1.forecast_horizon, forecast2.forecast_horizon);
        
        // Both should have forecasts for the same series
        assert_eq!(forecast1.shape().0, forecast2.shape().0);
    }
}