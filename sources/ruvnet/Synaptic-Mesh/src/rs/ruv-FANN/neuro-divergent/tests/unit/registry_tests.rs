//! Unit tests for the model registry and factory system
//!
//! This module tests model registration, discovery, factory patterns,
//! and basic registry functionality.

use neuro_divergent::prelude::*;
use neuro_divergent::{AccuracyMetrics, NeuroDivergentError, NeuroDivergentResult};
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};
use neuro_divergent::core::{BaseModel, ModelConfig};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use num_traits::Float;
use proptest::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Mock Registry Infrastructure for Testing
// ============================================================================

/// Mock model information for registry
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MockModelInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub model_type: String,
    pub capabilities: Vec<String>,
    pub supported_data_types: Vec<String>,
}

impl MockModelInfo {
    pub fn new(name: String, model_type: String) -> Self {
        Self {
            name: name.clone(),
            version: "1.0.0".to_string(),
            description: format!("Mock {} model for testing", name),
            model_type,
            capabilities: vec!["forecasting".to_string(), "training".to_string()],
            supported_data_types: vec!["f32".to_string(), "f64".to_string()],
        }
    }
}

/// Mock model registry
#[derive(Clone, Debug)]
struct MockModelRegistry {
    models: Arc<Mutex<HashMap<String, MockModelInfo>>>,
}

impl MockModelRegistry {
    pub fn new() -> Self {
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn register(&self, name: &str, info: MockModelInfo) -> NeuroDivergentResult<()> {
        let mut models = self.models.lock().unwrap();
        
        if models.contains_key(name) {
            return Err(NeuroDivergentError::config(
                format!("Model '{}' is already registered", name)
            ));
        }
        
        models.insert(name.to_string(), info);
        Ok(())
    }
    
    pub fn unregister(&self, name: &str) -> NeuroDivergentResult<()> {
        let mut models = self.models.lock().unwrap();
        
        if models.remove(name).is_none() {
            return Err(NeuroDivergentError::config(
                format!("Model '{}' is not registered", name)
            ));
        }
        
        Ok(())
    }
    
    pub fn get_model_info(&self, name: &str) -> Option<MockModelInfo> {
        let models = self.models.lock().unwrap();
        models.get(name).cloned()
    }
    
    pub fn list_models(&self) -> Vec<String> {
        let models = self.models.lock().unwrap();
        models.keys().cloned().collect()
    }
    
    pub fn find_models_by_type(&self, model_type: &str) -> Vec<MockModelInfo> {
        let models = self.models.lock().unwrap();
        models.values()
            .filter(|info| info.model_type == model_type)
            .cloned()
            .collect()
    }
    
    pub fn find_models_by_capability(&self, capability: &str) -> Vec<MockModelInfo> {
        let models = self.models.lock().unwrap();
        models.values()
            .filter(|info| info.capabilities.contains(&capability.to_string()))
            .cloned()
            .collect()
    }
}

/// Mock model factory
#[derive(Clone, Debug)]
struct MockModelFactory {
    registry: MockModelRegistry,
}

impl MockModelFactory {
    pub fn new(registry: MockModelRegistry) -> Self {
        Self { registry }
    }
    
    pub fn create_model<T: Float>(&self, name: &str) -> NeuroDivergentResult<Box<dyn ModelBuilder<T>>> {
        let model_info = self.registry.get_model_info(name)
            .ok_or_else(|| NeuroDivergentError::config(
                format!("Model '{}' not found in registry", name)
            ))?;
        
        // In a real implementation, this would use a registry of constructors
        match model_info.model_type.as_str() {
            "MockMLP" => Ok(Box::new(MockMLPBuilder::new())),
            "MockLSTM" => Ok(Box::new(MockLSTMBuilder::new())),
            _ => Err(NeuroDivergentError::config(
                format!("Unknown model type: {}", model_info.model_type)
            ))
        }
    }
    
    pub fn get_available_models(&self) -> Vec<String> {
        self.registry.list_models()
    }
}

/// Trait for model builders
trait ModelBuilder<T: Float>: Send + Sync {
    fn build(&self, config: HashMap<String, String>) -> NeuroDivergentResult<Box<dyn TestableModel<T>>>;
    fn get_default_config(&self) -> HashMap<String, String>;
    fn validate_config(&self, config: &HashMap<String, String>) -> NeuroDivergentResult<()>;
}

/// Simplified testable model trait
trait TestableModel<T: Float>: Send + Sync {
    fn name(&self) -> &str;
    fn model_type(&self) -> &str;
    fn fit(&mut self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()>;
    fn predict(&self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<ForecastDataFrame<T>>;
    fn is_fitted(&self) -> bool;
}

/// Mock MLP model builder
struct MockMLPBuilder;

impl MockMLPBuilder {
    fn new() -> Self {
        Self
    }
}

impl<T: Float> ModelBuilder<T> for MockMLPBuilder {
    fn build(&self, config: HashMap<String, String>) -> NeuroDivergentResult<Box<dyn TestableModel<T>>> {
        self.validate_config(&config)?;
        
        let hidden_size = config.get("hidden_size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
            
        Ok(Box::new(MockMLPModel::new("MockMLP".to_string(), hidden_size)))
    }
    
    fn get_default_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("hidden_size".to_string(), "64".to_string());
        config.insert("learning_rate".to_string(), "0.001".to_string());
        config.insert("max_epochs".to_string(), "100".to_string());
        config
    }
    
    fn validate_config(&self, config: &HashMap<String, String>) -> NeuroDivergentResult<()> {
        if let Some(hidden_size_str) = config.get("hidden_size") {
            let hidden_size: usize = hidden_size_str.parse()
                .map_err(|_| NeuroDivergentError::config("Invalid hidden_size format"))?;
            
            if hidden_size == 0 {
                return Err(NeuroDivergentError::config("hidden_size must be greater than 0"));
            }
        }
        
        Ok(())
    }
}

/// Mock LSTM model builder
struct MockLSTMBuilder;

impl MockLSTMBuilder {
    fn new() -> Self {
        Self
    }
}

impl<T: Float> ModelBuilder<T> for MockLSTMBuilder {
    fn build(&self, config: HashMap<String, String>) -> NeuroDivergentResult<Box<dyn TestableModel<T>>> {
        self.validate_config(&config)?;
        
        let hidden_size = config.get("hidden_size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(128);
            
        Ok(Box::new(MockLSTMModel::new("MockLSTM".to_string(), hidden_size)))
    }
    
    fn get_default_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("hidden_size".to_string(), "128".to_string());
        config.insert("num_layers".to_string(), "2".to_string());
        config.insert("dropout".to_string(), "0.1".to_string());
        config
    }
    
    fn validate_config(&self, config: &HashMap<String, String>) -> NeuroDivergentResult<()> {
        if let Some(num_layers_str) = config.get("num_layers") {
            let num_layers: usize = num_layers_str.parse()
                .map_err(|_| NeuroDivergentError::config("Invalid num_layers format"))?;
            
            if num_layers == 0 {
                return Err(NeuroDivergentError::config("num_layers must be greater than 0"));
            }
        }
        
        Ok(())
    }
}

/// Mock MLP model implementation
struct MockMLPModel<T: Float> {
    name: String,
    hidden_size: usize,
    fitted: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> MockMLPModel<T> {
    fn new(name: String, hidden_size: usize) -> Self {
        Self {
            name,
            hidden_size,
            fitted: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float> TestableModel<T> for MockMLPModel<T> {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn model_type(&self) -> &str {
        "MockMLP"
    }
    
    fn fit(&mut self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()> {
        if data.shape().0 == 0 {
            return Err(NeuroDivergentError::data("No data to fit"));
        }
        self.fitted = true;
        Ok(())
    }
    
    fn predict(&self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        if !self.fitted {
            return Err(NeuroDivergentError::prediction("Model not fitted"));
        }
        
        // Create mock forecast data
        use polars::prelude::*;
        let df = df! {
            "unique_id" => vec!["test"],
            "ds" => vec![1],
            "MockMLP" => vec![1.0],
        }.map_err(|e| NeuroDivergentError::prediction(format!("DataFrame creation failed: {}", e)))?;
        
        Ok(ForecastDataFrame::new(
            df,
            vec![self.name.clone()],
            1,
            None,
            TimeSeriesSchema::default(),
        ))
    }
    
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Mock LSTM model implementation
struct MockLSTMModel<T: Float> {
    name: String,
    hidden_size: usize,
    fitted: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> MockLSTMModel<T> {
    fn new(name: String, hidden_size: usize) -> Self {
        Self {
            name,
            hidden_size,
            fitted: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float> TestableModel<T> for MockLSTMModel<T> {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn model_type(&self) -> &str {
        "MockLSTM"
    }
    
    fn fit(&mut self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()> {
        if data.shape().0 == 0 {
            return Err(NeuroDivergentError::data("No data to fit"));
        }
        self.fitted = true;
        Ok(())
    }
    
    fn predict(&self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        if !self.fitted {
            return Err(NeuroDivergentError::prediction("Model not fitted"));
        }
        
        // Create mock forecast data
        use polars::prelude::*;
        let df = df! {
            "unique_id" => vec!["test"],
            "ds" => vec![1],
            "MockLSTM" => vec![2.0],
        }.map_err(|e| NeuroDivergentError::prediction(format!("DataFrame creation failed: {}", e)))?;
        
        Ok(ForecastDataFrame::new(
            df,
            vec![self.name.clone()],
            1,
            None,
            TimeSeriesSchema::default(),
        ))
    }
    
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ============================================================================
// Registry Tests
// ============================================================================

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn test_model_registry_registration() {
        let registry = MockModelRegistry::new();
        let model_info = MockModelInfo::new("TestMLP".to_string(), "MockMLP".to_string());
        
        // Register model
        let result = registry.register("test_mlp", model_info.clone());
        assert!(result.is_ok());
        
        // Verify registration
        let retrieved_info = registry.get_model_info("test_mlp");
        assert!(retrieved_info.is_some());
        let retrieved = retrieved_info.unwrap();
        assert_eq!(retrieved.name, model_info.name);
        assert_eq!(retrieved.model_type, model_info.model_type);
    }

    #[test]
    fn test_model_registry_duplicate_registration() {
        let registry = MockModelRegistry::new();
        let model_info = MockModelInfo::new("TestMLP".to_string(), "MockMLP".to_string());
        
        // Register model first time
        registry.register("test_mlp", model_info.clone()).unwrap();
        
        // Try to register again - should fail
        let result = registry.register("test_mlp", model_info);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_registry_unregistration() {
        let registry = MockModelRegistry::new();
        let model_info = MockModelInfo::new("TestMLP".to_string(), "MockMLP".to_string());
        
        // Register and then unregister
        registry.register("test_mlp", model_info).unwrap();
        let result = registry.unregister("test_mlp");
        assert!(result.is_ok());
        
        // Verify it's gone
        let retrieved_info = registry.get_model_info("test_mlp");
        assert!(retrieved_info.is_none());
    }

    #[test]
    fn test_model_registry_unregister_nonexistent() {
        let registry = MockModelRegistry::new();
        
        // Try to unregister non-existent model
        let result = registry.unregister("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_registry_list_models() {
        let registry = MockModelRegistry::new();
        
        let model1 = MockModelInfo::new("MLP1".to_string(), "MockMLP".to_string());
        let model2 = MockModelInfo::new("LSTM1".to_string(), "MockLSTM".to_string());
        
        registry.register("mlp1", model1).unwrap();
        registry.register("lstm1", model2).unwrap();
        
        let models = registry.list_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"mlp1".to_string()));
        assert!(models.contains(&"lstm1".to_string()));
    }

    #[test]
    fn test_model_registry_find_by_type() {
        let registry = MockModelRegistry::new();
        
        let mlp1 = MockModelInfo::new("MLP1".to_string(), "MockMLP".to_string());
        let mlp2 = MockModelInfo::new("MLP2".to_string(), "MockMLP".to_string());
        let lstm1 = MockModelInfo::new("LSTM1".to_string(), "MockLSTM".to_string());
        
        registry.register("mlp1", mlp1).unwrap();
        registry.register("mlp2", mlp2).unwrap();
        registry.register("lstm1", lstm1).unwrap();
        
        let mlp_models = registry.find_models_by_type("MockMLP");
        assert_eq!(mlp_models.len(), 2);
        
        let lstm_models = registry.find_models_by_type("MockLSTM");
        assert_eq!(lstm_models.len(), 1);
        
        let nonexistent_models = registry.find_models_by_type("NonExistent");
        assert_eq!(nonexistent_models.len(), 0);
    }

    #[test]
    fn test_model_registry_find_by_capability() {
        let registry = MockModelRegistry::new();
        
        let mut model_info = MockModelInfo::new("SpecialModel".to_string(), "Special".to_string());
        model_info.capabilities = vec!["forecasting".to_string(), "clustering".to_string()];
        
        registry.register("special", model_info).unwrap();
        
        let forecasting_models = registry.find_models_by_capability("forecasting");
        assert_eq!(forecasting_models.len(), 1);
        
        let clustering_models = registry.find_models_by_capability("clustering");
        assert_eq!(clustering_models.len(), 1);
        
        let missing_models = registry.find_models_by_capability("missing_capability");
        assert_eq!(missing_models.len(), 0);
    }
}

// ============================================================================
// Factory Tests
// ============================================================================

#[cfg(test)]
mod factory_tests {
    use super::*;

    #[test]
    fn test_model_factory_creation() {
        let registry = MockModelRegistry::new();
        let mlp_info = MockModelInfo::new("TestMLP".to_string(), "MockMLP".to_string());
        registry.register("test_mlp", mlp_info).unwrap();
        
        let factory = MockModelFactory::new(registry);
        let model_builder = factory.create_model::<f64>("test_mlp");
        assert!(model_builder.is_ok());
    }

    #[test]
    fn test_model_factory_nonexistent_model() {
        let registry = MockModelRegistry::new();
        let factory = MockModelFactory::new(registry);
        
        let result = factory.create_model::<f64>("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_factory_available_models() {
        let registry = MockModelRegistry::new();
        
        let mlp_info = MockModelInfo::new("TestMLP".to_string(), "MockMLP".to_string());
        let lstm_info = MockModelInfo::new("TestLSTM".to_string(), "MockLSTM".to_string());
        
        registry.register("test_mlp", mlp_info).unwrap();
        registry.register("test_lstm", lstm_info).unwrap();
        
        let factory = MockModelFactory::new(registry);
        let available = factory.get_available_models();
        
        assert_eq!(available.len(), 2);
        assert!(available.contains(&"test_mlp".to_string()));
        assert!(available.contains(&"test_lstm".to_string()));
    }
}

// ============================================================================
// Model Builder Tests
// ============================================================================

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn test_mlp_builder_default_config() {
        let builder = MockMLPBuilder::new();
        let config = builder.get_default_config();
        
        assert!(config.contains_key("hidden_size"));
        assert!(config.contains_key("learning_rate"));
        assert!(config.contains_key("max_epochs"));
    }

    #[test]
    fn test_mlp_builder_validation() {
        let builder = MockMLPBuilder::new();
        
        // Valid config
        let mut valid_config = HashMap::new();
        valid_config.insert("hidden_size".to_string(), "64".to_string());
        assert!(builder.validate_config(&valid_config).is_ok());
        
        // Invalid config - zero hidden size
        let mut invalid_config = HashMap::new();
        invalid_config.insert("hidden_size".to_string(), "0".to_string());
        assert!(builder.validate_config(&invalid_config).is_err());
        
        // Invalid config - non-numeric hidden size
        let mut invalid_config = HashMap::new();
        invalid_config.insert("hidden_size".to_string(), "invalid".to_string());
        assert!(builder.validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_lstm_builder_validation() {
        let builder = MockLSTMBuilder::new();
        
        // Valid config
        let mut valid_config = HashMap::new();
        valid_config.insert("num_layers".to_string(), "2".to_string());
        assert!(builder.validate_config(&valid_config).is_ok());
        
        // Invalid config - zero layers
        let mut invalid_config = HashMap::new();
        invalid_config.insert("num_layers".to_string(), "0".to_string());
        assert!(builder.validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_model_builder_build() {
        let builder = MockMLPBuilder::new();
        let config = builder.get_default_config();
        
        let model = builder.build(config);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.model_type(), "MockMLP");
        assert!(!model.is_fitted());
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_complete_registry_workflow() {
        // Create registry and register models
        let registry = MockModelRegistry::new();
        
        let mlp_info = MockModelInfo::new("ProductionMLP".to_string(), "MockMLP".to_string());
        let lstm_info = MockModelInfo::new("ProductionLSTM".to_string(), "MockLSTM".to_string());
        
        registry.register("prod_mlp", mlp_info).unwrap();
        registry.register("prod_lstm", lstm_info).unwrap();
        
        // Create factory and build models
        let factory = MockModelFactory::new(registry);
        
        let mlp_builder = factory.create_model::<f64>("prod_mlp").unwrap();
        let lstm_builder = factory.create_model::<f64>("prod_lstm").unwrap();
        
        // Build models with default configs
        let mlp_config = mlp_builder.get_default_config();
        let lstm_config = lstm_builder.get_default_config();
        
        let mut mlp_model = mlp_builder.build(mlp_config).unwrap();
        let mut lstm_model = lstm_builder.build(lstm_config).unwrap();
        
        // Create test data
        let data = df! {
            "unique_id" => ["test"],
            "ds" => [1],
            "y" => [10.0],
        }.unwrap();
        
        let ts_data = TimeSeriesDataFrame::<f64>::from_polars(
            data, 
            TimeSeriesSchema::default(), 
            None
        ).unwrap();
        
        // Train models
        mlp_model.fit(&ts_data).unwrap();
        lstm_model.fit(&ts_data).unwrap();
        
        assert!(mlp_model.is_fitted());
        assert!(lstm_model.is_fitted());
        
        // Generate predictions
        let mlp_forecast = mlp_model.predict(&ts_data).unwrap();
        let lstm_forecast = lstm_model.predict(&ts_data).unwrap();
        
        assert_eq!(mlp_forecast.models, vec!["ProductionMLP".to_string()]);
        assert_eq!(lstm_forecast.models, vec!["ProductionLSTM".to_string()]);
    }

    #[test]
    fn test_registry_concurrent_access() {
        use std::thread;
        use std::sync::Arc;
        
        let registry = Arc::new(MockModelRegistry::new());
        let mut handles = vec![];
        
        // Spawn multiple threads to register models concurrently
        for i in 0..5 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                let model_info = MockModelInfo::new(
                    format!("Model{}", i), 
                    "MockMLP".to_string()
                );
                registry_clone.register(&format!("model_{}", i), model_info)
            });
            handles.push(handle);
        }
        
        // Wait for all registrations to complete
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }
        
        // Verify all models were registered
        let models = registry.list_models();
        assert_eq!(models.len(), 5);
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
        fn test_registry_invariants(
            model_names in prop::collection::vec("[a-z_][a-z0-9_]*", 1..10)
        ) {
            let registry = MockModelRegistry::new();
            let unique_names: std::collections::HashSet<_> = model_names.iter().collect();
            
            // Register all unique models
            for name in &unique_names {
                let model_info = MockModelInfo::new(
                    format!("Model_{}", name), 
                    "MockMLP".to_string()
                );
                let result = registry.register(name, model_info);
                assert!(result.is_ok());
            }
            
            // Verify count matches
            let registered_models = registry.list_models();
            assert_eq!(registered_models.len(), unique_names.len());
            
            // Verify all names are present
            for name in &unique_names {
                assert!(registered_models.contains(&name.to_string()));
                assert!(registry.get_model_info(name).is_some());
            }
        }

        #[test]
        fn test_model_builder_config_validation(
            hidden_size in 1usize..1000
        ) {
            let builder = MockMLPBuilder::new();
            let mut config = HashMap::new();
            config.insert("hidden_size".to_string(), hidden_size.to_string());
            
            // All positive hidden sizes should be valid
            assert!(builder.validate_config(&config).is_ok());
            
            // Building should succeed
            let model = builder.build(config);
            assert!(model.is_ok());
            
            let model = model.unwrap();
            assert_eq!(model.model_type(), "MockMLP");
            assert!(!model.is_fitted());
        }
    }
}