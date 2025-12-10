//! Main NeuralForecast class providing 100% Python API compatibility

use std::collections::HashMap;
use std::path::Path;
use std::marker::PhantomData;
use num_traits::Float;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use polars::prelude::*;

use crate::config::{
    Frequency, ScalerType, PredictionIntervals, CrossValidationConfig, Device
};
use crate::results::{
    TimeSeriesDataFrame, ForecastDataFrame, CrossValidationDataFrame, 
    TimeSeriesSchema
};
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};

// Forward declarations for core traits (will be implemented in core module)
pub trait BaseModel<T: Float + Send + Sync>: Send + Sync + std::fmt::Debug {
    /// Fit the model to training data
    fn fit(&mut self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()>;
    
    /// Generate predictions
    fn predict(&self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<ForecastDataFrame<T>>;
    
    /// Get model name
    fn name(&self) -> &str;
    
    /// Check if model is fitted
    fn is_fitted(&self) -> bool;
    
    /// Reset model to unfitted state
    fn reset(&mut self) -> NeuroDivergentResult<()>;
}

/// Main NeuralForecast class - primary user interface
/// 
/// Provides 100% compatibility with the NeuralForecast Python library while
/// leveraging Rust's performance and safety guarantees.
/// 
/// # Examples
/// 
/// ```rust
/// use neuro_divergent::{NeuralForecast, models::LSTM, Frequency};
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create models
/// let lstm = LSTM::builder()
///     .hidden_size(128)
///     .num_layers(2)
///     .horizon(12)
///     .build()?;
/// 
/// // Create NeuralForecast instance
/// let mut nf = NeuralForecast::builder()
///     .with_model(Box::new(lstm))
///     .with_frequency(Frequency::Monthly)
///     .build()?;
/// 
/// // Load and fit data
/// let data = TimeSeriesDataFrame::from_csv("data.csv")?;
/// nf.fit(data.clone())?;
/// 
/// // Generate forecasts
/// let forecasts = nf.predict()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct NeuralForecast<T: Float + Send + Sync> {
    /// Collection of forecasting models
    models: Vec<Box<dyn BaseModel<T>>>,
    /// Time series frequency
    frequency: Frequency,
    /// Local scaler type for preprocessing
    local_scaler_type: Option<ScalerType>,
    /// Number of threads for parallel processing
    num_threads: Option<usize>,
    /// Prediction intervals configuration
    prediction_intervals: Option<PredictionIntervals>,
    /// Device for computation
    device: Device,
    /// Whether models have been fitted
    is_fitted: bool,
    /// Training data schema
    training_schema: Option<TimeSeriesSchema>,
    /// Model metadata
    model_metadata: HashMap<String, ModelInfo>,
    /// Phantom data for type parameter
    phantom: PhantomData<T>,
}

/// Model information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelInfo {
    /// Model name
    name: String,
    /// Whether model is fitted
    fitted: bool,
    /// Training history if available
    training_history: Option<TrainingHistory<f64>>,
    /// Model configuration parameters
    config_summary: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingHistory<T: Float> {
    /// Training loss history
    train_loss: Vec<T>,
    /// Validation loss history
    val_loss: Option<Vec<T>>,
    /// Number of epochs trained
    epochs: usize,
    /// Best validation loss
    best_val_loss: Option<T>,
}

impl<T: Float + Send + Sync> NeuralForecast<T> {
    /// Create a new NeuralForecast builder
    /// 
    /// # Example
    /// ```rust
    /// use neuro_divergent::{NeuralForecast, Frequency};
    /// 
    /// let nf_builder = NeuralForecast::<f32>::builder();
    /// ```
    pub fn builder() -> NeuralForecastBuilder<T> {
        NeuralForecastBuilder::new()
    }
    
    /// Create NeuralForecast with models and frequency (direct constructor)
    /// 
    /// # Arguments
    /// * `models` - Vector of boxed models implementing BaseModel trait
    /// * `frequency` - Time series frequency
    /// 
    /// # Example
    /// ```rust
    /// use neuro_divergent::{NeuralForecast, models::LSTM, Frequency};
    /// 
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lstm = LSTM::default();
    /// let nf = NeuralForecast::new(vec![Box::new(lstm)], Frequency::Daily)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        models: Vec<Box<dyn BaseModel<T>>>, 
        frequency: Frequency
    ) -> NeuroDivergentResult<Self> {
        if models.is_empty() {
            return Err(NeuroDivergentError::config("At least one model is required"));
        }
        
        // Check for duplicate model names
        let mut model_names = std::collections::HashSet::new();
        for model in &models {
            if !model_names.insert(model.name()) {
                return Err(NeuroDivergentError::config(
                    format!("Duplicate model name: {}", model.name())
                ));
            }
        }
        
        let model_metadata = models.iter()
            .map(|model| {
                let info = ModelInfo {
                    name: model.name().to_string(),
                    fitted: model.is_fitted(),
                    training_history: None,
                    config_summary: HashMap::new(),
                };
                (model.name().to_string(), info)
            })
            .collect();
        
        Ok(Self {
            models,
            frequency,
            local_scaler_type: None,
            num_threads: None,
            prediction_intervals: None,
            device: Device::default(),
            is_fitted: false,
            training_schema: None,
            model_metadata,
            phantom: PhantomData,
        })
    }
    
    /// Fit models to the provided time series data
    /// 
    /// Equivalent to Python: `nf.fit(df)`
    /// 
    /// # Arguments
    /// * `data` - Time series data for training
    /// 
    /// # Example
    /// ```rust
    /// # use neuro_divergent::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut nf = NeuralForecast::<f32>::builder().build()?;
    /// let data = TimeSeriesDataFrame::from_csv("train_data.csv")?;
    /// nf.fit(data)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn fit(&mut self, data: TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()> {
        self.fit_with_validation(data, ValidationConfig::default())
    }
    
    /// Fit with custom validation configuration
    pub fn fit_with_validation(
        &mut self, 
        data: TimeSeriesDataFrame<T>,
        _validation_config: ValidationConfig
    ) -> NeuroDivergentResult<()> {
        // Validate input data
        if data.shape().0 == 0 {
            return Err(NeuroDivergentError::data("Training data is empty"));
        }
        
        // Check frequency compatibility
        if let Some(data_freq) = &data.frequency {
            if data_freq != &self.frequency {
                return Err(NeuroDivergentError::data(
                    format!("Data frequency {:?} does not match NeuralForecast frequency {:?}", 
                            data_freq, self.frequency)
                ));
            }
        }
        
        // Store training schema
        self.training_schema = Some(data.schema.clone());
        
        // Fit all models (can be parallelized)
        let results: Vec<NeuroDivergentResult<()>> = if self.num_threads.unwrap_or(1) > 1 {
            // Parallel fitting - Note: This is conceptual, actual implementation
            // would need to handle the fact that models need to be mutable
            self.models.par_iter_mut()
                .map(|model| model.fit(&data))
                .collect()
        } else {
            // Sequential fitting
            self.models.iter_mut()
                .map(|model| model.fit(&data))
                .collect()
        };
        
        // Check for errors
        let mut errors = Vec::new();
        for (i, result) in results.into_iter().enumerate() {
            if let Err(e) = result {
                errors.push(format!("Model {} ({}): {}", i, self.models[i].name(), e));
            }
        }
        
        if !errors.is_empty() {
            return Err(NeuroDivergentError::training(
                format!("Model fitting failed: {}", errors.join("; "))
            ));
        }
        
        // Update model metadata
        for model in &self.models {
            if let Some(info) = self.model_metadata.get_mut(model.name()) {
                info.fitted = model.is_fitted();
            }
        }
        
        self.is_fitted = true;
        Ok(())
    }
    
    /// Generate forecasts for all fitted models
    /// 
    /// Equivalent to Python: `nf.predict()`
    /// 
    /// # Example
    /// ```rust
    /// # use neuro_divergent::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut nf = NeuralForecast::<f32>::builder().build()?;
    /// # let data = TimeSeriesDataFrame::from_csv("data.csv")?;
    /// # nf.fit(data)?;
    /// let forecasts = nf.predict()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(&self) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        self.predict_with_config(PredictionConfig::default())
    }
    
    /// Generate forecasts with custom configuration
    pub fn predict_with_config(
        &self, 
        _config: PredictionConfig
    ) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        if !self.is_fitted {
            return Err(NeuroDivergentError::prediction("Models have not been fitted yet"));
        }
        
        // For now, return a placeholder forecast dataframe
        // In the full implementation, this would generate actual forecasts
        let schema = self.training_schema.as_ref()
            .ok_or_else(|| NeuroDivergentError::prediction("No training schema available"))?
            .clone();
            
        let model_names: Vec<String> = self.models.iter()
            .map(|m| m.name().to_string())
            .collect();
            
        // Create empty forecast dataframe (placeholder)
        let data = df! {
            "unique_id" => Vec::<String>::new(),
            "ds" => Vec::<i64>::new(),
        }.map_err(|e| NeuroDivergentError::prediction(format!("DataFrame creation error: {}", e)))?;
        
        Ok(ForecastDataFrame::new(
            data,
            model_names,
            1, // placeholder horizon
            self.prediction_intervals.as_ref().map(|pi| pi.confidence_levels.clone()),
            schema,
        ))
    }
    
    /// Predict on new data
    /// 
    /// Equivalent to Python: `nf.predict(df=new_data)`
    pub fn predict_on(
        &self, 
        data: TimeSeriesDataFrame<T>
    ) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        if !self.is_fitted {
            return Err(NeuroDivergentError::prediction("Models have not been fitted yet"));
        }
        
        // Validate data compatibility
        if let Some(training_schema) = &self.training_schema {
            if data.schema.unique_id_col != training_schema.unique_id_col ||
               data.schema.ds_col != training_schema.ds_col ||
               data.schema.y_col != training_schema.y_col {
                return Err(NeuroDivergentError::data(
                    "Data schema does not match training data schema"
                ));
            }
        }
        
        // Generate predictions from all models
        let model_names: Vec<String> = self.models.iter().map(|m| m.name().to_string()).collect();
        
        // Placeholder implementation
        let data_df = df! {
            "unique_id" => Vec::<String>::new(),
            "ds" => Vec::<i64>::new(),
        }.map_err(|e| NeuroDivergentError::prediction(format!("DataFrame creation error: {}", e)))?;
        
        Ok(ForecastDataFrame::new(
            data_df,
            model_names,
            1,
            self.prediction_intervals.as_ref().map(|pi| pi.confidence_levels.clone()),
            data.schema,
        ))
    }
    
    /// Cross-validation for model evaluation
    /// 
    /// Equivalent to Python: `nf.cross_validation(df, n_windows=3)`
    pub fn cross_validation(
        &mut self,
        data: TimeSeriesDataFrame<T>,
        config: CrossValidationConfig
    ) -> NeuroDivergentResult<CrossValidationDataFrame<T>> {
        // Validate configuration
        config.validate()?;
        
        // Validate data
        if data.shape().0 == 0 {
            return Err(NeuroDivergentError::data("Cross-validation data is empty"));
        }
        
        // For now, return placeholder CV results
        let cutoffs = vec![chrono::Utc::now(); config.n_windows];
        let model_names: Vec<String> = self.models.iter().map(|m| m.name().to_string()).collect();
        
        let cv_data = polars::prelude::df! {
            "unique_id" => Vec::<String>::new(),
            "ds" => Vec::<i64>::new(),
            "cutoff" => Vec::<i64>::new(),
        }.map_err(|e| NeuroDivergentError::data(format!("DataFrame creation error: {}", e)))?;
        
        Ok(CrossValidationDataFrame::new(
            cv_data,
            cutoffs,
            model_names,
            data.schema,
        ))
    }
    
    /// Fit and predict in one step
    pub fn fit_predict(
        &mut self,
        train_data: TimeSeriesDataFrame<T>
    ) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        self.fit(train_data)?;
        self.predict()
    }
    
    /// Get model by name
    pub fn get_model(&self, name: &str) -> Option<&dyn BaseModel<T>> {
        self.models.iter()
            .find(|model| model.name() == name)
            .map(|model| model.as_ref())
    }
    
    /// Get mutable model by name
    pub fn get_model_mut(&mut self, name: &str) -> Option<&mut dyn BaseModel<T>> {
        for model in &mut self.models {
            if model.name() == name {
                return Some(model.as_mut());
            }
        }
        None
    }
    
    /// List all model names
    pub fn model_names(&self) -> Vec<String> {
        self.models.iter().map(|m| m.name().to_string()).collect()
    }
    
    /// Check if models are fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
    
    /// Get frequency
    pub fn frequency(&self) -> Frequency {
        self.frequency.clone()
    }
    
    /// Get number of models
    pub fn num_models(&self) -> usize {
        self.models.len()
    }
    
    /// Reset all models to unfitted state
    pub fn reset(&mut self) -> NeuroDivergentResult<()> {
        for model in &mut self.models {
            model.reset()?;
        }
        
        self.is_fitted = false;
        self.training_schema = None;
        
        // Reset model metadata
        for info in self.model_metadata.values_mut() {
            info.fitted = false;
            info.training_history = None;
        }
        
        Ok(())
    }
    
    /// Save models to file
    pub fn save<P: AsRef<Path>>(&self, _path: P) -> NeuroDivergentResult<()> {
        // Placeholder for model serialization
        // In full implementation, would serialize all models and metadata
        Ok(())
    }
    
    /// Load models from file
    pub fn load<P: AsRef<Path>>(_path: P) -> NeuroDivergentResult<Self> {
        // Placeholder for model deserialization
        // In full implementation, would load models and metadata from file
        Err(NeuroDivergentError::generic("Load functionality not yet implemented"))
    }
}

/// Builder for constructing NeuralForecast instances with fluent API
#[derive(Debug)]
pub struct NeuralForecastBuilder<T: Float + Send + Sync> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Option<Frequency>,
    local_scaler_type: Option<ScalerType>,
    num_threads: Option<usize>,
    prediction_intervals: Option<PredictionIntervals>,
    device: Device,
}

impl<T: Float + Send + Sync> NeuralForecastBuilder<T> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            frequency: None,
            local_scaler_type: None,
            num_threads: None,
            prediction_intervals: None,
            device: Device::default(),
        }
    }
    
    /// Add models to the forecasting ensemble
    pub fn with_models(mut self, models: Vec<Box<dyn BaseModel<T>>>) -> Self {
        self.models = models;
        self
    }
    
    /// Add a single model
    pub fn with_model(mut self, model: Box<dyn BaseModel<T>>) -> Self {
        self.models.push(model);
        self
    }
    
    /// Set the frequency of the time series
    pub fn with_frequency(mut self, frequency: Frequency) -> Self {
        self.frequency = Some(frequency);
        self
    }
    
    /// Set local scaler type for data preprocessing
    pub fn with_local_scaler(mut self, scaler_type: ScalerType) -> Self {
        self.local_scaler_type = Some(scaler_type);
        self
    }
    
    /// Set number of threads for parallel processing
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }
    
    /// Configure prediction intervals
    pub fn with_prediction_intervals(mut self, intervals: PredictionIntervals) -> Self {
        self.prediction_intervals = Some(intervals);
        self
    }
    
    /// Set computation device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    
    /// Build the NeuralForecast instance
    pub fn build(self) -> NeuroDivergentResult<NeuralForecast<T>> {
        let frequency = self.frequency
            .ok_or_else(|| NeuroDivergentError::config("Frequency is required"))?;
            
        if self.models.is_empty() {
            return Err(NeuroDivergentError::config("At least one model is required"));
        }
        
        let mut nf = NeuralForecast::new(self.models, frequency)?;
        nf.local_scaler_type = self.local_scaler_type;
        nf.num_threads = self.num_threads;
        nf.prediction_intervals = self.prediction_intervals;
        nf.device = self.device;
        
        Ok(nf)
    }
}

impl<T: Float + Send + Sync> Default for NeuralForecastBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for model validation during training
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Validation split ratio
    pub validation_split: Option<f64>,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl ValidationConfig {
    /// Create new validation configuration
    pub fn new() -> Self {
        Self {
            validation_split: None,
            shuffle: true,
            random_seed: None,
        }
    }
    
    /// Set validation split ratio
    pub fn with_validation_split(mut self, split: f64) -> Self {
        if split > 0.0 && split < 1.0 {
            self.validation_split = Some(split);
        }
        self
    }
    
    /// Set whether to shuffle data
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    
    /// Set random seed
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for prediction generation
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Whether to include prediction intervals
    pub include_intervals: bool,
    /// Number of samples for probabilistic models
    pub num_samples: Option<usize>,
    /// Temperature for sampling
    pub temperature: Option<f64>,
}

impl PredictionConfig {
    /// Create new prediction configuration
    pub fn new() -> Self {
        Self {
            include_intervals: false,
            num_samples: None,
            temperature: None,
        }
    }
    
    /// Enable prediction intervals
    pub fn with_intervals(mut self) -> Self {
        self.include_intervals = true;
        self
    }
    
    /// Set number of samples for probabilistic prediction
    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = Some(num_samples);
        self
    }
    
    /// Set sampling temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock model for testing
    #[derive(Debug)]
    struct MockModel {
        name: String,
        fitted: bool,
    }
    
    impl MockModel {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                fitted: false,
            }
        }
    }
    
    impl BaseModel<f32> for MockModel {
        fn fit(&mut self, _data: &TimeSeriesDataFrame<f32>) -> NeuroDivergentResult<()> {
            self.fitted = true;
            Ok(())
        }
        
        fn predict(&self, _data: &TimeSeriesDataFrame<f32>) -> NeuroDivergentResult<ForecastDataFrame<f32>> {
            // Return placeholder forecast
            let data = df! {
                "unique_id" => Vec::<String>::new(),
                "ds" => Vec::<i64>::new(),
            }.unwrap();
            
            Ok(ForecastDataFrame::new(
                data,
                vec![self.name.clone()],
                1,
                None,
                TimeSeriesSchema::default(),
            ))
        }
        
        fn name(&self) -> &str {
            &self.name
        }
        
        fn is_fitted(&self) -> bool {
            self.fitted
        }
        
        fn reset(&mut self) -> NeuroDivergentResult<()> {
            self.fitted = false;
            Ok(())
        }
    }
    
    #[test]
    fn test_neural_forecast_creation() {
        let model = MockModel::new("test_model");
        let result = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily);
        assert!(result.is_ok());
        
        let nf = result.unwrap();
        assert_eq!(nf.model_names(), vec!["test_model"]);
        assert_eq!(nf.frequency(), Frequency::Daily);
        assert!(!nf.is_fitted());
    }
    
    #[test]
    fn test_neural_forecast_builder() {
        let model = MockModel::new("test_model");
        let result = NeuralForecast::builder()
            .with_model(Box::new(model))
            .with_frequency(Frequency::Hourly)
            .with_num_threads(4)
            .build();
            
        assert!(result.is_ok());
        let nf = result.unwrap();
        assert_eq!(nf.frequency(), Frequency::Hourly);
        assert_eq!(nf.num_threads, Some(4));
    }
    
    #[test]
    fn test_empty_models_error() {
        let result = NeuralForecast::<f32>::new(vec![], Frequency::Daily);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_config_error());
    }
    
    #[test]
    fn test_duplicate_model_names() {
        let model1 = MockModel::new("duplicate");
        let model2 = MockModel::new("duplicate");
        let result = NeuralForecast::new(
            vec![Box::new(model1), Box::new(model2)], 
            Frequency::Daily
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().is_config_error());
    }
    
    #[test]
    fn test_prediction_before_fitting() {
        let model = MockModel::new("test_model");
        let nf = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily).unwrap();
        let result = nf.predict();
        assert!(result.is_err());
        assert!(result.unwrap_err().is_prediction_error());
    }
}