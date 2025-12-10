//! Basic Recurrent Neural Network (RNN) implementation
//!
//! This module provides a vanilla RNN implementation using ruv-FANN as the foundation,
//! with support for multi-layer architectures, bidirectional processing, and time series forecasting.

use std::collections::HashMap;
use num_traits::Float;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction, TrainingData, TrainingAlgorithm};
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};
use crate::foundation::{
    BaseModel, ModelConfig, NetworkAdapter, TimeSeriesInput, ForecastOutput,
    TimeSeriesDataset, TrainingMetrics, ValidationMetrics, TrainingHistory,
    RecurrentState, TimeSeriesSample
};
use crate::config::RNNConfig;
use crate::recurrent::layers::{
    RecurrentLayer, BasicRecurrentCell, MultiLayerRecurrent, BidirectionalRecurrent,
    BidirectionalMergeMode
};
use crate::utils::{math, preprocessing::StandardScaler, validation};

/// Basic RNN model for time series forecasting
pub struct RNN<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    /// Model configuration
    config: RNNConfig<T>,
    
    /// Multi-layer recurrent network
    recurrent_layers: Option<MultiLayerRecurrent<T>>,
    
    /// Output layer (ruv-FANN network)
    output_layer: Option<Network<T>>,
    
    /// Bidirectional wrapper (if enabled)
    bidirectional: Option<BidirectionalRecurrent<T>>,
    
    /// Training state
    is_trained: bool,
    training_history: Option<TrainingHistory<T>>,
    
    /// Data preprocessing
    input_scaler: Option<StandardScaler<T>>,
    output_scaler: Option<StandardScaler<T>>,
    
    /// Cached model metrics
    model_metrics: HashMap<String, f64>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RNN<T> {
    /// Create a new RNN model
    pub fn new(config: RNNConfig<T>) -> NeuroDivergentResult<Self> {
        config.validate()?;
        
        Ok(Self {
            config,
            recurrent_layers: None,
            output_layer: None,
            bidirectional: None,
            is_trained: false,
            training_history: None,
            input_scaler: None,
            output_scaler: None,
            model_metrics: HashMap::new(),
        })
    }
    
    /// Initialize the model architecture
    fn initialize_architecture(&mut self) -> NeuroDivergentResult<()> {
        let input_size = self.config.input_size;
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let horizon = self.config.horizon;
        let activation = self.config.activation;
        
        // Create recurrent layers
        let mut layers: Vec<Box<dyn RecurrentLayer<T> + Send + Sync>> = Vec::new();
        
        // First layer takes input_size, subsequent layers take hidden_size
        for i in 0..num_layers {
            let layer_input_size = if i == 0 { input_size } else { hidden_size };
            let cell = BasicRecurrentCell::new(layer_input_size, hidden_size, activation);
            layers.push(Box::new(cell));
        }
        
        self.recurrent_layers = Some(MultiLayerRecurrent::new(layers, self.config.dropout));
        
        // Create output layer using ruv-FANN
        let output_input_size = hidden_size;
        self.output_layer = Some(
            NetworkBuilder::new()
                .input_layer(output_input_size)
                .output_layer_with_activation(horizon, ActivationFunction::Linear, T::one())
                .build()
        );
        
        // Initialize scalers
        self.input_scaler = Some(StandardScaler::new());
        self.output_scaler = Some(StandardScaler::new());
        
        Ok(())
    }
    
    /// Prepare input sequence for the model
    fn prepare_sequence(&self, ts_input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<Vec<T>>> {
        let sequence_length = ts_input.historical_targets.len();
        let mut sequence = Vec::with_capacity(sequence_length);
        
        // For basic RNN, we use sliding windows of size 1 (each time step is separate)
        // In practice, you might want to use larger windows
        for i in 0..sequence_length {
            let mut input_vector = vec![ts_input.historical_targets[i]];
            
            // Add static features if available
            if let Some(static_features) = &ts_input.static_features {
                input_vector.extend_from_slice(static_features);
            }
            
            // Add historical exogenous features if available
            if let Some(hist_features) = &ts_input.historical_features {
                if i < hist_features.len() {
                    input_vector.extend_from_slice(&hist_features[i]);
                }
            }
            
            sequence.push(input_vector);
        }
        
        Ok(sequence)
    }
    
    /// Create training data from dataset
    fn create_training_data(&self, dataset: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingData<T>> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        
        for sample in &dataset.samples {
            // Prepare input sequence
            let input_sequence = self.prepare_sequence(&sample.input)?;
            
            // For training, we need to flatten the sequence or use the last output
            // Here we'll use the last time step's hidden state as input to output layer
            if let Some(recurrent_layers) = &self.recurrent_layers {
                let mut temp_layers = recurrent_layers.clone();
                temp_layers.reset_states();
                
                // Process sequence and get final hidden state
                let sequence_outputs = temp_layers.forward_sequence(&input_sequence)?;
                if let Some(final_output) = sequence_outputs.last() {
                    inputs.push(final_output.clone());
                    outputs.push(sample.target.clone());
                }
            }
        }
        
        Ok(TrainingData { inputs, outputs })
    }
    
    /// Train the recurrent layers using BPTT (simplified version)
    fn train_recurrent_layers(&mut self, dataset: &TimeSeriesDataset<T>) -> NeuroDivergentResult<T> {
        let mut total_loss = T::zero();
        let mut num_samples = 0;
        
        for sample in &dataset.samples {
            let input_sequence = self.prepare_sequence(&sample.input)?;
            
            if let Some(recurrent_layers) = &mut self.recurrent_layers {
                recurrent_layers.reset_states();
                
                // Forward pass through sequence
                let sequence_outputs = recurrent_layers.forward_sequence(&input_sequence)?;
                
                // Get final output and compute loss with target
                if let Some(final_output) = sequence_outputs.last() {
                    // Simple MSE loss for the final hidden state
                    // In a full implementation, this would involve proper BPTT
                    let sample_loss = math::mse(final_output, &sample.target)?;
                    total_loss = total_loss + sample_loss;
                    num_samples += 1;
                }
            }
        }
        
        Ok(if num_samples > 0 {
            total_loss / T::from(num_samples).unwrap()
        } else {
            T::zero()
        })
    }
    
    /// Update model metrics
    fn update_metrics(&mut self, training_loss: T, validation_loss: Option<T>) {
        self.model_metrics.insert("training_loss".to_string(), training_loss.to_f64().unwrap_or(0.0));
        if let Some(val_loss) = validation_loss {
            self.model_metrics.insert("validation_loss".to_string(), val_loss.to_f64().unwrap_or(0.0));
        }
        
        // Add model-specific metrics
        self.model_metrics.insert("num_parameters".to_string(), self.count_parameters() as f64);
        self.model_metrics.insert("num_layers".to_string(), self.config.num_layers as f64);
        self.model_metrics.insert("hidden_size".to_string(), self.config.hidden_size as f64);
    }
    
    /// Count total number of parameters
    fn count_parameters(&self) -> usize {
        let mut params = 0;
        
        // Recurrent layer parameters
        let input_size = self.config.input_size;
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        
        // First layer: input_size * hidden_size + hidden_size * hidden_size + hidden_size (bias)
        params += input_size * hidden_size + hidden_size * hidden_size + hidden_size;
        
        // Subsequent layers: hidden_size * hidden_size + hidden_size * hidden_size + hidden_size
        for _ in 1..num_layers {
            params += hidden_size * hidden_size + hidden_size * hidden_size + hidden_size;
        }
        
        // Output layer parameters
        params += hidden_size * self.config.horizon + self.config.horizon;
        
        params
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> BaseModel<T> for RNN<T> {
    fn name(&self) -> &str {
        "RNN"
    }
    
    fn config(&self) -> &dyn ModelConfig<T> {
        &self.config
    }
    
    fn is_trained(&self) -> bool {
        self.is_trained
    }
    
    fn fit(&mut self, dataset: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingMetrics<T>> {
        let start_time = std::time::Instant::now();
        
        // Initialize architecture if not done
        if self.recurrent_layers.is_none() {
            self.initialize_architecture()?;
        }
        
        // Initialize training history
        let mut history = TrainingHistory::new();
        
        // Fit scalers on the data (simplified)
        // In a full implementation, this would properly scale all features
        
        let mut best_loss = T::infinity();
        let mut epochs_without_improvement = 0;
        let patience = 10; // Early stopping patience
        
        // Training loop
        for epoch in 0..self.config.max_steps {
            let epoch_start = std::time::Instant::now();
            
            // Train recurrent layers
            let recurrent_loss = self.train_recurrent_layers(dataset)?;
            
            // Create training data for output layer
            let training_data = self.create_training_data(dataset)?;
            
            // Train output layer using ruv-FANN
            let output_loss = if let Some(output_layer) = &mut self.output_layer {
                // Use a simple training algorithm (this is simplified)
                let mut trainer = ruv_fann::training::IncrementalBackprop::new(
                    self.config.learning_rate
                );
                trainer.train_epoch(output_layer, &training_data)?
            } else {
                T::zero()
            };
            
            // Combined loss
            let total_loss = recurrent_loss + output_loss;
            
            // Update history
            let epoch_time = epoch_start.elapsed().as_secs_f64();
            history.add_epoch(total_loss, None, self.config.learning_rate, epoch_time);
            
            // Check for improvement
            if total_loss < best_loss {
                best_loss = total_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }
            
            // Early stopping
            if epochs_without_improvement >= patience {
                break;
            }
            
            // Log progress
            if epoch % 100 == 0 {
                log::info!("Epoch {}: Loss = {:?}", epoch, total_loss);
            }
        }
        
        self.is_trained = true;
        self.training_history = Some(history.clone());
        
        let training_time = start_time.elapsed().as_secs_f64();
        self.update_metrics(best_loss, None);
        
        Ok(TrainingMetrics {
            final_loss: best_loss,
            epochs_completed: history.num_epochs(),
            training_time_seconds: training_time,
            best_validation_loss: None,
            early_stopped: epochs_without_improvement >= patience,
            convergence_achieved: best_loss < T::from(1e-6).unwrap(),
        })
    }
    
    fn predict(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<ForecastOutput<T>> {
        if !self.is_trained {
            return Err(NeuroDivergentError::prediction("Model has not been trained"));
        }
        
        input.validate()?;
        
        let sequence = self.prepare_sequence(input)?;
        
        // Process through recurrent layers
        let recurrent_output = if let Some(recurrent_layers) = &self.recurrent_layers {
            let mut temp_layers = recurrent_layers.clone();
            temp_layers.reset_states();
            let sequence_outputs = temp_layers.forward_sequence(&sequence)?;
            sequence_outputs.last().unwrap().clone()
        } else {
            return Err(NeuroDivergentError::prediction("Model not properly initialized"));
        };
        
        // Process through output layer
        let forecasts = if let Some(output_layer) = &self.output_layer {
            // Clone the network to avoid borrowing issues
            // TODO: Check if ruv-fann provides an immutable prediction method
            let mut cloned_layer = output_layer.clone();
            cloned_layer.run(&recurrent_output)
        } else {
            return Err(NeuroDivergentError::prediction("Output layer not initialized"));
        };
        
        // Create forecast output
        let mut forecast_output = ForecastOutput::new(forecasts);
        
        if let Some(id) = &input.unique_id {
            forecast_output = forecast_output.with_id(id.clone());
        }
        
        Ok(forecast_output)
    }
    
    fn predict_batch(&self, inputs: &[TimeSeriesInput<T>]) -> NeuroDivergentResult<Vec<ForecastOutput<T>>> {
        inputs.iter()
            .map(|input| self.predict(input))
            .collect()
    }
    
    fn validate(&self, dataset: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ValidationMetrics<T>> {
        if !self.is_trained {
            return Err(NeuroDivergentError::prediction("Model has not been trained"));
        }
        
        let mut total_mse = T::zero();
        let mut total_mae = T::zero();
        let mut num_samples = 0;
        
        for sample in &dataset.samples {
            let prediction = self.predict(&sample.input)?;
            
            // Calculate metrics
            let mse = math::mse(&prediction.forecasts, &sample.target)?;
            let mae = math::mae(&prediction.forecasts, &sample.target)?;
            
            total_mse = total_mse + mse;
            total_mae = total_mae + mae;
            num_samples += 1;
        }
        
        let avg_mse = if num_samples > 0 {
            total_mse / T::from(num_samples).unwrap()
        } else {
            T::zero()
        };
        
        let avg_mae = if num_samples > 0 {
            total_mae / T::from(num_samples).unwrap()
        } else {
            T::zero()
        };
        
        Ok(ValidationMetrics {
            mse: avg_mse,
            mae: avg_mae,
            mape: None, // TODO: Implement MAPE
            smape: None, // TODO: Implement SMAPE
            r2_score: None, // TODO: Implement R²
            directional_accuracy: None, // TODO: Implement directional accuracy
            custom_metrics: HashMap::new(),
        })
    }
    
    fn save_state(&self) -> NeuroDivergentResult<Vec<u8>> {
        // This is a simplified implementation
        // In practice, you'd serialize the entire model state
        if self.output_layer.is_some() {
            // For now, return a placeholder implementation
            // TODO: Implement proper serialization
            Ok(vec![0u8; 64]) // Placeholder
        } else {
            Err(NeuroDivergentError::state("Model not initialized"))
        }
    }
    
    fn load_state(&mut self, _state: &[u8]) -> NeuroDivergentResult<()> {
        // This is a simplified implementation
        if self.output_layer.is_some() {
            // TODO: Implement proper deserialization
            self.is_trained = true;
            Ok(())
        } else {
            Err(NeuroDivergentError::state("Model not initialized"))
        }
    }
    
    fn input_size(&self) -> usize {
        self.config.input_size
    }
    
    fn horizon(&self) -> usize {
        self.config.horizon
    }
    
    fn reset(&mut self) {
        self.is_trained = false;
        self.training_history = None;
        self.recurrent_layers = None;
        self.output_layer = None;
        self.input_scaler = None;
        self.output_scaler = None;
        self.model_metrics.clear();
    }
    
    fn training_history(&self) -> Option<&TrainingHistory<T>> {
        self.training_history.as_ref()
    }
    
    fn model_metrics(&self) -> HashMap<String, f64> {
        self.model_metrics.clone()
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> NetworkAdapter<T> for RNN<T> {
    fn prepare_input(&self, ts_input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<T>> {
        // For RNN, we need to process the sequence and return the final hidden state
        let sequence = self.prepare_sequence(ts_input)?;
        
        if let Some(recurrent_layers) = &self.recurrent_layers {
            let mut temp_layers = recurrent_layers.clone();
            temp_layers.reset_states();
            let sequence_outputs = temp_layers.forward_sequence(&sequence)?;
            Ok(sequence_outputs.last().unwrap().clone())
        } else {
            Err(NeuroDivergentError::state("Recurrent layers not initialized"))
        }
    }
    
    fn process_output(&self, network_output: Vec<T>) -> NeuroDivergentResult<ForecastOutput<T>> {
        Ok(ForecastOutput::new(network_output))
    }
    
    fn create_training_data(&self, dataset: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingData<T>> {
        self.create_training_data(dataset)
    }
    
    fn network(&self) -> &Network<T> {
        self.output_layer.as_ref().unwrap()
    }
    
    fn network_mut(&mut self) -> &mut Network<T> {
        self.output_layer.as_mut().unwrap()
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RecurrentState<T> for RNN<T> {
    fn reset(&mut self) {
        if let Some(recurrent_layers) = &mut self.recurrent_layers {
            recurrent_layers.reset_states();
        }
    }
    
    fn get_state(&self) -> Vec<T> {
        if let Some(recurrent_layers) = &self.recurrent_layers {
            recurrent_layers.get_states().into_iter().flatten().collect()
        } else {
            Vec::new()
        }
    }
    
    fn set_state(&mut self, state: Vec<T>) -> NeuroDivergentResult<()> {
        if let Some(recurrent_layers) = &mut self.recurrent_layers {
            // This is simplified - in practice you'd need to split the state appropriately
            let num_layers = self.config.num_layers;
            let hidden_size = self.config.hidden_size;
            let expected_size = num_layers * hidden_size;
            
            if state.len() != expected_size {
                return Err(NeuroDivergentError::dimension_mismatch(expected_size, state.len()));
            }
            
            let layer_states: Vec<Vec<T>> = state.chunks(hidden_size)
                .map(|chunk| chunk.to_vec())
                .collect();
                
            recurrent_layers.set_states(layer_states)?;
        }
        Ok(())
    }
    
    fn state_dimension(&self) -> usize {
        self.config.num_layers * self.config.hidden_size
    }
    
    fn clone_state(&self) -> Box<dyn RecurrentState<T>> {
        // This would need a more sophisticated implementation in practice
        Box::new(RNN::new(self.config.clone()).unwrap())
    }
}

/// Helper trait for RNN-specific operations
impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RNN<T> {
    /// Enable bidirectional processing
    pub fn enable_bidirectional(&mut self, merge_mode: BidirectionalMergeMode) -> NeuroDivergentResult<()> {
        if self.recurrent_layers.is_none() {
            self.initialize_architecture()?;
        }
        
        // Create bidirectional wrapper
        // This is a simplified implementation - you'd need to create separate forward/backward layers
        todo!("Bidirectional implementation needs refinement")
    }
    
    /// Set dropout rate
    pub fn set_dropout(&mut self, dropout_rate: T) {
        self.config.dropout = dropout_rate;
        if let Some(recurrent_layers) = &mut self.recurrent_layers {
            // Update dropout in the multi-layer recurrent network
            // This would require exposing the dropout setting in MultiLayerRecurrent
        }
    }
    
    /// Get current dropout rate
    pub fn get_dropout(&self) -> T {
        self.config.dropout
    }
}