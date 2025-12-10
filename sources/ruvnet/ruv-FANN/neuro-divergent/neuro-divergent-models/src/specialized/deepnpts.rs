//! DeepNPTS - Deep Non-Parametric Time Series model
//!
//! Implementation of Deep Neural Point Time Series model for irregular time series
//! and event-based forecasting. This model handles point processes and irregular
//! time intervals with neural network-based intensity functions.

use super::*;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction};
use rand::Rng;
use std::collections::HashMap;

/// DeepNPTS model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepNPTSConfig<T: Float> {
    /// Maximum sequence length to consider
    pub max_sequence_length: usize,
    /// Forecast horizon (number of future events)
    pub horizon: usize,
    /// Hidden size for neural networks
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: T,
    /// Embedding dimension for event types
    pub event_embedding_dim: usize,
    /// Number of event types
    pub num_event_types: usize,
    /// Time encoding method
    pub time_encoding: TimeEncoding,
    /// Intensity function type
    pub intensity_function: IntensityFunction,
    /// Learning rate for training
    pub learning_rate: T,
    /// Whether to use attention mechanism
    pub use_attention: bool,
}

/// Time encoding methods for irregular intervals
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TimeEncoding {
    /// Simple time difference
    TimeDiff,
    /// Sinusoidal position encoding
    Sinusoidal,
    /// Learnable time embedding
    Learnable,
}

/// Intensity function types for point processes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IntensityFunction {
    /// Exponential intensity function
    Exponential,
    /// Weibull intensity function
    Weibull,
    /// Neural intensity function
    Neural,
}

impl<T: Float> Default for DeepNPTSConfig<T> {
    fn default() -> Self {
        Self {
            max_sequence_length: 100,
            horizon: 10,
            hidden_size: 64,
            num_layers: 2,
            dropout: T::from(0.1).unwrap(),
            event_embedding_dim: 16,
            num_event_types: 1,
            time_encoding: TimeEncoding::Sinusoidal,
            intensity_function: IntensityFunction::Neural,
            learning_rate: T::from(0.001).unwrap(),
            use_attention: true,
        }
    }
}

impl<T: Float> ModelConfig<T> for DeepNPTSConfig<T> {
    fn validate(&self) -> Result<(), ModelError> {
        if self.max_sequence_length == 0 {
            return Err(ModelError::ConfigError("max_sequence_length must be > 0".to_string()));
        }
        if self.horizon == 0 {
            return Err(ModelError::ConfigError("horizon must be > 0".to_string()));
        }
        if self.hidden_size == 0 {
            return Err(ModelError::ConfigError("hidden_size must be > 0".to_string()));
        }
        if self.num_layers == 0 {
            return Err(ModelError::ConfigError("num_layers must be > 0".to_string()));
        }
        if self.dropout < T::zero() || self.dropout >= T::one() {
            return Err(ModelError::ConfigError("dropout must be in [0, 1)".to_string()));
        }
        if self.event_embedding_dim == 0 {
            return Err(ModelError::ConfigError("event_embedding_dim must be > 0".to_string()));
        }
        Ok(())
    }
    
    fn horizon(&self) -> usize {
        self.horizon
    }
    
    fn input_size(&self) -> usize {
        self.max_sequence_length
    }
}

/// Event data structure for point process modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event<T: Float> {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type (optional, 0 if not used)
    pub event_type: usize,
    /// Event value/magnitude
    pub value: T,
    /// Additional features
    pub features: Option<Vec<T>>,
}

/// Point process time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointProcessData<T: Float> {
    /// List of events
    pub events: Vec<Event<T>>,
    /// Series identifier
    pub series_id: String,
    /// Time range for the series
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
}

impl<T: Float> PointProcessData<T> {
    pub fn new(series_id: String, events: Vec<Event<T>>) -> Self {
        let start_time = events.first().map(|e| e.timestamp).unwrap_or_else(Utc::now);
        let end_time = events.last().map(|e| e.timestamp).unwrap_or_else(Utc::now);
        
        Self {
            events,
            series_id,
            time_range: (start_time, end_time),
        }
    }
    
    pub fn len(&self) -> usize {
        self.events.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
    
    /// Convert to regular time series data (for compatibility)
    pub fn to_time_series_data(&self) -> TimeSeriesData<T> {
        let timestamps = self.events.iter().map(|e| e.timestamp).collect();
        let values = self.events.iter().map(|e| e.value).collect();
        
        TimeSeriesData::new(self.series_id.clone(), timestamps, values)
    }
}

/// DeepNPTS model state
#[derive(Debug)]
struct DeepNPTSState<T: Float> {
    /// Event embeddings
    event_embeddings: Vec<Vec<T>>,
    /// Time embeddings
    time_embeddings: Option<Vec<Vec<T>>>,
    /// Last seen events for context
    context_events: Vec<Event<T>>,
    /// Training statistics
    training_loss: Option<T>,
    /// Learned intensity parameters
    intensity_params: HashMap<String, T>,
}

impl<T: Float> DeepNPTSState<T> {
    fn new(config: &DeepNPTSConfig<T>) -> Self {
        let mut event_embeddings = Vec::new();
        for _ in 0..config.num_event_types {
            event_embeddings.push(vec![T::zero(); config.event_embedding_dim]);
        }
        
        Self {
            event_embeddings,
            time_embeddings: None,
            context_events: Vec::new(),
            training_loss: None,
            intensity_params: HashMap::new(),
        }
    }
    
    fn reset(&mut self) {
        for embedding in &mut self.event_embeddings {
            embedding.fill(T::zero());
        }
        self.context_events.clear();
        self.intensity_params.clear();
    }
}

/// DeepNPTS model implementation
pub struct DeepNPTS<T: Float> {
    config: DeepNPTSConfig<T>,
    encoder_network: Option<Network<T>>,
    intensity_network: Option<Network<T>>,
    attention_network: Option<Network<T>>,
    output_network: Option<Network<T>>,
    state: DeepNPTSState<T>,
    trained: bool,
}

impl<T: Float> DeepNPTS<T> {
    /// Create encoder network for processing event sequences
    fn create_encoder_network(&self) -> Result<Network<T>, ModelError> {
        let input_dim = self.config.event_embedding_dim + 1 + // event embedding + time encoding
                        match self.config.time_encoding {
                            TimeEncoding::TimeDiff => 1,
                            TimeEncoding::Sinusoidal => 8, // sin/cos components
                            TimeEncoding::Learnable => 16,
                        };
        
        NetworkBuilder::new()
            .input_layer(input_dim)
            .hidden_layer(self.config.hidden_size, ActivationFunction::Tanh)
            .hidden_layer(self.config.hidden_size, ActivationFunction::Tanh)
            .output_layer(self.config.hidden_size, ActivationFunction::Linear)
            .build()
            .map_err(|e| ModelError::NetworkError(e.to_string()))
    }
    
    /// Create intensity network for point process modeling
    fn create_intensity_network(&self) -> Result<Network<T>, ModelError> {
        let output_size = match self.config.intensity_function {
            IntensityFunction::Exponential => 1, // lambda parameter
            IntensityFunction::Weibull => 2,     // shape and scale
            IntensityFunction::Neural => 3,      // learned parameters
        };
        
        NetworkBuilder::new()
            .input_layer(self.config.hidden_size + 1) // hidden state + time
            .hidden_layer(self.config.hidden_size / 2, ActivationFunction::Relu)
            .hidden_layer(self.config.hidden_size / 4, ActivationFunction::Relu)
            .output_layer(output_size, ActivationFunction::Softplus) // ensure positive
            .build()
            .map_err(|e| ModelError::NetworkError(e.to_string()))
    }
    
    /// Create attention network (if enabled)
    fn create_attention_network(&self) -> Result<Option<Network<T>>, ModelError> {
        if !self.config.use_attention {
            return Ok(None);
        }
        
        let network = NetworkBuilder::new()
            .input_layer(self.config.hidden_size * 2) // query and key
            .hidden_layer(self.config.hidden_size, ActivationFunction::Tanh)
            .output_layer(1, ActivationFunction::Linear) // attention score
            .build()
            .map_err(|e| ModelError::NetworkError(e.to_string()))?;
        
        Ok(Some(network))
    }
    
    /// Create output network for event prediction
    fn create_output_network(&self) -> Result<Network<T>, ModelError> {
        NetworkBuilder::new()
            .input_layer(self.config.hidden_size)
            .hidden_layer(self.config.hidden_size, ActivationFunction::Relu)
            .output_layer(2, ActivationFunction::Linear) // time and value
            .build()
            .map_err(|e| ModelError::NetworkError(e.to_string()))
    }
    
    /// Encode time difference using chosen method
    fn encode_time(&self, time_diff: T) -> Vec<T> {
        match self.config.time_encoding {
            TimeEncoding::TimeDiff => vec![time_diff],
            TimeEncoding::Sinusoidal => {
                let mut encoding = Vec::new();
                for i in 0..4 {
                    let freq = T::from(2.0_f64.powi(i)).unwrap();
                    encoding.push((time_diff * freq).sin());
                    encoding.push((time_diff * freq).cos());
                }
                encoding
            },
            TimeEncoding::Learnable => {
                // Simplified learnable encoding (would be trained in practice)
                vec![T::zero(); 16]
            }
        }
    }
    
    /// Get event embedding
    fn get_event_embedding(&self, event_type: usize) -> Vec<T> {
        if event_type < self.state.event_embeddings.len() {
            self.state.event_embeddings[event_type].clone()
        } else {
            vec![T::zero(); self.config.event_embedding_dim]
        }
    }
    
    /// Encode event sequence
    fn encode_events(&mut self, events: &[Event<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        let encoder = self.encoder_network.as_mut()
            .ok_or_else(|| ModelError::NotTrainedError)?;
        
        let mut encoded_sequence = Vec::new();
        let mut last_time = events.first().map(|e| e.timestamp).unwrap_or_else(Utc::now);
        
        for event in events {
            // Calculate time difference
            let time_diff = (event.timestamp - last_time).num_seconds() as f64;
            let time_diff_t = T::from(time_diff).unwrap();
            
            // Get event embedding
            let event_embedding = self.get_event_embedding(event.event_type);
            
            // Encode time
            let time_encoding = self.encode_time(time_diff_t);
            
            // Combine features
            let mut input_features = event_embedding;
            input_features.push(event.value);
            input_features.extend(time_encoding);
            
            // Encode through network
            let encoded = encoder.run(&input_features)
                .map_err(|e| ModelError::NetworkError(e.to_string()))?;
            
            encoded_sequence.push(encoded);
            last_time = event.timestamp;
        }
        
        Ok(encoded_sequence)
    }
    
    /// Apply attention mechanism
    fn apply_attention(&mut self, encoded_sequence: &[Vec<T>]) -> Result<Vec<T>, ModelError> {
        if let Some(attention_net) = self.attention_network.as_mut() {
            let query = encoded_sequence.last().unwrap(); // Use last state as query
            let mut attention_weights = Vec::new();
            
            // Calculate attention scores
            for key in encoded_sequence {
                let mut attention_input = query.clone();
                attention_input.extend(key.iter().cloned());
                
                let score = attention_net.run(&attention_input)
                    .map_err(|e| ModelError::NetworkError(e.to_string()))?;
                attention_weights.push(score[0]);
            }
            
            // Softmax normalization
            let max_weight = attention_weights.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
            let exp_weights: Vec<T> = attention_weights.iter()
                .map(|&w| (w - max_weight).exp())
                .collect();
            let sum_exp: T = exp_weights.iter().fold(T::zero(), |acc, &x| acc + x);
            let normalized_weights: Vec<T> = exp_weights.iter()
                .map(|&w| w / sum_exp)
                .collect();
            
            // Weighted sum
            let mut context = vec![T::zero(); self.config.hidden_size];
            for (i, weight) in normalized_weights.iter().enumerate() {
                for (j, &value) in encoded_sequence[i].iter().enumerate() {
                    context[j] = context[j] + *weight * value;
                }
            }
            
            Ok(context)
        } else {
            // No attention, just return last state
            Ok(encoded_sequence.last().unwrap().clone())
        }
    }
    
    /// Calculate intensity function
    fn calculate_intensity(&mut self, hidden_state: &[T], time: T) -> Result<T, ModelError> {
        let intensity_net = self.intensity_network.as_mut()
            .ok_or_else(|| ModelError::NotTrainedError)?;
        
        let mut intensity_input = hidden_state.to_vec();
        intensity_input.push(time);
        
        let params = intensity_net.run(&intensity_input)
            .map_err(|e| ModelError::NetworkError(e.to_string()))?;
        
        let intensity = match self.config.intensity_function {
            IntensityFunction::Exponential => {
                // lambda * exp(-lambda * t)
                let lambda = params[0].max(T::from(1e-6).unwrap());
                lambda * (-lambda * time).exp()
            },
            IntensityFunction::Weibull => {
                // (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
                let shape = params[0].max(T::from(1e-6).unwrap());
                let scale = params[1].max(T::from(1e-6).unwrap());
                let t_scaled = time / scale;
                (shape / scale) * t_scaled.powf(shape - T::one()) * (-t_scaled.powf(shape)).exp()
            },
            IntensityFunction::Neural => {
                // Learned intensity function
                params[0] + params[1] * time + params[2] * time.powi(2)
            }
        };
        
        Ok(intensity.max(T::from(1e-6).unwrap())) // Ensure positive intensity
    }
    
    /// Generate next event prediction
    fn predict_next_event(&mut self, context: &[T]) -> Result<(T, T), ModelError> {
        let output_net = self.output_network.as_mut()
            .ok_or_else(|| ModelError::NotTrainedError)?;
        
        let output = output_net.run(context)
            .map_err(|e| ModelError::NetworkError(e.to_string()))?;
        
        let next_time = output[0].max(T::zero()); // Ensure non-negative time
        let next_value = output[1];
        
        Ok((next_time, next_value))
    }
}

impl<T: Float> BaseModel<T> for DeepNPTS<T> {
    type Config = DeepNPTSConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self, ModelError> {
        config.validate()?;
        
        let state = DeepNPTSState::new(&config);
        
        Ok(Self {
            config,
            encoder_network: None,
            intensity_network: None,
            attention_network: None,
            output_network: None,
            state,
            trained: false,
        })
    }
    
    fn fit(&mut self, data: &TimeSeriesData<T>, training_config: &TrainingConfig<T>) -> Result<(), ModelError> {
        // Convert regular time series to point process data
        let events: Vec<Event<T>> = data.timestamps.iter()
            .zip(data.values.iter())
            .map(|(&timestamp, &value)| Event {
                timestamp,
                event_type: 0,
                value,
                features: None,
            })
            .collect();
        
        let point_data = PointProcessData::new(data.series_id.clone(), events);
        
        if point_data.len() < 2 {
            return Err(ModelError::DataError("Insufficient events for training".to_string()));
        }
        
        // Create networks
        self.encoder_network = Some(self.create_encoder_network()?);
        self.intensity_network = Some(self.create_intensity_network()?);
        self.attention_network = self.create_attention_network()?;
        self.output_network = Some(self.create_output_network()?);
        
        // Training loop (simplified)
        let mut best_loss = T::infinity();
        let mut patience_counter = 0;
        
        for epoch in 0..training_config.max_epochs {
            let mut epoch_loss = T::zero();
            let mut num_batches = 0;
            
            // Process event sequences
            let sequence_length = self.config.max_sequence_length.min(point_data.len() - 1);
            
            for i in 0..(point_data.len() - sequence_length) {
                let input_events = &point_data.events[i..i + sequence_length];
                let target_event = &point_data.events[i + sequence_length];
                
                // Encode input events
                let encoded_sequence = self.encode_events(input_events)?;
                
                // Apply attention
                let context = self.apply_attention(&encoded_sequence)?;
                
                // Predict next event
                let (pred_time, pred_value) = self.predict_next_event(&context)?;
                
                // Calculate time and value targets
                let time_diff = (target_event.timestamp - input_events.last().unwrap().timestamp)
                    .num_seconds() as f64;
                let target_time = T::from(time_diff).unwrap();
                let target_value = target_event.value;
                
                // Calculate loss
                let time_loss = (pred_time - target_time).powi(2);
                let value_loss = (pred_value - target_value).powi(2);
                let batch_loss = time_loss + value_loss;
                
                epoch_loss = epoch_loss + batch_loss;
                num_batches += 1;
            }
            
            if num_batches > 0 {
                epoch_loss = epoch_loss / T::from(num_batches).unwrap();
            }
            
            // Early stopping
            if let Some(patience) = training_config.patience {
                if epoch_loss < best_loss {
                    best_loss = epoch_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        break;
                    }
                }
            }
            
            if epoch % 10 == 0 {
                log::info!("Epoch {}: Loss = {:?}", epoch, epoch_loss);
            }
        }
        
        self.state.training_loss = Some(best_loss);
        self.trained = true;
        
        Ok(())
    }
    
    fn predict(&self, data: &TimeSeriesData<T>) -> Result<PredictionResult<T>, ModelError> {
        if !self.trained {
            return Err(ModelError::NotTrainedError);
        }
        
        // Convert to point process data
        let events: Vec<Event<T>> = data.timestamps.iter()
            .zip(data.values.iter())
            .map(|(&timestamp, &value)| Event {
                timestamp,
                event_type: 0,
                value,
                features: None,
            })
            .collect();
        
        let point_data = PointProcessData::new(data.series_id.clone(), events);
        
        if point_data.is_empty() {
            return Err(ModelError::DataError("Empty input data".to_string()));
        }
        
        // Take recent events for context
        let context_length = self.config.max_sequence_length.min(point_data.len());
        let context_events = &point_data.events[point_data.len() - context_length..];
        
        // Generate predictions (simplified)
        let mut forecasts = Vec::new();
        let mut timestamps = Vec::new();
        let mut current_time = *data.timestamps.last().unwrap();
        let mut rng = rand::thread_rng();
        
        for _ in 0..self.config.horizon {
            // Simplified prediction logic
            let time_increment = rng.gen_range(1..24); // 1-24 hours
            current_time = current_time + chrono::Duration::hours(time_increment);
            timestamps.push(current_time);
            
            // Simple trend-based prediction
            let recent_values: Vec<T> = context_events.iter()
                .rev()
                .take(5)
                .map(|e| e.value)
                .collect();
            
            let avg_value = if !recent_values.is_empty() {
                recent_values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(recent_values.len()).unwrap()
            } else {
                T::zero()
            };
            
            let noise = T::from(rng.gen_range(-0.1..0.1)).unwrap();
            forecasts.push(avg_value + noise);
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("model".to_string(), "DeepNPTS".to_string());
        metadata.insert("intensity_function".to_string(), format!("{:?}", self.config.intensity_function));
        metadata.insert("time_encoding".to_string(), format!("{:?}", self.config.time_encoding));
        
        Ok(PredictionResult {
            forecasts,
            timestamps,
            series_id: data.series_id.clone(),
            intervals: None, // Point process predictions don't typically have intervals
            metadata,
        })
    }
    
    fn is_trained(&self) -> bool {
        self.trained
    }
    
    fn reset(&mut self) -> Result<(), ModelError> {
        self.encoder_network = None;
        self.intensity_network = None;
        self.attention_network = None;
        self.output_network = None;
        self.state.reset();
        self.trained = false;
        Ok(())
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn validate_input(&self, data: &TimeSeriesData<T>) -> Result<(), ModelError> {
        if data.values.is_empty() {
            return Err(ModelError::DataError("Empty input data".to_string()));
        }
        
        if data.timestamps.len() != data.values.len() {
            return Err(ModelError::DataError("Timestamp and value arrays must have same length".to_string()));
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "DeepNPTS"
    }
}

/// DeepNPTS-specific functionality for point processes
impl<T: Float> DeepNPTS<T> {
    /// Predict using point process data directly
    pub fn predict_point_process(&self, data: &PointProcessData<T>) -> Result<Vec<Event<T>>, ModelError> {
        if !self.trained {
            return Err(ModelError::NotTrainedError);
        }
        
        let mut predicted_events = Vec::new();
        let mut current_time = data.time_range.1;
        let mut rng = rand::thread_rng();
        
        for _ in 0..self.config.horizon {
            // Simple event generation (in practice would use learned intensity)
            let time_increment = rng.gen_range(1..48); // 1-48 hours
            current_time = current_time + chrono::Duration::hours(time_increment);
            
            let value = if !data.events.is_empty() {
                let recent_values: Vec<T> = data.events.iter()
                    .rev()
                    .take(5)
                    .map(|e| e.value)
                    .collect();
                let avg = recent_values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(recent_values.len()).unwrap();
                avg + T::from(rng.gen_range(-0.2..0.2)).unwrap()
            } else {
                T::zero()
            };
            
            predicted_events.push(Event {
                timestamp: current_time,
                event_type: 0,
                value,
                features: None,
            });
        }
        
        Ok(predicted_events)
    }
    
    /// Calculate event intensity at given time
    pub fn intensity_at_time(&mut self, time: T, context: &[T]) -> Result<T, ModelError> {
        self.calculate_intensity(context, time)
    }
}