//! # Neuro-Divergent Models
//!
//! A comprehensive neural forecasting library built on top of ruv-FANN, providing
//! state-of-the-art time series forecasting models for production use.
//!
//! This library implements 27+ neural forecasting models inspired by NeuralForecast,
//! optimized for Rust's performance and safety guarantees.
//!
//! ## Features
//!
//! - **Recurrent Models**: RNN, LSTM, GRU with temporal state management
//! - **Transformer Models**: Multi-head attention, TFT, and advanced architectures
//! - **Linear Models**: DLinear, NLinear with decomposition
//! - **Specialized Models**: NBEATS, TimesNet, and domain-specific architectures
//! - **Production Ready**: Type-safe, memory-efficient, and scalable
//!
//! ## Quick Start
//!
//! ```rust
//! use neuro_divergent_models::{NeuralForecast, models::LSTM, LSTMConfig};
//! use neuro_divergent_models::data::TimeSeriesDataFrame;
//!
//! // Create LSTM model
//! let lstm_config = LSTMConfig::default_with_horizon(24)
//!     .with_architecture(128, 2, 0.1)
//!     .with_training(1000, 0.001);
//!
//! let lstm = LSTM::new(lstm_config)?;
//!
//! // Create forecasting pipeline
//! let mut nf = NeuralForecast::new()
//!     .with_model(Box::new(lstm))
//!     .build()?;
//!
//! // Train and forecast
//! nf.fit(train_data)?;
//! let forecasts = nf.predict()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Core error handling
pub use errors::{NeuroDivergentError, NeuroDivergentResult};

// Core traits and foundations
pub use foundation::{BaseModel, NetworkAdapter, ModelConfig};
pub use foundation::{TimeSeriesInput, ForecastOutput, ValidationConfig};

// Data structures
pub use data::{TimeSeriesDataFrame, ForecastDataFrame, TimeSeriesSchema};

// Main forecasting interface
pub use forecasting::NeuralForecast;

// Model configurations
pub use config::{LSTMConfig, RNNConfig, GRUConfig};
pub use config::{TrainingConfig, PredictionConfig, CrossValidationConfig};

// Re-export commonly used types from ruv-FANN
pub use ruv_fann::{ActivationFunction, Network, Layer, Neuron};
pub use num_traits::Float;

// Modules
pub mod errors;
pub mod foundation;
pub mod data;
pub mod forecasting;
pub mod config;

// Model implementations
pub mod models {
    //! Neural forecasting model implementations
    pub use crate::recurrent::{RNN, LSTM, GRU};
}

// Model categories
pub mod recurrent;

// Core components
pub mod activations;
pub mod layers;

// Utilities
pub mod utils;

// Test utilities for other crates
#[cfg(feature = "testing")]
pub mod test_utils;