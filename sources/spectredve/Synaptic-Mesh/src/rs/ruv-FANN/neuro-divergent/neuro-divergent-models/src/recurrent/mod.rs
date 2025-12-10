//! Recurrent Neural Network Models
//!
//! This module provides implementations of various recurrent neural network architectures
//! for time series forecasting, including basic RNN, LSTM, and GRU models.

pub mod layers;
pub mod rnn;

// Re-export main model types
pub use rnn::RNN;

// Re-export layer components
pub use layers::{
    RecurrentLayer, 
    BasicRecurrentCell, 
    MultiLayerRecurrent, 
    BidirectionalRecurrent,
    BidirectionalMergeMode
};

// LSTM and GRU models will be implemented here
// For now, we'll create type aliases to the base RNN until they're fully implemented
pub type LSTM<T> = RNN<T>;
pub type GRU<T> = RNN<T>;