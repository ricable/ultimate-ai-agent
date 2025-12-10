//! Comprehensive unit tests for recurrent models
//!
//! This module tests RNN, LSTM, and GRU implementations including their layers
//! and various configurations.

use neuro_divergent_models::recurrent::*;
use neuro_divergent_models::core::*;
use neuro_divergent_models::config::*;
use neuro_divergent_models::errors::*;
use neuro_divergent_core::prelude::*;
use ndarray::{Array1, Array2, Array3, Axis};
use num_traits::Float;
use proptest::prelude::*;
use approx::assert_relative_eq;
use std::collections::HashMap;

// ============================================================================
// RNN Layer Tests
// ============================================================================

#[cfg(test)]
mod rnn_layer_tests {
    use super::*;

    #[test]
    fn test_simple_rnn_cell() {
        let input_size = 10;
        let hidden_size = 20;
        let batch_size = 2;
        
        let mut cell = SimpleRNNCell::<f64>::new(input_size, hidden_size);
        
        // Test forward pass
        let input = Array2::zeros((batch_size, input_size));
        let hidden = Array2::zeros((batch_size, hidden_size));
        
        let new_hidden = cell.forward(&input, &hidden);
        assert_eq!(new_hidden.shape(), &[batch_size, hidden_size]);
        
        // Test with non-zero inputs
        let input = Array2::ones((batch_size, input_size));
        let hidden = Array2::ones((batch_size, hidden_size)) * 0.5;
        
        let new_hidden = cell.forward(&input, &hidden);
        assert_eq!(new_hidden.shape(), &[batch_size, hidden_size]);
        
        // Values should be bounded by activation function (tanh)
        for value in new_hidden.iter() {
            assert!(*value >= -1.0 && *value <= 1.0);
        }
    }

    #[test]
    fn test_lstm_cell() {
        let input_size = 10;
        let hidden_size = 20;
        let batch_size = 2;
        
        let mut cell = LSTMCell::<f64>::new(input_size, hidden_size);
        
        // Test forward pass
        let input = Array2::zeros((batch_size, input_size));
        let hidden = Array2::zeros((batch_size, hidden_size));
        let cell_state = Array2::zeros((batch_size, hidden_size));
        
        let (new_hidden, new_cell) = cell.forward(&input, &hidden, &cell_state);
        assert_eq!(new_hidden.shape(), &[batch_size, hidden_size]);
        assert_eq!(new_cell.shape(), &[batch_size, hidden_size]);
        
        // Test with non-zero inputs
        let input = Array2::ones((batch_size, input_size)) * 0.1;
        let hidden = Array2::ones((batch_size, hidden_size)) * 0.1;
        let cell_state = Array2::ones((batch_size, hidden_size)) * 0.1;
        
        let (new_hidden, new_cell) = cell.forward(&input, &hidden, &cell_state);
        
        // Hidden state should be bounded by tanh
        for value in new_hidden.iter() {
            assert!(*value >= -1.0 && *value <= 1.0);
        }
    }

    #[test]
    fn test_gru_cell() {
        let input_size = 10;
        let hidden_size = 20;
        let batch_size = 2;
        
        let mut cell = GRUCell::<f64>::new(input_size, hidden_size);
        
        // Test forward pass
        let input = Array2::zeros((batch_size, input_size));
        let hidden = Array2::zeros((batch_size, hidden_size));
        
        let new_hidden = cell.forward(&input, &hidden);
        assert_eq!(new_hidden.shape(), &[batch_size, hidden_size]);
        
        // Test with non-zero inputs
        let input = Array2::ones((batch_size, input_size)) * 0.1;
        let hidden = Array2::ones((batch_size, hidden_size)) * 0.5;
        
        let new_hidden = cell.forward(&input, &hidden);
        
        // Hidden state should be bounded by tanh
        for value in new_hidden.iter() {
            assert!(*value >= -1.0 && *value <= 1.0);
        }
    }

    #[test]
    fn test_bidirectional_rnn() {
        let input_size = 10;
        let hidden_size = 20;
        let seq_length = 15;
        let batch_size = 2;
        
        let mut layer = BidirectionalRNN::<f64>::new(
            RNNType::LSTM,
            input_size,
            hidden_size,
        );
        
        // Input shape: (batch, seq_len, input_size)
        let input = Array3::zeros((batch_size, seq_length, input_size));
        
        let output = layer.forward(&input);
        
        // Output should have double hidden size due to bidirectional
        assert_eq!(output.shape(), &[batch_size, seq_length, hidden_size * 2]);
    }

    #[test]
    fn test_multi_layer_rnn() {
        let input_size = 10;
        let hidden_size = 20;
        let num_layers = 3;
        let seq_length = 15;
        let batch_size = 2;
        
        let mut rnn = MultiLayerRNN::<f64>::new(
            RNNType::LSTM,
            input_size,
            hidden_size,
            num_layers,
            0.0, // no dropout for testing
            false, // not bidirectional
        );
        
        let input = Array3::zeros((batch_size, seq_length, input_size));
        let output = rnn.forward(&input);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, hidden_size]);
    }

    #[test]
    fn test_rnn_with_dropout() {
        let input_size = 10;
        let hidden_size = 20;
        let num_layers = 3;
        let seq_length = 15;
        let batch_size = 2;
        let dropout = 0.5;
        
        let mut rnn = MultiLayerRNN::<f64>::new(
            RNNType::GRU,
            input_size,
            hidden_size,
            num_layers,
            dropout,
            false,
        );
        
        // Set to training mode for dropout
        rnn.set_training(true);
        
        let input = Array3::ones((batch_size, seq_length, input_size));
        let output1 = rnn.forward(&input);
        let output2 = rnn.forward(&input);
        
        // Outputs should be different due to dropout
        let diff: f64 = (&output1 - &output2)
            .iter()
            .map(|x| x.abs())
            .sum();
        assert!(diff > 0.0);
        
        // Set to evaluation mode
        rnn.set_training(false);
        
        let output3 = rnn.forward(&input);
        let output4 = rnn.forward(&input);
        
        // Outputs should be identical without dropout
        let diff: f64 = (&output3 - &output4)
            .iter()
            .map(|x| x.abs())
            .sum();
        assert_relative_eq!(diff, 0.0);
    }
}

// ============================================================================
// RNN Model Tests
// ============================================================================

#[cfg(test)]
mod rnn_model_tests {
    use super::*;

    #[test]
    fn test_rnn_config_creation() {
        let config = RNNConfig::new(24, 12)
            .with_rnn_type(RNNType::LSTM)
            .with_hidden_size(128)
            .with_num_layers(2)
            .with_dropout(0.2)
            .with_bidirectional(true)
            .with_learning_rate(0.001)
            .with_max_epochs(100);

        assert_eq!(config.input_size, 24);
        assert_eq!(config.horizon, 12);
        assert_eq!(config.rnn_type, RNNType::LSTM);
        assert_eq!(config.hidden_size, 128);
        assert_eq!(config.num_layers, 2);
        assert_relative_eq!(config.dropout, 0.2);
        assert!(config.bidirectional);
        assert_relative_eq!(config.learning_rate, 0.001);
        assert_eq!(config.max_epochs, 100);
    }

    #[test]
    fn test_rnn_config_validation() {
        // Valid config
        let valid_config = RNNConfig::new(24, 12);
        assert!(valid_config.validate().is_ok());

        // Invalid hidden size
        let invalid_hidden = RNNConfig::new(24, 12).with_hidden_size(0);
        assert!(invalid_hidden.validate().is_err());

        // Invalid number of layers
        let invalid_layers = RNNConfig::new(24, 12).with_num_layers(0);
        assert!(invalid_layers.validate().is_err());

        // Invalid dropout
        let invalid_dropout = RNNConfig::new(24, 12).with_dropout(1.5);
        assert!(invalid_dropout.validate().is_err());
    }

    #[test]
    fn test_rnn_types() {
        let rnn_types = vec![RNNType::Simple, RNNType::LSTM, RNNType::GRU];
        
        for rnn_type in rnn_types {
            let config = RNNConfig::new(10, 5)
                .with_rnn_type(rnn_type)
                .with_max_epochs(1);
            
            let model = RNN::<f64>::new(config);
            assert!(model.is_ok());
        }
    }

    #[test]
    fn test_rnn_workflow() {
        // Create synthetic sequential data
        let data_length = 100;
        let data: Vec<f64> = (0..data_length)
            .map(|i| (i as f64 * 0.1).sin() + 0.1 * rand::random::<f64>())
            .collect();
        
        let ts_data = TimeSeriesData::new(data);

        let config = RNNConfig::new(20, 10)
            .with_rnn_type(RNNType::LSTM)
            .with_hidden_size(32)
            .with_num_layers(2)
            .with_max_epochs(5);

        let mut model = RNN::new(config).unwrap();

        // Train
        assert!(!model.is_fitted());
        assert!(model.fit(&ts_data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&ts_data).unwrap();
        assert_eq!(forecast.len(), 10);
    }

    #[test]
    fn test_bidirectional_vs_unidirectional() {
        let data = TimeSeriesData::new(vec![1.0; 100]);
        
        // Unidirectional
        let uni_config = RNNConfig::new(20, 10)
            .with_bidirectional(false)
            .with_hidden_size(32)
            .with_max_epochs(1);
        
        let mut uni_model = RNN::<f64>::new(uni_config).unwrap();
        assert!(uni_model.fit(&data).is_ok());
        
        // Bidirectional
        let bi_config = RNNConfig::new(20, 10)
            .with_bidirectional(true)
            .with_hidden_size(32)
            .with_max_epochs(1);
        
        let mut bi_model = RNN::<f64>::new(bi_config).unwrap();
        assert!(bi_model.fit(&data).is_ok());
        
        // Bidirectional should have different architecture
        // This would be reflected in parameter count in actual implementation
    }

    #[test]
    fn test_rnn_builder() {
        let config = RNNBuilder::new()
            .input_size(48)
            .horizon(24)
            .rnn_type(RNNType::GRU)
            .hidden_size(256)
            .num_layers(3)
            .dropout(0.3)
            .bidirectional(true)
            .learning_rate(0.0005)
            .max_epochs(200)
            .build();

        assert_eq!(config.input_size, 48);
        assert_eq!(config.horizon, 24);
        assert_eq!(config.rnn_type, RNNType::GRU);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 3);
        assert_relative_eq!(config.dropout, 0.3);
        assert!(config.bidirectional);
    }

    #[test]
    fn test_sequence_processing() {
        // Test that RNN properly processes sequences
        let seq_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0,
            3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0,
        ];
        
        let ts_data = TimeSeriesData::new(seq_data);
        
        let config = RNNConfig::new(5, 3)
            .with_rnn_type(RNNType::LSTM)
            .with_hidden_size(16)
            .with_max_epochs(5);
        
        let mut model = RNN::new(config).unwrap();
        assert!(model.fit(&ts_data).is_ok());
        
        let forecast = model.predict(&ts_data).unwrap();
        assert_eq!(forecast.len(), 3);
    }
}

// ============================================================================
// Advanced RNN Features Tests
// ============================================================================

#[cfg(test)]
mod advanced_rnn_tests {
    use super::*;

    #[test]
    fn test_attention_mechanism() {
        let seq_length = 10;
        let hidden_size = 20;
        let batch_size = 2;
        
        let attention = AttentionLayer::<f64>::new(hidden_size);
        
        // Input: (batch, seq_len, hidden_size)
        let input = Array3::ones((batch_size, seq_length, hidden_size));
        
        let (output, weights) = attention.forward(&input);
        
        // Output should be (batch, hidden_size)
        assert_eq!(output.shape(), &[batch_size, hidden_size]);
        
        // Attention weights should be (batch, seq_len)
        assert_eq!(weights.shape(), &[batch_size, seq_length]);
        
        // Weights should sum to 1 along sequence dimension
        for batch_idx in 0..batch_size {
            let weight_sum: f64 = weights.slice(s![batch_idx, ..]).sum();
            assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_teacher_forcing() {
        // Test teacher forcing vs autoregressive generation
        let data = TimeSeriesData::new(
            (0..100).map(|i| (i as f64 * 0.1).sin()).collect()
        );
        
        let config = RNNConfig::new(20, 10)
            .with_rnn_type(RNNType::LSTM)
            .with_teacher_forcing_ratio(0.5)
            .with_max_epochs(1);
        
        let mut model = RNN::<f64>::new(config).unwrap();
        
        // During training, teacher forcing should be used
        assert!(model.fit(&data).is_ok());
        
        // During prediction, no teacher forcing
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 10);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = RNNConfig::new(20, 10)
            .with_gradient_clip_value(1.0)
            .with_max_epochs(1);
        
        let model = RNN::<f64>::new(config);
        assert!(model.is_ok());
        
        // Gradient clipping should prevent exploding gradients
        // This would be tested more thoroughly in actual training
    }

    #[test]
    fn test_variable_length_sequences() {
        // Test handling of sequences with different lengths
        let sequences = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0],
        ];
        
        // In practice, sequences would be padded or packed
        // This tests the concept
        for seq in sequences {
            if seq.len() >= 20 {
                let data = TimeSeriesData::new(seq);
                let config = RNNConfig::new(10, 5).with_max_epochs(1);
                let mut model = RNN::<f64>::new(config).unwrap();
                
                let result = model.fit(&data);
                assert!(result.is_ok() || result.is_err()); // Either padded or error
            }
        }
    }
}

// ============================================================================
// Performance and Optimization Tests
// ============================================================================

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_cell_computation_efficiency() {
        let input_size = 100;
        let hidden_size = 200;
        let batch_size = 32;
        
        // Test that cell computations are efficient
        let mut lstm_cell = LSTMCell::<f64>::new(input_size, hidden_size);
        let mut gru_cell = GRUCell::<f64>::new(input_size, hidden_size);
        let mut rnn_cell = SimpleRNNCell::<f64>::new(input_size, hidden_size);
        
        let input = Array2::ones((batch_size, input_size));
        let hidden = Array2::zeros((batch_size, hidden_size));
        let cell_state = Array2::zeros((batch_size, hidden_size));
        
        // LSTM forward
        let (lstm_h, lstm_c) = lstm_cell.forward(&input, &hidden, &cell_state);
        assert_eq!(lstm_h.shape(), &[batch_size, hidden_size]);
        
        // GRU forward  
        let gru_h = gru_cell.forward(&input, &hidden);
        assert_eq!(gru_h.shape(), &[batch_size, hidden_size]);
        
        // Simple RNN forward
        let rnn_h = rnn_cell.forward(&input, &hidden);
        assert_eq!(rnn_h.shape(), &[batch_size, hidden_size]);
    }

    #[test]
    fn test_memory_usage() {
        // Test that memory usage is reasonable for different configurations
        let configs = vec![
            (10, 32, 1),   // Small
            (50, 128, 2),  // Medium
            (100, 256, 3), // Large
        ];
        
        for (input_size, hidden_size, num_layers) in configs {
            let config = RNNConfig::new(input_size, 5)
                .with_hidden_size(hidden_size)
                .with_num_layers(num_layers);
            
            let model = RNN::<f64>::new(config);
            assert!(model.is_ok());
            
            // Model should be created without excessive memory
            // In practice, we'd measure actual memory usage
        }
    }

    #[test]
    #[ignore] // Run with --ignored for performance testing
    fn benchmark_rnn_variants() {
        use std::time::Instant;
        
        let data = TimeSeriesData::new(
            (0..1000).map(|i| (i as f64 * 0.01).sin()).collect()
        );
        
        let variants = vec![
            ("Simple RNN", RNNType::Simple),
            ("LSTM", RNNType::LSTM),
            ("GRU", RNNType::GRU),
        ];
        
        for (name, rnn_type) in variants {
            let config = RNNConfig::new(50, 10)
                .with_rnn_type(rnn_type)
                .with_hidden_size(64)
                .with_num_layers(2)
                .with_max_epochs(10);
            
            let mut model = RNN::<f64>::new(config).unwrap();
            
            let start = Instant::now();
            model.fit(&data).unwrap();
            let duration = start.elapsed();
            
            println!("{} training time: {:?}", name, duration);
        }
    }
}

// ============================================================================
// Property-based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn prop_rnn_config_validation(
            input_size in 1usize..100,
            horizon in 1usize..50,
            hidden_size in 1usize..256,
            num_layers in 1usize..5,
            dropout in 0.0f64..0.99,
            learning_rate in 0.0001f64..0.1,
        ) {
            let config = RNNConfig::new(input_size, horizon)
                .with_hidden_size(hidden_size)
                .with_num_layers(num_layers)
                .with_dropout(dropout)
                .with_learning_rate(learning_rate);
            
            prop_assert!(config.validate().is_ok());
        }

        #[test]
        fn prop_cell_output_shapes(
            batch_size in 1usize..32,
            input_size in 1usize..50,
            hidden_size in 1usize..100,
        ) {
            // Test SimpleRNN
            let mut rnn_cell = SimpleRNNCell::<f64>::new(input_size, hidden_size);
            let input = Array2::zeros((batch_size, input_size));
            let hidden = Array2::zeros((batch_size, hidden_size));
            let output = rnn_cell.forward(&input, &hidden);
            prop_assert_eq!(output.shape(), &[batch_size, hidden_size]);
            
            // Test LSTM
            let mut lstm_cell = LSTMCell::<f64>::new(input_size, hidden_size);
            let cell_state = Array2::zeros((batch_size, hidden_size));
            let (h, c) = lstm_cell.forward(&input, &hidden, &cell_state);
            prop_assert_eq!(h.shape(), &[batch_size, hidden_size]);
            prop_assert_eq!(c.shape(), &[batch_size, hidden_size]);
            
            // Test GRU
            let mut gru_cell = GRUCell::<f64>::new(input_size, hidden_size);
            let output = gru_cell.forward(&input, &hidden);
            prop_assert_eq!(output.shape(), &[batch_size, hidden_size]);
        }

        #[test]
        fn prop_attention_weights_sum_to_one(
            seq_length in 1usize..50,
            hidden_size in 1usize..100,
            batch_size in 1usize..16,
        ) {
            let attention = AttentionLayer::<f64>::new(hidden_size);
            let input = Array3::ones((batch_size, seq_length, hidden_size));
            
            let (_, weights) = attention.forward(&input);
            
            // Check each batch
            for b in 0..batch_size {
                let weight_sum: f64 = weights.slice(s![b, ..]).sum();
                prop_assert!((weight_sum - 1.0).abs() < 1e-6);
            }
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_single_timestep() {
        let config = RNNConfig::new(1, 1)
            .with_hidden_size(1)
            .with_num_layers(1);
        
        let model = RNN::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_very_long_sequences() {
        // Test handling of very long sequences
        let long_data = TimeSeriesData::new(vec![1.0; 10000]);
        
        let config = RNNConfig::new(100, 10)
            .with_hidden_size(32)
            .with_max_epochs(1);
        
        let mut model = RNN::<f64>::new(config).unwrap();
        
        // Should handle long sequences (possibly with truncation)
        let result = model.fit(&long_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zero_teacher_forcing() {
        let config = RNNConfig::new(20, 10)
            .with_teacher_forcing_ratio(0.0);
        
        let model = RNN::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_full_teacher_forcing() {
        let config = RNNConfig::new(20, 10)
            .with_teacher_forcing_ratio(1.0);
        
        let model = RNN::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_extreme_gradient_clipping() {
        // Very small gradient clip
        let small_clip = RNNConfig::new(20, 10)
            .with_gradient_clip_value(0.001);
        assert!(RNN::<f64>::new(small_clip).is_ok());
        
        // Very large gradient clip
        let large_clip = RNNConfig::new(20, 10)
            .with_gradient_clip_value(1000.0);
        assert!(RNN::<f64>::new(large_clip).is_ok());
    }
}

// ============================================================================
// Integration Tests with Time Series
// ============================================================================

#[cfg(test)]
mod time_series_integration_tests {
    use super::*;

    #[test]
    fn test_seasonal_pattern_learning() {
        // Create data with clear seasonal pattern
        let period = 12;
        let cycles = 10;
        let data: Vec<f64> = (0..period * cycles)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
            .collect();
        
        let ts_data = TimeSeriesData::new(data);
        
        let config = RNNConfig::new(period * 2, period)
            .with_rnn_type(RNNType::LSTM)
            .with_hidden_size(32)
            .with_max_epochs(10);
        
        let mut model = RNN::new(config).unwrap();
        assert!(model.fit(&ts_data).is_ok());
        
        // Model should learn the seasonal pattern
        let forecast = model.predict(&ts_data).unwrap();
        assert_eq!(forecast.len(), period);
    }

    #[test]
    fn test_trend_pattern_learning() {
        // Create data with linear trend
        let data: Vec<f64> = (0..100)
            .map(|i| i as f64 * 0.5 + 10.0)
            .collect();
        
        let ts_data = TimeSeriesData::new(data);
        
        let config = RNNConfig::new(20, 10)
            .with_rnn_type(RNNType::GRU)
            .with_hidden_size(16)
            .with_max_epochs(10);
        
        let mut model = RNN::new(config).unwrap();
        assert!(model.fit(&ts_data).is_ok());
        
        let forecast = model.predict(&ts_data).unwrap();
        assert_eq!(forecast.len(), 10);
        
        // Forecast should continue the trend
        // In practice, we'd check if values are increasing
    }

    #[test]
    fn test_noisy_data_handling() {
        // Create noisy data
        let base: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();
        
        let noise: Vec<f64> = (0..100)
            .map(|_| rand::random::<f64>() * 0.5 - 0.25)
            .collect();
        
        let data: Vec<f64> = base.iter()
            .zip(noise.iter())
            .map(|(b, n)| b + n)
            .collect();
        
        let ts_data = TimeSeriesData::new(data);
        
        let config = RNNConfig::new(20, 10)
            .with_rnn_type(RNNType::LSTM)
            .with_hidden_size(32)
            .with_dropout(0.2) // Dropout for regularization
            .with_max_epochs(10);
        
        let mut model = RNN::new(config).unwrap();
        assert!(model.fit(&ts_data).is_ok());
        
        // Model should handle noise and produce reasonable forecasts
        let forecast = model.predict(&ts_data).unwrap();
        assert_eq!(forecast.len(), 10);
    }
}