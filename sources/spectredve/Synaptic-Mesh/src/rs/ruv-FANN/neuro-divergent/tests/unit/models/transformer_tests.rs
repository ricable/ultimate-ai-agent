//! Comprehensive unit tests for transformer models
//!
//! This module tests Transformer, Informer, Autoformer, and TFT implementations
//! including attention mechanisms, decomposition layers, and various configurations.

use neuro_divergent_models::transformer::*;
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
// Attention Mechanism Tests
// ============================================================================

#[cfg(test)]
mod attention_tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product_attention() {
        let batch_size = 2;
        let seq_length = 10;
        let d_model = 64;
        
        let attention = ScaledDotProductAttention::<f64>::new(d_model);
        
        // Create Q, K, V matrices
        let q = Array3::ones((batch_size, seq_length, d_model));
        let k = Array3::ones((batch_size, seq_length, d_model));
        let v = Array3::ones((batch_size, seq_length, d_model));
        
        let (output, weights) = attention.forward(&q, &k, &v, None);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
        assert_eq!(weights.shape(), &[batch_size, seq_length, seq_length]);
        
        // Check attention weights sum to 1
        for b in 0..batch_size {
            for i in 0..seq_length {
                let weight_sum: f64 = weights.slice(s![b, i, ..]).sum();
                assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_multi_head_attention() {
        let batch_size = 2;
        let seq_length = 8;
        let d_model = 128;
        let n_heads = 8;
        
        let mha = MultiHeadAttention::<f64>::new(d_model, n_heads);
        
        let input = Array3::ones((batch_size, seq_length, d_model));
        let output = mha.forward(&input, &input, &input, None);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
    }

    #[test]
    fn test_attention_masking() {
        let batch_size = 2;
        let seq_length = 6;
        let d_model = 32;
        
        let attention = ScaledDotProductAttention::<f64>::new(d_model);
        
        // Create causal mask (lower triangular)
        let mut mask = Array2::zeros((seq_length, seq_length));
        for i in 0..seq_length {
            for j in i+1..seq_length {
                mask[[i, j]] = f64::NEG_INFINITY;
            }
        }
        
        let q = Array3::ones((batch_size, seq_length, d_model));
        let k = Array3::ones((batch_size, seq_length, d_model));
        let v = Array3::ones((batch_size, seq_length, d_model));
        
        let (output, weights) = attention.forward(&q, &k, &v, Some(&mask));
        
        // Check that masked positions have zero weight
        for b in 0..batch_size {
            for i in 0..seq_length {
                for j in i+1..seq_length {
                    assert!(weights[[b, i, j]] < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_probsparse_attention() {
        // Informer's ProbSparse attention
        let batch_size = 2;
        let seq_length = 100;
        let d_model = 64;
        let sampling_factor = 5;
        
        let attention = ProbSparseAttention::<f64>::new(d_model, sampling_factor);
        
        let q = Array3::ones((batch_size, seq_length, d_model));
        let k = Array3::ones((batch_size, seq_length, d_model));
        let v = Array3::ones((batch_size, seq_length, d_model));
        
        let output = attention.forward(&q, &k, &v);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
        
        // ProbSparse should be more efficient than full attention
        // Only top-k queries are computed
    }

    #[test]
    fn test_auto_correlation() {
        // Autoformer's auto-correlation mechanism
        let batch_size = 2;
        let seq_length = 48;
        let d_model = 64;
        let top_k = 5;
        
        let auto_corr = AutoCorrelation::<f64>::new(d_model, top_k);
        
        let input = Array3::from_shape_fn((batch_size, seq_length, d_model), |(b, t, d)| {
            (t as f64 * 0.1).sin() + d as f64 * 0.01
        });
        
        let output = auto_corr.forward(&input);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
        
        // Auto-correlation should capture periodic patterns
    }
}

// ============================================================================
// Transformer Model Tests
// ============================================================================

#[cfg(test)]
mod transformer_model_tests {
    use super::*;

    #[test]
    fn test_transformer_config() {
        let config = TransformerConfig::new(48, 24)
            .with_d_model(512)
            .with_n_heads(8)
            .with_n_encoder_layers(6)
            .with_n_decoder_layers(6)
            .with_d_ff(2048)
            .with_dropout(0.1)
            .with_activation(ActivationType::ReLU)
            .with_positional_encoding(PositionalEncodingType::Sinusoidal);

        assert_eq!(config.input_size, 48);
        assert_eq!(config.horizon, 24);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.n_encoder_layers, 6);
        assert_eq!(config.n_decoder_layers, 6);
    }

    #[test]
    fn test_transformer_validation() {
        // Valid config
        let valid = TransformerConfig::new(48, 24);
        assert!(valid.validate().is_ok());

        // Invalid d_model (not divisible by n_heads)
        let invalid_d_model = TransformerConfig::new(48, 24)
            .with_d_model(100)
            .with_n_heads(8); // 100 not divisible by 8
        assert!(invalid_d_model.validate().is_err());

        // Zero layers
        let zero_layers = TransformerConfig::new(48, 24)
            .with_n_encoder_layers(0);
        assert!(zero_layers.validate().is_err());
    }

    #[test]
    fn test_positional_encoding() {
        let seq_length = 100;
        let d_model = 128;
        
        // Sinusoidal encoding
        let sin_encoding = create_sinusoidal_encoding::<f64>(seq_length, d_model);
        assert_eq!(sin_encoding.shape(), &[seq_length, d_model]);
        
        // Learnable encoding
        let learn_encoding = LearnablePositionalEncoding::<f64>::new(seq_length, d_model);
        let output = learn_encoding.forward(seq_length);
        assert_eq!(output.shape(), &[seq_length, d_model]);
    }

    #[test]
    fn test_transformer_encoder_decoder() {
        let batch_size = 2;
        let src_length = 48;
        let tgt_length = 24;
        let d_model = 128;
        
        let config = TransformerConfig::new(src_length, tgt_length)
            .with_d_model(d_model)
            .with_n_heads(8)
            .with_n_encoder_layers(2)
            .with_n_decoder_layers(2);
        
        let transformer = Transformer::<f64>::new(config).unwrap();
        
        let src = Array3::ones((batch_size, src_length, d_model));
        let tgt = Array3::ones((batch_size, tgt_length, d_model));
        
        let output = transformer.forward(&src, &tgt, None, None);
        assert_eq!(output.shape(), &[batch_size, tgt_length, d_model]);
    }

    #[test]
    fn test_transformer_workflow() {
        let data = TimeSeriesData::new(
            (0..200).map(|i| (i as f64 * 0.05).sin()).collect()
        );

        let config = TransformerConfig::new(48, 24)
            .with_d_model(64)
            .with_n_heads(4)
            .with_n_encoder_layers(2)
            .with_n_decoder_layers(2)
            .with_max_epochs(5);

        let mut model = Transformer::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 24);
    }
}

// ============================================================================
// Informer Tests
// ============================================================================

#[cfg(test)]
mod informer_tests {
    use super::*;

    #[test]
    fn test_informer_config() {
        let config = InformerConfig::new(96, 48)
            .with_d_model(512)
            .with_n_heads(8)
            .with_e_layers(3)
            .with_d_layers(2)
            .with_factor(5)
            .with_distil(true)
            .with_activation(ActivationType::GELU);

        assert_eq!(config.input_size, 96);
        assert_eq!(config.horizon, 48);
        assert_eq!(config.factor, 5);
        assert!(config.distil);
    }

    #[test]
    fn test_probsparse_self_attention() {
        let batch_size = 2;
        let L_Q = 96;  // Query length
        let L_K = 96;  // Key length
        let d_model = 64;
        let n_heads = 8;
        let factor = 5;
        
        let attention = ProbAttention::<f64>::new(d_model, n_heads, factor);
        
        let queries = Array3::ones((batch_size, L_Q, d_model));
        let keys = Array3::ones((batch_size, L_K, d_model));
        let values = Array3::ones((batch_size, L_K, d_model));
        
        let output = attention.forward(&queries, &keys, &values, None);
        assert_eq!(output.shape(), &[batch_size, L_Q, d_model]);
    }

    #[test]
    fn test_distilling_layer() {
        let batch_size = 2;
        let seq_length = 96;
        let d_model = 64;
        
        let distil = DistillingLayer::<f64>::new();
        
        let input = Array3::ones((batch_size, seq_length, d_model));
        let output = distil.forward(&input);
        
        // Distilling reduces sequence length by half
        assert_eq!(output.shape(), &[batch_size, seq_length / 2, d_model]);
    }

    #[test]
    fn test_informer_encoder_stack() {
        let config = InformerConfig::new(96, 48)
            .with_d_model(128)
            .with_n_heads(8)
            .with_e_layers(3)
            .with_distil(true);
        
        let encoder = InformerEncoder::<f64>::new(&config);
        
        let batch_size = 2;
        let input = Array3::ones((batch_size, 96, 128));
        
        let output = encoder.forward(&input, None);
        
        // With distilling, output length is reduced
        assert!(output.shape()[1] < 96);
    }

    #[test]
    fn test_informer_workflow() {
        let data = TimeSeriesData::new(
            (0..500).map(|i| {
                let trend = i as f64 * 0.01;
                let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin();
                trend + seasonal
            }).collect()
        );

        let config = InformerConfig::new(96, 48)
            .with_d_model(64)
            .with_n_heads(4)
            .with_e_layers(2)
            .with_d_layers(1)
            .with_factor(3)
            .with_max_epochs(5);

        let mut model = Informer::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 48);
    }
}

// ============================================================================
// Autoformer Tests
// ============================================================================

#[cfg(test)]
mod autoformer_tests {
    use super::*;

    #[test]
    fn test_autoformer_config() {
        let config = AutoformerConfig::new(96, 48)
            .with_d_model(512)
            .with_n_heads(8)
            .with_e_layers(2)
            .with_d_layers(1)
            .with_moving_avg_window(25)
            .with_factor(3)
            .with_activation(ActivationType::GELU);

        assert_eq!(config.input_size, 96);
        assert_eq!(config.horizon, 48);
        assert_eq!(config.moving_avg_window, 25);
    }

    #[test]
    fn test_series_decomposition() {
        let batch_size = 2;
        let seq_length = 100;
        let d_model = 64;
        let kernel_size = 25;
        
        let decomp = SeriesDecomposition::<f64>::new(kernel_size);
        
        // Create synthetic data with trend and seasonal
        let input = Array3::from_shape_fn((batch_size, seq_length, d_model), |(b, t, d)| {
            let trend = t as f64 * 0.1;
            let seasonal = (t as f64 * 2.0 * std::f64::consts::PI / 24.0).sin();
            trend + seasonal
        });
        
        let (seasonal, trend) = decomp.forward(&input);
        
        assert_eq!(seasonal.shape(), &[batch_size, seq_length, d_model]);
        assert_eq!(trend.shape(), &[batch_size, seq_length, d_model]);
        
        // Seasonal + trend should equal input
        let reconstructed = &seasonal + &trend;
        for i in 0..batch_size {
            for j in kernel_size/2..seq_length-kernel_size/2 {
                for k in 0..d_model {
                    assert_relative_eq!(
                        reconstructed[[i, j, k]],
                        input[[i, j, k]],
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    #[test]
    fn test_auto_correlation_layer() {
        let batch_size = 2;
        let seq_length = 48;
        let d_model = 64;
        let n_heads = 8;
        let c = 3;
        
        let layer = AutoCorrelationLayer::<f64>::new(d_model, n_heads, c);
        
        let input = Array3::from_shape_fn((batch_size, seq_length, d_model), |(b, t, d)| {
            (t as f64 * 0.2).sin()
        });
        
        let output = layer.forward(&input, None);
        assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
    }

    #[test]
    fn test_autoformer_encoder_decoder() {
        let config = AutoformerConfig::new(96, 48)
            .with_d_model(64)
            .with_n_heads(4)
            .with_e_layers(2)
            .with_d_layers(1)
            .with_moving_avg_window(25);
        
        let autoformer = Autoformer::<f64>::new(config).unwrap();
        
        let batch_size = 2;
        let src = Array3::ones((batch_size, 96, 64));
        let tgt = Array3::ones((batch_size, 48, 64));
        
        let output = autoformer.forward(&src, &tgt);
        assert_eq!(output.shape(), &[batch_size, 48, 64]);
    }

    #[test]
    fn test_autoformer_workflow() {
        // Create data with clear decomposition
        let data = TimeSeriesData::new(
            (0..400).map(|i| {
                let trend = i as f64 * 0.05;
                let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin() * 3.0;
                let noise = rand::random::<f64>() * 0.1;
                trend + seasonal + noise
            }).collect()
        );

        let config = AutoformerConfig::new(96, 48)
            .with_d_model(64)
            .with_n_heads(4)
            .with_e_layers(2)
            .with_d_layers(1)
            .with_moving_avg_window(25)
            .with_max_epochs(5);

        let mut model = Autoformer::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 48);
    }
}

// ============================================================================
// Temporal Fusion Transformer (TFT) Tests
// ============================================================================

#[cfg(test)]
mod tft_tests {
    use super::*;

    #[test]
    fn test_tft_config() {
        let config = TFTConfig::new(168, 24)
            .with_hidden_size(160)
            .with_lstm_layers(2)
            .with_attention_heads(4)
            .with_dropout(0.1)
            .with_static_covariates(5)
            .with_time_varying_known(3)
            .with_time_varying_unknown(2)
            .with_quantiles(vec![0.1, 0.5, 0.9]);

        assert_eq!(config.input_size, 168);
        assert_eq!(config.horizon, 24);
        assert_eq!(config.hidden_size, 160);
        assert_eq!(config.quantiles, vec![0.1, 0.5, 0.9]);
    }

    #[test]
    fn test_variable_selection_network() {
        let batch_size = 2;
        let seq_length = 48;
        let n_features = 10;
        let hidden_size = 64;
        
        let vsn = VariableSelectionNetwork::<f64>::new(n_features, hidden_size);
        
        let input = Array3::ones((batch_size, seq_length, n_features));
        let (selected, weights) = vsn.forward(&input);
        
        assert_eq!(selected.shape(), &[batch_size, seq_length, hidden_size]);
        assert_eq!(weights.shape(), &[batch_size, n_features]);
        
        // Weights should sum to approximately n_features (before softmax)
        for b in 0..batch_size {
            let weight_sum: f64 = weights.slice(s![b, ..]).sum();
            assert!(weight_sum > 0.0);
        }
    }

    #[test]
    fn test_gated_residual_network() {
        let input_size = 64;
        let hidden_size = 32;
        let output_size = 64;
        let batch_size = 2;
        let seq_length = 10;
        
        let grn = GatedResidualNetwork::<f64>::new(
            input_size,
            hidden_size,
            output_size,
            Some(input_size),
        );
        
        let input = Array3::ones((batch_size, seq_length, input_size));
        let output = grn.forward(&input, None);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, output_size]);
    }

    #[test]
    fn test_temporal_self_attention() {
        let batch_size = 2;
        let seq_length = 24;
        let d_model = 64;
        let n_heads = 4;
        
        let attention = InterpretableMultiHeadAttention::<f64>::new(d_model, n_heads);
        
        let input = Array3::ones((batch_size, seq_length, d_model));
        let (output, weights) = attention.forward(&input);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
        assert_eq!(weights.shape(), &[batch_size, n_heads, seq_length, seq_length]);
    }

    #[test]
    fn test_quantile_output() {
        let batch_size = 2;
        let hidden_size = 64;
        let horizon = 24;
        let quantiles = vec![0.1, 0.5, 0.9];
        
        let output_layer = QuantileOutput::<f64>::new(
            hidden_size,
            horizon,
            quantiles.clone(),
        );
        
        let input = Array2::ones((batch_size, hidden_size));
        let output = output_layer.forward(&input);
        
        // Output shape: (batch, horizon, num_quantiles)
        assert_eq!(output.shape(), &[batch_size, horizon, quantiles.len()]);
    }

    #[test]
    fn test_tft_workflow() {
        // Create data with covariates
        let n_samples = 500;
        let target: Vec<f64> = (0..n_samples).map(|i| {
            (i as f64 * 0.05).sin() + i as f64 * 0.01
        }).collect();
        
        // Time-varying known (e.g., day of week, hour)
        let known_features = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                (i % 7) as f64  // Day of week
            } else {
                (i % 24) as f64  // Hour of day
            }
        });
        
        let data = TimeSeriesData::new(target)
            .with_exogenous(known_features);

        let config = TFTConfig::new(168, 24)
            .with_hidden_size(64)
            .with_lstm_layers(1)
            .with_attention_heads(4)
            .with_time_varying_known(2)
            .with_quantiles(vec![0.1, 0.5, 0.9])
            .with_max_epochs(5);

        let mut model = TFT::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.values.len(), 24); // Median predictions
        
        // Get quantile predictions
        let quantile_forecast = model.predict_quantiles(&data).unwrap();
        assert_eq!(quantile_forecast.shape(), &[24, 3]); // 24 timesteps, 3 quantiles
    }
}

// ============================================================================
// Common Transformer Tests
// ============================================================================

#[cfg(test)]
mod common_transformer_tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        let activations = vec![
            ActivationType::ReLU,
            ActivationType::GELU,
            ActivationType::Swish,
            ActivationType::GLU,
        ];
        
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        
        for activation in activations {
            let output = apply_activation(&input, activation);
            assert_eq!(output.len(), input.len());
            
            match activation {
                ActivationType::ReLU => {
                    assert_eq!(output[0], 0.0);
                    assert_eq!(output[2], 0.0);
                    assert_eq!(output[4], 2.0);
                }
                _ => {
                    // Other activations have different behaviors
                }
            }
        }
    }

    #[test]
    fn test_feedforward_network() {
        let batch_size = 2;
        let seq_length = 10;
        let d_model = 64;
        let d_ff = 256;
        
        let ffn = FeedForwardNetwork::<f64>::new(d_model, d_ff, ActivationType::ReLU, 0.1);
        
        let input = Array3::ones((batch_size, seq_length, d_model));
        let output = ffn.forward(&input);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
    }

    #[test]
    fn test_layer_normalization() {
        let batch_size = 2;
        let seq_length = 10;
        let d_model = 64;
        
        let layer_norm = LayerNorm::<f64>::new(d_model);
        
        let input = Array3::from_shape_fn((batch_size, seq_length, d_model), |(b, t, d)| {
            (b + t + d) as f64
        });
        
        let output = layer_norm.forward(&input);
        assert_eq!(output.shape(), input.shape());
        
        // Check normalization
        for b in 0..batch_size {
            for t in 0..seq_length {
                let slice = output.slice(s![b, t, ..]);
                let mean: f64 = slice.mean().unwrap();
                let var: f64 = slice.var(0.0);
                
                assert_relative_eq!(mean, 0.0, epsilon = 1e-5);
                assert_relative_eq!(var, 1.0, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_embedding_layers() {
        let batch_size = 2;
        let seq_length = 10;
        let vocab_size = 1000;
        let d_model = 64;
        
        let embedding = Embedding::<f64>::new(vocab_size, d_model);
        
        // Input token indices
        let input = Array2::from_shape_fn((batch_size, seq_length), |(b, t)| {
            ((b * seq_length + t) % vocab_size) as i64
        });
        
        let output = embedding.forward(&input);
        assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
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
        fn prop_attention_weights_valid(
            batch_size in 1usize..8,
            seq_length in 1usize..50,
            d_model in 8usize..128,
            n_heads in vec![1usize, 2, 4, 8],
        ) {
            let d_model = (d_model / n_heads) * n_heads; // Ensure divisibility
            
            let mha = MultiHeadAttention::<f64>::new(d_model, n_heads);
            let input = Array3::zeros((batch_size, seq_length, d_model));
            
            let output = mha.forward(&input, &input, &input, None);
            prop_assert_eq!(output.shape(), &[batch_size, seq_length, d_model]);
        }

        #[test]
        fn prop_series_decomposition_reconstruction(
            batch_size in 1usize..4,
            seq_length in 25usize..100,
            d_model in 1usize..32,
            kernel_size in 3usize..25,
        ) {
            let kernel_size = if kernel_size % 2 == 0 { kernel_size + 1 } else { kernel_size };
            
            let decomp = SeriesDecomposition::<f64>::new(kernel_size);
            let input = Array3::zeros((batch_size, seq_length, d_model));
            
            let (seasonal, trend) = decomp.forward(&input);
            
            prop_assert_eq!(seasonal.shape(), input.shape());
            prop_assert_eq!(trend.shape(), input.shape());
        }

        #[test]
        fn prop_positional_encoding_bounded(
            seq_length in 1usize..200,
            d_model in 16usize..256,
        ) {
            let encoding = create_sinusoidal_encoding::<f64>(seq_length, d_model);
            
            prop_assert_eq!(encoding.shape(), &[seq_length, d_model]);
            
            // All values should be bounded between -1 and 1
            for val in encoding.iter() {
                prop_assert!(*val >= -1.0 && *val <= 1.0);
            }
        }
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_attention_complexity() {
        // Standard attention: O(L^2 * d)
        // ProbSparse attention: O(L * log(L) * d)
        // Auto-correlation: O(L * log(L) * d)
        
        let configs = vec![
            ("Short", 48, 64),
            ("Medium", 192, 128),
            ("Long", 720, 256),
        ];
        
        for (name, seq_length, d_model) in configs {
            println!("Testing {} sequence attention", name);
            
            // Standard attention should work for all
            let standard = ScaledDotProductAttention::<f64>::new(d_model);
            
            // ProbSparse for longer sequences
            if seq_length > 100 {
                let probsparse = ProbSparseAttention::<f64>::new(d_model, 5);
            }
        }
    }

    #[test]
    #[ignore] // Run with --ignored for benchmarking
    fn benchmark_transformer_variants() {
        use std::time::Instant;
        
        let data = TimeSeriesData::new(
            (0..1000).map(|i| (i as f64 * 0.01).sin()).collect()
        );
        
        // Benchmark standard Transformer
        let transformer_config = TransformerConfig::new(96, 24)
            .with_d_model(64)
            .with_n_heads(4)
            .with_max_epochs(5);
        
        let mut transformer = Transformer::<f64>::new(transformer_config).unwrap();
        let start = Instant::now();
        transformer.fit(&data).unwrap();
        println!("Transformer training time: {:?}", start.elapsed());
        
        // Benchmark Informer
        let informer_config = InformerConfig::new(96, 24)
            .with_d_model(64)
            .with_n_heads(4)
            .with_max_epochs(5);
        
        let mut informer = Informer::<f64>::new(informer_config).unwrap();
        let start = Instant::now();
        informer.fit(&data).unwrap();
        println!("Informer training time: {:?}", start.elapsed());
        
        // Benchmark Autoformer
        let autoformer_config = AutoformerConfig::new(96, 24)
            .with_d_model(64)
            .with_n_heads(4)
            .with_max_epochs(5);
        
        let mut autoformer = Autoformer::<f64>::new(autoformer_config).unwrap();
        let start = Instant::now();
        autoformer.fit(&data).unwrap();
        println!("Autoformer training time: {:?}", start.elapsed());
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_single_head_attention() {
        let d_model = 64;
        let mha = MultiHeadAttention::<f64>::new(d_model, 1);
        
        let input = Array3::ones((2, 10, d_model));
        let output = mha.forward(&input, &input, &input, None);
        
        assert_eq!(output.shape(), &[2, 10, d_model]);
    }

    #[test]
    fn test_minimal_transformer() {
        let config = TransformerConfig::new(10, 5)
            .with_d_model(16)
            .with_n_heads(1)
            .with_n_encoder_layers(1)
            .with_n_decoder_layers(1)
            .with_d_ff(32);
        
        let model = Transformer::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_very_deep_transformer() {
        let config = TransformerConfig::new(48, 24)
            .with_d_model(64)
            .with_n_heads(4)
            .with_n_encoder_layers(12)
            .with_n_decoder_layers(12);
        
        let model = Transformer::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_extreme_attention_heads() {
        // Many heads with small dimension per head
        let config = TransformerConfig::new(48, 24)
            .with_d_model(512)
            .with_n_heads(64); // 8 dims per head
        
        let model = Transformer::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_single_timestep_input() {
        let config = TransformerConfig::new(1, 1)
            .with_d_model(16)
            .with_n_heads(1);
        
        let model = Transformer::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_autoformer_no_decomposition() {
        // Moving average window larger than sequence
        let config = AutoformerConfig::new(10, 5)
            .with_moving_avg_window(25);
        
        let model = Autoformer::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_tft_no_covariates() {
        // TFT without any covariates
        let config = TFTConfig::new(48, 24)
            .with_static_covariates(0)
            .with_time_varying_known(0)
            .with_time_varying_unknown(0);
        
        let model = TFT::<f64>::new(config);
        assert!(model.is_ok());
    }
}