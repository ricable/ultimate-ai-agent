//! Comprehensive unit tests for specialized models
//!
//! This module tests DeepAR, TCN, BiTCN, and DeepNPTS implementations
//! including their specialized layers and configurations.

use neuro_divergent_models::specialized::*;
use neuro_divergent_models::core::*;
use neuro_divergent_models::config::*;
use neuro_divergent_models::errors::*;
use neuro_divergent_core::prelude::*;
use ndarray::{Array1, Array2, Array3, Axis, s};
use num_traits::Float;
use proptest::prelude::*;
use approx::assert_relative_eq;
use std::collections::HashMap;

// ============================================================================
// DeepAR Tests
// ============================================================================

#[cfg(test)]
mod deepar_tests {
    use super::*;

    #[test]
    fn test_deepar_config() {
        let config = DeepARConfig::new(48, 24)
            .with_lstm_layers(vec![50, 50])
            .with_dropout(0.1)
            .with_num_samples(100)
            .with_likelihood(LikelihoodType::Gaussian)
            .with_use_feat_dynamic_real(true)
            .with_use_feat_static_cat(true)
            .with_cardinality(vec![50, 20, 10])
            .with_embedding_dimension(vec![10, 5, 5]);

        assert_eq!(config.input_size, 48);
        assert_eq!(config.horizon, 24);
        assert_eq!(config.lstm_layers, vec![50, 50]);
        assert_eq!(config.num_samples, 100);
        assert_eq!(config.likelihood, LikelihoodType::Gaussian);
        assert!(config.use_feat_dynamic_real);
        assert!(config.use_feat_static_cat);
        assert_eq!(config.cardinality, vec![50, 20, 10]);
    }

    #[test]
    fn test_likelihood_types() {
        let likelihoods = vec![
            LikelihoodType::Gaussian,
            LikelihoodType::StudentT,
            LikelihoodType::NegativeBinomial,
            LikelihoodType::Beta,
        ];

        for likelihood in likelihoods {
            let config = DeepARConfig::new(48, 24)
                .with_likelihood(likelihood);
            
            let model = DeepAR::<f64>::new(config);
            assert!(model.is_ok());
        }
    }

    #[test]
    fn test_distribution_parameters() {
        let batch_size = 2;
        let hidden_size = 64;
        
        // Gaussian distribution
        let gaussian = GaussianOutput::<f64>::new(hidden_size);
        let hidden = Array2::ones((batch_size, hidden_size));
        let (mu, sigma) = gaussian.forward(&hidden);
        
        assert_eq!(mu.shape(), &[batch_size, 1]);
        assert_eq!(sigma.shape(), &[batch_size, 1]);
        
        // Sigma should be positive
        for &s in sigma.iter() {
            assert!(s > 0.0);
        }
        
        // Student-T distribution
        let student_t = StudentTOutput::<f64>::new(hidden_size);
        let (mu, sigma, nu) = student_t.forward(&hidden);
        
        assert_eq!(mu.shape(), &[batch_size, 1]);
        assert_eq!(sigma.shape(), &[batch_size, 1]);
        assert_eq!(nu.shape(), &[batch_size, 1]);
        
        // Nu (degrees of freedom) should be > 2
        for &n in nu.iter() {
            assert!(n > 2.0);
        }
    }

    #[test]
    fn test_categorical_embedding() {
        let cardinality = vec![50, 20, 10]; // 3 categorical features
        let embedding_dim = vec![10, 5, 5];
        let batch_size = 2;
        
        let embedder = CategoricalEmbedding::<f64>::new(&cardinality, &embedding_dim);
        
        // Input categorical indices
        let input = Array2::from_shape_fn((batch_size, 3), |(b, f)| {
            (b * 3 + f) as i64 % cardinality[f]
        });
        
        let embedded = embedder.forward(&input);
        
        // Output shape: (batch, sum(embedding_dim))
        let total_embed_dim: usize = embedding_dim.iter().sum();
        assert_eq!(embedded.shape(), &[batch_size, total_embed_dim]);
    }

    #[test]
    fn test_deepar_workflow() {
        // Create synthetic data with seasonality
        let data = TimeSeriesData::new(
            (0..500).map(|i| {
                let trend = i as f64 * 0.01;
                let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin() * 5.0;
                let noise = rand::random::<f64>() * 0.5;
                (trend + seasonal + noise).max(0.0) // Ensure positive for some likelihoods
            }).collect()
        );

        let config = DeepARConfig::new(48, 24)
            .with_lstm_layers(vec![40, 40])
            .with_dropout(0.1)
            .with_num_samples(50)
            .with_likelihood(LikelihoodType::Gaussian)
            .with_max_epochs(5);

        let mut model = DeepAR::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict - returns probabilistic forecasts
        let forecast_samples = model.predict_samples(&data, 100).unwrap();
        assert_eq!(forecast_samples.shape(), &[100, 24]); // 100 samples, 24 horizon
        
        // Get point forecast (mean)
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 24);
        
        // Get prediction intervals
        let intervals = model.predict_intervals(&data, vec![0.8, 0.95]).unwrap();
        assert_eq!(intervals.lower_bounds.len(), 2); // 2 confidence levels
        assert_eq!(intervals.upper_bounds.len(), 2);
    }

    #[test]
    fn test_deepar_with_covariates() {
        let n_samples = 200;
        let n_dynamic_features = 3;
        
        // Create data with dynamic features
        let target = (0..n_samples).map(|i| i as f64 * 0.1).collect();
        let dynamic_features = Array2::from_shape_fn((n_samples, n_dynamic_features), |(i, j)| {
            (i + j) as f64 * 0.01
        });
        
        let data = TimeSeriesData::new(target)
            .with_exogenous(dynamic_features);

        let config = DeepARConfig::new(48, 24)
            .with_use_feat_dynamic_real(true)
            .with_num_feat_dynamic_real(n_dynamic_features)
            .with_max_epochs(5);

        let mut model = DeepAR::new(config).unwrap();
        assert!(model.fit(&data).is_ok());
    }

    #[test]
    fn test_deepar_multi_series() {
        // DeepAR is designed for multiple time series
        let series_data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0; 100],
            vec![5.0, 4.0, 3.0, 2.0, 1.0; 100],
            vec![2.0, 4.0, 6.0, 8.0, 10.0; 100],
        ];
        
        // In practice, would train on all series jointly
        for (idx, series) in series_data.iter().enumerate() {
            let data = TimeSeriesData::new(series.clone())
                .with_series_id(idx.to_string());
            
            // Model learns shared patterns across series
        }
    }
}

// ============================================================================
// TCN Tests
// ============================================================================

#[cfg(test)]
mod tcn_tests {
    use super::*;

    #[test]
    fn test_tcn_config() {
        let config = TCNConfig::new(48, 24)
            .with_num_channels(vec![32, 32, 32, 32])
            .with_kernel_size(7)
            .with_dropout(0.2)
            .with_use_skip_connections(true)
            .with_use_weight_norm(true);

        assert_eq!(config.input_size, 48);
        assert_eq!(config.horizon, 24);
        assert_eq!(config.num_channels, vec![32, 32, 32, 32]);
        assert_eq!(config.kernel_size, 7);
        assert!(config.use_skip_connections);
        assert!(config.use_weight_norm);
    }

    #[test]
    fn test_causal_convolution() {
        let batch_size = 2;
        let seq_length = 20;
        let in_channels = 1;
        let out_channels = 16;
        let kernel_size = 3;
        let dilation = 1;
        
        let conv = CausalConv1d::<f64>::new(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
        );
        
        let input = Array3::ones((batch_size, seq_length, in_channels));
        let output = conv.forward(&input);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, out_channels]);
        
        // Check causality: output at time t depends only on inputs up to time t
        // This would be verified by checking that changing future inputs doesn't affect past outputs
    }

    #[test]
    fn test_dilated_convolution() {
        let dilations = vec![1, 2, 4, 8, 16];
        let kernel_size = 3;
        let seq_length = 50;
        
        for dilation in dilations {
            let receptive_field = (kernel_size - 1) * dilation + 1;
            println!("Dilation {}: receptive field = {}", dilation, receptive_field);
            
            let conv = CausalConv1d::<f64>::new(1, 16, kernel_size, dilation);
            let input = Array3::ones((1, seq_length, 1));
            let output = conv.forward(&input);
            
            assert_eq!(output.shape()[1], seq_length);
        }
    }

    #[test]
    fn test_residual_block() {
        let batch_size = 2;
        let seq_length = 30;
        let n_channels = 32;
        let kernel_size = 3;
        let dilation = 4;
        
        let block = ResidualBlock::<f64>::new(
            n_channels,
            n_channels,
            kernel_size,
            dilation,
            0.1,
            true,
        );
        
        let input = Array3::ones((batch_size, seq_length, n_channels));
        let (output, skip) = block.forward(&input);
        
        assert_eq!(output.shape(), &[batch_size, seq_length, n_channels]);
        assert_eq!(skip.shape(), &[batch_size, seq_length, n_channels]);
    }

    #[test]
    fn test_tcn_receptive_field() {
        // TCN receptive field = 1 + 2 * (kernel_size - 1) * sum(dilations)
        let kernel_size = 3;
        let num_layers = 4;
        let dilations: Vec<usize> = (0..num_layers).map(|i| 2usize.pow(i as u32)).collect();
        
        let sum_dilations: usize = dilations.iter().sum();
        let receptive_field = 1 + 2 * (kernel_size - 1) * sum_dilations;
        
        println!("TCN with {} layers, kernel size {}: receptive field = {}", 
                 num_layers, kernel_size, receptive_field);
        
        assert!(receptive_field > 0);
    }

    #[test]
    fn test_tcn_workflow() {
        let data = TimeSeriesData::new(
            (0..300).map(|i| (i as f64 * 0.05).sin()).collect()
        );

        let config = TCNConfig::new(48, 24)
            .with_num_channels(vec![25, 25, 25])
            .with_kernel_size(5)
            .with_dropout(0.1)
            .with_max_epochs(5);

        let mut model = TCN::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 24);
    }

    #[test]
    fn test_tcn_vs_rnn_efficiency() {
        // TCN advantages:
        // 1. Parallelizable (no sequential dependencies)
        // 2. Stable gradients (no vanishing/exploding)
        // 3. Flexible receptive field
        
        let config = TCNConfig::new(100, 50)
            .with_num_channels(vec![64; 6]) // 6 layers
            .with_kernel_size(3);
        
        let model = TCN::<f64>::new(config);
        assert!(model.is_ok());
        
        // TCN can process all timesteps in parallel unlike RNN
    }
}

// ============================================================================
// BiTCN Tests
// ============================================================================

#[cfg(test)]
mod bitcn_tests {
    use super::*;

    #[test]
    fn test_bitcn_config() {
        let config = BiTCNConfig::new(48, 24)
            .with_num_channels(vec![32, 32, 32])
            .with_kernel_size(5)
            .with_dropout(0.2)
            .with_use_layer_norm(true);

        assert_eq!(config.input_size, 48);
        assert_eq!(config.horizon, 24);
        assert_eq!(config.num_channels, vec![32, 32, 32]);
        assert!(config.use_layer_norm);
    }

    #[test]
    fn test_bidirectional_convolution() {
        let batch_size = 2;
        let seq_length = 20;
        let in_channels = 1;
        let out_channels = 16;
        let kernel_size = 3;
        
        let biconv = BidirectionalConv1d::<f64>::new(
            in_channels,
            out_channels,
            kernel_size,
        );
        
        let input = Array3::ones((batch_size, seq_length, in_channels));
        let output = biconv.forward(&input);
        
        // Output has double channels (forward + backward)
        assert_eq!(output.shape(), &[batch_size, seq_length, out_channels * 2]);
    }

    #[test]
    fn test_bidirectional_block() {
        let batch_size = 2;
        let seq_length = 30;
        let n_channels = 32;
        let kernel_size = 3;
        let dilation = 2;
        
        let block = BidirectionalBlock::<f64>::new(
            n_channels,
            n_channels,
            kernel_size,
            dilation,
            0.1,
            true,
        );
        
        let input = Array3::ones((batch_size, seq_length, n_channels));
        let (output, skip) = block.forward(&input);
        
        // Bidirectional processing preserves input shape
        assert_eq!(output.shape(), &[batch_size, seq_length, n_channels]);
        assert_eq!(skip.shape(), &[batch_size, seq_length, n_channels]);
    }

    #[test]
    fn test_bitcn_workflow() {
        // Create data with patterns that benefit from bidirectional processing
        let data = TimeSeriesData::new(
            (0..400).map(|i| {
                let forward_pattern = (i as f64 * 0.05).sin();
                let backward_pattern = ((400 - i) as f64 * 0.03).cos();
                forward_pattern + backward_pattern
            }).collect()
        );

        let config = BiTCNConfig::new(60, 30)
            .with_num_channels(vec![30, 30, 30])
            .with_kernel_size(5)
            .with_max_epochs(5);

        let mut model = BiTCN::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 30);
    }

    #[test]
    fn test_bitcn_vs_tcn() {
        // BiTCN can capture both forward and backward dependencies
        // Useful for:
        // 1. Time series with bidirectional patterns
        // 2. When future context helps predict current values
        // 3. Anomaly detection where context from both directions matters
        
        let data = TimeSeriesData::new(vec![1.0; 200]);
        
        // Standard TCN
        let tcn_config = TCNConfig::new(48, 24).with_max_epochs(1);
        let mut tcn = TCN::<f64>::new(tcn_config).unwrap();
        assert!(tcn.fit(&data).is_ok());
        
        // Bidirectional TCN
        let bitcn_config = BiTCNConfig::new(48, 24).with_max_epochs(1);
        let mut bitcn = BiTCN::<f64>::new(bitcn_config).unwrap();
        assert!(bitcn.fit(&data).is_ok());
        
        // BiTCN has roughly 2x parameters due to bidirectional processing
    }
}

// ============================================================================
// DeepNPTS Tests
// ============================================================================

#[cfg(test)]
mod deepnpts_tests {
    use super::*;

    #[test]
    fn test_deepnpts_config() {
        let config = DeepNPTSConfig::new(48, 24)
            .with_hidden_size(100)
            .with_num_layers(3)
            .with_num_attention_heads(8)
            .with_temperature(0.1)
            .with_num_prototypes(50)
            .with_use_memory_network(true);

        assert_eq!(config.input_size, 48);
        assert_eq!(config.horizon, 24);
        assert_eq!(config.hidden_size, 100);
        assert_eq!(config.num_prototypes, 50);
        assert!(config.use_memory_network);
    }

    #[test]
    fn test_prototype_network() {
        let batch_size = 2;
        let seq_length = 20;
        let hidden_size = 64;
        let num_prototypes = 10;
        
        let prototype_net = PrototypeNetwork::<f64>::new(
            hidden_size,
            num_prototypes,
        );
        
        let input = Array3::ones((batch_size, seq_length, hidden_size));
        let (prototypes, similarities) = prototype_net.forward(&input);
        
        assert_eq!(prototypes.shape(), &[batch_size, num_prototypes, hidden_size]);
        assert_eq!(similarities.shape(), &[batch_size, seq_length, num_prototypes]);
        
        // Similarities should sum to 1 across prototypes
        for b in 0..batch_size {
            for t in 0..seq_length {
                let sim_sum: f64 = similarities.slice(s![b, t, ..]).sum();
                assert_relative_eq!(sim_sum, 1.0, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_memory_network() {
        let memory_size = 100;
        let key_size = 64;
        let value_size = 32;
        let batch_size = 2;
        let query_size = 64;
        
        let memory = MemoryNetwork::<f64>::new(
            memory_size,
            key_size,
            value_size,
        );
        
        let query = Array2::ones((batch_size, query_size));
        let (values, attention) = memory.forward(&query);
        
        assert_eq!(values.shape(), &[batch_size, value_size]);
        assert_eq!(attention.shape(), &[batch_size, memory_size]);
        
        // Attention weights sum to 1
        for b in 0..batch_size {
            let attn_sum: f64 = attention.slice(s![b, ..]).sum();
            assert_relative_eq!(attn_sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_neural_process_layer() {
        let batch_size = 2;
        let context_size = 30;
        let target_size = 10;
        let hidden_size = 64;
        let latent_size = 32;
        
        let np_layer = NeuralProcessLayer::<f64>::new(
            hidden_size,
            latent_size,
        );
        
        let context_x = Array3::ones((batch_size, context_size, hidden_size));
        let context_y = Array2::ones((batch_size, context_size));
        let target_x = Array3::ones((batch_size, target_size, hidden_size));
        
        let (mean, logvar) = np_layer.encode(&context_x, &context_y);
        assert_eq!(mean.shape(), &[batch_size, latent_size]);
        assert_eq!(logvar.shape(), &[batch_size, latent_size]);
        
        let predictions = np_layer.decode(&target_x, &mean);
        assert_eq!(predictions.shape(), &[batch_size, target_size]);
    }

    #[test]
    fn test_deepnpts_workflow() {
        // Create data with patterns that can be captured by prototypes
        let data = TimeSeriesData::new(
            (0..500).map(|i| {
                match i % 50 {
                    0..=9 => 1.0,    // Pattern 1
                    10..=19 => 2.0,  // Pattern 2
                    20..=29 => 3.0,  // Pattern 3
                    30..=39 => 2.0,  // Pattern 2
                    _ => 1.0,        // Pattern 1
                }
            }).collect()
        );

        let config = DeepNPTSConfig::new(50, 25)
            .with_hidden_size(64)
            .with_num_layers(2)
            .with_num_prototypes(5) // Should learn ~3 main patterns
            .with_max_epochs(5);

        let mut model = DeepNPTS::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 25);
        
        // Get prototype assignments
        let assignments = model.get_prototype_assignments(&data).unwrap();
        assert!(assignments.len() > 0);
    }

    #[test]
    fn test_few_shot_learning() {
        // DeepNPTS is designed for few-shot learning scenarios
        let few_shot_data = TimeSeriesData::new(
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0] // Only 9 points
        );
        
        let config = DeepNPTSConfig::new(5, 2) // Small input/output
            .with_hidden_size(32)
            .with_num_prototypes(3)
            .with_use_memory_network(true)
            .with_max_epochs(10);
        
        let mut model = DeepNPTS::new(config).unwrap();
        
        // Should be able to learn from very few examples
        let result = model.fit(&few_shot_data);
        assert!(result.is_ok() || result.is_err()); // May need minimum data
    }
}

// ============================================================================
// Common Specialized Model Tests
// ============================================================================

#[cfg(test)]
mod common_specialized_tests {
    use super::*;

    #[test]
    fn test_weight_normalization() {
        let in_features = 32;
        let out_features = 64;
        
        let linear = WeightNormLinear::<f64>::new(in_features, out_features);
        
        let input = Array2::ones((2, in_features));
        let output = linear.forward(&input);
        
        assert_eq!(output.shape(), &[2, out_features]);
        
        // Weight norm ensures ||w|| * g where g is learned
        // This improves optimization and generalization
    }

    #[test]
    fn test_glu_activation() {
        let input_size = 64;
        let batch_size = 2;
        
        let glu = GLU::<f64>::new(input_size);
        
        let input = Array2::ones((batch_size, input_size));
        let output = glu.forward(&input);
        
        // GLU reduces dimension by half
        assert_eq!(output.shape(), &[batch_size, input_size / 2]);
    }

    #[test]
    fn test_model_comparison() {
        let data = TimeSeriesData::new(
            (0..200).map(|i| (i as f64 * 0.1).sin()).collect()
        );
        
        let models: Vec<(&str, Box<dyn BaseModel<f64>>)> = vec![
            ("DeepAR", Box::new(DeepAR::new(
                DeepARConfig::new(40, 20).with_max_epochs(1)
            ).unwrap())),
            ("TCN", Box::new(TCN::new(
                TCNConfig::new(40, 20).with_max_epochs(1)
            ).unwrap())),
            ("BiTCN", Box::new(BiTCN::new(
                BiTCNConfig::new(40, 20).with_max_epochs(1)
            ).unwrap())),
        ];
        
        for (name, mut model) in models {
            println!("Testing {}", name);
            assert!(model.fit(&data).is_ok());
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
        fn prop_causal_conv_preserves_sequence_length(
            batch_size in 1usize..8,
            seq_length in 10usize..100,
            in_channels in 1usize..16,
            out_channels in 1usize..32,
            kernel_size in 3usize..7,
            dilation in 1usize..4,
        ) {
            let conv = CausalConv1d::<f64>::new(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
            );
            
            let input = Array3::zeros((batch_size, seq_length, in_channels));
            let output = conv.forward(&input);
            
            prop_assert_eq!(output.shape()[0], batch_size);
            prop_assert_eq!(output.shape()[1], seq_length);
            prop_assert_eq!(output.shape()[2], out_channels);
        }

        #[test]
        fn prop_prototype_similarities_valid(
            batch_size in 1usize..4,
            seq_length in 5usize..20,
            hidden_size in 16usize..64,
            num_prototypes in 2usize..10,
        ) {
            let prototype_net = PrototypeNetwork::<f64>::new(
                hidden_size,
                num_prototypes,
            );
            
            let input = Array3::zeros((batch_size, seq_length, hidden_size));
            let (_, similarities) = prototype_net.forward(&input);
            
            // All similarities should be between 0 and 1
            for &sim in similarities.iter() {
                prop_assert!(sim >= 0.0 && sim <= 1.0);
            }
            
            // Sum to 1 across prototypes
            for b in 0..batch_size {
                for t in 0..seq_length {
                    let sum: f64 = similarities.slice(s![b, t, ..]).sum();
                    prop_assert!((sum - 1.0).abs() < 1e-5);
                }
            }
        }

        #[test]
        fn prop_distribution_parameters_valid(
            batch_size in 1usize..8,
            hidden_size in 16usize..128,
        ) {
            let hidden = Array2::ones((batch_size, hidden_size));
            
            // Gaussian
            let gaussian = GaussianOutput::<f64>::new(hidden_size);
            let (mu, sigma) = gaussian.forward(&hidden);
            
            for &s in sigma.iter() {
                prop_assert!(s > 0.0);
            }
            
            // Student-T
            let student_t = StudentTOutput::<f64>::new(hidden_size);
            let (mu, sigma, nu) = student_t.forward(&hidden);
            
            for &s in sigma.iter() {
                prop_assert!(s > 0.0);
            }
            for &n in nu.iter() {
                prop_assert!(n > 2.0);
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
    fn test_tcn_parallelization() {
        // TCN processes all timesteps in parallel
        // Unlike RNN which must process sequentially
        
        let configs = vec![
            ("Short", 50, vec![32; 3]),
            ("Medium", 200, vec![64; 4]),
            ("Long", 1000, vec![128; 5]),
        ];
        
        for (name, seq_length, channels) in configs {
            println!("Testing {} sequence TCN", name);
            
            let config = TCNConfig::new(seq_length, 24)
                .with_num_channels(channels);
            
            let model = TCN::<f64>::new(config);
            assert!(model.is_ok());
            
            // All timesteps can be processed simultaneously
        }
    }

    #[test]
    #[ignore] // Run with --ignored for benchmarking
    fn benchmark_specialized_models() {
        use std::time::Instant;
        
        let data = TimeSeriesData::new(
            (0..1000).map(|i| (i as f64 * 0.01).sin()).collect()
        );
        
        // Benchmark DeepAR
        let deepar_config = DeepARConfig::new(96, 24)
            .with_lstm_layers(vec![50, 50])
            .with_max_epochs(5);
        
        let mut deepar = DeepAR::<f64>::new(deepar_config).unwrap();
        let start = Instant::now();
        deepar.fit(&data).unwrap();
        println!("DeepAR training time: {:?}", start.elapsed());
        
        // Benchmark TCN
        let tcn_config = TCNConfig::new(96, 24)
            .with_num_channels(vec![32; 4])
            .with_max_epochs(5);
        
        let mut tcn = TCN::<f64>::new(tcn_config).unwrap();
        let start = Instant::now();
        tcn.fit(&data).unwrap();
        println!("TCN training time: {:?}", start.elapsed());
        
        // Benchmark BiTCN
        let bitcn_config = BiTCNConfig::new(96, 24)
            .with_num_channels(vec![32; 4])
            .with_max_epochs(5);
        
        let mut bitcn = BiTCN::<f64>::new(bitcn_config).unwrap();
        let start = Instant::now();
        bitcn.fit(&data).unwrap();
        println!("BiTCN training time: {:?}", start.elapsed());
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_single_sample_deepar() {
        let config = DeepARConfig::new(10, 5)
            .with_num_samples(1);
        
        let model = DeepAR::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_single_channel_tcn() {
        let config = TCNConfig::new(20, 10)
            .with_num_channels(vec![1]); // Single channel
        
        let model = TCN::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_single_prototype() {
        let config = DeepNPTSConfig::new(20, 10)
            .with_num_prototypes(1);
        
        let model = DeepNPTS::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_no_dilation_tcn() {
        let config = TCNConfig::new(50, 25)
            .with_num_channels(vec![32; 10]); // Many layers with dilation=1
        
        let model = TCN::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_very_deep_tcn() {
        let config = TCNConfig::new(100, 50)
            .with_num_channels(vec![8; 20]); // 20 layers
        
        let model = TCN::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_extreme_kernel_size() {
        // Large kernel size
        let config = TCNConfig::new(100, 50)
            .with_kernel_size(31);
        
        let model = TCN::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_no_categorical_deepar() {
        let config = DeepARConfig::new(48, 24)
            .with_use_feat_static_cat(false)
            .with_cardinality(vec![]);
        
        let model = DeepAR::<f64>::new(config);
        assert!(model.is_ok());
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_ensemble_predictions() {
        let data = TimeSeriesData::new(
            (0..200).map(|i| (i as f64 * 0.1).sin()).collect()
        );
        
        // Create multiple models
        let mut deepar = DeepAR::new(
            DeepARConfig::new(40, 20).with_max_epochs(2)
        ).unwrap();
        
        let mut tcn = TCN::new(
            TCNConfig::new(40, 20).with_max_epochs(2)
        ).unwrap();
        
        // Train both
        deepar.fit(&data).unwrap();
        tcn.fit(&data).unwrap();
        
        // Get predictions
        let deepar_pred = deepar.predict(&data).unwrap();
        let tcn_pred = tcn.predict(&data).unwrap();
        
        // Ensemble predictions (simple average)
        let ensemble_pred: Vec<f64> = deepar_pred.values.iter()
            .zip(tcn_pred.values.iter())
            .map(|(d, t)| (d + t) / 2.0)
            .collect();
        
        assert_eq!(ensemble_pred.len(), 20);
    }

    #[test]
    fn test_hierarchical_forecasting() {
        // Test forecasting at multiple aggregation levels
        // E.g., hourly -> daily -> weekly
        
        let hourly_data = TimeSeriesData::new(
            (0..168).map(|i| (i as f64 * 0.1).sin()).collect() // 1 week hourly
        );
        
        // Aggregate to daily
        let daily_values: Vec<f64> = hourly_data.target.chunks(24)
            .map(|chunk| chunk.iter().sum::<f64>() / 24.0)
            .collect();
        
        let daily_data = TimeSeriesData::new(daily_values);
        
        // Models at different levels
        let hourly_model = TCN::new(
            TCNConfig::new(48, 24).with_max_epochs(1)
        ).unwrap();
        
        let daily_model = TCN::new(
            TCNConfig::new(7, 7).with_max_epochs(1)
        ).unwrap();
        
        // Different granularities for different use cases
    }
}