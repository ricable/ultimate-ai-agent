//! Comprehensive unit tests for advanced models
//!
//! This module tests N-BEATS, N-BEATS-X, and N-HiTS implementations including
//! their blocks, stacks, and various configurations.

use neuro_divergent_models::advanced::*;
use neuro_divergent_models::core::*;
use neuro_divergent_models::config::*;
use neuro_divergent_models::errors::*;
use neuro_divergent_core::prelude::*;
use ndarray::{Array1, Array2, Array3};
use num_traits::Float;
use proptest::prelude::*;
use approx::assert_relative_eq;
use std::collections::HashMap;

// ============================================================================
// N-BEATS Block Tests
// ============================================================================

#[cfg(test)]
mod nbeats_block_tests {
    use super::*;

    #[test]
    fn test_generic_block() {
        let input_size = 20;
        let theta_size = 10;
        let horizon = 5;
        let hidden_units = vec![32, 16];
        
        let mut block = GenericBlock::<f64>::new(
            input_size,
            theta_size,
            horizon,
            hidden_units,
        );
        
        let batch_size = 2;
        let input = Array2::ones((batch_size, input_size));
        
        let (backcast, forecast) = block.forward(&input);
        
        assert_eq!(backcast.shape(), &[batch_size, input_size]);
        assert_eq!(forecast.shape(), &[batch_size, horizon]);
    }

    #[test]
    fn test_trend_block() {
        let input_size = 20;
        let horizon = 10;
        let polynomial_degree = 3;
        let hidden_units = vec![32, 16];
        
        let mut block = TrendBlock::<f64>::new(
            input_size,
            horizon,
            polynomial_degree,
            hidden_units,
        );
        
        let batch_size = 2;
        let input = Array2::ones((batch_size, input_size));
        
        let (backcast, forecast) = block.forward(&input);
        
        assert_eq!(backcast.shape(), &[batch_size, input_size]);
        assert_eq!(forecast.shape(), &[batch_size, horizon]);
        
        // Trend should be smooth (polynomial)
        // Check that forecast represents a polynomial trend
    }

    #[test]
    fn test_seasonality_block() {
        let input_size = 24;
        let horizon = 12;
        let num_harmonics = 5;
        let hidden_units = vec![32, 16];
        
        let mut block = SeasonalityBlock::<f64>::new(
            input_size,
            horizon,
            num_harmonics,
            hidden_units,
        );
        
        let batch_size = 2;
        let input = Array2::ones((batch_size, input_size));
        
        let (backcast, forecast) = block.forward(&input);
        
        assert_eq!(backcast.shape(), &[batch_size, input_size]);
        assert_eq!(forecast.shape(), &[batch_size, horizon]);
        
        // Seasonality should be periodic (Fourier series)
    }

    #[test]
    fn test_block_basis_functions() {
        // Test polynomial basis for trend
        let degree = 3;
        let length = 10;
        let trend_basis = create_polynomial_basis::<f64>(degree, length);
        assert_eq!(trend_basis.shape(), &[degree + 1, length]);
        
        // Test Fourier basis for seasonality
        let num_harmonics = 5;
        let seasonality_basis = create_fourier_basis::<f64>(num_harmonics, length);
        assert_eq!(seasonality_basis.shape(), &[2 * num_harmonics, length]);
    }

    #[test]
    fn test_block_stacking() {
        let input_size = 20;
        let horizon = 10;
        let batch_size = 2;
        
        // Create multiple blocks
        let mut blocks: Vec<Box<dyn Block<f64>>> = vec![
            Box::new(TrendBlock::new(input_size, horizon, 2, vec![32])),
            Box::new(SeasonalityBlock::new(input_size, horizon, 3, vec![32])),
            Box::new(GenericBlock::new(input_size, 8, horizon, vec![32])),
        ];
        
        let input = Array2::ones((batch_size, input_size));
        let mut residual = input.clone();
        let mut total_forecast = Array2::zeros((batch_size, horizon));
        
        // Stack blocks
        for block in blocks.iter_mut() {
            let (backcast, forecast) = block.forward(&residual);
            residual = residual - backcast;
            total_forecast = total_forecast + forecast;
        }
        
        assert_eq!(total_forecast.shape(), &[batch_size, horizon]);
    }
}

// ============================================================================
// N-BEATS Model Tests
// ============================================================================

#[cfg(test)]
mod nbeats_model_tests {
    use super::*;

    #[test]
    fn test_nbeats_config_creation() {
        let config = NBEATSConfig::new(24, 12)
            .with_stack_types(vec![
                StackType::Trend(3),
                StackType::Seasonality(5),
                StackType::Generic,
            ])
            .with_num_blocks_per_stack(vec![3, 3, 3])
            .with_hidden_units_per_stack(vec![
                vec![256, 256],
                vec![256, 256],
                vec![256, 256],
            ])
            .with_share_weights(false)
            .with_learning_rate(0.001)
            .with_max_epochs(100);

        assert_eq!(config.input_size, 24);
        assert_eq!(config.horizon, 12);
        assert_eq!(config.stack_types.len(), 3);
        assert_eq!(config.num_blocks_per_stack, vec![3, 3, 3]);
        assert!(!config.share_weights);
    }

    #[test]
    fn test_nbeats_config_validation() {
        // Valid config
        let valid_config = NBEATSConfig::new(24, 12);
        assert!(valid_config.validate().is_ok());

        // Mismatched stack configuration
        let mismatched = NBEATSConfig::new(24, 12)
            .with_stack_types(vec![StackType::Trend(3), StackType::Generic])
            .with_num_blocks_per_stack(vec![3]); // Should be 2 elements
        assert!(mismatched.validate().is_err());

        // Empty stacks
        let empty_stacks = NBEATSConfig::new(24, 12)
            .with_stack_types(vec![]);
        assert!(empty_stacks.validate().is_err());
    }

    #[test]
    fn test_nbeats_interpretable() {
        // Interpretable N-BEATS with trend and seasonality
        let config = NBEATSConfig::new(48, 24)
            .with_stack_types(vec![
                StackType::Trend(2),
                StackType::Seasonality(5),
            ])
            .with_num_blocks_per_stack(vec![3, 3])
            .with_hidden_units_per_stack(vec![
                vec![256, 256],
                vec![256, 256],
            ]);

        let model = NBEATS::<f64>::new(config);
        assert!(model.is_ok());
        
        // Model should provide interpretable outputs
        // Trend stack captures long-term patterns
        // Seasonality stack captures periodic patterns
    }

    #[test]
    fn test_nbeats_generic() {
        // Generic N-BEATS for pure forecasting performance
        let config = NBEATSConfig::new(30, 15)
            .with_stack_types(vec![StackType::Generic; 5])
            .with_num_blocks_per_stack(vec![1; 5])
            .with_hidden_units_per_stack(vec![vec![512, 512]; 5]);

        let model = NBEATS::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_nbeats_workflow() {
        let data = TimeSeriesData::new(
            (0..200).map(|i| {
                let trend = i as f64 * 0.1;
                let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin() * 5.0;
                trend + seasonal
            }).collect()
        );

        let config = NBEATSConfig::new(48, 24)
            .with_stack_types(vec![
                StackType::Trend(3),
                StackType::Seasonality(5),
            ])
            .with_num_blocks_per_stack(vec![2, 2])
            .with_max_epochs(5);

        let mut model = NBEATS::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&data).unwrap();
        assert_eq!(forecast.len(), 24);
    }

    #[test]
    fn test_nbeats_ensemble_mode() {
        let config = NBEATSConfig::new(30, 15)
            .with_ensemble_mode(true)
            .with_num_ensemble_members(5);

        let model = NBEATS::<f64>::new(config);
        assert!(model.is_ok());
        
        // Ensemble should train multiple models and average predictions
    }

    #[test]
    fn test_weight_sharing() {
        let data = TimeSeriesData::new(vec![1.0; 100]);
        
        // Without weight sharing
        let config_no_share = NBEATSConfig::new(20, 10)
            .with_share_weights(false)
            .with_max_epochs(1);
        
        let mut model_no_share = NBEATS::<f64>::new(config_no_share).unwrap();
        assert!(model_no_share.fit(&data).is_ok());
        
        // With weight sharing
        let config_share = NBEATSConfig::new(20, 10)
            .with_share_weights(true)
            .with_max_epochs(1);
        
        let mut model_share = NBEATS::<f64>::new(config_share).unwrap();
        assert!(model_share.fit(&data).is_ok());
        
        // Models with weight sharing should have fewer parameters
    }
}

// ============================================================================
// N-BEATS-X Tests (with exogenous variables)
// ============================================================================

#[cfg(test)]
mod nbeatsx_tests {
    use super::*;

    #[test]
    fn test_nbeatsx_config() {
        let config = NBEATSXConfig::new(24, 12, 5)
            .with_static_covariates(3)
            .with_stack_types(vec![
                StackType::Trend(3),
                StackType::Seasonality(5),
                StackType::Exogenous,
                StackType::Generic,
            ])
            .with_covariate_projection_sizes(vec![16, 16, 32, 32]);

        assert_eq!(config.input_size, 24);
        assert_eq!(config.horizon, 12);
        assert_eq!(config.num_exogenous, 5);
        assert_eq!(config.num_static_covariates, Some(3));
        assert_eq!(config.stack_types.len(), 4);
    }

    #[test]
    fn test_exogenous_block() {
        let input_size = 20;
        let horizon = 10;
        let num_exogenous = 5;
        let hidden_units = vec![32, 16];
        let covariate_projection_size = 16;
        
        let mut block = ExogenousBlock::<f64>::new(
            input_size,
            horizon,
            num_exogenous,
            hidden_units,
            covariate_projection_size,
        );
        
        let batch_size = 2;
        let input = Array2::ones((batch_size, input_size));
        let exogenous = Array3::ones((batch_size, input_size + horizon, num_exogenous));
        
        let (backcast, forecast) = block.forward_with_covariates(&input, &exogenous);
        
        assert_eq!(backcast.shape(), &[batch_size, input_size]);
        assert_eq!(forecast.shape(), &[batch_size, horizon]);
    }

    #[test]
    fn test_nbeatsx_workflow() {
        // Create data with exogenous variables
        let n_samples = 200;
        let n_exogenous = 3;
        
        let target: Vec<f64> = (0..n_samples).map(|i| i as f64 * 0.1).collect();
        let exogenous = Array2::from_shape_fn((n_samples, n_exogenous), |(i, j)| {
            (i + j) as f64 * 0.01
        });
        
        let data = TimeSeriesData::new(target)
            .with_exogenous(exogenous);

        let config = NBEATSXConfig::new(48, 24, n_exogenous)
            .with_stack_types(vec![
                StackType::Trend(2),
                StackType::Exogenous,
                StackType::Generic,
            ])
            .with_num_blocks_per_stack(vec![2, 2, 2])
            .with_max_epochs(5);

        let mut model = NBEATSX::new(config).unwrap();

        // Train
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        // Predict with future exogenous
        let future_exogenous = Array2::ones((24, n_exogenous));
        let forecast = model.predict_with_exogenous(&data, &future_exogenous).unwrap();
        assert_eq!(forecast.len(), 24);
    }

    #[test]
    fn test_static_covariates() {
        let config = NBEATSXConfig::new(30, 15, 5)
            .with_static_covariates(3)
            .with_static_covariate_embedding_dim(8);

        let model = NBEATSX::<f64>::new(config);
        assert!(model.is_ok());
        
        // Static covariates should be embedded and used across all time steps
    }
}

// ============================================================================
// N-HiTS Tests
// ============================================================================

#[cfg(test)]
mod nhits_tests {
    use super::*;

    #[test]
    fn test_nhits_config() {
        let config = NHiTSConfig::new(60, 30)
            .with_num_stacks(3)
            .with_num_blocks(vec![1, 1, 1])
            .with_pooling_sizes(vec![4, 2, 1])
            .with_downsample_frequencies(vec![24, 12, 1])
            .with_hidden_units(vec![256, 128, 64])
            .with_interpolation_mode(InterpolationMode::Linear)
            .with_learning_rate(0.001);

        assert_eq!(config.input_size, 60);
        assert_eq!(config.horizon, 30);
        assert_eq!(config.num_stacks, 3);
        assert_eq!(config.pooling_sizes, vec![4, 2, 1]);
        assert_eq!(config.downsample_frequencies, vec![24, 12, 1]);
    }

    #[test]
    fn test_multi_rate_sampling() {
        let input_size = 96;
        let horizon = 48;
        let batch_size = 2;
        
        // Test different pooling sizes
        let pooling_sizes = vec![8, 4, 2, 1];
        
        for pool_size in pooling_sizes {
            let pooled_size = input_size / pool_size;
            let block = HierarchicalInterpolation::<f64>::new(
                pooled_size,
                horizon,
                pool_size,
                vec![128, 64],
                InterpolationMode::Linear,
            );
            
            let input = Array2::ones((batch_size, input_size));
            let pooled = maxpool_1d(&input, pool_size);
            
            assert_eq!(pooled.shape(), &[batch_size, pooled_size]);
        }
    }

    #[test]
    fn test_interpolation_modes() {
        let modes = vec![
            InterpolationMode::Linear,
            InterpolationMode::Cubic,
            InterpolationMode::Nearest,
        ];
        
        for mode in modes {
            let config = NHiTSConfig::new(48, 24)
                .with_interpolation_mode(mode)
                .with_max_epochs(1);
            
            let model = NHiTS::<f64>::new(config);
            assert!(model.is_ok());
        }
    }

    #[test]
    fn test_nhits_workflow() {
        // Create multi-scale data
        let data: Vec<f64> = (0..500).map(|i| {
            let daily = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin();
            let weekly = (i as f64 * 2.0 * std::f64::consts::PI / 168.0).sin() * 2.0;
            let trend = i as f64 * 0.01;
            daily + weekly + trend
        }).collect();
        
        let ts_data = TimeSeriesData::new(data);

        let config = NHiTSConfig::new(168, 48) // Week input, 2 days forecast
            .with_num_stacks(3)
            .with_pooling_sizes(vec![24, 6, 1]) // Day, 6-hour, hourly
            .with_downsample_frequencies(vec![48, 8, 1])
            .with_max_epochs(5);

        let mut model = NHiTS::new(config).unwrap();

        // Train
        assert!(model.fit(&ts_data).is_ok());
        assert!(model.is_fitted());

        // Predict
        let forecast = model.predict(&ts_data).unwrap();
        assert_eq!(forecast.len(), 48);
    }

    #[test]
    fn test_hierarchical_decomposition() {
        // N-HiTS should decompose signal at multiple rates
        let config = NHiTSConfig::new(96, 24)
            .with_num_stacks(3)
            .with_pooling_sizes(vec![8, 4, 1])
            .with_hierarchical_decomposition(true);

        let model = NHiTS::<f64>::new(config);
        assert!(model.is_ok());
        
        // Each stack handles different frequency components
        // Stack 1: Low frequency (pooling 8)
        // Stack 2: Medium frequency (pooling 4)  
        // Stack 3: High frequency (pooling 1)
    }

    #[test]
    fn test_long_horizon_efficiency() {
        // N-HiTS is designed for long-horizon forecasting
        let long_horizon = 720; // 30 days hourly
        
        let config = NHiTSConfig::new(720, long_horizon)
            .with_num_stacks(5)
            .with_pooling_sizes(vec![168, 24, 8, 4, 1]) // Week, day, 8h, 4h, hourly
            .with_max_epochs(1);

        let model = NHiTS::<f64>::new(config);
        assert!(model.is_ok());
        
        // Should handle long horizons efficiently through multi-rate sampling
    }
}

// ============================================================================
// Common Advanced Model Tests
// ============================================================================

#[cfg(test)]
mod common_advanced_tests {
    use super::*;

    #[test]
    fn test_residual_connections() {
        // All advanced models use residual connections
        let models = vec![
            ("N-BEATS", "residual stacks"),
            ("N-HiTS", "hierarchical residuals"),
        ];
        
        for (name, residual_type) in models {
            println!("Testing {} with {}", name, residual_type);
            // Residual connections help with training stability
        }
    }

    #[test]
    fn test_doubly_residual_stacking() {
        // N-BEATS uses doubly residual stacking
        let input_size = 30;
        let horizon = 15;
        let batch_size = 2;
        
        let input = Array2::ones((batch_size, input_size));
        
        // Forward through blocks
        let mut block_input = input.clone();
        let mut total_backcast = Array2::zeros((batch_size, input_size));
        let mut total_forecast = Array2::zeros((batch_size, horizon));
        
        for i in 0..3 {
            let mut block = GenericBlock::<f64>::new(
                input_size,
                8,
                horizon,
                vec![32],
            );
            
            let (backcast, forecast) = block.forward(&block_input);
            
            // Doubly residual
            block_input = block_input - &backcast;
            total_backcast = total_backcast + backcast;
            total_forecast = total_forecast + forecast;
        }
        
        // Check shapes preserved
        assert_eq!(total_forecast.shape(), &[batch_size, horizon]);
    }

    #[test]
    fn test_basis_function_projections() {
        // Test polynomial basis
        let poly_basis = create_polynomial_basis::<f64>(3, 10);
        assert_eq!(poly_basis.shape()[0], 4); // degree + 1
        
        // Test Fourier basis
        let fourier_basis = create_fourier_basis::<f64>(5, 10);
        assert_eq!(fourier_basis.shape()[0], 10); // 2 * num_harmonics
        
        // Basis functions should be orthogonal or well-conditioned
    }

    #[test]
    fn test_model_comparison() {
        let data = TimeSeriesData::new(
            (0..200).map(|i| (i as f64 * 0.1).sin()).collect()
        );
        
        // N-BEATS config
        let nbeats_config = NBEATSConfig::new(40, 20)
            .with_max_epochs(2);
        
        // N-HiTS config
        let nhits_config = NHiTSConfig::new(40, 20)
            .with_max_epochs(2);
        
        // Both should handle the same data
        let mut nbeats = NBEATS::<f64>::new(nbeats_config).unwrap();
        let mut nhits = NHiTS::<f64>::new(nhits_config).unwrap();
        
        assert!(nbeats.fit(&data).is_ok());
        assert!(nhits.fit(&data).is_ok());
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
        fn prop_block_output_shapes(
            batch_size in 1usize..16,
            input_size in 10usize..50,
            horizon in 5usize..30,
            hidden_size in 16usize..128,
        ) {
            // Generic block
            let mut generic = GenericBlock::<f64>::new(
                input_size,
                hidden_size / 2,
                horizon,
                vec![hidden_size],
            );
            
            let input = Array2::zeros((batch_size, input_size));
            let (backcast, forecast) = generic.forward(&input);
            
            prop_assert_eq!(backcast.shape(), &[batch_size, input_size]);
            prop_assert_eq!(forecast.shape(), &[batch_size, horizon]);
        }

        #[test]
        fn prop_pooling_preserves_batch_size(
            batch_size in 1usize..16,
            seq_length in 10usize..100,
            pool_size in 1usize..10,
        ) {
            let adjusted_length = (seq_length / pool_size) * pool_size;
            let input = Array2::ones((batch_size, adjusted_length));
            
            let pooled = maxpool_1d(&input, pool_size);
            prop_assert_eq!(pooled.shape()[0], batch_size);
            prop_assert_eq!(pooled.shape()[1], adjusted_length / pool_size);
        }

        #[test]
        fn prop_basis_functions_span_space(
            degree in 1usize..10,
            length in 10usize..50,
        ) {
            let poly_basis = create_polynomial_basis::<f64>(degree, length);
            prop_assert_eq!(poly_basis.shape(), &[degree + 1, length]);
            
            // First basis should be constant (all ones)
            let first_basis = poly_basis.slice(s![0, ..]);
            for val in first_basis.iter() {
                prop_assert!((*val - 1.0).abs() < 1e-10);
            }
        }
    }
}

// ============================================================================
// Performance and Scaling Tests
// ============================================================================

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_computational_complexity() {
        // N-BEATS: O(num_stacks * num_blocks * hidden_size^2)
        // N-HiTS: O(num_stacks * (input_size/pool_size) * hidden_size^2)
        
        let configs = vec![
            ("Small", 24, 12, vec![128]),
            ("Medium", 96, 48, vec![256, 128]),
            ("Large", 336, 168, vec![512, 256, 128]),
        ];
        
        for (name, input_size, horizon, hidden) in configs {
            println!("Testing {} configuration", name);
            
            let nbeats = NBEATSConfig::new(input_size, horizon)
                .with_hidden_units_per_stack(vec![hidden.clone(); 3]);
            
            let nhits = NHiTSConfig::new(input_size, horizon)
                .with_hidden_units(hidden);
            
            // Both should be constructible
            assert!(NBEATS::<f64>::new(nbeats).is_ok());
            assert!(NHiTS::<f64>::new(nhits).is_ok());
        }
    }

    #[test]
    #[ignore] // Run with --ignored for benchmarking
    fn benchmark_advanced_models() {
        use std::time::Instant;
        
        let data = TimeSeriesData::new(
            (0..5000).map(|i| (i as f64 * 0.01).sin()).collect()
        );
        
        // Benchmark N-BEATS
        let nbeats_config = NBEATSConfig::new(168, 24)
            .with_max_epochs(10);
        
        let mut nbeats = NBEATS::<f64>::new(nbeats_config).unwrap();
        
        let start = Instant::now();
        nbeats.fit(&data).unwrap();
        println!("N-BEATS training time: {:?}", start.elapsed());
        
        // Benchmark N-HiTS
        let nhits_config = NHiTSConfig::new(168, 24)
            .with_max_epochs(10);
        
        let mut nhits = NHiTS::<f64>::new(nhits_config).unwrap();
        
        let start = Instant::now();
        nhits.fit(&data).unwrap();
        println!("N-HiTS training time: {:?}", start.elapsed());
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_single_block_stack() {
        let config = NBEATSConfig::new(20, 10)
            .with_num_blocks_per_stack(vec![1]);
        
        let model = NBEATS::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_minimal_architecture() {
        // Minimal N-BEATS
        let nbeats_config = NBEATSConfig::new(10, 5)
            .with_stack_types(vec![StackType::Generic])
            .with_num_blocks_per_stack(vec![1])
            .with_hidden_units_per_stack(vec![vec![8]]);
        
        assert!(NBEATS::<f64>::new(nbeats_config).is_ok());
        
        // Minimal N-HiTS
        let nhits_config = NHiTSConfig::new(10, 5)
            .with_num_stacks(1)
            .with_pooling_sizes(vec![1])
            .with_hidden_units(vec![8]);
        
        assert!(NHiTS::<f64>::new(nhits_config).is_ok());
    }

    #[test]
    fn test_extreme_pooling() {
        // Pool entire sequence
        let config = NHiTSConfig::new(100, 10)
            .with_pooling_sizes(vec![100]);
        
        let model = NHiTS::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_very_long_horizon() {
        // Test with horizon longer than input
        let config = NBEATSConfig::new(24, 168); // 1 day input, 1 week forecast
        
        let model = NBEATS::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_high_degree_polynomial() {
        // High degree polynomial for trend
        let config = NBEATSConfig::new(50, 25)
            .with_stack_types(vec![StackType::Trend(10)]); // 10th degree polynomial
        
        let model = NBEATS::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_many_harmonics() {
        // Many harmonics for seasonality
        let config = NBEATSConfig::new(100, 50)
            .with_stack_types(vec![StackType::Seasonality(20)]); // 20 harmonics
        
        let model = NBEATS::<f64>::new(config);
        assert!(model.is_ok());
    }
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn test_polynomial_basis_properties() {
        let degree = 3;
        let length = 10;
        let basis = create_polynomial_basis::<f64>(degree, length);
        
        // Check dimensions
        assert_eq!(basis.shape(), &[degree + 1, length]);
        
        // Check polynomial properties
        // T_0 = 1 (constant)
        for i in 0..length {
            assert_relative_eq!(basis[[0, i]], 1.0);
        }
        
        // T_1 = t (linear)
        for i in 0..length {
            let t = -1.0 + 2.0 * i as f64 / (length - 1) as f64;
            assert_relative_eq!(basis[[1, i]], t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fourier_basis_properties() {
        let num_harmonics = 3;
        let length = 12;
        let basis = create_fourier_basis::<f64>(num_harmonics, length);
        
        // Check dimensions
        assert_eq!(basis.shape(), &[2 * num_harmonics, length]);
        
        // Check orthogonality (approximately)
        for i in 0..2*num_harmonics {
            for j in i+1..2*num_harmonics {
                let dot_product: f64 = basis.slice(s![i, ..])
                    .iter()
                    .zip(basis.slice(s![j, ..]).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                
                // Should be approximately orthogonal
                assert!(dot_product.abs() < 1.0);
            }
        }
    }

    #[test]
    fn test_maxpool_1d() {
        let input = Array2::from_shape_vec(
            (2, 8),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
        ).unwrap();
        
        let pooled = maxpool_1d(&input, 2);
        
        assert_eq!(pooled.shape(), &[2, 4]);
        assert_eq!(pooled[[0, 0]], 2.0); // max(1, 2)
        assert_eq!(pooled[[0, 1]], 4.0); // max(3, 4)
        assert_eq!(pooled[[1, 0]], 8.0); // max(8, 7)
        assert_eq!(pooled[[1, 1]], 6.0); // max(6, 5)
    }
}