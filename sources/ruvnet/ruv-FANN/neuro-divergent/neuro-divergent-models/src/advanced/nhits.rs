//! NHITS (Neural Hierarchical Interpolation for Time Series) Implementation
//! 
//! This module implements the NHITS model with multi-rate data sampling,
//! hierarchical temporal interpolation, and multi-step interpolation networks
//! for capturing different temporal resolutions efficiently.

use super::blocks::{
    BaseModel, ModelError, TimeSeriesData, MLPBlock, DenseLayer, 
    PoolingLayer, PoolingType, InterpolationLayer, InterpolationType,
    hierarchical_sample, calculate_expression_ratios, tensor_ops
};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use ruv_fann::ActivationFunction;
use std::collections::HashMap;

/// NHITS model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHITSConfig<T: Float> {
    /// Forecast horizon
    pub horizon: usize,
    /// Input window size
    pub input_size: usize,
    /// Multi-rate sampling configuration
    pub sampling_rates: Vec<usize>,
    /// MLP layer configurations for each resolution
    pub mlp_units: Vec<Vec<usize>>,
    /// Number of blocks per resolution level
    pub n_blocks: Vec<usize>,
    /// Pooling configuration
    pub pooling_modes: Vec<PoolingType>,
    /// Interpolation method
    pub interpolation_mode: InterpolationType,
    /// Activation function
    pub activation: ActivationFunction,
    /// Training parameters
    pub learning_rate: T,
    pub max_epochs: usize,
    pub batch_size: usize,
    /// Regularization
    pub dropout_rate: T,
    pub weight_decay: T,
    /// Loss function
    pub loss_function: LossFunction,
    /// Stack residual connections
    pub stack_residual: bool,
    /// Share parameters across blocks
    pub shared_weights: bool,
}

/// Loss function types for NHITS
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossFunction {
    MAE,  // Mean Absolute Error
    MSE,  // Mean Squared Error  
    MAPE, // Mean Absolute Percentage Error
    SMAPE, // Symmetric Mean Absolute Percentage Error
}

impl<T: Float> Default for NHITSConfig<T> {
    fn default() -> Self {
        Self {
            horizon: 1,
            input_size: 32,
            sampling_rates: vec![1, 2, 4],
            mlp_units: vec![
                vec![512, 512, 512],
                vec![512, 512, 512], 
                vec![512, 512, 512],
            ],
            n_blocks: vec![1, 1, 1],
            pooling_modes: vec![PoolingType::Max, PoolingType::Average, PoolingType::Max],
            interpolation_mode: InterpolationType::Linear,
            activation: ActivationFunction::ReLU,
            learning_rate: T::from(0.001).unwrap(),
            max_epochs: 100,
            batch_size: 32,
            dropout_rate: T::from(0.1).unwrap(),
            weight_decay: T::from(1e-4).unwrap(),
            loss_function: LossFunction::MAE,
            stack_residual: true,
            shared_weights: false,
        }
    }
}

/// Multi-resolution block for NHITS
#[derive(Debug, Clone)]
pub struct NHITSBlock<T: Float> {
    /// MLP for feature extraction at this resolution
    pub mlp: MLPBlock<T>,
    /// Pooling layer for downsampling
    pub pooling: PoolingLayer<T>,
    /// Interpolation layer for upsampling
    pub interpolation: InterpolationLayer<T>,
    /// Linear layer for backcast
    pub backcast_linear: DenseLayer<T>,
    /// Linear layer for forecast
    pub forecast_linear: DenseLayer<T>,
    /// Resolution level parameters
    pub sampling_rate: usize,
    pub input_size: usize,
    pub output_size: usize,
    /// Expression ratio for this resolution
    pub expression_ratio: T,
}

impl<T: Float> NHITSBlock<T> {
    /// Create a new NHITS block
    pub fn new(
        input_size: usize,
        output_size: usize,
        sampling_rate: usize,
        mlp_units: Vec<usize>,
        pooling_mode: PoolingType,
        interpolation_mode: InterpolationType,
        activation: ActivationFunction,
        expression_ratio: T,
    ) -> Result<Self, ModelError> {
        if mlp_units.is_empty() {
            return Err(ModelError::ConfigurationError("MLP units cannot be empty".to_string()));
        }
        
        // Calculate pooled input size
        let pooled_size = (input_size + sampling_rate - 1) / sampling_rate;
        
        // Create MLP for this resolution
        let mlp = MLPBlock::new(pooled_size, mlp_units.clone(), activation)?;
        let mlp_output_size = mlp_units.last().unwrap();
        
        // Create pooling layer
        let pooling = PoolingLayer::new(pooling_mode, sampling_rate, Some(sampling_rate));
        
        // Create interpolation layer
        let interpolation = InterpolationLayer::new(interpolation_mode, input_size);
        
        // Create linear layers for backcast and forecast
        let backcast_linear = DenseLayer::new(*mlp_output_size, pooled_size, ActivationFunction::Linear);
        let forecast_linear = DenseLayer::new(*mlp_output_size, 
            (output_size + sampling_rate - 1) / sampling_rate, ActivationFunction::Linear);
        
        Ok(Self {
            mlp,
            pooling,
            interpolation,
            backcast_linear,
            forecast_linear,
            sampling_rate,
            input_size,
            output_size,
            expression_ratio,
        })
    }
    
    /// Forward pass through the block
    pub fn forward(&mut self, input: &[T]) -> Result<(Vec<T>, Vec<T>), ModelError> {
        if input.len() != self.input_size {
            return Err(ModelError::DimensionMismatch {
                expected: self.input_size,
                actual: input.len(),
            });
        }
        
        // Downsample input using pooling
        let pooled_input = self.pooling.forward(input);
        
        // Forward through MLP
        let mlp_output = self.mlp.forward(&pooled_input)?;
        
        // Generate backcast and forecast at downsampled resolution
        let pooled_backcast = self.backcast_linear.forward(&mlp_output)?;
        let pooled_forecast = self.forecast_linear.forward(&mlp_output)?;
        
        // Upsample backcast to original resolution
        let backcast = self.interpolation.forward(&pooled_backcast);
        
        // Upsample forecast to target resolution
        let forecast_interpolation = InterpolationLayer::new(
            InterpolationType::Linear, 
            self.output_size
        );
        let forecast = forecast_interpolation.forward(&pooled_forecast);
        
        // Apply expression ratio
        let scaled_backcast: Vec<T> = backcast.iter().map(|&x| x * self.expression_ratio).collect();
        let scaled_forecast: Vec<T> = forecast.iter().map(|&x| x * self.expression_ratio).collect();
        
        Ok((scaled_backcast, scaled_forecast))
    }
    
    /// Get the effective receptive field size at this resolution
    pub fn receptive_field_size(&self) -> usize {
        self.input_size / self.sampling_rate
    }
}

/// Multi-resolution stack for NHITS
#[derive(Debug, Clone)]
pub struct NHITSStack<T: Float> {
    /// Blocks at this resolution level
    pub blocks: Vec<NHITSBlock<T>>,
    /// Sampling rate for this stack
    pub sampling_rate: usize,  
    /// Expression ratio for this stack
    pub expression_ratio: T,
    /// Stack configuration
    pub n_blocks: usize,
}

impl<T: Float> NHITSStack<T> {
    /// Create a new NHITS stack
    pub fn new(
        input_size: usize,
        output_size: usize,
        sampling_rate: usize,
        n_blocks: usize,
        mlp_units: Vec<usize>,
        pooling_mode: PoolingType,
        interpolation_mode: InterpolationType,
        activation: ActivationFunction,
        expression_ratio: T,
    ) -> Result<Self, ModelError> {
        let mut blocks = Vec::with_capacity(n_blocks);
        
        for _ in 0..n_blocks {
            let block = NHITSBlock::new(
                input_size,
                output_size,
                sampling_rate,
                mlp_units.clone(),
                pooling_mode,
                interpolation_mode,
                activation,
                expression_ratio,
            )?;
            blocks.push(block);
        }
        
        Ok(Self {
            blocks,
            sampling_rate,
            expression_ratio,
            n_blocks,
        })
    }
    
    /// Forward pass through the stack
    pub fn forward(&mut self, input: &[T]) -> Result<(Vec<T>, Vec<T>), ModelError> {
        let mut current_input = input.to_vec();
        let mut stack_forecast = vec![T::zero(); self.blocks[0].output_size];
        
        // Process each block in the stack
        for block in &mut self.blocks {
            let (backcast, forecast) = block.forward(&current_input)?;
            
            // Update residual input for next block (if input sizes match)
            if backcast.len() == current_input.len() {
                current_input = tensor_ops::subtract(&current_input, &backcast)?;
            }
            
            // Accumulate forecasts
            if forecast.len() == stack_forecast.len() {
                stack_forecast = tensor_ops::add(&stack_forecast, &forecast)?;
            }
        }
        
        Ok((current_input, stack_forecast))
    }
}

/// Main NHITS model
#[derive(Debug, Clone)]
pub struct NHITS<T: Float> {
    /// Model configuration
    pub config: NHITSConfig<T>,
    /// Multi-resolution stacks
    pub stacks: Vec<NHITSStack<T>>,
    /// Expression ratios for each resolution
    pub expression_ratios: Vec<T>,
    /// Training state
    pub is_fitted: bool,
    /// Training history
    pub training_history: Vec<T>,
}

impl<T: Float> NHITS<T> {
    /// Create a new NHITS model
    pub fn new(config: NHITSConfig<T>) -> Result<Self, ModelError> {
        config.validate()?;
        
        // Calculate expression ratios for different sampling rates
        let expression_ratios = calculate_expression_ratios(&config.sampling_rates);
        
        let mut stacks = Vec::with_capacity(config.sampling_rates.len());
        
        for (i, &sampling_rate) in config.sampling_rates.iter().enumerate() {
            let stack = NHITSStack::new(
                config.input_size,
                config.horizon,
                sampling_rate,
                config.n_blocks[i],
                config.mlp_units[i].clone(),
                config.pooling_modes[i],
                config.interpolation_mode,
                config.activation,
                expression_ratios[i],
            )?;
            stacks.push(stack);
        }
        
        Ok(Self {
            config,
            stacks,
            expression_ratios,
            is_fitted: false,
            training_history: Vec::new(),
        })
    }
    
    /// Create NHITS model with default multi-scale configuration
    pub fn multi_scale(horizon: usize, input_size: usize, max_resolution_levels: usize) -> Result<Self, ModelError> {
        // Generate sampling rates as powers of 2
        let mut sampling_rates = Vec::with_capacity(max_resolution_levels);
        let mut mlp_units = Vec::with_capacity(max_resolution_levels);
        let mut n_blocks = Vec::with_capacity(max_resolution_levels);
        let mut pooling_modes = Vec::with_capacity(max_resolution_levels);
        
        for i in 0..max_resolution_levels {
            sampling_rates.push(2_usize.pow(i as u32));
            mlp_units.push(vec![512, 512, 512]);
            n_blocks.push(1);
            pooling_modes.push(if i % 2 == 0 { PoolingType::Max } else { PoolingType::Average });
        }
        
        let config = NHITSConfig {
            horizon,
            input_size,
            sampling_rates,
            mlp_units,
            n_blocks,
            pooling_modes,
            interpolation_mode: InterpolationType::Linear,
            activation: ActivationFunction::ReLU,
            stack_residual: true,
            ..Default::default()
        };
        
        Self::new(config)
    }
    
    /// Forward pass through all resolution stacks
    pub fn forward(&mut self, input: &[T]) -> Result<Vec<T>, ModelError> {
        if input.len() != self.config.input_size {
            return Err(ModelError::DimensionMismatch {
                expected: self.config.input_size,
                actual: input.len(),
            });
        }
        
        let mut current_residual = input.to_vec();
        let mut total_forecast = vec![T::zero(); self.config.horizon];
        
        // Process each resolution stack
        for stack in &mut self.stacks {
            let (residual, forecast) = stack.forward(&current_residual)?;
            
            // Update residual for next stack (if using stack residuals)
            if self.config.stack_residual && residual.len() == current_residual.len() {
                current_residual = residual;
            }
            
            // Accumulate forecasts from all resolutions
            if forecast.len() == total_forecast.len() {
                total_forecast = tensor_ops::add(&total_forecast, &forecast)?;
            }
        }
        
        Ok(total_forecast)
    }
    
    /// Get multi-resolution forecasts (for analysis purposes)
    pub fn multi_resolution_forecast(&mut self, input: &[T]) -> Result<Vec<Vec<T>>, ModelError> {
        if input.len() != self.config.input_size {
            return Err(ModelError::DimensionMismatch {
                expected: self.config.input_size,
                actual: input.len(),
            });
        }
        
        let mut multi_res_forecasts = Vec::with_capacity(self.stacks.len());
        let mut current_residual = input.to_vec();
        
        for stack in &mut self.stacks {
            let (residual, forecast) = stack.forward(&current_residual)?;
            multi_res_forecasts.push(forecast);
            
            if self.config.stack_residual && residual.len() == current_residual.len() {
                current_residual = residual;
            }
        }
        
        Ok(multi_res_forecasts)
    }
    
    /// Analyze temporal patterns at different resolutions
    pub fn analyze_resolution_contributions(&mut self, input: &[T]) -> Result<ResolutionAnalysis<T>, ModelError> {
        let multi_res_forecasts = self.multi_resolution_forecast(input)?;
        let total_forecast = self.forward(input)?;
        
        // Calculate contribution weights for each resolution
        let mut contributions = Vec::with_capacity(self.stacks.len());
        
        for (i, forecast) in multi_res_forecasts.iter().enumerate() {
            let forecast_magnitude: T = forecast.iter().map(|&x| x.abs()).sum();
            let total_magnitude: T = total_forecast.iter().map(|&x| x.abs()).sum();
            
            let contribution_ratio = if total_magnitude != T::zero() {
                forecast_magnitude / total_magnitude
            } else {
                T::zero()
            };
            
            contributions.push(ResolutionContribution {
                sampling_rate: self.config.sampling_rates[i],
                expression_ratio: self.expression_ratios[i],
                contribution_ratio,
                forecast_magnitude,
                receptive_field_size: self.stacks[i].blocks[0].receptive_field_size(),
            });
        }
        
        Ok(ResolutionAnalysis {
            total_forecast,
            resolution_forecasts: multi_res_forecasts,
            contributions,
        })
    }
    
    /// Get model complexity (parameter count)
    pub fn complexity(&self) -> usize {
        self.stacks.iter()
            .enumerate()
            .map(|(i, stack)| {
                let block_params = self.estimate_block_parameters(i);
                stack.n_blocks * block_params
            })
            .sum()
    }
    
    /// Estimate parameters for a block at given resolution level
    fn estimate_block_parameters(&self, stack_index: usize) -> usize {
        let sampling_rate = self.config.sampling_rates[stack_index];
        let mlp_units = &self.config.mlp_units[stack_index];
        
        let pooled_input_size = (self.config.input_size + sampling_rate - 1) / sampling_rate;
        let pooled_output_size = (self.config.horizon + sampling_rate - 1) / sampling_rate;
        
        let mut params = 0;
        let mut prev_size = pooled_input_size;
        
        // MLP parameters
        for &size in mlp_units {
            params += prev_size * size + size; // weights + biases
            prev_size = size;
        }
        
        // Backcast and forecast linear layers
        params += prev_size * pooled_input_size; // backcast
        params += prev_size * pooled_output_size; // forecast
        
        params
    }
    
    /// Calculate loss
    fn calculate_loss(&self, predictions: &[T], targets: &[T]) -> Result<T, ModelError> {
        if predictions.len() != targets.len() {
            return Err(ModelError::DimensionMismatch {
                expected: targets.len(),
                actual: predictions.len(),
            });
        }
        
        match self.config.loss_function {
            LossFunction::MAE => {
                let mae = predictions.iter()
                    .zip(targets.iter())
                    .map(|(&pred, &target)| (pred - target).abs())
                    .sum::<T>() / T::from(predictions.len()).unwrap();
                Ok(mae)
            },
            LossFunction::MSE => {
                let mse = predictions.iter()
                    .zip(targets.iter())
                    .map(|(&pred, &target)| (pred - target).powi(2))
                    .sum::<T>() / T::from(predictions.len()).unwrap();
                Ok(mse)
            },
            LossFunction::MAPE => {
                let mape = predictions.iter()
                    .zip(targets.iter())
                    .filter(|(_, &target)| target != T::zero())
                    .map(|(&pred, &target)| ((pred - target) / target).abs())
                    .sum::<T>() / T::from(predictions.len()).unwrap() * T::from(100.0).unwrap();
                Ok(mape)
            },
            LossFunction::SMAPE => {
                let smape = predictions.iter()
                    .zip(targets.iter())
                    .map(|(&pred, &target)| {
                        let denominator = (pred.abs() + target.abs()) / T::from(2.0).unwrap();
                        if denominator != T::zero() {
                            (pred - target).abs() / denominator
                        } else {
                            T::zero()
                        }
                    })
                    .sum::<T>() / T::from(predictions.len()).unwrap() * T::from(100.0).unwrap();
                Ok(smape)
            },
        }
    }
}

/// Resolution contribution analysis
#[derive(Debug, Clone)]
pub struct ResolutionContribution<T: Float> {
    pub sampling_rate: usize,
    pub expression_ratio: T,
    pub contribution_ratio: T,
    pub forecast_magnitude: T,
    pub receptive_field_size: usize,
}

/// Resolution analysis results
#[derive(Debug, Clone)]
pub struct ResolutionAnalysis<T: Float> {
    pub total_forecast: Vec<T>,
    pub resolution_forecasts: Vec<Vec<T>>,
    pub contributions: Vec<ResolutionContribution<T>>,
}

impl<T: Float> BaseModel<T> for NHITS<T> {
    type Config = NHITSConfig<T>;
    
    fn name(&self) -> &'static str {
        "NHITS"
    }
    
    fn horizon(&self) -> usize {
        self.config.horizon
    }
    
    fn input_size(&self) -> usize {
        self.config.input_size
    }
    
    fn fit(&mut self, data: &TimeSeriesData<T>) -> Result<(), ModelError> {
        if data.series.len() < self.config.input_size + self.config.horizon {
            return Err(ModelError::InvalidInput(
                "Insufficient data length for training".to_string()
            ));
        }
        
        // Create training windows for multi-resolution learning
        let mut training_losses = Vec::new();
        let n_windows = data.series.len() - self.config.input_size - self.config.horizon + 1;
        
        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = T::zero();
            
            for i in 0..n_windows {
                let input_window = &data.series[i..i + self.config.input_size];
                let target_window = &data.series[i + self.config.input_size..i + self.config.input_size + self.config.horizon];
                
                let prediction = self.forward(input_window)?;
                let loss = self.calculate_loss(&prediction, target_window)?;
                epoch_loss = epoch_loss + loss;
            }
            
            epoch_loss = epoch_loss / T::from(n_windows).unwrap();
            training_losses.push(epoch_loss);
            
            // Simple early stopping
            if training_losses.len() > 10 {
                let recent_avg = training_losses[training_losses.len() - 5..].iter().sum::<T>() / T::from(5).unwrap();
                let earlier_avg = training_losses[training_losses.len() - 10..training_losses.len() - 5].iter().sum::<T>() / T::from(5).unwrap();
                
                if recent_avg >= earlier_avg {
                    break; // Stop if not improving
                }
            }
        }
        
        self.training_history = training_losses;
        self.is_fitted = true;
        
        Ok(())
    }
    
    fn predict(&self, data: &TimeSeriesData<T>) -> Result<Vec<T>, ModelError> {
        if !self.is_fitted {
            return Err(ModelError::ConfigurationError(
                "Model must be fitted before prediction".to_string()
            ));
        }
        
        if data.series.len() < self.config.input_size {
            return Err(ModelError::InvalidInput(
                "Insufficient data length for prediction".to_string()
            ));
        }
        
        let input_window = &data.series[data.series.len() - self.config.input_size..];
        let mut model_copy = self.clone();
        model_copy.forward(input_window)
    }
    
    fn predict_quantiles(&self, data: &TimeSeriesData<T>, _quantiles: &[T]) -> Result<Vec<Vec<T>>, ModelError> {
        // NHITS is deterministic, return point predictions for all quantiles
        let point_prediction = self.predict(data)?;
        Ok(vec![point_prediction; _quantiles.len()])
    }
    
    fn parameters_count(&self) -> usize {
        self.complexity()
    }
    
    fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

impl<T: Float> NHITSConfig<T> {
    /// Validate NHITS configuration
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.horizon == 0 {
            return Err(ModelError::ConfigurationError("Horizon must be positive".to_string()));
        }
        
        if self.input_size == 0 {
            return Err(ModelError::ConfigurationError("Input size must be positive".to_string()));
        }
        
        if self.sampling_rates.is_empty() {
            return Err(ModelError::ConfigurationError("At least one sampling rate is required".to_string()));
        }
        
        if self.mlp_units.len() != self.sampling_rates.len() {
            return Err(ModelError::ConfigurationError("MLP units must match sampling rates length".to_string()));
        }
        
        if self.n_blocks.len() != self.sampling_rates.len() {
            return Err(ModelError::ConfigurationError("n_blocks must match sampling rates length".to_string()));
        }
        
        if self.pooling_modes.len() != self.sampling_rates.len() {
            return Err(ModelError::ConfigurationError("Pooling modes must match sampling rates length".to_string()));
        }
        
        // Check that sampling rates are valid
        for &rate in &self.sampling_rates {
            if rate == 0 {
                return Err(ModelError::ConfigurationError("Sampling rates must be positive".to_string()));
            }
            if rate > self.input_size {
                return Err(ModelError::ConfigurationError("Sampling rate cannot exceed input size".to_string()));
            }
        }
        
        // Check MLP configurations
        for (i, units) in self.mlp_units.iter().enumerate() {
            if units.is_empty() {
                return Err(ModelError::ConfigurationError(format!("MLP units for resolution {} cannot be empty", i)));
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nhits_creation() {
        let config = NHITSConfig::<f32>::default();
        let model = NHITS::new(config);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.name(), "NHITS");
        assert!(!model.is_fitted());
    }
    
    #[test]
    fn test_multi_scale_nhits() {
        let model = NHITS::<f32>::multi_scale(12, 48, 3);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.horizon(), 12);
        assert_eq!(model.input_size(), 48);
        assert_eq!(model.stacks.len(), 3);
        assert_eq!(model.config.sampling_rates, vec![1, 2, 4]);
    }
    
    #[test]
    fn test_nhits_block_forward() {
        let mut block = NHITSBlock::<f32>::new(
            16,
            8,
            2,
            vec![32, 16],
            PoolingType::Average,
            InterpolationType::Linear,
            ActivationFunction::ReLU,
            0.5,
        ).unwrap();
        
        let input = vec![1.0; 16];
        let result = block.forward(&input);
        assert!(result.is_ok());
        
        let (backcast, forecast) = result.unwrap();
        assert_eq!(backcast.len(), 16);
        assert_eq!(forecast.len(), 8);
    }
    
    #[test]
    fn test_expression_ratios() {
        let rates = vec![1, 2, 4, 8];
        let ratios = calculate_expression_ratios::<f32>(&rates);
        assert_eq!(ratios.len(), 4);
        assert_eq!(ratios[3], 1.0); // Highest rate should have ratio 1.0
        assert!(ratios[0] < ratios[1]); // Lower rates should have lower ratios
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = NHITSConfig::<f32>::default();
        assert!(config.validate().is_ok());
        
        // Test invalid horizon
        config.horizon = 0;
        assert!(config.validate().is_err());
        
        // Reset and test mismatched vector lengths
        config = NHITSConfig::default();
        config.mlp_units.pop();
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_hierarchical_sampling() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rates = vec![1, 2, 4];
        let samples = hierarchical_sample(&input, &rates);
        
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0], input); // Rate 1: no sampling
        assert_eq!(samples[1], vec![1.0, 3.0, 5.0, 7.0]); // Rate 2: every 2nd
        assert_eq!(samples[2], vec![1.0, 5.0]); // Rate 4: every 4th
    }
    
    #[test]
    fn test_multi_resolution_forecast() {
        let mut model = NHITS::<f32>::multi_scale(6, 24, 2).unwrap();
        let input = vec![1.0; 24];
        
        let multi_res_result = model.multi_resolution_forecast(&input);
        assert!(multi_res_result.is_ok());
        
        let forecasts = multi_res_result.unwrap();
        assert_eq!(forecasts.len(), 2); // Two resolution levels
        assert_eq!(forecasts[0].len(), 6); // Horizon size
        assert_eq!(forecasts[1].len(), 6);
    }
    
    #[test]
    fn test_resolution_analysis() {
        let mut model = NHITS::<f32>::multi_scale(4, 16, 2).unwrap();
        let input = vec![1.0; 16];
        
        let analysis = model.analyze_resolution_contributions(&input);
        assert!(analysis.is_ok());
        
        let analysis = analysis.unwrap();
        assert_eq!(analysis.contributions.len(), 2);
        assert_eq!(analysis.resolution_forecasts.len(), 2);
        assert_eq!(analysis.total_forecast.len(), 4);
    }
}