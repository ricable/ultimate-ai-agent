//! NBEATS (Neural Basis Expansion Analysis for Time Series) Implementation
//! 
//! This module implements the NBEATS model with doubly residual stacking,
//! supporting both interpretable and generic architectures with trend
//! and seasonality decomposition capabilities.

use super::blocks::{
    BaseModel, ModelError, TimeSeriesData, BasisFunction, StackType, 
    MLPBlock, ResidualBlock, DenseLayer, tensor_ops
};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use ruv_fann::ActivationFunction;
use std::collections::HashMap;

/// NBEATS model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBEATSConfig<T: Float> {
    /// Forecast horizon
    pub horizon: usize,
    /// Input window size (lookback period)
    pub input_size: usize,
    /// Stack types and their configuration
    pub stacks: Vec<StackConfig<T>>,
    /// Shared weights across blocks in the same stack
    pub shared_weights: bool,
    /// Global activation function
    pub activation: ActivationFunction,
    /// Learning rate for training
    pub learning_rate: T,
    /// Number of training epochs
    pub max_epochs: usize,
    /// Early stopping patience
    pub patience: Option<usize>,
    /// Loss function type
    pub loss_function: LossFunction,
}

/// Configuration for individual stacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackConfig<T: Float> {
    /// Type of stack (Generic, Trend, or Seasonality)
    pub stack_type: StackType,
    /// Number of blocks in the stack
    pub n_blocks: usize,
    /// Layer sizes for each block's MLP
    pub layer_widths: Vec<usize>,
    /// Basis function configuration
    pub basis_function: BasisFunction<T>,
    /// Theta size for basis expansion
    pub theta_size: usize,
}

/// Loss function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossFunction {
    MAE,  // Mean Absolute Error
    MSE,  // Mean Squared Error
    MAPE, // Mean Absolute Percentage Error
}

impl<T: Float> Default for NBEATSConfig<T> {
    fn default() -> Self {
        Self {
            horizon: 1,
            input_size: 2,
            stacks: vec![
                StackConfig {
                    stack_type: StackType::Trend,
                    n_blocks: 3,
                    layer_widths: vec![512, 512, 512, 512],
                    basis_function: BasisFunction::Polynomial { degree: 3 },
                    theta_size: 4,
                },
                StackConfig {
                    stack_type: StackType::Seasonality,
                    n_blocks: 3,
                    layer_widths: vec![512, 512, 512, 512],
                    basis_function: BasisFunction::Fourier { harmonics: 8 },
                    theta_size: 17, // 1 + 2*8
                },
            ],
            shared_weights: true,
            activation: ActivationFunction::ReLU,
            learning_rate: T::from(0.001).unwrap(),
            max_epochs: 100,
            patience: Some(10),
            loss_function: LossFunction::MAE,
        }
    }
}

/// Individual NBEATS block
#[derive(Debug, Clone)]
pub struct NBEATSBlock<T: Float> {
    /// MLP for feature extraction
    pub mlp: MLPBlock<T>,
    /// Linear layer for backcast coefficients
    pub backcast_linear: DenseLayer<T>,
    /// Linear layer for forecast coefficients  
    pub forecast_linear: DenseLayer<T>,
    /// Basis function for interpretability
    pub basis_function: BasisFunction<T>,
    /// Block type
    pub block_type: StackType,
    /// Input and output sizes
    pub input_size: usize,
    pub output_size: usize,
    /// Theta size for basis expansion
    pub theta_size: usize,
}

impl<T: Float> NBEATSBlock<T> {
    /// Create a new NBEATS block
    pub fn new(
        input_size: usize,
        output_size: usize,
        layer_widths: Vec<usize>,
        activation: ActivationFunction,
        basis_function: BasisFunction<T>,
        block_type: StackType,
        theta_size: usize,
    ) -> Result<Self, ModelError> {
        if layer_widths.is_empty() {
            return Err(ModelError::ConfigurationError("Layer widths cannot be empty".to_string()));
        }
        
        // Create MLP backbone
        let mlp = MLPBlock::new(input_size, layer_widths.clone(), activation)?;
        let mlp_output_size = layer_widths.last().unwrap();
        
        // Create linear layers for backcast and forecast coefficients
        let backcast_linear = DenseLayer::new(*mlp_output_size, theta_size, ActivationFunction::Linear);
        let forecast_linear = DenseLayer::new(*mlp_output_size, theta_size, ActivationFunction::Linear);
        
        Ok(Self {
            mlp,
            backcast_linear,
            forecast_linear,
            basis_function,
            block_type,
            input_size,
            output_size,
            theta_size,
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
        
        // Forward through MLP backbone
        let mlp_output = self.mlp.forward(input)?;
        
        // Generate backcast and forecast coefficients
        let backcast_coeffs = self.backcast_linear.forward(&mlp_output)?;
        let forecast_coeffs = self.forecast_linear.forward(&mlp_output)?;
        
        // Apply basis functions
        let backcast = self.apply_basis(&backcast_coeffs, self.input_size)?;
        let forecast = self.apply_basis(&forecast_coeffs, self.output_size)?;
        
        Ok((backcast, forecast))
    }
    
    /// Apply basis function to coefficients
    fn apply_basis(&self, coefficients: &[T], output_size: usize) -> Result<Vec<T>, ModelError> {
        let basis_matrix = self.basis_function.generate_basis(coefficients.len(), output_size);
        
        // Matrix-vector multiplication: basis_matrix^T * coefficients
        let mut result = vec![T::zero(); output_size];
        
        for (i, row) in basis_matrix.iter().enumerate() {
            for (j, &basis_val) in row.iter().enumerate() {
                if i < coefficients.len() && j < result.len() {
                    result[j] = result[j] + coefficients[i] * basis_val;
                }
            }
        }
        
        Ok(result)
    }
}

/// NBEATS stack containing multiple blocks
#[derive(Debug, Clone)]
pub struct NBEATSStack<T: Float> {
    /// Blocks in the stack
    pub blocks: Vec<NBEATSBlock<T>>,
    /// Stack configuration
    pub config: StackConfig<T>,
    /// Whether weights are shared across blocks
    pub shared_weights: bool,
}

impl<T: Float> NBEATSStack<T> {
    /// Create a new NBEATS stack
    pub fn new(
        input_size: usize,
        output_size: usize,
        config: StackConfig<T>,
        activation: ActivationFunction,
        shared_weights: bool,
    ) -> Result<Self, ModelError> {
        let mut blocks = Vec::with_capacity(config.n_blocks);
        
        for _ in 0..config.n_blocks {
            let block = NBEATSBlock::new(
                input_size,
                output_size,
                config.layer_widths.clone(),
                activation,
                config.basis_function.clone(),
                config.stack_type,
                config.theta_size,
            )?;
            blocks.push(block);
        }
        
        Ok(Self {
            blocks,
            config,
            shared_weights,
        })
    }
    
    /// Forward pass through the stack with residual connections
    pub fn forward(&mut self, input: &[T]) -> Result<(Vec<T>, Vec<T>), ModelError> {
        let mut current_input = input.to_vec();
        let mut stack_forecast = vec![T::zero(); self.blocks[0].output_size];
        
        // Process each block in the stack
        for block in &mut self.blocks {
            let (backcast, forecast) = block.forward(&current_input)?;
            
            // Update residual input for next block
            current_input = tensor_ops::subtract(&current_input, &backcast)?;
            
            // Accumulate forecasts
            stack_forecast = tensor_ops::add(&stack_forecast, &forecast)?;
        }
        
        // Return final residual and accumulated forecast
        Ok((current_input, stack_forecast))
    }
}

/// Main NBEATS model
#[derive(Debug, Clone)]
pub struct NBEATS<T: Float> {
    /// Model configuration
    pub config: NBEATSConfig<T>,
    /// Stack collection
    pub stacks: Vec<NBEATSStack<T>>,
    /// Training state
    pub is_fitted: bool,
    /// Training history
    pub training_history: Vec<T>,
}

impl<T: Float> NBEATS<T> {
    /// Create a new NBEATS model
    pub fn new(config: NBEATSConfig<T>) -> Result<Self, ModelError> {
        config.validate()?;
        
        let mut stacks = Vec::with_capacity(config.stacks.len());
        
        for stack_config in &config.stacks {
            let stack = NBEATSStack::new(
                config.input_size,
                config.horizon,
                stack_config.clone(),
                config.activation,
                config.shared_weights,
            )?;
            stacks.push(stack);
        }
        
        Ok(Self {
            config,
            stacks,
            is_fitted: false,
            training_history: Vec::new(),
        })
    }
    
    /// Create interpretable NBEATS model
    pub fn interpretable(horizon: usize, input_size: usize) -> Result<Self, ModelError> {
        let config = NBEATSConfig {
            horizon,
            input_size,
            stacks: vec![
                StackConfig {
                    stack_type: StackType::Trend,
                    n_blocks: 3,
                    layer_widths: vec![512, 512, 512, 512],
                    basis_function: BasisFunction::Polynomial { degree: 3 },
                    theta_size: 4,
                },
                StackConfig {
                    stack_type: StackType::Seasonality,
                    n_blocks: 3,
                    layer_widths: vec![512, 512, 512, 512],
                    basis_function: BasisFunction::Fourier { harmonics: horizon / 2 },
                    theta_size: 1 + 2 * (horizon / 2), // DC + cos/sin pairs
                },
            ],
            ..Default::default()
        };
        
        Self::new(config)
    }
    
    /// Create generic NBEATS model
    pub fn generic(horizon: usize, input_size: usize, n_stacks: usize) -> Result<Self, ModelError> {
        let mut stacks = Vec::with_capacity(n_stacks);
        
        for _ in 0..n_stacks {
            stacks.push(StackConfig {
                stack_type: StackType::Generic,
                n_blocks: 3,
                layer_widths: vec![512, 512, 512, 512],
                basis_function: BasisFunction::Generic,
                theta_size: input_size.max(horizon),
            });
        }
        
        let config = NBEATSConfig {
            horizon,
            input_size,
            stacks,
            ..Default::default()
        };
        
        Self::new(config)
    }
    
    /// Forward pass through all stacks
    pub fn forward(&mut self, input: &[T]) -> Result<Vec<T>, ModelError> {
        if input.len() != self.config.input_size {
            return Err(ModelError::DimensionMismatch {
                expected: self.config.input_size,
                actual: input.len(),
            });
        }
        
        let mut current_residual = input.to_vec();
        let mut total_forecast = vec![T::zero(); self.config.horizon];
        
        // Forward through each stack
        for stack in &mut self.stacks {
            let (residual, forecast) = stack.forward(&current_residual)?;
            
            // Update residual for next stack
            current_residual = residual;
            
            // Accumulate forecasts
            total_forecast = tensor_ops::add(&total_forecast, &forecast)?;
        }
        
        Ok(total_forecast)
    }
    
    /// Decompose forecast into trend and seasonality components (for interpretable model)
    pub fn decompose_forecast(&mut self, input: &[T]) -> Result<DecomposedForecast<T>, ModelError> {
        if !self.is_interpretable() {
            return Err(ModelError::ConfigurationError(
                "Decomposition only available for interpretable models".to_string()
            ));
        }
        
        let mut trend_component = vec![T::zero(); self.config.horizon];
        let mut seasonal_component = vec![T::zero(); self.config.horizon];
        let mut current_residual = input.to_vec();
        
        for stack in &mut self.stacks {
            let (residual, forecast) = stack.forward(&current_residual)?;
            current_residual = residual;
            
            match stack.config.stack_type {
                StackType::Trend => {
                    trend_component = tensor_ops::add(&trend_component, &forecast)?;
                },
                StackType::Seasonality => {
                    seasonal_component = tensor_ops::add(&seasonal_component, &forecast)?;
                },
                StackType::Generic => {
                    // Add to trend component as fallback
                    trend_component = tensor_ops::add(&trend_component, &forecast)?;
                },
            }
        }
        
        let total_forecast = tensor_ops::add(&trend_component, &seasonal_component)?;
        
        Ok(DecomposedForecast {
            total: total_forecast,
            trend: trend_component,
            seasonal: seasonal_component,
        })
    }
    
    /// Check if model is interpretable
    pub fn is_interpretable(&self) -> bool {
        self.config.stacks.iter().any(|s| matches!(s.stack_type, StackType::Trend | StackType::Seasonality))
    }
    
    /// Get model complexity (total parameters count)
    pub fn complexity(&self) -> usize {
        self.stacks.iter()
            .map(|stack| stack.blocks.len() * self.estimate_block_parameters())
            .sum()
    }
    
    /// Estimate parameters in a single block
    fn estimate_block_parameters(&self) -> usize {
        // Rough estimation based on MLP layers and linear projections
        let mut params = 0;
        
        // MLP parameters (weights + biases)
        let layer_widths = &self.config.stacks[0].layer_widths;
        let mut prev_size = self.config.input_size;
        
        for &size in layer_widths {
            params += prev_size * size + size; // weights + biases
            prev_size = size;
        }
        
        // Backcast and forecast linear layers
        let theta_size = self.config.stacks[0].theta_size;
        params += prev_size * theta_size * 2; // backcast + forecast
        
        params
    }
    
    /// Simple training procedure (placeholder for full implementation)
    fn train_step(&mut self, input: &[T], target: &[T]) -> Result<T, ModelError> {
        let prediction = self.forward(input)?;
        let loss = self.calculate_loss(&prediction, target)?;
        
        // TODO: Implement backpropagation using ruv-FANN training algorithms
        // This would require extending ruv-FANN with custom training procedures
        
        Ok(loss)
    }
    
    /// Calculate loss based on configured loss function
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
        }
    }
}

/// Decomposed forecast components
#[derive(Debug, Clone)]
pub struct DecomposedForecast<T: Float> {
    pub total: Vec<T>,
    pub trend: Vec<T>,
    pub seasonal: Vec<T>,
}

impl<T: Float> BaseModel<T> for NBEATS<T> {
    type Config = NBEATSConfig<T>;
    
    fn name(&self) -> &'static str {
        "NBEATS"
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
        
        // Create training windows
        let mut training_losses = Vec::new();
        let n_windows = data.series.len() - self.config.input_size - self.config.horizon + 1;
        
        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = T::zero();
            
            for i in 0..n_windows {
                let input_window = &data.series[i..i + self.config.input_size];
                let target_window = &data.series[i + self.config.input_size..i + self.config.input_size + self.config.horizon];
                
                let loss = self.train_step(input_window, target_window)?;
                epoch_loss = epoch_loss + loss;
            }
            
            epoch_loss = epoch_loss / T::from(n_windows).unwrap();
            training_losses.push(epoch_loss);
            
            // Early stopping check
            if let Some(patience) = self.config.patience {
                if training_losses.len() > patience {
                    let recent_losses = &training_losses[training_losses.len() - patience..];
                    let improving = recent_losses.windows(2).any(|w| w[1] < w[0]);
                    
                    if !improving {
                        break;
                    }
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
        
        // Use the last input_size points for prediction
        let input_window = &data.series[data.series.len() - self.config.input_size..];
        
        // Create mutable copy for forward pass
        let mut model_copy = self.clone();
        model_copy.forward(input_window)
    }
    
    fn predict_quantiles(&self, data: &TimeSeriesData<T>, _quantiles: &[T]) -> Result<Vec<Vec<T>>, ModelError> {
        // For deterministic NBEATS, return point predictions for all quantiles
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

impl<T: Float> NBEATSConfig<T> {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.horizon == 0 {
            return Err(ModelError::ConfigurationError("Horizon must be positive".to_string()));
        }
        
        if self.input_size == 0 {
            return Err(ModelError::ConfigurationError("Input size must be positive".to_string()));
        }
        
        if self.stacks.is_empty() {
            return Err(ModelError::ConfigurationError("At least one stack is required".to_string()));
        }
        
        for (i, stack) in self.stacks.iter().enumerate() {
            if stack.n_blocks == 0 {
                return Err(ModelError::ConfigurationError(
                    format!("Stack {} must have at least one block", i)
                ));
            }
            
            if stack.layer_widths.is_empty() {
                return Err(ModelError::ConfigurationError(
                    format!("Stack {} must have at least one layer", i)
                ));
            }
            
            if stack.theta_size == 0 {
                return Err(ModelError::ConfigurationError(
                    format!("Stack {} theta size must be positive", i)
                ));
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nbeats_creation() {
        let config = NBEATSConfig::<f32>::default();
        let model = NBEATS::new(config);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.name(), "NBEATS");
        assert!(!model.is_fitted());
    }
    
    #[test]
    fn test_interpretable_nbeats() {
        let model = NBEATS::<f32>::interpretable(12, 24);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert!(model.is_interpretable());
        assert_eq!(model.horizon(), 12);
        assert_eq!(model.input_size(), 24);
    }
    
    #[test]
    fn test_generic_nbeats() {
        let model = NBEATS::<f32>::generic(6, 12, 3);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.horizon(), 6);
        assert_eq!(model.input_size(), 12);
        assert_eq!(model.stacks.len(), 3);
    }
    
    #[test]
    fn test_nbeats_block_forward() {
        let mut block = NBEATSBlock::<f32>::new(
            10,
            5,
            vec![64, 32],
            ActivationFunction::ReLU,
            BasisFunction::Generic,
            StackType::Generic,
            10,
        ).unwrap();
        
        let input = vec![1.0; 10];
        let result = block.forward(&input);
        assert!(result.is_ok());
        
        let (backcast, forecast) = result.unwrap();
        assert_eq!(backcast.len(), 10);
        assert_eq!(forecast.len(), 5);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = NBEATSConfig::<f32>::default();
        assert!(config.validate().is_ok());
        
        config.horizon = 0;
        assert!(config.validate().is_err());
        
        config.horizon = 1;
        config.stacks.clear();
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_basis_function_application() {
        let block = NBEATSBlock::<f32>::new(
            5,
            3,
            vec![16],
            ActivationFunction::ReLU,
            BasisFunction::Polynomial { degree: 2 },
            StackType::Trend,
            3,
        ).unwrap();
        
        let coeffs = vec![1.0, 0.5, 0.25];
        let result = block.apply_basis(&coeffs, 4);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 4);
    }
}