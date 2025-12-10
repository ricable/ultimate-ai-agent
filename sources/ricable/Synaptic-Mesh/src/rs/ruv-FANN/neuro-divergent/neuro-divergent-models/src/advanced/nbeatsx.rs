//! NBEATSx (Extended NBEATS with Exogenous Variables) Implementation
//! 
//! This module extends the NBEATS architecture to handle exogenous variables,
//! supporting both historical and future known variables in the forecasting process.

use super::blocks::{
    BaseModel, ModelError, TimeSeriesData, BasisFunction, StackType, 
    MLPBlock, ResidualBlock, DenseLayer, tensor_ops
};
use super::nbeats::{NBEATSConfig, StackConfig, LossFunction, DecomposedForecast};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use ruv_fann::ActivationFunction;
use std::collections::HashMap;

/// NBEATSx model configuration extending NBEATS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBEATSxConfig<T: Float> {
    /// Base NBEATS configuration
    pub base_config: NBEATSConfig<T>,
    /// Number of historical exogenous features
    pub hist_exog_size: usize,
    /// Number of future exogenous features
    pub futr_exog_size: usize,
    /// Static features size
    pub stat_exog_size: usize,
    /// Exogenous variables integration strategy
    pub exog_integration: ExogIntegration,
    /// Separate processing for different exogenous types
    pub separate_exog_processing: bool,
    /// Exogenous variables normalization
    pub normalize_exog: bool,
}

/// Exogenous variables integration strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExogIntegration {
    /// Concatenate with time series input
    Concatenation,
    /// Add separate MLP processing paths
    SeparateMLPs,
    /// Use attention-like mechanisms
    AttentionBased,
    /// Hierarchical integration
    Hierarchical,
}

impl<T: Float> Default for NBEATSxConfig<T> {
    fn default() -> Self {
        Self {
            base_config: NBEATSConfig::default(),
            hist_exog_size: 0,
            futr_exog_size: 0,
            stat_exog_size: 0,
            exog_integration: ExogIntegration::Concatenation,
            separate_exog_processing: false,
            normalize_exog: true,
        }
    }
}

/// Extended NBEATS block with exogenous variables support
#[derive(Debug, Clone)]
pub struct NBEATSxBlock<T: Float> {
    /// Main time series MLP
    pub main_mlp: MLPBlock<T>,
    /// Historical exogenous variables MLP
    pub hist_exog_mlp: Option<MLPBlock<T>>,
    /// Future exogenous variables MLP  
    pub futr_exog_mlp: Option<MLPBlock<T>>,
    /// Static exogenous variables MLP
    pub stat_exog_mlp: Option<MLPBlock<T>>,
    /// Fusion layer for combining different inputs
    pub fusion_layer: DenseLayer<T>,
    /// Linear layer for backcast coefficients
    pub backcast_linear: DenseLayer<T>,
    /// Linear layer for forecast coefficients
    pub forecast_linear: DenseLayer<T>,
    /// Basis function for interpretability
    pub basis_function: BasisFunction<T>,
    /// Block configuration
    pub block_type: StackType,
    pub input_size: usize,
    pub output_size: usize,
    pub theta_size: usize,
    /// Exogenous variables configuration
    pub exog_config: ExogConfiguration,
}

/// Exogenous variables configuration
#[derive(Debug, Clone)]
pub struct ExogConfiguration {
    pub hist_exog_size: usize,
    pub futr_exog_size: usize,
    pub stat_exog_size: usize,
    pub integration: ExogIntegration,
}

impl<T: Float> NBEATSxBlock<T> {
    /// Create a new NBEATSx block
    pub fn new(
        input_size: usize,
        output_size: usize,
        layer_widths: Vec<usize>,
        activation: ActivationFunction,
        basis_function: BasisFunction<T>,
        block_type: StackType,
        theta_size: usize,
        exog_config: ExogConfiguration,
    ) -> Result<Self, ModelError> {
        if layer_widths.is_empty() {
            return Err(ModelError::ConfigurationError("Layer widths cannot be empty".to_string()));
        }
        
        // Calculate effective input size based on integration strategy
        let effective_input_size = match exog_config.integration {
            ExogIntegration::Concatenation => {
                input_size + exog_config.hist_exog_size + exog_config.futr_exog_size + exog_config.stat_exog_size
            },
            _ => input_size,
        };
        
        // Create main MLP
        let main_mlp = MLPBlock::new(effective_input_size, layer_widths.clone(), activation)?;
        let mlp_output_size = layer_widths.last().unwrap();
        
        // Create exogenous MLPs based on integration strategy
        let hist_exog_mlp = if exog_config.hist_exog_size > 0 && matches!(exog_config.integration, ExogIntegration::SeparateMLPs | ExogIntegration::AttentionBased | ExogIntegration::Hierarchical) {
            Some(MLPBlock::new(exog_config.hist_exog_size, vec![layer_widths[0] / 2], activation)?)
        } else {
            None
        };
        
        let futr_exog_mlp = if exog_config.futr_exog_size > 0 && matches!(exog_config.integration, ExogIntegration::SeparateMLPs | ExogIntegration::AttentionBased | ExogIntegration::Hierarchical) {
            Some(MLPBlock::new(exog_config.futr_exog_size, vec![layer_widths[0] / 2], activation)?)
        } else {
            None
        };
        
        let stat_exog_mlp = if exog_config.stat_exog_size > 0 && matches!(exog_config.integration, ExogIntegration::SeparateMLPs | ExogIntegration::AttentionBased | ExogIntegration::Hierarchical) {
            Some(MLPBlock::new(exog_config.stat_exog_size, vec![layer_widths[0] / 4], activation)?)
        } else {
            None
        };
        
        // Calculate fusion layer input size
        let fusion_input_size = match exog_config.integration {
            ExogIntegration::Concatenation => *mlp_output_size,
            ExogIntegration::SeparateMLPs | ExogIntegration::AttentionBased | ExogIntegration::Hierarchical => {
                let mut size = *mlp_output_size;
                if hist_exog_mlp.is_some() { size += layer_widths[0] / 2; }
                if futr_exog_mlp.is_some() { size += layer_widths[0] / 2; }
                if stat_exog_mlp.is_some() { size += layer_widths[0] / 4; }
                size
            },
        };
        
        let fusion_layer = DenseLayer::new(fusion_input_size, *mlp_output_size, activation);
        
        // Create linear layers for backcast and forecast coefficients
        let backcast_linear = DenseLayer::new(*mlp_output_size, theta_size, ActivationFunction::Linear);
        let forecast_linear = DenseLayer::new(*mlp_output_size, theta_size, ActivationFunction::Linear);
        
        Ok(Self {
            main_mlp,
            hist_exog_mlp,
            futr_exog_mlp,
            stat_exog_mlp,
            fusion_layer,
            backcast_linear,
            forecast_linear,
            basis_function,
            block_type,
            input_size,
            output_size,
            theta_size,
            exog_config,
        })
    }
    
    /// Forward pass through the block with exogenous variables
    pub fn forward(
        &mut self, 
        input: &[T],
        hist_exog: Option<&[T]>,
        futr_exog: Option<&[T]>,
        stat_exog: Option<&[T]>,
    ) -> Result<(Vec<T>, Vec<T>), ModelError> {
        if input.len() != self.input_size {
            return Err(ModelError::DimensionMismatch {
                expected: self.input_size,
                actual: input.len(),
            });
        }
        
        // Process inputs based on integration strategy
        let fusion_input = match self.exog_config.integration {
            ExogIntegration::Concatenation => {
                self.process_concatenation(input, hist_exog, futr_exog, stat_exog)?
            },
            ExogIntegration::SeparateMLPs => {
                self.process_separate_mlps(input, hist_exog, futr_exog, stat_exog)?
            },
            ExogIntegration::AttentionBased => {
                self.process_attention_based(input, hist_exog, futr_exog, stat_exog)?
            },
            ExogIntegration::Hierarchical => {
                self.process_hierarchical(input, hist_exog, futr_exog, stat_exog)?
            },
        };
        
        // Generate backcast and forecast coefficients
        let backcast_coeffs = self.backcast_linear.forward(&fusion_input)?;
        let forecast_coeffs = self.forecast_linear.forward(&fusion_input)?;
        
        // Apply basis functions
        let backcast = self.apply_basis(&backcast_coeffs, self.input_size)?;
        let forecast = self.apply_basis(&forecast_coeffs, self.output_size)?;
        
        Ok((backcast, forecast))
    }
    
    /// Process inputs using concatenation strategy
    fn process_concatenation(
        &mut self,
        input: &[T],
        hist_exog: Option<&[T]>,
        futr_exog: Option<&[T]>,
        stat_exog: Option<&[T]>,
    ) -> Result<Vec<T>, ModelError> {
        let mut combined_input = input.to_vec();
        
        if let Some(hist) = hist_exog {
            if hist.len() != self.exog_config.hist_exog_size {
                return Err(ModelError::DimensionMismatch {
                    expected: self.exog_config.hist_exog_size,
                    actual: hist.len(),
                });
            }
            combined_input.extend_from_slice(hist);
        } else if self.exog_config.hist_exog_size > 0 {
            combined_input.extend(vec![T::zero(); self.exog_config.hist_exog_size]);
        }
        
        if let Some(futr) = futr_exog {
            if futr.len() != self.exog_config.futr_exog_size {
                return Err(ModelError::DimensionMismatch {
                    expected: self.exog_config.futr_exog_size,
                    actual: futr.len(),
                });
            }
            combined_input.extend_from_slice(futr);
        } else if self.exog_config.futr_exog_size > 0 {
            combined_input.extend(vec![T::zero(); self.exog_config.futr_exog_size]);
        }
        
        if let Some(stat) = stat_exog {
            if stat.len() != self.exog_config.stat_exog_size {
                return Err(ModelError::DimensionMismatch {
                    expected: self.exog_config.stat_exog_size,
                    actual: stat.len(),
                });
            }
            combined_input.extend_from_slice(stat);
        } else if self.exog_config.stat_exog_size > 0 {
            combined_input.extend(vec![T::zero(); self.exog_config.stat_exog_size]);
        }
        
        self.main_mlp.forward(&combined_input)
    }
    
    /// Process inputs using separate MLPs strategy
    fn process_separate_mlps(
        &mut self,
        input: &[T],
        hist_exog: Option<&[T]>,
        futr_exog: Option<&[T]>,
        stat_exog: Option<&[T]>,
    ) -> Result<Vec<T>, ModelError> {
        let main_output = self.main_mlp.forward(input)?;
        let mut combined_outputs = main_output;
        
        if let (Some(hist), Some(ref mut hist_mlp)) = (hist_exog, &mut self.hist_exog_mlp) {
            let hist_output = hist_mlp.forward(hist)?;
            combined_outputs.extend(hist_output);
        }
        
        if let (Some(futr), Some(ref mut futr_mlp)) = (futr_exog, &mut self.futr_exog_mlp) {
            let futr_output = futr_mlp.forward(futr)?;
            combined_outputs.extend(futr_output);
        }
        
        if let (Some(stat), Some(ref mut stat_mlp)) = (stat_exog, &mut self.stat_exog_mlp) {
            let stat_output = stat_mlp.forward(stat)?;
            combined_outputs.extend(stat_output);
        }
        
        self.fusion_layer.forward(&combined_outputs)
    }
    
    /// Process inputs using attention-based strategy
    fn process_attention_based(
        &mut self,
        input: &[T],
        hist_exog: Option<&[T]>,
        futr_exog: Option<&[T]>,
        stat_exog: Option<&[T]>,
    ) -> Result<Vec<T>, ModelError> {
        // Simplified attention mechanism using weighted combination
        let main_output = self.main_mlp.forward(input)?;
        let mut weighted_outputs = main_output.clone();
        let mut weights_sum = T::one();
        
        if let (Some(hist), Some(ref mut hist_mlp)) = (hist_exog, &mut self.hist_exog_mlp) {
            let hist_output = hist_mlp.forward(hist)?;
            let attention_weight = self.compute_attention_weight(&main_output, &hist_output)?;
            
            for (i, &hist_val) in hist_output.iter().enumerate().take(weighted_outputs.len()) {
                weighted_outputs[i] = weighted_outputs[i] + attention_weight * hist_val;
            }
            weights_sum = weights_sum + attention_weight;
        }
        
        if let (Some(futr), Some(ref mut futr_mlp)) = (futr_exog, &mut self.futr_exog_mlp) {
            let futr_output = futr_mlp.forward(futr)?;
            let attention_weight = self.compute_attention_weight(&main_output, &futr_output)?;
            
            for (i, &futr_val) in futr_output.iter().enumerate().take(weighted_outputs.len()) {
                weighted_outputs[i] = weighted_outputs[i] + attention_weight * futr_val;
            }
            weights_sum = weights_sum + attention_weight;
        }
        
        // Normalize by total weights
        for val in &mut weighted_outputs {
            *val = *val / weights_sum;
        }
        
        self.fusion_layer.forward(&weighted_outputs)
    }
    
    /// Process inputs using hierarchical strategy
    fn process_hierarchical(
        &mut self,
        input: &[T],
        hist_exog: Option<&[T]>,
        futr_exog: Option<&[T]>,
        stat_exog: Option<&[T]>,
    ) -> Result<Vec<T>, ModelError> {
        // Hierarchical processing: static -> historical -> future -> main
        let mut current_input = input.to_vec();
        
        // Process static features first (global context)
        if let (Some(stat), Some(ref mut stat_mlp)) = (stat_exog, &mut self.stat_exog_mlp) {
            let stat_output = stat_mlp.forward(stat)?;
            // Broadcast static features to time series length
            for (i, val) in current_input.iter_mut().enumerate() {
                if i < stat_output.len() {
                    *val = *val + stat_output[i % stat_output.len()];
                }
            }
        }
        
        // Process historical exogenous variables
        if let (Some(hist), Some(ref mut hist_mlp)) = (hist_exog, &mut self.hist_exog_mlp) {
            let hist_output = hist_mlp.forward(hist)?;
            current_input = tensor_ops::add(&current_input, &hist_output[..current_input.len().min(hist_output.len())])?;
        }
        
        // Main processing
        let main_output = self.main_mlp.forward(&current_input)?;
        
        // Process future exogenous variables (for forecast enhancement)
        let mut final_output = main_output;
        if let (Some(futr), Some(ref mut futr_mlp)) = (futr_exog, &mut self.futr_exog_mlp) {
            let futr_output = futr_mlp.forward(futr)?;
            final_output.extend(futr_output);
        }
        
        self.fusion_layer.forward(&final_output)
    }
    
    /// Compute attention weight between two feature vectors
    fn compute_attention_weight(&self, query: &[T], key: &[T]) -> Result<T, ModelError> {
        // Simplified attention: normalized dot product
        let min_len = query.len().min(key.len());
        let dot_product = tensor_ops::dot(&query[..min_len], &key[..min_len])?;
        
        // Normalize by vector lengths
        let query_norm = query.iter().map(|&x| x.powi(2)).sum::<T>().sqrt();
        let key_norm = key.iter().map(|&x| x.powi(2)).sum::<T>().sqrt();
        
        if query_norm == T::zero() || key_norm == T::zero() {
            Ok(T::zero())
        } else {
            Ok(dot_product / (query_norm * key_norm))
        }
    }
    
    /// Apply basis function to coefficients
    fn apply_basis(&self, coefficients: &[T], output_size: usize) -> Result<Vec<T>, ModelError> {
        let basis_matrix = self.basis_function.generate_basis(coefficients.len(), output_size);
        
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

/// Extended NBEATS stack with exogenous variables support
#[derive(Debug, Clone)]
pub struct NBEATSxStack<T: Float> {
    /// Blocks in the stack
    pub blocks: Vec<NBEATSxBlock<T>>,
    /// Stack configuration
    pub config: StackConfig<T>,
    /// Exogenous configuration
    pub exog_config: ExogConfiguration,
    /// Whether weights are shared across blocks
    pub shared_weights: bool,
}

impl<T: Float> NBEATSxStack<T> {
    /// Create a new NBEATSx stack
    pub fn new(
        input_size: usize,
        output_size: usize,
        config: StackConfig<T>,
        exog_config: ExogConfiguration,
        activation: ActivationFunction,
        shared_weights: bool,
    ) -> Result<Self, ModelError> {
        let mut blocks = Vec::with_capacity(config.n_blocks);
        
        for _ in 0..config.n_blocks {
            let block = NBEATSxBlock::new(
                input_size,
                output_size,
                config.layer_widths.clone(),
                activation,
                config.basis_function.clone(),
                config.stack_type,
                config.theta_size,
                exog_config.clone(),
            )?;
            blocks.push(block);
        }
        
        Ok(Self {
            blocks,
            config,
            exog_config,
            shared_weights,
        })
    }
    
    /// Forward pass through the stack with exogenous variables
    pub fn forward(
        &mut self, 
        input: &[T],
        hist_exog: Option<&[T]>,
        futr_exog: Option<&[T]>,
        stat_exog: Option<&[T]>,
    ) -> Result<(Vec<T>, Vec<T>), ModelError> {
        let mut current_input = input.to_vec();
        let mut stack_forecast = vec![T::zero(); self.blocks[0].output_size];
        
        for block in &mut self.blocks {
            let (backcast, forecast) = block.forward(&current_input, hist_exog, futr_exog, stat_exog)?;
            
            // Update residual input for next block
            current_input = tensor_ops::subtract(&current_input, &backcast)?;
            
            // Accumulate forecasts
            stack_forecast = tensor_ops::add(&stack_forecast, &forecast)?;
        }
        
        Ok((current_input, stack_forecast))
    }
}

/// Main NBEATSx model
#[derive(Debug, Clone)]
pub struct NBEATSx<T: Float> {
    /// Model configuration
    pub config: NBEATSxConfig<T>,
    /// Stack collection
    pub stacks: Vec<NBEATSxStack<T>>,
    /// Training state
    pub is_fitted: bool,
    /// Training history
    pub training_history: Vec<T>,
}

impl<T: Float> NBEATSx<T> {
    /// Create a new NBEATSx model
    pub fn new(config: NBEATSxConfig<T>) -> Result<Self, ModelError> {
        config.validate()?;
        
        let exog_config = ExogConfiguration {
            hist_exog_size: config.hist_exog_size,
            futr_exog_size: config.futr_exog_size,
            stat_exog_size: config.stat_exog_size,
            integration: config.exog_integration,
        };
        
        let mut stacks = Vec::with_capacity(config.base_config.stacks.len());
        
        for stack_config in &config.base_config.stacks {
            let stack = NBEATSxStack::new(
                config.base_config.input_size,
                config.base_config.horizon,
                stack_config.clone(),
                exog_config.clone(),
                config.base_config.activation,
                config.base_config.shared_weights,
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
    
    /// Forward pass through all stacks with exogenous variables
    pub fn forward(
        &mut self, 
        input: &[T],
        hist_exog: Option<&[T]>,
        futr_exog: Option<&[T]>,
        stat_exog: Option<&[T]>,
    ) -> Result<Vec<T>, ModelError> {
        if input.len() != self.config.base_config.input_size {
            return Err(ModelError::DimensionMismatch {
                expected: self.config.base_config.input_size,
                actual: input.len(),
            });
        }
        
        let mut current_residual = input.to_vec();
        let mut total_forecast = vec![T::zero(); self.config.base_config.horizon];
        
        for stack in &mut self.stacks {
            let (residual, forecast) = stack.forward(&current_residual, hist_exog, futr_exog, stat_exog)?;
            
            current_residual = residual;
            total_forecast = tensor_ops::add(&total_forecast, &forecast)?;
        }
        
        Ok(total_forecast)
    }
    
    /// Create interpretable NBEATSx model
    pub fn interpretable(
        horizon: usize, 
        input_size: usize,
        hist_exog_size: usize,
        futr_exog_size: usize,
        stat_exog_size: usize,
    ) -> Result<Self, ModelError> {
        let base_config = NBEATSConfig {
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
                    theta_size: 1 + 2 * (horizon / 2),
                },
            ],
            ..Default::default()
        };
        
        let config = NBEATSxConfig {
            base_config,
            hist_exog_size,
            futr_exog_size,
            stat_exog_size,
            exog_integration: ExogIntegration::SeparateMLPs,
            separate_exog_processing: true,
            normalize_exog: true,
        };
        
        Self::new(config)
    }
}

impl<T: Float> BaseModel<T> for NBEATSx<T> {
    type Config = NBEATSxConfig<T>;
    
    fn name(&self) -> &'static str {
        "NBEATSx"
    }
    
    fn horizon(&self) -> usize {
        self.config.base_config.horizon
    }
    
    fn input_size(&self) -> usize {
        self.config.base_config.input_size
    }
    
    fn fit(&mut self, data: &TimeSeriesData<T>) -> Result<(), ModelError> {
        if data.series.len() < self.config.base_config.input_size + self.config.base_config.horizon {
            return Err(ModelError::InvalidInput(
                "Insufficient data length for training".to_string()
            ));
        }
        
        // Validate exogenous data dimensions if provided
        if let Some(ref exog) = data.exogenous {
            if !exog.is_empty() {
                let expected_exog_size = self.config.hist_exog_size + self.config.futr_exog_size;
                if exog.len() != expected_exog_size {
                    return Err(ModelError::DimensionMismatch {
                        expected: expected_exog_size,
                        actual: exog.len(),
                    });
                }
            }
        }
        
        // Training logic would be implemented here
        // This is a placeholder implementation
        self.is_fitted = true;
        
        Ok(())
    }
    
    fn predict(&self, data: &TimeSeriesData<T>) -> Result<Vec<T>, ModelError> {
        if !self.is_fitted {
            return Err(ModelError::ConfigurationError(
                "Model must be fitted before prediction".to_string()
            ));
        }
        
        if data.series.len() < self.config.base_config.input_size {
            return Err(ModelError::InvalidInput(
                "Insufficient data length for prediction".to_string()
            ));
        }
        
        let input_window = &data.series[data.series.len() - self.config.base_config.input_size..];
        
        // Extract exogenous variables if available
        let hist_exog = data.exogenous.as_ref()
            .and_then(|exog| {
                if self.config.hist_exog_size > 0 && !exog.is_empty() {
                    Some(&exog[0][..self.config.hist_exog_size])
                } else {
                    None
                }
            });
            
        let futr_exog = data.exogenous.as_ref()
            .and_then(|exog| {
                if self.config.futr_exog_size > 0 && exog.len() > 1 {
                    Some(&exog[1][..self.config.futr_exog_size])
                } else {
                    None
                }
            });
            
        let stat_exog = data.static_features.as_ref()
            .and_then(|stat| {
                if self.config.stat_exog_size > 0 {
                    Some(stat.as_slice())
                } else {
                    None
                }
            });
        
        let mut model_copy = self.clone();
        model_copy.forward(input_window, hist_exog, futr_exog, stat_exog)
    }
    
    fn predict_quantiles(&self, data: &TimeSeriesData<T>, _quantiles: &[T]) -> Result<Vec<Vec<T>>, ModelError> {
        let point_prediction = self.predict(data)?;
        Ok(vec![point_prediction; _quantiles.len()])
    }
    
    fn parameters_count(&self) -> usize {
        // Estimate parameters including exogenous processing layers
        let base_params = self.stacks.iter()
            .map(|stack| stack.blocks.len() * self.estimate_block_parameters())
            .sum::<usize>();
            
        // Add exogenous processing parameters
        let exog_params = self.config.hist_exog_size * 64 + 
                         self.config.futr_exog_size * 64 + 
                         self.config.stat_exog_size * 32;
                         
        base_params + exog_params
    }
    
    fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

impl<T: Float> NBEATSx<T> {
    fn estimate_block_parameters(&self) -> usize {
        // Enhanced parameter estimation including exogenous processing
        let layer_widths = &self.config.base_config.stacks[0].layer_widths;
        let mut params = 0;
        
        // Main MLP parameters
        let effective_input_size = match self.config.exog_integration {
            ExogIntegration::Concatenation => {
                self.config.base_config.input_size + 
                self.config.hist_exog_size + 
                self.config.futr_exog_size + 
                self.config.stat_exog_size
            },
            _ => self.config.base_config.input_size,
        };
        
        let mut prev_size = effective_input_size;
        for &size in layer_widths {
            params += prev_size * size + size;
            prev_size = size;
        }
        
        // Backcast and forecast linear layers
        let theta_size = self.config.base_config.stacks[0].theta_size;
        params += prev_size * theta_size * 2;
        
        params
    }
}

impl<T: Float> NBEATSxConfig<T> {
    /// Validate NBEATSx configuration
    pub fn validate(&self) -> Result<(), ModelError> {
        self.base_config.validate()?;
        
        if self.hist_exog_size > 0 && self.futr_exog_size > 0 {
            // Ensure compatibility between historical and future exogenous sizes
            if self.hist_exog_size > self.base_config.input_size * 2 {
                return Err(ModelError::ConfigurationError(
                    "Historical exogenous size too large relative to input size".to_string()
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
    fn test_nbeatsx_creation() {
        let config = NBEATSxConfig::<f32>::default();
        let model = NBEATSx::new(config);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.name(), "NBEATSx");
        assert!(!model.is_fitted());
    }
    
    #[test]
    fn test_interpretable_nbeatsx() {
        let model = NBEATSx::<f32>::interpretable(12, 24, 5, 3, 2);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.horizon(), 12);
        assert_eq!(model.input_size(), 24);
        assert_eq!(model.config.hist_exog_size, 5);
        assert_eq!(model.config.futr_exog_size, 3);
        assert_eq!(model.config.stat_exog_size, 2);
    }
    
    #[test]
    fn test_exog_integration_strategies() {
        for integration in [
            ExogIntegration::Concatenation,
            ExogIntegration::SeparateMLPs,
            ExogIntegration::AttentionBased,
            ExogIntegration::Hierarchical,
        ] {
            let mut config = NBEATSxConfig::<f32>::default();
            config.exog_integration = integration;
            config.hist_exog_size = 3;
            config.futr_exog_size = 2;
            
            let model = NBEATSx::new(config);
            assert!(model.is_ok());
        }
    }
    
    #[test]
    fn test_nbeatsx_block_forward() {
        let exog_config = ExogConfiguration {
            hist_exog_size: 3,
            futr_exog_size: 2,
            stat_exog_size: 1,
            integration: ExogIntegration::Concatenation,
        };
        
        let mut block = NBEATSxBlock::<f32>::new(
            10,
            5,
            vec![64, 32],
            ActivationFunction::ReLU,
            BasisFunction::Generic,
            StackType::Generic,
            10,
            exog_config,
        ).unwrap();
        
        let input = vec![1.0; 10];
        let hist_exog = vec![0.5; 3];
        let futr_exog = vec![0.3; 2];
        let stat_exog = vec![0.1; 1];
        
        let result = block.forward(&input, Some(&hist_exog), Some(&futr_exog), Some(&stat_exog));
        assert!(result.is_ok());
        
        let (backcast, forecast) = result.unwrap();
        assert_eq!(backcast.len(), 10);
        assert_eq!(forecast.len(), 5);
    }
}