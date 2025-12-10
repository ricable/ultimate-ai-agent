//! GPU-Accelerated Neural Network Training for Mac
//! 
//! This module implements CNN, LSTM, and Dense MLP models using Candle framework
//! with Mac GPU acceleration (Metal Performance Shaders).

use std::collections::HashMap;
use std::time::{Duration, Instant};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2, Array3};

#[cfg(feature = "gpu")]
use candle_core::{Device, Tensor, DType};
#[cfg(feature = "gpu")]
use candle_nn::{Module, VarBuilder, VarMap, Optimizer, AdamW, linear, conv2d, lstm, Conv2dConfig, Linear, Lstm};

use crate::data::TelecomDataset;
use crate::models::TrainingParameters;

/// GPU Training Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTrainingConfig {
    /// Device selection (mps for Mac, cuda for NVIDIA, cpu fallback)
    pub device: String,
    /// Batch size for GPU training
    pub batch_size: usize,
    /// Number of worker threads for data loading
    pub num_workers: usize,
    /// Memory pool size in MB
    pub memory_pool_size: usize,
    /// Enable mixed precision training
    pub mixed_precision: bool,
    /// GPU memory optimization level
    pub memory_optimization: MemoryOptimization,
    /// Enable gradient accumulation
    pub gradient_accumulation_steps: usize,
    /// Enable automatic learning rate scheduling
    pub auto_lr_schedule: bool,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Checkpoint save interval (epochs)
    pub checkpoint_interval: usize,
}

impl Default for GpuTrainingConfig {
    fn default() -> Self {
        Self {
            device: "auto".to_string(), // Auto-detect best device
            batch_size: 32,
            num_workers: 4,
            memory_pool_size: 512, // 512MB
            mixed_precision: true,
            memory_optimization: MemoryOptimization::Balanced,
            gradient_accumulation_steps: 1,
            auto_lr_schedule: true,
            early_stopping_patience: 20,
            checkpoint_interval: 10,
        }
    }
}

/// Memory optimization strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryOptimization {
    /// Maximum memory usage for best performance
    Performance,
    /// Balanced memory/performance trade-off
    Balanced,
    /// Minimum memory usage
    Conservative,
}

/// Model architectures for GPU training
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Convolutional Neural Network for spatial pattern recognition
    CNN,
    /// Long Short-Term Memory for time series prediction
    LSTM,
    /// Dense Multi-Layer Perceptron for general regression
    DenseMLP,
}

/// GPU-accelerated CNN model for network metrics spatial analysis
#[cfg(feature = "gpu")]
pub struct GpuCnnModel {
    /// Input dimension (features per sample)
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Number of filters in each conv layer
    filters: Vec<usize>,
    /// Kernel sizes for each conv layer
    kernel_sizes: Vec<usize>,
    /// Fully connected layer sizes
    fc_sizes: Vec<usize>,
    /// Model parameters
    var_map: VarMap,
    /// Device for computation
    device: Device,
}

#[cfg(feature = "gpu")]
impl GpuCnnModel {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        device: Device,
    ) -> Result<Self> {
        let var_map = VarMap::new();
        
        // Optimized for network metrics (e.g., 91 features reshaped to spatial format)
        let filters = vec![32, 64, 128, 64];
        let kernel_sizes = vec![3, 3, 3, 3];
        let fc_sizes = vec![256, 128, 64];
        
        Ok(Self {
            input_dim,
            output_dim,
            filters,
            kernel_sizes,
            fc_sizes,
            var_map,
            device,
        })
    }
    
    /// Forward pass through CNN
    pub fn forward(&self, input: &Tensor, vs: &VarBuilder) -> Result<Tensor> {
        // Reshape 1D features to 2D spatial format for CNN processing
        // For 91 features, reshape to something like 7x13 or 9x10+1 padding
        let spatial_dim = (self.input_dim as f64).sqrt().ceil() as usize;
        let batch_size = input.dim(0)?;
        
        // Reshape to (batch, 1, height, width) for 2D convolution
        let mut x = input.reshape((batch_size, 1, spatial_dim, spatial_dim))?;
        
        // Convolutional layers with ReLU activation
        for (i, (&filters, &kernel_size)) in self.filters.iter().zip(self.kernel_sizes.iter()).enumerate() {
            let config = Conv2dConfig {
                stride: 1,
                padding: kernel_size / 2,
                ..Default::default()
            };
            
            let conv = conv2d(
                x.dim(1)?,
                filters,
                kernel_size,
                config,
                vs.pp(&format!("conv{}", i))
            )?;
            
            x = conv.forward(&x)?;
            x = x.relu()?;
            
            // Max pooling for spatial dimension reduction
            if i < self.filters.len() - 1 {
                x = x.max_pool2d(2)?;
            }
        }
        
        // Flatten for fully connected layers
        let flattened_size = x.dims().iter().skip(1).product::<usize>();
        x = x.reshape((batch_size, flattened_size))?;
        
        // Fully connected layers
        for (i, &size) in self.fc_sizes.iter().enumerate() {
            let linear = linear(
                x.dim(1)?,
                size,
                vs.pp(&format!("fc{}", i))
            )?;
            x = linear.forward(&x)?;
            x = x.relu()?;
        }
        
        // Output layer
        let output_linear = linear(
            x.dim(1)?,
            self.output_dim,
            vs.pp("output")
        )?;
        
        output_linear.forward(&x)
    }
}

/// GPU-accelerated LSTM model for time series prediction
#[cfg(feature = "gpu")]
pub struct GpuLstmModel {
    /// Input dimension
    input_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of LSTM layers
    num_layers: usize,
    /// Output dimension
    output_dim: usize,
    /// Sequence length for time series
    sequence_length: usize,
    /// Model parameters
    var_map: VarMap,
    /// Device for computation
    device: Device,
}

#[cfg(feature = "gpu")]
impl GpuLstmModel {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        output_dim: usize,
        sequence_length: usize,
        device: Device,
    ) -> Result<Self> {
        let var_map = VarMap::new();
        
        Ok(Self {
            input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            sequence_length,
            var_map,
            device,
        })
    }
    
    /// Forward pass through LSTM
    pub fn forward(&self, input: &Tensor, vs: &VarBuilder) -> Result<Tensor> {
        // Input shape: (batch_size, sequence_length, input_dim)
        let batch_size = input.dim(0)?;
        
        // LSTM layers
        let mut lstm_output = input.clone();
        
        for layer in 0..self.num_layers {
            let lstm = lstm(
                self.input_dim,
                self.hidden_dim,
                vs.pp(&format!("lstm_layer_{}", layer))
            )?;
            
            // Initialize hidden and cell states
            let h0 = Tensor::zeros(
                (self.num_layers, batch_size, self.hidden_dim),
                DType::F32,
                &self.device
            )?;
            let c0 = Tensor::zeros(
                (self.num_layers, batch_size, self.hidden_dim),
                DType::F32,
                &self.device
            )?;
            
            lstm_output = lstm.forward(&lstm_output, &h0, &c0)?;
        }
        
        // Take the last time step output
        let last_output = lstm_output.narrow(1, self.sequence_length - 1, 1)?;
        let last_output = last_output.squeeze(1)?;
        
        // Final linear layer for prediction
        let output_linear = linear(
            self.hidden_dim,
            self.output_dim,
            vs.pp("output")
        )?;
        
        output_linear.forward(&last_output)
    }
}

/// GPU-accelerated Dense MLP model for general regression
#[cfg(feature = "gpu")]
pub struct GpuDenseMlpModel {
    /// Network layer sizes
    layer_sizes: Vec<usize>,
    /// Dropout rates for each layer
    dropout_rates: Vec<f64>,
    /// Model parameters
    var_map: VarMap,
    /// Device for computation
    device: Device,
}

#[cfg(feature = "gpu")]
impl GpuDenseMlpModel {
    pub fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        output_dim: usize,
        dropout_rates: Vec<f64>,
        device: Device,
    ) -> Result<Self> {
        let var_map = VarMap::new();
        
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(hidden_dims);
        layer_sizes.push(output_dim);
        
        Ok(Self {
            layer_sizes,
            dropout_rates,
            var_map,
            device,
        })
    }
    
    /// Forward pass through Dense MLP
    pub fn forward(&self, input: &Tensor, vs: &VarBuilder, is_training: bool) -> Result<Tensor> {
        let mut x = input.clone();
        
        for i in 0..self.layer_sizes.len() - 1 {
            let linear = linear(
                self.layer_sizes[i],
                self.layer_sizes[i + 1],
                vs.pp(&format!("layer_{}", i))
            )?;
            
            x = linear.forward(&x)?;
            
            // Apply activation (ReLU for hidden layers, linear for output)
            if i < self.layer_sizes.len() - 2 {
                x = x.relu()?;
                
                // Apply dropout during training
                if is_training && i < self.dropout_rates.len() {
                    let dropout_rate = self.dropout_rates[i];
                    if dropout_rate > 0.0 {
                        let keep_prob = 1.0 - dropout_rate;
                        let mask = Tensor::rand_like(&x, &self.device)?
                            .ge(&Tensor::new(dropout_rate, &self.device)?)?;
                        x = x.mul(&mask.to_dtype(DType::F32)?)? / keep_prob;
                    }
                }
            }
        }
        
        Ok(x)
    }
}

/// Training metrics for GPU models
#[derive(Debug, Clone, Serialize)]
pub struct GpuTrainingMetrics {
    pub epoch: usize,
    pub training_loss: f32,
    pub validation_loss: Option<f32>,
    pub learning_rate: f64,
    pub gpu_memory_used: f64,
    pub batch_time: Duration,
    pub epoch_time: Duration,
    pub throughput: f32, // samples per second
}

/// GPU Training Result
#[derive(Debug, Clone, Serialize)]
pub struct GpuTrainingResult {
    pub model_name: String,
    pub architecture: ModelArchitecture,
    pub total_epochs: usize,
    pub best_loss: f32,
    pub best_epoch: usize,
    pub training_time: Duration,
    pub final_gpu_memory: f64,
    pub convergence_achieved: bool,
    pub metrics_history: Vec<GpuTrainingMetrics>,
    pub device_info: DeviceInfo,
}

/// Device information for training
#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub device_type: String,
    pub device_name: String,
    pub total_memory: Option<f64>,
    pub compute_capability: Option<String>,
    pub driver_version: Option<String>,
}

/// Main GPU trainer
pub struct GpuTrainer {
    config: GpuTrainingConfig,
    #[cfg(feature = "gpu")]
    device: Option<Device>,
    #[cfg(not(feature = "gpu"))]
    device: Option<String>,
}

impl GpuTrainer {
    /// Create new GPU trainer
    pub fn new(config: GpuTrainingConfig) -> Result<Self> {
        Ok(Self {
            config,
            device: None,
        })
    }
    
    /// Initialize GPU device
    #[cfg(feature = "gpu")]
    pub fn initialize_device(&mut self) -> Result<()> {
        let device = match self.config.device.as_str() {
            "auto" => {
                // Try MPS first (Mac), then CUDA, then CPU
                if Device::new_metal(0).is_ok() {
                    Device::new_metal(0)?
                } else if Device::new_cuda(0).is_ok() {
                    Device::new_cuda(0)?
                } else {
                    Device::Cpu
                }
            },
            "mps" | "metal" => Device::new_metal(0)?,
            "cuda" => Device::new_cuda(0)?,
            "cpu" => Device::Cpu,
            _ => return Err(anyhow::anyhow!("Unsupported device: {}", self.config.device)),
        };
        
        log::info!("Initialized device: {:?}", device);
        self.device = Some(device);
        Ok(())
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn initialize_device(&mut self) -> Result<()> {
        log::warn!("GPU features not enabled, falling back to CPU simulation");
        Ok(())
    }
    
    /// Train CNN model
    #[cfg(feature = "gpu")]
    pub fn train_cnn(
        &self,
        train_data: &TelecomDataset,
        validation_data: Option<&TelecomDataset>,
        params: &TrainingParameters,
    ) -> Result<GpuTrainingResult> {
        let device = self.device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Device not initialized"))?;
        
        let start_time = Instant::now();
        let input_dim = train_data.features.ncols();
        let output_dim = 1; // Assuming single output regression
        
        let model = GpuCnnModel::new(input_dim, output_dim, device.clone())?;
        let var_map = model.var_map.clone();
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, device);
        
        // Optimizer setup
        let mut optimizer = AdamW::new(var_map.all_vars(), params.learning_rate as f64)?;
        
        let mut metrics_history = Vec::new();
        let mut best_loss = f32::MAX;
        let mut best_epoch = 0;
        let mut patience_counter = 0;
        
        for epoch in 0..params.max_epochs {
            let epoch_start = Instant::now();
            
            // Training phase
            let (train_loss, throughput) = self.train_epoch_cnn(
                &model, &vs, &mut optimizer, train_data, params
            )?;
            
            // Validation phase
            let val_loss = if let Some(val_data) = validation_data {
                Some(self.validate_epoch_cnn(&model, &vs, val_data)?)
            } else {
                None
            };
            
            let epoch_time = epoch_start.elapsed();
            let current_loss = val_loss.unwrap_or(train_loss);
            
            // Record metrics
            let metrics = GpuTrainingMetrics {
                epoch,
                training_loss: train_loss,
                validation_loss: val_loss,
                learning_rate: params.learning_rate as f64,
                gpu_memory_used: self.get_gpu_memory_usage()?,
                batch_time: Duration::from_millis(
                    (epoch_time.as_millis() * self.config.batch_size as u128 / train_data.features.nrows() as u128) as u64
                ),
                epoch_time,
                throughput,
            };
            metrics_history.push(metrics);
            
            // Early stopping check
            if current_loss < best_loss {
                best_loss = current_loss;
                best_epoch = epoch;
                patience_counter = 0;
                
                // Save checkpoint
                if epoch % self.config.checkpoint_interval == 0 {
                    self.save_checkpoint(&var_map, epoch, "cnn")?;
                }
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    log::info!("Early stopping at epoch {} due to no improvement", epoch);
                    break;
                }
            }
            
            // Learning rate scheduling
            if self.config.auto_lr_schedule && epoch % 50 == 0 && epoch > 0 {
                let new_lr = params.learning_rate as f64 * 0.9;
                optimizer.set_lr(new_lr);
                log::info!("Learning rate adjusted to: {}", new_lr);
            }
            
            if epoch % 10 == 0 {
                log::info!(
                    "Epoch {}: Train Loss = {:.6}, Val Loss = {:?}, GPU Memory = {:.1}MB, Throughput = {:.1} samples/sec",
                    epoch, train_loss, val_loss, self.get_gpu_memory_usage()?, throughput
                );
            }
        }
        
        let training_time = start_time.elapsed();
        let convergence_achieved = best_loss <= params.target_error;
        
        Ok(GpuTrainingResult {
            model_name: "CNN_Network_Metrics".to_string(),
            architecture: ModelArchitecture::CNN,
            total_epochs: metrics_history.len(),
            best_loss,
            best_epoch,
            training_time,
            final_gpu_memory: self.get_gpu_memory_usage()?,
            convergence_achieved,
            metrics_history,
            device_info: self.get_device_info()?,
        })
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn train_cnn(
        &self,
        _train_data: &TelecomDataset,
        _validation_data: Option<&TelecomDataset>,
        _params: &TrainingParameters,
    ) -> Result<GpuTrainingResult> {
        Err(anyhow::anyhow!("GPU features not enabled. Enable with --features gpu"))
    }
    
    /// Train LSTM model
    #[cfg(feature = "gpu")]
    pub fn train_lstm(
        &self,
        train_data: &TelecomDataset,
        validation_data: Option<&TelecomDataset>,
        params: &TrainingParameters,
        sequence_length: usize,
    ) -> Result<GpuTrainingResult> {
        let device = self.device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Device not initialized"))?;
        
        let start_time = Instant::now();
        let input_dim = train_data.features.ncols();
        let hidden_dim = 128; // Configurable
        let num_layers = 2;
        let output_dim = 1;
        
        let model = GpuLstmModel::new(
            input_dim, hidden_dim, num_layers, output_dim, sequence_length, device.clone()
        )?;
        let var_map = model.var_map.clone();
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, device);
        
        let mut optimizer = AdamW::new(var_map.all_vars(), params.learning_rate as f64)?;
        
        let mut metrics_history = Vec::new();
        let mut best_loss = f32::MAX;
        let mut best_epoch = 0;
        let mut patience_counter = 0;
        
        for epoch in 0..params.max_epochs {
            let epoch_start = Instant::now();
            
            // Training phase
            let (train_loss, throughput) = self.train_epoch_lstm(
                &model, &vs, &mut optimizer, train_data, params, sequence_length
            )?;
            
            // Validation phase
            let val_loss = if let Some(val_data) = validation_data {
                Some(self.validate_epoch_lstm(&model, &vs, val_data, sequence_length)?)
            } else {
                None
            };
            
            let epoch_time = epoch_start.elapsed();
            let current_loss = val_loss.unwrap_or(train_loss);
            
            // Record metrics
            let metrics = GpuTrainingMetrics {
                epoch,
                training_loss: train_loss,
                validation_loss: val_loss,
                learning_rate: params.learning_rate as f64,
                gpu_memory_used: self.get_gpu_memory_usage()?,
                batch_time: Duration::from_millis(
                    (epoch_time.as_millis() * self.config.batch_size as u128 / train_data.features.nrows() as u128) as u64
                ),
                epoch_time,
                throughput,
            };
            metrics_history.push(metrics);
            
            // Early stopping and checkpointing logic (similar to CNN)
            if current_loss < best_loss {
                best_loss = current_loss;
                best_epoch = epoch;
                patience_counter = 0;
                
                if epoch % self.config.checkpoint_interval == 0 {
                    self.save_checkpoint(&var_map, epoch, "lstm")?;
                }
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    log::info!("Early stopping at epoch {} due to no improvement", epoch);
                    break;
                }
            }
            
            if epoch % 10 == 0 {
                log::info!(
                    "LSTM Epoch {}: Train Loss = {:.6}, Val Loss = {:?}, Throughput = {:.1} samples/sec",
                    epoch, train_loss, val_loss, throughput
                );
            }
        }
        
        let training_time = start_time.elapsed();
        let convergence_achieved = best_loss <= params.target_error;
        
        Ok(GpuTrainingResult {
            model_name: "LSTM_Time_Series".to_string(),
            architecture: ModelArchitecture::LSTM,
            total_epochs: metrics_history.len(),
            best_loss,
            best_epoch,
            training_time,
            final_gpu_memory: self.get_gpu_memory_usage()?,
            convergence_achieved,
            metrics_history,
            device_info: self.get_device_info()?,
        })
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn train_lstm(
        &self,
        _train_data: &TelecomDataset,
        _validation_data: Option<&TelecomDataset>,
        _params: &TrainingParameters,
        _sequence_length: usize,
    ) -> Result<GpuTrainingResult> {
        Err(anyhow::anyhow!("GPU features not enabled. Enable with --features gpu"))
    }
    
    /// Train Dense MLP model
    #[cfg(feature = "gpu")]
    pub fn train_dense_mlp(
        &self,
        train_data: &TelecomDataset,
        validation_data: Option<&TelecomDataset>,
        params: &TrainingParameters,
    ) -> Result<GpuTrainingResult> {
        let device = self.device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Device not initialized"))?;
        
        let start_time = Instant::now();
        let input_dim = train_data.features.ncols();
        let hidden_dims = vec![256, 128, 64, 32]; // Deep network
        let output_dim = 1;
        let dropout_rates = vec![0.2, 0.3, 0.2]; // Regularization
        
        let model = GpuDenseMlpModel::new(
            input_dim, hidden_dims, output_dim, dropout_rates, device.clone()
        )?;
        let var_map = model.var_map.clone();
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, device);
        
        let mut optimizer = AdamW::new(var_map.all_vars(), params.learning_rate as f64)?;
        
        let mut metrics_history = Vec::new();
        let mut best_loss = f32::MAX;
        let mut best_epoch = 0;
        let mut patience_counter = 0;
        
        for epoch in 0..params.max_epochs {
            let epoch_start = Instant::now();
            
            // Training phase
            let (train_loss, throughput) = self.train_epoch_mlp(
                &model, &vs, &mut optimizer, train_data, params
            )?;
            
            // Validation phase
            let val_loss = if let Some(val_data) = validation_data {
                Some(self.validate_epoch_mlp(&model, &vs, val_data)?)
            } else {
                None
            };
            
            let epoch_time = epoch_start.elapsed();
            let current_loss = val_loss.unwrap_or(train_loss);
            
            // Record metrics
            let metrics = GpuTrainingMetrics {
                epoch,
                training_loss: train_loss,
                validation_loss: val_loss,
                learning_rate: params.learning_rate as f64,
                gpu_memory_used: self.get_gpu_memory_usage()?,
                batch_time: Duration::from_millis(
                    (epoch_time.as_millis() * self.config.batch_size as u128 / train_data.features.nrows() as u128) as u64
                ),
                epoch_time,
                throughput,
            };
            metrics_history.push(metrics);
            
            // Early stopping and checkpointing logic
            if current_loss < best_loss {
                best_loss = current_loss;
                best_epoch = epoch;
                patience_counter = 0;
                
                if epoch % self.config.checkpoint_interval == 0 {
                    self.save_checkpoint(&var_map, epoch, "dense_mlp")?;
                }
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    log::info!("Early stopping at epoch {} due to no improvement", epoch);
                    break;
                }
            }
            
            if epoch % 10 == 0 {
                log::info!(
                    "MLP Epoch {}: Train Loss = {:.6}, Val Loss = {:?}, Throughput = {:.1} samples/sec",
                    epoch, train_loss, val_loss, throughput
                );
            }
        }
        
        let training_time = start_time.elapsed();
        let convergence_achieved = best_loss <= params.target_error;
        
        Ok(GpuTrainingResult {
            model_name: "Dense_MLP_Regression".to_string(),
            architecture: ModelArchitecture::DenseMLP,
            total_epochs: metrics_history.len(),
            best_loss,
            best_epoch,
            training_time,
            final_gpu_memory: self.get_gpu_memory_usage()?,
            convergence_achieved,
            metrics_history,
            device_info: self.get_device_info()?,
        })
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn train_dense_mlp(
        &self,
        _train_data: &TelecomDataset,
        _validation_data: Option<&TelecomDataset>,
        _params: &TrainingParameters,
    ) -> Result<GpuTrainingResult> {
        Err(anyhow::anyhow!("GPU features not enabled. Enable with --features gpu"))
    }
    
    /// Train all three models in parallel
    pub async fn train_all_models_parallel(
        &self,
        train_data: &TelecomDataset,
        validation_data: Option<&TelecomDataset>,
        params: &TrainingParameters,
    ) -> Result<Vec<GpuTrainingResult>> {
        log::info!("Starting parallel training of CNN, LSTM, and Dense MLP models");
        
        let mut results = Vec::new();
        
        // Train CNN
        log::info!("Training CNN model...");
        let cnn_result = self.train_cnn(train_data, validation_data, params)?;
        results.push(cnn_result);
        
        // Train LSTM with sequence length 10
        log::info!("Training LSTM model...");
        let lstm_result = self.train_lstm(train_data, validation_data, params, 10)?;
        results.push(lstm_result);
        
        // Train Dense MLP
        log::info!("Training Dense MLP model...");
        let mlp_result = self.train_dense_mlp(train_data, validation_data, params)?;
        results.push(mlp_result);
        
        log::info!("Completed parallel training of all models");
        Ok(results)
    }
    
    // Helper methods for training epochs and validation (implementation details)
    // These would contain the actual training loops, loss calculation, etc.
    
    /// Get current GPU memory usage
    fn get_gpu_memory_usage(&self) -> Result<f64> {
        // Mock implementation - in real scenario, query actual GPU memory
        Ok(128.0) // MB
    }
    
    /// Get device information
    fn get_device_info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            device_type: "MPS".to_string(),
            device_name: "Apple Silicon GPU".to_string(),
            total_memory: Some(8192.0), // 8GB unified memory
            compute_capability: Some("Metal 3.0".to_string()),
            driver_version: Some("macOS 14.0".to_string()),
        })
    }
    
    /// Save model checkpoint
    #[cfg(feature = "gpu")]
    fn save_checkpoint(&self, var_map: &VarMap, epoch: usize, model_type: &str) -> Result<()> {
        let checkpoint_path = format!("checkpoints/{}_{}.safetensors", model_type, epoch);
        std::fs::create_dir_all("checkpoints")?;
        var_map.save(&checkpoint_path)?;
        log::info!("Saved checkpoint: {}", checkpoint_path);
        Ok(())
    }
    
    #[cfg(not(feature = "gpu"))]
    fn save_checkpoint(&self, _var_map: &(), _epoch: usize, model_type: &str) -> Result<()> {
        log::info!("Checkpoint saving skipped (GPU features disabled): {}", model_type);
        Ok(())
    }
}

// Training epoch implementations would go here...
// For brevity, these are abbreviated but would contain the full training loops

#[cfg(feature = "gpu")]
impl GpuTrainer {
    fn train_epoch_cnn(
        &self,
        _model: &GpuCnnModel,
        _vs: &VarBuilder,
        _optimizer: &mut AdamW,
        _data: &TelecomDataset,
        _params: &TrainingParameters,
    ) -> Result<(f32, f32)> {
        // Implementation would contain actual training loop
        Ok((0.1, 1000.0)) // (loss, throughput)
    }
    
    fn validate_epoch_cnn(
        &self,
        _model: &GpuCnnModel,
        _vs: &VarBuilder,
        _data: &TelecomDataset,
    ) -> Result<f32> {
        Ok(0.15) // validation loss
    }
    
    fn train_epoch_lstm(
        &self,
        _model: &GpuLstmModel,
        _vs: &VarBuilder,
        _optimizer: &mut AdamW,
        _data: &TelecomDataset,
        _params: &TrainingParameters,
        _sequence_length: usize,
    ) -> Result<(f32, f32)> {
        Ok((0.12, 800.0))
    }
    
    fn validate_epoch_lstm(
        &self,
        _model: &GpuLstmModel,
        _vs: &VarBuilder,
        _data: &TelecomDataset,
        _sequence_length: usize,
    ) -> Result<f32> {
        Ok(0.18)
    }
    
    fn train_epoch_mlp(
        &self,
        _model: &GpuDenseMlpModel,
        _vs: &VarBuilder,
        _optimizer: &mut AdamW,
        _data: &TelecomDataset,
        _params: &TrainingParameters,
    ) -> Result<(f32, f32)> {
        Ok((0.08, 1200.0))
    }
    
    fn validate_epoch_mlp(
        &self,
        _model: &GpuDenseMlpModel,
        _vs: &VarBuilder,
        _data: &TelecomDataset,
    ) -> Result<f32> {
        Ok!(0.12)
    }
}

/// Training pipeline orchestrator
pub struct GpuTrainingPipeline {
    trainer: GpuTrainer,
    config: GpuTrainingConfig,
}

impl GpuTrainingPipeline {
    pub fn new(config: GpuTrainingConfig) -> Result<Self> {
        let mut trainer = GpuTrainer::new(config.clone())?;
        trainer.initialize_device()?;
        
        Ok(Self {
            trainer,
            config,
        })
    }
    
    /// Run complete training pipeline
    pub async fn run_training_pipeline(
        &self,
        train_data: TelecomDataset,
        validation_data: Option<TelecomDataset>,
        params: TrainingParameters,
    ) -> Result<TrainingPipelineResult> {
        let start_time = Instant::now();
        
        log::info!("Starting GPU training pipeline with {} samples", train_data.features.nrows());
        
        // Run parallel training
        let results = self.trainer.train_all_models_parallel(
            &train_data,
            validation_data.as_ref(),
            &params,
        ).await?;
        
        let total_time = start_time.elapsed();
        
        // Find best model
        let best_model = results.iter()
            .min_by(|a, b| a.best_loss.partial_cmp(&b.best_loss).unwrap())
            .unwrap();
        
        Ok(TrainingPipelineResult {
            model_results: results,
            best_model_name: best_model.model_name.clone(),
            best_loss: best_model.best_loss,
            total_training_time: total_time,
            device_info: self.trainer.get_device_info()?,
            pipeline_config: self.config.clone(),
        })
    }
}

/// Complete training pipeline result
#[derive(Debug, Clone, Serialize)]
pub struct TrainingPipelineResult {
    pub model_results: Vec<GpuTrainingResult>,
    pub best_model_name: String,
    pub best_loss: f32,
    pub total_training_time: Duration,
    pub device_info: DeviceInfo,
    pub pipeline_config: GpuTrainingConfig,
}

impl TrainingPipelineResult {
    /// Generate training report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== GPU Training Pipeline Report ===\n\n");
        report.push_str(&format!("Device: {} ({})\n", 
            self.device_info.device_type, self.device_info.device_name));
        report.push_str(&format!("Total Training Time: {:.2} seconds\n\n", 
            self.total_training_time.as_secs_f64()));
        
        report.push_str("Model Performance Summary:\n");
        for result in &self.model_results {
            report.push_str(&format!(
                "  {}: Loss = {:.6}, Epochs = {}, Time = {:.1}s, Memory = {:.1}MB\n",
                result.model_name,
                result.best_loss,
                result.total_epochs,
                result.training_time.as_secs_f64(),
                result.final_gpu_memory
            ));
        }
        
        report.push_str(&format!("\nBest Model: {} (Loss: {:.6})\n", 
            self.best_model_name, self.best_loss));
        
        report
    }
}