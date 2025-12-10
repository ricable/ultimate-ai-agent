//! # Loss Functions for Neural Forecasting
//!
//! Comprehensive collection of loss functions specifically designed for time series forecasting,
//! including both point forecasting and probabilistic forecasting losses.
//!
//! ## Point Forecasting Losses
//!
//! - **MAE**: Mean Absolute Error
//! - **MSE**: Mean Squared Error  
//! - **RMSE**: Root Mean Squared Error
//! - **MAPE**: Mean Absolute Percentage Error
//! - **SMAPE**: Symmetric Mean Absolute Percentage Error
//! - **MASE**: Mean Absolute Scaled Error
//!
//! ## Probabilistic Forecasting Losses
//!
//! - **NegativeLogLikelihood**: For probabilistic models
//! - **PinballLoss**: For quantile forecasting
//! - **CRPS**: Continuous Ranked Probability Score
//!
//! ## Distribution-Specific Losses
//!
//! - **GaussianNLL**: Gaussian Negative Log-Likelihood
//! - **PoissonNLL**: Poisson Negative Log-Likelihood
//! - **NegativeBinomialNLL**: Negative Binomial NLL
//!
//! ## Robust Losses
//!
//! - **HuberLoss**: Robust to outliers
//! - **QuantileLoss**: For quantile regression
//!
//! ## Custom Losses
//!
//! - **ScaledLoss**: Scale-invariant loss
//! - **SeasonalLoss**: Seasonal-aware loss

use num_traits::Float;
use std::marker::PhantomData;
use crate::{TrainingError, TrainingResult};

/// Core trait for loss functions in neural forecasting
pub trait LossFunction<T: Float + Send + Sync>: Send + Sync {
    /// Calculate the forward loss
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T>;
    
    /// Calculate the backward gradients
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>>;
    
    /// Get the name of the loss function
    fn name(&self) -> &'static str;
    
    /// Whether the loss function requires additional parameters
    fn requires_params(&self) -> bool { false }
    
    /// Set additional parameters if needed
    fn set_params(&mut self, _params: &[T]) -> TrainingResult<()> { Ok(()) }
}

/// Loss function wrapper that can be used with different concrete types
#[derive(Clone)]
pub enum Loss<T: Float + Send + Sync> {
    MAE(MAELoss<T>),
    MSE(MSELoss<T>),
    RMSE(RMSELoss<T>),
    MAPE(MAPELoss<T>),
    SMAPE(SMAPELoss<T>),
    MASE(MASELoss<T>),
    NegativeLogLikelihood(NegativeLogLikelihoodLoss<T>),
    PinballLoss(PinballLoss<T>),
    CRPS(CRPSLoss<T>),
    GaussianNLL(GaussianNLLLoss<T>),
    PoissonNLL(PoissonNLLLoss<T>),
    NegativeBinomialNLL(NegativeBinomialNLLLoss<T>),
    HuberLoss(HuberLoss<T>),
    QuantileLoss(QuantileLoss<T>),
    ScaledLoss(ScaledLoss<T>),
    SeasonalLoss(SeasonalLoss<T>),
}

impl<T: Float + Send + Sync> LossFunction<T> for Loss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        match self {
            Loss::MAE(loss) => loss.forward(predictions, targets),
            Loss::MSE(loss) => loss.forward(predictions, targets),
            Loss::RMSE(loss) => loss.forward(predictions, targets),
            Loss::MAPE(loss) => loss.forward(predictions, targets),
            Loss::SMAPE(loss) => loss.forward(predictions, targets),
            Loss::MASE(loss) => loss.forward(predictions, targets),
            Loss::NegativeLogLikelihood(loss) => loss.forward(predictions, targets),
            Loss::PinballLoss(loss) => loss.forward(predictions, targets),
            Loss::CRPS(loss) => loss.forward(predictions, targets),
            Loss::GaussianNLL(loss) => loss.forward(predictions, targets),
            Loss::PoissonNLL(loss) => loss.forward(predictions, targets),
            Loss::NegativeBinomialNLL(loss) => loss.forward(predictions, targets),
            Loss::HuberLoss(loss) => loss.forward(predictions, targets),
            Loss::QuantileLoss(loss) => loss.forward(predictions, targets),
            Loss::ScaledLoss(loss) => loss.forward(predictions, targets),
            Loss::SeasonalLoss(loss) => loss.forward(predictions, targets),
        }
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        match self {
            Loss::MAE(loss) => loss.backward(predictions, targets),
            Loss::MSE(loss) => loss.backward(predictions, targets),
            Loss::RMSE(loss) => loss.backward(predictions, targets),
            Loss::MAPE(loss) => loss.backward(predictions, targets),
            Loss::SMAPE(loss) => loss.backward(predictions, targets),
            Loss::MASE(loss) => loss.backward(predictions, targets),
            Loss::NegativeLogLikelihood(loss) => loss.backward(predictions, targets),
            Loss::PinballLoss(loss) => loss.backward(predictions, targets),
            Loss::CRPS(loss) => loss.backward(predictions, targets),
            Loss::GaussianNLL(loss) => loss.backward(predictions, targets),
            Loss::PoissonNLL(loss) => loss.backward(predictions, targets),
            Loss::NegativeBinomialNLL(loss) => loss.backward(predictions, targets),
            Loss::HuberLoss(loss) => loss.backward(predictions, targets),
            Loss::QuantileLoss(loss) => loss.backward(predictions, targets),
            Loss::ScaledLoss(loss) => loss.backward(predictions, targets),
            Loss::SeasonalLoss(loss) => loss.backward(predictions, targets),
        }
    }
    
    fn name(&self) -> &'static str {
        match self {
            Loss::MAE(loss) => loss.name(),
            Loss::MSE(loss) => loss.name(),
            Loss::RMSE(loss) => loss.name(),
            Loss::MAPE(loss) => loss.name(),
            Loss::SMAPE(loss) => loss.name(),
            Loss::MASE(loss) => loss.name(),
            Loss::NegativeLogLikelihood(loss) => loss.name(),
            Loss::PinballLoss(loss) => loss.name(),
            Loss::CRPS(loss) => loss.name(),
            Loss::GaussianNLL(loss) => loss.name(),
            Loss::PoissonNLL(loss) => loss.name(),
            Loss::NegativeBinomialNLL(loss) => loss.name(),
            Loss::HuberLoss(loss) => loss.name(),
            Loss::QuantileLoss(loss) => loss.name(),
            Loss::ScaledLoss(loss) => loss.name(),
            Loss::SeasonalLoss(loss) => loss.name(),
        }
    }
}

// =============================================================================
// Point Forecasting Losses
// =============================================================================

/// Mean Absolute Error (MAE)
#[derive(Clone)]
pub struct MAELoss<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> MAELoss<T> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for MAELoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let sum = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                if pred > target {
                    T::one() / n
                } else if pred < target {
                    -T::one() / n
                } else {
                    T::zero()
                }
            })
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "MAE"
    }
}

impl<T: Float + Send + Sync> Default for MAELoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean Squared Error (MSE)
#[derive(Clone)]
pub struct MSELoss<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> MSELoss<T> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for MSELoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let sum = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let two = T::from(2.0).unwrap();
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| two * (pred - target) / n)
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "MSE"
    }
}

impl<T: Float + Send + Sync> Default for MSELoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Root Mean Squared Error (RMSE)
#[derive(Clone)]
pub struct RMSELoss<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> RMSELoss<T> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for RMSELoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        let mse_loss = MSELoss::new();
        let mse = mse_loss.forward(predictions, targets)?;
        Ok(mse.sqrt())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        let mse_loss = MSELoss::new();
        let mse = mse_loss.forward(predictions, targets)?;
        let rmse = mse.sqrt();
        
        if rmse.is_zero() {
            return Ok(vec![T::zero(); predictions.len()]);
        }
        
        let mse_gradients = mse_loss.backward(predictions, targets)?;
        let scale = T::one() / (T::from(2.0).unwrap() * rmse);
        
        Ok(mse_gradients.iter().map(|&g| g * scale).collect())
    }
    
    fn name(&self) -> &'static str {
        "RMSE"
    }
}

impl<T: Float + Send + Sync> Default for RMSELoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean Absolute Percentage Error (MAPE)
#[derive(Clone)]
pub struct MAPELoss<T: Float + Send + Sync> {
    epsilon: T,
}

impl<T: Float + Send + Sync> MAPELoss<T> {
    pub fn new() -> Self {
        Self { 
            epsilon: T::from(1e-8).unwrap() 
        }
    }
    
    pub fn with_epsilon(epsilon: T) -> Self {
        Self { epsilon }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for MAPELoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let sum = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let denominator = target.abs().max(self.epsilon);
                (pred - target).abs() / denominator
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        let hundred = T::from(100.0).unwrap();
        Ok(hundred * sum / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let hundred = T::from(100.0).unwrap();
        
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let denominator = target.abs().max(self.epsilon);
                let sign = if pred > target { T::one() } else { -T::one() };
                hundred * sign / (n * denominator)
            })
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "MAPE"
    }
}

impl<T: Float + Send + Sync> Default for MAPELoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Symmetric Mean Absolute Percentage Error (SMAPE)
#[derive(Clone)]
pub struct SMAPELoss<T: Float + Send + Sync> {
    epsilon: T,
}

impl<T: Float + Send + Sync> SMAPELoss<T> {
    pub fn new() -> Self {
        Self { 
            epsilon: T::from(1e-8).unwrap() 
        }
    }
    
    pub fn with_epsilon(epsilon: T) -> Self {
        Self { epsilon }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for SMAPELoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let sum = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let numerator = (pred - target).abs();
                let denominator = (pred.abs() + target.abs()).max(self.epsilon);
                numerator / denominator
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        let hundred = T::from(100.0).unwrap();
        Ok(hundred * sum / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let hundred = T::from(100.0).unwrap();
        let two = T::from(2.0).unwrap();
        
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let abs_sum = (pred.abs() + target.abs()).max(self.epsilon);
                let diff = pred - target;
                let abs_diff = diff.abs();
                
                let sign_diff = if pred > target { T::one() } else { -T::one() };
                let sign_pred = if pred > T::zero() { T::one() } else { -T::one() };
                
                let numerator = sign_diff * abs_sum - abs_diff * sign_pred;
                let denominator = abs_sum * abs_sum;
                
                hundred * numerator / (n * denominator)
            })
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "SMAPE"
    }
}

impl<T: Float + Send + Sync> Default for SMAPELoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean Absolute Scaled Error (MASE)
#[derive(Clone)]
pub struct MASELoss<T: Float + Send + Sync> {
    seasonal_naive_error: T,
    epsilon: T,
}

impl<T: Float + Send + Sync> MASELoss<T> {
    pub fn new(seasonal_naive_error: T) -> Self {
        Self { 
            seasonal_naive_error,
            epsilon: T::from(1e-8).unwrap()
        }
    }
    
    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for MASELoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let mae = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .fold(T::zero(), |acc, x| acc + x) / T::from(predictions.len()).unwrap();
        
        let denominator = self.seasonal_naive_error.max(self.epsilon);
        Ok(mae / denominator)
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let denominator = self.seasonal_naive_error.max(self.epsilon);
        
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let sign = if pred > target { T::one() } else { -T::one() };
                sign / (n * denominator)
            })
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "MASE"
    }
    
    fn requires_params(&self) -> bool {
        true
    }
}

// =============================================================================
// Probabilistic Forecasting Losses  
// =============================================================================

/// Negative Log-Likelihood Loss for probabilistic forecasting
#[derive(Clone)]
pub struct NegativeLogLikelihoodLoss<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> NegativeLogLikelihoodLoss<T> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for NegativeLogLikelihoodLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        // predictions should contain [mean, log_var] for each target
        if predictions.len() != targets.len() * 2 {
            return Err(TrainingError::LossError("Predictions should contain mean and log_var".to_string()));
        }
        
        let mut nll = T::zero();
        let two = T::from(2.0).unwrap();
        let pi = T::from(std::f64::consts::PI).unwrap();
        
        for i in 0..targets.len() {
            let mean = predictions[i * 2];
            let log_var = predictions[i * 2 + 1];
            let target = targets[i];
            
            let var = log_var.exp();
            let diff = target - mean;
            
            nll = nll + log_var / two + (diff * diff) / (two * var) + (two * pi).ln() / two;
        }
        
        Ok(nll / T::from(targets.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() * 2 {
            return Err(TrainingError::LossError("Predictions should contain mean and log_var".to_string()));
        }
        
        let mut gradients = vec![T::zero(); predictions.len()];
        let n = T::from(targets.len()).unwrap();
        let two = T::from(2.0).unwrap();
        
        for i in 0..targets.len() {
            let mean = predictions[i * 2];
            let log_var = predictions[i * 2 + 1];
            let target = targets[i];
            
            let var = log_var.exp();
            let diff = target - mean;
            
            // Gradient w.r.t. mean
            gradients[i * 2] = -diff / (var * n);
            
            // Gradient w.r.t. log_var
            gradients[i * 2 + 1] = (T::one() / two - diff * diff / (two * var)) / n;
        }
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "NegativeLogLikelihood"
    }
}

impl<T: Float + Send + Sync> Default for NegativeLogLikelihoodLoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Pinball Loss for quantile forecasting
#[derive(Clone)]
pub struct PinballLoss<T: Float + Send + Sync> {
    quantile: T,
}

impl<T: Float + Send + Sync> PinballLoss<T> {
    pub fn new(quantile: T) -> Self {
        Self { quantile }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for PinballLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let sum = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = target - pred;
                if diff >= T::zero() {
                    self.quantile * diff
                } else {
                    (self.quantile - T::one()) * diff
                }
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                if target >= pred {
                    -self.quantile / n
                } else {
                    (T::one() - self.quantile) / n
                }
            })
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "PinballLoss"
    }
    
    fn requires_params(&self) -> bool {
        true
    }
}

/// Continuous Ranked Probability Score (CRPS)
#[derive(Clone)]
pub struct CRPSLoss<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> CRPSLoss<T> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for CRPSLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        // For simplicity, assume Gaussian distribution with mean and std
        if predictions.len() != targets.len() * 2 {
            return Err(TrainingError::LossError("Predictions should contain mean and std".to_string()));
        }
        
        let mut crps = T::zero();
        let sqrt_pi = T::from(std::f64::consts::PI.sqrt()).unwrap();
        
        for i in 0..targets.len() {
            let mean = predictions[i * 2];
            let std = predictions[i * 2 + 1];
            let target = targets[i];
            
            let z = (target - mean) / std;
            let phi_z = (-z * z / T::from(2.0).unwrap()).exp() / sqrt_pi;
            let erf_z = erf_approx(z / T::from(2.0_f64.sqrt()).unwrap());
            
            crps = crps + std * (z * erf_z + phi_z - T::one() / sqrt_pi);
        }
        
        Ok(crps / T::from(targets.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        // Simplified gradient approximation
        let mut gradients = vec![T::zero(); predictions.len()];
        let n = T::from(targets.len()).unwrap();
        
        for i in 0..targets.len() {
            let mean = predictions[i * 2];
            let std = predictions[i * 2 + 1];
            let target = targets[i];
            
            let z = (target - mean) / std;
            let erf_z = erf_approx(z / T::from(2.0_f64.sqrt()).unwrap());
            
            gradients[i * 2] = -erf_z / n;
            gradients[i * 2 + 1] = (T::one() - erf_z * z) / n;
        }
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "CRPS"
    }
}

impl<T: Float + Send + Sync> Default for CRPSLoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Distribution-Specific Losses
// =============================================================================

/// Gaussian Negative Log-Likelihood
#[derive(Clone)]
pub struct GaussianNLLLoss<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> GaussianNLLLoss<T> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for GaussianNLLLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        // Same as NegativeLogLikelihoodLoss but more explicit
        let nll_loss = NegativeLogLikelihoodLoss::new();
        nll_loss.forward(predictions, targets)
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        let nll_loss = NegativeLogLikelihoodLoss::new();
        nll_loss.backward(predictions, targets)
    }
    
    fn name(&self) -> &'static str {
        "GaussianNLL"
    }
}

impl<T: Float + Send + Sync> Default for GaussianNLLLoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Poisson Negative Log-Likelihood
#[derive(Clone)]
pub struct PoissonNLLLoss<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> PoissonNLLLoss<T> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for PoissonNLLLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let mut nll = T::zero();
        
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let lambda = pred.exp(); // Ensure positive rate parameter
            nll = nll + lambda - target * lambda.ln() + log_factorial_approx(target);
        }
        
        Ok(nll / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let lambda = pred.exp();
                (lambda - target) / n
            })
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "PoissonNLL"
    }
}

impl<T: Float + Send + Sync> Default for PoissonNLLLoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Negative Binomial Negative Log-Likelihood
#[derive(Clone)]
pub struct NegativeBinomialNLLLoss<T: Float + Send + Sync> {
    r: T, // Dispersion parameter
}

impl<T: Float + Send + Sync> NegativeBinomialNLLLoss<T> {
    pub fn new(r: T) -> Self {
        Self { r }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for NegativeBinomialNLLLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let mut nll = T::zero();
        
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let mu = pred.exp();
            let p = self.r / (self.r + mu);
            
            nll = nll - log_gamma_approx(target + self.r) + log_gamma_approx(self.r)
                + log_factorial_approx(target) - target * (T::one() - p).ln() - self.r * p.ln();
        }
        
        Ok(nll / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let mu = pred.exp();
                let gradient = (mu - target) / (mu + self.r);
                gradient / n
            })
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "NegativeBinomialNLL"
    }
    
    fn requires_params(&self) -> bool {
        true
    }
}

// =============================================================================
// Robust Losses
// =============================================================================

/// Huber Loss (robust to outliers)
#[derive(Clone)]
pub struct HuberLoss<T: Float + Send + Sync> {
    delta: T,
}

impl<T: Float + Send + Sync> HuberLoss<T> {
    pub fn new(delta: T) -> Self {
        Self { delta }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for HuberLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let sum = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = (pred - target).abs();
                if diff <= self.delta {
                    diff * diff / T::from(2.0).unwrap()
                } else {
                    self.delta * diff - self.delta * self.delta / T::from(2.0).unwrap()
                }
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(predictions.len()).unwrap();
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                let abs_diff = diff.abs();
                
                if abs_diff <= self.delta {
                    diff / n
                } else {
                    self.delta * diff.signum() / n
                }
            })
            .collect();
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "HuberLoss"
    }
    
    fn requires_params(&self) -> bool {
        true
    }
}

/// Quantile Loss for quantile regression
#[derive(Clone)]
pub struct QuantileLoss<T: Float + Send + Sync> {
    quantiles: Vec<T>,
}

impl<T: Float + Send + Sync> QuantileLoss<T> {
    pub fn new(quantiles: Vec<T>) -> Self {
        Self { quantiles }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for QuantileLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() * self.quantiles.len() {
            return Err(TrainingError::LossError("Predictions length mismatch".to_string()));
        }
        
        let mut total_loss = T::zero();
        
        for (i, &quantile) in self.quantiles.iter().enumerate() {
            for j in 0..targets.len() {
                let pred = predictions[j * self.quantiles.len() + i];
                let target = targets[j];
                let diff = target - pred;
                
                if diff >= T::zero() {
                    total_loss = total_loss + quantile * diff;
                } else {
                    total_loss = total_loss + (quantile - T::one()) * diff;
                }
            }
        }
        
        Ok(total_loss / T::from(predictions.len()).unwrap())
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        if predictions.len() != targets.len() * self.quantiles.len() {
            return Err(TrainingError::LossError("Predictions length mismatch".to_string()));
        }
        
        let mut gradients = vec![T::zero(); predictions.len()];
        let n = T::from(predictions.len()).unwrap();
        
        for (i, &quantile) in self.quantiles.iter().enumerate() {
            for j in 0..targets.len() {
                let idx = j * self.quantiles.len() + i;
                let pred = predictions[idx];
                let target = targets[j];
                
                gradients[idx] = if target >= pred {
                    -quantile / n
                } else {
                    (T::one() - quantile) / n
                };
            }
        }
        
        Ok(gradients)
    }
    
    fn name(&self) -> &'static str {
        "QuantileLoss"
    }
    
    fn requires_params(&self) -> bool {
        true
    }
}

// =============================================================================
// Custom Losses
// =============================================================================

/// Scale-invariant loss function
#[derive(Clone)]
pub struct ScaledLoss<T: Float + Send + Sync> {
    base_loss: Box<Loss<T>>,
    scale_factor: T,
}

impl<T: Float + Send + Sync> ScaledLoss<T> {
    pub fn new(base_loss: Loss<T>, scale_factor: T) -> Self {
        Self { 
            base_loss: Box::new(base_loss),
            scale_factor,
        }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for ScaledLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        let base_loss = self.base_loss.forward(predictions, targets)?;
        Ok(base_loss / self.scale_factor)
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        let base_gradients = self.base_loss.backward(predictions, targets)?;
        Ok(base_gradients.iter().map(|&g| g / self.scale_factor).collect())
    }
    
    fn name(&self) -> &'static str {
        "ScaledLoss"
    }
    
    fn requires_params(&self) -> bool {
        true
    }
}

/// Seasonal-aware loss function
#[derive(Clone)]
pub struct SeasonalLoss<T: Float + Send + Sync> {
    base_loss: Box<Loss<T>>,
    seasonal_weights: Vec<T>,
}

impl<T: Float + Send + Sync> SeasonalLoss<T> {
    pub fn new(base_loss: Loss<T>, seasonal_weights: Vec<T>) -> Self {
        Self { 
            base_loss: Box::new(base_loss),
            seasonal_weights,
        }
    }
}

impl<T: Float + Send + Sync> LossFunction<T> for SeasonalLoss<T> {
    fn forward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        if predictions.len() != targets.len() {
            return Err(TrainingError::LossError("Dimension mismatch".to_string()));
        }
        
        let base_loss = self.base_loss.forward(predictions, targets)?;
        
        // Apply seasonal weighting
        let seasonal_factor = if !self.seasonal_weights.is_empty() {
            let season_idx = predictions.len() % self.seasonal_weights.len();
            self.seasonal_weights[season_idx]
        } else {
            T::one()
        };
        
        Ok(base_loss * seasonal_factor)
    }
    
    fn backward(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        let base_gradients = self.base_loss.backward(predictions, targets)?;
        
        let seasonal_factor = if !self.seasonal_weights.is_empty() {
            let season_idx = predictions.len() % self.seasonal_weights.len();
            self.seasonal_weights[season_idx]
        } else {
            T::one()
        };
        
        Ok(base_gradients.iter().map(|&g| g * seasonal_factor).collect())
    }
    
    fn name(&self) -> &'static str {
        "SeasonalLoss"
    }
    
    fn requires_params(&self) -> bool {
        true
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Approximation of the error function (erf)
fn erf_approx<T: Float + Send + Sync>(x: T) -> T {
    // Abramowitz and Stegun approximation
    let a1 = T::from(0.254829592).unwrap();
    let a2 = T::from(-0.284496736).unwrap();
    let a3 = T::from(1.421413741).unwrap();
    let a4 = T::from(-1.453152027).unwrap();
    let a5 = T::from(1.061405429).unwrap();
    let p = T::from(0.3275911).unwrap();
    
    let sign = if x >= T::zero() { T::one() } else { -T::one() };
    let x = x.abs();
    
    let t = T::one() / (T::one() + p * x);
    let y = T::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

/// Stirling's approximation for log factorial
fn log_factorial_approx<T: Float + Send + Sync>(n: T) -> T {
    if n <= T::one() {
        return T::zero();
    }
    
    let ln_2pi = T::from((2.0 * std::f64::consts::PI).ln()).unwrap();
    n * n.ln() - n + (ln_2pi / T::from(2.0).unwrap() + n.ln() / T::from(2.0).unwrap())
}

/// Approximation of log gamma function
fn log_gamma_approx<T: Float + Send + Sync>(x: T) -> T {
    if x <= T::zero() {
        return T::zero();
    }
    
    // Stirling's approximation
    let ln_2pi = T::from((2.0 * std::f64::consts::PI).ln()).unwrap();
    (x - T::from(0.5).unwrap()) * x.ln() - x + ln_2pi / T::from(2.0).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_mae_loss() {
        let loss = MAELoss::new();
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 2.5];
        
        let result = loss.forward(&predictions, &targets).unwrap();
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
        
        let gradients = loss.backward(&predictions, &targets).unwrap();
        let expected = vec![-1.0/3.0, -1.0/3.0, 1.0/3.0];
        for (g, e) in gradients.iter().zip(expected.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_mse_loss() {
        let loss = MSELoss::new();
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];
        
        let result = loss.forward(&predictions, &targets).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
        
        let gradients = loss.backward(&predictions, &targets).unwrap();
        for g in gradients.iter() {
            assert_relative_eq!(*g, 0.0, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_mape_loss() {
        let loss = MAPELoss::new();
        let predictions = vec![100.0, 200.0];
        let targets = vec![110.0, 180.0];
        
        let result = loss.forward(&predictions, &targets).unwrap();
        // MAPE = 100 * (|100-110|/110 + |200-180|/180) / 2
        // = 100 * (10/110 + 20/180) / 2 ≈ 10.1
        assert_relative_eq!(result, 10.101010, epsilon = 1e-5);
    }
    
    #[test]
    fn test_pinball_loss() {
        let loss = PinballLoss::new(0.5);
        let predictions = vec![1.0, 2.0];
        let targets = vec![1.5, 1.5];
        
        let result = loss.forward(&predictions, &targets).unwrap();
        // Pinball(0.5) = 0.5 * max(y-pred, 0) + 0.5 * max(pred-y, 0)
        assert_relative_eq!(result, 0.25, epsilon = 1e-6);
        
        let gradients = loss.backward(&predictions, &targets).unwrap();
        assert_eq!(gradients.len(), 2);
    }
    
    #[test]
    fn test_huber_loss() {
        let loss = HuberLoss::new(1.0);
        let predictions = vec![0.0, 2.0];
        let targets = vec![0.5, 0.0];
        
        let result = loss.forward(&predictions, &targets).unwrap();
        // First term: |0-0.5| = 0.5 <= 1, so (0.5)^2/2 = 0.125
        // Second term: |2-0| = 2 > 1, so 1*2 - 1^2/2 = 1.5
        // Average: (0.125 + 1.5) / 2 = 0.8125
        assert_relative_eq!(result, 0.8125, epsilon = 1e-6);
    }
}