//! Basic neural forecasting model implementations
//!
//! This module contains implementations of fundamental forecasting models:
//! - MLP: Multi-Layer Perceptron for non-linear time series forecasting
//! - DLinear: Direct Linear decomposition model
//! - NLinear: Normalized Linear model
//! - MLPMultivariate: Multi-variate extension of MLP

pub mod mlp;
pub mod dlinear;
pub mod nlinear;
pub mod mlp_multivariate;

// Re-export the main model types
pub use mlp::{MLP, MLPConfig, MLPBuilder};
pub use dlinear::{DLinear, DLinearConfig, DLinearBuilder};
pub use nlinear::{NLinear, NLinearConfig, NLinearBuilder};
pub use mlp_multivariate::{MLPMultivariate, MLPMultivariateConfig, MLPMultivariateBuilder};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BaseModel, TimeSeriesData};
    use approx::assert_relative_eq;

    #[test]
    fn test_all_models_creation() {
        // Test MLP creation
        let mlp_config = MLPConfig::new(10, 5);
        let mlp = MLP::<f64>::new(mlp_config);
        assert!(mlp.is_ok());

        // Test DLinear creation  
        let dlinear_config = DLinearConfig::new(10, 5);
        let dlinear = DLinear::<f64>::new(dlinear_config);
        assert!(dlinear.is_ok());

        // Test NLinear creation
        let nlinear_config = NLinearConfig::new(10, 5);
        let nlinear = NLinear::<f64>::new(nlinear_config);
        assert!(nlinear.is_ok());

        // Test MLPMultivariate creation
        let mlp_mv_config = MLPMultivariateConfig::new(2, 10, 5);
        let mlp_mv = MLPMultivariate::<f64>::new(mlp_mv_config);
        assert!(mlp_mv.is_ok());
    }

    #[test]
    fn test_basic_workflow() {
        // Create some test data
        let target = (0..20).map(|i| i as f64 * 0.1).collect();
        let data = TimeSeriesData::new(target);

        // Test MLP workflow
        let config = MLPConfig::new(10, 5);
        let mut model = MLP::<f64>::new(config).unwrap();
        
        assert!(!model.is_fitted());
        
        // Note: Actual fitting would require more data and proper training
        // This is just testing the interface
        let validation = model.validate_input(&data);
        assert!(validation.is_ok());
    }
}