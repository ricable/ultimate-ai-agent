//! Differential Privacy Engine
//!
//! Implements differential privacy mechanisms for protecting sensitive genomic data.

use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum PrivacyError {
    #[error("Insufficient privacy budget: requested {requested}, available {available}")]
    InsufficientBudget { requested: f64, available: f64 },
    #[error("Invalid privacy parameter: {0}")]
    InvalidParameter(String),
    #[error("Anonymization failed: {0}")]
    AnonymizationError(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivacyBudget {
    pub user_id: String,
    pub total_epsilon: f64,
    pub consumed_epsilon: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivacyConfig {
    pub default_epsilon: f64,
    pub default_delta: f64,
    pub sensitivity: f64,
    pub noise_mechanism: NoiseMechanism,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NoiseMechanism {
    Laplace,
    Gaussian,
    Exponential,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            default_epsilon: 1.0,
            default_delta: 1e-5,
            sensitivity: 1.0,
            noise_mechanism: NoiseMechanism::Laplace,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PrivateData {
    pub data: serde_json::Value,
    pub budget_consumed: f64,
    pub mechanism_used: String,
}

pub struct PrivacyEngine {
    config: PrivacyConfig,
    budgets: Arc<RwLock<std::collections::HashMap<String, PrivacyBudget>>>,
}

impl PrivacyEngine {
    pub fn new(config: PrivacyConfig) -> Self {
        Self {
            config,
            budgets: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Initialize privacy budget for a user
    pub async fn initialize_budget(&self, user_id: String, total_epsilon: f64) -> anyhow::Result<()> {
        let budget = PrivacyBudget {
            user_id: user_id.clone(),
            total_epsilon,
            consumed_epsilon: 0.0,
            last_updated: chrono::Utc::now(),
        };

        self.budgets.write().await.insert(user_id.clone(), budget);
        info!("Initialized privacy budget for user: {}, epsilon: {}", user_id, total_epsilon);
        Ok(())
    }

    /// Check available privacy budget
    pub async fn check_budget(&self, user_id: &str) -> Option<f64> {
        self.budgets
            .read()
            .await
            .get(user_id)
            .map(|b| b.total_epsilon - b.consumed_epsilon)
    }

    /// Apply differential privacy noise to data
    pub async fn apply_noise(
        &self,
        data: &serde_json::Value,
        epsilon: f64,
    ) -> Result<PrivateData, PrivacyError> {
        debug!("Applying differential privacy with epsilon: {}", epsilon);

        if epsilon <= 0.0 || epsilon > 10.0 {
            return Err(PrivacyError::InvalidParameter(
                format!("Epsilon must be between 0 and 10, got {}", epsilon),
            ));
        }

        let noisy_data = match self.config.noise_mechanism {
            NoiseMechanism::Laplace => self.add_laplace_noise(data, epsilon)?,
            NoiseMechanism::Gaussian => self.add_gaussian_noise(data, epsilon)?,
            NoiseMechanism::Exponential => self.add_exponential_noise(data, epsilon)?,
        };

        Ok(PrivateData {
            data: noisy_data,
            budget_consumed: epsilon,
            mechanism_used: format!("{:?}", self.config.noise_mechanism),
        })
    }

    /// Add Laplace noise (satisfies epsilon-differential privacy)
    fn add_laplace_noise(
        &self,
        data: &serde_json::Value,
        epsilon: f64,
    ) -> Result<serde_json::Value, PrivacyError> {
        let scale = self.config.sensitivity / epsilon;
        let mut rng = rand::thread_rng();

        let noisy_data = match data {
            serde_json::Value::Number(n) => {
                if let Some(val) = n.as_f64() {
                    let noise: f64 = self.sample_laplace(scale, &mut rng);
                    serde_json::json!(val + noise)
                } else {
                    data.clone()
                }
            }
            serde_json::Value::Array(arr) => {
                let noisy_arr: Vec<_> = arr
                    .iter()
                    .map(|v| self.add_laplace_noise(v, epsilon))
                    .collect::<Result<_, _>>()?;
                serde_json::json!(noisy_arr)
            }
            serde_json::Value::Object(obj) => {
                let mut noisy_obj = serde_json::Map::new();
                for (key, value) in obj {
                    noisy_obj.insert(key.clone(), self.add_laplace_noise(value, epsilon)?);
                }
                serde_json::json!(noisy_obj)
            }
            _ => data.clone(),
        };

        Ok(noisy_data)
    }

    /// Add Gaussian noise (satisfies (epsilon, delta)-differential privacy)
    fn add_gaussian_noise(
        &self,
        data: &serde_json::Value,
        epsilon: f64,
    ) -> Result<serde_json::Value, PrivacyError> {
        let sigma = self.config.sensitivity * (2.0 * (1.25 / self.config.default_delta).ln()).sqrt() / epsilon;
        let normal = Normal::new(0.0, sigma)
            .map_err(|e| PrivacyError::InvalidParameter(format!("Invalid normal distribution: {}", e)))?;
        let mut rng = rand::thread_rng();

        let noisy_data = match data {
            serde_json::Value::Number(n) => {
                if let Some(val) = n.as_f64() {
                    let noise = normal.sample(&mut rng);
                    serde_json::json!(val + noise)
                } else {
                    data.clone()
                }
            }
            _ => data.clone(), // Simplified for other types
        };

        Ok(noisy_data)
    }

    /// Add exponential noise
    fn add_exponential_noise(
        &self,
        data: &serde_json::Value,
        epsilon: f64,
    ) -> Result<serde_json::Value, PrivacyError> {
        // Exponential mechanism implementation
        // Simplified for demonstration
        self.add_laplace_noise(data, epsilon)
    }

    /// Sample from Laplace distribution
    fn sample_laplace(&self, scale: f64, rng: &mut impl rand::Rng) -> f64 {
        let u: f64 = rng.gen_range(-0.5..0.5);
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    /// Anonymize identifiable data
    pub async fn anonymize_data(&self, data: serde_json::Value) -> Result<serde_json::Value, PrivacyError> {
        debug!("Anonymizing data");

        // k-anonymity and l-diversity implementation
        // This is a simplified version - production would use proper anonymization
        let anonymized = match data {
            serde_json::Value::Object(mut obj) => {
                // Remove direct identifiers
                obj.remove("patient_id");
                obj.remove("sample_id");
                obj.remove("name");
                obj.remove("email");

                // Generalize quasi-identifiers
                if let Some(age) = obj.get("age").and_then(|v| v.as_i64()) {
                    let age_range = (age / 10) * 10;
                    obj.insert("age_range".to_string(), serde_json::json!(format!("{}-{}", age_range, age_range + 9)));
                    obj.remove("age");
                }

                serde_json::json!(obj)
            }
            _ => data,
        };

        Ok(anonymized)
    }

    /// Consume privacy budget
    pub async fn consume_budget(&self, user_id: &str, epsilon: f64) -> Result<(), PrivacyError> {
        let mut budgets = self.budgets.write().await;

        if let Some(budget) = budgets.get_mut(user_id) {
            let available = budget.total_epsilon - budget.consumed_epsilon;

            if epsilon > available {
                warn!("Insufficient privacy budget for user: {}", user_id);
                return Err(PrivacyError::InsufficientBudget {
                    requested: epsilon,
                    available,
                });
            }

            budget.consumed_epsilon += epsilon;
            budget.last_updated = chrono::Utc::now();

            info!("Consumed {} epsilon for user {}, remaining: {}",
                  epsilon, user_id, budget.total_epsilon - budget.consumed_epsilon);

            Ok(())
        } else {
            Err(PrivacyError::InvalidParameter(format!("No budget found for user: {}", user_id)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_privacy_budget_initialization() {
        let engine = PrivacyEngine::new(PrivacyConfig::default());
        engine.initialize_budget("user1".to_string(), 10.0).await.unwrap();

        let budget = engine.check_budget("user1").await;
        assert_eq!(budget, Some(10.0));
    }

    #[tokio::test]
    async fn test_laplace_noise_application() {
        let engine = PrivacyEngine::new(PrivacyConfig::default());
        let data = serde_json::json!(100.0);

        let result = engine.apply_noise(&data, 1.0).await.unwrap();
        assert_eq!(result.budget_consumed, 1.0);
    }

    #[test]
    fn test_anonymize_data() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let engine = PrivacyEngine::new(PrivacyConfig::default());
            let data = serde_json::json!({
                "patient_id": "12345",
                "name": "John Doe",
                "age": 35,
                "gene_expression": 100
            });

            let result = engine.anonymize_data(data).await.unwrap();
            assert!(!result.as_object().unwrap().contains_key("patient_id"));
            assert!(!result.as_object().unwrap().contains_key("name"));
        });
    }
}
