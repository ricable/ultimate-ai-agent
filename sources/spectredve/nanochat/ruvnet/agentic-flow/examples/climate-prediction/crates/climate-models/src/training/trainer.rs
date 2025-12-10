//! Model training orchestration

use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

use crate::{Result, ModelError};

/// Configuration for model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Optimizer type (adam, sgd, etc.)
    pub optimizer: String,

    /// Weight decay (L2 regularization)
    pub weight_decay: f64,

    /// Learning rate scheduler
    pub lr_scheduler: Option<LRScheduler>,

    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,

    /// Validation split ratio
    pub validation_split: f64,

    /// Random seed for reproducibility
    pub random_seed: u64,

    /// Device for training (cpu, cuda, etc.)
    pub device: String,

    /// Number of worker threads
    pub num_workers: usize,

    /// Save checkpoint every N epochs
    pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            optimizer: "adam".to_string(),
            weight_decay: 1e-5,
            lr_scheduler: Some(LRScheduler::ReduceLROnPlateau {
                factor: 0.5,
                patience: 10,
                min_lr: 1e-6,
            }),
            early_stopping_patience: Some(20),
            validation_split: 0.2,
            random_seed: 42,
            device: "cpu".to_string(),
            num_workers: 4,
            checkpoint_interval: 10,
        }
    }
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LRScheduler {
    /// Reduce LR when metric plateaus
    ReduceLROnPlateau {
        factor: f64,
        patience: usize,
        min_lr: f64,
    },
    /// Cosine annealing schedule
    CosineAnnealing {
        t_max: usize,
        eta_min: f64,
    },
    /// Step decay
    StepLR {
        step_size: usize,
        gamma: f64,
    },
    /// Exponential decay
    ExponentialLR {
        gamma: f64,
    },
}

/// Training statistics for one epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochStats {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub duration_secs: f64,
}

/// Model trainer for orchestrating the training process
///
/// This trainer coordinates with Python scripts for actual training,
/// as PyTorch provides better support for training complex models.
pub struct ModelTrainer {
    config: TrainingConfig,
    training_history: Vec<EpochStats>,
}

impl ModelTrainer {
    /// Create a new trainer with configuration
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            training_history: Vec::new(),
        }
    }

    /// Train an LSTM model using Python script
    pub async fn train_lstm(
        &mut self,
        data_path: &Path,
        output_path: &Path,
    ) -> Result<()> {
        info!("Starting LSTM training with config: {:?}", self.config);

        // Save training config
        let config_path = output_path.join("training_config.json");
        self.save_config(&config_path)?;

        // Call Python training script
        let status = tokio::process::Command::new("python3")
            .arg("python/train_lstm.py")
            .arg("--data")
            .arg(data_path)
            .arg("--output")
            .arg(output_path)
            .arg("--config")
            .arg(&config_path)
            .status()
            .await
            .map_err(|e| ModelError::TrainingFailed(format!("Failed to run Python script: {}", e)))?;

        if !status.success() {
            return Err(ModelError::TrainingFailed(
                format!("Python training script failed with status: {}", status)
            ));
        }

        // Load training history
        let history_path = output_path.join("training_history.json");
        self.load_history(&history_path)?;

        info!("LSTM training completed successfully");
        Ok(())
    }

    /// Train an FNO model using Python script
    pub async fn train_fno(
        &mut self,
        data_path: &Path,
        output_path: &Path,
    ) -> Result<()> {
        info!("Starting FNO training with config: {:?}", self.config);

        let config_path = output_path.join("training_config.json");
        self.save_config(&config_path)?;

        let status = tokio::process::Command::new("python3")
            .arg("python/train_fno.py")
            .arg("--data")
            .arg(data_path)
            .arg("--output")
            .arg(output_path)
            .arg("--config")
            .arg(&config_path)
            .status()
            .await
            .map_err(|e| ModelError::TrainingFailed(format!("Failed to run Python script: {}", e)))?;

        if !status.success() {
            return Err(ModelError::TrainingFailed(
                format!("Python training script failed with status: {}", status)
            ));
        }

        let history_path = output_path.join("training_history.json");
        self.load_history(&history_path)?;

        info!("FNO training completed successfully");
        Ok(())
    }

    /// Save training configuration to file
    fn save_config(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load training history from file
    fn load_history(&mut self, path: &Path) -> Result<()> {
        if !path.exists() {
            warn!("Training history file not found: {}", path.display());
            return Ok(());
        }

        let json = std::fs::read_to_string(path)?;
        self.training_history = serde_json::from_str(&json)?;
        Ok(())
    }

    /// Get the training history
    pub fn history(&self) -> &[EpochStats] {
        &self.training_history
    }

    /// Get the best epoch based on validation loss
    pub fn best_epoch(&self) -> Option<&EpochStats> {
        self.training_history
            .iter()
            .filter(|stats| stats.val_loss.is_some())
            .min_by(|a, b| {
                a.val_loss
                    .unwrap()
                    .partial_cmp(&b.val_loss.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Check if early stopping criteria is met
    pub fn should_stop_early(&self, current_epoch: usize) -> bool {
        if let Some(patience) = self.config.early_stopping_patience {
            if let Some(best) = self.best_epoch() {
                return current_epoch - best.epoch > patience;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.learning_rate, 0.001);
    }

    #[test]
    fn test_best_epoch_selection() {
        let mut trainer = ModelTrainer::new(TrainingConfig::default());

        trainer.training_history = vec![
            EpochStats {
                epoch: 0,
                train_loss: 1.0,
                val_loss: Some(1.2),
                learning_rate: 0.001,
                duration_secs: 10.0,
            },
            EpochStats {
                epoch: 1,
                train_loss: 0.8,
                val_loss: Some(0.9),
                learning_rate: 0.001,
                duration_secs: 10.0,
            },
            EpochStats {
                epoch: 2,
                train_loss: 0.6,
                val_loss: Some(1.0),
                learning_rate: 0.001,
                duration_secs: 10.0,
            },
        ];

        let best = trainer.best_epoch().unwrap();
        assert_eq!(best.epoch, 1);
        assert_eq!(best.val_loss.unwrap(), 0.9);
    }
}
