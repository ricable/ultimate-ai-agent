//! Model training and evaluation

pub mod evaluator;
pub mod trainer;

pub use evaluator::ModelEvaluator;
pub use trainer::{ModelTrainer, TrainingConfig};
