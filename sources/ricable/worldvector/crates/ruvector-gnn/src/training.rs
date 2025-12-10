//! Training utilities for GNN models.
//!
//! Provides training loop utilities, optimizers, and loss functions.

use crate::error::{GnnError, Result};
use crate::search::cosine_similarity;
use ndarray::Array2;

/// Optimizer types
#[derive(Debug, Clone)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    Sgd {
        /// Learning rate
        learning_rate: f32,
        /// Momentum coefficient (0.0 = no momentum, 0.9 = standard)
        momentum: f32,
    },
    /// Adam optimizer
    Adam {
        /// Learning rate
        learning_rate: f32,
        /// Beta1 parameter (exponential decay rate for first moment)
        beta1: f32,
        /// Beta2 parameter (exponential decay rate for second moment)
        beta2: f32,
        /// Epsilon for numerical stability
        epsilon: f32,
    },
}

/// Optimizer state storage
#[derive(Debug)]
enum OptimizerState {
    /// SGD with momentum state
    Sgd {
        /// Momentum buffer (velocity)
        velocity: Option<Array2<f32>>,
    },
    /// Adam optimizer state
    Adam {
        /// First moment estimate (mean of gradients)
        m: Option<Array2<f32>>,
        /// Second moment estimate (uncentered variance of gradients)
        v: Option<Array2<f32>>,
        /// Timestep counter
        t: usize,
    },
}

/// Optimizer for parameter updates
pub struct Optimizer {
    optimizer_type: OptimizerType,
    state: OptimizerState,
}

impl Optimizer {
    /// Create a new optimizer
    pub fn new(optimizer_type: OptimizerType) -> Self {
        let state = match &optimizer_type {
            OptimizerType::Sgd { .. } => OptimizerState::Sgd { velocity: None },
            OptimizerType::Adam { .. } => OptimizerState::Adam {
                m: None,
                v: None,
                t: 0,
            },
        };

        Self {
            optimizer_type,
            state,
        }
    }

    /// Perform optimization step
    ///
    /// Updates parameters in-place based on gradients using the configured optimizer.
    ///
    /// # Arguments
    /// * `params` - Parameters to update (modified in-place)
    /// * `grads` - Gradients for the parameters
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(GnnError)` if shapes don't match or other errors occur
    pub fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()> {
        // Validate shapes match
        if params.shape() != grads.shape() {
            return Err(GnnError::dimension_mismatch(
                format!("{:?}", params.shape()),
                format!("{:?}", grads.shape()),
            ));
        }

        match (&self.optimizer_type, &mut self.state) {
            (OptimizerType::Sgd { learning_rate, momentum }, OptimizerState::Sgd { velocity }) => {
                Self::sgd_step_with_momentum(params, grads, *learning_rate, *momentum, velocity)
            }
            (
                OptimizerType::Adam {
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                },
                OptimizerState::Adam { m, v, t },
            ) => Self::adam_step(params, grads, *learning_rate, *beta1, *beta2, *epsilon, m, v, t),
            _ => {
                return Err(GnnError::invalid_input(
                    "Optimizer type and state mismatch",
                ))
            }
        }
    }

    /// SGD optimization step with momentum
    ///
    /// Implements: v_t = momentum * v_{t-1} + learning_rate * grad
    ///             params = params - v_t
    fn sgd_step_with_momentum(
        params: &mut Array2<f32>,
        grads: &Array2<f32>,
        learning_rate: f32,
        momentum: f32,
        velocity: &mut Option<Array2<f32>>,
    ) -> Result<()> {
        if momentum == 0.0 {
            // Simple SGD without momentum
            *params -= &(grads * learning_rate);
        } else {
            // SGD with momentum
            if velocity.is_none() {
                // Initialize velocity buffer
                *velocity = Some(Array2::zeros(params.dim()));
            }

            if let Some(v) = velocity {
                // Update velocity: v = momentum * v + learning_rate * grad
                let new_velocity = v.mapv(|x| x * momentum) + grads * learning_rate;
                *v = new_velocity;

                // Update parameters: params = params - v
                *params -= &*v;
            }
        }

        Ok(())
    }

    /// Adam optimization step
    ///
    /// Implements the Adam algorithm:
    /// 1. m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    /// 2. v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    /// 3. m_hat = m_t / (1 - beta1^t)
    /// 4. v_hat = v_t / (1 - beta2^t)
    /// 5. params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
    #[allow(clippy::too_many_arguments)]
    fn adam_step(
        params: &mut Array2<f32>,
        grads: &Array2<f32>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        m: &mut Option<Array2<f32>>,
        v: &mut Option<Array2<f32>>,
        t: &mut usize,
    ) -> Result<()> {
        // Initialize moment buffers if needed
        if m.is_none() {
            *m = Some(Array2::zeros(params.dim()));
        }
        if v.is_none() {
            *v = Some(Array2::zeros(params.dim()));
        }

        // Increment timestep
        *t += 1;
        let timestep = *t as f32;

        if let (Some(m_buf), Some(v_buf)) = (m, v) {
            // Update biased first moment estimate
            // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            let new_m = m_buf.mapv(|x| x * beta1) + grads * (1.0 - beta1);
            *m_buf = new_m;

            // Update biased second raw moment estimate
            // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            let grads_squared = grads.mapv(|x| x * x);
            let new_v = v_buf.mapv(|x| x * beta2) + grads_squared * (1.0 - beta2);
            *v_buf = new_v;

            // Compute bias-corrected first moment estimate
            // m_hat = m_t / (1 - beta1^t)
            let bias_correction1 = 1.0 - beta1.powi(*t as i32);
            let m_hat = m_buf.mapv(|x| x / bias_correction1);

            // Compute bias-corrected second raw moment estimate
            // v_hat = v_t / (1 - beta2^t)
            let bias_correction2 = 1.0 - beta2.powi(*t as i32);
            let v_hat = v_buf.mapv(|x| x / bias_correction2);

            // Update parameters
            // params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
            let update = m_hat.iter().zip(v_hat.iter()).map(|(&m_val, &v_val)| {
                learning_rate * m_val / (v_val.sqrt() + epsilon)
            });

            for (param, upd) in params.iter_mut().zip(update) {
                *param -= upd;
            }
        }

        Ok(())
    }
}

/// Loss function types
#[derive(Debug, Clone, Copy)]
pub enum LossType {
    /// Mean Squared Error
    Mse,
    /// Cross Entropy
    CrossEntropy,
    /// Binary Cross Entropy
    BinaryCrossEntropy,
}

/// TODO: Implement loss functions
pub struct Loss;

impl Loss {
    /// TODO: Compute loss
    pub fn compute(
        loss_type: LossType,
        predictions: &Array2<f32>,
        targets: &Array2<f32>,
    ) -> Result<f32> {
        unimplemented!("TODO: Implement loss computation")
    }

    /// TODO: Compute loss gradient
    pub fn gradient(
        loss_type: LossType,
        predictions: &Array2<f32>,
        targets: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        unimplemented!("TODO: Implement loss gradient")
    }
}

/// TODO: Implement training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Loss type
    pub loss_type: LossType,
    /// Optimizer type
    pub optimizer_type: OptimizerType,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            loss_type: LossType::Mse,
            optimizer_type: OptimizerType::Adam {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        }
    }
}

/// Configuration for contrastive learning training
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Batch size for training
    pub batch_size: usize,
    /// Number of negative samples per positive
    pub n_negatives: usize,
    /// Temperature parameter for contrastive loss
    pub temperature: f32,
    /// Learning rate for optimization
    pub learning_rate: f32,
    /// Number of updates before flushing to storage
    pub flush_threshold: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 256,
            n_negatives: 64,
            temperature: 0.07,
            learning_rate: 0.001,
            flush_threshold: 1000,
        }
    }
}

/// Configuration for online learning
#[derive(Debug, Clone)]
pub struct OnlineConfig {
    /// Number of local optimization steps
    pub local_steps: usize,
    /// Whether to propagate updates to neighbors
    pub propagate_updates: bool,
}

impl Default for OnlineConfig {
    fn default() -> Self {
        Self {
            local_steps: 5,
            propagate_updates: true,
        }
    }
}

/// Compute InfoNCE contrastive loss
///
/// InfoNCE (Information Noise-Contrastive Estimation) loss is used for contrastive learning.
/// It maximizes agreement between anchor and positive samples while minimizing agreement
/// with negative samples.
///
/// # Arguments
/// * `anchor` - The anchor embedding vector
/// * `positives` - Positive example embeddings (similar to anchor)
/// * `negatives` - Negative example embeddings (dissimilar to anchor)
/// * `temperature` - Temperature scaling parameter (lower = sharper distinctions)
///
/// # Returns
/// * The computed loss value (lower is better)
///
/// # Example
/// ```
/// use ruvector_gnn::training::info_nce_loss;
///
/// let anchor = vec![1.0, 0.0, 0.0];
/// let positive = vec![0.9, 0.1, 0.0];
/// let negative1 = vec![0.0, 1.0, 0.0];
/// let negative2 = vec![0.0, 0.0, 1.0];
///
/// let loss = info_nce_loss(
///     &anchor,
///     &[&positive],
///     &[&negative1, &negative2],
///     0.07
/// );
/// assert!(loss > 0.0);
/// ```
pub fn info_nce_loss(
    anchor: &[f32],
    positives: &[&[f32]],
    negatives: &[&[f32]],
    temperature: f32,
) -> f32 {
    if positives.is_empty() {
        return 0.0;
    }

    // Compute similarities with positives (scaled by temperature)
    let pos_sims: Vec<f32> = positives
        .iter()
        .map(|pos| cosine_similarity(anchor, pos) / temperature)
        .collect();

    // Compute similarities with negatives (scaled by temperature)
    let neg_sims: Vec<f32> = negatives
        .iter()
        .map(|neg| cosine_similarity(anchor, neg) / temperature)
        .collect();

    // For each positive, compute the InfoNCE loss using log-sum-exp trick for numerical stability
    let mut total_loss = 0.0;
    for &pos_sim in &pos_sims {
        // Use log-sum-exp trick to avoid overflow
        // log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
        // = pos_sim - log(exp(pos_sim) + sum(exp(neg_sim)))
        // = pos_sim - log_sum_exp([pos_sim, neg_sims...])

        // Collect all logits for log-sum-exp
        let mut all_logits = vec![pos_sim];
        all_logits.extend(&neg_sims);

        // Compute log-sum-exp with numerical stability
        let max_logit = all_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = max_logit
            + all_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .sum::<f32>()
                .ln();

        // Loss = -log(exp(pos_sim) / sum_exp) = -(pos_sim - log_sum_exp)
        total_loss -= pos_sim - log_sum_exp;
    }

    // Average over positives
    total_loss / positives.len() as f32
}

/// Compute local contrastive loss for graph structures
///
/// This loss encourages node embeddings to be similar to their neighbors
/// and dissimilar to non-neighbors in the graph.
///
/// # Arguments
/// * `node_embedding` - The embedding of the target node
/// * `neighbor_embeddings` - Embeddings of neighbor nodes
/// * `non_neighbor_embeddings` - Embeddings of non-neighbor nodes
/// * `temperature` - Temperature scaling parameter
///
/// # Returns
/// * The computed loss value (lower is better)
///
/// # Example
/// ```
/// use ruvector_gnn::training::local_contrastive_loss;
///
/// let node = vec![1.0, 0.0, 0.0];
/// let neighbor = vec![0.9, 0.1, 0.0];
/// let non_neighbor1 = vec![0.0, 1.0, 0.0];
/// let non_neighbor2 = vec![0.0, 0.0, 1.0];
///
/// let loss = local_contrastive_loss(
///     &node,
///     &[neighbor],
///     &[non_neighbor1, non_neighbor2],
///     0.07
/// );
/// assert!(loss > 0.0);
/// ```
pub fn local_contrastive_loss(
    node_embedding: &[f32],
    neighbor_embeddings: &[Vec<f32>],
    non_neighbor_embeddings: &[Vec<f32>],
    temperature: f32,
) -> f32 {
    if neighbor_embeddings.is_empty() {
        return 0.0;
    }

    // Convert to slices for info_nce_loss
    let positives: Vec<&[f32]> = neighbor_embeddings.iter().map(|v| v.as_slice()).collect();
    let negatives: Vec<&[f32]> = non_neighbor_embeddings
        .iter()
        .map(|v| v.as_slice())
        .collect();

    info_nce_loss(node_embedding, &positives, &negatives, temperature)
}

/// Perform a single SGD (Stochastic Gradient Descent) optimization step
///
/// Updates the embedding in-place by subtracting the scaled gradient.
///
/// # Arguments
/// * `embedding` - The embedding to update (modified in-place)
/// * `grad` - The gradient vector
/// * `learning_rate` - The learning rate (step size)
///
/// # Example
/// ```
/// use ruvector_gnn::training::sgd_step;
///
/// let mut embedding = vec![1.0, 2.0, 3.0];
/// let gradient = vec![0.1, -0.2, 0.3];
/// let learning_rate = 0.01;
///
/// sgd_step(&mut embedding, &gradient, learning_rate);
///
/// // Embedding is now updated: embedding[i] -= learning_rate * grad[i]
/// assert!((embedding[0] - 0.999).abs() < 1e-6);
/// assert!((embedding[1] - 2.002).abs() < 1e-6);
/// assert!((embedding[2] - 2.997).abs() < 1e-6);
/// ```
pub fn sgd_step(embedding: &mut [f32], grad: &[f32], learning_rate: f32) {
    assert_eq!(
        embedding.len(),
        grad.len(),
        "Embedding and gradient must have the same length"
    );

    for (emb, &g) in embedding.iter_mut().zip(grad.iter()) {
        *emb -= learning_rate * g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_config_default() {
        let config = TrainConfig::default();
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.n_negatives, 64);
        assert_eq!(config.temperature, 0.07);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.flush_threshold, 1000);
    }

    #[test]
    fn test_online_config_default() {
        let config = OnlineConfig::default();
        assert_eq!(config.local_steps, 5);
        assert!(config.propagate_updates);
    }

    #[test]
    fn test_info_nce_loss_basic() {
        // Anchor and positive are similar
        let anchor = vec![1.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0];

        // Negatives are orthogonal
        let negative1 = vec![0.0, 1.0, 0.0];
        let negative2 = vec![0.0, 0.0, 1.0];

        let loss = info_nce_loss(&anchor, &[&positive], &[&negative1, &negative2], 0.07);

        // Loss should be positive
        assert!(loss > 0.0);

        // Loss should be reasonable (not infinite or NaN)
        assert!(loss.is_finite());
    }

    #[test]
    fn test_info_nce_loss_perfect_match() {
        // Anchor and positive are identical
        let anchor = vec![1.0, 0.0, 0.0];
        let positive = vec![1.0, 0.0, 0.0];

        // Negatives are very different
        let negative1 = vec![0.0, 1.0, 0.0];
        let negative2 = vec![0.0, 0.0, 1.0];

        let loss = info_nce_loss(&anchor, &[&positive], &[&negative1, &negative2], 0.07);

        // Loss should be lower for perfect match
        assert!(loss < 1.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_info_nce_loss_no_positives() {
        let anchor = vec![1.0, 0.0, 0.0];
        let negative1 = vec![0.0, 1.0, 0.0];

        let loss = info_nce_loss(&anchor, &[], &[&negative1], 0.07);

        // Should return 0.0 when no positives
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_info_nce_loss_temperature_effect() {
        let anchor = vec![1.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0];
        let negative = vec![0.0, 1.0, 0.0];

        // Test with reasonable temperature values
        // Very low temperatures can cause numerical issues, so we use 0.07 (standard) and 1.0
        let loss_low_temp = info_nce_loss(&anchor, &[&positive], &[&negative], 0.07);
        let loss_high_temp = info_nce_loss(&anchor, &[&positive], &[&negative], 1.0);

        // Both should be positive and finite
        assert!(
            loss_low_temp > 0.0 && loss_low_temp.is_finite(),
            "Low temp loss should be positive and finite, got: {}",
            loss_low_temp
        );
        assert!(
            loss_high_temp > 0.0 && loss_high_temp.is_finite(),
            "High temp loss should be positive and finite, got: {}",
            loss_high_temp
        );

        // With standard temperature, the loss should be reasonable
        assert!(loss_low_temp < 10.0, "Loss should not be too large");
        assert!(loss_high_temp < 10.0, "Loss should not be too large");
    }

    #[test]
    fn test_local_contrastive_loss_basic() {
        let node = vec![1.0, 0.0, 0.0];
        let neighbor = vec![0.9, 0.1, 0.0];
        let non_neighbor1 = vec![0.0, 1.0, 0.0];
        let non_neighbor2 = vec![0.0, 0.0, 1.0];

        let loss =
            local_contrastive_loss(&node, &[neighbor], &[non_neighbor1, non_neighbor2], 0.07);

        // Loss should be positive and finite
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_local_contrastive_loss_multiple_neighbors() {
        let node = vec![1.0, 0.0, 0.0];
        let neighbor1 = vec![0.9, 0.1, 0.0];
        let neighbor2 = vec![0.95, 0.05, 0.0];
        let non_neighbor = vec![0.0, 1.0, 0.0];

        let loss = local_contrastive_loss(&node, &[neighbor1, neighbor2], &[non_neighbor], 0.07);

        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_local_contrastive_loss_no_neighbors() {
        let node = vec![1.0, 0.0, 0.0];
        let non_neighbor = vec![0.0, 1.0, 0.0];

        let loss = local_contrastive_loss(&node, &[], &[non_neighbor], 0.07);

        // Should return 0.0 when no neighbors
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_sgd_step_basic() {
        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradient = vec![0.1, -0.2, 0.3];
        let learning_rate = 0.01;

        sgd_step(&mut embedding, &gradient, learning_rate);

        // Expected: embedding[i] -= learning_rate * grad[i]
        assert!((embedding[0] - 0.999).abs() < 1e-6); // 1.0 - 0.01 * 0.1
        assert!((embedding[1] - 2.002).abs() < 1e-6); // 2.0 - 0.01 * (-0.2)
        assert!((embedding[2] - 2.997).abs() < 1e-6); // 3.0 - 0.01 * 0.3
    }

    #[test]
    fn test_sgd_step_zero_gradient() {
        let mut embedding = vec![1.0, 2.0, 3.0];
        let original = embedding.clone();
        let gradient = vec![0.0, 0.0, 0.0];
        let learning_rate = 0.01;

        sgd_step(&mut embedding, &gradient, learning_rate);

        // Embedding should not change with zero gradient
        assert_eq!(embedding, original);
    }

    #[test]
    fn test_sgd_step_zero_learning_rate() {
        let mut embedding = vec![1.0, 2.0, 3.0];
        let original = embedding.clone();
        let gradient = vec![0.1, 0.2, 0.3];
        let learning_rate = 0.0;

        sgd_step(&mut embedding, &gradient, learning_rate);

        // Embedding should not change with zero learning rate
        assert_eq!(embedding, original);
    }

    #[test]
    fn test_sgd_step_large_learning_rate() {
        let mut embedding = vec![10.0, 20.0, 30.0];
        let gradient = vec![1.0, 2.0, 3.0];
        let learning_rate = 5.0;

        sgd_step(&mut embedding, &gradient, learning_rate);

        // Expected: embedding[i] -= learning_rate * grad[i]
        assert!((embedding[0] - 5.0).abs() < 1e-5); // 10.0 - 5.0 * 1.0
        assert!((embedding[1] - 10.0).abs() < 1e-5); // 20.0 - 5.0 * 2.0
        assert!((embedding[2] - 15.0).abs() < 1e-5); // 30.0 - 5.0 * 3.0
    }

    #[test]
    #[should_panic(expected = "Embedding and gradient must have the same length")]
    fn test_sgd_step_mismatched_lengths() {
        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradient = vec![0.1, 0.2]; // Wrong length

        sgd_step(&mut embedding, &gradient, 0.01);
    }

    #[test]
    fn test_info_nce_loss_multiple_positives() {
        let anchor = vec![1.0, 0.0, 0.0];
        let positive1 = vec![0.9, 0.1, 0.0];
        let positive2 = vec![0.95, 0.05, 0.0];
        let negative = vec![0.0, 1.0, 0.0];

        let loss = info_nce_loss(&anchor, &[&positive1, &positive2], &[&negative], 0.07);

        // Loss should be positive and finite
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_contrastive_loss_gradient_property() {
        // Test that loss decreases when positive becomes more similar
        let anchor = vec![1.0, 0.0, 0.0];
        let positive_far = vec![0.5, 0.5, 0.0];
        let positive_close = vec![0.9, 0.1, 0.0];
        let negative = vec![0.0, 1.0, 0.0];

        let loss_far = info_nce_loss(&anchor, &[&positive_far], &[&negative], 0.07);
        let loss_close = info_nce_loss(&anchor, &[&positive_close], &[&negative], 0.07);

        // Loss should be lower when positive is closer to anchor
        assert!(loss_close < loss_far);
    }

    #[test]
    fn test_sgd_optimizer_basic() {
        let optimizer_type = OptimizerType::Sgd {
            learning_rate: 0.1,
            momentum: 0.0,
        };
        let mut optimizer = Optimizer::new(optimizer_type);

        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        let result = optimizer.step(&mut params, &grads);
        assert!(result.is_ok());

        // Expected: params[i] -= learning_rate * grads[i]
        assert!((params[[0, 0]] - 0.99).abs() < 1e-6); // 1.0 - 0.1 * 0.1
        assert!((params[[0, 1]] - 1.98).abs() < 1e-6); // 2.0 - 0.1 * 0.2
        assert!((params[[1, 0]] - 2.97).abs() < 1e-6); // 3.0 - 0.1 * 0.3
        assert!((params[[1, 1]] - 3.96).abs() < 1e-6); // 4.0 - 0.1 * 0.4
    }

    #[test]
    fn test_sgd_optimizer_with_momentum() {
        let optimizer_type = OptimizerType::Sgd {
            learning_rate: 0.1,
            momentum: 0.9,
        };
        let mut optimizer = Optimizer::new(optimizer_type);

        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        // First step
        let result = optimizer.step(&mut params, &grads);
        assert!(result.is_ok());

        // First step should be same as SGD without momentum (velocity starts at 0)
        assert!((params[[0, 0]] - 0.99).abs() < 1e-6);

        // Second step should use accumulated momentum
        let result = optimizer.step(&mut params, &grads);
        assert!(result.is_ok());

        // With momentum, the update should be larger
        assert!(params[[0, 0]] < 0.99);
    }

    #[test]
    fn test_adam_optimizer_basic() {
        let optimizer_type = OptimizerType::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut optimizer = Optimizer::new(optimizer_type);

        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        let original_params = params.clone();
        let result = optimizer.step(&mut params, &grads);
        assert!(result.is_ok());

        // Parameters should be updated (decreased in the direction of gradients)
        assert!(params[[0, 0]] < original_params[[0, 0]]);
        assert!(params[[0, 1]] < original_params[[0, 1]]);
        assert!(params[[1, 0]] < original_params[[1, 0]]);
        assert!(params[[1, 1]] < original_params[[1, 1]]);

        // Check that all values are finite
        assert!(params.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_adam_optimizer_multiple_steps() {
        let optimizer_type = OptimizerType::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut optimizer = Optimizer::new(optimizer_type);

        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let initial_params = params.clone();

        // Perform multiple steps
        for _ in 0..10 {
            let result = optimizer.step(&mut params, &grads);
            assert!(result.is_ok());
            assert!(params.iter().all(|&x| x.is_finite()));
        }

        // After multiple steps, parameters should have decreased (gradients are positive)
        assert!(params[[0, 0]] < initial_params[[0, 0]]);
        assert!(params[[1, 1]] < initial_params[[1, 1]]);
        // All parameters should have moved
        for i in 0..2 {
            for j in 0..2 {
                assert!(params[[i, j]] < initial_params[[i, j]]);
            }
        }
    }

    #[test]
    fn test_adam_bias_correction() {
        let optimizer_type = OptimizerType::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut optimizer = Optimizer::new(optimizer_type.clone());

        let mut params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let grads = Array2::from_shape_vec((1, 1), vec![0.1]).unwrap();

        // First step should have strong bias correction
        let result = optimizer.step(&mut params, &grads);
        assert!(result.is_ok());
        let first_update = 1.0 - params[[0, 0]];

        // Reset optimizer
        let mut optimizer = Optimizer::new(optimizer_type);
        let mut params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        // Perform 100 steps, last step should have less bias correction effect
        for _ in 0..100 {
            let _ = optimizer.step(&mut params, &grads);
        }

        // The bias correction effect should diminish over time
        assert!(first_update > 0.0);
    }

    #[test]
    fn test_optimizer_shape_mismatch() {
        let optimizer_type = OptimizerType::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut optimizer = Optimizer::new(optimizer_type);

        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

        let result = optimizer.step(&mut params, &grads);
        assert!(result.is_err());
        if let Err(GnnError::DimensionMismatch { expected, actual }) = result {
            assert!(expected.contains("2, 2"));
            assert!(actual.contains("3, 2"));
        } else {
            panic!("Expected DimensionMismatch error");
        }
    }

    #[test]
    fn test_adam_convergence() {
        // Test that Adam can minimize a simple quadratic function
        let optimizer_type = OptimizerType::Adam {
            learning_rate: 0.5,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut optimizer = Optimizer::new(optimizer_type);

        // Start with params far from optimum (0, 0)
        let mut params = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();

        // Gradient of f(x, y) = x^2 + y^2 is (2x, 2y)
        for _ in 0..200 {
            let grads =
                Array2::from_shape_vec((1, 2), vec![2.0 * params[[0, 0]], 2.0 * params[[0, 1]]])
                    .unwrap();
            let _ = optimizer.step(&mut params, &grads);
        }

        // Should converge close to (0, 0)
        assert!(params[[0, 0]].abs() < 0.5);
        assert!(params[[0, 1]].abs() < 0.5);
    }

    #[test]
    fn test_sgd_momentum_convergence() {
        // Test that SGD with momentum can minimize a simple quadratic function
        let optimizer_type = OptimizerType::Sgd {
            learning_rate: 0.01,
            momentum: 0.9,
        };
        let mut optimizer = Optimizer::new(optimizer_type);

        // Start with params far from optimum (0, 0)
        let mut params = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();

        // Gradient of f(x, y) = x^2 + y^2 is (2x, 2y)
        for _ in 0..200 {
            let grads =
                Array2::from_shape_vec((1, 2), vec![2.0 * params[[0, 0]], 2.0 * params[[0, 1]]])
                    .unwrap();
            let _ = optimizer.step(&mut params, &grads);
        }

        // Should converge close to (0, 0)
        assert!(params[[0, 0]].abs() < 0.5);
        assert!(params[[0, 1]].abs() < 0.5);
    }
}
