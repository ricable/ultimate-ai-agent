//! # ruvector-attention
//!
//! Attention mechanisms for ruvector, including geometric, graph, and sparse attention.
//!
//! This crate provides efficient implementations of various attention mechanisms:
//! - Scaled dot-product attention
//! - Multi-head attention with parallel processing
//! - Graph attention for GNN applications
//! - Geometric attention in hyperbolic spaces
//! - Sparse attention patterns
//!
//! ## Features
//!
//! - **SIMD Acceleration**: Optional SIMD optimizations for performance
//! - **Parallel Processing**: Rayon-based parallel head computation
//! - **WASM Support**: WebAssembly compilation support
//! - **NAPI Bindings**: Node.js bindings for JavaScript integration
//!
//! ## Example
//!
//! ```rust
//! use ruvector_attention::{
//!     attention::ScaledDotProductAttention,
//!     traits::Attention,
//! };
//!
//! // Create scaled dot-product attention
//! let attention = ScaledDotProductAttention::new(512);
//!
//! // Prepare inputs
//! let query = vec![1.0; 512];
//! let keys = vec![vec![0.5; 512], vec![0.3; 512]];
//! let values = vec![vec![1.0; 512], vec![2.0; 512]];
//!
//! let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
//! let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
//!
//! // Compute attention
//! let output = attention.compute(&query, &keys_refs, &values_refs).unwrap();
//! assert_eq!(output.len(), 512);
//! ```

pub mod attention;
pub mod config;
pub mod error;
pub mod traits;
pub mod utils;
pub mod hyperbolic;
pub mod sparse;
pub mod moe;
pub mod graph;
pub mod training;
pub mod sdk;

// Re-export main types
pub use attention::{MultiHeadAttention, ScaledDotProductAttention};
pub use config::{AttentionConfig, GraphAttentionConfig, SparseAttentionConfig};
pub use error::{AttentionError, AttentionResult};
pub use traits::{
    Attention, EdgeInfo, GeometricAttention, Gradients, GraphAttention, SparseAttention,
    SparseMask, TrainableAttention,
};
pub use hyperbolic::{
    poincare_distance, mobius_add, exp_map, log_map, project_to_ball,
    HyperbolicAttention, HyperbolicAttentionConfig,
    MixedCurvatureAttention, MixedCurvatureConfig,
};

// Sparse attention exports
pub use sparse::{
    SparseMaskBuilder, AttentionMask,
    LocalGlobalAttention, LinearAttention, FlashAttention,
};

// MoE exports
pub use moe::{
    MoEAttention, MoEConfig,
    Expert, ExpertType, StandardExpert, HyperbolicExpert, LinearExpert,
    Router, LearnedRouter, TopKRouting,
};

// Graph attention exports
pub use graph::{
    EdgeFeaturedAttention, EdgeFeaturedConfig,
    GraphRoPE, RoPEConfig,
    DualSpaceAttention, DualSpaceConfig,
};

// Training exports
pub use training::{
    Loss, InfoNCELoss, LocalContrastiveLoss, SpectralRegularization, Reduction,
    Optimizer, SGD, Adam, AdamW,
    CurriculumScheduler, CurriculumStage, TemperatureAnnealing, DecayType,
    NegativeMiner, HardNegativeMiner, MiningStrategy,
};

// SDK exports
pub use sdk::{AttentionBuilder, AttentionPipeline, presets};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_basic_attention_workflow() {
        let config = AttentionConfig::builder()
            .dim(64)
            .num_heads(4)
            .build()
            .unwrap();

        assert_eq!(config.dim, 64);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.head_dim(), 16);
    }
}
