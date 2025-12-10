//! Sparse attention mechanisms for efficient computation on long sequences
//!
//! This module provides sparse attention patterns that reduce complexity from O(n²) to sub-quadratic.

pub mod mask;
pub mod local_global;
pub mod linear;
pub mod flash;

pub use mask::{SparseMaskBuilder, AttentionMask};
pub use local_global::LocalGlobalAttention;
pub use linear::LinearAttention;
pub use flash::FlashAttention;
