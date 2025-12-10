//! # ruvector-attention SDK
//!
//! High-level, ergonomic APIs for building attention mechanisms.

pub mod builder;
pub mod pipeline;
pub mod presets;

pub use builder::{AttentionBuilder, AttentionType, scaled_dot, multi_head, flash};
pub use pipeline::{AttentionPipeline, PipelineStage, NormType};
pub use presets::{AttentionPreset, for_sequences, for_graphs, for_large_scale};
