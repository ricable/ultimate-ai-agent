//! # Ruvector Core
//!
//! High-performance Rust-native vector database with HNSW indexing and SIMD-optimized operations.
//!
//! ## Features
//!
//! - **HNSW Indexing**: O(log n) search with 95%+ recall
//! - **SIMD Optimizations**: 4-16x faster distance calculations
//! - **Quantization**: 4-32x memory compression
//! - **Zero-copy Memory**: Memory-mapped vectors for instant loading
//! - **AgenticDB Compatible**: Drop-in replacement with 10-100x speedup

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod advanced_features;

// AgenticDB requires storage feature
#[cfg(feature = "storage")]
pub mod agenticdb;

pub mod distance;
pub mod error;
pub mod index;
pub mod quantization;

// Storage backends - conditional compilation based on features
#[cfg(feature = "storage")]
pub mod storage;

#[cfg(not(feature = "storage"))]
pub mod storage_memory;

#[cfg(not(feature = "storage"))]
pub use storage_memory as storage;

pub mod types;
pub mod vector_db;

// Performance optimization modules
pub mod arena;
pub mod cache_optimized;
pub mod lockfree;
pub mod simd_intrinsics;

/// Advanced techniques: hypergraphs, learned indexes, neural hashing, TDA (Phase 6)
pub mod advanced;

// Re-exports
pub use advanced_features::{
    ConformalConfig, ConformalPredictor, EnhancedPQ, FilterExpression, FilterStrategy,
    FilteredSearch, HybridConfig, HybridSearch, MMRConfig, MMRSearch, PQConfig, PredictionSet,
    BM25,
};

#[cfg(feature = "storage")]
pub use agenticdb::AgenticDB;

pub use error::{Result, RuvectorError};
pub use types::{DistanceMetric, SearchQuery, SearchResult, VectorEntry, VectorId};
pub use vector_db::VectorDB;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        // Verify version matches workspace - use dynamic check instead of hardcoded value
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty(), "Version should not be empty");
        assert!(version.starts_with("0.1."), "Version should be 0.1.x");
    }
}
