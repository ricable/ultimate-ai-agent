//! # Immune Analyzer
//!
//! Differential expression and immune pathway analysis for CRISPR-Cas13.

pub mod deseq;
pub mod error;
pub mod normalization;
pub mod pathways;

pub use error::{AnalysisError, Result};
