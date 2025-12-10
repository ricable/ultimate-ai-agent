//! Hyperbolic Attention Module
//!
//! Implements attention mechanisms in hyperbolic space using the Poincaré ball model.

pub mod poincare;
pub mod hyperbolic_attention;
pub mod mixed_curvature;

pub use poincare::{
    poincare_distance,
    mobius_add,
    mobius_scalar_mult,
    exp_map,
    log_map,
    project_to_ball,
    frechet_mean,
};

pub use hyperbolic_attention::{
    HyperbolicAttention,
    HyperbolicAttentionConfig,
};

pub use mixed_curvature::{
    MixedCurvatureAttention,
    MixedCurvatureConfig,
};
