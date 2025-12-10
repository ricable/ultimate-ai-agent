//! Hyperbolic Attention Module
//!
//! Implements attention mechanisms in hyperbolic space using:
//! - Poincaré ball model (traditional)
//! - Lorentz hyperboloid model (novel - faster, more stable)

pub mod poincare;
pub mod hyperbolic_attention;
pub mod mixed_curvature;
pub mod lorentz_cascade;

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

// Novel Lorentz Cascade Attention (LCA)
pub use lorentz_cascade::{
    LorentzCascadeAttention,
    LCAConfig,
    CascadeHead,
    lorentz_distance,
    lorentz_inner,
    busemann_score,
    horosphere_attention_weights,
    einstein_midpoint,
    project_hyperboloid,
};
