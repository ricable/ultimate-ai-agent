#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Activation functions available for neurons
///
/// These functions are based on the FANN library's activation functions
/// and include both common neural network activation functions and
/// some specialized variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum ActivationFunction {
    /// Linear activation function: f(x) = x * steepness
    Linear,

    /// Threshold activation function: f(x) = 0 if x < 0, 1 if x >= 0
    /// Note: Cannot be used during training due to zero derivative
    Threshold,

    /// Symmetric threshold: f(x) = -1 if x < 0, 1 if x >= 0
    /// Note: Cannot be used during training due to zero derivative
    ThresholdSymmetric,

    /// Sigmoid activation function: f(x) = 1 / (1 + exp(-2 * steepness * x))
    /// Output range: (0, 1)
    #[default]
    Sigmoid,

    /// Symmetric sigmoid (tanh): f(x) = tanh(steepness * x)
    /// Output range: (-1, 1)
    SigmoidSymmetric,

    /// Hyperbolic tangent: alias for SigmoidSymmetric
    Tanh,

    /// Gaussian activation: f(x) = exp(-x * steepness * x * steepness)
    /// Output range: (0, 1], peaks at x=0
    Gaussian,

    /// Symmetric gaussian: f(x) = exp(-x * steepness * x * steepness) * 2 - 1
    /// Output range: (-1, 1], peaks at x=0
    GaussianSymmetric,

    /// Elliott activation: f(x) = ((x * steepness) / 2) / (1 + |x * steepness|) + 0.5
    /// Fast approximation to sigmoid, output range: (0, 1)
    Elliot,

    /// Symmetric Elliott: f(x) = (x * steepness) / (1 + |x * steepness|)
    /// Fast approximation to tanh, output range: (-1, 1)
    ElliotSymmetric,

    /// Bounded linear: f(x) = max(0, min(1, x * steepness))
    /// Output range: [0, 1]
    LinearPiece,

    /// Symmetric bounded linear: f(x) = max(-1, min(1, x * steepness))
    /// Output range: [-1, 1]
    LinearPieceSymmetric,

    /// Rectified Linear Unit (ReLU): f(x) = max(0, x)
    /// Output range: [0, ∞)
    ReLU,

    /// Leaky ReLU: f(x) = x if x > 0, 0.01 * x if x <= 0
    /// Output range: (-∞, ∞)
    ReLULeaky,

    /// Sine activation: f(x) = sin(x * steepness) / 2 + 0.5
    /// Output range: [0, 1]
    Sin,

    /// Cosine activation: f(x) = cos(x * steepness) / 2 + 0.5
    /// Output range: [0, 1]
    Cos,

    /// Symmetric sine: f(x) = sin(x * steepness)
    /// Output range: [-1, 1]
    SinSymmetric,

    /// Symmetric cosine: f(x) = cos(x * steepness)
    /// Output range: [-1, 1]
    CosSymmetric,
}

impl ActivationFunction {
    /// Returns the string name of the activation function
    pub fn name(&self) -> &'static str {
        match self {
            ActivationFunction::Linear => "Linear",
            ActivationFunction::Threshold => "Threshold",
            ActivationFunction::ThresholdSymmetric => "ThresholdSymmetric",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::SigmoidSymmetric => "SigmoidSymmetric",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::Gaussian => "Gaussian",
            ActivationFunction::GaussianSymmetric => "GaussianSymmetric",
            ActivationFunction::Elliot => "Elliot",
            ActivationFunction::ElliotSymmetric => "ElliotSymmetric",
            ActivationFunction::LinearPiece => "LinearPiece",
            ActivationFunction::LinearPieceSymmetric => "LinearPieceSymmetric",
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::ReLULeaky => "ReLULeaky",
            ActivationFunction::Sin => "Sin",
            ActivationFunction::Cos => "Cos",
            ActivationFunction::SinSymmetric => "SinSymmetric",
            ActivationFunction::CosSymmetric => "CosSymmetric",
        }
    }

    /// Returns whether this activation function can be used during training
    /// (i.e., has a computable derivative)
    pub fn is_trainable(&self) -> bool {
        !matches!(
            self,
            ActivationFunction::Threshold | ActivationFunction::ThresholdSymmetric
        )
    }

    /// Returns the output range of the activation function
    pub fn output_range(&self) -> (&'static str, &'static str) {
        match self {
            ActivationFunction::Linear => ("-inf", "inf"),
            ActivationFunction::Threshold => ("0", "1"),
            ActivationFunction::ThresholdSymmetric => ("-1", "1"),
            ActivationFunction::Sigmoid => ("0", "1"),
            ActivationFunction::SigmoidSymmetric | ActivationFunction::Tanh => ("-1", "1"),
            ActivationFunction::Gaussian => ("0", "1"),
            ActivationFunction::GaussianSymmetric => ("-1", "1"),
            ActivationFunction::Elliot => ("0", "1"),
            ActivationFunction::ElliotSymmetric => ("-1", "1"),
            ActivationFunction::LinearPiece => ("0", "1"),
            ActivationFunction::LinearPieceSymmetric => ("-1", "1"),
            ActivationFunction::ReLU => ("0", "inf"),
            ActivationFunction::ReLULeaky => ("-inf", "inf"),
            ActivationFunction::Sin | ActivationFunction::Cos => ("0", "1"),
            ActivationFunction::SinSymmetric | ActivationFunction::CosSymmetric => ("-1", "1"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_function_names() {
        assert_eq!(ActivationFunction::Sigmoid.name(), "Sigmoid");
        assert_eq!(ActivationFunction::ReLU.name(), "ReLU");
        assert_eq!(ActivationFunction::Tanh.name(), "Tanh");
    }

    #[test]
    fn test_trainable() {
        assert!(ActivationFunction::Sigmoid.is_trainable());
        assert!(ActivationFunction::ReLU.is_trainable());
        assert!(!ActivationFunction::Threshold.is_trainable());
        assert!(!ActivationFunction::ThresholdSymmetric.is_trainable());
    }

    #[test]
    fn test_output_ranges() {
        assert_eq!(ActivationFunction::Sigmoid.output_range(), ("0", "1"));
        assert_eq!(ActivationFunction::Tanh.output_range(), ("-1", "1"));
        assert_eq!(ActivationFunction::ReLU.output_range(), ("0", "inf"));
        assert_eq!(ActivationFunction::Linear.output_range(), ("-inf", "inf"));
    }
}
