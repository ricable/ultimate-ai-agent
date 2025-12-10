use num_traits::Float;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a connection between two neurons with a weight
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Connection<T: Float> {
    /// Index of the source neuron
    pub from_neuron: usize,
    /// Index of the destination neuron
    pub to_neuron: usize,
    /// The weight of the connection
    pub weight: T,
}

impl<T: Float> Connection<T> {
    /// Creates a new connection between two neurons
    ///
    /// # Arguments
    /// * `from_neuron` - Index of the source neuron
    /// * `to_neuron` - Index of the destination neuron
    /// * `weight` - Initial weight of the connection
    ///
    /// # Example
    /// ```
    /// use ruv_fann::Connection;
    ///
    /// let conn = Connection::new(0, 1, 0.5_f32);
    /// assert_eq!(conn.from_neuron, 0);
    /// assert_eq!(conn.to_neuron, 1);
    /// assert_eq!(conn.weight, 0.5_f32);
    /// ```
    pub fn new(from_neuron: usize, to_neuron: usize, weight: T) -> Self {
        Connection {
            from_neuron,
            to_neuron,
            weight,
        }
    }

    /// Updates the weight of the connection
    pub fn update_weight(&mut self, delta: T) {
        self.weight = self.weight + delta;
    }

    /// Sets the weight to a specific value
    pub fn set_weight(&mut self, weight: T) {
        self.weight = weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_connection_creation_f32() {
        let conn = Connection::new(0, 1, 0.5_f32);
        assert_eq!(conn.from_neuron, 0);
        assert_eq!(conn.to_neuron, 1);
        assert_relative_eq!(conn.weight, 0.5_f32);
    }

    #[test]
    fn test_connection_creation_f64() {
        let conn = Connection::new(0, 1, 0.5_f64);
        assert_eq!(conn.from_neuron, 0);
        assert_eq!(conn.to_neuron, 1);
        assert_relative_eq!(conn.weight, 0.5_f64);
    }

    #[test]
    fn test_update_weight() {
        let mut conn = Connection::new(0, 1, 0.5_f32);
        conn.update_weight(0.2);
        assert_relative_eq!(conn.weight, 0.7_f32);

        conn.update_weight(-0.3);
        assert_relative_eq!(conn.weight, 0.4_f32);
    }

    #[test]
    fn test_set_weight() {
        let mut conn = Connection::new(0, 1, 0.5_f32);
        conn.set_weight(0.9);
        assert_relative_eq!(conn.weight, 0.9_f32);
    }
}
