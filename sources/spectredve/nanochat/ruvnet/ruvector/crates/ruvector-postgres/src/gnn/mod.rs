//! Graph Neural Network (GNN) module for ruvector-postgres
//!
//! This module provides graph neural network layers and operations
//! for PostgreSQL, enabling efficient graph learning on relational data.

pub mod aggregators;
pub mod gcn;
pub mod graphsage;
pub mod message_passing;
pub mod operators;

// Re-export key types and traits
pub use aggregators::{max_aggregate, mean_aggregate, sum_aggregate, AggregationMethod};
pub use gcn::GCNLayer;
pub use graphsage::GraphSAGELayer;
pub use message_passing::{build_adjacency_list, propagate, MessagePassing};
pub use operators::{ruvector_gcn_forward, ruvector_gnn_aggregate, ruvector_message_pass};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Ensure all public exports are accessible
        let _ = AggregationMethod::Sum;
        let _ = AggregationMethod::Mean;
        let _ = AggregationMethod::Max;
    }
}
