pub mod gnn;
pub mod topology;
pub mod message_passing;
pub mod spatial_temporal;
pub mod cuda_kernels;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::dijkstra;
use std::collections::HashMap;
use ndarray::{Array2, Array3, ArrayView2, s};

/// Main PFS Twin module for Digital Twin neural models
pub struct PfsTwin {
    /// Network topology graph
    topology: NetworkTopology,
    /// Graph Neural Network model
    gnn: GraphNeuralNetwork,
    /// Spatial-temporal convolution layer
    st_conv: SpatialTemporalConv,
    /// Message passing neural network
    mpnn: MessagePassingNN,
}

/// Network element types in the RAN topology
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetworkElementType {
    GNB,    // gNodeB (5G base station)
    DU,     // Distributed Unit
    CU,     // Centralized Unit
    Cell,   // Radio cell
    UE,     // User Equipment
}

/// Network element with attributes
#[derive(Debug, Clone)]
pub struct NetworkElement {
    pub id: String,
    pub element_type: NetworkElementType,
    pub features: Vec<f32>,
    pub position: Option<(f64, f64, f64)>, // 3D coordinates
}

/// Edge types in the network topology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    Hierarchy,      // gNB -> DU -> CU
    Neighbor,       // Cell neighbor relations
    Connection,     // UE connections
    Backhaul,       // Network backhaul links
}

/// Network edge with attributes
#[derive(Debug, Clone)]
pub struct NetworkEdge {
    pub edge_type: EdgeType,
    pub weight: f32,
    pub features: Vec<f32>,
}

/// Network topology representation using petgraph
pub struct NetworkTopology {
    /// Directed graph representing the network
    graph: DiGraph<NetworkElement, NetworkEdge>,
    /// Node index mapping for fast lookup
    node_map: HashMap<String, NodeIndex>,
    /// Sparse adjacency matrix representation
    adjacency: SparseMatrix,
}

/// Sparse matrix representation for GPU-friendly operations
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Row indices (CSR format)
    row_ptr: Vec<usize>,
    /// Column indices
    col_idx: Vec<usize>,
    /// Values
    values: Vec<f32>,
    /// Matrix dimensions
    shape: (usize, usize),
}

impl NetworkTopology {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            adjacency: SparseMatrix::new(0, 0),
        }
    }

    /// Add a network element to the topology
    pub fn add_element(&mut self, element: NetworkElement) -> NodeIndex {
        let id = element.id.clone();
        let idx = self.graph.add_node(element);
        self.node_map.insert(id, idx);
        self.update_adjacency_matrix();
        idx
    }

    /// Add an edge between network elements
    pub fn add_edge(&mut self, from_id: &str, to_id: &str, edge: NetworkEdge) {
        if let (Some(&from_idx), Some(&to_idx)) = 
            (self.node_map.get(from_id), self.node_map.get(to_id)) {
            self.graph.add_edge(from_idx, to_idx, edge);
            self.update_adjacency_matrix();
        }
    }

    /// Update sparse adjacency matrix representation
    fn update_adjacency_matrix(&mut self) {
        let n = self.graph.node_count();
        let mut row_ptr = vec![0; n + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for (i, node) in self.graph.node_indices().enumerate() {
            let edges: Vec<_> = self.graph.edges(node).collect();
            row_ptr[i + 1] = row_ptr[i] + edges.len();
            
            for edge in edges {
                col_idx.push(edge.target().index());
                values.push(edge.weight().weight);
            }
        }

        self.adjacency = SparseMatrix {
            row_ptr,
            col_idx,
            values,
            shape: (n, n),
        };
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<&NetworkElement> {
        if let Some(&idx) = self.node_map.get(node_id) {
            self.graph.neighbors(idx)
                .filter_map(|n| self.graph.node_weight(n))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Incremental graph update
    pub fn incremental_update(&mut self, updates: Vec<TopologyUpdate>) {
        for update in updates {
            match update {
                TopologyUpdate::AddNode(element) => {
                    self.add_element(element);
                }
                TopologyUpdate::RemoveNode(id) => {
                    if let Some(&idx) = self.node_map.get(&id) {
                        self.graph.remove_node(idx);
                        self.node_map.remove(&id);
                        self.update_adjacency_matrix();
                    }
                }
                TopologyUpdate::AddEdge(from, to, edge) => {
                    self.add_edge(&from, &to, edge);
                }
                TopologyUpdate::UpdateNodeFeatures(id, features) => {
                    if let Some(&idx) = self.node_map.get(&id) {
                        if let Some(node) = self.graph.node_weight_mut(idx) {
                            node.features = features;
                        }
                    }
                }
            }
        }
    }
}

/// Topology update operations for incremental updates
#[derive(Debug, Clone)]
pub enum TopologyUpdate {
    AddNode(NetworkElement),
    RemoveNode(String),
    AddEdge(String, String, NetworkEdge),
    UpdateNodeFeatures(String, Vec<f32>),
}

/// Graph Neural Network implementation
pub struct GraphNeuralNetwork {
    /// Number of layers
    num_layers: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Layer weights
    weights: Vec<Array2<f32>>,
    /// Activation function
    activation: ActivationType,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    GELU,
}

impl GraphNeuralNetwork {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize) -> Self {
        let mut weights = Vec::with_capacity(num_layers);
        
        // Initialize weights
        weights.push(Array2::zeros((input_dim, hidden_dim)));
        for _ in 1..num_layers - 1 {
            weights.push(Array2::zeros((hidden_dim, hidden_dim)));
        }
        weights.push(Array2::zeros((hidden_dim, output_dim)));

        Self {
            num_layers,
            hidden_dim,
            output_dim,
            weights,
            activation: ActivationType::ReLU,
        }
    }

    /// Forward pass through the GNN
    pub fn forward(&self, node_features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        let mut h = node_features.clone();
        
        for i in 0..self.num_layers {
            // Graph convolution: H' = σ(AHW)
            h = self.graph_convolution(&h, adjacency, &self.weights[i]);
            
            // Apply activation
            if i < self.num_layers - 1 {
                h = self.apply_activation(&h);
            }
        }
        
        h
    }

    /// Graph convolution operation
    fn graph_convolution(&self, features: &Array2<f32>, adj: &SparseMatrix, weight: &Array2<f32>) -> Array2<f32> {
        // Sparse matrix multiplication followed by dense matrix multiplication
        let aggregated = self.sparse_dense_matmul(adj, features);
        aggregated.dot(weight)
    }

    /// Sparse-dense matrix multiplication
    fn sparse_dense_matmul(&self, sparse: &SparseMatrix, dense: &Array2<f32>) -> Array2<f32> {
        let (n_rows, n_cols) = sparse.shape;
        let mut result = Array2::zeros((n_rows, dense.ncols()));
        
        for i in 0..n_rows {
            let start = sparse.row_ptr[i];
            let end = sparse.row_ptr[i + 1];
            
            for j in start..end {
                let col = sparse.col_idx[j];
                let val = sparse.values[j];
                
                for k in 0..dense.ncols() {
                    result[[i, k]] += val * dense[[col, k]];
                }
            }
        }
        
        result
    }

    /// Apply activation function
    fn apply_activation(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            ActivationType::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationType::Tanh => x.mapv(|v| v.tanh()),
            ActivationType::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationType::GELU => x.mapv(|v| {
                0.5 * v * (1.0 + (0.7978845608 * (v + 0.044715 * v.powi(3))).tanh())
            }),
        }
    }
}

/// Spatial-Temporal Convolution for dynamic topology
pub struct SpatialTemporalConv {
    /// Spatial convolution weights
    spatial_weights: Array2<f32>,
    /// Temporal convolution weights
    temporal_weights: Array3<f32>,
    /// Temporal window size
    window_size: usize,
}

impl SpatialTemporalConv {
    pub fn new(spatial_dim: usize, temporal_dim: usize, window_size: usize) -> Self {
        Self {
            spatial_weights: Array2::zeros((spatial_dim, spatial_dim)),
            temporal_weights: Array3::zeros((window_size, temporal_dim, temporal_dim)),
            window_size,
        }
    }

    /// Apply spatial-temporal convolution
    pub fn forward(&self, spatial_features: &Array2<f32>, temporal_features: &Array3<f32>) -> Array2<f32> {
        // Apply spatial convolution
        let spatial_out = spatial_features.dot(&self.spatial_weights);
        
        // Apply temporal convolution
        let mut temporal_out = Array2::zeros(spatial_out.dim());
        
        for t in 0..self.window_size {
            let temporal_slice = temporal_features.index_axis(ndarray::Axis(0), t);
            let conv_result = temporal_slice.dot(&self.temporal_weights.index_axis(ndarray::Axis(0), t));
            temporal_out = temporal_out + conv_result;
        }
        
        // Combine spatial and temporal features
        spatial_out + temporal_out
    }
}

/// Message Passing Neural Network
pub struct MessagePassingNN {
    /// Message function weights
    message_weights: Array2<f32>,
    /// Update function weights
    update_weights: Array2<f32>,
    /// Aggregation type
    aggregation: AggregationType,
}

#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    Sum,
    Mean,
    Max,
    Attention,
}

impl MessagePassingNN {
    pub fn new(node_dim: usize, message_dim: usize, aggregation: AggregationType) -> Self {
        Self {
            message_weights: Array2::zeros((node_dim * 2, message_dim)),
            update_weights: Array2::zeros((node_dim + message_dim, node_dim)),
            aggregation,
        }
    }

    /// Message passing forward pass
    pub fn forward(&self, node_features: &Array2<f32>, edge_index: &[(usize, usize)]) -> Array2<f32> {
        let num_nodes = node_features.nrows();
        let node_dim = node_features.ncols();
        
        // Compute messages
        let messages = self.compute_messages(node_features, edge_index);
        
        // Aggregate messages
        let aggregated = self.aggregate_messages(&messages, edge_index, num_nodes);
        
        // Update node features
        self.update_nodes(node_features, &aggregated)
    }

    /// Compute messages between connected nodes
    fn compute_messages(&self, node_features: &Array2<f32>, edge_index: &[(usize, usize)]) -> Vec<Vec<f32>> {
        let mut messages = Vec::new();
        
        for &(src, dst) in edge_index {
            let src_features = node_features.row(src);
            let dst_features = node_features.row(dst);
            
            // Concatenate source and destination features
            let mut concat_features = Vec::with_capacity(src_features.len() + dst_features.len());
            concat_features.extend_from_slice(src_features.as_slice().unwrap());
            concat_features.extend_from_slice(dst_features.as_slice().unwrap());
            
            // Apply message function
            let concat_array = ArrayView2::from_shape((1, concat_features.len()), &concat_features).unwrap();
            let message = concat_array.dot(&self.message_weights);
            messages.push(message.to_vec());
        }
        
        messages
    }

    /// Aggregate messages per node
    fn aggregate_messages(&self, messages: &[Vec<f32>], edge_index: &[(usize, usize)], num_nodes: usize) -> Array2<f32> {
        let message_dim = messages[0].len();
        let mut aggregated = Array2::zeros((num_nodes, message_dim));
        let mut counts = vec![0.0; num_nodes];
        
        for (i, &(_, dst)) in edge_index.iter().enumerate() {
            match self.aggregation {
                AggregationType::Sum => {
                    for j in 0..message_dim {
                        aggregated[[dst, j]] += messages[i][j];
                    }
                }
                AggregationType::Mean => {
                    for j in 0..message_dim {
                        aggregated[[dst, j]] += messages[i][j];
                    }
                    counts[dst] += 1.0;
                }
                AggregationType::Max => {
                    for j in 0..message_dim {
                        aggregated[[dst, j]] = aggregated[[dst, j]].max(messages[i][j]);
                    }
                }
                AggregationType::Attention => {
                    // Simplified attention aggregation
                    let attention_weight = 1.0 / (1.0 + edge_index.len() as f32);
                    for j in 0..message_dim {
                        aggregated[[dst, j]] += attention_weight * messages[i][j];
                    }
                }
            }
        }
        
        // Normalize for mean aggregation
        if let AggregationType::Mean = self.aggregation {
            for i in 0..num_nodes {
                if counts[i] > 0.0 {
                    for j in 0..message_dim {
                        aggregated[[i, j]] /= counts[i];
                    }
                }
            }
        }
        
        aggregated
    }

    /// Update node features with aggregated messages
    fn update_nodes(&self, node_features: &Array2<f32>, aggregated_messages: &Array2<f32>) -> Array2<f32> {
        let num_nodes = node_features.nrows();
        let node_dim = node_features.ncols();
        let message_dim = aggregated_messages.ncols();
        
        let mut updated = Array2::zeros((num_nodes, node_dim));
        
        for i in 0..num_nodes {
            // Concatenate node features with aggregated messages
            let mut concat = Vec::with_capacity(node_dim + message_dim);
            concat.extend_from_slice(node_features.row(i).as_slice().unwrap());
            concat.extend_from_slice(aggregated_messages.row(i).as_slice().unwrap());
            
            // Apply update function
            let concat_array = ArrayView2::from_shape((1, concat.len()), &concat).unwrap();
            let updated_features = concat_array.dot(&self.update_weights);
            
            for j in 0..node_dim {
                updated[[i, j]] = updated_features[[0, j]];
            }
        }
        
        updated
    }
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
            shape: (rows, cols),
        }
    }

    /// Convert to dense matrix (for debugging/visualization)
    pub fn to_dense(&self) -> Array2<f32> {
        let (n_rows, n_cols) = self.shape;
        let mut dense = Array2::zeros((n_rows, n_cols));
        
        for i in 0..n_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            
            for j in start..end {
                let col = self.col_idx[j];
                let val = self.values[j];
                dense[[i, col]] = val;
            }
        }
        
        dense
    }
}

impl PfsTwin {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            topology: NetworkTopology::new(),
            gnn: GraphNeuralNetwork::new(input_dim, hidden_dim, output_dim, 3),
            st_conv: SpatialTemporalConv::new(hidden_dim, hidden_dim, 5),
            mpnn: MessagePassingNN::new(hidden_dim, hidden_dim, AggregationType::Mean),
        }
    }

    /// Process network topology and generate embeddings
    pub fn process_topology(&self, features: &Array2<f32>) -> Array2<f32> {
        // Apply GNN to get initial embeddings
        let gnn_output = self.gnn.forward(features, &self.topology.adjacency);
        
        // Apply message passing for refined embeddings
        let edge_index = self.get_edge_index();
        let mpnn_output = self.mpnn.forward(&gnn_output, &edge_index);
        
        mpnn_output
    }

    /// Get edge index representation for message passing
    fn get_edge_index(&self) -> Vec<(usize, usize)> {
        let mut edge_index = Vec::new();
        
        for node in self.topology.graph.node_indices() {
            for edge in self.topology.graph.edges(node) {
                edge_index.push((edge.source().index(), edge.target().index()));
            }
        }
        
        edge_index
    }

    /// Add network hierarchy (gNB -> DU -> CU)
    pub fn add_network_hierarchy(&mut self, gnb_id: &str, du_ids: &[String], cu_id: &str) {
        // Add gNB
        let gnb = NetworkElement {
            id: gnb_id.to_string(),
            element_type: NetworkElementType::GNB,
            features: vec![1.0, 0.0, 0.0, 0.0],
            position: None,
        };
        self.topology.add_element(gnb);
        
        // Add CU
        let cu = NetworkElement {
            id: cu_id.to_string(),
            element_type: NetworkElementType::CU,
            features: vec![0.0, 0.0, 1.0, 0.0],
            position: None,
        };
        self.topology.add_element(cu);
        
        // Add DUs and connections
        for du_id in du_ids {
            let du = NetworkElement {
                id: du_id.clone(),
                element_type: NetworkElementType::DU,
                features: vec![0.0, 1.0, 0.0, 0.0],
                position: None,
            };
            self.topology.add_element(du);
            
            // gNB -> DU connection
            let gnb_du_edge = NetworkEdge {
                edge_type: EdgeType::Hierarchy,
                weight: 1.0,
                features: vec![1.0],
            };
            self.topology.add_edge(gnb_id, du_id, gnb_du_edge);
            
            // DU -> CU connection
            let du_cu_edge = NetworkEdge {
                edge_type: EdgeType::Hierarchy,
                weight: 1.0,
                features: vec![1.0],
            };
            self.topology.add_edge(du_id, cu_id, du_cu_edge);
        }
    }

    /// Add cell neighbor relations
    pub fn add_cell_neighbors(&mut self, cell_id: &str, neighbor_ids: &[String]) {
        for neighbor_id in neighbor_ids {
            let edge = NetworkEdge {
                edge_type: EdgeType::Neighbor,
                weight: 1.0,
                features: vec![0.5],
            };
            self.topology.add_edge(cell_id, neighbor_id, edge);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_topology() {
        let mut topology = NetworkTopology::new();
        
        let gnb = NetworkElement {
            id: "gNB1".to_string(),
            element_type: NetworkElementType::GNB,
            features: vec![1.0, 0.0, 0.0],
            position: Some((0.0, 0.0, 0.0)),
        };
        
        let idx = topology.add_element(gnb);
        assert_eq!(topology.graph.node_count(), 1);
    }

    #[test]
    fn test_sparse_matrix() {
        let sparse = SparseMatrix {
            row_ptr: vec![0, 2, 3, 5],
            col_idx: vec![0, 2, 1, 0, 2],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            shape: (3, 3),
        };
        
        let dense = sparse.to_dense();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 2]], 2.0);
        assert_eq!(dense[[1, 1]], 3.0);
    }
}