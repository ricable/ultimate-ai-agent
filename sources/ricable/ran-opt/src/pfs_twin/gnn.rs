use ndarray::{Array2, Array3, ArrayView2, Axis};
use std::collections::HashMap;
use crate::pfs_twin::{SparseMatrix, ActivationType};

/// Graph Convolutional Network layer
pub struct GraphConvLayer {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Weight matrix
    weight: Array2<f32>,
    /// Bias vector
    bias: Array2<f32>,
    /// Whether to use bias
    use_bias: bool,
    /// Normalization type
    normalization: NormalizationType,
}

#[derive(Debug, Clone, Copy)]
pub enum NormalizationType {
    None,
    Symmetric,
    RowNorm,
    ColNorm,
}

impl GraphConvLayer {
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        Self {
            input_dim,
            output_dim,
            weight: Array2::zeros((input_dim, output_dim)),
            bias: Array2::zeros((1, output_dim)),
            use_bias,
            normalization: NormalizationType::Symmetric,
        }
    }

    /// Initialize weights using Xavier/Glorot initialization
    pub fn xavier_init(&mut self) {
        let fan_in = self.input_dim as f32;
        let fan_out = self.output_dim as f32;
        let bound = (6.0 / (fan_in + fan_out)).sqrt();
        
        // Initialize weights randomly (simplified for demo)
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                self.weight[[i, j]] = (rand::random::<f32>() - 0.5) * 2.0 * bound;
            }
        }
    }

    /// Forward pass
    pub fn forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        // Normalize adjacency matrix
        let norm_adj = self.normalize_adjacency(adjacency);
        
        // Apply linear transformation: XW
        let transformed = features.dot(&self.weight);
        
        // Apply graph convolution: AXW
        let output = self.sparse_dense_matmul(&norm_adj, &transformed);
        
        // Add bias if enabled
        if self.use_bias {
            output + &self.bias
        } else {
            output
        }
    }

    /// Normalize adjacency matrix
    fn normalize_adjacency(&self, adj: &SparseMatrix) -> SparseMatrix {
        match self.normalization {
            NormalizationType::None => adj.clone(),
            NormalizationType::Symmetric => self.symmetric_normalize(adj),
            NormalizationType::RowNorm => self.row_normalize(adj),
            NormalizationType::ColNorm => self.col_normalize(adj),
        }
    }

    /// Symmetric normalization: D^(-1/2) A D^(-1/2)
    fn symmetric_normalize(&self, adj: &SparseMatrix) -> SparseMatrix {
        let (n_rows, _) = adj.shape;
        let mut degrees = vec![0.0; n_rows];
        
        // Calculate degrees
        for i in 0..n_rows {
            let start = adj.row_ptr[i];
            let end = adj.row_ptr[i + 1];
            
            for j in start..end {
                degrees[i] += adj.values[j];
            }
        }
        
        // Calculate D^(-1/2)
        let mut inv_sqrt_deg = vec![0.0; n_rows];
        for i in 0..n_rows {
            if degrees[i] > 0.0 {
                inv_sqrt_deg[i] = 1.0 / degrees[i].sqrt();
            }
        }
        
        // Apply symmetric normalization
        let mut normalized = adj.clone();
        for i in 0..n_rows {
            let start = adj.row_ptr[i];
            let end = adj.row_ptr[i + 1];
            
            for j in start..end {
                let col = adj.col_idx[j];
                normalized.values[j] = adj.values[j] * inv_sqrt_deg[i] * inv_sqrt_deg[col];
            }
        }
        
        normalized
    }

    /// Row normalization: D^(-1) A
    fn row_normalize(&self, adj: &SparseMatrix) -> SparseMatrix {
        let (n_rows, _) = adj.shape;
        let mut degrees = vec![0.0; n_rows];
        
        // Calculate row sums
        for i in 0..n_rows {
            let start = adj.row_ptr[i];
            let end = adj.row_ptr[i + 1];
            
            for j in start..end {
                degrees[i] += adj.values[j];
            }
        }
        
        // Apply row normalization
        let mut normalized = adj.clone();
        for i in 0..n_rows {
            let start = adj.row_ptr[i];
            let end = adj.row_ptr[i + 1];
            
            if degrees[i] > 0.0 {
                for j in start..end {
                    normalized.values[j] = adj.values[j] / degrees[i];
                }
            }
        }
        
        normalized
    }

    /// Column normalization: A D^(-1)
    fn col_normalize(&self, adj: &SparseMatrix) -> SparseMatrix {
        let (n_rows, n_cols) = adj.shape;
        let mut degrees = vec![0.0; n_cols];
        
        // Calculate column sums
        for i in 0..n_rows {
            let start = adj.row_ptr[i];
            let end = adj.row_ptr[i + 1];
            
            for j in start..end {
                let col = adj.col_idx[j];
                degrees[col] += adj.values[j];
            }
        }
        
        // Apply column normalization
        let mut normalized = adj.clone();
        for i in 0..n_rows {
            let start = adj.row_ptr[i];
            let end = adj.row_ptr[i + 1];
            
            for j in start..end {
                let col = adj.col_idx[j];
                if degrees[col] > 0.0 {
                    normalized.values[j] = adj.values[j] / degrees[col];
                }
            }
        }
        
        normalized
    }

    /// Sparse-dense matrix multiplication
    fn sparse_dense_matmul(&self, sparse: &SparseMatrix, dense: &Array2<f32>) -> Array2<f32> {
        let (n_rows, _) = sparse.shape;
        let n_cols = dense.ncols();
        let mut result = Array2::zeros((n_rows, n_cols));
        
        for i in 0..n_rows {
            let start = sparse.row_ptr[i];
            let end = sparse.row_ptr[i + 1];
            
            for j in start..end {
                let col = sparse.col_idx[j];
                let val = sparse.values[j];
                
                for k in 0..n_cols {
                    result[[i, k]] += val * dense[[col, k]];
                }
            }
        }
        
        result
    }
}

/// Graph Attention Network layer
pub struct GraphAttentionLayer {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Weight matrices for each head
    weights: Vec<Array2<f32>>,
    /// Attention vectors for each head
    attention_vectors: Vec<Array2<f32>>,
    /// Dropout rate
    dropout: f32,
    /// Whether to use bias
    use_bias: bool,
}

impl GraphAttentionLayer {
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize, dropout: f32) -> Self {
        let mut weights = Vec::new();
        let mut attention_vectors = Vec::new();
        
        for _ in 0..num_heads {
            weights.push(Array2::zeros((input_dim, output_dim)));
            attention_vectors.push(Array2::zeros((2 * output_dim, 1)));
        }
        
        Self {
            input_dim,
            output_dim,
            num_heads,
            weights,
            attention_vectors,
            dropout,
            use_bias: true,
        }
    }

    /// Forward pass with multi-head attention
    pub fn forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        let mut head_outputs = Vec::new();
        
        for head in 0..self.num_heads {
            let head_output = self.attention_head_forward(features, adjacency, head);
            head_outputs.push(head_output);
        }
        
        // Concatenate or average head outputs
        self.combine_heads(&head_outputs)
    }

    /// Forward pass for a single attention head
    fn attention_head_forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix, head: usize) -> Array2<f32> {
        let num_nodes = features.nrows();
        
        // Linear transformation
        let h = features.dot(&self.weights[head]);
        
        // Compute attention coefficients
        let attention_coeff = self.compute_attention_coefficients(&h, adjacency, head);
        
        // Apply attention and aggregate
        self.aggregate_with_attention(&h, &attention_coeff, adjacency)
    }

    /// Compute attention coefficients
    fn compute_attention_coefficients(&self, features: &Array2<f32>, adjacency: &SparseMatrix, head: usize) -> SparseMatrix {
        let (n_rows, _) = adjacency.shape;
        let mut attention_values = Vec::new();
        let mut attention_indices = Vec::new();
        let mut attention_ptr = vec![0; n_rows + 1];
        
        for i in 0..n_rows {
            let start = adjacency.row_ptr[i];
            let end = adjacency.row_ptr[i + 1];
            
            let mut local_attention = Vec::new();
            
            for j in start..end {
                let neighbor = adjacency.col_idx[j];
                
                // Concatenate features for attention computation
                let mut concat_features = Vec::new();
                concat_features.extend_from_slice(features.row(i).as_slice().unwrap());
                concat_features.extend_from_slice(features.row(neighbor).as_slice().unwrap());
                
                // Compute attention score
                let concat_array = ArrayView2::from_shape((1, concat_features.len()), &concat_features).unwrap();
                let attention_score = concat_array.dot(&self.attention_vectors[head])[[0, 0]];
                
                local_attention.push((neighbor, attention_score));
            }
            
            // Apply softmax to attention scores
            let max_score = local_attention.iter().map(|(_, score)| *score).fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0;
            let mut softmax_attention = Vec::new();
            
            for (neighbor, score) in local_attention {
                let exp_score = (score - max_score).exp();
                sum_exp += exp_score;
                softmax_attention.push((neighbor, exp_score));
            }
            
            // Normalize and store
            for (neighbor, exp_score) in softmax_attention {
                let normalized_score = exp_score / sum_exp;
                attention_values.push(normalized_score);
                attention_indices.push(neighbor);
            }
            
            attention_ptr[i + 1] = attention_values.len();
        }
        
        SparseMatrix {
            row_ptr: attention_ptr,
            col_idx: attention_indices,
            values: attention_values,
            shape: adjacency.shape,
        }
    }

    /// Aggregate features with attention weights
    fn aggregate_with_attention(&self, features: &Array2<f32>, attention: &SparseMatrix, adjacency: &SparseMatrix) -> Array2<f32> {
        let (n_rows, n_cols) = (features.nrows(), features.ncols());
        let mut output = Array2::zeros((n_rows, n_cols));
        
        for i in 0..n_rows {
            let start = attention.row_ptr[i];
            let end = attention.row_ptr[i + 1];
            
            for j in start..end {
                let neighbor = attention.col_idx[j];
                let weight = attention.values[j];
                
                for k in 0..n_cols {
                    output[[i, k]] += weight * features[[neighbor, k]];
                }
            }
        }
        
        output
    }

    /// Combine multiple attention heads
    fn combine_heads(&self, head_outputs: &[Array2<f32>]) -> Array2<f32> {
        let (n_rows, n_cols) = (head_outputs[0].nrows(), head_outputs[0].ncols());
        let mut combined = Array2::zeros((n_rows, n_cols));
        
        // Average the heads
        for head_output in head_outputs {
            combined = combined + head_output;
        }
        
        combined / (self.num_heads as f32)
    }
}

/// GraphSAGE layer for inductive learning
pub struct GraphSAGELayer {
    /// Aggregator type
    aggregator: AggregatorType,
    /// Weight matrices
    self_weight: Array2<f32>,
    /// Neighbor weight matrix
    neighbor_weight: Array2<f32>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Normalization
    normalize: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum AggregatorType {
    Mean,
    MaxPool,
    LSTM,
    GCN,
}

impl GraphSAGELayer {
    pub fn new(input_dim: usize, output_dim: usize, aggregator: AggregatorType) -> Self {
        Self {
            aggregator,
            self_weight: Array2::zeros((input_dim, output_dim)),
            neighbor_weight: Array2::zeros((input_dim, output_dim)),
            input_dim,
            output_dim,
            normalize: true,
        }
    }

    /// Forward pass
    pub fn forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        // Aggregate neighbor features
        let neighbor_aggregated = self.aggregate_neighbors(features, adjacency);
        
        // Self transformation
        let self_transformed = features.dot(&self.self_weight);
        
        // Neighbor transformation
        let neighbor_transformed = neighbor_aggregated.dot(&self.neighbor_weight);
        
        // Combine
        let combined = self_transformed + neighbor_transformed;
        
        // Apply normalization
        if self.normalize {
            self.l2_normalize(&combined)
        } else {
            combined
        }
    }

    /// Aggregate neighbor features
    fn aggregate_neighbors(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        match self.aggregator {
            AggregatorType::Mean => self.mean_aggregation(features, adjacency),
            AggregatorType::MaxPool => self.max_pool_aggregation(features, adjacency),
            AggregatorType::LSTM => self.lstm_aggregation(features, adjacency),
            AggregatorType::GCN => self.gcn_aggregation(features, adjacency),
        }
    }

    /// Mean aggregation
    fn mean_aggregation(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        let (n_rows, n_cols) = (features.nrows(), features.ncols());
        let mut aggregated = Array2::zeros((n_rows, n_cols));
        let mut counts = vec![0; n_rows];
        
        for i in 0..n_rows {
            let start = adjacency.row_ptr[i];
            let end = adjacency.row_ptr[i + 1];
            
            for j in start..end {
                let neighbor = adjacency.col_idx[j];
                counts[i] += 1;
                
                for k in 0..n_cols {
                    aggregated[[i, k]] += features[[neighbor, k]];
                }
            }
            
            // Average
            if counts[i] > 0 {
                for k in 0..n_cols {
                    aggregated[[i, k]] /= counts[i] as f32;
                }
            }
        }
        
        aggregated
    }

    /// Max pooling aggregation
    fn max_pool_aggregation(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        let (n_rows, n_cols) = (features.nrows(), features.ncols());
        let mut aggregated = Array2::from_elem((n_rows, n_cols), f32::NEG_INFINITY);
        
        for i in 0..n_rows {
            let start = adjacency.row_ptr[i];
            let end = adjacency.row_ptr[i + 1];
            
            for j in start..end {
                let neighbor = adjacency.col_idx[j];
                
                for k in 0..n_cols {
                    aggregated[[i, k]] = aggregated[[i, k]].max(features[[neighbor, k]]);
                }
            }
        }
        
        // Replace -inf with 0 for nodes with no neighbors
        for i in 0..n_rows {
            for k in 0..n_cols {
                if aggregated[[i, k]] == f32::NEG_INFINITY {
                    aggregated[[i, k]] = 0.0;
                }
            }
        }
        
        aggregated
    }

    /// LSTM aggregation (simplified version)
    fn lstm_aggregation(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        // For simplicity, we'll use mean aggregation with permutation
        // In practice, this would involve an actual LSTM
        self.mean_aggregation(features, adjacency)
    }

    /// GCN-style aggregation
    fn gcn_aggregation(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        // Add self-loops and apply symmetric normalization
        let mut adj_with_self = adjacency.clone();
        
        // Add self-loops (simplified)
        for i in 0..adj_with_self.shape.0 {
            adj_with_self.row_ptr[i + 1] += 1;
            adj_with_self.col_idx.push(i);
            adj_with_self.values.push(1.0);
        }
        
        // Apply symmetric normalization and aggregate
        let conv_layer = GraphConvLayer::new(self.input_dim, self.input_dim, false);
        conv_layer.sparse_dense_matmul(&adj_with_self, features)
    }

    /// L2 normalization
    fn l2_normalize(&self, features: &Array2<f32>) -> Array2<f32> {
        let mut normalized = features.clone();
        
        for i in 0..features.nrows() {
            let mut norm = 0.0;
            for j in 0..features.ncols() {
                norm += features[[i, j]].powi(2);
            }
            norm = norm.sqrt();
            
            if norm > 0.0 {
                for j in 0..features.ncols() {
                    normalized[[i, j]] = features[[i, j]] / norm;
                }
            }
        }
        
        normalized
    }
}

/// Multi-layer GNN with different layer types
pub struct MultiLayerGNN {
    /// Layer configurations
    layers: Vec<GNNLayerType>,
    /// Activation functions
    activations: Vec<ActivationType>,
    /// Dropout rates
    dropout_rates: Vec<f32>,
}

pub enum GNNLayerType {
    GraphConv(GraphConvLayer),
    GraphAttention(GraphAttentionLayer),
    GraphSAGE(GraphSAGELayer),
}

impl MultiLayerGNN {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            activations: Vec::new(),
            dropout_rates: Vec::new(),
        }
    }

    /// Add a layer to the network
    pub fn add_layer(&mut self, layer: GNNLayerType, activation: ActivationType, dropout: f32) {
        self.layers.push(layer);
        self.activations.push(activation);
        self.dropout_rates.push(dropout);
    }

    /// Forward pass through all layers
    pub fn forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        let mut x = features.clone();
        
        for (i, layer) in self.layers.iter().enumerate() {
            // Apply layer
            x = match layer {
                GNNLayerType::GraphConv(conv) => conv.forward(&x, adjacency),
                GNNLayerType::GraphAttention(att) => att.forward(&x, adjacency),
                GNNLayerType::GraphSAGE(sage) => sage.forward(&x, adjacency),
            };
            
            // Apply activation
            x = self.apply_activation(&x, self.activations[i]);
            
            // Apply dropout (simplified - in practice would be random)
            if self.dropout_rates[i] > 0.0 {
                x = x * (1.0 - self.dropout_rates[i]);
            }
        }
        
        x
    }

    /// Apply activation function
    fn apply_activation(&self, x: &Array2<f32>, activation: ActivationType) -> Array2<f32> {
        match activation {
            ActivationType::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationType::Tanh => x.mapv(|v| v.tanh()),
            ActivationType::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationType::GELU => x.mapv(|v| {
                0.5 * v * (1.0 + (0.7978845608 * (v + 0.044715 * v.powi(3))).tanh())
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_conv_layer() {
        let mut layer = GraphConvLayer::new(3, 2, true);
        layer.xavier_init();
        
        let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sparse_adj = SparseMatrix {
            row_ptr: vec![0, 1, 2],
            col_idx: vec![1, 0],
            values: vec![1.0, 1.0],
            shape: (2, 2),
        };
        
        let output = layer.forward(&features, &sparse_adj);
        assert_eq!(output.shape(), (2, 2));
    }

    #[test]
    fn test_graph_attention_layer() {
        let layer = GraphAttentionLayer::new(3, 2, 2, 0.1);
        
        let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sparse_adj = SparseMatrix {
            row_ptr: vec![0, 1, 2],
            col_idx: vec![1, 0],
            values: vec![1.0, 1.0],
            shape: (2, 2),
        };
        
        let output = layer.forward(&features, &sparse_adj);
        assert_eq!(output.shape(), (2, 2));
    }
}