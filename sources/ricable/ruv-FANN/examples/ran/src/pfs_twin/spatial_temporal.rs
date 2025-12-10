use ndarray::{Array1, Array2, Array3, Array4, ArrayView3, Axis, s};
use std::collections::HashMap;
use crate::pfs_twin::{SparseMatrix, NetworkElement, NetworkElementType};

/// Spatial-Temporal Graph Convolutional Network for dynamic topology
pub struct SpatialTemporalGCN {
    /// Spatial convolution layers
    spatial_layers: Vec<SpatialConvLayer>,
    /// Temporal convolution layers
    temporal_layers: Vec<TemporalConvLayer>,
    /// Attention mechanism for temporal dynamics
    temporal_attention: TemporalAttention,
    /// Fusion mechanism for spatial-temporal features
    fusion_layer: SpatialTemporalFusion,
    /// Number of time steps
    time_steps: usize,
}

/// Spatial convolution layer for graph structure
pub struct SpatialConvLayer {
    /// Convolution type
    conv_type: SpatialConvType,
    /// Weight matrices
    weights: Array2<f32>,
    /// Bias
    bias: Array1<f32>,
    /// Activation function
    activation: ActivationFunction,
    /// Normalization
    normalization: Option<NormalizationType>,
}

/// Temporal convolution layer for time series
pub struct TemporalConvLayer {
    /// Convolution type
    conv_type: TemporalConvType,
    /// Kernel weights
    kernel: Array3<f32>,
    /// Bias
    bias: Array1<f32>,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    /// Dilation
    dilation: usize,
}

/// Temporal attention mechanism
pub struct TemporalAttention {
    /// Query transformation
    query_transform: Array2<f32>,
    /// Key transformation
    key_transform: Array2<f32>,
    /// Value transformation
    value_transform: Array2<f32>,
    /// Attention dimension
    attention_dim: usize,
}

/// Spatial-temporal fusion layer
pub struct SpatialTemporalFusion {
    /// Fusion type
    fusion_type: FusionType,
    /// Spatial gate
    spatial_gate: Array2<f32>,
    /// Temporal gate
    temporal_gate: Array2<f32>,
    /// Output projection
    output_projection: Array2<f32>,
}

#[derive(Debug, Clone, Copy)]
pub enum SpatialConvType {
    GCN,
    GraphSAGE,
    GAT,
    ChebNet,
}

#[derive(Debug, Clone, Copy)]
pub enum TemporalConvType {
    Conv1D,
    Dilated,
    Causal,
    Gated,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    ELU,
    Swish,
    GELU,
}

#[derive(Debug, Clone, Copy)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    GraphNorm,
}

#[derive(Debug, Clone, Copy)]
pub enum FusionType {
    Concatenation,
    Addition,
    Multiplication,
    Attention,
    Gated,
}

impl SpatialConvLayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        conv_type: SpatialConvType,
        activation: ActivationFunction,
    ) -> Self {
        Self {
            conv_type,
            weights: Array2::zeros((input_dim, output_dim)),
            bias: Array1::zeros(output_dim),
            activation,
            normalization: None,
        }
    }

    /// Initialize weights
    pub fn init_weights(&mut self) {
        let fan_in = self.weights.nrows() as f32;
        let fan_out = self.weights.ncols() as f32;
        let bound = (6.0 / (fan_in + fan_out)).sqrt();
        
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                self.weights[[i, j]] = (rand::random::<f32>() - 0.5) * 2.0 * bound;
            }
        }
    }

    /// Forward pass
    pub fn forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        let output = match self.conv_type {
            SpatialConvType::GCN => self.gcn_forward(features, adjacency),
            SpatialConvType::GraphSAGE => self.graphsage_forward(features, adjacency),
            SpatialConvType::GAT => self.gat_forward(features, adjacency),
            SpatialConvType::ChebNet => self.chebnet_forward(features, adjacency),
        };
        
        self.apply_activation(&output)
    }

    /// GCN forward pass
    fn gcn_forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        // Normalize adjacency matrix
        let normalized_adj = self.normalize_adjacency(adjacency);
        
        // Graph convolution: A * X * W
        let aggregated = self.sparse_dense_matmul(&normalized_adj, features);
        let output = aggregated.dot(&self.weights);
        
        // Add bias
        let mut result = output;
        for mut row in result.axis_iter_mut(Axis(0)) {
            row += &self.bias;
        }
        
        result
    }

    /// GraphSAGE forward pass
    fn graphsage_forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        // Aggregate neighbor features
        let neighbor_features = self.aggregate_neighbors(features, adjacency);
        
        // Concatenate self and neighbor features
        let num_nodes = features.nrows();
        let self_dim = features.ncols();
        let neighbor_dim = neighbor_features.ncols();
        let mut combined = Array2::zeros((num_nodes, self_dim + neighbor_dim));
        
        combined.slice_mut(s![.., ..self_dim]).assign(features);
        combined.slice_mut(s![.., self_dim..]).assign(&neighbor_features);
        
        // Apply linear transformation
        let reduced_weights = self.weights.slice(s![..combined.ncols(), ..]);
        combined.dot(&reduced_weights)
    }

    /// GAT forward pass (simplified)
    fn gat_forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        // Simplified GAT - would need attention mechanism
        self.gcn_forward(features, adjacency)
    }

    /// ChebNet forward pass
    fn chebnet_forward(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
        // Simplified ChebNet using first-order approximation
        let identity = self.create_identity_matrix(adjacency.shape.0);
        let laplacian = self.compute_laplacian(adjacency);
        
        // T_0(L) = I, T_1(L) = L
        let t0 = features.clone();
        let t1 = self.sparse_dense_matmul(&laplacian, features);
        
        // Combine with weights (simplified to 2 terms)
        let w0 = self.weights.slice(s![.., ..features.ncols()]);
        let w1 = self.weights.slice(s![.., features.ncols()..]);
        
        t0.dot(&w0) + t1.dot(&w1)
    }

    /// Normalize adjacency matrix (symmetric normalization)
    fn normalize_adjacency(&self, adj: &SparseMatrix) -> SparseMatrix {
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
        
        // Add self-loops and recalculate degrees
        for degree in degrees.iter_mut() {
            *degree += 1.0;
        }
        
        // Calculate D^(-1/2)
        let inv_sqrt_deg: Vec<f32> = degrees.iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();
        
        // Apply normalization
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

    /// Aggregate neighbor features
    fn aggregate_neighbors(&self, features: &Array2<f32>, adjacency: &SparseMatrix) -> Array2<f32> {
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

    /// Create identity matrix
    fn create_identity_matrix(&self, size: usize) -> SparseMatrix {
        let mut row_ptr = vec![0; size + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        
        for i in 0..size {
            row_ptr[i + 1] = row_ptr[i] + 1;
            col_idx.push(i);
            values.push(1.0);
        }
        
        SparseMatrix {
            row_ptr,
            col_idx,
            values,
            shape: (size, size),
        }
    }

    /// Compute Laplacian matrix
    fn compute_laplacian(&self, adj: &SparseMatrix) -> SparseMatrix {
        // Simplified Laplacian: L = D - A
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
        
        // Create degree matrix - adjacency matrix
        let mut laplacian = adj.clone();
        for i in 0..n_rows {
            let start = adj.row_ptr[i];
            let end = adj.row_ptr[i + 1];
            
            for j in start..end {
                if adj.col_idx[j] == i {
                    // Diagonal element: degree - self_loop
                    laplacian.values[j] = degrees[i] - adj.values[j];
                } else {
                    // Off-diagonal: -adjacency
                    laplacian.values[j] = -adj.values[j];
                }
            }
        }
        
        laplacian
    }

    /// Apply activation function
    fn apply_activation(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Tanh => x.mapv(|v| v.tanh()),
            ActivationFunction::ELU => x.mapv(|v| if v > 0.0 { v } else { (v.exp() - 1.0) }),
            ActivationFunction::Swish => x.mapv(|v| v / (1.0 + (-v).exp())),
            ActivationFunction::GELU => x.mapv(|v| {
                0.5 * v * (1.0 + (0.7978845608 * (v + 0.044715 * v.powi(3))).tanh())
            }),
        }
    }
}

impl TemporalConvLayer {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        conv_type: TemporalConvType,
    ) -> Self {
        Self {
            conv_type,
            kernel: Array3::zeros((output_channels, input_channels, kernel_size)),
            bias: Array1::zeros(output_channels),
            stride: 1,
            padding: 0,
            dilation: 1,
        }
    }

    /// Initialize weights
    pub fn init_weights(&mut self) {
        let fan_in = (self.kernel.shape()[1] * self.kernel.shape()[2]) as f32;
        let bound = (1.0 / fan_in).sqrt();
        
        for i in 0..self.kernel.shape()[0] {
            for j in 0..self.kernel.shape()[1] {
                for k in 0..self.kernel.shape()[2] {
                    self.kernel[[i, j, k]] = (rand::random::<f32>() - 0.5) * 2.0 * bound;
                }
            }
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        match self.conv_type {
            TemporalConvType::Conv1D => self.conv1d_forward(input),
            TemporalConvType::Dilated => self.dilated_conv_forward(input),
            TemporalConvType::Causal => self.causal_conv_forward(input),
            TemporalConvType::Gated => self.gated_conv_forward(input),
        }
    }

    /// Standard 1D convolution
    fn conv1d_forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, input_channels, seq_len) = input.dim();
        let (output_channels, _, kernel_size) = self.kernel.dim();
        
        let output_len = (seq_len + 2 * self.padding - kernel_size) / self.stride + 1;
        let mut output = Array3::zeros((batch_size, output_channels, output_len));
        
        for b in 0..batch_size {
            for out_c in 0..output_channels {
                for out_t in 0..output_len {
                    let mut sum = 0.0;
                    
                    for in_c in 0..input_channels {
                        for k in 0..kernel_size {
                            let in_t = out_t * self.stride + k;
                            if in_t < seq_len {
                                sum += input[[b, in_c, in_t]] * self.kernel[[out_c, in_c, k]];
                            }
                        }
                    }
                    
                    output[[b, out_c, out_t]] = sum + self.bias[out_c];
                }
            }
        }
        
        output
    }

    /// Dilated convolution
    fn dilated_conv_forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, input_channels, seq_len) = input.dim();
        let (output_channels, _, kernel_size) = self.kernel.dim();
        
        let effective_kernel_size = (kernel_size - 1) * self.dilation + 1;
        let output_len = (seq_len + 2 * self.padding - effective_kernel_size) / self.stride + 1;
        let mut output = Array3::zeros((batch_size, output_channels, output_len));
        
        for b in 0..batch_size {
            for out_c in 0..output_channels {
                for out_t in 0..output_len {
                    let mut sum = 0.0;
                    
                    for in_c in 0..input_channels {
                        for k in 0..kernel_size {
                            let in_t = out_t * self.stride + k * self.dilation;
                            if in_t < seq_len {
                                sum += input[[b, in_c, in_t]] * self.kernel[[out_c, in_c, k]];
                            }
                        }
                    }
                    
                    output[[b, out_c, out_t]] = sum + self.bias[out_c];
                }
            }
        }
        
        output
    }

    /// Causal convolution (no future information)
    fn causal_conv_forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, input_channels, seq_len) = input.dim();
        let (output_channels, _, kernel_size) = self.kernel.dim();
        
        let mut output = Array3::zeros((batch_size, output_channels, seq_len));
        
        for b in 0..batch_size {
            for out_c in 0..output_channels {
                for out_t in 0..seq_len {
                    let mut sum = 0.0;
                    
                    for in_c in 0..input_channels {
                        for k in 0..kernel_size {
                            let in_t = out_t as i32 - k as i32;
                            if in_t >= 0 && in_t < seq_len as i32 {
                                sum += input[[b, in_c, in_t as usize]] * self.kernel[[out_c, in_c, k]];
                            }
                        }
                    }
                    
                    output[[b, out_c, out_t]] = sum + self.bias[out_c];
                }
            }
        }
        
        output
    }

    /// Gated convolution
    fn gated_conv_forward(&self, input: &Array3<f32>) -> Array3<f32> {
        // Split channels for gating
        let conv_output = self.conv1d_forward(input);
        let (batch_size, channels, seq_len) = conv_output.dim();
        let half_channels = channels / 2;
        
        let mut gated_output = Array3::zeros((batch_size, half_channels, seq_len));
        
        for b in 0..batch_size {
            for c in 0..half_channels {
                for t in 0..seq_len {
                    let value = conv_output[[b, c, t]];
                    let gate = conv_output[[b, c + half_channels, t]].tanh();
                    gated_output[[b, c, t]] = value * gate;
                }
            }
        }
        
        gated_output
    }
}

impl TemporalAttention {
    pub fn new(input_dim: usize, attention_dim: usize) -> Self {
        Self {
            query_transform: Array2::zeros((input_dim, attention_dim)),
            key_transform: Array2::zeros((input_dim, attention_dim)),
            value_transform: Array2::zeros((input_dim, attention_dim)),
            attention_dim,
        }
    }

    /// Apply temporal attention
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, features, seq_len) = input.dim();
        let mut attended = Array3::zeros((batch_size, features, seq_len));
        
        for b in 0..batch_size {
            let batch_input = input.index_axis(Axis(0), b);
            let batch_attended = self.self_attention(&batch_input.to_owned());
            attended.index_axis_mut(Axis(0), b).assign(&batch_attended);
        }
        
        attended
    }

    /// Self-attention mechanism
    fn self_attention(&self, input: &Array2<f32>) -> Array2<f32> {
        let (features, seq_len) = input.dim();
        
        // Transpose for attention computation
        let input_t = input.t();
        
        // Compute queries, keys, values
        let queries = input_t.dot(&self.query_transform);
        let keys = input_t.dot(&self.key_transform);
        let values = input_t.dot(&self.value_transform);
        
        // Scaled dot-product attention
        let attention_scores = queries.dot(&keys.t()) / (self.attention_dim as f32).sqrt();
        let attention_weights = self.softmax(&attention_scores);
        
        // Apply attention
        let attended = attention_weights.dot(&values);
        
        // Transpose back
        attended.t().to_owned()
    }

    /// Softmax function
    fn softmax(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut result = x.clone();
        
        for mut row in result.axis_iter_mut(Axis(0)) {
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row.mapv_inplace(|v| (v - max_val).exp());
            let sum = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }
        
        result
    }
}

impl SpatialTemporalFusion {
    pub fn new(spatial_dim: usize, temporal_dim: usize, output_dim: usize, fusion_type: FusionType) -> Self {
        Self {
            fusion_type,
            spatial_gate: Array2::zeros((spatial_dim, output_dim)),
            temporal_gate: Array2::zeros((temporal_dim, output_dim)),
            output_projection: Array2::zeros((spatial_dim + temporal_dim, output_dim)),
        }
    }

    /// Fuse spatial and temporal features
    pub fn forward(&self, spatial_features: &Array2<f32>, temporal_features: &Array2<f32>) -> Array2<f32> {
        match self.fusion_type {
            FusionType::Concatenation => self.concatenation_fusion(spatial_features, temporal_features),
            FusionType::Addition => self.addition_fusion(spatial_features, temporal_features),
            FusionType::Multiplication => self.multiplication_fusion(spatial_features, temporal_features),
            FusionType::Attention => self.attention_fusion(spatial_features, temporal_features),
            FusionType::Gated => self.gated_fusion(spatial_features, temporal_features),
        }
    }

    /// Concatenation fusion
    fn concatenation_fusion(&self, spatial: &Array2<f32>, temporal: &Array2<f32>) -> Array2<f32> {
        let (batch_size, spatial_dim) = spatial.dim();
        let temporal_dim = temporal.ncols();
        
        let mut combined = Array2::zeros((batch_size, spatial_dim + temporal_dim));
        combined.slice_mut(s![.., ..spatial_dim]).assign(spatial);
        combined.slice_mut(s![.., spatial_dim..]).assign(temporal);
        
        combined.dot(&self.output_projection)
    }

    /// Addition fusion
    fn addition_fusion(&self, spatial: &Array2<f32>, temporal: &Array2<f32>) -> Array2<f32> {
        // Assume spatial and temporal have same dimensions
        spatial + temporal
    }

    /// Multiplication fusion
    fn multiplication_fusion(&self, spatial: &Array2<f32>, temporal: &Array2<f32>) -> Array2<f32> {
        // Element-wise multiplication
        spatial * temporal
    }

    /// Attention fusion
    fn attention_fusion(&self, spatial: &Array2<f32>, temporal: &Array2<f32>) -> Array2<f32> {
        // Simplified attention fusion
        let spatial_weight = 0.5;
        let temporal_weight = 0.5;
        
        spatial_weight * spatial + temporal_weight * temporal
    }

    /// Gated fusion
    fn gated_fusion(&self, spatial: &Array2<f32>, temporal: &Array2<f32>) -> Array2<f32> {
        // Compute gates
        let spatial_gate = spatial.dot(&self.spatial_gate);
        let temporal_gate = temporal.dot(&self.temporal_gate);
        
        // Apply sigmoid to gates
        let spatial_gate = spatial_gate.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let temporal_gate = temporal_gate.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        
        // Gated combination
        &spatial_gate * spatial + &temporal_gate * temporal
    }
}

impl SpatialTemporalGCN {
    pub fn new(
        spatial_dim: usize,
        temporal_dim: usize,
        hidden_dim: usize,
        time_steps: usize,
        num_layers: usize,
    ) -> Self {
        let mut spatial_layers = Vec::new();
        let mut temporal_layers = Vec::new();
        
        // Initialize spatial layers
        for i in 0..num_layers {
            let input_dim = if i == 0 { spatial_dim } else { hidden_dim };
            let output_dim = if i == num_layers - 1 { hidden_dim } else { hidden_dim };
            
            spatial_layers.push(SpatialConvLayer::new(
                input_dim,
                output_dim,
                SpatialConvType::GCN,
                ActivationFunction::ReLU,
            ));
        }
        
        // Initialize temporal layers
        for i in 0..num_layers {
            let input_channels = if i == 0 { temporal_dim } else { hidden_dim };
            let output_channels = if i == num_layers - 1 { hidden_dim } else { hidden_dim };
            
            temporal_layers.push(TemporalConvLayer::new(
                input_channels,
                output_channels,
                3, // kernel size
                TemporalConvType::Conv1D,
            ));
        }
        
        Self {
            spatial_layers,
            temporal_layers,
            temporal_attention: TemporalAttention::new(hidden_dim, hidden_dim),
            fusion_layer: SpatialTemporalFusion::new(hidden_dim, hidden_dim, hidden_dim, FusionType::Gated),
            time_steps,
        }
    }

    /// Forward pass through spatial-temporal GCN
    pub fn forward(
        &self,
        spatial_features: &Array2<f32>,
        temporal_features: &Array3<f32>,
        adjacency: &SparseMatrix,
    ) -> Array2<f32> {
        // Process spatial features
        let mut spatial_out = spatial_features.clone();
        for layer in &self.spatial_layers {
            spatial_out = layer.forward(&spatial_out, adjacency);
        }
        
        // Process temporal features
        let mut temporal_out = temporal_features.clone();
        for layer in &self.temporal_layers {
            temporal_out = layer.forward(&temporal_out);
        }
        
        // Apply temporal attention
        temporal_out = self.temporal_attention.forward(&temporal_out);
        
        // Aggregate temporal features (mean over time)
        let temporal_aggregated = temporal_out.mean_axis(Axis(2)).unwrap();
        
        // Fuse spatial and temporal features
        self.fusion_layer.forward(&spatial_out, &temporal_aggregated)
    }

    /// Initialize all weights
    pub fn init_weights(&mut self) {
        for layer in &mut self.spatial_layers {
            layer.init_weights();
        }
        
        for layer in &mut self.temporal_layers {
            layer.init_weights();
        }
    }
}

/// Dynamic topology updater for spatial-temporal features
pub struct DynamicTopologyProcessor {
    /// Feature history buffer
    feature_history: HashMap<String, Array2<f32>>,
    /// Topology change detector
    change_detector: TopologyChangeDetector,
    /// Incremental processor
    incremental_processor: IncrementalProcessor,
}

/// Topology change detection
pub struct TopologyChangeDetector {
    /// Previous adjacency matrix
    prev_adjacency: Option<SparseMatrix>,
    /// Change threshold
    change_threshold: f32,
}

/// Incremental processing for dynamic updates
pub struct IncrementalProcessor {
    /// Cached computations
    cached_features: HashMap<String, Array2<f32>>,
    /// Update strategy
    update_strategy: UpdateStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum UpdateStrategy {
    FullRecompute,
    Incremental,
    Selective,
}

impl TopologyChangeDetector {
    pub fn new(change_threshold: f32) -> Self {
        Self {
            prev_adjacency: None,
            change_threshold,
        }
    }

    /// Detect changes in topology
    pub fn detect_changes(&mut self, current_adjacency: &SparseMatrix) -> Vec<TopologyChange> {
        let mut changes = Vec::new();
        
        if let Some(prev) = &self.prev_adjacency {
            changes.extend(self.detect_edge_changes(prev, current_adjacency));
        }
        
        self.prev_adjacency = Some(current_adjacency.clone());
        changes
    }

    /// Detect edge changes
    fn detect_edge_changes(&self, prev: &SparseMatrix, current: &SparseMatrix) -> Vec<TopologyChange> {
        let mut changes = Vec::new();
        
        // Compare adjacency matrices
        if prev.shape != current.shape {
            changes.push(TopologyChange::StructuralChange);
            return changes;
        }
        
        // Check for edge weight changes
        let prev_dense = prev.to_dense();
        let current_dense = current.to_dense();
        
        for i in 0..prev_dense.nrows() {
            for j in 0..prev_dense.ncols() {
                let prev_val = prev_dense[[i, j]];
                let current_val = current_dense[[i, j]];
                
                if (prev_val - current_val).abs() > self.change_threshold {
                    changes.push(TopologyChange::EdgeWeightChange(i, j, prev_val, current_val));
                }
            }
        }
        
        changes
    }
}

#[derive(Debug, Clone)]
pub enum TopologyChange {
    StructuralChange,
    EdgeWeightChange(usize, usize, f32, f32),
    NodeFeatureChange(usize, Array1<f32>),
}

impl DynamicTopologyProcessor {
    pub fn new() -> Self {
        Self {
            feature_history: HashMap::new(),
            change_detector: TopologyChangeDetector::new(0.01),
            incremental_processor: IncrementalProcessor::new(UpdateStrategy::Incremental),
        }
    }

    /// Process dynamic topology updates
    pub fn process_update(
        &mut self,
        node_id: &str,
        features: &Array2<f32>,
        adjacency: &SparseMatrix,
    ) -> Array2<f32> {
        // Detect changes
        let changes = self.change_detector.detect_changes(adjacency);
        
        // Update feature history
        self.feature_history.insert(node_id.to_string(), features.clone());
        
        // Process based on changes
        self.incremental_processor.process_changes(&changes, features, adjacency)
    }
}

impl IncrementalProcessor {
    pub fn new(strategy: UpdateStrategy) -> Self {
        Self {
            cached_features: HashMap::new(),
            update_strategy: strategy,
        }
    }

    /// Process topology changes
    pub fn process_changes(
        &mut self,
        changes: &[TopologyChange],
        features: &Array2<f32>,
        adjacency: &SparseMatrix,
    ) -> Array2<f32> {
        match self.update_strategy {
            UpdateStrategy::FullRecompute => {
                // Clear cache and recompute everything
                self.cached_features.clear();
                features.clone()
            }
            UpdateStrategy::Incremental => {
                // Update only affected nodes
                self.incremental_update(changes, features, adjacency)
            }
            UpdateStrategy::Selective => {
                // Update based on change importance
                self.selective_update(changes, features, adjacency)
            }
        }
    }

    /// Incremental update
    fn incremental_update(
        &mut self,
        changes: &[TopologyChange],
        features: &Array2<f32>,
        adjacency: &SparseMatrix,
    ) -> Array2<f32> {
        let mut updated_features = features.clone();
        
        for change in changes {
            match change {
                TopologyChange::EdgeWeightChange(i, j, _, _) => {
                    // Update features for nodes i and j
                    // This is a simplified update - in practice would involve
                    // more sophisticated incremental computation
                    println!("Updating features for edge change: {} -> {}", i, j);
                }
                TopologyChange::NodeFeatureChange(i, new_features) => {
                    // Update specific node features
                    for (idx, &val) in new_features.iter().enumerate() {
                        if idx < updated_features.ncols() {
                            updated_features[[*i, idx]] = val;
                        }
                    }
                }
                TopologyChange::StructuralChange => {
                    // Major structural change - need full recompute
                    self.cached_features.clear();
                }
            }
        }
        
        updated_features
    }

    /// Selective update based on change importance
    fn selective_update(
        &mut self,
        changes: &[TopologyChange],
        features: &Array2<f32>,
        adjacency: &SparseMatrix,
    ) -> Array2<f32> {
        // Prioritize changes by importance
        let important_changes: Vec<_> = changes.iter()
            .filter(|change| self.is_important_change(change))
            .collect();
        
        if important_changes.is_empty() {
            // No important changes, return cached or current features
            features.clone()
        } else {
            // Process important changes
            self.incremental_update(&important_changes.into_iter().cloned().collect::<Vec<_>>(), features, adjacency)
        }
    }

    /// Determine if a change is important
    fn is_important_change(&self, change: &TopologyChange) -> bool {
        match change {
            TopologyChange::StructuralChange => true,
            TopologyChange::EdgeWeightChange(_, _, prev, current) => {
                (prev - current).abs() > 0.1 // Threshold for importance
            }
            TopologyChange::NodeFeatureChange(_, _) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_conv_layer() {
        let mut layer = SpatialConvLayer::new(3, 2, SpatialConvType::GCN, ActivationFunction::ReLU);
        layer.init_weights();
        
        let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let adj = SparseMatrix {
            row_ptr: vec![0, 1, 2],
            col_idx: vec![1, 0],
            values: vec![1.0, 1.0],
            shape: (2, 2),
        };
        
        let output = layer.forward(&features, &adj);
        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_temporal_conv_layer() {
        let mut layer = TemporalConvLayer::new(3, 2, 3, TemporalConvType::Conv1D);
        layer.init_weights();
        
        let input = Array3::from_shape_vec((1, 3, 5), vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
        ]).unwrap();
        
        let output = layer.forward(&input);
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 2);
    }

    #[test]
    fn test_spatial_temporal_gcn() {
        let mut stgcn = SpatialTemporalGCN::new(3, 2, 4, 5, 2);
        stgcn.init_weights();
        
        let spatial_features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let temporal_features = Array3::from_shape_vec((1, 2, 5), vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
        ]).unwrap();
        let adj = SparseMatrix {
            row_ptr: vec![0, 1, 2],
            col_idx: vec![1, 0],
            values: vec![1.0, 1.0],
            shape: (2, 2),
        };
        
        let output = stgcn.forward(&spatial_features, &temporal_features, &adj);
        assert_eq!(output.shape(), &[2, 4]);
    }
}