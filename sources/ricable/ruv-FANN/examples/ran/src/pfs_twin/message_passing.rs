use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, s};
use std::collections::HashMap;
use crate::pfs_twin::{SparseMatrix, NetworkElement, NetworkEdge, EdgeType};

/// Message Passing Neural Network for dynamic graph learning
pub struct MessagePassingNeuralNetwork {
    /// Message function
    message_fn: MessageFunction,
    /// Aggregation function
    aggregation_fn: AggregationFunction,
    /// Update function
    update_fn: UpdateFunction,
    /// Readout function
    readout_fn: ReadoutFunction,
    /// Number of message passing steps
    num_steps: usize,
}

/// Message function for computing messages between nodes
pub struct MessageFunction {
    /// Neural network layers
    layers: Vec<LinearLayer>,
    /// Activation function
    activation: ActivationFunction,
}

/// Aggregation function for combining messages
pub struct AggregationFunction {
    /// Aggregation type
    agg_type: AggregationType,
    /// Attention mechanism (if using attention aggregation)
    attention: Option<AttentionMechanism>,
}

/// Update function for updating node states
pub struct UpdateFunction {
    /// GRU or LSTM for state update
    recurrent_cell: RecurrentCell,
    /// Linear transformation
    linear: LinearLayer,
}

/// Readout function for graph-level predictions
pub struct ReadoutFunction {
    /// Set2Set or other advanced readout
    readout_type: ReadoutType,
    /// Final prediction layers
    prediction_layers: Vec<LinearLayer>,
}

#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    Sum,
    Mean,
    Max,
    Min,
    Attention,
    Set2Set,
}

#[derive(Debug, Clone, Copy)]
pub enum ReadoutType {
    GlobalSum,
    GlobalMean,
    GlobalMax,
    Set2Set,
    Attention,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    ELU,
    Swish,
}

/// Linear layer with weight matrix and bias
pub struct LinearLayer {
    /// Weight matrix
    weight: Array2<f32>,
    /// Bias vector
    bias: Array1<f32>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weight: Array2::zeros((input_dim, output_dim)),
            bias: Array1::zeros(output_dim),
            input_dim,
            output_dim,
        }
    }

    /// Initialize weights using Xavier initialization
    pub fn xavier_init(&mut self) {
        let fan_in = self.input_dim as f32;
        let fan_out = self.output_dim as f32;
        let bound = (6.0 / (fan_in + fan_out)).sqrt();
        
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                self.weight[[i, j]] = (rand::random::<f32>() - 0.5) * 2.0 * bound;
            }
        }
        
        // Initialize bias to zero
        self.bias.fill(0.0);
    }

    /// Forward pass
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let output = input.dot(&self.weight);
        
        // Add bias to each row
        let mut result = output;
        for mut row in result.axis_iter_mut(Axis(0)) {
            row += &self.bias;
        }
        
        result
    }
}

/// Recurrent cell (GRU or LSTM)
pub struct RecurrentCell {
    /// Cell type
    cell_type: CellType,
    /// Hidden dimension
    hidden_dim: usize,
    /// GRU parameters
    gru_params: Option<GRUParameters>,
    /// LSTM parameters
    lstm_params: Option<LSTMParameters>,
}

#[derive(Debug, Clone, Copy)]
pub enum CellType {
    GRU,
    LSTM,
}

/// GRU parameters
pub struct GRUParameters {
    /// Update gate weights
    w_update: Array2<f32>,
    /// Update gate bias
    b_update: Array1<f32>,
    /// Reset gate weights
    w_reset: Array2<f32>,
    /// Reset gate bias
    b_reset: Array1<f32>,
    /// New gate weights
    w_new: Array2<f32>,
    /// New gate bias
    b_new: Array1<f32>,
}

/// LSTM parameters
pub struct LSTMParameters {
    /// Input gate weights
    w_input: Array2<f32>,
    /// Input gate bias
    b_input: Array1<f32>,
    /// Forget gate weights
    w_forget: Array2<f32>,
    /// Forget gate bias
    b_forget: Array1<f32>,
    /// Cell gate weights
    w_cell: Array2<f32>,
    /// Cell gate bias
    b_cell: Array1<f32>,
    /// Output gate weights
    w_output: Array2<f32>,
    /// Output gate bias
    b_output: Array1<f32>,
}

impl RecurrentCell {
    pub fn new_gru(input_dim: usize, hidden_dim: usize) -> Self {
        let total_dim = input_dim + hidden_dim;
        
        let gru_params = GRUParameters {
            w_update: Array2::zeros((total_dim, hidden_dim)),
            b_update: Array1::zeros(hidden_dim),
            w_reset: Array2::zeros((total_dim, hidden_dim)),
            b_reset: Array1::zeros(hidden_dim),
            w_new: Array2::zeros((total_dim, hidden_dim)),
            b_new: Array1::zeros(hidden_dim),
        };
        
        Self {
            cell_type: CellType::GRU,
            hidden_dim,
            gru_params: Some(gru_params),
            lstm_params: None,
        }
    }

    pub fn new_lstm(input_dim: usize, hidden_dim: usize) -> Self {
        let total_dim = input_dim + hidden_dim;
        
        let lstm_params = LSTMParameters {
            w_input: Array2::zeros((total_dim, hidden_dim)),
            b_input: Array1::zeros(hidden_dim),
            w_forget: Array2::zeros((total_dim, hidden_dim)),
            b_forget: Array1::zeros(hidden_dim),
            w_cell: Array2::zeros((total_dim, hidden_dim)),
            b_cell: Array1::zeros(hidden_dim),
            w_output: Array2::zeros((total_dim, hidden_dim)),
            b_output: Array1::zeros(hidden_dim),
        };
        
        Self {
            cell_type: CellType::LSTM,
            hidden_dim,
            gru_params: None,
            lstm_params: Some(lstm_params),
        }
    }

    /// Forward pass through recurrent cell
    pub fn forward(&self, input: &Array1<f32>, hidden: &Array1<f32>) -> Array1<f32> {
        match self.cell_type {
            CellType::GRU => self.gru_forward(input, hidden),
            CellType::LSTM => self.lstm_forward(input, hidden),
        }
    }

    /// GRU forward pass
    fn gru_forward(&self, input: &Array1<f32>, hidden: &Array1<f32>) -> Array1<f32> {
        let params = self.gru_params.as_ref().unwrap();
        
        // Concatenate input and hidden
        let mut concat = Array1::zeros(input.len() + hidden.len());
        concat.slice_mut(s![..input.len()]).assign(input);
        concat.slice_mut(s![input.len()..]).assign(hidden);
        
        // Update gate
        let update_gate = self.sigmoid(&(concat.dot(&params.w_update) + &params.b_update));
        
        // Reset gate
        let reset_gate = self.sigmoid(&(concat.dot(&params.w_reset) + &params.b_reset));
        
        // New gate (with reset applied to hidden)
        let mut reset_hidden = hidden * &reset_gate;
        let mut new_concat = Array1::zeros(input.len() + hidden.len());
        new_concat.slice_mut(s![..input.len()]).assign(input);
        new_concat.slice_mut(s![input.len()..]).assign(&reset_hidden);
        
        let new_gate = self.tanh(&(new_concat.dot(&params.w_new) + &params.b_new));
        
        // Final hidden state
        let one_minus_update: Array1<f32> = Array1::ones(update_gate.len()) - &update_gate;
        &update_gate * hidden + &one_minus_update * &new_gate
    }

    /// LSTM forward pass
    fn lstm_forward(&self, input: &Array1<f32>, hidden: &Array1<f32>) -> Array1<f32> {
        let params = self.lstm_params.as_ref().unwrap();
        
        // Concatenate input and hidden
        let mut concat = Array1::zeros(input.len() + hidden.len());
        concat.slice_mut(s![..input.len()]).assign(input);
        concat.slice_mut(s![input.len()..]).assign(hidden);
        
        // Gates
        let input_gate = self.sigmoid(&(concat.dot(&params.w_input) + &params.b_input));
        let forget_gate = self.sigmoid(&(concat.dot(&params.w_forget) + &params.b_forget));
        let cell_gate = self.tanh(&(concat.dot(&params.w_cell) + &params.b_cell));
        let output_gate = self.sigmoid(&(concat.dot(&params.w_output) + &params.b_output));
        
        // Cell state (simplified - we're only returning hidden state)
        let cell_state = &input_gate * &cell_gate; // Simplified
        
        // Hidden state
        &output_gate * &self.tanh(&cell_state)
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Tanh activation
    fn tanh(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| v.tanh())
    }
}

/// Attention mechanism for message aggregation
pub struct AttentionMechanism {
    /// Attention type
    attention_type: AttentionType,
    /// Query transformation
    query_transform: LinearLayer,
    /// Key transformation
    key_transform: LinearLayer,
    /// Value transformation
    value_transform: LinearLayer,
    /// Output transformation
    output_transform: LinearLayer,
}

#[derive(Debug, Clone, Copy)]
pub enum AttentionType {
    Additive,
    Multiplicative,
    ScaledDotProduct,
}

impl AttentionMechanism {
    pub fn new(feature_dim: usize, attention_dim: usize, attention_type: AttentionType) -> Self {
        Self {
            attention_type,
            query_transform: LinearLayer::new(feature_dim, attention_dim),
            key_transform: LinearLayer::new(feature_dim, attention_dim),
            value_transform: LinearLayer::new(feature_dim, attention_dim),
            output_transform: LinearLayer::new(attention_dim, feature_dim),
        }
    }

    /// Apply attention to aggregate messages
    pub fn apply_attention(&self, queries: &Array2<f32>, keys: &Array2<f32>, values: &Array2<f32>) -> Array2<f32> {
        // Transform inputs
        let q = self.query_transform.forward(queries);
        let k = self.key_transform.forward(keys);
        let v = self.value_transform.forward(values);
        
        // Compute attention scores
        let attention_scores = match self.attention_type {
            AttentionType::Additive => self.additive_attention(&q, &k),
            AttentionType::Multiplicative => self.multiplicative_attention(&q, &k),
            AttentionType::ScaledDotProduct => self.scaled_dot_product_attention(&q, &k),
        };
        
        // Apply attention to values
        let attended = attention_scores.dot(&v);
        
        // Final transformation
        self.output_transform.forward(&attended)
    }

    /// Additive attention (Bahdanau attention)
    fn additive_attention(&self, queries: &Array2<f32>, keys: &Array2<f32>) -> Array2<f32> {
        let num_queries = queries.nrows();
        let num_keys = keys.nrows();
        let mut scores = Array2::zeros((num_queries, num_keys));
        
        for i in 0..num_queries {
            for j in 0..num_keys {
                let query = queries.row(i);
                let key = keys.row(j);
                let combined = &query + &key;
                
                // Simplified additive attention score
                let score = combined.iter().map(|x| x.tanh()).sum::<f32>();
                scores[[i, j]] = score;
            }
        }
        
        // Apply softmax
        self.softmax(&scores)
    }

    /// Multiplicative attention (Luong attention)
    fn multiplicative_attention(&self, queries: &Array2<f32>, keys: &Array2<f32>) -> Array2<f32> {
        let scores = queries.dot(&keys.t());
        self.softmax(&scores)
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(&self, queries: &Array2<f32>, keys: &Array2<f32>) -> Array2<f32> {
        let d_k = keys.ncols() as f32;
        let scores = queries.dot(&keys.t()) / d_k.sqrt();
        self.softmax(&scores)
    }

    /// Softmax activation
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

impl MessageFunction {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut layers = Vec::new();
        layers.push(LinearLayer::new(input_dim, hidden_dim));
        layers.push(LinearLayer::new(hidden_dim, output_dim));
        
        Self {
            layers,
            activation: ActivationFunction::ReLU,
        }
    }

    /// Compute messages between connected nodes
    pub fn compute_messages(&self, sender_features: &Array2<f32>, receiver_features: &Array2<f32>, edge_features: &Array2<f32>) -> Array2<f32> {
        // Concatenate sender, receiver, and edge features
        let batch_size = sender_features.nrows();
        let total_dim = sender_features.ncols() + receiver_features.ncols() + edge_features.ncols();
        
        let mut combined = Array2::zeros((batch_size, total_dim));
        let mut offset = 0;
        
        // Add sender features
        combined.slice_mut(s![.., offset..offset + sender_features.ncols()]).assign(sender_features);
        offset += sender_features.ncols();
        
        // Add receiver features
        combined.slice_mut(s![.., offset..offset + receiver_features.ncols()]).assign(receiver_features);
        offset += receiver_features.ncols();
        
        // Add edge features
        combined.slice_mut(s![.., offset..offset + edge_features.ncols()]).assign(edge_features);
        
        // Pass through neural network
        let mut output = combined;
        for layer in &self.layers {
            output = layer.forward(&output);
            output = self.apply_activation(&output);
        }
        
        output
    }

    /// Apply activation function
    fn apply_activation(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Tanh => x.mapv(|v| v.tanh()),
            ActivationFunction::ELU => x.mapv(|v| if v > 0.0 { v } else { (v.exp() - 1.0) }),
            ActivationFunction::Swish => x.mapv(|v| v / (1.0 + (-v).exp())),
        }
    }
}

impl AggregationFunction {
    pub fn new(agg_type: AggregationType, feature_dim: usize) -> Self {
        let attention = if let AggregationType::Attention = agg_type {
            Some(AttentionMechanism::new(feature_dim, feature_dim, AttentionType::ScaledDotProduct))
        } else {
            None
        };
        
        Self {
            agg_type,
            attention,
        }
    }

    /// Aggregate messages for each node
    pub fn aggregate(&self, messages: &Array2<f32>, edge_indices: &[(usize, usize)], num_nodes: usize) -> Array2<f32> {
        let message_dim = messages.ncols();
        let mut aggregated = Array2::zeros((num_nodes, message_dim));
        
        match self.agg_type {
            AggregationType::Sum => self.sum_aggregation(messages, edge_indices, &mut aggregated),
            AggregationType::Mean => self.mean_aggregation(messages, edge_indices, &mut aggregated),
            AggregationType::Max => self.max_aggregation(messages, edge_indices, &mut aggregated),
            AggregationType::Min => self.min_aggregation(messages, edge_indices, &mut aggregated),
            AggregationType::Attention => self.attention_aggregation(messages, edge_indices, &mut aggregated),
            AggregationType::Set2Set => self.set2set_aggregation(messages, edge_indices, &mut aggregated),
        }
        
        aggregated
    }

    /// Sum aggregation
    fn sum_aggregation(&self, messages: &Array2<f32>, edge_indices: &[(usize, usize)], aggregated: &mut Array2<f32>) {
        for (i, &(_, target)) in edge_indices.iter().enumerate() {
            let message = messages.row(i);
            let mut target_row = aggregated.row_mut(target);
            target_row += &message;
        }
    }

    /// Mean aggregation
    fn mean_aggregation(&self, messages: &Array2<f32>, edge_indices: &[(usize, usize)], aggregated: &mut Array2<f32>) {
        let mut counts = vec![0; aggregated.nrows()];
        
        for (i, &(_, target)) in edge_indices.iter().enumerate() {
            let message = messages.row(i);
            let mut target_row = aggregated.row_mut(target);
            target_row += &message;
            counts[target] += 1;
        }
        
        // Normalize by counts
        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                let mut row = aggregated.row_mut(i);
                row /= *count as f32;
            }
        }
    }

    /// Max aggregation
    fn max_aggregation(&self, messages: &Array2<f32>, edge_indices: &[(usize, usize)], aggregated: &mut Array2<f32>) {
        // Initialize with negative infinity
        aggregated.fill(f32::NEG_INFINITY);
        
        for (i, &(_, target)) in edge_indices.iter().enumerate() {
            let message = messages.row(i);
            let mut target_row = aggregated.row_mut(target);
            
            for (j, &msg_val) in message.iter().enumerate() {
                target_row[j] = target_row[j].max(msg_val);
            }
        }
        
        // Replace negative infinity with zeros
        aggregated.mapv_inplace(|v| if v == f32::NEG_INFINITY { 0.0 } else { v });
    }

    /// Min aggregation
    fn min_aggregation(&self, messages: &Array2<f32>, edge_indices: &[(usize, usize)], aggregated: &mut Array2<f32>) {
        // Initialize with positive infinity
        aggregated.fill(f32::INFINITY);
        
        for (i, &(_, target)) in edge_indices.iter().enumerate() {
            let message = messages.row(i);
            let mut target_row = aggregated.row_mut(target);
            
            for (j, &msg_val) in message.iter().enumerate() {
                target_row[j] = target_row[j].min(msg_val);
            }
        }
        
        // Replace positive infinity with zeros
        aggregated.mapv_inplace(|v| if v == f32::INFINITY { 0.0 } else { v });
    }

    /// Attention-based aggregation
    fn attention_aggregation(&self, messages: &Array2<f32>, edge_indices: &[(usize, usize)], aggregated: &mut Array2<f32>) {
        if let Some(attention) = &self.attention {
            // Group messages by target node
            let mut node_messages: HashMap<usize, Vec<usize>> = HashMap::new();
            
            for (i, &(_, target)) in edge_indices.iter().enumerate() {
                node_messages.entry(target).or_insert_with(Vec::new).push(i);
            }
            
            // Apply attention for each node
            for (target, message_indices) in node_messages {
                if message_indices.len() == 1 {
                    // Single message, no attention needed
                    let message = messages.row(message_indices[0]);
                    aggregated.row_mut(target).assign(&message);
                } else {
                    // Multiple messages, apply attention
                    let node_messages = Array2::from_shape_fn((message_indices.len(), messages.ncols()), |(i, j)| {
                        messages[[message_indices[i], j]]
                    });
                    
                    let attended = attention.apply_attention(&node_messages, &node_messages, &node_messages);
                    let aggregated_message = attended.mean_axis(Axis(0)).unwrap();
                    aggregated.row_mut(target).assign(&aggregated_message);
                }
            }
        }
    }

    /// Set2Set aggregation (simplified)
    fn set2set_aggregation(&self, messages: &Array2<f32>, edge_indices: &[(usize, usize)], aggregated: &mut Array2<f32>) {
        // Simplified Set2Set - in practice would use LSTM-based Set2Set
        self.mean_aggregation(messages, edge_indices, aggregated);
    }
}

impl UpdateFunction {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            recurrent_cell: RecurrentCell::new_gru(input_dim, hidden_dim),
            linear: LinearLayer::new(input_dim + hidden_dim, hidden_dim),
        }
    }

    /// Update node states
    pub fn update_states(&self, node_states: &Array2<f32>, aggregated_messages: &Array2<f32>) -> Array2<f32> {
        let num_nodes = node_states.nrows();
        let mut updated_states = Array2::zeros((num_nodes, self.recurrent_cell.hidden_dim));
        
        for i in 0..num_nodes {
            let current_state = node_states.row(i);
            let message = aggregated_messages.row(i);
            
            // Use recurrent cell to update state
            let new_state = self.recurrent_cell.forward(&message.to_owned(), &current_state.to_owned());
            updated_states.row_mut(i).assign(&new_state);
        }
        
        updated_states
    }
}

impl MessagePassingNeuralNetwork {
    pub fn new(
        node_feature_dim: usize,
        edge_feature_dim: usize,
        hidden_dim: usize,
        num_steps: usize,
        agg_type: AggregationType,
    ) -> Self {
        let message_input_dim = node_feature_dim * 2 + edge_feature_dim;
        
        Self {
            message_fn: MessageFunction::new(message_input_dim, hidden_dim, hidden_dim),
            aggregation_fn: AggregationFunction::new(agg_type, hidden_dim),
            update_fn: UpdateFunction::new(hidden_dim, hidden_dim),
            readout_fn: ReadoutFunction::new(hidden_dim, hidden_dim),
            num_steps,
        }
    }

    /// Forward pass through the message passing network
    pub fn forward(
        &self,
        node_features: &Array2<f32>,
        edge_features: &Array2<f32>,
        edge_indices: &[(usize, usize)],
    ) -> Array2<f32> {
        let mut current_states = node_features.clone();
        
        // Message passing steps
        for _ in 0..self.num_steps {
            // Compute messages
            let messages = self.compute_messages(&current_states, edge_features, edge_indices);
            
            // Aggregate messages
            let aggregated = self.aggregation_fn.aggregate(&messages, edge_indices, current_states.nrows());
            
            // Update node states
            current_states = self.update_fn.update_states(&current_states, &aggregated);
        }
        
        current_states
    }

    /// Compute messages between all connected nodes
    fn compute_messages(
        &self,
        node_states: &Array2<f32>,
        edge_features: &Array2<f32>,
        edge_indices: &[(usize, usize)],
    ) -> Array2<f32> {
        let num_edges = edge_indices.len();
        let message_dim = self.message_fn.layers.last().unwrap().output_dim;
        
        let mut messages = Array2::zeros((num_edges, message_dim));
        
        for (i, &(source, target)) in edge_indices.iter().enumerate() {
            let sender_features = node_states.row(source).to_owned().insert_axis(Axis(0));
            let receiver_features = node_states.row(target).to_owned().insert_axis(Axis(0));
            let edge_feature = edge_features.row(i).to_owned().insert_axis(Axis(0));
            
            let message = self.message_fn.compute_messages(&sender_features, &receiver_features, &edge_feature);
            messages.row_mut(i).assign(&message.row(0));
        }
        
        messages
    }

    /// Graph-level prediction using readout function
    pub fn graph_prediction(&self, node_states: &Array2<f32>) -> Array1<f32> {
        self.readout_fn.readout(node_states)
    }
}

impl ReadoutFunction {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut prediction_layers = Vec::new();
        prediction_layers.push(LinearLayer::new(input_dim, input_dim));
        prediction_layers.push(LinearLayer::new(input_dim, output_dim));
        
        Self {
            readout_type: ReadoutType::GlobalMean,
            prediction_layers,
        }
    }

    /// Readout function for graph-level features
    pub fn readout(&self, node_states: &Array2<f32>) -> Array1<f32> {
        let graph_representation = match self.readout_type {
            ReadoutType::GlobalSum => node_states.sum_axis(Axis(0)),
            ReadoutType::GlobalMean => node_states.mean_axis(Axis(0)).unwrap(),
            ReadoutType::GlobalMax => {
                let mut max_vals = Array1::from_elem(node_states.ncols(), f32::NEG_INFINITY);
                for row in node_states.axis_iter(Axis(0)) {
                    for (i, &val) in row.iter().enumerate() {
                        max_vals[i] = max_vals[i].max(val);
                    }
                }
                max_vals
            }
            ReadoutType::Set2Set => {
                // Simplified Set2Set
                node_states.mean_axis(Axis(0)).unwrap()
            }
            ReadoutType::Attention => {
                // Simplified attention readout
                node_states.mean_axis(Axis(0)).unwrap()
            }
        };
        
        // Pass through prediction layers
        let mut output = graph_representation.insert_axis(Axis(0));
        for layer in &self.prediction_layers {
            output = layer.forward(&output);
        }
        
        output.remove_axis(Axis(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let mut layer = LinearLayer::new(3, 2);
        layer.xavier_init();
        
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = layer.forward(&input);
        
        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_gru_cell() {
        let cell = RecurrentCell::new_gru(3, 4);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let hidden = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        
        let output = cell.forward(&input, &hidden);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_message_passing_network() {
        let mpnn = MessagePassingNeuralNetwork::new(3, 2, 4, 2, AggregationType::Mean);
        
        let node_features = Array2::from_shape_vec((3, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        
        let edge_features = Array2::from_shape_vec((2, 2), vec![
            0.1, 0.2,
            0.3, 0.4,
        ]).unwrap();
        
        let edge_indices = vec![(0, 1), (1, 2)];
        
        let output = mpnn.forward(&node_features, &edge_features, &edge_indices);
        assert_eq!(output.shape(), &[3, 4]);
    }
}