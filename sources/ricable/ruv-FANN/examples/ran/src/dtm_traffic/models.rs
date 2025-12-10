use ndarray::{Array2, Array3, Array4, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
// use std::f32::consts::PI;

use crate::dtm_traffic::{PredictorConfig, cuda_kernels};

/// LSTM cell with custom gates for traffic prediction
#[derive(Debug)]
pub struct LSTMCell {
    // Weight matrices
    pub w_ii: Array2<f32>,  // Input to input gate
    pub w_if: Array2<f32>,  // Input to forget gate
    pub w_ig: Array2<f32>,  // Input to cell gate
    pub w_io: Array2<f32>,  // Input to output gate
    
    pub w_hi: Array2<f32>,  // Hidden to input gate
    pub w_hf: Array2<f32>,  // Hidden to forget gate
    pub w_hg: Array2<f32>,  // Hidden to cell gate
    pub w_ho: Array2<f32>,  // Hidden to output gate
    
    // Bias vectors
    pub b_i: Array2<f32>,   // Input gate bias
    pub b_f: Array2<f32>,   // Forget gate bias
    pub b_g: Array2<f32>,   // Cell gate bias
    pub b_o: Array2<f32>,   // Output gate bias
    
    // Peephole connections (optional)
    pub w_ci: Option<Array2<f32>>,  // Cell to input gate
    pub w_cf: Option<Array2<f32>>,  // Cell to forget gate
    pub w_co: Option<Array2<f32>>,  // Cell to output gate
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize, use_peepholes: bool) -> Self {
        let glorot_scale = (2.0 / (input_size + hidden_size) as f32).sqrt();
        
        // Initialize weights with Glorot uniform initialization
        let w_ii = Array2::random((input_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_if = Array2::random((input_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_ig = Array2::random((input_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_io = Array2::random((input_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        
        let w_hi = Array2::random((hidden_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_hf = Array2::random((hidden_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_hg = Array2::random((hidden_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_ho = Array2::random((hidden_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        
        // Initialize biases to small values
        let b_i = Array2::zeros((1, hidden_size));
        let b_f = Array2::ones((1, hidden_size));  // Forget gate bias initialized to 1
        let b_g = Array2::zeros((1, hidden_size));
        let b_o = Array2::zeros((1, hidden_size));
        
        // Peephole connections
        let (w_ci, w_cf, w_co) = if use_peepholes {
            (
                Some(Array2::random((hidden_size, hidden_size), Uniform::new(-0.01, 0.01))),
                Some(Array2::random((hidden_size, hidden_size), Uniform::new(-0.01, 0.01))),
                Some(Array2::random((hidden_size, hidden_size), Uniform::new(-0.01, 0.01))),
            )
        } else {
            (None, None, None)
        };
        
        Self {
            w_ii, w_if, w_ig, w_io,
            w_hi, w_hf, w_hg, w_ho,
            b_i, b_f, b_g, b_o,
            w_ci, w_cf, w_co,
        }
    }
    
    /// Forward pass through LSTM cell
    pub fn forward(&self, x: &Array2<f32>, h_prev: &Array2<f32>, c_prev: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let batch_size = x.shape()[0];
        let hidden_size = h_prev.shape()[1];
        
        // Input gate
        let mut i_gate = x.dot(&self.w_ii) + h_prev.dot(&self.w_hi) + &self.b_i;
        if let Some(ref w_ci) = self.w_ci {
            i_gate = i_gate + c_prev.dot(w_ci);
        }
        let i_gate = sigmoid(&i_gate);
        
        // Forget gate
        let mut f_gate = x.dot(&self.w_if) + h_prev.dot(&self.w_hf) + &self.b_f;
        if let Some(ref w_cf) = self.w_cf {
            f_gate = f_gate + c_prev.dot(w_cf);
        }
        let f_gate = sigmoid(&f_gate);
        
        // Cell gate
        let g_gate = x.dot(&self.w_ig) + h_prev.dot(&self.w_hg) + &self.b_g;
        let g_gate = tanh(&g_gate);
        
        // New cell state
        let c_new = &f_gate * c_prev + &i_gate * &g_gate;
        
        // Output gate
        let mut o_gate = x.dot(&self.w_io) + h_prev.dot(&self.w_ho) + &self.b_o;
        if let Some(ref w_co) = self.w_co {
            o_gate = o_gate + c_new.dot(w_co);
        }
        let o_gate = sigmoid(&o_gate);
        
        // New hidden state
        let h_new = &o_gate * &tanh(&c_new);
        
        (h_new, c_new)
    }
    
    /// Forward pass using CUDA kernels for acceleration
    pub fn forward_cuda(&self, x: &Array2<f32>, h_prev: &Array2<f32>, c_prev: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        cuda_kernels::lstm_forward_cuda(
            x, h_prev, c_prev,
            &self.w_ii, &self.w_if, &self.w_ig, &self.w_io,
            &self.w_hi, &self.w_hf, &self.w_hg, &self.w_ho,
            &self.b_i, &self.b_f, &self.b_g, &self.b_o,
            self.w_ci.as_ref(), self.w_cf.as_ref(), self.w_co.as_ref(),
        )
    }
}

/// GRU cell with custom gates
#[derive(Debug)]
pub struct GRUCell {
    // Weight matrices
    pub w_ir: Array2<f32>,  // Input to reset gate
    pub w_iz: Array2<f32>,  // Input to update gate
    pub w_in: Array2<f32>,  // Input to new gate
    
    pub w_hr: Array2<f32>,  // Hidden to reset gate
    pub w_hz: Array2<f32>,  // Hidden to update gate
    pub w_hn: Array2<f32>,  // Hidden to new gate
    
    // Bias vectors
    pub b_r: Array2<f32>,   // Reset gate bias
    pub b_z: Array2<f32>,   // Update gate bias
    pub b_n: Array2<f32>,   // New gate bias
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let glorot_scale = (2.0 / (input_size + hidden_size) as f32).sqrt();
        
        // Initialize weights
        let w_ir = Array2::random((input_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_iz = Array2::random((input_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_in = Array2::random((input_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        
        let w_hr = Array2::random((hidden_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_hz = Array2::random((hidden_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        let w_hn = Array2::random((hidden_size, hidden_size), Uniform::new(-glorot_scale, glorot_scale));
        
        // Initialize biases
        let b_r = Array2::zeros((1, hidden_size));
        let b_z = Array2::zeros((1, hidden_size));
        let b_n = Array2::zeros((1, hidden_size));
        
        Self {
            w_ir, w_iz, w_in,
            w_hr, w_hz, w_hn,
            b_r, b_z, b_n,
        }
    }
    
    /// Forward pass through GRU cell
    pub fn forward(&self, x: &Array2<f32>, h_prev: &Array2<f32>) -> Array2<f32> {
        // Reset gate
        let r_gate = sigmoid(&(x.dot(&self.w_ir) + h_prev.dot(&self.w_hr) + &self.b_r));
        
        // Update gate
        let z_gate = sigmoid(&(x.dot(&self.w_iz) + h_prev.dot(&self.w_hz) + &self.b_z));
        
        // New gate
        let n_gate = tanh(&(x.dot(&self.w_in) + (&r_gate * h_prev).dot(&self.w_hn) + &self.b_n));
        
        // New hidden state
        &z_gate * h_prev + (1.0 - &z_gate) * &n_gate
    }
    
    /// Forward pass using CUDA kernels
    pub fn forward_cuda(&self, x: &Array2<f32>, h_prev: &Array2<f32>) -> Array2<f32> {
        cuda_kernels::gru_forward_cuda(
            x, h_prev,
            &self.w_ir, &self.w_iz, &self.w_in,
            &self.w_hr, &self.w_hz, &self.w_hn,
            &self.b_r, &self.b_z, &self.b_n,
        )
    }
}

/// LSTM-based traffic predictor
pub struct LSTMPredictor {
    cells: Vec<LSTMCell>,
    output_layer: Array2<f32>,
    hidden_size: usize,
    use_cuda: bool,
}

impl LSTMPredictor {
    pub fn new(config: &PredictorConfig) -> Self {
        let input_size = 22;  // Feature dimension
        let hidden_size = config.lstm_hidden_size;
        let output_size = config.forecast_horizons.len() * 21;  // Horizons * output features
        
        // Stack 2 LSTM layers
        let cells = vec![
            LSTMCell::new(input_size, hidden_size, true),
            LSTMCell::new(hidden_size, hidden_size, true),
        ];
        
        let output_layer = Array2::random((hidden_size, output_size), Uniform::new(-0.1, 0.1));
        
        Self {
            cells,
            output_layer,
            hidden_size,
            use_cuda: cuda_kernels::is_cuda_available(),
        }
    }
    
    pub fn train(&mut self, features: &Array4<f32>, targets: &Array4<f32>, epochs: usize) -> Result<(), String> {
        // Simplified training loop - in practice, use proper optimizer
        for epoch in 0..epochs {
            let loss = self.forward_backward(features, targets)?;
            if epoch % 10 == 0 {
                println!("LSTM Epoch {}: Loss = {:.4}", epoch, loss);
            }
        }
        Ok(())
    }
    
    pub fn predict(&self, features: &Array3<f32>) -> Result<Array3<f32>, String> {
        let batch_size = features.shape()[0];
        let seq_length = features.shape()[1];
        
        // Initialize hidden and cell states
        let mut h_states = vec![Array2::zeros((batch_size, self.hidden_size)); self.cells.len()];
        let mut c_states = vec![Array2::zeros((batch_size, self.hidden_size)); self.cells.len()];
        
        // Process sequence
        for t in 0..seq_length {
            let x_t = features.slice(s![.., t, ..]).to_owned();
            
            // First LSTM layer
            let (h_new, c_new) = if self.use_cuda {
                self.cells[0].forward_cuda(&x_t, &h_states[0], &c_states[0])
            } else {
                self.cells[0].forward(&x_t, &h_states[0], &c_states[0])
            };
            h_states[0] = h_new;
            c_states[0] = c_new;
            
            // Second LSTM layer
            let (h_new, c_new) = if self.use_cuda {
                self.cells[1].forward_cuda(&h_states[0], &h_states[1], &c_states[1])
            } else {
                self.cells[1].forward(&h_states[0], &h_states[1], &c_states[1])
            };
            h_states[1] = h_new;
            c_states[1] = c_new;
        }
        
        // Output projection
        let output = h_states[1].dot(&self.output_layer);
        
        // Reshape to [batch, horizons, features]
        let n_horizons = output.shape()[1] / 21;
        Ok(output.into_shape((batch_size, n_horizons, 21)).unwrap())
    }
    
    fn forward_backward(&mut self, features: &Array4<f32>, targets: &Array4<f32>) -> Result<f32, String> {
        // Simplified forward-backward pass
        // In practice, implement proper backpropagation through time (BPTT)
        Ok(0.01)  // Placeholder loss
    }
}

/// GRU-based traffic predictor
pub struct GRUPredictor {
    cells: Vec<GRUCell>,
    output_layer: Array2<f32>,
    hidden_size: usize,
    use_cuda: bool,
}

impl GRUPredictor {
    pub fn new(config: &PredictorConfig) -> Self {
        let input_size = 22;
        let hidden_size = config.gru_hidden_size;
        let output_size = config.forecast_horizons.len() * 21;
        
        let cells = vec![
            GRUCell::new(input_size, hidden_size),
            GRUCell::new(hidden_size, hidden_size),
        ];
        
        let output_layer = Array2::random((hidden_size, output_size), Uniform::new(-0.1, 0.1));
        
        Self {
            cells,
            output_layer,
            hidden_size,
            use_cuda: cuda_kernels::is_cuda_available(),
        }
    }
    
    pub fn train(&mut self, features: &Array4<f32>, targets: &Array4<f32>, epochs: usize) -> Result<(), String> {
        for epoch in 0..epochs {
            let loss = self.forward_backward(features, targets)?;
            if epoch % 10 == 0 {
                println!("GRU Epoch {}: Loss = {:.4}", epoch, loss);
            }
        }
        Ok(())
    }
    
    pub fn predict(&self, features: &Array3<f32>) -> Result<Array3<f32>, String> {
        let batch_size = features.shape()[0];
        let seq_length = features.shape()[1];
        
        let mut h_states = vec![Array2::zeros((batch_size, self.hidden_size)); self.cells.len()];
        
        for t in 0..seq_length {
            let x_t = features.slice(s![.., t, ..]).to_owned();
            
            // First GRU layer
            h_states[0] = if self.use_cuda {
                self.cells[0].forward_cuda(&x_t, &h_states[0])
            } else {
                self.cells[0].forward(&x_t, &h_states[0])
            };
            
            // Second GRU layer
            h_states[1] = if self.use_cuda {
                self.cells[1].forward_cuda(&h_states[0], &h_states[1])
            } else {
                self.cells[1].forward(&h_states[0], &h_states[1])
            };
        }
        
        let output = h_states[1].dot(&self.output_layer);
        let n_horizons = output.shape()[1] / 21;
        Ok(output.into_shape((batch_size, n_horizons, 21)).unwrap())
    }
    
    fn forward_backward(&mut self, features: &Array4<f32>, targets: &Array4<f32>) -> Result<f32, String> {
        Ok(0.01)  // Placeholder
    }
}

/// Temporal Convolutional Network block
#[derive(Debug)]
pub struct TCNBlock {
    conv1: Conv1D,
    conv2: Conv1D,
    dropout_rate: f32,
    residual_conv: Option<Conv1D>,
}

impl TCNBlock {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, dilation: usize, dropout_rate: f32) -> Self {
        let padding = (kernel_size - 1) * dilation;
        
        let conv1 = Conv1D::new(in_channels, out_channels, kernel_size, dilation, padding);
        let conv2 = Conv1D::new(out_channels, out_channels, kernel_size, dilation, padding);
        
        let residual_conv = if in_channels != out_channels {
            Some(Conv1D::new(in_channels, out_channels, 1, 1, 0))
        } else {
            None
        };
        
        Self {
            conv1,
            conv2,
            dropout_rate,
            residual_conv,
        }
    }
    
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        // First convolution
        let out1 = self.conv1.forward(x);
        let out1 = relu(&out1);
        let out1 = dropout(&out1, self.dropout_rate);
        
        // Second convolution
        let out2 = self.conv2.forward(&out1);
        let out2 = relu(&out2);
        let out2 = dropout(&out2, self.dropout_rate);
        
        // Residual connection
        let residual = if let Some(ref conv) = self.residual_conv {
            conv.forward(x)
        } else {
            x.clone()
        };
        
        &out2 + &residual
    }
}

/// 1D Convolution layer
#[derive(Debug)]
pub struct Conv1D {
    weights: Array3<f32>,  // [out_channels, in_channels, kernel_size]
    bias: Array2<f32>,     // [1, out_channels]
    kernel_size: usize,
    dilation: usize,
    padding: usize,
}

impl Conv1D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, dilation: usize, padding: usize) -> Self {
        let scale = (2.0 / (in_channels * kernel_size) as f32).sqrt();
        let weights = Array3::random((out_channels, in_channels, kernel_size), Uniform::new(-scale, scale));
        let bias = Array2::zeros((1, out_channels));
        
        Self {
            weights,
            bias,
            kernel_size,
            dilation,
            padding,
        }
    }
    
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];
        let in_channels = x.shape()[2];
        let out_channels = self.weights.shape()[0];
        
        // Apply padding
        let padded_length = seq_length + 2 * self.padding;
        let mut padded = Array3::zeros((batch_size, padded_length, in_channels));
        padded.slice_mut(s![.., self.padding..self.padding + seq_length, ..]).assign(x);
        
        // Compute output length
        let out_length = (padded_length - self.dilation * (self.kernel_size - 1) - 1) + 1;
        let mut output = Array3::zeros((batch_size, out_length, out_channels));
        
        // Perform convolution
        for b in 0..batch_size {
            for t in 0..out_length {
                for oc in 0..out_channels {
                    let mut sum = self.bias[[0, oc]];
                    
                    for ic in 0..in_channels {
                        for k in 0..self.kernel_size {
                            let idx = t + k * self.dilation;
                            if idx < padded_length {
                                sum += padded[[b, idx, ic]] * self.weights[[oc, ic, k]];
                            }
                        }
                    }
                    
                    output[[b, t, oc]] = sum;
                }
            }
        }
        
        output
    }
}

/// TCN-based traffic predictor
pub struct TCNPredictor {
    blocks: Vec<TCNBlock>,
    output_layer: Array2<f32>,
    use_cuda: bool,
}

impl TCNPredictor {
    pub fn new(config: &PredictorConfig) -> Self {
        let input_size = 22;
        let n_channels = config.tcn_filters;
        let output_size = config.forecast_horizons.len() * 21;
        
        // Build TCN blocks with increasing dilations
        let mut blocks = Vec::new();
        let mut in_channels = input_size;
        
        for &dilation in &config.tcn_dilations {
            blocks.push(TCNBlock::new(
                in_channels,
                n_channels,
                config.tcn_kernel_size,
                dilation,
                config.dropout_rate,
            ));
            in_channels = n_channels;
        }
        
        let output_layer = Array2::random((n_channels, output_size), Uniform::new(-0.1, 0.1));
        
        Self {
            blocks,
            output_layer,
            use_cuda: cuda_kernels::is_cuda_available(),
        }
    }
    
    pub fn train(&mut self, features: &Array4<f32>, targets: &Array4<f32>, epochs: usize) -> Result<(), String> {
        for epoch in 0..epochs {
            let loss = self.forward_backward(features, targets)?;
            if epoch % 10 == 0 {
                println!("TCN Epoch {}: Loss = {:.4}", epoch, loss);
            }
        }
        Ok(())
    }
    
    pub fn predict(&self, features: &Array3<f32>) -> Result<Array3<f32>, String> {
        let batch_size = features.shape()[0];
        
        // Transform input: [batch, seq, features] -> [batch, seq, channels]
        let mut x = features.clone();
        
        // Pass through TCN blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }
        
        // Global average pooling over sequence dimension
        let pooled = x.mean_axis(Axis(1)).unwrap();
        
        // Output projection
        let output = pooled.dot(&self.output_layer);
        
        // Reshape to [batch, horizons, features]
        let n_horizons = output.shape()[1] / 21;
        Ok(output.into_shape((batch_size, n_horizons, 21)).unwrap())
    }
    
    fn forward_backward(&mut self, features: &Array4<f32>, targets: &Array4<f32>) -> Result<f32, String> {
        Ok(0.01)  // Placeholder
    }
}

// Activation functions
fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|a| 1.0 / (1.0 + (-a).exp()))
}

fn tanh(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|a| a.tanh())
}

fn relu(x: &Array3<f32>) -> Array3<f32> {
    x.mapv(|a| a.max(0.0))
}

fn dropout(x: &Array3<f32>, rate: f32) -> Array3<f32> {
    // Simplified dropout - in practice, use proper random mask during training
    if rate > 0.0 {
        x * (1.0 - rate)
    } else {
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lstm_cell() {
        let cell = LSTMCell::new(10, 20, true);
        let x = Array2::zeros((2, 10));
        let h = Array2::zeros((2, 20));
        let c = Array2::zeros((2, 20));
        
        let (h_new, c_new) = cell.forward(&x, &h, &c);
        assert_eq!(h_new.shape(), &[2, 20]);
        assert_eq!(c_new.shape(), &[2, 20]);
    }
    
    #[test]
    fn test_gru_cell() {
        let cell = GRUCell::new(10, 20);
        let x = Array2::zeros((2, 10));
        let h = Array2::zeros((2, 20));
        
        let h_new = cell.forward(&x, &h);
        assert_eq!(h_new.shape(), &[2, 20]);
    }
    
    #[test]
    fn test_conv1d() {
        let conv = Conv1D::new(3, 16, 3, 1, 1);
        let x = Array3::zeros((2, 10, 3));
        
        let output = conv.forward(&x);
        assert_eq!(output.shape(), &[2, 10, 16]);
    }
}