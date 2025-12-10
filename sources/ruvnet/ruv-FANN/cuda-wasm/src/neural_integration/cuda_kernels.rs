//! Pre-optimized CUDA Kernels for Neural Operations
//!
//! This module contains hand-optimized CUDA kernels for common neural network
//! operations, designed for maximum performance and efficiency.

use super::{ActivationFunction, NeuralResult, NeuralIntegrationError};

/// Collection of optimized CUDA kernels for neural operations
pub struct OptimizedKernels;

impl OptimizedKernels {
    /// Get optimized matrix multiplication kernel
    pub fn matrix_multiply_kernel(rows_a: usize, cols_a: usize, cols_b: usize) -> &'static str {
        r#"
extern "C" __global__ void optimized_matrix_multiply(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tiling
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Calculate global position
    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + 31) / 32; ++t) {
        // Load tiles into shared memory
        int a_col = t * 32 + tx;
        int b_row = t * 32 + ty;
        
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum using shared memory
        #pragma unroll 32
        for (int k = 0; k < 32; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"#
    }
    
    /// Get optimized vector operations kernel
    pub fn vector_operations_kernel() -> &'static str {
        r#"
extern "C" __global__ void optimized_vector_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better memory coalescing
    for (int i = idx; i < size; i += stride) {
        result[i] = a[i] + b[i];
    }
}

extern "C" __global__ void optimized_vector_multiply(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        result[i] = a[i] * b[i];
    }
}

extern "C" __global__ void optimized_vector_scale(
    const float* __restrict__ input,
    float scale,
    float* __restrict__ result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        result[i] = input[i] * scale;
    }
}
"#
    }
    
    /// Get optimized activation functions kernel
    pub fn activation_functions_kernel() -> &'static str {
        r#"
// Fast approximation functions
__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float fast_tanh(float x) {
    // Fast tanh approximation using polynomial
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

__device__ __forceinline__ float fast_gelu(float x) {
    // Fast GELU approximation
    return 0.5f * x * (1.0f + fast_tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void optimized_activation_functions(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    int activation_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        float result;
        
        switch (activation_type) {
            case 0: // Sigmoid
                result = fast_sigmoid(x);
                break;
            case 1: // ReLU
                result = fmaxf(0.0f, x);
                break;
            case 2: // Tanh
                result = fast_tanh(x);
                break;
            case 3: // Leaky ReLU
                result = x > 0.0f ? x : 0.01f * x;
                break;
            case 4: // Swish
                result = x * fast_sigmoid(x);
                break;
            case 5: // GELU
                result = fast_gelu(x);
                break;
            default:
                result = x; // Linear
        }
        
        output[i] = result;
    }
}

extern "C" __global__ void optimized_activation_derivatives(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    int activation_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        float result;
        
        switch (activation_type) {
            case 0: { // Sigmoid derivative
                float s = fast_sigmoid(x);
                result = s * (1.0f - s);
                break;
            }
            case 1: // ReLU derivative
                result = x > 0.0f ? 1.0f : 0.0f;
                break;
            case 2: { // Tanh derivative
                float t = fast_tanh(x);
                result = 1.0f - t * t;
                break;
            }
            case 3: // Leaky ReLU derivative
                result = x > 0.0f ? 1.0f : 0.01f;
                break;
            case 4: { // Swish derivative
                float s = fast_sigmoid(x);
                result = s + x * s * (1.0f - s);
                break;
            }
            case 5: { // GELU derivative (approximation)
                float tanh_arg = 0.7978845608f * (x + 0.044715f * x * x * x);
                float tanh_val = fast_tanh(tanh_arg);
                result = 0.5f * (1.0f + tanh_val) + x * 0.5f * (1.0f - tanh_val * tanh_val) * 
                        0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
                break;
            }
            default:
                result = 1.0f; // Linear derivative
        }
        
        output[i] = result;
    }
}
"#
    }
    
    /// Get optimized convolution kernel
    pub fn convolution_kernel() -> &'static str {
        r#"
extern "C" __global__ void optimized_conv2d(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding
) {
    // Shared memory for input tile
    extern __shared__ float shared_input[];
    
    int batch = blockIdx.z;
    int out_channel = blockIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = threadIdx.y;
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    if (out_x >= out_width || out_y >= out_height) return;
    
    float sum = 0.0f;
    
    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y * stride - padding + ky;
                int in_x = out_x * stride - padding + kx;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = batch * in_channels * in_height * in_width +
                                   in_channel * in_height * in_width +
                                   in_y * in_width + in_x;
                    
                    int kernel_idx = out_channel * in_channels * kernel_size * kernel_size +
                                    in_channel * kernel_size * kernel_size +
                                    ky * kernel_size + kx;
                    
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }
    
    int output_idx = batch * out_channels * out_height * out_width +
                     out_channel * out_height * out_width +
                     out_y * out_width + out_x;
    
    output[output_idx] = sum;
}
"#
    }
    
    /// Get optimized forward propagation kernel
    pub fn forward_propagation_kernel() -> &'static str {
        r#"
extern "C" __global__ void optimized_forward_propagation(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int output_size,
    int activation_type
) {
    // Shared memory for weight tiles
    __shared__ float weight_tile[32][32];
    __shared__ float input_tile[32];
    
    int batch = blockIdx.z;
    int output_neuron = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (output_neuron >= output_size) return;
    
    float sum = 0.0f;
    
    // Process input in tiles
    for (int tile = 0; tile < (input_size + 31) / 32; ++tile) {
        int input_idx = tile * 32 + tx;
        
        // Load input tile
        if (input_idx < input_size && ty == 0) {
            input_tile[tx] = input[batch * input_size + input_idx];
        } else if (ty == 0) {
            input_tile[tx] = 0.0f;
        }
        
        // Load weight tile
        if (input_idx < input_size) {
            weight_tile[ty][tx] = weights[output_neuron * input_size + input_idx];
        } else {
            weight_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll 32
        for (int k = 0; k < 32; ++k) {
            sum += weight_tile[ty][k] * input_tile[k];
        }
        
        __syncthreads();
    }
    
    // Add bias
    sum += biases[output_neuron];
    
    // Apply activation function
    float result;
    switch (activation_type) {
        case 0: // Sigmoid
            result = 1.0f / (1.0f + expf(-sum));
            break;
        case 1: // ReLU
            result = fmaxf(0.0f, sum);
            break;
        case 2: // Tanh
            result = tanhf(sum);
            break;
        default:
            result = sum; // Linear
    }
    
    output[batch * output_size + output_neuron] = result;
}
"#
    }
    
    /// Get optimized backward propagation kernel
    pub fn backward_propagation_kernel() -> &'static str {
        r#"
extern "C" __global__ void optimized_backward_propagation(
    const float* __restrict__ delta_output,
    const float* __restrict__ weights,
    const float* __restrict__ activations,
    float* __restrict__ delta_input,
    float* __restrict__ weight_gradients,
    float* __restrict__ bias_gradients,
    int batch_size,
    int input_size,
    int output_size,
    int activation_type
) {
    __shared__ float delta_shared[32];
    __shared__ float activation_shared[32];
    
    int batch = blockIdx.z;
    int input_neuron = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    
    if (input_neuron >= input_size) return;
    
    float delta_sum = 0.0f;
    
    // Process output deltas in tiles
    for (int tile = 0; tile < (output_size + 31) / 32; ++tile) {
        int output_idx = tile * 32 + tx;
        
        // Load delta tile
        if (output_idx < output_size) {
            delta_shared[tx] = delta_output[batch * output_size + output_idx];
        } else {
            delta_shared[tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute delta contribution
        #pragma unroll 32
        for (int k = 0; k < 32; ++k) {
            int out_neuron = tile * 32 + k;
            if (out_neuron < output_size) {
                delta_sum += delta_shared[k] * weights[out_neuron * input_size + input_neuron];
            }
        }
        
        __syncthreads();
    }
    
    // Apply activation derivative
    float activation = activations[batch * input_size + input_neuron];
    float derivative;
    
    switch (activation_type) {
        case 0: // Sigmoid derivative
            derivative = activation * (1.0f - activation);
            break;
        case 1: // ReLU derivative
            derivative = activation > 0.0f ? 1.0f : 0.0f;
            break;
        case 2: // Tanh derivative
            derivative = 1.0f - activation * activation;
            break;
        default:
            derivative = 1.0f; // Linear derivative
    }
    
    delta_input[batch * input_size + input_neuron] = delta_sum * derivative;
}

extern "C" __global__ void optimized_compute_gradients(
    const float* __restrict__ delta_output,
    const float* __restrict__ input_activations,
    float* __restrict__ weight_gradients,
    float* __restrict__ bias_gradients,
    int batch_size,
    int input_size,
    int output_size,
    float learning_rate
) {
    int input_neuron = blockIdx.x * blockDim.x + threadIdx.x;
    int output_neuron = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (input_neuron >= input_size || output_neuron >= output_size) return;
    
    float gradient_sum = 0.0f;
    
    // Accumulate gradients across batch
    for (int batch = 0; batch < batch_size; ++batch) {
        float delta = delta_output[batch * output_size + output_neuron];
        float activation = input_activations[batch * input_size + input_neuron];
        gradient_sum += delta * activation;
    }
    
    // Update weight gradient
    int weight_idx = output_neuron * input_size + input_neuron;
    weight_gradients[weight_idx] = gradient_sum / batch_size;
    
    // Update bias gradient (only for first input neuron to avoid race conditions)
    if (input_neuron == 0) {
        float bias_gradient = 0.0f;
        for (int batch = 0; batch < batch_size; ++batch) {
            bias_gradient += delta_output[batch * output_size + output_neuron];
        }
        bias_gradients[output_neuron] = bias_gradient / batch_size;
    }
}
"#
    }
    
    /// Get optimized reduction operations kernel
    pub fn reduction_operations_kernel() -> &'static str {
        r#"
extern "C" __global__ void optimized_reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void optimized_reduce_max(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? input[i] : -INFINITY;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void optimized_softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int size
) {
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size) return;
    
    const float* batch_input = input + batch * size;
    float* batch_output = output + batch * size;
    
    // Find maximum value
    float local_max = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, batch_input[i]);
    }
    
    // Reduce maximum across threads
    local_max = blockReduceMax(local_max);
    if (tid == 0) max_val = local_max;
    __syncthreads();
    
    // Compute sum of exponentials
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        local_sum += expf(batch_input[i] - max_val);
    }
    
    // Reduce sum across threads
    local_sum = blockReduceSum(local_sum);
    if (tid == 0) sum_exp = local_sum;
    __syncthreads();
    
    // Compute softmax
    for (int i = tid; i < size; i += blockDim.x) {
        batch_output[i] = expf(batch_input[i] - max_val) / sum_exp;
    }
}

__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__device__ float blockReduceMax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
"#
    }
    
    /// Get kernel for specific activation function
    pub fn get_activation_kernel(function: ActivationFunction) -> NeuralResult<String> {
        let activation_code = match function {
            ActivationFunction::Sigmoid => "fast_sigmoid(x)",
            ActivationFunction::ReLU => "fmaxf(0.0f, x)",
            ActivationFunction::Tanh => "fast_tanh(x)",
            ActivationFunction::LeakyReLU => "x > 0.0f ? x : 0.01f * x",
            ActivationFunction::Swish => "x * fast_sigmoid(x)",
            ActivationFunction::GELU => "fast_gelu(x)",
        };
        
        let kernel = format!(r#"
{}

extern "C" __global__ void specialized_activation(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {{
        float x = input[i];
        output[i] = {};
    }}
}}
"#, Self::activation_functions_kernel(), activation_code);
        
        Ok(kernel)
    }
    
    /// Get all kernels as a single compilation unit
    pub fn get_combined_kernels() -> &'static str {
        r#"
// Combined optimized kernels for neural operations
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Fast math approximations
__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + fast_tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Warp-level reductions
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reductions
__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__device__ float blockReduceMax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// Include all optimized kernels
// Matrix operations
extern "C" __global__ void optimized_matrix_multiply(...);

// Vector operations  
extern "C" __global__ void optimized_vector_add(...);
extern "C" __global__ void optimized_vector_multiply(...);
extern "C" __global__ void optimized_vector_scale(...);

// Activation functions
extern "C" __global__ void optimized_activation_functions(...);
extern "C" __global__ void optimized_activation_derivatives(...);

// Neural network layers
extern "C" __global__ void optimized_forward_propagation(...);
extern "C" __global__ void optimized_backward_propagation(...);
extern "C" __global__ void optimized_compute_gradients(...);

// Convolution
extern "C" __global__ void optimized_conv2d(...);

// Reductions
extern "C" __global__ void optimized_reduce_sum(...);
extern "C" __global__ void optimized_reduce_max(...);
extern "C" __global__ void optimized_softmax(...);
"#
    }
}

/// Kernel configuration parameters
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub block_size: (u32, u32, u32),
    pub grid_size: (u32, u32, u32),
    pub shared_memory_size: u32,
}

impl KernelConfig {
    /// Create optimal configuration for matrix multiplication
    pub fn for_matrix_multiply(rows: usize, cols: usize) -> Self {
        let block_x = 32u32;
        let block_y = 32u32;
        let grid_x = (cols as u32).div_ceil(block_x);
        let grid_y = (rows as u32).div_ceil(block_y);
        
        Self {
            block_size: (block_x, block_y, 1),
            grid_size: (grid_x, grid_y, 1),
            shared_memory_size: block_x * block_y * 4 * 2, // Two tiles of f32
        }
    }
    
    /// Create optimal configuration for vector operations
    pub fn for_vector_operation(size: usize) -> Self {
        let block_size = 256u32;
        let grid_size = (size as u32).div_ceil(block_size);
        
        Self {
            block_size: (block_size, 1, 1),
            grid_size: (grid_size, 1, 1),
            shared_memory_size: 0,
        }
    }
    
    /// Create optimal configuration for convolution
    pub fn for_convolution(batch_size: usize, out_channels: usize, out_height: usize, out_width: usize) -> Self {
        let block_x = 16u32;
        let block_y = 16u32;
        let grid_x = (out_width as u32).div_ceil(block_x);
        let grid_y = (out_channels as u32).div_ceil(block_y);
        let grid_z = batch_size as u32;
        
        Self {
            block_size: (block_x, block_y, 1),
            grid_size: (grid_x, grid_y, grid_z),
            shared_memory_size: block_x * block_y * 4, // Shared input tile
        }
    }
}

/// Kernel launch parameters
#[derive(Debug, Clone)]
pub struct LaunchParams {
    pub config: KernelConfig,
    pub stream: Option<u64>, // CUDA stream handle
}

impl LaunchParams {
    pub fn new(config: KernelConfig) -> Self {
        Self {
            config,
            stream: None,
        }
    }
    
    pub fn with_stream(mut self, stream: u64) -> Self {
        self.stream = Some(stream);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kernel_availability() {
        let matrix_kernel = OptimizedKernels::matrix_multiply_kernel(4, 4, 4);
        assert!(matrix_kernel.contains("optimized_matrix_multiply"));
        assert!(matrix_kernel.contains("__shared__"));
    }
    
    #[test]
    fn test_vector_kernels() {
        let vector_kernel = OptimizedKernels::vector_operations_kernel();
        assert!(vector_kernel.contains("optimized_vector_add"));
        assert!(vector_kernel.contains("grid-stride loop"));
    }
    
    #[test]
    fn test_activation_kernels() {
        let activation_kernel = OptimizedKernels::activation_functions_kernel();
        assert!(activation_kernel.contains("fast_sigmoid"));
        assert!(activation_kernel.contains("fast_gelu"));
    }
    
    #[test]
    fn test_kernel_config() {
        let config = KernelConfig::for_matrix_multiply(128, 128);
        assert_eq!(config.block_size, (32, 32, 1));
        assert_eq!(config.grid_size, (4, 4, 1));
    }
    
    #[test]
    fn test_activation_kernel_generation() {
        let kernel = OptimizedKernels::get_activation_kernel(ActivationFunction::ReLU).unwrap();
        assert!(kernel.contains("fmaxf"));
        assert!(kernel.contains("specialized_activation"));
    }
}