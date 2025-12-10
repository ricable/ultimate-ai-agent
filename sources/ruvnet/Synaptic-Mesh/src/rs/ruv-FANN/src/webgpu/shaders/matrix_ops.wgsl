// Matrix-vector multiplication shader for WebGPU
// Optimized for neural network operations

const WORKGROUP_SIZE: u32 = 64u;

struct MatrixDimensions {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0)
var<storage, read> matrix: array<f32>;

@group(0) @binding(1)
var<storage, read> vector: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> dims: MatrixDimensions;

@compute @workgroup_size(WORKGROUP_SIZE)
fn matrix_vector_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    // Bounds check
    if (row >= dims.rows) {
        return;
    }
    
    var sum = 0.0f;
    let row_start = row * dims.cols;
    
    // Unrolled loop for better performance
    var col = 0u;
    let unroll_factor = 4u;
    
    // Process 4 elements at a time
    while (col + unroll_factor <= dims.cols) {
        sum += matrix[row_start + col] * vector[col];
        sum += matrix[row_start + col + 1u] * vector[col + 1u];
        sum += matrix[row_start + col + 2u] * vector[col + 2u];
        sum += matrix[row_start + col + 3u] * vector[col + 3u];
        col += unroll_factor;
    }
    
    // Handle remainder
    while (col < dims.cols) {
        sum += matrix[row_start + col] * vector[col];
        col += 1u;
    }
    
    output[row] = sum;
}

// Fused matrix-vector multiply with activation function
@compute @workgroup_size(WORKGROUP_SIZE)
fn matrix_vector_multiply_relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    if (row >= dims.rows) {
        return;
    }
    
    var sum = 0.0f;
    let row_start = row * dims.cols;
    
    // Compute matrix-vector product
    for (var col = 0u; col < dims.cols; col += 1u) {
        sum += matrix[row_start + col] * vector[col];
    }
    
    // Apply ReLU activation inline
    output[row] = max(0.0f, sum);
}

// Batch matrix-vector multiplication
struct BatchDimensions {
    rows: u32,
    cols: u32,
    batch_size: u32,
}

@group(0) @binding(3)
var<uniform> batch_dims: BatchDimensions;

@compute @workgroup_size(WORKGROUP_SIZE)
fn batch_matrix_vector_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.y;
    let row = global_id.x;
    
    if (batch_idx >= batch_dims.batch_size || row >= batch_dims.rows) {
        return;
    }
    
    var sum = 0.0f;
    let row_start = row * batch_dims.cols;
    let vector_offset = batch_idx * batch_dims.cols;
    let output_offset = batch_idx * batch_dims.rows;
    
    // Compute matrix-vector product for this batch element
    for (var col = 0u; col < batch_dims.cols; col += 1u) {
        sum += matrix[row_start + col] * vector[vector_offset + col];
    }
    
    output[output_offset + row] = sum;
}