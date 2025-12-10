@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec2<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let cols = dimensions.y;
    
    if (row >= dimensions.x) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var col: u32 = 0u; col < cols; col = col + 1u) {
        sum += matrix[row * cols + col] * vector[col];
    }
    
    result[row] = sum;
}