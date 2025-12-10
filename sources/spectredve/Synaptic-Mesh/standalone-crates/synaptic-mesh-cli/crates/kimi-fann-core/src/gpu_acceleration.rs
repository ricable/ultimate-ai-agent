//! GPU Acceleration for Neural Networks
//! 
//! WebGPU-based acceleration for parallel neural network operations,
//! targeting 10-100x speedup for large models with compute shaders.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;
use js_sys::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// GPU acceleration manager for neural networks
#[derive(Debug)]
pub struct GPUAccelerator {
    device: Option<GpuDevice>,
    queue: Option<GpuQueue>,
    compute_pipelines: HashMap<String, GpuComputePipeline>,
    buffers: HashMap<String, GpuBuffer>,
    is_available: bool,
    performance_metrics: GPUPerformanceMetrics,
}

impl GPUAccelerator {
    /// Create new GPU accelerator
    pub async fn new() -> Self {
        let mut accelerator = Self {
            device: None,
            queue: None,
            compute_pipelines: HashMap::new(),
            buffers: HashMap::new(),
            is_available: false,
            performance_metrics: GPUPerformanceMetrics::new(),
        };
        
        if let Err(_) = accelerator.initialize_webgpu().await {
            web_sys::console::warn_1(&"WebGPU not available, falling back to CPU".into());
        }
        
        accelerator
    }
    
    /// Initialize WebGPU context
    async fn initialize_webgpu(&mut self) -> Result<(), JsValue> {
        // Check if WebGPU is available
        let window = web_sys::window().ok_or("No window object")?;
        let navigator = window.navigator();
        
        // Get GPU adapter
        let gpu = js_sys::Reflect::get(&navigator, &"gpu".into())?;
        if gpu.is_undefined() {
            return Err("WebGPU not supported".into());
        }
        
        let gpu: GpuNavigator = gpu.into();
        let adapter_options = GpuRequestAdapterOptions::new();
        let adapter_promise = gpu.request_adapter_with_options(&adapter_options);
        let adapter: GpuAdapter = JsFuture::from(adapter_promise).await?.into();
        
        // Request device
        let device_descriptor = GpuDeviceDescriptor::new();
        let device_promise = adapter.request_device_with_descriptor(&device_descriptor);
        let device: GpuDevice = JsFuture::from(device_promise).await?.into();
        
        let queue = device.queue();
        
        // Create compute shaders
        self.create_neural_compute_shaders(&device).await?;
        
        self.device = Some(device);
        self.queue = Some(queue);
        self.is_available = true;
        
        web_sys::console::log_1(&"WebGPU initialized successfully".into());
        Ok(())
    }
    
    /// Create compute shaders for neural operations
    async fn create_neural_compute_shaders(&mut self, device: &GpuDevice) -> Result<(), JsValue> {
        // Matrix multiplication shader
        let matmul_shader = self.create_matmul_shader(device)?;
        self.compute_pipelines.insert("matmul".to_string(), matmul_shader);
        
        // Activation function shaders
        let relu_shader = self.create_relu_shader(device)?;
        self.compute_pipelines.insert("relu".to_string(), relu_shader);
        
        let sigmoid_shader = self.create_sigmoid_shader(device)?;
        self.compute_pipelines.insert("sigmoid".to_string(), sigmoid_shader);
        
        // Feature extraction shader
        let feature_shader = self.create_feature_extraction_shader(device)?;
        self.compute_pipelines.insert("feature_extract".to_string(), feature_shader);
        
        Ok(())
    }
    
    /// Create matrix multiplication compute shader
    fn create_matmul_shader(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
            @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
            @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<uniform> dimensions: vec3<u32>; // M, N, K
            
            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let row = global_id.x;
                let col = global_id.y;
                let M = dimensions.x;
                let N = dimensions.y;
                let K = dimensions.z;
                
                if (row >= M || col >= N) {
                    return;
                }
                
                var sum: f32 = 0.0;
                for (var k: u32 = 0u; k < K; k++) {
                    sum += matrixA[row * K + k] * matrixB[k * N + col];
                }
                
                result[row * N + col] = sum;
            }
        "#;
        
        let shader_module = self.create_shader_module(device, shader_code)?;
        let pipeline_descriptor = GpuComputePipelineDescriptor::new("main", &shader_module);
        Ok(device.create_compute_pipeline(&pipeline_descriptor))
    }
    
    /// Create ReLU activation shader
    fn create_relu_shader(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&input)) {
                    return;
                }
                
                output[index] = max(0.0, input[index]);
            }
        "#;
        
        let shader_module = self.create_shader_module(device, shader_code)?;
        let pipeline_descriptor = GpuComputePipelineDescriptor::new("main", &shader_module);
        Ok(device.create_compute_pipeline(&pipeline_descriptor))
    }
    
    /// Create sigmoid activation shader
    fn create_sigmoid_shader(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&input)) {
                    return;
                }
                
                let x = input[index];
                // Fast sigmoid approximation: 0.5 + 0.25 * x * (1 - abs(x)/4) for |x| < 2
                let abs_x = abs(x);
                if (abs_x < 2.0) {
                    output[index] = 0.5 + 0.25 * x * (1.0 - abs_x / 4.0);
                } else {
                    output[index] = 1.0 / (1.0 + exp(-x));
                }
            }
        "#;
        
        let shader_module = self.create_shader_module(device, shader_code)?;
        let pipeline_descriptor = GpuComputePipelineDescriptor::new("main", &shader_module);
        Ok(device.create_compute_pipeline(&pipeline_descriptor))
    }
    
    /// Create feature extraction shader
    fn create_feature_extraction_shader(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
            @group(0) @binding(0) var<storage, read> text_features: array<f32>;
            @group(0) @binding(1) var<storage, read_write> neural_features: array<f32>;
            @group(0) @binding(2) var<uniform> feature_config: vec4<f32>; // input_size, output_size, scale, bias
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                let output_size = u32(feature_config.y);
                
                if (index >= output_size) {
                    return;
                }
                
                let input_size = u32(feature_config.x);
                let scale = feature_config.z;
                let bias = feature_config.w;
                
                var sum: f32 = 0.0;
                
                // Feature transformation with normalization
                if (index < input_size) {
                    sum = text_features[index] * scale + bias;
                } else {
                    // Generate synthetic features for remaining indices
                    let hash_val = f32(index * 37u + 17u) / 1000.0;
                    sum = sin(hash_val) * 0.5;
                }
                
                // Apply tanh activation for bounded output
                neural_features[index] = tanh(sum);
            }
        "#;
        
        let shader_module = self.create_shader_module(device, shader_code)?;
        let pipeline_descriptor = GpuComputePipelineDescriptor::new("main", &shader_module);
        Ok(device.create_compute_pipeline(&pipeline_descriptor))
    }
    
    /// Create shader module from WGSL code
    fn create_shader_module(&self, device: &GpuDevice, code: &str) -> Result<GpuShaderModule, JsValue> {
        let mut descriptor = GpuShaderModuleDescriptor::new(code);
        descriptor.label("Neural compute shader");
        Ok(device.create_shader_module(&descriptor))
    }
    
    /// Perform GPU-accelerated matrix multiplication
    pub async fn matrix_multiply_gpu(
        &mut self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        rows_a: u32,
        cols_a: u32,
        cols_b: u32,
    ) -> Result<Vec<f32>, JsValue> {
        if !self.is_available {
            return Err("GPU not available".into());
        }
        
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let pipeline = self.compute_pipelines.get("matmul").unwrap();
        
        // Create buffers
        let buffer_a = self.create_storage_buffer(device, matrix_a, "Matrix A")?;
        let buffer_b = self.create_storage_buffer(device, matrix_b, "Matrix B")?;
        
        let result_size = (rows_a * cols_b) as usize;
        let result_buffer = self.create_output_buffer(device, result_size * 4, "Result")?;
        
        // Create uniform buffer for dimensions
        let dimensions = [rows_a, cols_b, cols_a, 0u32]; // Pad to 4 elements
        let uniform_buffer = self.create_uniform_buffer(device, &dimensions, "Dimensions")?;
        
        // Create bind group
        let bind_group = self.create_matrix_multiply_bind_group(
            device, pipeline, &buffer_a, &buffer_b, &result_buffer, &uniform_buffer
        )?;
        
        // Execute compute shader
        let command_encoder = device.create_command_encoder();
        let compute_pass = command_encoder.begin_compute_pass();
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group);
        
        let workgroup_x = (rows_a + 15) / 16;
        let workgroup_y = (cols_b + 15) / 16;
        compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        compute_pass.end();
        
        queue.submit(&Array::from(&command_encoder.finish()));
        
        // Read result
        self.read_buffer_async(&result_buffer, result_size * 4).await
    }
    
    /// Perform GPU-accelerated activation function
    pub async fn activate_gpu(
        &mut self,
        input: &[f32],
        activation_type: ActivationType,
    ) -> Result<Vec<f32>, JsValue> {
        if !self.is_available {
            return Err("GPU not available".into());
        }
        
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        
        let pipeline_name = match activation_type {
            ActivationType::ReLU => "relu",
            ActivationType::Sigmoid => "sigmoid",
        };
        
        let pipeline = self.compute_pipelines.get(pipeline_name).unwrap();
        
        // Create buffers
        let input_buffer = self.create_storage_buffer(device, input, "Input")?;
        let output_buffer = self.create_output_buffer(device, input.len() * 4, "Output")?;
        
        // Create bind group
        let bind_group = self.create_activation_bind_group(
            device, pipeline, &input_buffer, &output_buffer
        )?;
        
        // Execute compute shader
        let command_encoder = device.create_command_encoder();
        let compute_pass = command_encoder.begin_compute_pass();
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group);
        
        let workgroups = (input.len() as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
        compute_pass.end();
        
        queue.submit(&Array::from(&command_encoder.finish()));
        
        // Read result
        self.read_buffer_async(&output_buffer, input.len() * 4).await
    }
    
    /// Create storage buffer for GPU data
    fn create_storage_buffer(&self, device: &GpuDevice, data: &[f32], label: &str) -> Result<GpuBuffer, JsValue> {
        let mut descriptor = GpuBufferDescriptor::new((data.len() * 4) as f64, GpuBufferUsage::STORAGE | GpuBufferUsage::COPY_DST);
        descriptor.label(label);
        
        let buffer = device.create_buffer(&descriptor);
        
        // Write data to buffer
        let queue = self.queue.as_ref().unwrap();
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        queue.write_buffer(&buffer, 0, data_bytes);
        
        Ok(buffer)
    }
    
    /// Create output buffer for GPU results
    fn create_output_buffer(&self, device: &GpuDevice, size: usize, label: &str) -> Result<GpuBuffer, JsValue> {
        let mut descriptor = GpuBufferDescriptor::new(size as f64, GpuBufferUsage::STORAGE | GpuBufferUsage::COPY_SRC);
        descriptor.label(label);
        Ok(device.create_buffer(&descriptor))
    }
    
    /// Create uniform buffer for shader constants
    fn create_uniform_buffer(&self, device: &GpuDevice, data: &[u32], label: &str) -> Result<GpuBuffer, JsValue> {
        let mut descriptor = GpuBufferDescriptor::new((data.len() * 4) as f64, GpuBufferUsage::UNIFORM | GpuBufferUsage::COPY_DST);
        descriptor.label(label);
        
        let buffer = device.create_buffer(&descriptor);
        
        let queue = self.queue.as_ref().unwrap();
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        queue.write_buffer(&buffer, 0, data_bytes);
        
        Ok(buffer)
    }
    
    /// Create bind group for matrix multiplication
    fn create_matrix_multiply_bind_group(
        &self,
        device: &GpuDevice,
        pipeline: &GpuComputePipeline,
        buffer_a: &GpuBuffer,
        buffer_b: &GpuBuffer,
        result_buffer: &GpuBuffer,
        uniform_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        let layout = pipeline.get_bind_group_layout(0);
        
        let entries = Array::new();
        
        // Matrix A
        let mut entry_a = GpuBindGroupEntry::new(0, buffer_a);
        entries.push(&entry_a);
        
        // Matrix B
        let mut entry_b = GpuBindGroupEntry::new(1, buffer_b);
        entries.push(&entry_b);
        
        // Result
        let mut entry_result = GpuBindGroupEntry::new(2, result_buffer);
        entries.push(&entry_result);
        
        // Dimensions
        let mut entry_uniform = GpuBindGroupEntry::new(3, uniform_buffer);
        entries.push(&entry_uniform);
        
        let mut descriptor = GpuBindGroupDescriptor::new(&entries, &layout);
        descriptor.label("Matrix multiply bind group");
        
        Ok(device.create_bind_group(&descriptor))
    }
    
    /// Create bind group for activation functions
    fn create_activation_bind_group(
        &self,
        device: &GpuDevice,
        pipeline: &GpuComputePipeline,
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        let layout = pipeline.get_bind_group_layout(0);
        
        let entries = Array::new();
        
        // Input
        let mut entry_input = GpuBindGroupEntry::new(0, input_buffer);
        entries.push(&entry_input);
        
        // Output
        let mut entry_output = GpuBindGroupEntry::new(1, output_buffer);
        entries.push(&entry_output);
        
        let mut descriptor = GpuBindGroupDescriptor::new(&entries, &layout);
        descriptor.label("Activation bind group");
        
        Ok(device.create_bind_group(&descriptor))
    }
    
    /// Read buffer data asynchronously
    async fn read_buffer_async(&self, buffer: &GpuBuffer, size: usize) -> Result<Vec<f32>, JsValue> {
        let device = self.device.as_ref().unwrap();
        
        // Create staging buffer
        let mut staging_descriptor = GpuBufferDescriptor::new(size as f64, GpuBufferUsage::COPY_DST | GpuBufferUsage::MAP_READ);
        staging_descriptor.label("Staging buffer");
        let staging_buffer = device.create_buffer(&staging_descriptor);
        
        // Copy data to staging buffer
        let command_encoder = device.create_command_encoder();
        command_encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size as f64);
        
        let queue = self.queue.as_ref().unwrap();
        queue.submit(&Array::from(&command_encoder.finish()));
        
        // Map and read staging buffer
        let map_promise = staging_buffer.map_async(GpuMapMode::READ(), 0, size as f64);
        JsFuture::from(map_promise).await?;
        
        let mapped_range = staging_buffer.get_mapped_range_with_u32_and_u32(0, size as u32);
        let data_view = Uint8Array::new(&mapped_range);
        let mut data_bytes = vec![0u8; size];
        data_view.copy_to(&mut data_bytes);
        
        staging_buffer.unmap();
        
        // Convert bytes to f32
        let float_count = size / 4;
        let mut result = Vec::with_capacity(float_count);
        for i in 0..float_count {
            let bytes = [
                data_bytes[i * 4],
                data_bytes[i * 4 + 1],
                data_bytes[i * 4 + 2],
                data_bytes[i * 4 + 3],
            ];
            result.push(f32::from_le_bytes(bytes));
        }
        
        Ok(result)
    }
    
    /// Get GPU performance metrics
    pub fn get_performance_metrics(&self) -> &GPUPerformanceMetrics {
        &self.performance_metrics
    }
    
    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.is_available
    }
    
    /// Benchmark GPU performance
    pub async fn benchmark_performance(&mut self) -> Result<GPUBenchmarkResults, JsValue> {
        if !self.is_available {
            return Err("GPU not available".into());
        }
        
        let mut results = GPUBenchmarkResults::new();
        
        // Benchmark matrix multiplication
        let matrix_size = 256;
        let matrix_a = vec![1.0f32; matrix_size * matrix_size];
        let matrix_b = vec![1.0f32; matrix_size * matrix_size];
        
        let start_time = js_sys::Date::now();
        let _result = self.matrix_multiply_gpu(&matrix_a, &matrix_b, matrix_size as u32, matrix_size as u32, matrix_size as u32).await?;
        let matmul_time = js_sys::Date::now() - start_time;
        
        results.matrix_multiply_ms = matmul_time;
        results.matrix_multiply_gflops = (2.0 * matrix_size as f64 * matrix_size as f64 * matrix_size as f64) / (matmul_time * 1_000_000.0);
        
        // Benchmark activation functions
        let activation_input = vec![0.5f32; 10000];
        
        let start_time = js_sys::Date::now();
        let _result = self.activate_gpu(&activation_input, ActivationType::ReLU).await?;
        let relu_time = js_sys::Date::now() - start_time;
        
        results.relu_activation_ms = relu_time;
        results.activation_throughput_ops_ms = activation_input.len() as f64 / relu_time;
        
        Ok(results)
    }
}

/// Activation function types for GPU acceleration
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
}

/// GPU performance metrics
#[derive(Debug, Clone)]
pub struct GPUPerformanceMetrics {
    pub total_operations: u64,
    pub total_gpu_time_ms: f64,
    pub average_operation_time_ms: f64,
    pub memory_transfers_mb: f64,
    pub compute_utilization: f64,
}

impl GPUPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            total_gpu_time_ms: 0.0,
            average_operation_time_ms: 0.0,
            memory_transfers_mb: 0.0,
            compute_utilization: 0.0,
        }
    }
}

/// GPU benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUBenchmarkResults {
    pub matrix_multiply_ms: f64,
    pub matrix_multiply_gflops: f64,
    pub relu_activation_ms: f64,
    pub activation_throughput_ops_ms: f64,
    pub memory_bandwidth_gb_s: f64,
    pub overall_score: f64,
}

impl GPUBenchmarkResults {
    pub fn new() -> Self {
        Self {
            matrix_multiply_ms: 0.0,
            matrix_multiply_gflops: 0.0,
            relu_activation_ms: 0.0,
            activation_throughput_ops_ms: 0.0,
            memory_bandwidth_gb_s: 0.0,
            overall_score: 0.0,
        }
    }
    
    /// Calculate overall performance score
    pub fn calculate_score(&mut self) {
        // Weighted score based on different metrics
        let matmul_score = (self.matrix_multiply_gflops / 100.0).min(1.0) * 40.0;
        let activation_score = (self.activation_throughput_ops_ms / 1000.0).min(1.0) * 30.0;
        let bandwidth_score = (self.memory_bandwidth_gb_s / 100.0).min(1.0) * 30.0;
        
        self.overall_score = matmul_score + activation_score + bandwidth_score;
    }
}

/// GPU acceleration fallback for non-WebGPU environments
pub struct GPUFallback;

impl GPUFallback {
    /// CPU-based matrix multiplication fallback
    pub fn matrix_multiply_cpu(
        matrix_a: &[f32],
        matrix_b: &[f32],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0; rows_a * cols_b];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a {
                    sum += matrix_a[i * cols_a + k] * matrix_b[k * cols_b + j];
                }
                result[i * cols_b + j] = sum;
            }
        }
        
        result
    }
    
    /// CPU-based activation function fallback
    pub fn activate_cpu(input: &[f32], activation_type: ActivationType) -> Vec<f32> {
        match activation_type {
            ActivationType::ReLU => input.iter().map(|&x| x.max(0.0)).collect(),
            ActivationType::Sigmoid => input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    #[wasm_bindgen_test]
    fn test_gpu_fallback_matrix_multiply() {
        let matrix_a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let matrix_b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        
        let result = GPUFallback::matrix_multiply_cpu(&matrix_a, &matrix_b, 2, 2, 2);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }
    
    #[wasm_bindgen_test]
    fn test_gpu_fallback_relu() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let result = GPUFallback::activate_cpu(&input, ActivationType::ReLU);
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    }
    
    #[wasm_bindgen_test]
    fn test_gpu_fallback_sigmoid() {
        let input = vec![0.0];
        let result = GPUFallback::activate_cpu(&input, ActivationType::Sigmoid);
        assert!((result[0] - 0.5).abs() < 0.001);
    }
}