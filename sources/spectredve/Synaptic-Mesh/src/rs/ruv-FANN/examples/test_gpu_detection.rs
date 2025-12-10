// Simple test to check if GPU is detected

fn main() {
    #[cfg(feature = "gpu")]
    {
        use ruv_fann::webgpu::ComputeBackend;
        
        println!("GPU feature is enabled");
        
        // Test WebGPU availability
        match ruv_fann::webgpu::WebGPUBackend::<f32>::is_available() {
            true => {
                println!("WebGPU is available!");
                
                // Try to initialize
                match ruv_fann::webgpu::WebGPUBackend::<f32>::new() {
                    Ok(backend) => {
                        println!("WebGPU backend initialized successfully!");
                        let caps = backend.capabilities();
                        println!("Capabilities:");
                        println!("  Max buffer size: {} MB", caps.max_buffer_size / (1024 * 1024));
                        println!("  Max compute units: {}", caps.max_compute_units);
                        println!("  Supports F16: {}", caps.supports_f16);
                        println!("  Memory bandwidth: {} GB/s", caps.memory_bandwidth_gbps);
                    }
                    Err(e) => {
                        println!("Failed to initialize WebGPU backend: {}", e);
                    }
                }
            }
            false => {
                println!("WebGPU is not available");
            }
        }
        
        // Test training API
        println!("\nTesting training API:");
        println!("GPU available for training: {}", ruv_fann::training::is_gpu_available());
        println!("GPU capabilities: {}", ruv_fann::training::get_gpu_capabilities());
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature is not enabled. Compile with --features gpu");
    }
}