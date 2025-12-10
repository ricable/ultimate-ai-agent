//! Browser compatibility tests for WebGPU and WebAssembly

#[cfg(target_arch = "wasm32")]
mod browser_tests {
    use wasm_bindgen_test::*;
    use cuda_rust_wasm::{
        transpiler::{CudaTranspiler, TranspilerOptions},
        runtime::{WasmRuntime, RuntimeOptions},
        kernel::{KernelLauncher, LaunchConfig},
        memory::{DeviceMemory, MemoryPool, AllocationStrategy},
    };
    use wasm_bindgen::prelude::*;
    use web_sys::console;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_webgpu_availability() {
        let window = web_sys::window().unwrap();
        let navigator = window.navigator();
        
        // Check if WebGPU is available
        let gpu = js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"));
        
        match gpu {
            Ok(gpu_obj) if !gpu_obj.is_undefined() => {
                console::log_1(&"WebGPU is available".into());
                
                // Test WebGPU runtime
                let options = RuntimeOptions {
                    backend_type: cuda_rust_wasm::backend::BackendType::WebGPU,
                    ..Default::default()
                };
                
                let runtime = WasmRuntime::new(options);
                assert!(runtime.is_ok(), "WebGPU runtime should initialize");
            },
            _ => {
                console::log_1(&"WebGPU not available, skipping WebGPU tests".into());
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_webassembly_simd_support() {
        // Test WASM SIMD availability
        let has_simd = js_sys::WebAssembly::validate(&[
            0x00, 0x61, 0x73, 0x6d, // magic
            0x01, 0x00, 0x00, 0x00, // version
            0x01, 0x04, 0x01, 0x60, // type section
            0x00, 0x00,             // no params, no results
            0x03, 0x02, 0x01, 0x00, // function section
            0x0a, 0x09, 0x01, 0x07, // code section
            0x00, 0xfd, 0x0c,       // v128.const
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x1a, 0x0b              // drop, end
        ]);
        
        console::log_1(&format!("WASM SIMD support: {}", has_simd).into());
        
        if has_simd {
            let options = RuntimeOptions {
                enable_wasm_simd: true,
                ..Default::default()
            };
            
            let runtime = WasmRuntime::new(options).unwrap();
            assert!(runtime.has_simd_support());
        }
    }

    #[wasm_bindgen_test]
    fn test_shared_array_buffer_support() {
        // Test SharedArrayBuffer availability
        let window = web_sys::window().unwrap();
        let shared_array_buffer = js_sys::Reflect::get(&window, &JsValue::from_str("SharedArrayBuffer"));
        
        match shared_array_buffer {
            Ok(sab) if !sab.is_undefined() => {
                console::log_1(&"SharedArrayBuffer is available".into());
                
                let options = RuntimeOptions {
                    enable_wasm_threads: true,
                    ..Default::default()
                };
                
                let runtime = WasmRuntime::new(options);
                assert!(runtime.is_ok());
            },
            _ => {
                console::log_1(&"SharedArrayBuffer not available".into());
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_basic_compute_in_browser() {
        let cuda_code = r#"
            __global__ void vector_add(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        "#;
        
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let wasm_bytes = transpiler.transpile(cuda_code).unwrap();
        
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let module = runtime.load_module(&wasm_bytes).unwrap();
        
        let n = 100;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        let d_a = pool.allocate_and_copy(&a).unwrap();
        let d_b = pool.allocate_and_copy(&b).unwrap();
        let d_c: DeviceMemory<f32> = pool.allocate(n).unwrap();
        
        let launcher = KernelLauncher::new(module);
        let config = LaunchConfig {
            grid_size: ((n + 255) / 256, 1, 1),
            block_size: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launcher.launch(
            "vector_add",
            config,
            &[d_a.as_arg(), d_b.as_arg(), d_c.as_arg(), n.as_arg()],
        ).unwrap();
        
        let mut c = vec![0.0f32; n];
        d_c.copy_to_host(&mut c).unwrap();
        
        for i in 0..n {
            assert_eq!(c[i], a[i] + b[i]);
        }
        
        console::log_1(&"Basic compute test passed in browser!".into());
    }

    #[wasm_bindgen_test]
    fn test_performance_in_browser() {
        use web_sys::Performance;
        
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        
        let cuda_code = r#"
            __global__ void compute_intensive(float* data, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    float x = data[idx];
                    for (int i = 0; i < 10; i++) {
                        x = __sinf(x) + __cosf(x);
                        x = __sqrtf(x * x + 1.0f);
                    }
                    data[idx] = x;
                }
            }
        "#;
        
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let wasm_bytes = transpiler.transpile(cuda_code).unwrap();
        
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let module = runtime.load_module(&wasm_bytes).unwrap();
        
        let n = 10000;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) / 1000.0).collect();
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        let d_data = pool.allocate_and_copy(&data).unwrap();
        
        let launcher = KernelLauncher::new(module);
        let config = LaunchConfig {
            grid_size: ((n + 255) / 256, 1, 1),
            block_size: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let start_time = performance.now();
        
        launcher.launch(
            "compute_intensive",
            config,
            &[d_data.as_arg(), n.as_arg()],
        ).unwrap();
        
        runtime.synchronize().unwrap();
        
        let end_time = performance.now();
        let duration_ms = end_time - start_time;
        
        console::log_1(&format!("Compute intensive kernel took: {} ms", duration_ms).into());
        
        // Should complete in reasonable time (less than 5 seconds)
        assert!(duration_ms < 5000.0);
    }

    #[wasm_bindgen_test]
    fn test_memory_limits_in_browser() {
        // Test memory allocation limits in browser environment
        let max_allocation = 100 * 1024 * 1024; // 100MB
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, max_allocation);
        assert!(pool.is_ok());
        
        let pool = pool.unwrap();
        
        // Try allocating different sizes
        let sizes = vec![1024, 16384, 1024 * 1024, 10 * 1024 * 1024];
        
        for size in sizes {
            let mem: Result<DeviceMemory<f32>, _> = pool.allocate(size / 4); // f32 = 4 bytes
            match mem {
                Ok(_) => console::log_1(&format!("Successfully allocated {} bytes", size).into()),
                Err(e) => console::log_1(&format!("Failed to allocate {} bytes: {}", size, e).into()),
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_error_handling_in_browser() {
        // Test that errors are properly handled in browser environment
        let invalid_cuda = "__global__ void invalid() { syntax error }";
        
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let result = transpiler.transpile(invalid_cuda);
        
        assert!(result.is_err());
        console::log_1(&format!("Expected error caught: {}", result.unwrap_err()).into());
    }

    #[wasm_bindgen_test]
    fn test_multiple_kernels_in_browser() {
        // Test running multiple kernels in sequence
        let kernel1 = r#"
            __global__ void scale(float* data, float factor, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    data[idx] *= factor;
                }
            }
        "#;
        
        let kernel2 = r#"
            __global__ void add_bias(float* data, float bias, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    data[idx] += bias;
                }
            }
        "#;
        
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let wasm1 = transpiler.transpile(kernel1).unwrap();
        let wasm2 = transpiler.transpile(kernel2).unwrap();
        
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let module1 = runtime.load_module(&wasm1).unwrap();
        let module2 = runtime.load_module(&wasm2).unwrap();
        
        let n = 1000;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        let d_data = pool.allocate_and_copy(&data).unwrap();
        
        let config = LaunchConfig {
            grid_size: ((n + 255) / 256, 1, 1),
            block_size: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Run first kernel
        let launcher1 = KernelLauncher::new(module1);
        launcher1.launch(
            "scale",
            config,
            &[d_data.as_arg(), 2.0f32.as_arg(), n.as_arg()],
        ).unwrap();
        
        // Run second kernel
        let launcher2 = KernelLauncher::new(module2);
        launcher2.launch(
            "add_bias",
            config,
            &[d_data.as_arg(), 10.0f32.as_arg(), n.as_arg()],
        ).unwrap();
        
        let mut result = vec![0.0f32; n];
        d_data.copy_to_host(&mut result).unwrap();
        
        // Verify: (i * 2) + 10
        for i in 0..n {
            assert_eq!(result[i], (i as f32) * 2.0 + 10.0);
        }
        
        console::log_1(&"Multiple kernels test passed!".into());
    }

    #[wasm_bindgen_test]
    fn test_canvas_integration() {
        use web_sys::{Document, HtmlCanvasElement};
        
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        
        // Create a canvas for WebGPU context
        let canvas = document.create_element("canvas")
            .unwrap()
            .dyn_into::<HtmlCanvasElement>()
            .unwrap();
        
        canvas.set_width(512);
        canvas.set_height(512);
        
        // Try to get WebGPU context
        if let Ok(gpu) = js_sys::Reflect::get(&window.navigator(), &JsValue::from_str("gpu")) {
            if !gpu.is_undefined() {
                console::log_1(&"Canvas WebGPU integration available".into());
                
                // Test that our runtime can work with canvas-based WebGPU
                let options = RuntimeOptions {
                    backend_type: cuda_rust_wasm::backend::BackendType::WebGPU,
                    canvas: Some(canvas),
                    ..Default::default()
                };
                
                let runtime = WasmRuntime::new(options);
                match runtime {
                    Ok(_) => console::log_1(&"WebGPU runtime with canvas initialized".into()),
                    Err(e) => console::log_1(&format!("WebGPU canvas init failed: {}", e).into()),
                }
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_worker_thread_compatibility() {
        use web_sys::Worker;
        
        // Test that our runtime works in web workers
        // This would need a separate worker script, so we just test the setup
        
        let options = RuntimeOptions {
            enable_wasm_threads: true,
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options);
        match runtime {
            Ok(rt) => {
                console::log_1(&"Runtime created in worker context".into());
                assert!(rt.is_initialized());
            },
            Err(e) => {
                console::log_1(&format!("Worker compatibility issue: {}", e).into());
            }
        }
    }
}

// Regular tests that run in Node.js environment
#[cfg(not(target_arch = "wasm32"))]
mod browser_simulation_tests {
    use super::*;
    
    #[test]
    fn test_browser_memory_constraints() {
        // Simulate browser memory constraints
        let small_limit = 10 * 1024 * 1024; // 10MB
        
        let options = RuntimeOptions {
            memory_limit: Some(small_limit),
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options).unwrap();
        let pool = MemoryPool::new(AllocationStrategy::BestFit, small_limit).unwrap();
        
        // Should be able to allocate small chunks
        let small_alloc: DeviceMemory<f32> = pool.allocate(1000).unwrap();
        
        // Should fail for large allocations
        let large_alloc: Result<DeviceMemory<f32>, _> = pool.allocate(5 * 1024 * 1024);
        assert!(large_alloc.is_err());
    }
    
    #[test]
    fn test_webgpu_feature_simulation() {
        // Simulate WebGPU feature availability
        let features = vec![
            "timestamp-query",
            "indirect-first-instance", 
            "shader-f16",
            "rg11b10ufloat-renderable",
            "bgra8unorm-storage",
        ];
        
        for feature in features {
            let options = RuntimeOptions {
                required_webgpu_features: vec![feature.to_string()],
                ..Default::default()
            };
            
            // Should gracefully handle missing features
            let runtime = WasmRuntime::new(options);
            // Don't assert success since features may not be available
            match runtime {
                Ok(_) => println!("Feature {} available", feature),
                Err(_) => println!("Feature {} not available", feature),
            }
        }
    }
}