//! Cross-platform compatibility tests

use cuda_rust_wasm::{
    transpiler::{CudaTranspiler, TranspilerOptions},
    runtime::{WasmRuntime, RuntimeOptions},
    kernel::{KernelLauncher, LaunchConfig},
    memory::{DeviceMemory, MemoryPool, AllocationStrategy},
};
use std::env;
use std::process::Command;

#[cfg(test)]
mod cross_platform_tests {
    use super::*;

    #[test]
    fn test_basic_functionality_all_platforms() {
        let cuda_code = r#"
            __global__ void simple_add(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        "#;
        
        // Test compilation and execution on current platform
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let wasm_bytes = transpiler.transpile(cuda_code).unwrap();
        
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let module = runtime.load_module(&wasm_bytes).unwrap();
        
        // Test data
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
            "simple_add",
            config,
            &[d_a.as_arg(), d_b.as_arg(), d_c.as_arg(), n.as_arg()],
        ).unwrap();
        
        let mut c = vec![0.0f32; n];
        d_c.copy_to_host(&mut c).unwrap();
        
        // Verify results
        for i in 0..n {
            assert_eq!(c[i], a[i] + b[i]);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_linux_specific_features() {
        // Test Linux-specific optimizations
        let options = RuntimeOptions {
            use_system_allocator: true,
            enable_numa_awareness: true,
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options);
        assert!(runtime.is_ok());
        
        // Test large page support if available
        if runtime.as_ref().unwrap().has_large_page_support() {
            let pool = MemoryPool::new_with_large_pages(
                AllocationStrategy::BestFit, 
                100 * 1024 * 1024
            );
            assert!(pool.is_ok());
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_specific_features() {
        // Test macOS Metal backend integration
        let options = RuntimeOptions {
            prefer_metal_backend: true,
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options);
        // May fail if Metal not available, which is OK for older macOS
        match runtime {
            Ok(rt) => {
                assert!(rt.get_backend_info().name.contains("Metal") || 
                       rt.get_backend_info().name.contains("WASM"));
            },
            Err(_) => println!("Metal backend not available, using fallback"),
        }
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_windows_specific_features() {
        // Test Windows DirectX backend integration
        let options = RuntimeOptions {
            prefer_dx12_backend: true,
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options);
        // May fail if DirectX 12 not available
        match runtime {
            Ok(rt) => {
                let backend = rt.get_backend_info();
                assert!(backend.name.contains("DirectX") || 
                       backend.name.contains("WASM"));
            },
            Err(_) => println!("DirectX 12 backend not available, using fallback"),
        }
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_wasm_target_specific() {
        // Test WASM-specific features
        let options = RuntimeOptions {
            enable_wasm_simd: true,
            enable_wasm_threads: true,
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options).unwrap();
        
        // Test SIMD availability
        let simd_support = runtime.has_simd_support();
        println!("WASM SIMD support: {}", simd_support);
        
        // Test SharedArrayBuffer support
        let sab_support = runtime.has_shared_array_buffer_support();
        println!("SharedArrayBuffer support: {}", sab_support);
    }

    #[test]
    fn test_endianness_handling() {
        // Test data consistency across different endianness
        let test_data: Vec<u32> = vec![0x12345678, 0xABCDEF00, 0xDEADBEEF];
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 1024 * 1024).unwrap();
        let d_data = pool.allocate_and_copy(&test_data).unwrap();
        
        let mut readback = vec![0u32; test_data.len()];
        d_data.copy_to_host(&mut readback).unwrap();
        
        assert_eq!(test_data, readback);
    }

    #[test]
    fn test_float_precision_consistency() {
        // Test floating point precision across platforms
        let precision_test_code = r#"
            __global__ void precision_test(float* input, float* output, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    float x = input[idx];
                    // Operations that may have precision differences
                    x = __sinf(x);
                    x = __expf(x);
                    x = __logf(__fabsf(x) + 1e-8f);
                    output[idx] = x;
                }
            }
        "#;
        
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let wasm_bytes = transpiler.transpile(precision_test_code).unwrap();
        
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let module = runtime.load_module(&wasm_bytes).unwrap();
        
        let n = 100;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        let d_input = pool.allocate_and_copy(&input).unwrap();
        let d_output: DeviceMemory<f32> = pool.allocate(n).unwrap();
        
        let launcher = KernelLauncher::new(module);
        let config = LaunchConfig {
            grid_size: ((n + 255) / 256, 1, 1),
            block_size: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        launcher.launch(
            "precision_test",
            config,
            &[d_input.as_arg(), d_output.as_arg(), n.as_arg()],
        ).unwrap();
        
        let mut output = vec![0.0f32; n];
        d_output.copy_to_host(&mut output).unwrap();
        
        // Check that results are finite and reasonable
        for &val in &output {
            assert!(val.is_finite(), "Result should be finite");
        }
    }

    #[test]
    fn test_memory_alignment_requirements() {
        // Test different alignment requirements across platforms
        let alignments = vec![1, 4, 8, 16, 32, 64, 128, 256];
        
        for alignment in alignments {
            let options = RuntimeOptions {
                memory_alignment: alignment,
                ..Default::default()
            };
            
            let runtime = WasmRuntime::new(options);
            if runtime.is_ok() {
                let pool = MemoryPool::new(AllocationStrategy::BestFit, 1024 * 1024).unwrap();
                let mem: Result<DeviceMemory<f32>, _> = pool.allocate(1000);
                
                if let Ok(mem) = mem {
                    let ptr = mem.as_ptr() as usize;
                    assert_eq!(ptr % alignment, 0, "Memory not aligned to {} bytes", alignment);
                }
            }
        }
    }

    #[test]
    fn test_thread_safety_across_platforms() {
        use std::sync::{Arc, Barrier};
        use std::thread;
        
        let runtime = Arc::new(WasmRuntime::new(RuntimeOptions::default()).unwrap());
        let num_threads = std::thread::available_parallelism().unwrap().get().min(8);
        let barrier = Arc::new(Barrier::new(num_threads));
        
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let runtime = Arc::clone(&runtime);
                let barrier = Arc::clone(&barrier);
                
                thread::spawn(move || {
                    barrier.wait();
                    
                    // Perform thread-safe operations
                    for _ in 0..10 {
                        let pool = MemoryPool::new(AllocationStrategy::BestFit, 1024 * 1024).unwrap();
                        let mem: DeviceMemory<f32> = pool.allocate(100).unwrap();
                        drop(mem);
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_compilation_feature_detection() {
        // Test runtime feature detection
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let features = runtime.get_supported_features();
        
        println!("Supported features:");
        println!("  SIMD: {}", features.simd);
        println!("  Threads: {}", features.threads);
        println!("  Atomics: {}", features.atomics);
        println!("  Bulk Memory: {}", features.bulk_memory);
        println!("  Multi-value: {}", features.multi_value);
        println!("  Reference Types: {}", features.reference_types);
        
        // Ensure at least basic features are available
        assert!(features.basic_compute, "Basic compute should always be available");
    }

    #[test]
    fn test_error_message_consistency() {
        // Test that error messages are consistent across platforms
        let invalid_cuda = "__global__ void invalid_syntax( { invalid }";
        
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let result = transpiler.transpile(invalid_cuda);
        
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        
        // Error should contain useful information regardless of platform
        assert!(error_msg.contains("syntax") || 
                error_msg.contains("parse") || 
                error_msg.contains("invalid"));
    }

    #[ignore] // Only run manually to test system integration
    #[test]
    fn test_system_resource_integration() {
        // Test integration with system resources
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        
        // Check memory pressure handling
        let system_memory = runtime.get_system_memory_info().unwrap();
        println!("System memory: {} MB total, {} MB available", 
                 system_memory.total / (1024 * 1024),
                 system_memory.available / (1024 * 1024));
        
        // Don't allocate more than 10% of available memory
        let safe_allocation = system_memory.available / 10;
        if safe_allocation > 100 * 1024 * 1024 { // If more than 100MB available
            let pool = MemoryPool::new(AllocationStrategy::BestFit, safe_allocation);
            assert!(pool.is_ok());
        }
    }
}