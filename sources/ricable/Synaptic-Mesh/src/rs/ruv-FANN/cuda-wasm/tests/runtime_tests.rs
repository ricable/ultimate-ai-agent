//! Runtime system comprehensive tests

use cuda_rust_wasm::{
    runtime::{WasmRuntime, RuntimeOptions, DeviceInfo},
    backend::{BackendType, WasmBackend, WebGPUBackend},
    error::CudaError,
};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn test_runtime_initialization() {
        let runtime = WasmRuntime::new(RuntimeOptions::default());
        assert!(runtime.is_ok());
        
        let runtime = runtime.unwrap();
        assert!(runtime.is_initialized());
    }

    #[test]
    fn test_runtime_multiple_backends() {
        // Test WASM backend
        let wasm_options = RuntimeOptions {
            backend_type: BackendType::WASM,
            ..Default::default()
        };
        let wasm_runtime = WasmRuntime::new(wasm_options);
        assert!(wasm_runtime.is_ok());

        // Test WebGPU backend (if available)
        let webgpu_options = RuntimeOptions {
            backend_type: BackendType::WebGPU,
            enable_validation: true,
            ..Default::default()
        };
        let webgpu_runtime = WasmRuntime::new(webgpu_options);
        // May fail if WebGPU not available, which is OK
        match webgpu_runtime {
            Ok(_) => println!("WebGPU backend available"),
            Err(e) => println!("WebGPU backend not available: {}", e),
        }
    }

    #[test]
    fn test_device_enumeration() {
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let devices = runtime.enumerate_devices().unwrap();
        assert!(!devices.is_empty(), "Should have at least one device");
        
        for device in devices {
            assert!(!device.name.is_empty());
            assert!(device.compute_units > 0);
            assert!(device.memory_size > 0);
        }
    }

    #[test]
    fn test_runtime_configuration() {
        let options = RuntimeOptions {
            enable_debug: true,
            enable_validation: true,
            memory_limit: Some(100 * 1024 * 1024), // 100MB
            timeout: Some(Duration::from_secs(30)),
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options);
        assert!(runtime.is_ok());
        
        let runtime = runtime.unwrap();
        let config = runtime.get_configuration();
        assert!(config.debug_enabled);
        assert!(config.validation_enabled);
        assert_eq!(config.memory_limit, Some(100 * 1024 * 1024));
    }

    #[test]
    fn test_concurrent_runtime_access() {
        let runtime = Arc::new(WasmRuntime::new(RuntimeOptions::default()).unwrap());
        let num_threads = 4;
        let barrier = Arc::new(Barrier::new(num_threads));
        
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let runtime = Arc::clone(&runtime);
                let barrier = Arc::clone(&barrier);
                
                thread::spawn(move || {
                    barrier.wait();
                    
                    // Perform operations concurrently
                    for _ in 0..10 {
                        let devices = runtime.enumerate_devices().unwrap();
                        assert!(!devices.is_empty());
                        
                        let status = runtime.get_status();
                        assert!(status.is_active);
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_runtime_resource_cleanup() {
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let initial_memory = runtime.get_memory_usage().unwrap();
        
        // Create and destroy multiple modules
        for _ in 0..10 {
            let dummy_wasm = create_dummy_wasm_module();
            let module = runtime.load_module(&dummy_wasm).unwrap();
            drop(module);
        }
        
        // Force garbage collection
        runtime.force_gc().unwrap();
        
        let final_memory = runtime.get_memory_usage().unwrap();
        assert!(final_memory.used <= initial_memory.used + 1024 * 1024); // Allow 1MB growth
    }

    #[test]
    fn test_error_handling_and_recovery() {
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        
        // Test invalid module loading
        let invalid_wasm = vec![0x00, 0x61, 0x73, 0x6d]; // Incomplete WASM header
        let result = runtime.load_module(&invalid_wasm);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CudaError::InvalidModule(_)));
        
        // Runtime should still be functional after error
        let devices = runtime.enumerate_devices();
        assert!(devices.is_ok());
    }

    #[test]
    fn test_performance_monitoring() {
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        runtime.enable_profiling(true).unwrap();
        
        let start = Instant::now();
        
        // Perform some operations
        for _ in 0..100 {
            let _ = runtime.enumerate_devices();
        }
        
        let duration = start.elapsed();
        let metrics = runtime.get_performance_metrics().unwrap();
        
        assert!(metrics.total_operations > 0);
        assert!(metrics.average_operation_time.as_nanos() > 0);
        assert!(duration >= metrics.total_time);
    }

    #[test]
    fn test_memory_pressure_handling() {
        let options = RuntimeOptions {
            memory_limit: Some(10 * 1024 * 1024), // 10MB limit
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options).unwrap();
        
        // Try to allocate more memory than limit
        let large_data = vec![0u8; 20 * 1024 * 1024]; // 20MB
        let result = runtime.allocate_host_memory(&large_data);
        
        match result {
            Err(CudaError::OutOfMemory) => {}, // Expected
            Ok(_) => panic!("Should have failed with out of memory"),
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_timeout_handling() {
        let options = RuntimeOptions {
            timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        };
        
        let runtime = WasmRuntime::new(options).unwrap();
        
        // Simulate long-running operation
        let result = runtime.execute_with_timeout(|| {
            thread::sleep(Duration::from_millis(200));
            Ok(())
        });
        
        assert!(matches!(result, Err(CudaError::Timeout)));
    }

    // Helper function to create a minimal valid WASM module
    fn create_dummy_wasm_module() -> Vec<u8> {
        vec![
            0x00, 0x61, 0x73, 0x6d, // WASM magic
            0x01, 0x00, 0x00, 0x00, // WASM version
            // Empty module
        ]
    }
}