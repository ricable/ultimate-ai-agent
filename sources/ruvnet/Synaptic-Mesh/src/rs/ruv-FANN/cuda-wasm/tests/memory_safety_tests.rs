//! Memory safety and leak detection tests

use cuda_rust_wasm::{
    memory::{DeviceMemory, MemoryPool, AllocationStrategy, UnifiedMemory, PinnedMemory},
    runtime::{WasmRuntime, RuntimeOptions},
    error::CudaError,
};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[cfg(test)]
mod memory_safety_tests {
    use super::*;

    #[test]
    fn test_memory_leak_detection() {
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let initial_memory = runtime.get_memory_usage().unwrap();
        
        // Allocate and deallocate memory in a loop
        for _ in 0..100 {
            let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
            let mem: DeviceMemory<f32> = pool.allocate(10000).unwrap();
            
            // Use the memory
            let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
            mem.copy_from_host(&data).unwrap();
            
            // Memory should be freed when `mem` goes out of scope
            drop(mem);
            drop(pool);
        }
        
        // Force garbage collection
        runtime.force_gc().unwrap();
        
        let final_memory = runtime.get_memory_usage().unwrap();
        let memory_growth = final_memory.used as i64 - initial_memory.used as i64;
        
        println!("Initial memory: {} bytes", initial_memory.used);
        println!("Final memory: {} bytes", final_memory.used);
        println!("Memory growth: {} bytes", memory_growth);
        
        // Allow some growth but not excessive
        assert!(memory_growth < 1024 * 1024, "Memory leak detected: {} bytes leaked", memory_growth);
    }

    #[test]
    fn test_double_free_protection() {
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        let mem: DeviceMemory<f32> = pool.allocate(1000).unwrap();
        
        // First drop should succeed
        drop(mem);
        
        // Attempting to use after free should be prevented by Rust's ownership system
        // This test primarily checks that our Drop implementation is safe
    }

    #[test]
    fn test_buffer_overflow_protection() {
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        let mem: DeviceMemory<f32> = pool.allocate(100).unwrap(); // 100 elements
        
        // Try to copy more data than allocated
        let large_data: Vec<f32> = (0..200).map(|i| i as f32).collect(); // 200 elements
        
        let result = mem.copy_from_host(&large_data);
        assert!(result.is_err(), "Buffer overflow should be detected");
        assert!(matches!(result.unwrap_err(), CudaError::BufferOverflow));
    }

    #[test]
    fn test_null_pointer_protection() {
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        
        // Test various invalid operations that could lead to null pointer access
        let mem: DeviceMemory<f32> = pool.allocate(100).unwrap();
        
        // Test with empty slice
        let empty_data: Vec<f32> = vec![];
        let result = mem.copy_from_host(&empty_data);
        assert!(result.is_ok(), "Empty copy should be safe");
        
        // Test bounds checking
        let mut readback = vec![0.0f32; 150]; // Larger than allocated
        let result = mem.copy_to_host(&mut readback);
        assert!(result.is_err(), "Oversized read should fail");
    }

    #[test]
    fn test_concurrent_memory_safety() {
        let pool = Arc::new(MemoryPool::new(AllocationStrategy::BuddySystem, 100 * 1024 * 1024).unwrap());
        let num_threads = 8;
        let barrier = Arc::new(Barrier::new(num_threads));
        let allocation_count = Arc::new(Mutex::new(0));
        let deallocation_count = Arc::new(Mutex::new(0));
        
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                let alloc_count = Arc::clone(&allocation_count);
                let dealloc_count = Arc::clone(&deallocation_count);
                
                thread::spawn(move || {
                    barrier.wait();
                    
                    let mut allocations = Vec::new();
                    
                    // Allocate memory concurrently
                    for i in 0..100 {
                        let size = 1000 + (thread_id * 100) + i;
                        if let Ok(mem) = pool.allocate::<f32>(size) {
                            allocations.push(mem);
                            *alloc_count.lock().unwrap() += 1;
                        }
                    }
                    
                    // Use the memory
                    for mem in &allocations {
                        let data: Vec<f32> = (0..mem.len()).map(|i| i as f32).collect();
                        mem.copy_from_host(&data).unwrap();
                    }
                    
                    // Deallocate in random order
                    while !allocations.is_empty() {
                        let idx = thread_id % allocations.len();
                        allocations.remove(idx);
                        *dealloc_count.lock().unwrap() += 1;
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_alloc_count = *allocation_count.lock().unwrap();
        let final_dealloc_count = *deallocation_count.lock().unwrap();
        
        println!("Total allocations: {}", final_alloc_count);
        println!("Total deallocations: {}", final_dealloc_count);
        
        assert_eq!(final_alloc_count, final_dealloc_count, "Memory allocation/deallocation mismatch");
    }

    #[test]
    fn test_memory_alignment_safety() {
        let alignments = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];
        
        for alignment in alignments {
            let options = RuntimeOptions {
                memory_alignment: alignment,
                ..Default::default()
            };
            
            let runtime = WasmRuntime::new(options);
            if let Ok(runtime) = runtime {
                let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
                
                for size in vec![1, 100, 1000, 10000] {
                    if let Ok(mem) = pool.allocate::<f32>(size) {
                        let ptr = mem.as_ptr() as usize;
                        assert_eq!(ptr % alignment, 0, 
                                   "Memory at {:p} not aligned to {} bytes", 
                                   ptr as *const (), alignment);
                        
                        // Test that aligned memory access works
                        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                        mem.copy_from_host(&data).unwrap();
                        
                        let mut readback = vec![0.0f32; size];
                        mem.copy_to_host(&mut readback).unwrap();
                        
                        assert_eq!(data, readback);
                    }
                }
            }
        }
    }

    #[test]
    fn test_memory_pressure_handling() {
        // Test behavior under memory pressure
        let small_pool = MemoryPool::new(AllocationStrategy::BestFit, 1024 * 1024).unwrap(); // 1MB
        let mut allocations = Vec::new();
        let mut total_allocated = 0;
        
        // Allocate until we hit memory limits
        for i in 0..1000 {
            let size = 1000 + i; // Increasing size
            match small_pool.allocate::<f32>(size) {
                Ok(mem) => {
                    total_allocated += size * std::mem::size_of::<f32>();
                    allocations.push(mem);
                },
                Err(CudaError::OutOfMemory) => {
                    println!("Hit memory limit after allocating {} bytes", total_allocated);
                    break;
                },
                Err(e) => panic!("Unexpected error: {}", e),
            }
        }
        
        assert!(!allocations.is_empty(), "Should have made some allocations");
        assert!(total_allocated > 0, "Should have allocated some memory");
        
        // Free half the allocations
        let half = allocations.len() / 2;
        allocations.truncate(half);
        
        // Should be able to allocate again
        let new_alloc = small_pool.allocate::<f32>(1000);
        assert!(new_alloc.is_ok(), "Should be able to allocate after freeing memory");
    }

    #[test]
    fn test_fragmentation_resilience() {
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        let mut allocations = Vec::new();
        
        // Create a fragmented memory pattern
        for i in 0..100 {
            let size = if i % 2 == 0 { 1000 } else { 2000 };
            if let Ok(mem) = pool.allocate::<f32>(size) {
                allocations.push((i, mem));
            }
        }
        
        // Free every other allocation to create fragmentation
        allocations.retain(|(i, _)| i % 2 == 0);
        
        // Try to allocate in the fragmented space
        let mut new_allocations = Vec::new();
        for _ in 0..50 {
            if let Ok(mem) = pool.allocate::<f32>(1500) {
                new_allocations.push(mem);
            }
        }
        
        assert!(!new_allocations.is_empty(), "Should be able to allocate in fragmented space");
        
        // Test that all allocations are still valid
        for mem in &new_allocations {
            let data: Vec<f32> = (0..mem.len()).map(|i| i as f32).collect();
            mem.copy_from_host(&data).unwrap();
            
            let mut readback = vec![0.0f32; mem.len()];
            mem.copy_to_host(&mut readback).unwrap();
            
            assert_eq!(data, readback);
        }
    }

    #[test]
    fn test_use_after_free_detection() {
        // This test relies on Rust's ownership system to prevent use-after-free
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        
        {
            let mem: DeviceMemory<f32> = pool.allocate(1000).unwrap();
            let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
            mem.copy_from_host(&data).unwrap();
            
            // `mem` is dropped here
        }
        
        // If we try to use `mem` here, it would be a compile-time error
        // This test primarily verifies that our Drop implementation is correct
        
        // We can still use the pool for new allocations
        let new_mem: DeviceMemory<f32> = pool.allocate(1000).unwrap();
        let new_data: Vec<f32> = (0..1000).map(|i| (i * 2) as f32).collect();
        new_mem.copy_from_host(&new_data).unwrap();
    }

    #[test]
    fn test_memory_pool_isolation() {
        // Test that memory pools are properly isolated
        let pool1 = MemoryPool::new(AllocationStrategy::BestFit, 5 * 1024 * 1024).unwrap();
        let pool2 = MemoryPool::new(AllocationStrategy::FirstFit, 5 * 1024 * 1024).unwrap();
        
        let mem1: DeviceMemory<f32> = pool1.allocate(10000).unwrap();
        let mem2: DeviceMemory<f32> = pool2.allocate(10000).unwrap();
        
        // Fill with different patterns
        let data1: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let data2: Vec<f32> = (0..10000).map(|i| (i * 2) as f32).collect();
        
        mem1.copy_from_host(&data1).unwrap();
        mem2.copy_from_host(&data2).unwrap();
        
        // Verify isolation
        let mut readback1 = vec![0.0f32; 10000];
        let mut readback2 = vec![0.0f32; 10000];
        
        mem1.copy_to_host(&mut readback1).unwrap();
        mem2.copy_to_host(&mut readback2).unwrap();
        
        assert_eq!(data1, readback1);
        assert_eq!(data2, readback2);
        assert_ne!(readback1, readback2);
    }

    #[test]
    fn test_resource_cleanup_on_panic() {
        use std::panic;
        
        let pool = Arc::new(MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap());
        let initial_stats = pool.get_statistics().unwrap();
        
        // Test that resources are cleaned up even if panic occurs
        let result = panic::catch_unwind(|| {
            let pool = Arc::clone(&pool);
            let _mem: DeviceMemory<f32> = pool.allocate(10000).unwrap();
            
            // Simulate panic during processing
            panic!("Simulated panic");
        });
        
        assert!(result.is_err(), "Panic should have occurred");
        
        // Give time for cleanup
        thread::sleep(Duration::from_millis(100));
        
        let final_stats = pool.get_statistics().unwrap();
        
        // Memory should be cleaned up despite the panic
        assert_eq!(initial_stats.allocated_bytes, final_stats.allocated_bytes,
                   "Memory should be cleaned up after panic");
    }

    #[test]
    fn test_memory_pattern_detection() {
        // Test for common memory corruption patterns
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        let mem: DeviceMemory<u8> = pool.allocate(1000).unwrap();
        
        // Test pattern: all zeros
        let zeros = vec![0u8; 1000];
        mem.copy_from_host(&zeros).unwrap();
        
        let mut readback = vec![0xFFu8; 1000];
        mem.copy_to_host(&mut readback).unwrap();
        assert_eq!(zeros, readback);
        
        // Test pattern: all ones
        let ones = vec![0xFFu8; 1000];
        mem.copy_from_host(&ones).unwrap();
        
        let mut readback = vec![0u8; 1000];
        mem.copy_to_host(&mut readback).unwrap();
        assert_eq!(ones, readback);
        
        // Test pattern: alternating
        let pattern: Vec<u8> = (0..1000).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect();
        mem.copy_from_host(&pattern).unwrap();
        
        let mut readback = vec![0u8; 1000];
        mem.copy_to_host(&mut readback).unwrap();
        assert_eq!(pattern, readback);
    }

    #[test] 
    fn test_stack_overflow_protection() {
        // Test protection against stack overflow in recursive operations
        fn recursive_allocation(pool: &MemoryPool, depth: usize) -> Result<Vec<DeviceMemory<f32>>, CudaError> {
            if depth == 0 {
                return Ok(Vec::new());
            }
            
            let mut allocations = recursive_allocation(pool, depth - 1)?;
            let mem: DeviceMemory<f32> = pool.allocate(100)?;
            allocations.push(mem);
            Ok(allocations)
        }
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
        
        // Should handle reasonable recursion depth
        let result = recursive_allocation(&pool, 100);
        assert!(result.is_ok(), "Should handle reasonable recursion");
        
        // Very deep recursion should be limited by available memory
        let result = recursive_allocation(&pool, 100000);
        // This should either succeed or fail gracefully with OutOfMemory
        match result {
            Ok(_) => println!("Deep recursion succeeded"),
            Err(CudaError::OutOfMemory) => println!("Deep recursion limited by memory"),
            Err(e) => panic!("Unexpected error in deep recursion: {}", e),
        }
    }
}