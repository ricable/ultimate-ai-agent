//! WASM runtime backend implementation

use super::backend_trait::{BackendTrait, BackendCapabilities, MemcpyKind};
use crate::{Result, runtime_error};
use std::sync::Arc;
use async_trait::async_trait;

/// CPU-based runtime backend for WASM environments
pub struct WasmRuntime {
    capabilities: BackendCapabilities,
}

impl Default for WasmRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmRuntime {
    /// Create a new WASM runtime backend
    pub fn new() -> Self {
        Self {
            capabilities: BackendCapabilities {
                name: "WASM Runtime".to_string(),
                supports_cuda: false,
                supports_opencl: false,
                supports_vulkan: false,
                supports_webgpu: false,
                max_threads: 1,
                max_threads_per_block: 1,
                max_blocks_per_grid: 1,
                max_shared_memory: 0,
                supports_dynamic_parallelism: false,
                supports_unified_memory: false,
                max_grid_dim: [1, 1, 1],
                max_block_dim: [1, 1, 1],
                warp_size: 1,
            },
        }
    }
}

#[async_trait]
impl BackendTrait for WasmRuntime {
    fn name(&self) -> &str {
        &self.capabilities.name
    }
    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }
    
    async fn initialize(&mut self) -> Result<()> {
        // No initialization needed for WASM runtime
        Ok(())
    }
    
    async fn compile_kernel(&self, _source: &str) -> Result<Vec<u8>> {
        // For WASM runtime, we don't compile kernels
        Err(runtime_error!("Kernel compilation not supported on WASM runtime backend"))
    }
    
    async fn launch_kernel(
        &self,
        _kernel: &[u8],
        _grid: (u32, u32, u32),
        _block: (u32, u32, u32),
        _args: &[*const u8],
    ) -> Result<()> {
        Err(runtime_error!("Kernel launch not supported on WASM runtime backend"))
    }
    
    fn allocate_memory(&self, size: usize) -> Result<*mut u8> {
        // For CPU backend, we just use regular heap allocation
        let layout = std::alloc::Layout::from_size_align(size, 8)
            .map_err(|e| runtime_error!("Invalid layout: {}", e))?;
        
        let ptr = unsafe { std::alloc::alloc(layout) };
        
        if ptr.is_null() {
            return Err(runtime_error!("Failed to allocate {} bytes", size));
        }
        
        Ok(ptr)
    }
    
    fn free_memory(&self, ptr: *mut u8) -> Result<()> {
        // We don't track size, so we'll use a reasonable default alignment
        // In a real implementation, we'd need to track allocated sizes
        // For now, this is just a stub
        Ok(())
    }
    
    fn copy_memory(
        &self,
        dst: *mut u8,
        src: *const u8,
        size: usize,
        _kind: MemcpyKind,
    ) -> Result<()> {
        // Safety: This function assumes the caller has verified the pointers are valid
        // and don't overlap, as required by the trait contract
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
        Ok(())
    }
    
    fn synchronize(&self) -> Result<()> {
        // No-op for CPU backend
        Ok(())
    }
    
}