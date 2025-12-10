//! Common backend interface

use crate::Result;
use async_trait::async_trait;

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub name: String,
    pub supports_cuda: bool,
    pub supports_opencl: bool,
    pub supports_vulkan: bool,
    pub supports_webgpu: bool,
    pub max_threads: u32,
    pub max_threads_per_block: u32,
    pub max_blocks_per_grid: u32,
    pub max_shared_memory: usize,
    pub supports_dynamic_parallelism: bool,
    pub supports_unified_memory: bool,
    pub max_grid_dim: [u32; 3],
    pub max_block_dim: [u32; 3],
    pub warp_size: u32,
}

/// Common interface for all backends
#[async_trait]
pub trait BackendTrait: Send + Sync {
    /// Get backend name
    fn name(&self) -> &str;
    
    /// Get backend capabilities
    fn capabilities(&self) -> &BackendCapabilities;
    
    /// Initialize the backend
    async fn initialize(&mut self) -> Result<()>;
    
    /// Compile a kernel
    async fn compile_kernel(&self, source: &str) -> Result<Vec<u8>>;
    
    /// Launch a kernel
    async fn launch_kernel(
        &self,
        kernel: &[u8],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*const u8],
    ) -> Result<()>;
    
    /// Allocate device memory
    fn allocate_memory(&self, size: usize) -> Result<*mut u8>;
    
    /// Free device memory
    fn free_memory(&self, ptr: *mut u8) -> Result<()>;
    
    /// Copy memory
    fn copy_memory(
        &self,
        dst: *mut u8,
        src: *const u8,
        size: usize,
        kind: MemcpyKind,
    ) -> Result<()>;
    
    /// Synchronize device
    fn synchronize(&self) -> Result<()>;
}

/// Memory copy direction
#[derive(Debug, Clone, Copy)]
pub enum MemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost,
}