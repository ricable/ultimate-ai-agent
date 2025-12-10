//! Prelude module for convenient imports

pub use crate::error::{CudaRustError, Result};
pub use crate::runtime::{
    Runtime, Device, BackendType, Stream,
    Grid, Block, Dim3,
    launch_kernel, LaunchConfig, KernelFunction, ThreadContext,
};
pub use crate::memory::{DeviceBuffer, HostBuffer};

// Re-export macros - these are automatically available from #[macro_export]
// No need to re-export them here as they're already globally available