//! Unified memory management for CUDA-Rust
//!
//! Provides unified memory allocation that can be accessed by both host and device

use crate::{Result, memory_error};
use std::sync::Arc;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Unified memory allocation
pub struct UnifiedMemory {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl UnifiedMemory {
    /// Allocate unified memory
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(memory_error!("Cannot allocate zero-sized unified memory"));
        }

        let layout = Layout::from_size_align(size, 8)
            .map_err(|e| memory_error!("Invalid layout: {}", e))?;

        let ptr = unsafe { alloc(layout) };
        
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| memory_error!("Failed to allocate unified memory"))?;

        Ok(Self { ptr, size, layout })
    }

    /// Get a pointer to the memory
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr() as *const u8
    }

    /// Get a mutable pointer to the memory
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy data from host to unified memory
    pub fn copy_from_slice(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(memory_error!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                self.size
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr(), data.len());
        }

        Ok(())
    }

    /// Copy data from unified memory to host
    pub fn copy_to_slice(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(memory_error!(
                "Destination size {} exceeds buffer size {}",
                data.len(),
                self.size
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), data.as_mut_ptr(), data.len());
        }

        Ok(())
    }
}

impl Drop for UnifiedMemory {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

// Safety: UnifiedMemory can be safely sent between threads
unsafe impl Send for UnifiedMemory {}
unsafe impl Sync for UnifiedMemory {}

/// Shared unified memory handle
pub type SharedUnifiedMemory = Arc<UnifiedMemory>;

/// Create a new shared unified memory allocation
pub fn allocate_unified(size: usize) -> Result<SharedUnifiedMemory> {
    Ok(Arc::new(UnifiedMemory::new(size)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_memory_allocation() {
        let mem = UnifiedMemory::new(1024).unwrap();
        assert_eq!(mem.size(), 1024);
    }

    #[test]
    fn test_unified_memory_copy() {
        let mut mem = UnifiedMemory::new(256).unwrap();
        
        let data = vec![42u8; 256];
        mem.copy_from_slice(&data).unwrap();
        
        let mut output = vec![0u8; 256];
        mem.copy_to_slice(&mut output).unwrap();
        
        assert_eq!(data, output);
    }

    #[test]
    fn test_zero_size_allocation() {
        let result = UnifiedMemory::new(0);
        assert!(result.is_err());
    }
}