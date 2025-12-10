"""
Cache-optimized matrix operations using blocking, tiling, and memory-aware algorithms.
"""

import numpy as np
from typing import Tuple, Optional
import math


class CacheOptimizedOps:
    """Cache-efficient implementations of matrix operations."""
    
    # Cache sizes for typical modern CPUs (in KB)
    L1_CACHE_SIZE = 32 * 1024     # 32 KB
    L2_CACHE_SIZE = 256 * 1024    # 256 KB 
    L3_CACHE_SIZE = 8 * 1024 * 1024  # 8 MB
    
    @staticmethod
    def calculate_optimal_block_size(matrix_size: int, dtype_size: int = 4) -> int:
        """
        Calculate optimal block size based on cache size.
        Aim to fit 3 blocks in L2 cache (for A, B, and C blocks).
        """
        # Each block needs space for 3 matrices (A, B, C)
        available_cache = CacheOptimizedOps.L2_CACHE_SIZE // 3
        
        # Calculate block size that fits in cache
        # block_size^2 * dtype_size = available_cache
        block_size = int(math.sqrt(available_cache / dtype_size))
        
        # Round down to nearest power of 2 or multiple of 16 for better alignment
        block_size = (block_size // 16) * 16
        
        # Ensure block size divides matrix size reasonably
        while matrix_size % block_size > block_size // 2 and block_size > 16:
            block_size -= 16
            
        return max(16, min(block_size, matrix_size))
    
    @staticmethod
    def matrix_multiply_blocked(A: np.ndarray, B: np.ndarray, 
                               block_size: Optional[int] = None) -> np.ndarray:
        """
        Blocked matrix multiplication for better cache utilization.
        Divides matrices into blocks that fit in cache.
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        m, k = A.shape
        k2, n = B.shape
        
        # Auto-calculate block size if not provided
        if block_size is None:
            block_size = CacheOptimizedOps.calculate_optimal_block_size(min(m, n, k))
        
        C = np.zeros((m, n), dtype=A.dtype)
        
        # Block-wise multiplication
        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)
            for j in range(0, n, block_size):
                j_end = min(j + block_size, n)
                for l in range(0, k, block_size):
                    l_end = min(l + block_size, k)
                    
                    # Multiply blocks - these fit in cache
                    A_block = A[i:i_end, l:l_end]
                    B_block = B[l:l_end, j:j_end]
                    C[i:i_end, j:j_end] += A_block @ B_block
        
        return C
    
    @staticmethod
    def matrix_multiply_tiled_2d(A: np.ndarray, B: np.ndarray,
                                tile_m: int = 64, tile_n: int = 64, 
                                tile_k: int = 64) -> np.ndarray:
        """
        2D tiled matrix multiplication with different tile sizes for each dimension.
        Allows fine-tuning for specific cache hierarchies.
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        m, k = A.shape
        k2, n = B.shape
        C = np.zeros((m, n), dtype=A.dtype)
        
        # Tiled multiplication with register blocking
        for i0 in range(0, m, tile_m):
            i1 = min(i0 + tile_m, m)
            for j0 in range(0, n, tile_n):
                j1 = min(j0 + tile_n, n)
                for l0 in range(0, k, tile_k):
                    l1 = min(l0 + tile_k, k)
                    
                    # Inner kernel - fits in L1 cache
                    # Use explicit loops for potential compiler optimization
                    for i in range(i0, i1, 4):  # Unroll by 4
                        for j in range(j0, j1, 4):
                            # Register blocking
                            c00 = C[i, j] if i < m and j < n else 0
                            c01 = C[i, j+1] if i < m and j+1 < n else 0
                            c10 = C[i+1, j] if i+1 < m and j < n else 0
                            c11 = C[i+1, j+1] if i+1 < m and j+1 < n else 0
                            
                            for l in range(l0, l1):
                                a0 = A[i, l] if i < m else 0
                                a1 = A[i+1, l] if i+1 < m else 0
                                b0 = B[l, j] if j < n else 0
                                b1 = B[l, j+1] if j+1 < n else 0
                                
                                c00 += a0 * b0
                                c01 += a0 * b1
                                c10 += a1 * b0
                                c11 += a1 * b1
                            
                            if i < m and j < n:
                                C[i, j] = c00
                            if i < m and j+1 < n:
                                C[i, j+1] = c01
                            if i+1 < m and j < n:
                                C[i+1, j] = c10
                            if i+1 < m and j+1 < n:
                                C[i+1, j+1] = c11
        
        return C
    
    @staticmethod
    def matrix_transpose_blocked(A: np.ndarray, block_size: int = 64) -> np.ndarray:
        """
        Cache-friendly matrix transpose using blocking.
        Improves spatial locality by processing blocks that fit in cache.
        """
        m, n = A.shape
        AT = np.zeros((n, m), dtype=A.dtype)
        
        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)
            for j in range(0, n, block_size):
                j_end = min(j + block_size, n)
                
                # Transpose block - better cache locality
                AT[j:j_end, i:i_end] = A[i:i_end, j:j_end].T
        
        return AT
    
    @staticmethod
    def convolution_2d_tiled(image: np.ndarray, kernel: np.ndarray,
                            tile_size: int = 64) -> np.ndarray:
        """
        Tiled 2D convolution for better cache performance.
        Processes image in tiles to improve data locality.
        """
        if len(image.shape) != 2 or len(kernel.shape) != 2:
            raise ValueError("Both image and kernel must be 2D arrays")
        
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        output = np.zeros((out_h, out_w), dtype=image.dtype)
        
        # Process image in tiles
        for ti in range(0, out_h, tile_size):
            ti_end = min(ti + tile_size, out_h)
            for tj in range(0, out_w, tile_size):
                tj_end = min(tj + tile_size, out_w)
                
                # Process tile - better cache locality
                for i in range(ti, ti_end):
                    for j in range(tj, tj_end):
                        # Convolution for single output pixel
                        for ki in range(ker_h):
                            for kj in range(ker_w):
                                output[i, j] += image[i + ki, j + kj] * kernel[ki, kj]
        
        return output
    
    @staticmethod
    def convolution_2d_im2col(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Convolution using im2col transformation.
        Converts convolution to matrix multiplication for better cache usage.
        """
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        
        # Create im2col matrix - each column is a flattened patch
        col_matrix = np.zeros((ker_h * ker_w, out_h * out_w), dtype=image.dtype)
        
        col_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = image[i:i+ker_h, j:j+ker_w].flatten()
                col_matrix[:, col_idx] = patch
                col_idx += 1
        
        # Flatten kernel
        kernel_flat = kernel.flatten()
        
        # Convolution as matrix multiplication
        output_flat = kernel_flat @ col_matrix
        
        # Reshape to output dimensions
        output = output_flat.reshape(out_h, out_w)
        
        return output
    
    @staticmethod
    def matrix_vector_multiply_blocked(A: np.ndarray, x: np.ndarray,
                                     block_size: int = 256) -> np.ndarray:
        """
        Blocked matrix-vector multiplication for better cache usage.
        """
        if A.shape[1] != x.shape[0]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {x.shape}")
        
        m, n = A.shape
        y = np.zeros(m, dtype=A.dtype)
        
        # Process in blocks for better cache locality
        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)
            for j in range(0, n, block_size):
                j_end = min(j + block_size, n)
                
                # Block multiplication
                y[i:i_end] += A[i:i_end, j:j_end] @ x[j:j_end]
        
        return y
    
    @staticmethod
    def batch_matrix_multiply_blocked(A: np.ndarray, B: np.ndarray,
                                    block_size: int = 64) -> np.ndarray:
        """
        Blocked batch matrix multiplication.
        Reuses blocks across batch dimension for better cache efficiency.
        """
        if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        batch_size = A.shape[0]
        m, k = A.shape[1], A.shape[2]
        n = B.shape[2]
        C = np.zeros((batch_size, m, n), dtype=A.dtype)
        
        # Block across spatial dimensions, vectorize across batch
        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)
            for j in range(0, n, block_size):
                j_end = min(j + block_size, n)
                for l in range(0, k, block_size):
                    l_end = min(l + block_size, k)
                    
                    # Process all batches at once for this block
                    # Better cache reuse across batch dimension
                    A_block = A[:, i:i_end, l:l_end]
                    B_block = B[:, l:l_end, j:j_end]
                    C[:, i:i_end, j:j_end] += np.matmul(A_block, B_block)
        
        return C


def prefetch_data(data: np.ndarray) -> None:
    """
    Hint to prefetch data into cache.
    In practice, this would use compiler intrinsics or assembly.
    """
    # Touch data to bring into cache
    _ = data.sum()


def align_array(arr: np.ndarray, alignment: int = 64) -> np.ndarray:
    """
    Ensure array is aligned to cache line boundaries.
    Alignment should typically be 64 bytes for modern CPUs.
    """
    if arr.ctypes.data % alignment == 0:
        return arr
    
    # Create aligned copy
    aligned = np.empty(arr.shape, dtype=arr.dtype, order='C')
    aligned[:] = arr
    return aligned


if __name__ == "__main__":
    print("Cache-Optimized Matrix Operations Testing")
    print("=" * 50)
    
    # Test optimal block size calculation
    sizes = [128, 256, 512, 1024]
    for size in sizes:
        optimal_block = CacheOptimizedOps.calculate_optimal_block_size(size)
        print(f"Matrix size: {size}x{size}, Optimal block size: {optimal_block}")
    
    print("\nTesting blocked multiplication...")
    A = np.random.randn(512, 512).astype(np.float32)
    B = np.random.randn(512, 512).astype(np.float32)
    
    import time
    
    # Test different block sizes
    block_sizes = [16, 32, 64, 128, 256]
    for bs in block_sizes:
        start = time.perf_counter()
        C = CacheOptimizedOps.matrix_multiply_blocked(A, B, block_size=bs)
        end = time.perf_counter()
        
        elapsed = end - start
        gflops = (2 * 512**3 / 1e9) / elapsed
        print(f"Block size {bs}: {elapsed:.4f}s, {gflops:.2f} GFLOPS")