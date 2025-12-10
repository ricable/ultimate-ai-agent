"""
Vectorized and SIMD optimizations for matrix operations.
Uses NumPy's vectorization capabilities and explicit SIMD patterns.
"""

import numpy as np
from typing import Tuple, Optional
import numba
from numba import jit, prange, vectorize, float32, float64
import warnings


class VectorizedOps:
    """Vectorized and SIMD-optimized matrix operations."""
    
    @staticmethod
    def matrix_multiply_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Vectorized matrix multiplication using NumPy's optimized routines.
        NumPy automatically uses BLAS libraries with SIMD instructions.
        """
        # NumPy's @ operator uses optimized BLAS routines
        return A @ B
    
    @staticmethod
    def matrix_multiply_einsum(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using Einstein summation.
        Can be more efficient for certain tensor operations.
        """
        return np.einsum('ij,jk->ik', A, B, optimize=True)
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def matrix_multiply_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        JIT-compiled matrix multiplication with auto-vectorization.
        Numba compiles to machine code with SIMD instructions.
        """
        m, k = A.shape
        n = B.shape[1]
        C = np.zeros((m, n), dtype=A.dtype)
        
        # Parallel outer loops with vectorized inner loop
        for i in prange(m):
            for j in range(n):
                # This inner loop will be auto-vectorized by Numba
                for l in range(k):
                    C[i, j] += A[i, l] * B[l, j]
        
        return C
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def matrix_multiply_numba_tiled(A: np.ndarray, B: np.ndarray, 
                                   tile_size: int = 32) -> np.ndarray:
        """
        Tiled matrix multiplication with Numba JIT and SIMD.
        Combines cache optimization with vectorization.
        """
        m, k = A.shape
        n = B.shape[1]
        C = np.zeros((m, n), dtype=A.dtype)
        
        # Tiled multiplication with vectorization
        for i0 in prange(0, m, tile_size):
            for j0 in range(0, n, tile_size):
                for k0 in range(0, k, tile_size):
                    # Process tile
                    for i in range(i0, min(i0 + tile_size, m)):
                        for j in range(j0, min(j0 + tile_size, n)):
                            # Vectorizable inner loop
                            acc = C[i, j]
                            for l in range(k0, min(k0 + tile_size, k)):
                                acc += A[i, l] * B[l, j]
                            C[i, j] = acc
        
        return C
    
    @staticmethod
    def convolution_2d_vectorized(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Vectorized 2D convolution using stride tricks.
        Creates a view of the image with overlapping windows.
        """
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        
        # Create strided view of image for all patches at once
        shape = (out_h, out_w, ker_h, ker_w)
        strides = (*image.strides, *image.strides)
        
        # Create view without copying data
        patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
        
        # Vectorized convolution: element-wise multiply and sum
        # This uses SIMD instructions for the multiplication and reduction
        output = np.einsum('ijkl,kl->ij', patches, kernel, optimize=True)
        
        return output
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def convolution_2d_numba(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        JIT-compiled convolution with auto-vectorization.
        """
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        output = np.zeros((out_h, out_w), dtype=image.dtype)
        
        # Parallel outer loops
        for i in prange(out_h):
            for j in range(out_w):
                acc = 0.0
                # Inner loops will be vectorized
                for ki in range(ker_h):
                    for kj in range(ker_w):
                        acc += image[i + ki, j + kj] * kernel[ki, kj]
                output[i, j] = acc
        
        return output
    
    @staticmethod
    def batch_matrix_multiply_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Vectorized batch matrix multiplication.
        Uses optimized tensor operations.
        """
        # NumPy's matmul handles batch dimensions efficiently
        return np.matmul(A, B)
    
    @staticmethod
    def batch_matrix_multiply_einsum(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Batch matrix multiplication using Einstein summation.
        Can be more memory efficient for certain shapes.
        """
        return np.einsum('bij,bjk->bik', A, B, optimize=True)
    
    @staticmethod
    @vectorize([float32(float32, float32), float64(float64, float64)], 
               target='parallel')
    def vectorized_add(a, b):
        """Example of explicit SIMD operation using Numba vectorize."""
        return a + b
    
    @staticmethod
    @vectorize([float32(float32, float32), float64(float64, float64)], 
               target='parallel')
    def vectorized_multiply(a, b):
        """Vectorized element-wise multiplication."""
        return a * b
    
    @staticmethod
    def fused_multiply_add(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Fused multiply-add operation (FMA).
        Modern CPUs have dedicated FMA instructions.
        """
        # NumPy may use FMA instructions automatically
        return np.add(np.multiply(A, B), C)
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def dot_product_simd(a: np.ndarray, b: np.ndarray) -> float:
        """
        SIMD-optimized dot product.
        Numba will use vectorized instructions.
        """
        n = len(a)
        sum_val = 0.0
        
        # Process in chunks for better vectorization
        chunk_size = 8  # Typical SIMD width for AVX
        
        # Vectorized main loop
        for i in range(0, n - chunk_size + 1, chunk_size):
            # This will be compiled to SIMD instructions
            for j in range(chunk_size):
                sum_val += a[i + j] * b[i + j]
        
        # Handle remaining elements
        for i in range((n // chunk_size) * chunk_size, n):
            sum_val += a[i] * b[i]
        
        return sum_val
    
    @staticmethod
    def matrix_transpose_vectorized(A: np.ndarray) -> np.ndarray:
        """
        Vectorized matrix transpose.
        NumPy's transpose creates a view with swapped strides.
        """
        return A.T
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def matrix_transpose_blocked_simd(A: np.ndarray, block_size: int = 32) -> np.ndarray:
        """
        Cache-friendly transpose with SIMD optimization.
        """
        m, n = A.shape
        AT = np.zeros((n, m), dtype=A.dtype)
        
        # Blocked transpose for cache efficiency
        for i in prange(0, m, block_size):
            for j in range(0, n, block_size):
                # Process block with vectorization
                for bi in range(i, min(i + block_size, m)):
                    for bj in range(j, min(j + block_size, n)):
                        AT[bj, bi] = A[bi, bj]
        
        return AT


class SIMDPatterns:
    """Common SIMD patterns and operations."""
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def horizontal_sum(arr: np.ndarray) -> float:
        """Horizontal sum using SIMD reduction."""
        n = len(arr)
        sum_val = 0.0
        
        # Main vectorized loop
        for i in range(0, n - 3, 4):
            # Process 4 elements at once (simulating SIMD)
            sum_val += arr[i] + arr[i+1] + arr[i+2] + arr[i+3]
        
        # Handle remaining elements
        for i in range((n // 4) * 4, n):
            sum_val += arr[i]
        
        return sum_val
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def packed_arithmetic(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Perform multiple arithmetic operations in parallel.
        Simulates SIMD packed operations.
        """
        n = len(a)
        result = np.zeros(n, dtype=a.dtype)
        
        # Process multiple elements simultaneously
        for i in range(0, n - 3, 4):
            # These operations can be done in parallel on SIMD units
            result[i] = a[i] * b[i] + c[i]
            result[i+1] = a[i+1] * b[i+1] + c[i+1]
            result[i+2] = a[i+2] * b[i+2] + c[i+2]
            result[i+3] = a[i+3] * b[i+3] + c[i+3]
        
        # Handle remaining elements
        for i in range((n // 4) * 4, n):
            result[i] = a[i] * b[i] + c[i]
        
        return result
    
    @staticmethod
    def broadcast_operations(scalar: float, vector: np.ndarray) -> np.ndarray:
        """
        Broadcast scalar to vector - efficiently uses SIMD.
        """
        # NumPy broadcasting automatically uses SIMD
        return scalar * vector
    
    @staticmethod
    def masked_operations(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Masked SIMD operations for conditional processing.
        """
        # NumPy's masked operations are optimized
        return np.where(mask, data * 2, data)


def create_aligned_array(shape: Tuple[int, ...], dtype=np.float32, 
                        alignment: int = 32) -> np.ndarray:
    """
    Create an array aligned for optimal SIMD performance.
    Alignment should match SIMD register width (32 bytes for AVX).
    """
    # Calculate total size with padding
    total_elements = np.prod(shape)
    bytes_needed = total_elements * np.dtype(dtype).itemsize
    
    # Allocate aligned memory (would use posix_memalign in C)
    # NumPy arrays are usually aligned, but we ensure it here
    arr = np.empty(shape, dtype=dtype, order='C')
    
    # Check alignment
    if arr.ctypes.data % alignment != 0:
        warnings.warn(f"Array not aligned to {alignment} bytes")
    
    return arr


if __name__ == "__main__":
    print("Vectorized and SIMD Operations Testing")
    print("=" * 50)
    
    # Ensure Numba compilation
    print("Compiling Numba functions...")
    A_test = np.random.randn(64, 64).astype(np.float32)
    B_test = np.random.randn(64, 64).astype(np.float32)
    _ = VectorizedOps.matrix_multiply_numba(A_test, B_test)
    
    import time
    
    sizes = [256, 512, 1024]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Test NumPy (uses BLAS with SIMD)
        start = time.perf_counter()
        C_numpy = A @ B
        numpy_time = time.perf_counter() - start
        
        # Test einsum
        start = time.perf_counter()
        C_einsum = VectorizedOps.matrix_multiply_einsum(A, B)
        einsum_time = time.perf_counter() - start
        
        # Test Numba JIT
        start = time.perf_counter()
        C_numba = VectorizedOps.matrix_multiply_numba(A, B)
        numba_time = time.perf_counter() - start
        
        print(f"NumPy BLAS time: {numpy_time:.4f}s")
        print(f"Einsum time: {einsum_time:.4f}s")
        print(f"Numba JIT time: {numba_time:.4f}s")
        
        # Calculate GFLOPS
        flops = 2 * size**3
        print(f"NumPy GFLOPS: {(flops/1e9)/numpy_time:.2f}")
        print(f"Numba GFLOPS: {(flops/1e9)/numba_time:.2f}")
    
    # Test convolution
    print("\nConvolution Testing")
    img_size = 512
    kernel_size = 7
    image = np.random.randn(img_size, img_size).astype(np.float32)
    kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
    
    start = time.perf_counter()
    conv_vectorized = VectorizedOps.convolution_2d_vectorized(image, kernel)
    vec_time = time.perf_counter() - start
    
    start = time.perf_counter()
    conv_numba = VectorizedOps.convolution_2d_numba(image, kernel)
    numba_time = time.perf_counter() - start
    
    print(f"Vectorized convolution time: {vec_time:.4f}s")
    print(f"Numba convolution time: {numba_time:.4f}s")