"""
Baseline matrix operations without optimizations.
These implementations serve as reference for performance comparisons.
"""

import numpy as np
import time
from typing import Tuple, List
import sys


class BaselineMatrixOps:
    """Baseline implementations of matrix operations without optimizations."""
    
    @staticmethod
    def matrix_multiply_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive matrix multiplication using three nested loops.
        Time complexity: O(n^3)
        Cache efficiency: Poor
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        m, k = A.shape
        k2, n = B.shape
        C = np.zeros((m, n))
        
        # Three nested loops - poor cache performance
        for i in range(m):
            for j in range(n):
                for l in range(k):
                    C[i, j] += A[i, l] * B[l, j]
        
        return C
    
    @staticmethod
    def convolution_2d_naive(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Naive 2D convolution implementation.
        Time complexity: O(m*n*k*l) where image is m×n and kernel is k×l
        """
        if len(image.shape) != 2 or len(kernel.shape) != 2:
            raise ValueError("Both image and kernel must be 2D arrays")
        
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        
        # Output dimensions
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        output = np.zeros((out_h, out_w))
        
        # Four nested loops - very poor performance
        for i in range(out_h):
            for j in range(out_w):
                for ki in range(ker_h):
                    for kj in range(ker_w):
                        output[i, j] += image[i + ki, j + kj] * kernel[ki, kj]
        
        return output
    
    @staticmethod
    def matrix_transpose_naive(A: np.ndarray) -> np.ndarray:
        """
        Naive matrix transpose.
        Poor cache performance due to column-wise access.
        """
        m, n = A.shape
        AT = np.zeros((n, m))
        
        for i in range(m):
            for j in range(n):
                AT[j, i] = A[i, j]
        
        return AT
    
    @staticmethod
    def vector_dot_product_naive(a: np.ndarray, b: np.ndarray) -> float:
        """
        Naive dot product implementation.
        """
        if a.shape != b.shape:
            raise ValueError("Vectors must have same shape")
        
        result = 0.0
        for i in range(len(a)):
            result += a[i] * b[i]
        
        return result
    
    @staticmethod
    def matrix_vector_multiply_naive(A: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Naive matrix-vector multiplication.
        """
        if A.shape[1] != x.shape[0]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {x.shape}")
        
        m, n = A.shape
        y = np.zeros(m)
        
        for i in range(m):
            for j in range(n):
                y[i] += A[i, j] * x[j]
        
        return y
    
    @staticmethod
    def batch_matrix_multiply_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive batch matrix multiplication for 3D tensors.
        A: (batch_size, m, k)
        B: (batch_size, k, n)
        Returns: (batch_size, m, n)
        """
        if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        batch_size = A.shape[0]
        m, k = A.shape[1], A.shape[2]
        n = B.shape[2]
        C = np.zeros((batch_size, m, n))
        
        # Five nested loops - extremely poor performance
        for b in range(batch_size):
            for i in range(m):
                for j in range(n):
                    for l in range(k):
                        C[b, i, j] += A[b, i, l] * B[b, l, j]
        
        return C


def generate_test_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random test matrices for benchmarking."""
    np.random.seed(seed)
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    return A, B


def generate_conv_test_data(img_size: int, kernel_size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random image and kernel for convolution benchmarking."""
    np.random.seed(seed)
    image = np.random.randn(img_size, img_size).astype(np.float32)
    kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
    return image, kernel


if __name__ == "__main__":
    print("Baseline Matrix Operations - Performance Testing")
    print("=" * 50)
    
    # Test different sizes
    sizes = [64, 128, 256, 512]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        A, B = generate_test_data(size)
        
        # Time naive multiplication
        start = time.perf_counter()
        C = BaselineMatrixOps.matrix_multiply_naive(A, B)
        end = time.perf_counter()
        
        naive_time = end - start
        print(f"Naive multiplication time: {naive_time:.4f} seconds")
        
        # Calculate GFLOPS
        flops = 2 * size**3  # 2n^3 operations for matrix multiplication
        gflops = (flops / 1e9) / naive_time
        print(f"Performance: {gflops:.2f} GFLOPS")
        
        # Memory bandwidth estimate
        memory_accessed = 3 * size**2 * 4  # 3 matrices, 4 bytes per float
        bandwidth = (memory_accessed / 1e9) / naive_time
        print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")
    
    print("\n" + "=" * 50)
    print("Convolution Performance Testing")
    
    # Test convolution
    img_sizes = [128, 256, 512]
    kernel_sizes = [3, 5, 7]
    
    for img_size in img_sizes:
        for ker_size in kernel_sizes:
            print(f"\nImage: {img_size}x{img_size}, Kernel: {ker_size}x{ker_size}")
            image, kernel = generate_conv_test_data(img_size, ker_size)
            
            start = time.perf_counter()
            output = BaselineMatrixOps.convolution_2d_naive(image, kernel)
            end = time.perf_counter()
            
            conv_time = end - start
            print(f"Naive convolution time: {conv_time:.4f} seconds")
            
            # Calculate operations
            out_size = img_size - ker_size + 1
            ops = out_size * out_size * ker_size * ker_size * 2
            gflops = (ops / 1e9) / conv_time
            print(f"Performance: {gflops:.2f} GFLOPS")