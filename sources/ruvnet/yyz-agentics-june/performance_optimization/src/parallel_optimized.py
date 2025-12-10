"""
Parallel processing optimizations using multiprocessing and threading.
"""

import numpy as np
from typing import Tuple, Optional, List
import multiprocessing as mp
from multiprocessing import Pool, shared_memory
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from functools import partial
import queue


class ParallelOptimizedOps:
    """Parallel implementations of matrix operations."""
    
    def __init__(self, num_workers: Optional[int] = None):
        """Initialize with specified number of workers or auto-detect."""
        self.num_workers = num_workers or mp.cpu_count()
        
    @staticmethod
    def _multiply_row_range(args: Tuple[int, int, np.ndarray, np.ndarray]) -> np.ndarray:
        """Helper function to multiply a range of rows."""
        start_row, end_row, A_slice, B = args
        return A_slice @ B
    
    def matrix_multiply_multiprocess(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Parallel matrix multiplication using multiprocessing.
        Divides work by rows of A.
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        m, k = A.shape
        n = B.shape[1]
        
        # Calculate row chunks for each process
        chunk_size = max(1, m // self.num_workers)
        row_ranges = []
        
        for i in range(0, m, chunk_size):
            end = min(i + chunk_size, m)
            row_ranges.append((i, end))
        
        # Create pool and distribute work
        with Pool(processes=self.num_workers) as pool:
            # Prepare arguments for each worker
            args_list = [(start, end, A[start:end], B) 
                        for start, end in row_ranges]
            
            # Execute parallel multiplication
            results = pool.map(self._multiply_row_range, args_list)
        
        # Combine results
        C = np.vstack(results)
        return C
    
    def matrix_multiply_shared_memory(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Parallel matrix multiplication using shared memory.
        More efficient for large matrices as it avoids copying data.
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        m, k = A.shape
        n = B.shape[1]
        
        # Create shared memory for input matrices
        shm_a = shared_memory.SharedMemory(create=True, size=A.nbytes)
        shm_b = shared_memory.SharedMemory(create=True, size=B.nbytes)
        shm_c = shared_memory.SharedMemory(create=True, size=m * n * A.itemsize)
        
        # Copy data to shared memory
        np_array_a = np.ndarray(A.shape, dtype=A.dtype, buffer=shm_a.buf)
        np_array_b = np.ndarray(B.shape, dtype=B.dtype, buffer=shm_b.buf)
        np_array_c = np.ndarray((m, n), dtype=A.dtype, buffer=shm_c.buf)
        
        np_array_a[:] = A[:]
        np_array_b[:] = B[:]
        
        def worker(start_row, end_row, shm_names, shapes, dtype):
            """Worker function that operates on shared memory."""
            # Attach to existing shared memory
            existing_shm_a = shared_memory.SharedMemory(name=shm_names[0])
            existing_shm_b = shared_memory.SharedMemory(name=shm_names[1])
            existing_shm_c = shared_memory.SharedMemory(name=shm_names[2])
            
            # Create numpy arrays backed by shared memory
            A_shared = np.ndarray(shapes[0], dtype=dtype, buffer=existing_shm_a.buf)
            B_shared = np.ndarray(shapes[1], dtype=dtype, buffer=existing_shm_b.buf)
            C_shared = np.ndarray(shapes[2], dtype=dtype, buffer=existing_shm_c.buf)
            
            # Perform multiplication for assigned rows
            C_shared[start_row:end_row] = A_shared[start_row:end_row] @ B_shared
            
            # Clean up
            existing_shm_a.close()
            existing_shm_b.close()
            existing_shm_c.close()
        
        # Create processes
        processes = []
        chunk_size = max(1, m // self.num_workers)
        
        for i in range(0, m, chunk_size):
            end = min(i + chunk_size, m)
            p = mp.Process(
                target=worker,
                args=(i, end, [shm_a.name, shm_b.name, shm_c.name],
                     [A.shape, B.shape, (m, n)], A.dtype)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Copy result from shared memory
        result = np_array_c.copy()
        
        # Clean up shared memory
        shm_a.close()
        shm_b.close()
        shm_c.close()
        shm_a.unlink()
        shm_b.unlink()
        shm_c.unlink()
        
        return result
    
    def matrix_multiply_threaded(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Parallel matrix multiplication using threading.
        Best for I/O-bound operations or when using NumPy's internal threading.
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        m, k = A.shape
        n = B.shape[1]
        C = np.zeros((m, n), dtype=A.dtype)
        
        def multiply_rows(start_row: int, end_row: int):
            """Thread worker function."""
            C[start_row:end_row] = A[start_row:end_row] @ B
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            chunk_size = max(1, m // self.num_workers)
            futures = []
            
            for i in range(0, m, chunk_size):
                end = min(i + chunk_size, m)
                future = executor.submit(multiply_rows, i, end)
                futures.append(future)
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        return C
    
    def convolution_2d_parallel(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Parallel 2D convolution using multiprocessing.
        Divides output into tiles processed by different workers.
        """
        if len(image.shape) != 2 or len(kernel.shape) != 2:
            raise ValueError("Both image and kernel must be 2D arrays")
        
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        
        def convolve_tile(args):
            """Worker function to convolve a tile."""
            start_i, end_i, image, kernel = args
            tile_output = np.zeros((end_i - start_i, out_w), dtype=image.dtype)
            
            for i in range(start_i, end_i):
                for j in range(out_w):
                    for ki in range(ker_h):
                        for kj in range(ker_w):
                            tile_output[i - start_i, j] += (
                                image[i + ki, j + kj] * kernel[ki, kj]
                            )
            
            return start_i, tile_output
        
        # Divide work into chunks
        chunk_size = max(1, out_h // self.num_workers)
        args_list = []
        
        for i in range(0, out_h, chunk_size):
            end = min(i + chunk_size, out_h)
            args_list.append((i, end, image, kernel))
        
        # Process in parallel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(convolve_tile, args_list)
        
        # Combine results
        output = np.zeros((out_h, out_w), dtype=image.dtype)
        for start_i, tile_output in results:
            output[start_i:start_i + tile_output.shape[0]] = tile_output
        
        return output
    
    def batch_matrix_multiply_parallel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Parallel batch matrix multiplication.
        Distributes batches across workers.
        """
        if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
            raise ValueError(f"Invalid dimensions: {A.shape} x {B.shape}")
        
        batch_size = A.shape[0]
        
        def multiply_batch(args):
            """Worker function for batch multiplication."""
            start_batch, end_batch, A_batch, B_batch = args
            return np.matmul(A_batch, B_batch)
        
        # Divide batches among workers
        chunk_size = max(1, batch_size // self.num_workers)
        args_list = []
        
        for i in range(0, batch_size, chunk_size):
            end = min(i + chunk_size, batch_size)
            args_list.append((i, end, A[i:end], B[i:end]))
        
        # Process in parallel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(multiply_batch, args_list)
        
        # Combine results
        C = np.vstack(results)
        return C
    
    @staticmethod
    def parallel_reduce(data: np.ndarray, operation, num_workers: int = None) -> float:
        """
        Parallel reduction operation (sum, max, min, etc.).
        """
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        chunk_size = max(1, len(data) // num_workers)
        
        def reduce_chunk(chunk):
            return operation(chunk)
        
        # Divide data into chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process chunks in parallel
        with Pool(processes=num_workers) as pool:
            partial_results = pool.map(reduce_chunk, chunks)
        
        # Final reduction
        return operation(partial_results)


class OpenMPStyle:
    """
    OpenMP-style parallel patterns using Python threading.
    Note: Due to GIL, this is most effective with NumPy operations.
    """
    
    @staticmethod
    def parallel_for(func, iterations: int, num_threads: int = None):
        """
        Parallel for loop pattern similar to OpenMP.
        """
        if num_threads is None:
            num_threads = mp.cpu_count()
        
        def worker(start, end):
            for i in range(start, end):
                func(i)
        
        chunk_size = max(1, iterations // num_threads)
        threads = []
        
        for i in range(0, iterations, chunk_size):
            end = min(i + chunk_size, iterations)
            t = threading.Thread(target=worker, args=(i, end))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
    
    @staticmethod
    def parallel_reduce(data: np.ndarray, operation, num_threads: int = None):
        """
        Parallel reduction with thread-local storage.
        """
        if num_threads is None:
            num_threads = mp.cpu_count()
        
        chunk_size = max(1, len(data) // num_threads)
        results = [None] * num_threads
        
        def worker(thread_id, start, end):
            results[thread_id] = operation(data[start:end])
        
        threads = []
        for i in range(num_threads):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))
            if start < len(data):
                t = threading.Thread(target=worker, args=(i, start, end))
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join()
        
        # Final reduction
        return operation([r for r in results if r is not None])


def set_thread_affinity(thread_id: int, cpu_id: int):
    """
    Set thread CPU affinity for better cache locality.
    Platform-specific implementation required.
    """
    # This would use platform-specific APIs
    # Linux: sched_setaffinity
    # Windows: SetThreadAffinityMask
    pass


if __name__ == "__main__":
    print("Parallel Matrix Operations Testing")
    print("=" * 50)
    print(f"Number of CPU cores: {mp.cpu_count()}")
    
    # Test parallel multiplication
    sizes = [512, 1024]
    parallel_ops = ParallelOptimizedOps()
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        import time
        
        # Test multiprocess
        start = time.perf_counter()
        C_mp = parallel_ops.matrix_multiply_multiprocess(A, B)
        mp_time = time.perf_counter() - start
        
        # Test threaded
        start = time.perf_counter()
        C_thread = parallel_ops.matrix_multiply_threaded(A, B)
        thread_time = time.perf_counter() - start
        
        # Test NumPy (uses internal threading)
        start = time.perf_counter()
        C_numpy = A @ B
        numpy_time = time.perf_counter() - start
        
        print(f"Multiprocess time: {mp_time:.4f}s")
        print(f"Threaded time: {thread_time:.4f}s")
        print(f"NumPy time: {numpy_time:.4f}s")
        
        # Verify correctness
        print(f"Multiprocess correct: {np.allclose(C_mp, C_numpy)}")
        print(f"Threaded correct: {np.allclose(C_thread, C_numpy)}")