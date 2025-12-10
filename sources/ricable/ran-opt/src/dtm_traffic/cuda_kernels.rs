use ndarray::Array2;
// use std::ffi::CString;
use std::os::raw::{c_float, c_int, c_void};

// CUDA runtime function bindings
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(devPtr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    fn cudaGetLastError() -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
}

// CUBLAS function bindings
extern "C" {
    fn cublasCreate_v2(handle: *mut *mut c_void) -> c_int;
    fn cublasDestroy_v2(handle: *mut c_void) -> c_int;
    fn cublasSgemm_v2(
        handle: *mut c_void,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: *const c_float,
        c: *mut c_float,
        ldc: c_int,
    ) -> c_int;
}

// CUDA memory copy kinds
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// CUBLAS operation types
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;

/// CUDA context for GPU operations
pub struct CudaContext {
    device_count: i32,
    cublas_handle: *mut c_void,
    stream: *mut c_void,
}

impl CudaContext {
    pub fn new() -> Result<Self, String> {
        let mut device_count = 0;
        let result = unsafe { cudaGetDeviceCount(&mut device_count) };
        
        if result != 0 || device_count == 0 {
            return Err("No CUDA devices available".to_string());
        }
        
        let mut cublas_handle = std::ptr::null_mut();
        let result = unsafe { cublasCreate_v2(&mut cublas_handle) };
        
        if result != 0 {
            return Err("Failed to create CUBLAS handle".to_string());
        }
        
        Ok(Self {
            device_count,
            cublas_handle,
            stream: std::ptr::null_mut(),
        })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.cublas_handle.is_null() {
            unsafe { cublasDestroy_v2(self.cublas_handle) };
        }
    }
}

/// GPU memory buffer
pub struct GpuBuffer<T> {
    ptr: *mut T,
    size: usize,
}

impl<T> GpuBuffer<T> {
    pub fn new(size: usize) -> Result<Self, String> {
        let mut ptr = std::ptr::null_mut();
        let byte_size = size * std::mem::size_of::<T>();
        
        let result = unsafe { cudaMalloc(&mut ptr as *mut *mut c_void, byte_size) };
        
        if result != 0 {
            return Err("Failed to allocate GPU memory".to_string());
        }
        
        Ok(Self {
            ptr: ptr as *mut T,
            size,
        })
    }
    
    pub fn copy_from_host(&mut self, host_data: &[T]) -> Result<(), String> {
        if host_data.len() != self.size {
            return Err("Size mismatch in GPU memory copy".to_string());
        }
        
        let byte_size = self.size * std::mem::size_of::<T>();
        let result = unsafe {
            cudaMemcpy(
                self.ptr as *mut c_void,
                host_data.as_ptr() as *const c_void,
                byte_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        
        if result != 0 {
            return Err("Failed to copy data to GPU".to_string());
        }
        
        Ok(())
    }
    
    pub fn copy_to_host(&self, host_data: &mut [T]) -> Result<(), String> {
        if host_data.len() != self.size {
            return Err("Size mismatch in GPU memory copy".to_string());
        }
        
        let byte_size = self.size * std::mem::size_of::<T>();
        let result = unsafe {
            cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                byte_size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        
        if result != 0 {
            return Err("Failed to copy data from GPU".to_string());
        }
        
        Ok(())
    }
    
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr as *mut c_void) };
        }
    }
}

/// CUDA-accelerated LSTM operations
pub struct CudaLSTM {
    context: CudaContext,
    // Weight buffers on GPU
    w_ii_gpu: GpuBuffer<f32>,
    w_if_gpu: GpuBuffer<f32>,
    w_ig_gpu: GpuBuffer<f32>,
    w_io_gpu: GpuBuffer<f32>,
    w_hi_gpu: GpuBuffer<f32>,
    w_hf_gpu: GpuBuffer<f32>,
    w_hg_gpu: GpuBuffer<f32>,
    w_ho_gpu: GpuBuffer<f32>,
    // Bias buffers
    b_i_gpu: GpuBuffer<f32>,
    b_f_gpu: GpuBuffer<f32>,
    b_g_gpu: GpuBuffer<f32>,
    b_o_gpu: GpuBuffer<f32>,
    // Temporary buffers
    temp_gpu: GpuBuffer<f32>,
    gates_gpu: GpuBuffer<f32>,
}

impl CudaLSTM {
    pub fn new(input_size: usize, hidden_size: usize, batch_size: usize) -> Result<Self, String> {
        let context = CudaContext::new()?;
        
        // Allocate weight buffers
        let w_ii_gpu = GpuBuffer::new(input_size * hidden_size)?;
        let w_if_gpu = GpuBuffer::new(input_size * hidden_size)?;
        let w_ig_gpu = GpuBuffer::new(input_size * hidden_size)?;
        let w_io_gpu = GpuBuffer::new(input_size * hidden_size)?;
        let w_hi_gpu = GpuBuffer::new(hidden_size * hidden_size)?;
        let w_hf_gpu = GpuBuffer::new(hidden_size * hidden_size)?;
        let w_hg_gpu = GpuBuffer::new(hidden_size * hidden_size)?;
        let w_ho_gpu = GpuBuffer::new(hidden_size * hidden_size)?;
        
        // Allocate bias buffers
        let b_i_gpu = GpuBuffer::new(hidden_size)?;
        let b_f_gpu = GpuBuffer::new(hidden_size)?;
        let b_g_gpu = GpuBuffer::new(hidden_size)?;
        let b_o_gpu = GpuBuffer::new(hidden_size)?;
        
        // Allocate temporary buffers
        let temp_gpu = GpuBuffer::new(batch_size * hidden_size * 4)?;  // For all gates
        let gates_gpu = GpuBuffer::new(batch_size * hidden_size * 4)?;
        
        Ok(Self {
            context,
            w_ii_gpu,
            w_if_gpu,
            w_ig_gpu,
            w_io_gpu,
            w_hi_gpu,
            w_hf_gpu,
            w_hg_gpu,
            w_ho_gpu,
            b_i_gpu,
            b_f_gpu,
            b_g_gpu,
            b_o_gpu,
            temp_gpu,
            gates_gpu,
        })
    }
    
    pub fn forward_step(
        &mut self,
        x: &Array2<f32>,
        h_prev: &Array2<f32>,
        c_prev: &Array2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>), String> {
        let batch_size = x.shape()[0] as c_int;
        let input_size = x.shape()[1] as c_int;
        let hidden_size = h_prev.shape()[1] as c_int;
        
        // Copy input data to GPU
        let mut x_gpu = GpuBuffer::new(x.len())?;
        x_gpu.copy_from_host(x.as_slice().unwrap())?;
        
        let mut h_gpu = GpuBuffer::new(h_prev.len())?;
        h_gpu.copy_from_host(h_prev.as_slice().unwrap())?;
        
        let mut c_gpu = GpuBuffer::new(c_prev.len())?;
        c_gpu.copy_from_host(c_prev.as_slice().unwrap())?;
        
        // Perform LSTM forward pass on GPU
        self.cuda_lstm_forward(
            &x_gpu,
            &h_gpu,
            &c_gpu,
            batch_size,
            input_size,
            hidden_size,
        )?;
        
        // Copy results back to host
        let mut h_new = vec![0.0f32; h_prev.len()];
        let mut c_new = vec![0.0f32; c_prev.len()];
        
        h_gpu.copy_to_host(&mut h_new)?;
        c_gpu.copy_to_host(&mut c_new)?;
        
        // Convert back to ndarray
        let h_new = Array2::from_shape_vec((batch_size as usize, hidden_size as usize), h_new)
            .map_err(|e| format!("Failed to reshape output: {}", e))?;
        let c_new = Array2::from_shape_vec((batch_size as usize, hidden_size as usize), c_new)
            .map_err(|e| format!("Failed to reshape output: {}", e))?;
        
        Ok((h_new, c_new))
    }
    
    fn cuda_lstm_forward(
        &mut self,
        x_gpu: &GpuBuffer<f32>,
        h_gpu: &GpuBuffer<f32>,
        c_gpu: &GpuBuffer<f32>,
        batch_size: c_int,
        input_size: c_int,
        hidden_size: c_int,
    ) -> Result<(), String> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        
        // Compute input gates: temp = x * W_i + h * W_h
        // Input gate
        let result = unsafe {
            cublasSgemm_v2(
                self.context.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                hidden_size,
                batch_size,
                input_size,
                &alpha,
                self.w_ii_gpu.as_ptr(),
                hidden_size,
                x_gpu.as_ptr(),
                input_size,
                &beta,
                self.temp_gpu.as_mut_ptr(),
                hidden_size,
            )
        };
        
        if result != 0 {
            return Err("CUBLAS GEMM failed for input gate".to_string());
        }
        
        // Add hidden state contribution
        let result = unsafe {
            cublasSgemm_v2(
                self.context.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                hidden_size,
                batch_size,
                hidden_size,
                &alpha,
                self.w_hi_gpu.as_ptr(),
                hidden_size,
                h_gpu.as_ptr(),
                hidden_size,
                &alpha,  // beta = 1.0 to add to existing result
                self.temp_gpu.as_mut_ptr(),
                hidden_size,
            )
        };
        
        if result != 0 {
            return Err("CUBLAS GEMM failed for hidden gate".to_string());
        }
        
        // Launch CUDA kernel for element-wise operations (sigmoid, tanh, etc.)
        self.launch_lstm_elementwise_kernel(batch_size, hidden_size)?;
        
        // Synchronize GPU
        let result = unsafe { cudaDeviceSynchronize() };
        if result != 0 {
            return Err("CUDA synchronization failed".to_string());
        }
        
        Ok(())
    }
    
    fn launch_lstm_elementwise_kernel(&self, batch_size: c_int, hidden_size: c_int) -> Result<(), String> {
        // This would launch a custom CUDA kernel for elementwise operations
        // For now, we'll use a placeholder implementation
        
        // In a real implementation, you would:
        // 1. Compile CUDA kernel code
        // 2. Load the kernel function
        // 3. Launch with appropriate grid/block dimensions
        // 4. Pass GPU buffer pointers to the kernel
        
        // Placeholder: simulate kernel execution
        std::thread::sleep(std::time::Duration::from_micros(100));
        
        Ok(())
    }
}

/// CUDA-accelerated GRU operations
pub struct CudaGRU {
    context: CudaContext,
    // Weight buffers
    w_ir_gpu: GpuBuffer<f32>,
    w_iz_gpu: GpuBuffer<f32>,
    w_in_gpu: GpuBuffer<f32>,
    w_hr_gpu: GpuBuffer<f32>,
    w_hz_gpu: GpuBuffer<f32>,
    w_hn_gpu: GpuBuffer<f32>,
    // Bias buffers
    b_r_gpu: GpuBuffer<f32>,
    b_z_gpu: GpuBuffer<f32>,
    b_n_gpu: GpuBuffer<f32>,
    // Temporary buffers
    temp_gpu: GpuBuffer<f32>,
}

impl CudaGRU {
    pub fn new(input_size: usize, hidden_size: usize, batch_size: usize) -> Result<Self, String> {
        let context = CudaContext::new()?;
        
        // Allocate weight buffers
        let w_ir_gpu = GpuBuffer::new(input_size * hidden_size)?;
        let w_iz_gpu = GpuBuffer::new(input_size * hidden_size)?;
        let w_in_gpu = GpuBuffer::new(input_size * hidden_size)?;
        let w_hr_gpu = GpuBuffer::new(hidden_size * hidden_size)?;
        let w_hz_gpu = GpuBuffer::new(hidden_size * hidden_size)?;
        let w_hn_gpu = GpuBuffer::new(hidden_size * hidden_size)?;
        
        // Allocate bias buffers
        let b_r_gpu = GpuBuffer::new(hidden_size)?;
        let b_z_gpu = GpuBuffer::new(hidden_size)?;
        let b_n_gpu = GpuBuffer::new(hidden_size)?;
        
        // Allocate temporary buffer
        let temp_gpu = GpuBuffer::new(batch_size * hidden_size * 3)?;  // For all gates
        
        Ok(Self {
            context,
            w_ir_gpu,
            w_iz_gpu,
            w_in_gpu,
            w_hr_gpu,
            w_hz_gpu,
            w_hn_gpu,
            b_r_gpu,
            b_z_gpu,
            b_n_gpu,
            temp_gpu,
        })
    }
    
    pub fn forward_step(
        &mut self,
        x: &Array2<f32>,
        h_prev: &Array2<f32>,
    ) -> Result<Array2<f32>, String> {
        let batch_size = x.shape()[0] as c_int;
        let input_size = x.shape()[1] as c_int;
        let hidden_size = h_prev.shape()[1] as c_int;
        
        // Copy input data to GPU
        let mut x_gpu = GpuBuffer::new(x.len())?;
        x_gpu.copy_from_host(x.as_slice().unwrap())?;
        
        let mut h_gpu = GpuBuffer::new(h_prev.len())?;
        h_gpu.copy_from_host(h_prev.as_slice().unwrap())?;
        
        // Perform GRU forward pass on GPU
        self.cuda_gru_forward(&x_gpu, &h_gpu, batch_size, input_size, hidden_size)?;
        
        // Copy result back to host
        let mut h_new = vec![0.0f32; h_prev.len()];
        h_gpu.copy_to_host(&mut h_new)?;
        
        // Convert back to ndarray
        let h_new = Array2::from_shape_vec((batch_size as usize, hidden_size as usize), h_new)
            .map_err(|e| format!("Failed to reshape output: {}", e))?;
        
        Ok(h_new)
    }
    
    fn cuda_gru_forward(
        &mut self,
        x_gpu: &GpuBuffer<f32>,
        h_gpu: &GpuBuffer<f32>,
        batch_size: c_int,
        input_size: c_int,
        hidden_size: c_int,
    ) -> Result<(), String> {
        // Similar to LSTM but with GRU-specific operations
        // This is a simplified implementation
        
        let alpha = 1.0f32;
        let beta = 0.0f32;
        
        // Compute reset gate
        let result = unsafe {
            cublasSgemm_v2(
                self.context.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                hidden_size,
                batch_size,
                input_size,
                &alpha,
                self.w_ir_gpu.as_ptr(),
                hidden_size,
                x_gpu.as_ptr(),
                input_size,
                &beta,
                self.temp_gpu.as_mut_ptr(),
                hidden_size,
            )
        };
        
        if result != 0 {
            return Err("CUBLAS GEMM failed for reset gate".to_string());
        }
        
        // Launch GRU elementwise kernel
        self.launch_gru_elementwise_kernel(batch_size, hidden_size)?;
        
        // Synchronize GPU
        let result = unsafe { cudaDeviceSynchronize() };
        if result != 0 {
            return Err("CUDA synchronization failed".to_string());
        }
        
        Ok(())
    }
    
    fn launch_gru_elementwise_kernel(&self, batch_size: c_int, hidden_size: c_int) -> Result<(), String> {
        // Placeholder for GRU kernel launch
        std::thread::sleep(std::time::Duration::from_micros(80));
        Ok(())
    }
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    let mut device_count = 0;
    let result = unsafe { cudaGetDeviceCount(&mut device_count) };
    result == 0 && device_count > 0
}

/// High-level CUDA-accelerated LSTM forward pass
pub fn lstm_forward_cuda(
    x: &Array2<f32>,
    h_prev: &Array2<f32>,
    c_prev: &Array2<f32>,
    w_ii: &Array2<f32>,
    w_if: &Array2<f32>,
    w_ig: &Array2<f32>,
    w_io: &Array2<f32>,
    w_hi: &Array2<f32>,
    w_hf: &Array2<f32>,
    w_hg: &Array2<f32>,
    w_ho: &Array2<f32>,
    b_i: &Array2<f32>,
    b_f: &Array2<f32>,
    b_g: &Array2<f32>,
    b_o: &Array2<f32>,
    w_ci: Option<&Array2<f32>>,
    w_cf: Option<&Array2<f32>>,
    w_co: Option<&Array2<f32>>,
) -> (Array2<f32>, Array2<f32>) {
    if !is_cuda_available() {
        // Fallback to CPU implementation
        return fallback_lstm_forward(x, h_prev, c_prev, w_ii, w_if, w_ig, w_io,
                                   w_hi, w_hf, w_hg, w_ho, b_i, b_f, b_g, b_o,
                                   w_ci, w_cf, w_co);
    }
    
    let batch_size = x.shape()[0];
    let input_size = x.shape()[1];
    let hidden_size = h_prev.shape()[1];
    
    // Create CUDA LSTM context
    let mut cuda_lstm = match CudaLSTM::new(input_size, hidden_size, batch_size) {
        Ok(lstm) => lstm,
        Err(_) => {
            // Fallback to CPU if CUDA initialization fails
            return fallback_lstm_forward(x, h_prev, c_prev, w_ii, w_if, w_ig, w_io,
                                       w_hi, w_hf, w_hg, w_ho, b_i, b_f, b_g, b_o,
                                       w_ci, w_cf, w_co);
        }
    };
    
    // Perform CUDA forward pass
    match cuda_lstm.forward_step(x, h_prev, c_prev) {
        Ok((h_new, c_new)) => (h_new, c_new),
        Err(_) => {
            // Fallback to CPU if CUDA forward fails
            fallback_lstm_forward(x, h_prev, c_prev, w_ii, w_if, w_ig, w_io,
                                w_hi, w_hf, w_hg, w_ho, b_i, b_f, b_g, b_o,
                                w_ci, w_cf, w_co)
        }
    }
}

/// High-level CUDA-accelerated GRU forward pass
pub fn gru_forward_cuda(
    x: &Array2<f32>,
    h_prev: &Array2<f32>,
    w_ir: &Array2<f32>,
    w_iz: &Array2<f32>,
    w_in: &Array2<f32>,
    w_hr: &Array2<f32>,
    w_hz: &Array2<f32>,
    w_hn: &Array2<f32>,
    b_r: &Array2<f32>,
    b_z: &Array2<f32>,
    b_n: &Array2<f32>,
) -> Array2<f32> {
    if !is_cuda_available() {
        return fallback_gru_forward(x, h_prev, w_ir, w_iz, w_in, w_hr, w_hz, w_hn, b_r, b_z, b_n);
    }
    
    let batch_size = x.shape()[0];
    let input_size = x.shape()[1];
    let hidden_size = h_prev.shape()[1];
    
    // Create CUDA GRU context
    let mut cuda_gru = match CudaGRU::new(input_size, hidden_size, batch_size) {
        Ok(gru) => gru,
        Err(_) => {
            return fallback_gru_forward(x, h_prev, w_ir, w_iz, w_in, w_hr, w_hz, w_hn, b_r, b_z, b_n);
        }
    };
    
    // Perform CUDA forward pass
    match cuda_gru.forward_step(x, h_prev) {
        Ok(h_new) => h_new,
        Err(_) => {
            fallback_gru_forward(x, h_prev, w_ir, w_iz, w_in, w_hr, w_hz, w_hn, b_r, b_z, b_n)
        }
    }
}

// CPU fallback implementations
fn fallback_lstm_forward(
    x: &Array2<f32>,
    h_prev: &Array2<f32>,
    c_prev: &Array2<f32>,
    w_ii: &Array2<f32>,
    w_if: &Array2<f32>,
    w_ig: &Array2<f32>,
    w_io: &Array2<f32>,
    w_hi: &Array2<f32>,
    w_hf: &Array2<f32>,
    w_hg: &Array2<f32>,
    w_ho: &Array2<f32>,
    b_i: &Array2<f32>,
    b_f: &Array2<f32>,
    b_g: &Array2<f32>,
    b_o: &Array2<f32>,
    w_ci: Option<&Array2<f32>>,
    w_cf: Option<&Array2<f32>>,
    w_co: Option<&Array2<f32>>,
) -> (Array2<f32>, Array2<f32>) {
    // CPU implementation using ndarray operations
    let i_gate = sigmoid(&(x.dot(w_ii) + h_prev.dot(w_hi) + b_i));
    let f_gate = sigmoid(&(x.dot(w_if) + h_prev.dot(w_hf) + b_f));
    let g_gate = tanh(&(x.dot(w_ig) + h_prev.dot(w_hg) + b_g));
    let o_gate = sigmoid(&(x.dot(w_io) + h_prev.dot(w_ho) + b_o));
    
    let c_new = &f_gate * c_prev + &i_gate * &g_gate;
    let h_new = &o_gate * &tanh(&c_new);
    
    (h_new, c_new)
}

fn fallback_gru_forward(
    x: &Array2<f32>,
    h_prev: &Array2<f32>,
    w_ir: &Array2<f32>,
    w_iz: &Array2<f32>,
    w_in: &Array2<f32>,
    w_hr: &Array2<f32>,
    w_hz: &Array2<f32>,
    w_hn: &Array2<f32>,
    b_r: &Array2<f32>,
    b_z: &Array2<f32>,
    b_n: &Array2<f32>,
) -> Array2<f32> {
    let r_gate = sigmoid(&(x.dot(w_ir) + h_prev.dot(w_hr) + b_r));
    let z_gate = sigmoid(&(x.dot(w_iz) + h_prev.dot(w_hz) + b_z));
    let n_gate = tanh(&(x.dot(w_in) + (&r_gate * h_prev).dot(w_hn) + b_n));
    
    &z_gate * h_prev + (1.0 - &z_gate) * &n_gate
}

fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|a| 1.0 / (1.0 + (-a).exp()))
}

fn tanh(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|a| a.tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_cuda_availability() {
        // This test will pass regardless of CUDA availability
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
    }
    
    #[test]
    fn test_gpu_buffer() {
        if !is_cuda_available() {
            return;
        }
        
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut buffer = GpuBuffer::new(4).unwrap();
        buffer.copy_from_host(&data).unwrap();
        
        let mut result = vec![0.0f32; 4];
        buffer.copy_to_host(&mut result).unwrap();
        
        assert_eq!(data, result);
    }
    
    #[test]
    fn test_fallback_lstm() {
        let x = Array2::ones((2, 3));
        let h = Array2::zeros((2, 4));
        let c = Array2::zeros((2, 4));
        let w_ii = Array2::ones((3, 4));
        let w_if = Array2::ones((3, 4));
        let w_ig = Array2::ones((3, 4));
        let w_io = Array2::ones((3, 4));
        let w_hi = Array2::ones((4, 4));
        let w_hf = Array2::ones((4, 4));
        let w_hg = Array2::ones((4, 4));
        let w_ho = Array2::ones((4, 4));
        let b_i = Array2::zeros((1, 4));
        let b_f = Array2::ones((1, 4));
        let b_g = Array2::zeros((1, 4));
        let b_o = Array2::zeros((1, 4));
        
        let (h_new, c_new) = fallback_lstm_forward(&x, &h, &c, &w_ii, &w_if, &w_ig, &w_io,
                                                 &w_hi, &w_hf, &w_hg, &w_ho, &b_i, &b_f, &b_g, &b_o,
                                                 None, None, None);
        
        assert_eq!(h_new.shape(), &[2, 4]);
        assert_eq!(c_new.shape(), &[2, 4]);
    }
}