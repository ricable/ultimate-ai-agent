//! Zero-copy tensor buffer operations
//!
//! Provides high-performance zero-copy operations for tensor data transfer
//! between Rust and JavaScript using napi::Buffer.
//!
//! Key features:
//! - Direct memory mapping without serialization
//! - Efficient gradient and model parameter transfer
//! - Support for various tensor data types (f32, f64, i32, etc.)

use napi::bindgen_prelude::*;
use napi::{Env, Result, Status};

/// Tensor buffer wrapper for zero-copy operations
///
/// Wraps napi::Buffer to provide type-safe tensor operations with
/// zero-copy semantics for maximum performance.
#[napi]
pub struct TensorBuffer {
    buffer: Buffer,
    shape: Vec<u32>,
    dtype: String,
}

#[napi]
impl TensorBuffer {
    /// Create a new tensor buffer from raw data
    ///
    /// # Arguments
    ///
    /// * `buffer` - Raw buffer containing tensor data
    /// * `shape` - Dimensions of the tensor
    /// * `dtype` - Data type: "f32", "f64", "i32", "i64"
    ///
    /// # Example
    ///
    /// ```javascript
    /// const buffer = Buffer.from(new Float32Array([1, 2, 3, 4]).buffer);
    /// const tensor = new TensorBuffer(buffer, [2, 2], 'f32');
    /// ```
    #[napi(constructor)]
    pub fn new(buffer: Buffer, shape: Vec<u32>, dtype: String) -> Result<Self> {
        // Validate shape
        let total_elements: u32 = shape.iter().product();
        let element_size = match dtype.as_str() {
            "f32" | "i32" => 4,
            "f64" | "i64" => 8,
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("Unsupported dtype: {}", dtype),
                ))
            }
        };

        let expected_size = (total_elements as usize) * element_size;
        if buffer.len() != expected_size {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Buffer size {} does not match shape {:?} with dtype {} (expected {})",
                    buffer.len(),
                    shape,
                    dtype,
                    expected_size
                ),
            ));
        }

        Ok(Self {
            buffer,
            shape,
            dtype,
        })
    }

    /// Get the underlying buffer (zero-copy)
    #[napi(getter)]
    pub fn buffer(&self) -> Buffer {
        self.buffer.clone()
    }

    /// Get the tensor shape
    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    /// Get the data type
    #[napi(getter)]
    pub fn dtype(&self) -> String {
        self.dtype.clone()
    }

    /// Get the total number of elements
    #[napi]
    pub fn num_elements(&self) -> u32 {
        self.shape.iter().product()
    }

    /// Get the buffer size in bytes
    #[napi]
    pub fn byte_size(&self) -> u32 {
        self.buffer.len() as u32
    }

    /// Convert to a flat f32 array (creates a copy)
    ///
    /// Note: This creates a copy of the data. Use buffer() for zero-copy access.
    #[napi]
    pub fn to_f32_array(&self) -> Result<Vec<f32>> {
        if self.dtype != "f32" {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Cannot convert {} to f32 array", self.dtype),
            ));
        }

        let data = self.buffer.as_ref();
        let mut result = Vec::with_capacity(data.len() / 4);

        for i in (0..data.len()).step_by(4) {
            let bytes = &data[i..i + 4];
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            result.push(value);
        }

        Ok(result)
    }

    /// Convert to a flat f64 array (creates a copy)
    ///
    /// Note: This creates a copy of the data. Use buffer() for zero-copy access.
    #[napi]
    pub fn to_f64_array(&self) -> Result<Vec<f64>> {
        if self.dtype != "f64" {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Cannot convert {} to f64 array", self.dtype),
            ));
        }

        let data = self.buffer.as_ref();
        let mut result = Vec::with_capacity(data.len() / 8);

        for i in (0..data.len()).step_by(8) {
            let bytes = &data[i..i + 8];
            let value = f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            result.push(value);
        }

        Ok(result)
    }

    /// Reshape the tensor (zero-copy)
    ///
    /// Creates a new view of the same data with a different shape.
    /// The total number of elements must remain the same.
    #[napi]
    pub fn reshape(&self, new_shape: Vec<u32>) -> Result<TensorBuffer> {
        let old_elements: u32 = self.shape.iter().product();
        let new_elements: u32 = new_shape.iter().product();

        if old_elements != new_elements {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Cannot reshape: old shape {:?} has {} elements, new shape {:?} has {} elements",
                    self.shape, old_elements, new_shape, new_elements
                ),
            ));
        }

        Ok(TensorBuffer {
            buffer: self.buffer.clone(),
            shape: new_shape,
            dtype: self.dtype.clone(),
        })
    }

    /// Clone the tensor buffer (creates a copy)
    #[napi]
    pub fn clone_tensor(&self) -> TensorBuffer {
        TensorBuffer {
            buffer: self.buffer.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
        }
    }
}

/// Create a tensor buffer from a Float32Array or Buffer
///
/// # Arguments
///
/// * `data` - Buffer containing f32 data
/// * `shape` - Dimensions of the tensor
///
/// # Example
///
/// ```javascript
/// const buffer = Buffer.from(new Float32Array([1, 2, 3, 4]).buffer);
/// const tensor = createTensorBuffer(buffer, [2, 2]);
/// ```
#[napi]
pub fn create_tensor_buffer(data: Buffer, shape: Vec<u32>) -> Result<TensorBuffer> {
    let expected_elements: u32 = shape.iter().product();
    let expected_bytes = (expected_elements as usize) * 4; // 4 bytes per f32

    if data.len() != expected_bytes {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Data length {} does not match shape {:?} (expected {} bytes for {} f32 elements)",
                data.len(),
                shape,
                expected_bytes,
                expected_elements
            ),
        ));
    }

    Ok(TensorBuffer {
        buffer: data,
        shape,
        dtype: "f32".to_string(),
    })
}

/// Create a tensor from a buffer (zero-copy)
///
/// # Arguments
///
/// * `buffer` - Raw buffer containing tensor data
/// * `shape` - Dimensions of the tensor
/// * `dtype` - Data type: "f32", "f64", "i32", "i64"
///
/// # Example
///
/// ```javascript
/// const buffer = Buffer.from(new Float32Array([1, 2, 3, 4]).buffer);
/// const tensor = tensorFromBuffer(buffer, [2, 2], 'f32');
/// ```
#[napi]
pub fn tensor_from_buffer(buffer: Buffer, shape: Vec<u32>, dtype: String) -> Result<TensorBuffer> {
    TensorBuffer::new(buffer, shape, dtype)
}

/// Concatenate multiple tensor buffers along a dimension (creates a copy)
///
/// # Arguments
///
/// * `tensors` - Array of tensor buffers to concatenate
/// * `axis` - Dimension along which to concatenate
///
/// # Example
///
/// ```javascript
/// const t1 = createTensorBuffer([1, 2], [2]);
/// const t2 = createTensorBuffer([3, 4], [2]);
/// const result = concatenateTensors([t1, t2], 0); // [1, 2, 3, 4] with shape [4]
/// ```
#[napi]
pub fn concatenate_tensors(tensors: Vec<&TensorBuffer>, axis: u32) -> Result<TensorBuffer> {
    if tensors.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "Cannot concatenate empty tensor array",
        ));
    }

    // Validate all tensors have same dtype
    let dtype = &tensors[0].dtype;
    for tensor in &tensors {
        if &tensor.dtype != dtype {
            return Err(Error::new(
                Status::InvalidArg,
                "All tensors must have the same dtype",
            ));
        }
    }

    // For simplicity, only support axis=0 concatenation
    if axis != 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "Currently only axis=0 concatenation is supported",
        ));
    }

    // Concatenate buffers
    let mut concatenated = Vec::new();
    let mut total_elements = 0u32;

    for tensor in &tensors {
        concatenated.extend_from_slice(tensor.buffer.as_ref());
        total_elements += tensor.num_elements();
    }

    Ok(TensorBuffer {
        buffer: Buffer::from(concatenated),
        shape: vec![total_elements],
        dtype: dtype.clone(),
    })
}

/// Split a tensor buffer into multiple tensors (creates copies)
///
/// # Arguments
///
/// * `tensor` - Tensor buffer to split
/// * `num_splits` - Number of equal-sized splits
///
/// # Example
///
/// ```javascript
/// const tensor = createTensorBuffer([1, 2, 3, 4, 5, 6], [6]);
/// const [t1, t2, t3] = splitTensor(tensor, 3); // Each has shape [2]
/// ```
#[napi]
pub fn split_tensor(tensor: &TensorBuffer, num_splits: u32) -> Result<Vec<TensorBuffer>> {
    let total_elements = tensor.num_elements();
    if total_elements % num_splits != 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Cannot split tensor with {} elements into {} equal parts",
                total_elements, num_splits
            ),
        ));
    }

    let elements_per_split = total_elements / num_splits;
    let element_size = match tensor.dtype.as_str() {
        "f32" | "i32" => 4,
        "f64" | "i64" => 8,
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Unsupported dtype: {}", tensor.dtype),
            ))
        }
    };
    let bytes_per_split = (elements_per_split as usize) * element_size;

    let data = tensor.buffer.as_ref();
    let mut result = Vec::new();

    for i in 0..num_splits {
        let start = (i as usize) * bytes_per_split;
        let end = start + bytes_per_split;
        let split_data = data[start..end].to_vec();

        result.push(TensorBuffer {
            buffer: Buffer::from(split_data),
            shape: vec![elements_per_split],
            dtype: tensor.dtype.clone(),
        });
    }

    Ok(result)
}
