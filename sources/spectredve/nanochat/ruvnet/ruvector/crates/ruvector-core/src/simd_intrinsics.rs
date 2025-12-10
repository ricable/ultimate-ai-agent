//! Custom SIMD intrinsics for performance-critical operations
//!
//! This module provides hand-optimized SIMD implementations using AVX2/AVX-512
//! for distance calculations and other vectorized operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized euclidean distance using AVX2
/// Falls back to scalar implementation if AVX2 is not available
#[inline]
pub fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { euclidean_distance_avx2_impl(a, b) }
        } else {
            euclidean_distance_scalar(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        euclidean_distance_scalar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    // SECURITY: Ensure both arrays have the same length to prevent out-of-bounds access
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    // Process 8 floats at a time
    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;

        // Load 8 floats from each array
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));

        // Compute difference: (a - b)
        let diff = _mm256_sub_ps(va, vb);

        // Square the difference: (a - b)^2
        let sq = _mm256_mul_ps(diff, diff);

        // Accumulate
        sum = _mm256_add_ps(sum, sq);
    }

    // Horizontal sum of the 8 floats in the AVX register
    let sum_arr: [f32; 8] = std::mem::transmute(sum);
    let mut total = sum_arr.iter().sum::<f32>();

    // Handle remaining elements (if len not divisible by 8)
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}

/// SIMD-optimized dot product using AVX2
#[inline]
pub fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { dot_product_avx2_impl(a, b) }
        } else {
            dot_product_scalar(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_product_scalar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    // SECURITY: Ensure both arrays have the same length to prevent out-of-bounds access
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }

    let sum_arr: [f32; 8] = std::mem::transmute(sum);
    let mut total = sum_arr.iter().sum::<f32>();

    for i in (chunks * 8)..len {
        total += a[i] * b[i];
    }

    total
}

/// SIMD-optimized cosine similarity using AVX2
#[inline]
pub fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { cosine_similarity_avx2_impl(a, b) }
        } else {
            cosine_similarity_scalar(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        cosine_similarity_scalar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cosine_similarity_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    // SECURITY: Ensure both arrays have the same length to prevent out-of-bounds access
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));

        // Dot product
        dot = _mm256_add_ps(dot, _mm256_mul_ps(va, vb));

        // Norms
        norm_a = _mm256_add_ps(norm_a, _mm256_mul_ps(va, va));
        norm_b = _mm256_add_ps(norm_b, _mm256_mul_ps(vb, vb));
    }

    let dot_arr: [f32; 8] = std::mem::transmute(dot);
    let norm_a_arr: [f32; 8] = std::mem::transmute(norm_a);
    let norm_b_arr: [f32; 8] = std::mem::transmute(norm_b);

    let mut dot_sum = dot_arr.iter().sum::<f32>();
    let mut norm_a_sum = norm_a_arr.iter().sum::<f32>();
    let mut norm_b_sum = norm_b_arr.iter().sum::<f32>();

    for i in (chunks * 8)..len {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }

    dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt())
}

// Scalar fallback implementations

fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance_avx2() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = euclidean_distance_avx2(&a, &b);
        let expected = euclidean_distance_scalar(&a, &b);

        assert!(
            (result - expected).abs() < 0.001,
            "AVX2 result {} differs from scalar result {}",
            result,
            expected
        );
    }

    #[test]
    fn test_dot_product_avx2() {
        let a = vec![1.0; 16];
        let b = vec![2.0; 16];

        let result = dot_product_avx2(&a, &b);
        assert!((result - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_avx2() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let result = cosine_similarity_avx2(&a, &b);
        assert!((result - 1.0).abs() < 0.001);
    }
}
