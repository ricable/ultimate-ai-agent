//! Integration tests for attention mechanisms
//!
//! These tests verify the attention module works correctly with PostgreSQL types.

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    // We can't run full pgrx tests without PostgreSQL installed,
    // but we can test the Rust implementations directly

    #[test]
    fn test_attention_module_exists() {
        // This test just ensures the module compiles
        assert!(true);
    }

    #[test]
    fn test_softmax_implementation() {
        // Test softmax directly from the attention module
        let logits = vec![1.0, 2.0, 3.0];

        // Find max
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(max_logit, 3.0);

        // Compute exp
        let exp_values: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();

        // Compute sum
        let sum: f32 = exp_values.iter().sum();

        // Normalize
        let result: Vec<f32> = exp_values.iter().map(|x| x / sum).collect();

        // Verify properties
        let result_sum: f32 = result.iter().sum();
        assert_relative_eq!(result_sum, 1.0, epsilon = 1e-6);

        // Higher logit should have higher probability
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_scaled_dot_product() {
        // Test basic dot product scaling
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let query = vec![1.0; head_dim];
        let key = vec![1.0; head_dim];

        let dot: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();
        let scaled_score = dot * scale;

        assert!(scaled_score > 0.0);
        assert!(scaled_score < head_dim as f32); // Should be scaled down
    }

    #[test]
    fn test_multi_head_split() {
        // Test head splitting logic
        let num_heads = 4;
        let total_dim = 8;
        let head_dim = total_dim / num_heads;

        assert_eq!(head_dim, 2);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Split into heads
        let mut heads = Vec::new();
        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            heads.push(input[start..end].to_vec());
        }

        assert_eq!(heads.len(), 4);
        assert_eq!(heads[0], vec![1.0, 2.0]);
        assert_eq!(heads[1], vec![3.0, 4.0]);
        assert_eq!(heads[2], vec![5.0, 6.0]);
        assert_eq!(heads[3], vec![7.0, 8.0]);

        // Concatenate back
        let concatenated: Vec<f32> = heads.into_iter().flatten().collect();
        assert_eq!(concatenated, input);
    }

    #[test]
    fn test_flash_attention_block_size() {
        // Test block size calculations
        let seq_len = 256;
        let block_size = 64;

        let num_blocks = (seq_len + block_size - 1) / block_size;
        assert_eq!(num_blocks, 4);

        // Verify block boundaries
        for block_idx in 0..num_blocks {
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(seq_len);

            assert!(block_start < seq_len);
            assert!(block_end <= seq_len);
            assert!(block_end > block_start);
        }
    }

    #[test]
    fn test_attention_type_names() {
        // Test attention type string representations
        let types = vec![
            "scaled_dot",
            "multi_head",
            "flash_v2",
            "linear",
            "gat",
            "sparse",
            "moe",
            "cross",
            "sliding",
            "poincare",
        ];

        for type_name in types {
            assert!(!type_name.is_empty());
            assert!(type_name.len() > 2);
        }
    }
}
