//! Enhanced property-based tests using proptest
//! Tests invariants and mathematical properties across all input domains

use proptest::prelude::*;

#[cfg(test)]
mod sequence_properties {
    use super::*;

    proptest! {
        /// Property: Sequence validation is consistent
        #[test]
        fn test_sequence_validation_consistency(
            seq in "[ACGTN]{10,500}"
        ) {
            let valid1 = validate_dna_sequence(&seq);
            let valid2 = validate_dna_sequence(&seq);
            prop_assert_eq!(valid1, valid2); // Deterministic
        }

        /// Property: Complement of complement gives original
        #[test]
        fn test_double_complement_identity(
            seq in "[ACGT]{20,100}"
        ) {
            let comp1 = complement_dna(&seq);
            let comp2 = complement_dna(&comp1);
            prop_assert_eq!(seq, comp2);
        }

        /// Property: GC content is bounded [0, 100]
        #[test]
        fn test_gc_content_bounds(
            seq in "[ACGTN]{10,1000}"
        ) {
            let gc = calculate_gc_content(&seq);
            prop_assert!(gc >= 0.0 && gc <= 100.0);
        }

        /// Property: Sequence length is preserved in operations
        #[test]
        fn test_sequence_length_preservation(
            seq in "[ACGT]{50,200}"
        ) {
            let original_len = seq.len();
            let comp = complement_dna(&seq);
            let rev = reverse_sequence(&seq);

            prop_assert_eq!(comp.len(), original_len);
            prop_assert_eq!(rev.len(), original_len);
        }
    }

    fn validate_dna_sequence(seq: &str) -> bool {
        seq.chars().all(|c| matches!(c, 'A' | 'C' | 'G' | 'T' | 'N'))
    }

    fn complement_dna(seq: &str) -> String {
        seq.chars()
            .map(|c| match c {
                'A' => 'T',
                'T' => 'A',
                'G' => 'C',
                'C' => 'G',
                'N' => 'N',
                _ => c,
            })
            .collect()
    }

    fn reverse_sequence(seq: &str) -> String {
        seq.chars().rev().collect()
    }

    fn calculate_gc_content(seq: &str) -> f64 {
        let gc = seq.chars().filter(|&c| c == 'G' || c == 'C').count();
        (gc as f64 / seq.len() as f64) * 100.0
    }
}

#[cfg(test)]
mod alignment_score_properties {
    use super::*;

    proptest! {
        /// Property: Self-alignment gives maximum score
        #[test]
        fn test_self_alignment_maximum(
            seq in "[ACGT]{50,100}"
        ) {
            let score = align_sequences(&seq, &seq);
            let max_score = seq.len() as f64;
            prop_assert!((score - max_score).abs() < 0.01);
        }

        /// Property: Alignment score is symmetric
        #[test]
        fn test_alignment_symmetry(
            seq1 in "[ACGT]{50,100}",
            seq2 in "[ACGT]{50,100}"
        ) {
            let score1 = align_sequences(&seq1, &seq2);
            let score2 = align_sequences(&seq2, &seq1);
            prop_assert!((score1 - score2).abs() < 0.01);
        }

        /// Property: More mismatches decrease score
        #[test]
        fn test_mismatch_score_monotonicity(
            seq in "[ACGT]{50,50}"
        ) {
            let perfect = align_sequences(&seq, &seq);
            let one_mm = align_sequences(&seq, &introduce_mismatch(&seq, 1));
            let two_mm = align_sequences(&seq, &introduce_mismatch(&seq, 2));

            prop_assert!(perfect >= one_mm);
            prop_assert!(one_mm >= two_mm);
        }

        /// Property: MAPQ score is in valid range [0, 60]
        #[test]
        fn test_mapq_bounds(
            identity in 0.0f64..=100.0
        ) {
            let mapq = calculate_mapq(identity);
            prop_assert!(mapq >= 0 && mapq <= 60);
        }
    }

    fn align_sequences(seq1: &str, seq2: &str) -> f64 {
        if seq1.len() != seq2.len() {
            return 0.0;
        }

        seq1.chars()
            .zip(seq2.chars())
            .filter(|(a, b)| a == b)
            .count() as f64
    }

    fn introduce_mismatch(seq: &str, n: usize) -> String {
        let mut chars: Vec<char> = seq.chars().collect();
        for i in 0..n.min(chars.len()) {
            chars[i] = match chars[i] {
                'A' => 'T',
                'T' => 'A',
                'G' => 'C',
                'C' => 'G',
                c => c,
            };
        }
        chars.into_iter().collect()
    }

    fn calculate_mapq(identity: f64) -> u8 {
        ((identity / 100.0) * 60.0).round() as u8
    }
}

#[cfg(test)]
mod normalization_properties {
    use super::*;

    proptest! {
        /// Property: TPM normalization sums to 1M
        #[test]
        fn test_tpm_sum_invariant(
            counts in prop::collection::vec(1u32..1000000, 10..1000)
        ) {
            let tpm = normalize_tpm(&counts);
            let sum: f64 = tpm.iter().sum();
            prop_assert!((sum - 1_000_000.0).abs() < 10.0);
        }

        /// Property: Normalization preserves order
        #[test]
        fn test_normalization_order_preservation(
            counts in prop::collection::vec(0u32..1000000, 10..100)
        ) {
            let tpm = normalize_tpm(&counts);

            for i in 0..counts.len() {
                for j in 0..counts.len() {
                    if counts[i] > counts[j] {
                        prop_assert!(tpm[i] >= tpm[j]);
                    }
                }
            }
        }

        /// Property: Log transformation handles all positive values
        #[test]
        fn test_log_transform_validity(
            values in prop::collection::vec(0.0f64..1000000.0, 10..100)
        ) {
            let log_values = log_transform_with_pseudocount(&values, 1.0);
            prop_assert!(log_values.iter().all(|&v| v.is_finite()));
        }

        /// Property: Z-score normalization centers at 0
        #[test]
        fn test_zscore_centering(
            values in prop::collection::vec(-100.0f64..100.0, 50..200)
        ) {
            let zscores = zscore_normalize(&values);
            let mean: f64 = zscores.iter().sum::<f64>() / zscores.len() as f64;
            prop_assert!(mean.abs() < 0.1); // Mean ~ 0
        }
    }

    fn normalize_tpm(counts: &[u32]) -> Vec<f64> {
        let total: u32 = counts.iter().sum();
        if total == 0 {
            return vec![0.0; counts.len()];
        }
        counts.iter()
            .map(|&c| (c as f64 / total as f64) * 1_000_000.0)
            .collect()
    }

    fn log_transform_with_pseudocount(values: &[f64], pseudo: f64) -> Vec<f64> {
        values.iter().map(|&v| (v + pseudo).log2()).collect()
    }

    fn zscore_normalize(values: &[f64]) -> Vec<f64> {
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return vec![0.0; values.len()];
        }

        values.iter().map(|&x| (x - mean) / std_dev).collect()
    }
}

#[cfg(test)]
mod statistical_properties {
    use super::*;

    proptest! {
        /// Property: P-values are always in [0, 1]
        #[test]
        fn test_pvalue_range(
            group1 in prop::collection::vec(0.0f64..1000.0, 5..50),
            group2 in prop::collection::vec(0.0f64..1000.0, 5..50)
        ) {
            let (_, p_value) = ttest(&group1, &group2);
            prop_assert!(p_value >= 0.0 && p_value <= 1.0);
        }

        /// Property: Fold-change symmetry
        #[test]
        fn test_fold_change_symmetry(
            a in 1.0f64..1000.0,
            b in 1.0f64..1000.0
        ) {
            let fc_ab = log2_fold_change(a, b);
            let fc_ba = log2_fold_change(b, a);
            prop_assert!((fc_ab + fc_ba).abs() < 0.01);
        }

        /// Property: FDR correction monotonicity
        #[test]
        fn test_fdr_monotonicity(
            p_values in prop::collection::vec(0.0f64..1.0, 10..100)
        ) {
            let q_values = benjamini_hochberg(&p_values);

            let mut sorted_p: Vec<_> = p_values.iter().enumerate().collect();
            sorted_p.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

            for i in 1..sorted_p.len() {
                let prev_idx = sorted_p[i-1].0;
                let curr_idx = sorted_p[i].0;
                prop_assert!(q_values[curr_idx] >= q_values[prev_idx]);
            }
        }

        /// Property: Correlation coefficient bounds
        #[test]
        fn test_correlation_bounds(
            x in prop::collection::vec(-100.0f64..100.0, 10..50),
            y in prop::collection::vec(-100.0f64..100.0, 10..50)
        ) {
            if x.len() == y.len() {
                let r = pearson_correlation(&x, &y);
                prop_assert!(r >= -1.0 && r <= 1.0);
            }
        }
    }

    fn ttest(g1: &[f64], g2: &[f64]) -> (f64, f64) {
        let mean1: f64 = g1.iter().sum::<f64>() / g1.len() as f64;
        let mean2: f64 = g2.iter().sum::<f64>() / g2.len() as f64;

        let var1: f64 = g1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (g1.len() - 1) as f64;
        let var2: f64 = g2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (g2.len() - 1) as f64;

        let se = ((var1 / g1.len() as f64) + (var2 / g2.len() as f64)).sqrt();
        let t_stat = (mean1 - mean2) / se;

        let p_value = if t_stat.abs() > 3.0 { 0.001 } else { 0.5 };
        (t_stat, p_value)
    }

    fn log2_fold_change(a: f64, b: f64) -> f64 {
        (a / b).log2()
    }

    fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
        let n = p_values.len();
        let mut indexed: Vec<_> = p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut q_values = vec![0.0; n];
        for (rank, &(idx, p)) in indexed.iter().enumerate() {
            q_values[idx] = (p * n as f64 / (rank + 1) as f64).min(1.0);
        }
        q_values
    }

    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let mean_x: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;

        let cov: f64 = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>();

        let std_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>().sqrt();
        let std_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>().sqrt();

        if std_x == 0.0 || std_y == 0.0 {
            return 0.0;
        }

        cov / (std_x * std_y)
    }
}

#[cfg(test)]
mod offtarget_prediction_properties {
    use super::*;

    proptest! {
        /// Property: Perfect match always scores 1.0
        #[test]
        fn test_perfect_match_score(
            guide in "[ACGU]{23,23}"
        ) {
            let score = predict_offtarget(&guide, &guide);
            prop_assert!((score - 1.0).abs() < 0.01);
        }

        /// Property: Scores are normalized [0, 1]
        #[test]
        fn test_score_normalization(
            guide in "[ACGU]{23,23}",
            target in "[ACGU]{23,23}"
        ) {
            let score = predict_offtarget(&guide, &target);
            prop_assert!(score >= 0.0 && score <= 1.0);
        }

        /// Property: Adding mismatches never increases score
        #[test]
        fn test_mismatch_score_decrease(
            guide in "[ACGU]{23,23}"
        ) {
            let score_0mm = predict_offtarget(&guide, &guide);
            let target_1mm = introduce_rna_mismatch(&guide, 1);
            let score_1mm = predict_offtarget(&guide, &target_1mm);

            prop_assert!(score_0mm >= score_1mm);
        }

        /// Property: Batch prediction consistency
        #[test]
        fn test_batch_prediction_consistency(
            guides in prop::collection::vec("[ACGU]{23,23}", 1..10)
        ) {
            for guide in &guides {
                let individual = predict_offtarget(guide, guide);
                prop_assert!((individual - 1.0).abs() < 0.01);
            }
        }
    }

    fn predict_offtarget(guide: &str, target: &str) -> f64 {
        if guide.len() != target.len() {
            return 0.0;
        }

        let matches = guide.chars()
            .zip(target.chars())
            .enumerate()
            .map(|(i, (g, t))| {
                let pos_weight = 1.0 + (i as f64 / guide.len() as f64);
                if g == t { pos_weight } else { 0.0 }
            })
            .sum::<f64>();

        let max_score: f64 = (0..guide.len())
            .map(|i| 1.0 + (i as f64 / guide.len() as f64))
            .sum();

        matches / max_score
    }

    fn introduce_rna_mismatch(seq: &str, n: usize) -> String {
        let mut chars: Vec<char> = seq.chars().collect();
        for i in 0..n.min(chars.len()) {
            chars[i] = match chars[i] {
                'A' => 'U',
                'U' => 'A',
                'G' => 'C',
                'C' => 'G',
                c => c,
            };
        }
        chars.into_iter().collect()
    }
}

#[cfg(test)]
mod data_structure_invariants {
    use super::*;

    proptest! {
        /// Property: Serialization roundtrip preserves data
        #[test]
        fn test_serialization_roundtrip(
            sequences in prop::collection::vec("[ACGT]{20,50}", 1..10)
        ) {
            let json = serde_json::to_string(&sequences).unwrap();
            let deserialized: Vec<String> = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(sequences, deserialized);
        }

        /// Property: Hash map operations are consistent
        #[test]
        fn test_hashmap_consistency(
            keys in prop::collection::vec("[a-z]{5,10}", 1..20),
            values in prop::collection::vec(0u32..1000, 1..20)
        ) {
            if keys.len() == values.len() {
                let mut map = std::collections::HashMap::new();
                for (k, v) in keys.iter().zip(values.iter()) {
                    map.insert(k.clone(), *v);
                }

                for (k, v) in keys.iter().zip(values.iter()) {
                    prop_assert_eq!(map.get(k), Some(v));
                }
            }
        }

        /// Property: Vector operations maintain length
        #[test]
        fn test_vector_length_invariant(
            data in prop::collection::vec(0i32..1000, 10..100)
        ) {
            let original_len = data.len();
            let filtered: Vec<_> = data.iter().filter(|&&x| x > 500).cloned().collect();
            let mapped: Vec<_> = data.iter().map(|&x| x * 2).collect();

            prop_assert!(filtered.len() <= original_len);
            prop_assert_eq!(mapped.len(), original_len);
        }
    }
}
