//! Unit tests for data-models crate
//! Tests validation, serialization, and core data structure behavior

use data_models::error::{DataModelError, Result};
use data_models::sequencing::FastqRecord;
use data_models::targets::{CrisprTarget, GuideRNA};
use data_models::expression::ExpressionProfile;
use data_models::metadata::JobMetadata;

#[cfg(test)]
mod fastq_tests {
    use super::*;

    #[test]
    fn test_fastq_record_validation() {
        // Valid FASTQ record
        let valid_record = FastqRecord::new(
            "read1".to_string(),
            "ACGTACGTACGT".to_string(),
            "IIIIIIIIIIII".to_string(),
        );
        assert!(valid_record.is_ok());

        // Invalid: sequence and quality length mismatch
        let invalid_record = FastqRecord::new(
            "read2".to_string(),
            "ACGT".to_string(),
            "II".to_string(),
        );
        assert!(invalid_record.is_err());
    }

    #[test]
    fn test_fastq_quality_scores() {
        let record = FastqRecord::new(
            "test".to_string(),
            "ACGT".to_string(),
            "ABCD".to_string(),
        ).unwrap();

        let phred_scores = record.phred_scores();
        assert_eq!(phred_scores.len(), 4);
        // ASCII 'A' = 65, Phred+33 encoding: 65-33 = 32
        assert_eq!(phred_scores[0], 32);
    }

    #[test]
    fn test_fastq_average_quality() {
        let record = FastqRecord::new(
            "test".to_string(),
            "ACGT".to_string(),
            "IIII".to_string(),
        ).unwrap();

        let avg_qual = record.average_quality();
        assert!(avg_qual > 35.0); // 'I' = ASCII 73, Phred = 40
    }

    #[test]
    fn test_fastq_gc_content() {
        let record = FastqRecord::new(
            "test".to_string(),
            "ACGT".to_string(),
            "IIII".to_string(),
        ).unwrap();

        let gc = record.gc_content();
        assert_eq!(gc, 50.0); // 2 out of 4 bases are G or C
    }
}

#[cfg(test)]
mod crispr_target_tests {
    use super::*;

    #[test]
    fn test_guide_rna_validation() {
        // Valid 23nt guide for Cas13
        let valid_guide = GuideRNA::new("ACGUACGUACGUACGUACGUACG".to_string());
        assert!(valid_guide.is_ok());

        // Invalid: too short
        let short_guide = GuideRNA::new("ACGU".to_string());
        assert!(short_guide.is_err());

        // Invalid: contains invalid characters
        let invalid_chars = GuideRNA::new("ACGUXACGUACGUACGUACGUACG".to_string());
        assert!(invalid_chars.is_err());
    }

    #[test]
    fn test_crispr_target_creation() {
        let guide = GuideRNA::new("ACGUACGUACGUACGUACGUACG".to_string()).unwrap();

        let target = CrisprTarget::builder()
            .guide_rna(guide.clone())
            .target_gene("IL6".to_string())
            .pam_sequence("H".to_string()) // Cas13 PFS
            .strand("+".to_string())
            .build();

        assert!(target.is_ok());
        let target = target.unwrap();
        assert_eq!(target.target_gene, "IL6");
        assert_eq!(target.guide_rna.sequence().len(), 23);
    }

    #[test]
    fn test_guide_rna_complement() {
        let guide = GuideRNA::new("ACGU".to_string()).unwrap();
        let complement = guide.complement();
        assert_eq!(complement, "UGCA");
    }

    #[test]
    fn test_guide_rna_gc_content() {
        let guide = GuideRNA::new("ACGUACGUACGUACGUACGUACG".to_string()).unwrap();
        let gc = guide.gc_content();
        assert!(gc >= 45.0 && gc <= 55.0); // Should be balanced
    }
}

#[cfg(test)]
mod expression_tests {
    use super::*;

    #[test]
    fn test_expression_profile_normalization() {
        let mut profile = ExpressionProfile::new("sample1".to_string());
        profile.add_gene_count("GENE1", 100);
        profile.add_gene_count("GENE2", 200);
        profile.add_gene_count("GENE3", 300);

        let normalized = profile.tpm_normalize();

        // TPM should sum to ~1,000,000
        let sum: f64 = normalized.values().sum();
        assert!((sum - 1_000_000.0).abs() < 100.0); // Allow small floating point error
    }

    #[test]
    fn test_expression_fold_change() {
        let mut control = ExpressionProfile::new("control".to_string());
        control.add_gene_count("GENE1", 100);

        let mut treated = ExpressionProfile::new("treated".to_string());
        treated.add_gene_count("GENE1", 400);

        let fc = ExpressionProfile::fold_change(&control, &treated, "GENE1");
        assert_eq!(fc, 4.0); // 400/100 = 4x
    }

    #[test]
    fn test_expression_log2_fold_change() {
        let mut control = ExpressionProfile::new("control".to_string());
        control.add_gene_count("GENE1", 100);

        let mut treated = ExpressionProfile::new("treated".to_string());
        treated.add_gene_count("GENE1", 400);

        let log2fc = ExpressionProfile::log2_fold_change(&control, &treated, "GENE1");
        assert!((log2fc - 2.0).abs() < 0.01); // log2(4) = 2
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_job_metadata_creation() {
        let metadata = JobMetadata::new(
            "Test Job".to_string(),
            "user123".to_string(),
        );

        assert!(metadata.job_id.is_some());
        assert_eq!(metadata.status, "pending");
        assert!(metadata.created_at.is_some());
    }

    #[test]
    fn test_job_status_transitions() {
        let mut metadata = JobMetadata::new(
            "Test".to_string(),
            "user".to_string(),
        );

        metadata.set_status("running");
        assert_eq!(metadata.status, "running");
        assert!(metadata.started_at.is_some());

        metadata.set_status("completed");
        assert_eq!(metadata.status, "completed");
        assert!(metadata.completed_at.is_some());
    }

    #[test]
    fn test_job_duration_calculation() {
        let mut metadata = JobMetadata::new("Test".to_string(), "user".to_string());

        metadata.set_status("running");
        std::thread::sleep(std::time::Duration::from_millis(100));
        metadata.set_status("completed");

        let duration = metadata.duration_ms();
        assert!(duration >= 100);
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_dna_sequence_validation() {
        assert!(validate_dna_sequence("ACGT"));
        assert!(validate_dna_sequence("ACGTN")); // N allowed
        assert!(!validate_dna_sequence("ACGTX")); // X invalid
        assert!(!validate_dna_sequence("acgt")); // lowercase invalid
    }

    #[test]
    fn test_rna_sequence_validation() {
        assert!(validate_rna_sequence("ACGU"));
        assert!(validate_rna_sequence("ACGUN")); // N allowed
        assert!(!validate_rna_sequence("ACGT")); // T not allowed in RNA
        assert!(!validate_rna_sequence("acgu")); // lowercase invalid
    }

    #[test]
    fn test_quality_string_validation() {
        assert!(validate_quality_string("IIIII"));
        assert!(validate_quality_string("!!!!!")); // Minimum quality
        assert!(validate_quality_string("JJJJJ")); // High quality
        assert!(!validate_quality_string("12345")); // Invalid chars
    }

    fn validate_dna_sequence(seq: &str) -> bool {
        seq.chars().all(|c| matches!(c, 'A' | 'C' | 'G' | 'T' | 'N'))
    }

    fn validate_rna_sequence(seq: &str) -> bool {
        seq.chars().all(|c| matches!(c, 'A' | 'C' | 'G' | 'U' | 'N'))
    }

    fn validate_quality_string(qual: &str) -> bool {
        qual.chars().all(|c| c as u8 >= 33 && c as u8 <= 126)
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_fastq_record_serialization() {
        let record = FastqRecord::new(
            "read1".to_string(),
            "ACGT".to_string(),
            "IIII".to_string(),
        ).unwrap();

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: FastqRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record.id(), deserialized.id());
        assert_eq!(record.sequence(), deserialized.sequence());
    }

    #[test]
    fn test_expression_profile_serialization() {
        let mut profile = ExpressionProfile::new("sample1".to_string());
        profile.add_gene_count("GENE1", 100);

        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: ExpressionProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(profile.sample_id(), deserialized.sample_id());
    }
}
