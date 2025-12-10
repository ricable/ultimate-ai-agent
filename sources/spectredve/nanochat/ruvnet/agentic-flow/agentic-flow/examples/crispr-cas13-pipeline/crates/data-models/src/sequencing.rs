//! Sequencing data structures for FASTQ and BAM formats

use crate::error::{DataModelError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Represents a single sequencing read from FASTQ format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FastqRecord {
    /// Unique identifier for this read
    pub id: String,
    /// DNA/RNA sequence
    pub sequence: String,
    /// Quality scores (Phred+33 encoded)
    pub quality: Vec<u8>,
    /// Optional description
    pub description: Option<String>,
}

impl FastqRecord {
    /// Create a new FASTQ record
    pub fn new(id: String, sequence: String, quality: Vec<u8>) -> Result<Self> {
        if sequence.len() != quality.len() {
            return Err(DataModelError::InvalidSequence(
                "Sequence and quality length mismatch".to_string(),
            ));
        }

        if !Self::is_valid_sequence(&sequence) {
            return Err(DataModelError::InvalidSequence(format!(
                "Invalid nucleotides in sequence: {}",
                sequence
            )));
        }

        Ok(Self {
            id,
            sequence,
            quality,
            description: None,
        })
    }

    /// Validate nucleotide sequence (DNA/RNA)
    fn is_valid_sequence(seq: &str) -> bool {
        seq.chars()
            .all(|c| matches!(c.to_ascii_uppercase(), 'A' | 'C' | 'G' | 'T' | 'U' | 'N'))
    }

    /// Get average quality score
    pub fn average_quality(&self) -> f64 {
        if self.quality.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.quality.iter().map(|&q| (q - 33) as u64).sum();
        sum as f64 / self.quality.len() as f64
    }

    /// Get GC content percentage
    pub fn gc_content(&self) -> f64 {
        let gc_count = self
            .sequence
            .chars()
            .filter(|&c| matches!(c.to_ascii_uppercase(), 'G' | 'C'))
            .count();
        (gc_count as f64 / self.sequence.len() as f64) * 100.0
    }
}

/// Genomic coordinates
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenomicCoordinate {
    /// Chromosome/contig name index
    pub reference_id: u32,
    /// 0-based start position
    pub start: u64,
    /// 0-based end position (exclusive)
    pub end: u64,
    /// Strand (true = forward, false = reverse)
    pub strand: bool,
}

impl GenomicCoordinate {
    /// Create new genomic coordinate
    pub fn new(reference_id: u32, start: u64, end: u64, strand: bool) -> Result<Self> {
        if start >= end {
            return Err(DataModelError::InvalidRange { start, end });
        }
        Ok(Self {
            reference_id,
            start,
            end,
            strand,
        })
    }

    /// Get the length of this region
    pub fn length(&self) -> u64 {
        self.end - self.start
    }

    /// Check if two coordinates overlap
    pub fn overlaps(&self, other: &Self) -> bool {
        self.reference_id == other.reference_id && self.start < other.end && other.start < self.end
    }
}

/// Represents an aligned read from BAM format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedRead {
    /// Unique read identifier
    pub id: Uuid,
    /// Read name from sequencer
    pub qname: String,
    /// Genomic position
    pub position: GenomicCoordinate,
    /// Mapping quality (0-255)
    pub mapq: u8,
    /// CIGAR string
    pub cigar: String,
    /// Read sequence
    pub sequence: String,
    /// Quality scores
    pub quality: Vec<u8>,
    /// SAM flags
    pub flags: u16,
    /// Number of mismatches
    pub nm: Option<u32>,
}

impl AlignedRead {
    /// Check if read is mapped
    pub fn is_mapped(&self) -> bool {
        (self.flags & 0x4) == 0
    }

    /// Check if read is reverse complemented
    pub fn is_reverse(&self) -> bool {
        (self.flags & 0x10) != 0
    }

    /// Check if read is primary alignment
    pub fn is_primary(&self) -> bool {
        (self.flags & 0x100) == 0
    }

    /// Check if read passes quality filters
    pub fn is_qc_pass(&self) -> bool {
        (self.flags & 0x200) == 0
    }

    /// Check if read is PCR or optical duplicate
    pub fn is_duplicate(&self) -> bool {
        (self.flags & 0x400) != 0
    }
}

/// Alignment statistics for a read or set of reads
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlignmentStats {
    pub total_reads: u64,
    pub mapped_reads: u64,
    pub unmapped_reads: u64,
    pub duplicates: u64,
    pub average_mapq: f64,
    pub average_coverage: f64,
}

impl AlignmentStats {
    /// Calculate mapping rate percentage
    pub fn mapping_rate(&self) -> f64 {
        if self.total_reads == 0 {
            return 0.0;
        }
        (self.mapped_reads as f64 / self.total_reads as f64) * 100.0
    }

    /// Calculate duplication rate percentage
    pub fn duplication_rate(&self) -> f64 {
        if self.total_reads == 0 {
            return 0.0;
        }
        (self.duplicates as f64 / self.total_reads as f64) * 100.0
    }
}

impl fmt::Display for AlignmentStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AlignmentStats {{ total: {}, mapped: {} ({:.2}%), duplicates: {} ({:.2}%), avg_mapq: {:.2} }}",
            self.total_reads,
            self.mapped_reads,
            self.mapping_rate(),
            self.duplicates,
            self.duplication_rate(),
            self.average_mapq
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastq_record_creation() {
        let record = FastqRecord::new(
            "read1".to_string(),
            "ACGTACGT".to_string(),
            vec![40, 40, 40, 40, 40, 40, 40, 40],
        );
        assert!(record.is_ok());
    }

    #[test]
    fn test_fastq_invalid_sequence() {
        let record = FastqRecord::new(
            "read1".to_string(),
            "ACGTXYZ".to_string(),
            vec![40, 40, 40, 40, 40, 40, 40],
        );
        assert!(record.is_err());
    }

    #[test]
    fn test_gc_content() {
        let record =
            FastqRecord::new("read1".to_string(), "GCGCGCGC".to_string(), vec![40; 8]).unwrap();
        assert_eq!(record.gc_content(), 100.0);
    }

    #[test]
    fn test_genomic_coordinate_overlap() {
        let coord1 = GenomicCoordinate::new(0, 100, 200, true).unwrap();
        let coord2 = GenomicCoordinate::new(0, 150, 250, true).unwrap();
        assert!(coord1.overlaps(&coord2));
    }
}
