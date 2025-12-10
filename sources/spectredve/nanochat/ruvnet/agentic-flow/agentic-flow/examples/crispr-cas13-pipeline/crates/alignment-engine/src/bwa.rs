//! BWA (Burrows-Wheeler Aligner) wrapper implementation

use crate::error::{AlignmentError, Result};
use crate::{Aligner, AlignmentConfig};
use async_trait::async_trait;
use data_models::sequencing::{AlignedRead, FastqRecord, GenomicCoordinate};
use rayon::prelude::*;
use std::path::Path;
use std::process::{Command, Stdio};
use uuid::Uuid;

/// BWA aligner implementation
pub struct BwaAligner {
    config: AlignmentConfig,
    reference_indexed: bool,
}

impl BwaAligner {
    /// Create a new BWA aligner
    pub fn new(config: AlignmentConfig) -> Result<Self> {
        if !Path::new(&config.reference_path).exists() {
            return Err(AlignmentError::NotFound(format!(
                "Reference file not found: {}",
                config.reference_path
            )));
        }

        Ok(Self {
            config,
            reference_indexed: false,
        })
    }

    /// Index the reference genome if not already indexed
    pub fn index_reference(&mut self) -> Result<()> {
        tracing::info!("Indexing reference genome: {}", self.config.reference_path);

        let output = Command::new("bwa")
            .arg("index")
            .arg(&self.config.reference_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| AlignmentError::AlignmentFailed(format!("BWA index failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AlignmentError::AlignmentFailed(format!(
                "BWA index failed: {}",
                stderr
            )));
        }

        self.reference_indexed = true;
        tracing::info!("Reference indexing completed");
        Ok(())
    }

    /// Align reads using BWA MEM algorithm
    fn align_with_bwa_mem(&self, reads: &[FastqRecord]) -> Result<Vec<AlignedRead>> {
        // In a real implementation, this would:
        // 1. Write reads to temporary FASTQ file
        // 2. Run BWA MEM command
        // 3. Parse SAM/BAM output
        // 4. Convert to AlignedRead structures

        // For now, return a mock implementation
        let aligned: Vec<AlignedRead> = reads
            .par_iter()
            .filter_map(|read| self.mock_align_single(read).ok().flatten())
            .collect();

        Ok(aligned)
    }

    /// Mock alignment for a single read (placeholder for actual BWA integration)
    fn mock_align_single(&self, read: &FastqRecord) -> Result<Option<AlignedRead>> {
        // This is a simplified mock - real implementation would call BWA
        let avg_quality = read.average_quality();

        if avg_quality < self.config.min_mapq as f64 {
            return Ok(None);
        }

        let position = GenomicCoordinate::new(0, 1000, 1000 + read.sequence.len() as u64, true)?;

        let aligned = AlignedRead {
            id: Uuid::new_v4(),
            qname: read.id.clone(),
            position,
            mapq: avg_quality as u8,
            cigar: format!("{}M", read.sequence.len()),
            sequence: read.sequence.clone(),
            quality: read.quality.clone(),
            flags: 0,
            nm: Some(0),
        };

        Ok(Some(aligned))
    }

    /// Calculate alignment score for a read
    fn calculate_alignment_score(&self, read: &FastqRecord) -> i32 {
        // Simplified scoring based on quality and length
        let avg_qual = read.average_quality();
        let length = read.sequence.len() as i32;
        ((avg_qual * length as f64) / 40.0) as i32
    }
}

#[async_trait]
impl Aligner for BwaAligner {
    async fn align_read(&self, read: &FastqRecord) -> Result<Option<AlignedRead>> {
        self.mock_align_single(read)
    }

    async fn align_batch(&self, reads: Vec<FastqRecord>) -> Result<Vec<AlignedRead>> {
        tokio::task::spawn_blocking({
            let config = self.config.clone();
            move || {
                let aligner = BwaAligner {
                    config,
                    reference_indexed: true,
                };
                aligner.align_with_bwa_mem(&reads)
            }
        })
        .await
        .map_err(|e| AlignmentError::AlignmentFailed(format!("Task join error: {}", e)))?
    }

    fn name(&self) -> &str {
        "BWA-MEM"
    }

    fn version(&self) -> &str {
        "0.7.17"
    }
}

/// CIGAR string parser for alignment operations
pub struct CigarParser;

impl CigarParser {
    /// Parse CIGAR string into operations
    pub fn parse(cigar: &str) -> Result<Vec<CigarOp>> {
        let mut ops = Vec::new();
        let mut num_str = String::new();

        for ch in cigar.chars() {
            if ch.is_ascii_digit() {
                num_str.push(ch);
            } else {
                let length: u32 = num_str
                    .parse()
                    .map_err(|_| AlignmentError::Parse(format!("Invalid CIGAR: {}", cigar)))?;

                let op = match ch {
                    'M' => CigarOp::Match(length),
                    'I' => CigarOp::Insertion(length),
                    'D' => CigarOp::Deletion(length),
                    'N' => CigarOp::Skip(length),
                    'S' => CigarOp::SoftClip(length),
                    'H' => CigarOp::HardClip(length),
                    'P' => CigarOp::Padding(length),
                    '=' => CigarOp::SequenceMatch(length),
                    'X' => CigarOp::SequenceMismatch(length),
                    _ => {
                        return Err(AlignmentError::Parse(format!(
                            "Unknown CIGAR operation: {}",
                            ch
                        )))
                    }
                };

                ops.push(op);
                num_str.clear();
            }
        }

        Ok(ops)
    }

    /// Calculate reference length consumed by CIGAR operations
    pub fn reference_length(ops: &[CigarOp]) -> u32 {
        ops.iter()
            .map(|op| match op {
                CigarOp::Match(len)
                | CigarOp::Deletion(len)
                | CigarOp::Skip(len)
                | CigarOp::SequenceMatch(len)
                | CigarOp::SequenceMismatch(len) => *len,
                _ => 0,
            })
            .sum()
    }

    /// Calculate query length consumed by CIGAR operations
    pub fn query_length(ops: &[CigarOp]) -> u32 {
        ops.iter()
            .map(|op| match op {
                CigarOp::Match(len)
                | CigarOp::Insertion(len)
                | CigarOp::SoftClip(len)
                | CigarOp::SequenceMatch(len)
                | CigarOp::SequenceMismatch(len) => *len,
                _ => 0,
            })
            .sum()
    }
}

/// CIGAR operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CigarOp {
    Match(u32),
    Insertion(u32),
    Deletion(u32),
    Skip(u32),
    SoftClip(u32),
    HardClip(u32),
    Padding(u32),
    SequenceMatch(u32),
    SequenceMismatch(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cigar_parser() {
        let cigar = "10M2I5M3D8M";
        let ops = CigarParser::parse(cigar).unwrap();

        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0], CigarOp::Match(10));
        assert_eq!(ops[1], CigarOp::Insertion(2));
        assert_eq!(ops[2], CigarOp::Match(5));
        assert_eq!(ops[3], CigarOp::Deletion(3));
        assert_eq!(ops[4], CigarOp::Match(8));
    }

    #[test]
    fn test_cigar_lengths() {
        let ops = vec![
            CigarOp::Match(10),
            CigarOp::Insertion(2),
            CigarOp::Deletion(3),
        ];

        assert_eq!(CigarParser::reference_length(&ops), 13);
        assert_eq!(CigarParser::query_length(&ops), 12);
    }
}
