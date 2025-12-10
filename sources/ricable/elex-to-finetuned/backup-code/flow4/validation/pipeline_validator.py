"""
Comprehensive pipeline validation for Flow4 document processing.

This module provides validation for each step of the Flow4 pipeline:
- Document extraction and discovery
- Document conversion (HTML/PDF to Markdown)
- Document concatenation and cleaning
- Document chunking
- RAG dataset creation
"""

import os
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PipelineValidationReport:
    """Comprehensive pipeline validation report."""
    timestamp: str
    pipeline_version: str
    validation_results: Dict[str, List[ValidationResult]]
    overall_score: float
    passed_checks: int
    total_checks: int
    recommendations: List[str]
    errors: List[str]


class PipelineValidator:
    """Validates each step of the Flow4 document processing pipeline."""

    def __init__(self, pipeline_output_dir: str = "pipeline_output"):
        """Initialize the pipeline validator.
        
        Args:
            pipeline_output_dir: Directory containing pipeline outputs
        """
        self.output_dir = Path(pipeline_output_dir)
        self.validation_results = {}
        self.errors = []
        self.recommendations = []

    def validate_extraction_phase(self, input_path: str) -> List[ValidationResult]:
        """Validate document extraction and discovery phase.
        
        Args:
            input_path: Path to input ZIP or directory
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check if input exists
        if not os.path.exists(input_path):
            results.append(ValidationResult(
                passed=False,
                message=f"Input path does not exist: {input_path}",
                score=0.0
            ))
            return results
        
        results.append(ValidationResult(
            passed=True,
            message="Input path exists and is accessible",
            score=1.0
        ))
        
        # Check input type and structure
        if input_path.lower().endswith('.zip'):
            results.extend(self._validate_zip_structure(input_path))
        else:
            results.extend(self._validate_directory_structure(input_path))
        
        return results

    def _validate_zip_structure(self, zip_path: str) -> List[ValidationResult]:
        """Validate ZIP file structure."""
        results = []
        
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Count different file types
                html_files = [f for f in file_list if f.lower().endswith('.html')]
                pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]
                
                if not html_files and not pdf_files:
                    results.append(ValidationResult(
                        passed=False,
                        message="No HTML or PDF files found in ZIP archive",
                        score=0.0
                    ))
                else:
                    results.append(ValidationResult(
                        passed=True,
                        message=f"Found {len(html_files)} HTML and {len(pdf_files)} PDF files",
                        score=1.0,
                        metadata={"html_count": len(html_files), "pdf_count": len(pdf_files)}
                    ))
                
                # Check for reasonable file sizes
                total_size = sum(zip_ref.getinfo(f).file_size for f in file_list)
                if total_size > 500 * 1024 * 1024:  # 500MB
                    results.append(ValidationResult(
                        passed=False,
                        message=f"ZIP archive very large ({total_size / (1024*1024):.1f}MB). May cause memory issues.",
                        score=0.5
                    ))
                else:
                    results.append(ValidationResult(
                        passed=True,
                        message=f"ZIP archive size reasonable ({total_size / (1024*1024):.1f}MB)",
                        score=1.0
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                message=f"Error reading ZIP file: {str(e)}",
                score=0.0
            ))
        
        return results

    def _validate_directory_structure(self, dir_path: str) -> List[ValidationResult]:
        """Validate directory structure."""
        results = []
        
        try:
            # Find HTML and PDF files
            html_files = list(Path(dir_path).glob("**/*.html"))
            pdf_files = list(Path(dir_path).glob("**/*.pdf"))
            
            if not html_files and not pdf_files:
                results.append(ValidationResult(
                    passed=False,
                    message="No HTML or PDF files found in directory",
                    score=0.0
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Found {len(html_files)} HTML and {len(pdf_files)} PDF files",
                    score=1.0,
                    metadata={"html_count": len(html_files), "pdf_count": len(pdf_files)}
                ))
            
            # Check directory depth (too deep might indicate issues)
            max_depth = max(len(f.parts) - len(Path(dir_path).parts) for f in html_files + pdf_files) if html_files or pdf_files else 0
            if max_depth > 5:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Directory structure very deep (max depth: {max_depth}). May cause path issues.",
                    score=0.5
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Directory structure reasonable (max depth: {max_depth})",
                    score=1.0
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                message=f"Error scanning directory: {str(e)}",
                score=0.0
            ))
        
        return results

    def validate_conversion_phase(self) -> List[ValidationResult]:
        """Validate document conversion phase."""
        results = []
        markdown_dir = self.output_dir / "markdown"
        
        # Check if markdown directory exists
        if not markdown_dir.exists():
            results.append(ValidationResult(
                passed=False,
                message="Markdown output directory does not exist",
                score=0.0
            ))
            return results
        
        # Count markdown files
        markdown_files = list(markdown_dir.glob("*.md"))
        if not markdown_files:
            results.append(ValidationResult(
                passed=False,
                message="No markdown files found in output directory",
                score=0.0
            ))
            return results
        
        results.append(ValidationResult(
            passed=True,
            message=f"Found {len(markdown_files)} converted markdown files",
            score=1.0,
            metadata={"markdown_count": len(markdown_files)}
        ))
        
        # Validate individual markdown files
        results.extend(self._validate_markdown_files(markdown_files))
        
        return results

    def _validate_markdown_files(self, markdown_files: List[Path]) -> List[ValidationResult]:
        """Validate individual markdown files."""
        results = []
        
        valid_files = 0
        total_size = 0
        empty_files = 0
        
        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    empty_files += 1
                    continue
                
                total_size += len(content)
                
                # Check for Docling conversion artifacts
                if "Document converted using" in content:
                    valid_files += 1
                elif len(content) > 100:  # At least some meaningful content
                    valid_files += 1
                
            except Exception as e:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Error reading {md_file.name}: {str(e)}",
                    score=0.0
                ))
        
        # Report empty files
        if empty_files > 0:
            results.append(ValidationResult(
                passed=False,
                message=f"Found {empty_files} empty markdown files",
                score=0.5,
                metadata={"empty_files": empty_files}
            ))
        
        # Report conversion quality
        success_rate = valid_files / len(markdown_files) if markdown_files else 0
        if success_rate < 0.8:
            results.append(ValidationResult(
                passed=False,
                message=f"Low conversion success rate: {success_rate:.1%}",
                score=success_rate
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                message=f"Good conversion success rate: {success_rate:.1%}",
                score=success_rate
            ))
        
        # Check average file size
        avg_size = total_size / valid_files if valid_files > 0 else 0
        if avg_size < 500:
            results.append(ValidationResult(
                passed=False,
                message=f"Converted files very small (avg: {avg_size:.0f} chars). May indicate conversion issues.",
                score=0.5
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                message=f"Reasonable converted file sizes (avg: {avg_size:.0f} chars)",
                score=1.0
            ))
        
        return results

    def validate_concatenation_phase(self) -> List[ValidationResult]:
        """Validate document concatenation phase."""
        results = []
        combined_dir = self.output_dir / "combined"
        combined_file = combined_dir / "combined_document.md"
        
        # Check if combined file exists
        if not combined_file.exists():
            results.append(ValidationResult(
                passed=False,
                message="Combined document file does not exist",
                score=0.0
            ))
            return results
        
        try:
            with open(combined_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic content validation
            if not content.strip():
                results.append(ValidationResult(
                    passed=False,
                    message="Combined document is empty",
                    score=0.0
                ))
                return results
            
            results.append(ValidationResult(
                passed=True,
                message=f"Combined document created ({len(content):,} characters)",
                score=1.0,
                metadata={"content_length": len(content)}
            ))
            
            # Check for proper content merging
            results.extend(self._validate_combined_content(content))
            
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                message=f"Error reading combined document: {str(e)}",
                score=0.0
            ))
        
        return results

    def _validate_combined_content(self, content: str) -> List[ValidationResult]:
        """Validate combined document content quality."""
        results = []
        
        # Check for reasonable content length
        if len(content) < 10000:  # 10KB minimum
            results.append(ValidationResult(
                passed=False,
                message=f"Combined document very short ({len(content):,} chars). May indicate incomplete processing.",
                score=0.5
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                message=f"Combined document has substantial content ({len(content):,} chars)",
                score=1.0
            ))
        
        # Check for excessive repetition
        lines = content.split('\n')
        unique_lines = len(set(line.strip() for line in lines if line.strip()))
        repetition_ratio = unique_lines / len(lines) if lines else 0
        
        if repetition_ratio < 0.7:
            results.append(ValidationResult(
                passed=False,
                message=f"High content repetition detected (uniqueness: {repetition_ratio:.1%})",
                score=repetition_ratio
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                message=f"Good content diversity (uniqueness: {repetition_ratio:.1%})",
                score=1.0
            ))
        
        # Check for cleaning artifacts
        artifacts = [
            "radioequipmentclockreference",
            "column 2 =",
            "column 3 =",
            "--- =",
            "document.md"
        ]
        
        artifact_count = sum(content.lower().count(artifact) for artifact in artifacts)
        if artifact_count > 10:
            results.append(ValidationResult(
                passed=False,
                message=f"Document contains cleaning artifacts ({artifact_count} instances)",
                score=0.5,
                metadata={"artifact_count": artifact_count}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                message="Document appears well-cleaned",
                score=1.0
            ))
        
        return results

    def validate_chunking_phase(self) -> List[ValidationResult]:
        """Validate document chunking phase."""
        results = []
        chunks_dir = self.output_dir / "chunks"
        
        # Check if chunks directory exists
        if not chunks_dir.exists():
            results.append(ValidationResult(
                passed=False,
                message="Chunks output directory does not exist",
                score=0.0
            ))
            return results
        
        # Count chunk files
        chunk_files = list(chunks_dir.glob("chunk_*.json"))
        if not chunk_files:
            results.append(ValidationResult(
                passed=False,
                message="No chunk files found",
                score=0.0
            ))
            return results
        
        results.append(ValidationResult(
            passed=True,
            message=f"Found {len(chunk_files)} chunk files",
            score=1.0,
            metadata={"chunk_count": len(chunk_files)}
        ))
        
        # Validate chunk files
        results.extend(self._validate_chunk_files(chunk_files))
        
        return results

    def _validate_chunk_files(self, chunk_files: List[Path]) -> List[ValidationResult]:
        """Validate individual chunk files."""
        results = []
        
        chunk_sizes = []
        valid_chunks = 0
        empty_chunks = 0
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                # Validate chunk structure
                required_fields = ['id', 'text', 'source']
                missing_fields = [field for field in required_fields if field not in chunk_data]
                
                if missing_fields:
                    results.append(ValidationResult(
                        passed=False,
                        message=f"Chunk {chunk_file.name} missing fields: {missing_fields}",
                        score=0.0
                    ))
                    continue
                
                text_content = chunk_data.get('text', '')
                if not text_content.strip():
                    empty_chunks += 1
                    continue
                
                chunk_sizes.append(len(text_content))
                valid_chunks += 1
                
            except Exception as e:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Error reading chunk {chunk_file.name}: {str(e)}",
                    score=0.0
                ))
        
        # Report chunk statistics
        if chunk_sizes:
            avg_size = statistics.mean(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            
            # Check chunk size distribution
            if avg_size < 200:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Average chunk size very small ({avg_size:.0f} chars)",
                    score=0.5
                ))
            elif avg_size > 4000:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Average chunk size very large ({avg_size:.0f} chars)",
                    score=0.5
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Good chunk size distribution (avg: {avg_size:.0f} chars)",
                    score=1.0,
                    metadata={
                        "avg_size": avg_size,
                        "min_size": min_size,
                        "max_size": max_size
                    }
                ))
        
        # Report empty chunks
        if empty_chunks > 0:
            results.append(ValidationResult(
                passed=False,
                message=f"Found {empty_chunks} empty chunks",
                score=0.5,
                metadata={"empty_chunks": empty_chunks}
            ))
        
        # Overall chunking quality
        success_rate = valid_chunks / len(chunk_files) if chunk_files else 0
        if success_rate < 0.9:
            results.append(ValidationResult(
                passed=False,
                message=f"Low chunking success rate: {success_rate:.1%}",
                score=success_rate
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                message=f"Good chunking success rate: {success_rate:.1%}",
                score=success_rate
            ))
        
        return results

    def validate_rag_datasets(self) -> List[ValidationResult]:
        """Validate RAG dataset creation."""
        results = []
        rag_dir = self.output_dir / "rag_datasets"
        
        # Check if RAG datasets directory exists
        if not rag_dir.exists():
            results.append(ValidationResult(
                passed=False,
                message="RAG datasets directory does not exist",
                score=0.0
            ))
            return results
        
        # Find dataset files
        dataset_files = list(rag_dir.glob("*.json")) + list(rag_dir.glob("*.jsonl"))
        if not dataset_files:
            results.append(ValidationResult(
                passed=False,
                message="No RAG dataset files found",
                score=0.0
            ))
            return results
        
        results.append(ValidationResult(
            passed=True,
            message=f"Found {len(dataset_files)} dataset files",
            score=1.0,
            metadata={"dataset_count": len(dataset_files)}
        ))
        
        # Validate each dataset file
        for dataset_file in dataset_files:
            results.extend(self._validate_dataset_file(dataset_file))
        
        return results

    def _validate_dataset_file(self, dataset_file: Path) -> List[ValidationResult]:
        """Validate individual dataset file."""
        results = []
        
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                if dataset_file.suffix == '.jsonl':
                    # JSONL format
                    lines = f.readlines()
                    entries = []
                    for line in lines:
                        if line.strip():
                            entries.append(json.loads(line))
                else:
                    # JSON format
                    data = json.load(f)
                    if isinstance(data, dict) and 'data' in data:
                        entries = data['data']
                    elif isinstance(data, list):
                        entries = data
                    else:
                        entries = [data]
            
            if not entries:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Dataset {dataset_file.name} is empty",
                    score=0.0
                ))
                return results
            
            results.append(ValidationResult(
                passed=True,
                message=f"Dataset {dataset_file.name} contains {len(entries)} entries",
                score=1.0,
                metadata={"entry_count": len(entries)}
            ))
            
            # Validate entry structure
            valid_entries = 0
            for entry in entries[:10]:  # Check first 10 entries
                if isinstance(entry, dict) and ('text' in entry or 'instruction' in entry or 'messages' in entry):
                    valid_entries += 1
            
            if valid_entries < len(entries[:10]) * 0.8:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Dataset {dataset_file.name} has malformed entries",
                    score=0.5
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Dataset {dataset_file.name} has well-formed entries",
                    score=1.0
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                message=f"Error reading dataset {dataset_file.name}: {str(e)}",
                score=0.0
            ))
        
        return results

    def run_full_validation(self, input_path: str) -> PipelineValidationReport:
        """Run complete pipeline validation.
        
        Args:
            input_path: Path to original input (ZIP or directory)
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Starting comprehensive pipeline validation...")
        
        # Phase 1: Extraction validation
        logger.info("Validating extraction phase...")
        self.validation_results['extraction'] = self.validate_extraction_phase(input_path)
        
        # Phase 2: Conversion validation
        logger.info("Validating conversion phase...")
        self.validation_results['conversion'] = self.validate_conversion_phase()
        
        # Phase 3: Concatenation validation
        logger.info("Validating concatenation phase...")
        self.validation_results['concatenation'] = self.validate_concatenation_phase()
        
        # Phase 4: Chunking validation
        logger.info("Validating chunking phase...")
        self.validation_results['chunking'] = self.validate_chunking_phase()
        
        # Phase 5: RAG datasets validation
        logger.info("Validating RAG datasets...")
        self.validation_results['rag_datasets'] = self.validate_rag_datasets()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Calculate overall score
        all_results = []
        for phase_results in self.validation_results.values():
            all_results.extend(phase_results)
        
        passed_checks = sum(1 for result in all_results if result.passed)
        total_checks = len(all_results)
        overall_score = sum(result.score for result in all_results) / total_checks if total_checks > 0 else 0
        
        # Create report
        report = PipelineValidationReport(
            timestamp=datetime.now().isoformat(),
            pipeline_version="Flow4-v0.2.0",
            validation_results=self.validation_results,
            overall_score=overall_score,
            passed_checks=passed_checks,
            total_checks=total_checks,
            recommendations=self.recommendations,
            errors=self.errors
        )
        
        logger.info(f"Pipeline validation completed. Score: {overall_score:.2f}")
        return report

    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        # Analyze results and generate recommendations
        all_results = []
        for phase_results in self.validation_results.values():
            all_results.extend(phase_results)
        
        failed_results = [r for r in all_results if not r.passed]
        
        if failed_results:
            self.recommendations.append(f"Address {len(failed_results)} failed validation checks")
        
        # Phase-specific recommendations
        for phase, results in self.validation_results.items():
            phase_failed = [r for r in results if not r.passed]
            if phase_failed:
                self.recommendations.append(f"Review {phase} phase: {len(phase_failed)} issues found")
        
        # Quality-based recommendations
        avg_scores = {}
        for phase, results in self.validation_results.items():
            if results:
                avg_scores[phase] = sum(r.score for r in results) / len(results)
        
        low_quality_phases = [phase for phase, score in avg_scores.items() if score < 0.7]
        if low_quality_phases:
            self.recommendations.append(f"Improve quality in phases: {', '.join(low_quality_phases)}")

    def save_report(self, report: PipelineValidationReport, output_file: str):
        """Save validation report to file."""
        # Convert ValidationResult objects to dictionaries for JSON serialization
        serializable_results = {}
        for phase, results in report.validation_results.items():
            serializable_results[phase] = [
                {
                    'passed': r.passed,
                    'message': r.message,
                    'score': r.score,
                    'metadata': r.metadata
                }
                for r in results
            ]
        
        report_data = {
            'timestamp': report.timestamp,
            'pipeline_version': report.pipeline_version,
            'validation_results': serializable_results,
            'overall_score': report.overall_score,
            'passed_checks': report.passed_checks,
            'total_checks': report.total_checks,
            'recommendations': report.recommendations,
            'errors': report.errors
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to {output_file}")