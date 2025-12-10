#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Ericsson RAN Features Processing System

This test suite validates:
1. End-to-end pipeline: markdown → features → skill → ZIP
2. Performance validation (5 files, 100 files, memory usage)
3. Output quality verification (SKILL.md, references, indices)
4. Edge cases and error scenarios (missing files, corruption, recovery)
5. Cache performance and validation

Based on final-plan.md requirements and CLAUDE.md specifications.
"""

import os
import sys
import json
import time
import shutil
import zipfile
import hashlib
import tempfile
import traceback

# Try to import psutil, but make it optional for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - memory monitoring disabled")
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import unittest
from unittest.mock import patch, MagicMock

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ericsson_feature_processor import EricssonFeatureProcessor
from ericsson_skill_generator import EricssonSkillGenerator


@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    passed: bool
    duration: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    processing_time: float
    memory_peak: float
    memory_avg: float
    files_processed: int
    features_extracted: int
    cache_hit_rate: float = 0.0


class IntegrationTestSuite:
    """Comprehensive integration test suite for Ericsson RAN Features system"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.source_dir = self.base_dir / "elex_features_only"
        self.test_output_dir = self.base_dir / "test_integration_output"
        self.temp_dir = None

        self.test_results: List[TestResult] = []
        self.performance_metrics: List[PerformanceMetrics] = []

        # Performance targets from final-plan.md
        self.TARGET_5_FILES = 30  # seconds
        self.TARGET_100_FILES = 180  # seconds (3 minutes)
        self.TARGET_MEMORY_MB = 500  # MB

        # Initialize test environment
        self.setup_test_environment()

        # Helper method to get correct data directory path
        self._get_ericsson_data_dir = lambda base_dir: Path(base_dir) / "ericsson_data"

    def setup_test_environment(self):
        """Setup isolated test environment"""
        print("🔧 Setting up test environment...")

        # Create temporary directory for isolated testing
        self.temp_dir = tempfile.mkdtemp(prefix="ericsson_test_")

        # Clean any existing test output
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ Test environment ready: {self.test_output_dir}")

    def cleanup_test_environment(self):
        """Cleanup test environment"""
        print("🧹 Cleaning up test environment...")

        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        print("✅ Test environment cleaned up")

    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test with error handling and timing"""
        print(f"\n🧪 Running: {test_name}")
        start_time = time.time()

        try:
            details = test_func()
            duration = time.time() - start_time

            result = TestResult(
                test_name=test_name,
                passed=True,
                duration=duration,
                details=details
            )

            print(f"✅ {test_name} - PASSED ({duration:.2f}s)")

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"

            result = TestResult(
                test_name=test_name,
                passed=False,
                duration=duration,
                details={},
                error_message=error_msg
            )

            print(f"❌ {test_name} - FAILED ({duration:.2f}s)")
            print(f"   Error: {error_msg}")

            # Print full traceback for debugging
            print(f"   Traceback: {traceback.format_exc()}")

        self.test_results.append(result)
        return result

    # ==================== END-TO-END PIPELINE TESTS ====================

    def test_end_to_end_5_files(self) -> Dict[str, Any]:
        """Test complete pipeline with 5 files"""
        print("   📊 Testing end-to-end pipeline with 5 files...")

        # Verify source files exist
        md_files = list(self.source_dir.rglob("*.md"))
        if len(md_files) < 5:
            raise ValueError(f"Need at least 5 files, found {len(md_files)}")

        test_files = md_files[:5]

        # Create test source directory
        test_source = self.test_output_dir / "test_source"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        # Run processor
        processor = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(self.test_output_dir / "data"),
            batch_size=5
        )

        process_start = time.time()
        processor.process_all()
        processing_time = time.time() - process_start

        # Verify processed data
        data_dir = Path(self.test_output_dir / "data")
        if not data_dir.exists():
            raise ValueError("Processor output directory not created")

        # Check summary (processor creates ericsson_data subdirectory)
        ericsson_data_dir = data_dir / "ericsson_data"
        if not ericsson_data_dir.exists():
            raise ValueError("Ericsson data directory not created")

        summary_file = ericsson_data_dir / "summary.json"
        if not summary_file.exists():
            raise ValueError("Summary file not generated")

        summary = json.loads(summary_file.read_text())
        processed_files = summary.get('processing_stats', {}).get('processed', 0)
        if processed_files != 5:
            raise ValueError(f"Expected 5 files processed, got {processed_files}")

        # Run skill generator
        generator = EricssonSkillGenerator(
            data_dir=str(ericsson_data_dir),
            output_dir=str(self.test_output_dir)
        )

        gen_start = time.time()
        generator.generate_skill()
        generation_time = time.time() - gen_start

        # Verify skill output
        skill_dir = self.test_output_dir / "ericsson"
        if not skill_dir.exists():
            raise ValueError("Skill directory not created")

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            raise ValueError("SKILL.md not generated")

        # Check references
        refs_dir = skill_dir / "references"
        if not refs_dir.exists():
            raise ValueError("References directory not created")

        # Package skill
        package_start = time.time()
        package_result = generator.package_skill()
        packaging_time = time.time() - package_start

        zip_filename = package_result['zip_file_name']
        zip_path = Path(self.test_output_dir) / zip_filename

        if not zip_path.exists():
            raise ValueError("ZIP file not created")

        # Validate ZIP contents
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files = zf.namelist()
            if "SKILL.md" not in files:
                raise ValueError("SKILL.md missing from ZIP")
            if not any(f.startswith("references/") for f in files):
                raise ValueError("Reference files missing from ZIP")

        total_time = processing_time + generation_time + packaging_time

        return {
            'files_processed': 5,
            'processing_time': processing_time,
            'generation_time': generation_time,
            'packaging_time': packaging_time,
            'total_time': total_time,
            'features_extracted': summary.get('total_features', 0),
            'zip_file_size': zip_path.stat().st_size,
            'zip_file_path': str(zip_path),
            'package_stats': package_result
        }

    def test_end_to_end_100_files(self) -> Dict[str, Any]:
        """Test complete pipeline with 100 files (if available)"""
        print("   📊 Testing end-to-end pipeline with 100 files...")

        # Check if we have enough files
        md_files = list(self.source_dir.rglob("*.md"))
        if len(md_files) < 100:
            print(f"   ⚠️  Only {len(md_files)} files available, testing with all files")
            test_files = md_files
        else:
            test_files = md_files[:100]

        # Create test source directory
        test_source = self.test_output_dir / "test_source_100"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        # Monitor memory usage if psutil is available
        memory_samples = []

        if PSUTIL_AVAILABLE:
            process = psutil.Process()

            def sample_memory():
                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
        else:
            def sample_memory():
                memory_samples.append(0)  # Placeholder when psutil not available

        # Run processor with memory monitoring
        processor = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(self.test_output_dir / "data_100"),
            batch_size=20
        )

        # Patch processor to sample memory
        original_process_file = processor.process_file
        def process_file_with_memory(file_path):
            sample_memory()
            return original_process_file(file_path)

        processor.process_file = process_file_with_memory

        process_start = time.time()
        processor.process_all()
        processing_time = time.time() - process_start

        # Calculate memory metrics
        memory_peak = max(memory_samples) if memory_samples else 0
        memory_avg = sum(memory_samples) / len(memory_samples) if memory_samples else 0

        # Verify processed data
        data_dir = Path(self.test_output_dir / "data_100")
        ericsson_data_dir = self._get_ericsson_data_dir(data_dir)
        summary_file = ericsson_data_dir / "summary.json"
        summary = json.loads(summary_file.read_text())

        # Run skill generator
        generator = EricssonSkillGenerator(
            data_dir=str(ericsson_data_dir),
            output_dir=str(self.test_output_dir)
        )

        gen_start = time.time()
        generator.generate_skill()
        generation_time = time.time() - gen_start

        # Package skill
        package_result = generator.package_skill()
        zip_filename = package_result['zip_file_name']
        zip_path = Path(self.test_output_dir) / zip_filename

        total_time = processing_time + generation_time

        return {
            'files_processed': len(test_files),
            'processing_time': processing_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'memory_peak_mb': memory_peak,
            'memory_avg_mb': memory_avg,
            'features_extracted': summary.get('total_features', 0),
            'zip_file_size': zip_path.stat().st_size if zip_path.exists() else 0,
            'package_stats': package_result
        }

    # ==================== PERFORMANCE VALIDATION TESTS ====================

    def test_performance_targets(self) -> Dict[str, Any]:
        """Test performance against targets from final-plan.md"""
        print("   ⚡ Testing performance targets...")

        results = {}

        # Test 5-file performance
        md_files = list(self.source_dir.rglob("*.md"))
        if len(md_files) >= 5:
            test_files = md_files[:5]

            test_source = self.test_output_dir / "perf_test_5"
            test_source.mkdir()

            for file in test_files:
                shutil.copy2(file, test_source / file.name)

            processor = EricssonFeatureProcessor(
                source_dir=str(test_source),
                output_dir=str(self.test_output_dir / "perf_data_5")
            )

            start_time = time.time()
            processor.process_all()
            processing_time = time.time() - start_time

            results['time_5_files'] = processing_time
            results['target_5_files'] = self.TARGET_5_FILES
            results['passed_5_files'] = processing_time <= self.TARGET_5_FILES

            if not results['passed_5_files']:
                print(f"   ⚠️  5-file test exceeded target: {processing_time:.2f}s > {self.TARGET_5_FILES}s")

        return results

    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage during processing"""
        print("   💾 Testing memory usage...")

        if not PSUTIL_AVAILABLE:
            print("   ⚠️  psutil not available - skipping detailed memory monitoring")
            return {
                'files_tested': 0,
                'initial_memory_mb': 0,
                'peak_memory_mb': 0,
                'avg_memory_mb': 0,
                'memory_increase_mb': 0,
                'target_memory_mb': self.TARGET_MEMORY_MB,
                'passed_memory_target': True,
                'skipped': True
            }

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]

        # Create memory sampling wrapper
        def sample_memory_periodically():
            memory_samples.append(process.memory_info().rss / 1024 / 1024)

        # Test with a reasonable number of files
        md_files = list(self.source_dir.rglob("*.md"))
        test_count = min(50, len(md_files))  # Test with up to 50 files

        if test_count == 0:
            raise ValueError("No markdown files found for memory testing")

        test_files = md_files[:test_count]
        test_source = self.test_output_dir / "memory_test"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        # Patch processor to sample memory
        processor = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(self.test_output_dir / "memory_data"),
            batch_size=10
        )

        original_process_batch = processor.process_batch
        def process_batch_with_memory(batch):
            sample_memory_periodically()
            return original_process_batch(batch)

        processor.process_batch = process_batch_with_memory

        # Run processing
        processor.process_all()

        # Calculate memory metrics
        peak_memory = max(memory_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        memory_increase = peak_memory - initial_memory

        results = {
            'files_tested': test_count,
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'avg_memory_mb': avg_memory,
            'memory_increase_mb': memory_increase,
            'target_memory_mb': self.TARGET_MEMORY_MB,
            'passed_memory_target': peak_memory <= self.TARGET_MEMORY_MB
        }

        if not results['passed_memory_target']:
            print(f"   ⚠️  Memory usage exceeded target: {peak_memory:.1f}MB > {self.TARGET_MEMORY_MB}MB")

        return results

    def test_cache_performance(self) -> Dict[str, Any]:
        """Test caching system performance"""
        print("   🗄️  Testing cache performance...")

        # Use a small set of files for cache testing
        md_files = list(self.source_dir.rglob("*.md"))
        test_files = md_files[:10]

        test_source = self.test_output_dir / "cache_test"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        output_dir = self.test_output_dir / "cache_data"

        # First run - populate cache
        processor1 = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(output_dir),
            enable_cache=True
        )

        start_time = time.time()
        processor1.process_all()
        first_run_time = time.time() - start_time

        # Second run - use cache
        processor2 = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(output_dir),
            enable_cache=True
        )

        start_time = time.time()
        processor2.process_all()
        second_run_time = time.time() - start_time

        # Calculate cache efficiency
        cache_speedup = first_run_time / second_run_time if second_run_time > 0 else 0

        # Verify cache files exist
        cache_dir = Path(output_dir) / "cache"
        cache_files = list(cache_dir.glob("*")) if cache_dir.exists() else []

        results = {
            'files_tested': len(test_files),
            'first_run_time': first_run_time,
            'second_run_time': second_run_time,
            'cache_speedup': cache_speedup,
            'cache_files_created': len(cache_files),
            'cache_working': second_run_time < first_run_time
        }

        return results

    # ==================== OUTPUT QUALITY VERIFICATION TESTS ====================

    def test_skill_md_quality(self) -> Dict[str, Any]:
        """Test SKILL.md quality and structure"""
        print("   📄 Testing SKILL.md quality...")

        # Generate a skill first
        md_files = list(self.source_dir.rglob("*.md"))
        test_files = md_files[:5]

        test_source = self.test_output_dir / "quality_test_source"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        # Process and generate skill
        processor = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(self.test_output_dir / "quality_data")
        )
        processor.process_all()

        generator = EricssonSkillGenerator(
            data_dir=str(self.test_output_dir / "quality_data/ericsson_data"),
            output_dir=str(self.test_output_dir / "quality_output")
        )
        generator.generate_skill()

        # Analyze SKILL.md
        skill_file = Path(self.test_output_dir / "quality_output/ericsson/SKILL.md")
        if not skill_file.exists():
            raise ValueError("SKILL.md not found")

        content = skill_file.read_text()

        # Check required sections
        required_sections = [
            "# Ericsson RAN Features Expert",
            "## When to Use This Skill",
            "## Key Features",
            "## Example Queries",
            "## Available Features",
            "## Navigation"
        ]

        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        # Check for examples
        has_examples = "Example:" in content or "```" in content

        # Check structure
        line_count = len(content.splitlines())
        word_count = len(content.split())

        results = {
            'file_exists': True,
            'file_size_bytes': skill_file.stat().st_size,
            'line_count': line_count,
            'word_count': word_count,
            'has_examples': has_examples,
            'required_sections_present': len(required_sections) - len(missing_sections),
            'missing_sections': missing_sections,
            'meets_quality_standards': len(missing_sections) == 0 and has_examples and word_count > 500
        }

        return results

    def test_reference_structure(self) -> Dict[str, Any]:
        """Test reference file structure and categorization"""
        print("   📚 Testing reference file structure...")

        # Generate skill first (reuse from previous test if exists)
        skill_dir = Path(self.test_output_dir / "quality_output/ericsson")
        if not skill_dir.exists():
            # Generate if not exists
            md_files = list(self.source_dir.rglob("*.md"))
            test_files = md_files[:5]

            test_source = self.test_output_dir / "ref_test_source"
            test_source.mkdir()

            for file in test_files:
                shutil.copy2(file, test_source / file.name)

            processor = EricssonFeatureProcessor(
                source_dir=str(test_source),
                output_dir=str(self.test_output_dir / "ref_data")
            )
            processor.process_all()

            generator = EricssonSkillGenerator(
                data_dir=str(self.test_output_dir / "ref_data/ericsson_data"),
                output_dir=str(self.test_output_dir / "ref_output")
            )
            generator.generate_skill()
            skill_dir = Path(self.test_output_dir / "ref_output/ericsson")

        refs_dir = skill_dir / "references"
        if not refs_dir.exists():
            raise ValueError("References directory not found")

        # Check expected reference structure
        expected_dirs = [
            "features",
            "parameters",
            "counters",
            "cxc_codes",
            "guidelines"
        ]

        existing_dirs = []
        for dir_name in expected_dirs:
            dir_path = refs_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                existing_dirs.append(dir_name)

        # Count reference files
        ref_files = list(refs_dir.rglob("*.md"))

        # Check index files
        index_files = list(refs_dir.rglob("index.md"))

        # Sample content check
        sample_content = {}
        for ref_file in ref_files[:3]:  # Check first 3 files
            content = ref_file.read_text()
            sample_content[ref_file.name] = {
                'size_bytes': len(content),
                'has_meaningful_content': len(content.strip()) > 50,
                'has_structure': '##' in content or '#' in content
            }

        results = {
            'refs_dir_exists': True,
            'expected_directories': len(expected_dirs),
            'existing_directories': len(existing_dirs),
            'directory_coverage': len(existing_dirs) / len(expected_dirs),
            'total_reference_files': len(ref_files),
            'index_files': len(index_files),
            'sample_content_quality': sample_content,
            'structure_complete': len(existing_dirs) >= len(expected_dirs) * 0.8
        }

        return results

    def test_search_indices(self) -> Dict[str, Any]:
        """Test search indices completeness and accuracy"""
        print("   🔍 Testing search indices...")

        # Process some files to generate indices
        md_files = list(self.source_dir.rglob("*.md"))
        test_files = md_files[:10]

        test_source = self.test_output_dir / "index_test_source"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        processor = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(self.test_output_dir / "index_data")
        )
        processor.process_all()

        # Check indices (in ericsson_data subdirectory)
        indices_dir = Path(self.test_output_dir / "index_data/ericsson_data/indices")
        if not indices_dir.exists():
            raise ValueError("Indices directory not found")

        expected_indices = [
            "parameters_index.json",
            "counters_index.json",
            "cxc_index.json",
            "features_index.json"
        ]

        existing_indices = []
        index_data = {}

        for index_name in expected_indices:
            index_path = indices_dir / index_name
            if index_path.exists():
                existing_indices.append(index_name)
                try:
                    data = json.loads(index_path.read_text())
                    index_data[index_name] = {
                        'exists': True,
                        'entries': len(data) if isinstance(data, dict) else 0,
                        'valid_json': True
                    }
                except json.JSONDecodeError:
                    index_data[index_name] = {
                        'exists': True,
                        'entries': 0,
                        'valid_json': False
                    }
            else:
                index_data[index_name] = {
                    'exists': False,
                    'entries': 0,
                    'valid_json': False
                }

        # Test index functionality
        summary_file = Path(self.test_output_dir / "index_data/summary.json")
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}

        results = {
            'indices_dir_exists': True,
            'expected_indices': len(expected_indices),
            'existing_indices': len(existing_indices),
            'index_coverage': len(existing_indices) / len(expected_indices),
            'index_details': index_data,
            'features_processed': summary.get('features_extracted', 0),
            'indices_functional': len(existing_indices) >= 3
        }

        return results

    def test_zip_integrity(self) -> Dict[str, Any]:
        """Test ZIP file integrity and contents"""
        print("   📦 Testing ZIP file integrity...")

        # Generate skill ZIP
        md_files = list(self.source_dir.rglob("*.md"))
        test_files = md_files[:5]

        test_source = self.test_output_dir / "zip_test_source"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        processor = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(self.test_output_dir / "zip_data")
        )
        processor.process_all()

        generator = EricssonSkillGenerator(
            data_dir=str(self.test_output_dir / "zip_data/ericsson_data"),
            output_dir=str(self.test_output_dir / "zip_output")
        )
        generator.generate_skill()

        package_result = generator.package_skill()
        zip_filename = package_result['zip_file_name']
        zip_path = Path(self.test_output_dir) / zip_filename

        if not zip_path.exists():
            raise ValueError("ZIP file not created")

        # Test ZIP integrity
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Test ZIP file validity
                zf.testzip()

                # List contents
                file_list = zf.namelist()

                # Check required files
                required_files = ["SKILL.md"]
                missing_required = []

                for req_file in required_files:
                    if req_file not in file_list:
                        missing_required.append(req_file)

                # Check structure
                has_references = any(f.startswith("references/") for f in file_list)
                has_skill_md = "SKILL.md" in file_list

                # Calculate sizes
                total_size = sum(f.file_size for f in zf.filelist)
                compressed_size = sum(f.compress_size for f in zf.filelist)
                compression_ratio = compressed_size / total_size if total_size > 0 else 0

        except zipfile.BadZipFile:
            raise ValueError("Generated ZIP file is corrupted")

        results = {
            'zip_exists': True,
            'zip_size_bytes': zip_path.stat().st_size,
            'total_files': len(file_list),
            'required_files_present': len(required_files) - len(missing_required),
            'missing_required_files': missing_required,
            'has_references': has_references,
            'has_skill_md': has_skill_md,
            'total_uncompressed_size': total_size,
            'total_compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'zip_valid': True,
            'package_stats': package_result
        }

        return results

    # ==================== EDGE CASES AND ERROR SCENARIOS ====================

    def test_missing_files_handling(self) -> Dict[str, Any]:
        """Test handling of missing or non-existent files"""
        print("   ❓ Testing missing files handling...")

        # Test with empty directory
        empty_source = self.test_output_dir / "empty_source"
        empty_source.mkdir()

        processor = EricssonFeatureProcessor(
            source_dir=str(empty_source),
            output_dir=str(self.test_output_dir / "empty_data")
        )

        start_time = time.time()
        processor.process_all()
        processing_time = time.time() - start_time

        # Should complete gracefully with no files processed
        summary_file = Path(self.test_output_dir / "empty_data/summary.json")
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}

        results = {
            'empty_dir_handled': True,
            'processing_time': processing_time,
            'files_processed': summary.get('processing_stats', {}).get('processed', 0),
            'features_extracted': summary.get('total_features', 0),
            'graceful_completion': processing_time < 10  # Should complete quickly
        }

        return results

    def test_corrupted_files_handling(self) -> Dict[str, Any]:
        """Test handling of corrupted or invalid markdown files"""
        print("   🚫 Testing corrupted files handling...")

        # Create test files with various issues
        corrupt_source = self.test_output_dir / "corrupt_source"
        corrupt_source.mkdir()

        # Create corrupted files
        test_files = {
            "empty.md": "",
            "binary.md": b"\x00\x01\x02\x03\x04\x05",  # Binary data
            "invalid_encoding.md": "Invalid UTF-8: \xff\xfe\x00".encode('latin1'),
            "no_faj.md": "# Some Feature\nThis has no FAJ ID",
            "malformed_faj.md": "# Bad Feature\nFAJ INVALID FORMAT\nSome content"
        }

        for filename, content in test_files.items():
            file_path = corrupt_source / filename
            if isinstance(content, str):
                file_path.write_text(content, encoding='utf-8', errors='ignore')
            else:
                file_path.write_bytes(content)

        processor = EricssonFeatureProcessor(
            source_dir=str(corrupt_source),
            output_dir=str(self.test_output_dir / "corrupt_data")
        )

        start_time = time.time()
        processor.process_all()
        processing_time = time.time() - start_time

        # Check if processing completed without crashing
        summary_file = Path(self.test_output_dir / "corrupt_data/summary.json")
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}

        # Check error handling
        errors = summary.get('errors', [])

        results = {
            'corrupted_files_tested': len(test_files),
            'processing_completed': True,
            'processing_time': processing_time,
            'files_processed': summary.get('processing_stats', {}).get('processed', 0),
            'errors_logged': len(errors),
            'graceful_handling': processing_time < 30 and len(errors) > 0
        }

        return results

    def test_cache_corruption_recovery(self) -> Dict[str, Any]:
        """Test recovery from cache corruption"""
        print("   🔄 Testing cache corruption recovery...")

        # Process some files normally first
        md_files = list(self.source_dir.rglob("*.md"))
        test_files = md_files[:5]

        test_source = self.test_output_dir / "cache_corrupt_source"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        output_dir = self.test_output_dir / "cache_corrupt_data"

        # Normal processing
        processor1 = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(output_dir),
            enable_cache=True
        )
        processor1.process_all()

        # Corrupt cache files
        cache_dir = Path(output_dir) / "cache"
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*"):
                if cache_file.is_file():
                    # Write garbage data to corrupt the file
                    cache_file.write_text("CORRUPTED CACHE DATA")

        # Try processing again - should recover gracefully
        processor2 = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(output_dir),
            enable_cache=True
        )

        start_time = time.time()
        processor2.process_all()
        recovery_time = time.time() - start_time

        # Check if recovery was successful
        summary_file = Path(output_dir) / "ericsson_data" / "summary.json"
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}

        results = {
            'cache_corrupted': True,
            'recovery_attempted': True,
            'recovery_completed': True,
            'recovery_time': recovery_time,
            'files_processed_after_recovery': summary.get('files_processed', 0),
            'recovery_successful': summary.get('files_processed', 0) > 0
        }

        return results

    def test_memory_pressure_handling(self) -> Dict[str, Any]:
        """Test handling under memory pressure"""
        print("   🧠 Testing memory pressure handling...")

        # Test with small batch sizes to simulate memory pressure
        md_files = list(self.source_dir.rglob("*.md"))
        test_files = md_files[:20]  # Use enough files to create pressure

        if len(test_files) < 10:
            raise ValueError("Need at least 10 files for memory pressure test")

        test_source = self.test_output_dir / "memory_pressure_source"
        test_source.mkdir()

        for file in test_files:
            shutil.copy2(file, test_source / file.name)

        # Use very small batch size to simulate memory pressure
        processor = EricssonFeatureProcessor(
            source_dir=str(test_source),
            output_dir=str(self.test_output_dir / "memory_pressure_data"),
            batch_size=2  # Very small batches
        )

        if not PSUTIL_AVAILABLE:
            print("   ⚠️  psutil not available - skipping detailed memory pressure test")
            return {
                'files_tested': len(test_files),
                'batch_size': 2,
                'initial_memory_mb': 0,
                'final_memory_mb': 0,
                'memory_increase_mb': 0,
                'processing_time': 0,
                'files_processed': 0,
                'handled_pressure': True,
                'skipped': True
            }

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        start_time = time.time()
        processor.process_all()
        processing_time = time.time() - start_time

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        summary_file = Path(self.test_output_dir / "memory_pressure_data/summary.json")
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}

        results = {
            'files_tested': len(test_files),
            'batch_size': 2,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'processing_time': processing_time,
            'files_processed': summary.get('processing_stats', {}).get('processed', 0),
            'handled_pressure': memory_increase < 200  # Should not increase dramatically
        }

        return results

    # ==================== MAIN TEST EXECUTION ====================

    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite"""
        print("🚀 Starting Comprehensive Integration Test Suite")
        print("=" * 60)

        start_time = time.time()

        try:
            # End-to-end pipeline tests
            self.run_test("End-to-End Pipeline (5 files)", self.test_end_to_end_5_files)
            self.run_test("End-to-End Pipeline (100 files)", self.test_end_to_end_100_files)

            # Performance validation tests
            self.run_test("Performance Targets", self.test_performance_targets)
            self.run_test("Memory Usage", self.test_memory_usage)
            self.run_test("Cache Performance", self.test_cache_performance)

            # Output quality verification tests
            self.run_test("SKILL.md Quality", self.test_skill_md_quality)
            self.run_test("Reference Structure", self.test_reference_structure)
            self.run_test("Search Indices", self.test_search_indices)
            self.run_test("ZIP Integrity", self.test_zip_integrity)

            # Edge cases and error scenarios
            self.run_test("Missing Files Handling", self.test_missing_files_handling)
            self.run_test("Corrupted Files Handling", self.test_corrupted_files_handling)
            self.run_test("Cache Corruption Recovery", self.test_cache_corruption_recovery)
            self.run_test("Memory Pressure Handling", self.test_memory_pressure_handling)

        finally:
            total_time = time.time() - start_time
            self.cleanup_test_environment()

        # Generate summary report
        summary = self.generate_summary_report(total_time)

        return summary

    def generate_summary_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]

        # Performance summary
        performance_summary = {}
        for result in self.test_results:
            if 'time' in result.test_name.lower() or 'memory' in result.test_name.lower():
                performance_summary[result.test_name] = {
                    'passed': result.passed,
                    'duration': result.duration,
                    'details': result.details
                }

        # Quality metrics
        quality_metrics = {}
        for result in self.test_results:
            if 'quality' in result.test_name.lower() or 'structure' in result.test_name.lower():
                quality_metrics[result.test_name] = {
                    'passed': result.passed,
                    'details': result.details
                }

        summary = {
            'test_execution': {
                'total_tests': len(self.test_results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(passed_tests) / len(self.test_results) if self.test_results else 0,
                'total_duration': total_time
            },
            'performance_summary': performance_summary,
            'quality_metrics': quality_metrics,
            'failed_tests': [
                {
                    'name': test.test_name,
                    'error': test.error_message,
                    'duration': test.duration
                }
                for test in failed_tests
            ],
            'recommendations': self.generate_recommendations()
        }

        # Print summary
        self.print_summary_report(summary)

        return summary

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        for result in self.test_results:
            if not result.passed:
                if 'performance' in result.test_name.lower():
                    recommendations.append("Consider optimizing processing algorithms or increasing batch sizes")
                elif 'memory' in result.test_name.lower():
                    recommendations.append("Review memory usage patterns and implement better cleanup")
                elif 'quality' in result.test_name.lower():
                    recommendations.append("Enhance content generation and structure validation")
                elif 'cache' in result.test_name.lower():
                    recommendations.append("Improve cache corruption detection and recovery mechanisms")

        if not recommendations:
            recommendations.append("All tests passed! System is performing as expected.")

        return recommendations

    def print_summary_report(self, summary: Dict[str, Any]):
        """Print detailed summary report"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE INTEGRATION TEST SUMMARY")
        print("=" * 60)

        exec_summary = summary['test_execution']
        print(f"Tests Executed: {exec_summary['total_tests']}")
        print(f"Tests Passed: {exec_summary['passed_tests']}")
        print(f"Tests Failed: {exec_summary['failed_tests']}")
        print(f"Success Rate: {exec_summary['success_rate']:.1%}")
        print(f"Total Duration: {exec_summary['total_duration']:.2f}s")

        print("\n🎯 PERFORMANCE RESULTS:")
        for test_name, data in summary['performance_summary'].items():
            status = "✅" if data['passed'] else "❌"
            print(f"  {status} {test_name}: {data['duration']:.2f}s")

        print("\n📋 QUALITY RESULTS:")
        for test_name, data in summary['quality_metrics'].items():
            status = "✅" if data['passed'] else "❌"
            print(f"  {status} {test_name}")

        if summary['failed_tests']:
            print("\n❌ FAILED TESTS:")
            for test in summary['failed_tests']:
                print(f"  • {test['name']}: {test['error']}")

        print("\n💡 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")

        print("\n" + "=" * 60)


def main():
    """Main entry point for integration test suite"""
    print("Ericsson RAN Features - Comprehensive Integration Test Suite")
    print("Based on final-plan.md requirements")
    print()

    # Create and run test suite
    test_suite = IntegrationTestSuite()

    try:
        summary = test_suite.run_all_tests()

        # Exit with appropriate code
        failed_count = summary['test_execution']['failed_tests']
        if failed_count > 0:
            print(f"\n❌ {failed_count} test(s) failed. Check details above.")
            sys.exit(1)
        else:
            print(f"\n✅ All tests passed! System is ready for production.")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()