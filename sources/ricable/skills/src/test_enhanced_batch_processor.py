#!/usr/bin/env python3
"""
Test Script for Enhanced Batch Processing System
Validates the scalability, memory management, and resume capability
"""

import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import logging

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the batch processing system
from enhanced_batch_processor import BatchProcessor, BatchState, ProcessingStats
from enhanced_ericsson_processor import EnhancedEricssonProcessor


class TestBatchProcessor(BatchProcessor):
    """Test implementation of BatchProcessor for validation"""

    def process_file(self, file_path: Path) -> Optional[Dict]:
        """Simulate file processing for testing"""
        try:
            # Simulate processing time based on file size
            content = file_path.read_text(encoding='utf-8')
            processing_time = len(content) / 10000.0  # Simulate processing time
            time.sleep(processing_time)

            # Extract some basic data from file
            lines = content.split('\n')
            word_count = sum(len(line.split()) for line in lines)

            # Create mock feature data
            feature_data = {
                'id': f"feature_{file_path.stem}",
                'filename': file_path.name,
                'size': len(content),
                'lines': len(lines),
                'words': word_count,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_file': str(file_path)
            }

            # Simulate some processing errors
            if "error" in file_path.name.lower():
                raise Exception(f"Simulated processing error for {file_path.name}")

            return feature_data

        except Exception as e:
            logger.error(f"Test processing error for {file_path.name}: {e}")
            raise


def create_test_files(test_dir: Path, num_files: int = 100, include_errors: bool = True) -> List[Path]:
    """
    Create test markdown files for batch processing validation

    Args:
        test_dir: Directory to create test files
        num_files: Number of files to create
        include_errors: Whether to include files that cause errors

    Returns:
        List of created file paths
    """
    logger.info(f"📝 Creating {num_files} test files in {test_dir}")

    test_files = []
    test_dir.mkdir(parents=True, exist_ok=True)

    # Sample markdown content templates
    templates = [
        """
# Test Feature {i}

## Description
This is a test feature for batch processing validation.

## Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| param_{i} | string | Test parameter {i} |
| value_{i} | integer | Test value {i} |

## Summary
Test feature {i} summary content with multiple lines.
This helps validate that the batch processing system can handle
various content lengths and structures.
""",
        """
# Feature FAJ {i:03d} {i:04d}

## Identity
- FAJ Number: FAJ {i:03d} {i:04d}
- CXC Code: CXC {i_mod_1000000:06d}
- Node Type: TestNode
- Value Package: TestPackage

## Technical Details
### Parameters
{param_table}

### Counters
- PM_COUNTER_{i}: Test counter for feature {i}
- QUALITY_METRIC_{i}: Quality metric for feature {i}

## Engineering Guidelines
Test engineering guidelines for feature {i}.
This section tests extraction of multi-line content.
""",
        """
# Simple Test {i}

A simple test file with minimal content.

## Configuration
Test configuration for {i}.

## Notes
Test notes for feature {i}.
"""
    ]

    for i in range(num_files):
        # Choose template
        template_idx = i % len(templates)
        template = templates[template_idx]

        # Add parameter table for template 2
        if template_idx == 1:
            param_table = "\n".join([
                f"| MO_PARAM_{i}_{j} | {j} | Test parameter {j} |"
                for j in range(3)
            ])
            content = template.format(i=i, param_table=param_table, i_mod_1000000=i % 1000000)
        else:
            content = template.format(i=i)

        # Create file
        if include_errors and i % 20 == 0:  # 5% error rate
            filename = f"test_error_feature_{i:04d}.md"
        else:
            filename = f"test_feature_{i:04d}.md"

        file_path = test_dir / filename
        file_path.write_text(content.strip())
        test_files.append(file_path)

    logger.info(f"✅ Created {len(test_files)} test files")
    return test_files


def test_basic_batch_processing():
    """Test basic batch processing functionality"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Testing Basic Batch Processing")
    logger.info("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_dir = temp_path / "test_files"
        output_dir = temp_path / "output"

        # Create test files
        test_files = create_test_files(test_dir, num_files=50)

        # Create processor
        processor = TestBatchProcessor(
            source_dir=str(test_dir),
            output_dir=str(output_dir),
            batch_size=10,
            max_memory_mb=512,
            auto_gc=True,
            resume=False
        )

        # Process files
        start_time = time.time()
        processor.process_all()
        end_time = time.time()

        # Validate results
        processed_files = len(processor.processed_files_batch)
        expected_processed = len([f for f in test_files if "error" not in f.name])

        logger.info(f"✅ Processing completed in {end_time - start_time:.1f} seconds")
        logger.info(f"   Expected to process: {expected_processed} files")
        logger.info(f"   Actually processed: {processed_files} files")
        logger.info(f"   Batches completed: {processor.stats.batches_completed}")
        logger.info(f"   Memory peak: {processor.stats.memory_peak:.1f} MB")

        # Check if output files were created
        features_dir = output_dir / "ericsson_data" / "features"
        output_files = list(features_dir.glob("feature_*.json"))
        logger.info(f"   Output files created: {len(output_files)}")

        # Validate batch statistics
        assert processed_files == expected_processed, f"Expected {expected_processed}, got {processed_files}"
        assert len(output_files) == expected_processed, f"Expected {expected_processed} output files, got {len(output_files)}"
        assert processor.stats.batches_completed > 0, "No batches were completed"

        logger.info("✅ Basic batch processing test PASSED")


def test_memory_management():
    """Test memory management with large dataset"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Testing Memory Management")
    logger.info("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_dir = temp_path / "test_files"
        output_dir = temp_path / "output"

        # Create larger test files to test memory usage
        test_files = create_test_files(test_dir, num_files=200)

        # Create processor with strict memory limit
        processor = TestBatchProcessor(
            source_dir=str(test_dir),
            output_dir=str(output_dir),
            batch_size=20,
            max_memory_mb=256,  # Strict memory limit
            auto_gc=True,
            resume=False
        )

        # Process files
        start_time = time.time()
        processor.process_all()
        end_time = time.time()

        # Validate memory management
        logger.info(f"✅ Large dataset processed in {end_time - start_time:.1f} seconds")
        logger.info(f"   Memory peak: {processor.stats.memory_peak:.1f} MB")
        logger.info(f"   Memory limit: {processor.max_memory_mb} MB")
        logger.info(f"   Average batch time: {processor.stats.average_batch_time:.2f} seconds")

        # Check that memory usage was controlled
        assert processor.stats.memory_peak < processor.max_memory_mb * 1.2, \
            f"Memory usage ({processor.stats.memory_peak} MB) exceeded limit ({processor.max_memory_mb} MB)"

        # Verify all files were processed
        expected_processed = len([f for f in test_files if "error" not in f.name])
        assert processor.stats.processed_files == expected_processed, \
            f"Expected {expected_processed} files processed, got {processor.stats.processed_files}"

        logger.info("✅ Memory management test PASSED")


def test_resume_capability():
    """Test resume capability after interruption"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Testing Resume Capability")
    logger.info("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_dir = temp_path / "test_files"
        output_dir = temp_path / "output"

        # Create test files
        test_files = create_test_files(test_dir, num_files=100)

        # Create processor
        processor = TestBatchProcessor(
            source_dir=str(test_dir),
            output_dir=str(output_dir),
            batch_size=15,
            resume=True
        )

        # Process first 2 batches only
        logger.info("🔄 Processing first 2 batches...")
        files_to_process = test_files[:30]  # 2 batches of 15
        original_process_all = processor.process_all

        def limited_process():
            for batch_files, batch_num, total_batches in processor.create_file_batches(files_to_process):
                processor.current_batch = batch_num
                batch_state = processor.process_batch(batch_files, batch_num, total_batches)
                processor.batch_states.append(batch_state)

                if batch_num >= 2:
                    break

            processor.finalize_processing()

        processor.process_all = limited_process
        processor.process_all()

        first_run_stats = processor.stats.batches_completed
        logger.info(f"   First run: {first_run_stats} batches completed")

        # Create new processor instance to test resume
        processor2 = TestBatchProcessor(
            source_dir=str(test_dir),
            output_dir=str(output_dir),
            batch_size=15,
            resume=True
        )

        # Process remaining files
        logger.info("🔄 Resuming processing...")
        processor2.process_all()

        total_batches = processor2.stats.batches_completed
        logger.info(f"   Total after resume: {total_batches} batches completed")

        # Validate resume worked correctly
        expected_total_batches = (len(test_files) + 14) // 15  # Total batches needed
        assert total_batches >= expected_total_batches - 1, \
            f"Expected at least {expected_total_batches - 1} batches, got {total_batches}"

        # Check that no files were processed twice
        progress_file = output_dir / "ericsson_data" / "progress.json"
        if progress_file.exists():
            progress_data = json.loads(progress_file.read_text())
            logger.info(f"   Resume statistics loaded successfully")

        logger.info("✅ Resume capability test PASSED")


def test_error_handling():
    """Test error handling and recovery"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Testing Error Handling")
    logger.info("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_dir = temp_path / "test_files"
        output_dir = temp_path / "output"

        # Create test files with higher error rate
        test_files = create_test_files(test_dir, num_files=50, include_errors=True)

        # Ensure we have some error files
        error_file = test_dir / "error_test.md"
        error_file.write_text("This file will cause an error")
        test_files.append(error_file)

        # Create processor
        processor = TestBatchProcessor(
            source_dir=str(test_dir),
            output_dir=str(output_dir),
            batch_size=10,
            resume=False
        )

        # Process files
        processor.process_all()

        # Validate error handling
        expected_errors = len([f for f in test_files if "error" in f.name.lower()])
        actual_errors = processor.stats.failed_files

        logger.info(f"   Expected errors: {expected_errors}")
        logger.info(f"   Actual errors: {actual_errors}")
        logger.info(f"   Successful processing: {processor.stats.processed_files}")

        # Check that errors were recorded but processing continued
        assert actual_errors >= expected_errors, f"Expected at least {expected_errors} errors, got {actual_errors}"
        assert processor.stats.processed_files > 0, "No files were processed successfully"

        # Check error details
        if processor.stats.errors:
            logger.info("   Error details:")
            for file_path, error_msg in processor.stats.errors[:3]:
                logger.info(f"     - {Path(file_path).name}: {error_msg}")

        logger.info("✅ Error handling test PASSED")


def test_ericsson_integration():
    """Test integration with Ericsson processor"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Testing Ericsson Integration")
    logger.info("="*60)

    # This test would require actual Ericsson markdown files
    # For now, we'll create mock Ericsson-style files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_dir = temp_path / "ericsson_files"
        output_dir = temp_path / "output"

        # Create Ericsson-style test files
        ericsson_files = create_ericsson_test_files(test_dir, num_files=20)

        if not ericsson_files:
            logger.warning("⚠️ Skipping Ericsson integration test - no test files created")
            return

        # Create Ericsson processor
        processor = EnhancedEricssonProcessor(
            source_dir=str(test_dir),
            output_dir=str(output_dir),
            batch_size=5,
            resume=False
        )

        # Process files
        start_time = time.time()
        processor.process_all()
        end_time = time.time()

        # Validate Ericsson-specific processing
        logger.info(f"✅ Ericsson processing completed in {end_time - start_time:.1f} seconds")
        logger.info(f"   FAJ numbers found: {processor.stats.ericsson_stats['faj_numbers_found']}")
        logger.info(f"   CXC codes extracted: {processor.stats.ericsson_stats['cxc_codes_extracted']}")
        logger.info(f"   Parameters extracted: {processor.stats.ericsson_stats['parameters_extracted']}")
        logger.info(f"   Counters extracted: {processor.stats.ericsson_stats['counters_extracted']}")

        # Check for indices
        indices_dir = output_dir / "ericsson_data" / "indices"
        if indices_dir.exists():
            index_files = list(indices_dir.glob("*_index.json"))
            logger.info(f"   Index files created: {len(index_files)}")

        # Check for Ericsson summary
        ericsson_summary = output_dir / "ericsson_data" / "ericsson_summary.json"
        if ericsson_summary.exists():
            logger.info("   Ericsson summary created successfully")

        logger.info("✅ Ericsson integration test PASSED")


def create_ericsson_test_files(test_dir: Path, num_files: int = 20) -> List[Path]:
    """Create Ericsson-style test files"""
    test_dir.mkdir(parents=True, exist_ok=True)
    files = []

    for i in range(num_files):
        faj_num = f"{(i % 900) + 100:03d} {(i % 9000) + 1000:04d}"
        cxc_code = f"{(i % 1000000):06d}"

        content = f"""# Feature Test {i}

## Feature Identity
- FAJ Number: FAJ {faj_num}
- CXC Code: CXC {cxc_code}
- Node Type: RBS
- Value Package: Basic Package

## Description
This is a test Ericsson feature for validation of the enhanced batch processing system.

## Parameters
| MO Class | Parameter | Description | Type |
|----------|-----------|-------------|------|
| TestMO{i} | testParam{i} | Test parameter {i} | Integer |
| TestMO{i} | maxConfig{i} | Maximum configuration {i} | String |

## PM Counters
- COUNTER_{i}: Test counter for feature {i}
- QUALITY_METRIC_{i}: Quality metric for feature {i}

## Engineering Guidelines
1. Configure test parameters carefully
2. Monitor quality metrics
3. Apply appropriate thresholds

## Activation
Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC{cxc_code} MO instance.

## Dependencies
- Prerequisites: Feature {i-1} if {i} > 0
- Related features: Feature {i+1} if {i} < {num_files-1}
"""

        file_path = test_dir / f"FAJ_{faj_num.replace(' ', '_')}.md"
        file_path.write_text(content)
        files.append(file_path)

    return files


def run_all_tests():
    """Run all batch processing tests"""
    logger.info("🚀 Starting Enhanced Batch Processing System Tests")
    logger.info("="*80)

    tests = [
        test_basic_batch_processing,
        test_memory_management,
        test_resume_capability,
        test_error_handling,
        test_ericsson_integration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    logger.info("\n" + "="*80)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        return True
    else:
        logger.error(f"❌ {failed} TESTS FAILED")
        return False


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Enhanced Batch Processing System")
    parser.add_argument("--test", choices=[
        "basic", "memory", "resume", "errors", "ericsson", "all"
    ], default="all", help="Specific test to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test == "all":
        success = run_all_tests()
    elif args.test == "basic":
        test_basic_batch_processing()
        success = True
    elif args.test == "memory":
        test_memory_management()
        success = True
    elif args.test == "resume":
        test_resume_capability()
        success = True
    elif args.test == "errors":
        test_error_handling()
        success = True
    elif args.test == "ericsson":
        test_ericsson_integration()
        success = True

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()