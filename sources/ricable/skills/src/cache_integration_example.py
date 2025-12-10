#!/usr/bin/env python3
"""
Example demonstrating how to integrate the caching system with the existing Ericsson feature processor.
Shows various usage patterns and best practices.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List

from cache_manager import CacheManager
from cached_feature_processor import CachedFeatureProcessor


def example_basic_usage():
    """Basic usage example of the caching system"""
    print("=== Basic Usage Example ===")

    # Initialize cached processor
    processor = CachedFeatureProcessor("example_cache")

    # Process files with automatic caching
    source_dir = "elex_features_only"  # Replace with actual path
    if os.path.exists(source_dir):
        print(f"Processing files from {source_dir}")

        with processor.cache_manager:
            features = processor.process_all_files(source_dir, batch_size=20)

            # Get processing report
            report = processor.get_processing_report()
            print(f"Processed {len(features)} features")
            print(f"Cache hit rate: {report['cache_statistics']['hit_rate']}%")
    else:
        print(f"Source directory {source_dir} not found")


def example_cache_management():
    """Example of cache management operations"""
    print("\n=== Cache Management Example ===")

    # Initialize cache manager
    cache_dir = "example_cache_management"
    cache = CacheManager(cache_dir)

    try:
        # Create test file
        test_file = Path("test_feature.md")
        test_file.write_text("# Test Feature\nFAJ 123 4567\nTest content")

        # Check if cached (should be False first time)
        is_cached, data = cache.is_cached(str(test_file))
        print(f"First check - Cached: {is_cached}")

        # Cache test data
        test_feature = {
            'id': 'FAJ 123 4567',
            'name': 'Test Feature',
            'description': 'Test description'
        }
        cache.cache_feature(str(test_file), test_feature, processing_time=0.5)

        # Check again (should be True)
        is_cached, data = cache.is_cached(str(test_file))
        print(f"Second check - Cached: {is_cached}")
        print(f"Cached data: {data['id'] if data else 'None'}")

        # Get cache information
        info = cache.get_cache_info()
        print(f"Cache info: {info}")

        # Force reprocessing
        cache.force_reprocess(str(test_file))
        is_cached, data = cache.is_cached(str(test_file))
        print(f"After force reprocess - Cached: {is_cached}")

    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        cache.clear_cache()


def example_incremental_processing():
    """Example of incremental processing with file changes"""
    print("\n=== Incremental Processing Example ===")

    # Initialize processor
    processor = CachedFeatureProcessor("incremental_cache")

    try:
        # Create test files
        test_files = []
        for i in range(3):
            file_path = Path(f"test_feature_{i}.md")
            content = f"""# Test Feature {i}
FAJ 123 456{i}
Description for feature {i}
## Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| param{i} | int | Parameter {i} |
"""
            file_path.write_text(content)
            test_files.append(file_path)

        print("First processing run:")
        with processor.cache_manager:
            # Create a temporary source directory
            source_dir = Path("temp_source")
            source_dir.mkdir(exist_ok=True)

            for i, file_path in enumerate(test_files):
                dest_path = source_dir / file_path.name
                file_path.rename(dest_path)
                test_files[i] = dest_path

            # Process files
            features = processor.process_all_files(str(source_dir))
            report = processor.get_processing_report()

            print(f"Processed {len(features)} features")
            print(f"Files processed: {report['processing_statistics']['files_processed']}")
            print(f"Files from cache: {report['processing_statistics']['files_from_cache']}")

        print("\nSecond processing run (should use cache):")
        with processor.cache_manager:
            features = processor.process_all_files(str(source_dir))
            report = processor.get_processing_report()

            print(f"Processed {len(features)} features")
            print(f"Files processed: {report['processing_statistics']['files_processed']}")
            print(f"Files from cache: {report['processing_statistics']['files_from_cache']}")
            print(f"Cache hit rate: {report['cache_statistics']['hit_rate']}%")

        # Modify one file
        if test_files:
            test_files[0].write_text("# Modified Test Feature 0\nFAJ 123 4560\nModified content")
            print(f"\nModified {test_files[0].name}")

        print("\nThird processing run (one file should be reprocessed):")
        with processor.cache_manager:
            features = processor.process_all_files(str(source_dir))
            report = processor.get_processing_report()

            print(f"Processed {len(features)} features")
            print(f"Files processed: {report['processing_statistics']['files_processed']}")
            print(f"Files from cache: {report['processing_statistics']['files_from_cache']}")

    finally:
        # Cleanup
        for file_path in test_files:
            if file_path.exists():
                file_path.unlink()

        source_dir = Path("temp_source")
        if source_dir.exists():
            source_dir.rmdir()

        processor.cache_manager.clear_cache()


def example_error_handling():
    """Example of error handling and recovery"""
    print("\n=== Error Handling Example ===")

    # Initialize processor
    processor = CachedFeatureProcessor("error_handling_cache")

    try:
        # Create test files - one valid, one invalid
        valid_file = Path("valid_feature.md")
        valid_file.write_text("# Valid Feature\nFAJ 123 4567\nValid content")

        invalid_file = Path("invalid_feature.md")
        invalid_file.write_text("No FAJ ID here\nJust some text")

        # Process files
        source_dir = Path("error_test_source")
        source_dir.mkdir(exist_ok=True)

        valid_file.rename(source_dir / valid_file.name)
        invalid_file.rename(source_dir / invalid_file.name)

        with processor.cache_manager:
            features = processor.process_all_files(str(source_dir))
            report = processor.get_processing_report()

            print(f"Total files scanned: {report['processing_statistics']['files_scanned']}")
            print(f"Files processed: {report['processing_statistics']['files_processed']}")
            print(f"Files failed: {report['processing_statistics']['files_failed']}")
            print(f"Success rate: {report['processing_statistics']['success_rate']}%")

            # Show which files were cached
            cached_files = processor.cache_manager.list_cached_files()
            print(f"Cached files: {len(cached_files)}")
            for file_path in cached_files:
                print(f"  - {file_path}")

    finally:
        # Cleanup
        if source_dir.exists():
            for file_path in source_dir.glob("*"):
                file_path.unlink()
            source_dir.rmdir()

        processor.cache_manager.clear_cache()


def example_performance_monitoring():
    """Example of performance monitoring and optimization"""
    print("\n=== Performance Monitoring Example ===")

    # Create multiple test files
    test_files = []
    source_dir = Path("performance_test_source")
    source_dir.mkdir(exist_ok=True)

    try:
        # Create test files with varying complexity
        for i in range(10):
            file_path = source_dir / f"feature_{i}.md"
            content = f"""# Feature {i}
FAJ 123 456{i}
Description for feature {i}
"""

            # Add varying amounts of content
            for j in range(i * 5):
                content += f"Additional content line {j}\n"

            file_path.write_text(content)
            test_files.append(file_path)

        # Initialize processor with performance tracking
        processor = CachedFeatureProcessor("performance_cache")

        print("First run (cold cache):")
        start_time = time.time()
        with processor.cache_manager:
            features = processor.process_all_files(str(source_dir), batch_size=5)
            first_run_time = time.time() - start_time
            report = processor.get_processing_report()

            print(f"Processing time: {first_run_time:.2f}s")
            print(f"Files processed: {report['processing_statistics']['files_processed']}")
            print(f"Average processing time: {report['processing_statistics']['average_processing_time_seconds']:.3f}s")

        print("\nSecond run (warm cache):")
        start_time = time.time()
        with processor.cache_manager:
            features = processor.process_all_files(str(source_dir), batch_size=5)
            second_run_time = time.time() - start_time
            report = processor.get_processing_report()

            print(f"Processing time: {second_run_time:.2f}s")
            print(f"Files from cache: {report['processing_statistics']['files_from_cache']}")
            print(f"Cache efficiency: {report['performance']['cache_efficiency']:.1f}%")
            print(f"Time saved: {report['processing_statistics']['cache_time_saved_seconds']:.2f}s")
            print(f"Speedup: {first_run_time/max(second_run_time, 0.001):.1f}x")

    finally:
        # Cleanup
        for file_path in test_files:
            if file_path.exists():
                file_path.unlink()
        source_dir.rmdir()


def main():
    """Run all examples"""
    print("Caching System Integration Examples")
    print("=" * 50)

    try:
        # Run examples
        example_basic_usage()
        example_cache_management()
        example_incremental_processing()
        example_error_handling()
        example_performance_monitoring()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()